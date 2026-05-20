"""
Eval script for Splatter Image (objaverse/GSO model) on custom dataset.

Dataset structure:
    input_root/
        archive_xxx/
            object_name/
                rgb/
                    000.png   <-- front view (elev=0, azim=0, fov=49.13, dist=2.0, OpenGL)

    eval_root/
        archive_xxx/
            object_name/
                rgb/
                    000.png .. 015.png   <-- 16 GT novel views

Usage:
    python eval_splatter_image.py \
        --input_root  /path/to/input_dataset \
        --eval_root   /path/to/eval_dataset \
        --model_path  /path/to/model_latest.pth \
        --config_path /path/to/.hydra/config.yaml \
        --output_dir  ./splatter_eval_out \
        --save_vis    10
"""

import os
import argparse
import json
import math
import numpy as np
import torch
import torchvision
import lpips as lpips_lib
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf


# ------------------------------------------------------------------ #
#  Camera helpers                                                     #
# ------------------------------------------------------------------ #

def get_projection_matrix(znear, zfar, fov_deg):
    """Row-major projection matrix (matches Splatter Image graphics_utils)."""
    fov = math.radians(fov_deg)
    t   = math.tan(fov / 2) * znear
    P   = torch.zeros(4, 4)
    P[0, 0] =  2.0 * znear / (2 * t)
    P[1, 1] =  2.0 * znear / (2 * t)
    P[3, 2] =  1.0
    P[2, 2] =  zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.transpose(0, 1)   # row-major


def orbit_camera_opengl(elev_deg, azim_deg, dist=2.0):
    """c2w (4x4) in OpenGL convention (y-up, z-backward)."""
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x =  dist * math.cos(elev) * math.sin(azim)
    y =  dist * math.sin(elev)
    z =  dist * math.cos(elev) * math.cos(azim)
    cam_pos = np.array([x, y, z], dtype=np.float32)

    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up    = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] =  right
    c2w[:3, 1] =  up
    c2w[:3, 2] = -forward
    c2w[:3, 3] =  cam_pos
    return torch.from_numpy(c2w)


OPENGL_TO_COLMAP = torch.tensor([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=torch.float32)


def c2w_to_splatter(c2w_opengl):
    """OpenGL c2w -> (world_view_transform, view_world_transform, camera_center) row-major."""
    w2c = OPENGL_TO_COLMAP @ torch.inverse(c2w_opengl)
    wvt = w2c.transpose(0, 1)
    vwt = torch.inverse(w2c).transpose(0, 1)
    cc  = vwt[3, :3].clone()
    return wvt, vwt, cc


# ------------------------------------------------------------------ #
#  16 novel-view poses                                                #
# ------------------------------------------------------------------ #

NOVEL_VIEW_PARAMS = (
    [(30.0, 45.0 * i) for i in range(8)] +
    [(60.0, 45.0 * i) for i in range(8)]
)


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #

class Metricator:
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)

    @torch.no_grad()
    def compute(self, pred, gt):
        mse  = torch.mean((pred - gt) ** 2)
        psnr = -10 * torch.log10(mse + 1e-8).item()
        from utils.loss_utils import ssim as ssim_fn
        ssim = ssim_fn(pred, gt).item()
        
        import torch.nn.functional as F
        pred_256 = F.interpolate(pred.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)
        gt_256   = F.interpolate(gt.unsqueeze(0),   (256, 256), mode='bilinear', align_corners=False)
        lpips = self.lpips_net(
            pred_256 * 2 - 1,
            gt_256   * 2 - 1
        ).item()
        return psnr, ssim, lpips


# ------------------------------------------------------------------ #
#  Helper: resolve eval path (mirrors _resolve_eval_item_path)        #
# ------------------------------------------------------------------ #

def resolve_eval_path(eval_root, archive_name, obj_name, view_idx):
    """Return path to GT novel view image, or None if not found."""
    # try same archive first
    for candidate in [archive_name] + sorted(os.listdir(eval_root)):
        p = os.path.join(eval_root, candidate, obj_name, "rgb", f"{view_idx:03d}.png")
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config & model
    cfg = OmegaConf.load(args.config_path)

    from scene.gaussian_predictor import GaussianSplatPredictor
    from gaussian_renderer import render_predicted

    model = GaussianSplatPredictor(cfg)
    ckpt  = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print("Loaded Splatter Image model.")

    fov   = cfg.data.fov                  # 49.134...
    znear = cfg.data.znear                # 0.8
    zfar  = cfg.data.zfar                 # 3.2
    res   = cfg.data.training_resolution  # 128
    dist  = 2.0

    proj_mat = get_projection_matrix(znear, zfar, fov).to(device)
    render_proj_mat = get_projection_matrix(znear, zfar, args.render_fov).to(device)
    bg       = torch.tensor([1., 1., 1.], dtype=torch.float32, device=device)
    metricator = Metricator(device)

    # precompute novel-view camera matrices
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)  # cho phép sửa cfg
    cfg.data.fov = args.render_fov
    
    novel_cameras = []
    for elev, azim in NOVEL_VIEW_PARAMS:
        c2w = orbit_camera_opengl(elev, azim, args.render_dist)
        wvt, vwt, cc = c2w_to_splatter(c2w)
        novel_cameras.append({
            "world_view_transform": wvt.to(device),
            "full_proj_transform":  (wvt.to(device) @ render_proj_mat),
            "camera_center":        cc.to(device),
        })

    # source camera: view 000, elev=0, azim=0
    src_c2w      = orbit_camera_opengl(0.0, 0.0, dist)
    src_wvt, src_vwt, _ = c2w_to_splatter(src_c2w)

    from utils.general_utils import matrix_to_quaternion
    src_R    = src_vwt[:3, :3].transpose(0, 1)
    src_quat = matrix_to_quaternion(src_R).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,4]
    src_vwt_b = src_vwt.unsqueeze(0).unsqueeze(0).to(device)                     # [1,1,4,4]

    # collect objects: input_root/archive_xxx/obj_name/rgb/000.png
    object_entries = []
    for archive in sorted(os.listdir(args.input_root)):
        archive_path = os.path.join(args.input_root, archive)
        if not os.path.isdir(archive_path):
            continue
        for obj_name in sorted(os.listdir(archive_path)):
            rgb_dir = os.path.join(archive_path, obj_name, "rgb")
            if os.path.isdir(rgb_dir):
                object_entries.append((archive, obj_name))

    print(f"Found {len(object_entries)} objects.")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    all_psnr, all_ssim, all_lpips = [], [], []
    results_per_object = {}

    for obj_idx, (archive_name, obj_name) in enumerate(tqdm(object_entries)):
        obj_dir  = os.path.join(args.input_root, archive_name, obj_name)
        img_path = os.path.join(obj_dir, "rgb", "000.png")
        if not os.path.exists(img_path):
            print(f"[WARN] 000.png not found for {archive_name}/{obj_name}, skipping.")
            continue

        # load & preprocess input image
        img_pil = Image.open(img_path).convert("RGBA")
        img_pil = torchvision.transforms.functional.resize(
            img_pil, res,
            interpolation=torchvision.transforms.InterpolationMode.LANCZOS
        )
        img_t   = torchvision.transforms.functional.pil_to_tensor(img_pil) / 255.0  # [4,H,W]
        alpha   = img_t[3:4]
        img_rgb = (img_t[:3] * alpha + (1 - alpha)).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,H,W]

        # run model
        reconstruction = model(img_rgb, src_vwt_b, src_quat, None)

        # render & score 16 novel views
        psnr_obj, ssim_obj, lpips_obj = [], [], []
        vis_dir = os.path.join(args.output_dir, f"{archive_name}_{obj_name}") if args.output_dir else None

        for v_idx, cam in enumerate(novel_cameras):
            pred = render_predicted(
                {k: v[0].contiguous() for k, v in reconstruction.items()},
                cam["world_view_transform"],
                cam["full_proj_transform"],
                cam["camera_center"],
                bg, cfg, focals_pixels=None,
                render_res=512
            )["render"]   # [3, H, W]

            if vis_dir and obj_idx < args.save_vis:
                os.makedirs(vis_dir, exist_ok=True)
                torchvision.utils.save_image(pred, os.path.join(vis_dir, f"pred_{v_idx:03d}.png"))

            gt_path = resolve_eval_path(args.eval_root, archive_name, obj_name, v_idx)
            if gt_path is None:
                continue

            gt_pil = Image.open(gt_path).convert("RGBA")
            gt_pil = torchvision.transforms.functional.resize(
                gt_pil, 512,
                interpolation=torchvision.transforms.InterpolationMode.LANCZOS
            )
            gt_t   = torchvision.transforms.functional.pil_to_tensor(gt_pil) / 255.0
            gt_a   = gt_t[3:4]
            gt_rgb = (gt_t[:3] * gt_a + (1 - gt_a)).to(device)

            # skip empty frames
            if torch.all(gt_rgb >= 0.999) or torch.all(gt_rgb <= 0.001):
                continue

            if vis_dir and obj_idx < args.save_vis:
                torchvision.utils.save_image(gt_rgb, os.path.join(vis_dir, f"gt_{v_idx:03d}.png"))

            p, s, l = metricator.compute(pred.clamp(0, 1), gt_rgb)
            psnr_obj.append(p)
            ssim_obj.append(s)
            lpips_obj.append(l)

        if not psnr_obj:
            print(f"[WARN] No valid GT views for {archive_name}/{obj_name}, skipping.")
            continue

        mp = sum(psnr_obj)  / len(psnr_obj)
        ms = sum(ssim_obj)  / len(ssim_obj)
        ml = sum(lpips_obj) / len(lpips_obj)
        all_psnr.append(mp); all_ssim.append(ms); all_lpips.append(ml)
        results_per_object[f"{archive_name}/{obj_name}"] = {"psnr": mp, "ssim": ms, "lpips": ml}

    scores = {
        "PSNR":  sum(all_psnr)  / len(all_psnr),
        "SSIM":  sum(all_ssim)  / len(all_ssim),
        "LPIPS": sum(all_lpips) / len(all_lpips),
        "num_objects": len(all_psnr),
    }
    print("\n===== Splatter Image Eval Results =====")
    print(f"  PSNR : {scores['PSNR']:.4f}")
    print(f"  SSIM : {scores['SSIM']:.4f}")
    print(f"  LPIPS: {scores['LPIPS']:.4f}")
    print(f"  Objects evaluated: {scores['num_objects']}")

    if args.output_dir:
        with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
            json.dump({"aggregate": scores, "per_object": results_per_object}, f, indent=4)
        print(f"Saved to {args.output_dir}/scores.json")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root",  required=True,
                        help="Root of input dataset (archive_xxx/obj/rgb/000.png)")
    parser.add_argument("--eval_root",   required=True,
                        help="Root of eval dataset (archive_xxx/obj/rgb/000..015.png)")
    parser.add_argument("--model_path",  required=True,
                        help="Path to model_latest.pth")
    parser.add_argument("--config_path", required=True,
                        help="Path to .hydra/config.yaml")
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--save_vis",    type=int, default=0,
                        help="How many objects to save renders for")
    parser.add_argument("--render_fov",  type=float, default=60.0)
    parser.add_argument("--render_dist", type=float, default=1.5)
    args = parser.parse_args()
    main(args)
