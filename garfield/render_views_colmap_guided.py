#!/usr/bin/env python3
"""
SCRIPT 1: Render Gaussian Splatting views for labeling
Environment: nerfstudio3

Renders clean views of the building from multiple angles and saves:
- RGB images (JPG)
- Camera parameters (JSON) for projecting cluster points later

Usage:
    conda activate nerfstudio3
    python render_views_for_labeling.py \
        --config outputs/PFTdrone/garfield-gauss/2026-05-03_101858/config.yml \
        --output-dir outputs/labeling_views
"""
import yaml
import torch
import numpy as np
from pathlib import Path
import argparse
import os
import json
import cv2

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras


def build_c2w_matrix(position, centroid):
    """Build camera-to-world matrix looking at centroid from position."""
    position = np.array(position, dtype=np.float32)
    centroid = np.array(centroid, dtype=np.float32)

    forward = centroid - position
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0], dtype=np.float32)
    else:
        right = right / right_norm

    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = position

    return c2w[:3, :]  # 3x4


def main():
    parser = argparse.ArgumentParser(description="Render views for labeling")
    parser.add_argument("--config", type=Path,
                        required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/labeling_views"))
    parser.add_argument("--config-yaml", type=Path, required=True,
                    help="Pipeline config YAML containing labeling view settings")
    args = parser.parse_args()
    with open(args.config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    n_low = int(cfg["labeling_views"]["n_azimuth_low"])
    n_mid = int(cfg["labeling_views"]["n_azimuth_mid"])
    n_top = int(cfg["labeling_views"]["n_top"])

    building_radius = float(cfg["labeling_views"].get("building_radius", 0.7))
    low_elevation = float(cfg["labeling_views"].get("low_elevation", 15))
    mid_elevation = float(cfg["labeling_views"].get("mid_elevation", 45))
    top_elevation = float(cfg["labeling_views"].get("top_elevation", 85))

    patch_test_center = np.array(cfg["labeling_views"]["patch_test_center"], dtype=np.float32)
    patch_test_radius = float(cfg["labeling_views"]["patch_test_radius"])

    building_center = np.array(
        cfg["garfield"].get("crop_center", [0.02, -0.05, -0.15]),
        dtype=np.float32
    )

    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Gaussian Splatting model...")
    original_cwd = os.getcwd()
    os.chdir(Path(__file__).resolve().parents[1])
    config, pipeline, checkpoint_path, step = eval_setup(args.config, test_mode='test')
    pipeline.eval()

    # Enable crop
    crop_center = torch.tensor([0.02, -0.05, -0.15], device=pipeline.device)
    crop_scale = torch.tensor([1.0, 0.91, 0.19], device=pipeline.device)
    pipeline.model.crop_enabled = True
    pipeline.model.crop_min = crop_center - crop_scale / 2
    pipeline.model.crop_max = crop_center + crop_scale / 2
    pipeline.model.crop_bg_color = torch.tensor([1.0, 1.0, 1.0], device=pipeline.device)

    # Get training camera intrinsics
    train_cameras = pipeline.datamanager.train_dataset.cameras
    fx = train_cameras.fx[0].item()
    fy = train_cameras.fy[0].item()
    cx = train_cameras.cx[0].item()
    cy = train_cameras.cy[0].item()
    img_w = int(train_cameras.width[0].item())
    img_h = int(train_cameras.height[0].item())
    os.chdir(original_cwd)
    print(f"✓ Model loaded. Camera: {img_w}x{img_h}")

    # COLMAP/Nerfstudio-guided viewpoints
    training_poses = np.load("outputs/towerlsu/training_camera_poses.npy")

    # Sample across the entire acquisition path so low, middle, and high views are preserved.
    n_guided_views = min(96, len(training_poses))
    selected_indices = np.linspace(
        0, len(training_poses) - 1, n_guided_views, dtype=int
    )

    views = []
    for j, pose_idx in enumerate(selected_indices):
        c2w = training_poses[pose_idx].astype(np.float32).copy()

        # Small deterministic perturbation so the rendered pose differs slightly
        # from the original training camera while preserving its useful viewpoint.
        yaw_deg = 3.0 if j % 2 == 0 else -3.0
        yaw = np.radians(yaw_deg)

        yaw_local = np.array([
            [ np.cos(yaw), 0.0, np.sin(yaw)],
            [ 0.0,         1.0, 0.0        ],
            [-np.sin(yaw), 0.0, np.cos(yaw)]
        ], dtype=np.float32)

        c2w[:, :3] = c2w[:, :3] @ yaw_local

        # Tiny position offset along the camera-right direction.
        offset = 0.008 if j % 2 == 0 else -0.008
        c2w[:, 3] += offset * c2w[:, 0]

        views.append({
            "pose_index": int(pose_idx),
            "c2w": c2w,
            "type": "colmap_guided",
            "azimuth": 0,
            "elevation": 0,
        })

    print(f"Rendering {len(views)} COLMAP-guided views...")

    all_view_params = []

    for i, view in enumerate(views):
        c2w = view["c2w"]
        c2w_tensor = torch.from_numpy(c2w)

        # Create nerfstudio Camera
        camera = Cameras(
            camera_to_worlds=c2w_tensor.unsqueeze(0).to(pipeline.device),
            fx=torch.tensor([[fx]], device=pipeline.device),
            fy=torch.tensor([[fy]], device=pipeline.device),
            cx=torch.tensor([[cx]], device=pipeline.device),
            cy=torch.tensor([[cy]], device=pipeline.device),
            width=torch.tensor([[img_w]], device=pipeline.device),
            height=torch.tensor([[img_h]], device=pipeline.device),
        )

        # Render
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].cpu().numpy()
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Save image
        view_name = f"view_{i:03d}_az{int(view['azimuth']):03d}_el{int(round(view['elevation'])):02d}"
        img_path = output_dir / f"{view_name}.jpg"
        cv2.imwrite(str(img_path), rgb_bgr)

        # Store camera params for projection later
        view_params = {
            'view_name': view_name,
            'image_file': f"{view_name}.jpg",
            'c2w': c2w.tolist(),
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'img_w': img_w, 'img_h': img_h,
            'azimuth': view['azimuth'],
            'elevation': view['elevation'],
            'type': view['type'],
        }
        all_view_params.append(view_params)

        print(f"  [{i+1}/{len(views)}] {view_name}.jpg")

    # Save all camera parameters
    with open(output_dir / "view_params.json", 'w') as f:
        json.dump(all_view_params, f, indent=2)

    print(f"\n✓ Saved {len(views)} images + view_params.json to {output_dir}")
    print("Next: conda activate sam3 && python run_sam3_pcs.py")


if __name__ == "__main__":
    main()
