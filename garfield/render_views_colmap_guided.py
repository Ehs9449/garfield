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


def fit_radius(bbox_min, bbox_max, fx, fy, cx, cy, margin=1.15):
    """Camera distance from the building centre so the whole crop box fits.

    Uses the narrower of the two half field-of-view angles, so the building
    fits in both image directions.
    """
    extent = np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32)
    half_diag = 0.5 * float(np.linalg.norm(extent))

    half_fov_x = np.arctan(cx / fx)
    half_fov_y = np.arctan(cy / fy)
    half_fov = float(min(half_fov_x, half_fov_y))

    return margin * half_diag / np.tan(half_fov)


def build_rings(cfg_views):
    """Return a list of {elevation, n_azimuth} rings.

    Preferred form in config.yaml:

        labeling_views:
          rings:
            - {elevation: 5,  n_azimuth: 16}
            - {elevation: 20, n_azimuth: 16}
            - {elevation: 40, n_azimuth: 12}

    Falls back to the older low/mid keys so existing configs keep working.
    """
    rings = cfg_views.get("rings")

    if rings:
        return [
            {
                "elevation": float(r["elevation"]),
                "n_azimuth": int(r["n_azimuth"]),
            }
            for r in rings
            if int(r["n_azimuth"]) > 0
        ]

    rings = []

    n_low = int(cfg_views.get("n_azimuth_low", 0))
    if n_low > 0:
        rings.append({
            "elevation": float(cfg_views.get("low_elevation", 15)),
            "n_azimuth": n_low,
        })

    n_mid = int(cfg_views.get("n_azimuth_mid", 0))
    if n_mid > 0:
        rings.append({
            "elevation": float(cfg_views.get("mid_elevation", 45)),
            "n_azimuth": n_mid,
        })

    return rings


def ring_camera_positions(center, radius, elevation_deg, n_azimuth, azimuth_offset=0.0):
    """Camera positions on one horizontal ring around the building centre."""
    center = np.asarray(center, dtype=np.float32)
    el = np.radians(float(elevation_deg))

    positions = []
    for k in range(int(n_azimuth)):
        az_deg = (azimuth_offset + 360.0 * k / float(n_azimuth)) % 360.0
        az = np.radians(az_deg)

        direction = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ], dtype=np.float32)

        positions.append((center + radius * direction, az_deg, float(elevation_deg)))

    return positions


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
    cfg_views = cfg["labeling_views"]

    rings = build_rings(cfg_views)
    n_top = int(cfg_views.get("n_top", 1))
    top_elevation = float(cfg_views.get("top_elevation", 89))
    azimuth_offset = float(cfg_views.get("azimuth_offset", 0.0))
    radius_factor = float(cfg_views.get("radius_factor", 1.15))
    forced_radius = cfg_views.get("building_radius", None)

    building_center = np.array(
        cfg["garfield"].get("crop_center", [0.02, -0.05, -0.15]),
        dtype=np.float32
    )
    building_scale = np.array(
        cfg["garfield"].get("crop_scale", [1.0, 0.91, 0.19]),
        dtype=np.float32
    )

    bbox_min = building_center - building_scale / 2
    bbox_max = building_center + building_scale / 2

    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove images from earlier runs. Otherwise old views stay in the
    # folder, get uploaded to the HPC, and SAM3 spends its time on views
    # that view_params.json no longer mentions.
    stale = sorted(output_dir.glob("*.jpg"))
    for old in stale:
        old.unlink()
    if stale:
        print(f"Removed {len(stale)} image(s) from a previous run")

    # Load model
    print("Loading Gaussian Splatting model...")
    original_cwd = os.getcwd()
    os.chdir(Path(__file__).resolve().parents[1])
    config, pipeline, checkpoint_path, step = eval_setup(args.config, test_mode='test')
    pipeline.eval()

    # Enable crop (from config, not hard-coded)
    crop_center = torch.tensor(building_center.tolist(), device=pipeline.device)
    crop_scale = torch.tensor(building_scale.tolist(), device=pipeline.device)
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

    # ------------------------------------------------------------
    # Ring viewpoints around the building
    #
    # Cameras sit on horizontal rings at several elevations and always
    # look at the building centre. This replaces the old COLMAP-guided
    # poses, which were oblique drone shots at uneven angles.
    # ------------------------------------------------------------
    if forced_radius is not None:
        radius = float(forced_radius)
        print(f"Using fixed camera radius from config: {radius:.3f}")
    else:
        radius = fit_radius(bbox_min, bbox_max, fx, fy, cx, cy, margin=radius_factor)
        print(f"Computed camera radius so the building fits the image: {radius:.3f}")

    views = []

    for ring in rings:
        for position, az_deg, el_deg in ring_camera_positions(
            building_center,
            radius,
            ring["elevation"],
            ring["n_azimuth"],
            azimuth_offset,
        ):
            views.append({
                "c2w": build_c2w_matrix(position, building_center),
                "type": "ring",
                "azimuth": az_deg,
                "elevation": el_deg,
            })

    # Top-down views for the roof.
    for k in range(n_top):
        az_deg = (azimuth_offset + 360.0 * k / max(n_top, 1)) % 360.0
        position = ring_camera_positions(
            building_center, radius, top_elevation, 1, az_deg
        )[0][0]

        views.append({
            "c2w": build_c2w_matrix(position, building_center),
            "type": "top",
            "azimuth": az_deg,
            "elevation": top_elevation,
        })

    print(f"Rendering {len(views)} ring views "
          f"({len(rings)} rings + {n_top} top view(s))...")

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
