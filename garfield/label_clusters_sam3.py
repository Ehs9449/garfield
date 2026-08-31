#!/usr/bin/env python3
"""
Semantic Cluster Labeling via SAM 3 PCS + Gaussian Splatting

For each rendered view:
1. Render clean Gaussian Splatting image
2. Run SAM 3 PCS with building element text prompts → labeled masks
3. Project ALL cluster points into the rendered view
4. For each cluster, count overlap with each PCS mask
5. Majority vote across views → final label per cluster

Usage:
    python label_clusters_sam3.py \
        --config outputs/PFTdrone/garfield-gauss/2026-05-03_101858/config.yml \
        --features-dir outputs/ortho_projection_cropped \
        --labels-npy outputs/ortho_projection_cropped/cluster_labels.npy \
        --output-dir outputs/semantic_labels_sam3
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os
import json
import cv2
from collections import Counter, defaultdict

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras


# Building element text prompts for SAM 3 PCS
BUILDING_PROMPTS = [
    "window",
    "wall",
    "roof",
    "door",
    "entrance",
    "column",
    "parapet",
    "balcony",
    "staircase",
    "HVAC equipment",
    "ground",
    "vegetation",
    "mechanical equipment",
    "facade panel",
    "canopy",
]


def load_sam3(device):
    """Load SAM 3 model."""
    try:
        from ultralytics import SAM3
        model = SAM3("sam3l.pt")  # downloads automatically on first use
        print("✓ Loaded SAM 3 (large)")
        return model
    except ImportError:
        print("ERROR: pip install ultralytics")
        print("  Make sure you have ultralytics >= 8.3.237")
        return None


def run_sam3_pcs(model, image_path, prompts):
    """
    Run SAM 3 PCS on an image with text prompts.
    
    Returns:
        list of dicts: [{'label': 'window', 'mask': np.array(H,W bool)}, ...]
    """
    results = model(str(image_path), texts=prompts)
    
    labeled_masks = []
    
    for result in results:
        if result.masks is None:
            continue
        
        masks = result.masks.data.cpu().numpy()  # N x H x W
        
        # Get class indices and names
        if hasattr(result, 'names') and result.names:
            for i in range(len(masks)):
                cls_idx = int(result.boxes.cls[i]) if result.boxes is not None else i
                label = result.names.get(cls_idx, prompts[min(i, len(prompts)-1)])
                labeled_masks.append({
                    'label': label,
                    'mask': masks[i].astype(bool),
                    'confidence': float(result.boxes.conf[i]) if result.boxes is not None else 1.0,
                })
        else:
            # Fallback: assign prompts in order
            for i in range(len(masks)):
                labeled_masks.append({
                    'label': prompts[min(i, len(prompts)-1)],
                    'mask': masks[i].astype(bool),
                    'confidence': 1.0,
                })
    
    return labeled_masks


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

    return torch.from_numpy(c2w[:3, :4])


def project_points_to_camera(points_3d, c2w, fx, fy, cx, cy, img_w, img_h):
    """Project 3D points into a nerfstudio camera."""
    c2w_np = c2w.numpy() if isinstance(c2w, torch.Tensor) else c2w
    c2w_4x4 = np.eye(4, dtype=np.float32)
    c2w_4x4[:3, :] = c2w_np
    w2c = np.linalg.inv(c2w_4x4)

    N = len(points_3d)
    pts_homo = np.hstack([points_3d, np.ones((N, 1), dtype=np.float32)])
    pts_cam = (w2c @ pts_homo.T).T[:, :3]

    depth = -pts_cam[:, 2]
    in_front = depth > 0.01

    px = np.full(N, -1, dtype=np.int32)
    py = np.full(N, -1, dtype=np.int32)

    if in_front.sum() > 0:
        px[in_front] = (fx * pts_cam[in_front, 0] / depth[in_front] + cx).astype(np.int32)
        py[in_front] = (fy * (-pts_cam[in_front, 1]) / depth[in_front] + cy).astype(np.int32)

    valid = in_front & (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    return px, py, valid


def label_clusters(config_path, features_dir, labels_path, output_dir, num_views=8):
    
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################
    # STEP 1: Load data
    ############################################################
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    points = np.load(features_dir / "points.npy")
    labels = np.load(labels_path)
    n_clusters = int(labels.max()) + 1
    print(f"  Points: {len(points)}, Clusters: {n_clusters}")

    # Load SAM 3
    sam3_model = load_sam3(device)
    if sam3_model is None:
        return

    ############################################################
    # STEP 2: Load Gaussian Splatting model
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 2: Loading Gaussian Splatting model")
    print("=" * 60)

    original_cwd = os.getcwd()
    os.chdir("/home/eaghae1")

    config, pipeline, checkpoint_path, step = eval_setup(config_path, test_mode='test')
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
    print(f"  Camera: {img_w}x{img_h}, fx={fx:.1f}")

    os.chdir(original_cwd)
    print("✓ Model loaded")

    ############################################################
    # STEP 3: Define viewpoints around the building
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 3: Defining viewpoints")
    print("=" * 60)

    building_center = np.array([0.02, -0.05, -0.15], dtype=np.float32)
    building_radius = 0.7  # far enough to see building facades clearly

    view_configs = []
    # Facade-level views every 45 degrees
    for az in range(0, 360, 45):
        view_configs.append((az, 15))
    # Elevated views for roof
    for az in range(0, 360, 90):
        view_configs.append((az, 45))

    view_configs = view_configs[:num_views]
    print(f"  Using {len(view_configs)} viewpoints")

    ############################################################
    # STEP 4: For each view, render + SAM3 PCS + match clusters
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 4: Rendering, segmenting, and matching")
    print("=" * 60)

    # Initialize vote storage: cluster_id -> list of label votes
    cluster_votes = defaultdict(list)

    for view_idx, (az_deg, el_deg) in enumerate(view_configs):
        print(f"\n--- View {view_idx+1}/{len(view_configs)}: az={az_deg}° el={el_deg}° ---")

        az = np.radians(az_deg)
        el = np.radians(el_deg)

        # Camera position
        cam_pos = [
            building_center[0] + building_radius * np.cos(az) * np.cos(el),
            building_center[1] + building_radius * np.sin(az) * np.cos(el),
            building_center[2] + building_radius * np.sin(el),
        ]

        c2w = build_c2w_matrix(cam_pos, building_center)

        # Create nerfstudio Camera
        camera = Cameras(
            camera_to_worlds=c2w.unsqueeze(0).to(pipeline.device),
            fx=torch.tensor([[fx]], device=pipeline.device),
            fy=torch.tensor([[fy]], device=pipeline.device),
            cx=torch.tensor([[cx]], device=pipeline.device),
            cy=torch.tensor([[cy]], device=pipeline.device),
            width=torch.tensor([[img_w]], device=pipeline.device),
            height=torch.tensor([[img_h]], device=pipeline.device),
        )

        # Render Gaussian Splatting view
        print(f"  Rendering...", end=" ", flush=True)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].cpu().numpy()
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Save rendered image
        render_path = renders_dir / f"view_az{az_deg:03d}_el{el_deg:02d}.jpg"
        cv2.imwrite(str(render_path), rgb_bgr)
        print(f"saved.", flush=True)

        # Run SAM 3 PCS on rendered image
        print(f"  Running SAM 3 PCS...", end=" ", flush=True)
        try:
            labeled_masks = run_sam3_pcs(sam3_model, render_path, BUILDING_PROMPTS)
            print(f"found {len(labeled_masks)} masks")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if len(labeled_masks) == 0:
            print(f"  No masks detected, skipping view")
            continue

        # Print detected elements
        label_counts = Counter(m['label'] for m in labeled_masks)
        print(f"  Detected: {dict(label_counts)}")

        # Save visualization of PCS masks
        vis = rgb_bgr.copy()
        for m in labeled_masks:
            color = np.random.randint(0, 255, 3).tolist()
            mask_overlay = np.zeros_like(vis)
            mask_overlay[m['mask']] = color
            vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.4, 0)
        pcs_path = renders_dir / f"view_az{az_deg:03d}_el{el_deg:02d}_pcs.jpg"
        cv2.imwrite(str(pcs_path), vis)

        # Project ALL points into this camera
        px, py, valid = project_points_to_camera(
            points.astype(np.float32), c2w, fx, fy, cx, cy, img_w, img_h
        )

        n_visible = valid.sum()
        print(f"  Visible points: {n_visible} ({100*n_visible/len(points):.1f}%)")

        if n_visible < 100:
            continue

        # For each cluster, count how many projected points fall in each PCS mask
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_valid = valid & cluster_mask

            n_cluster_visible = cluster_valid.sum()
            if n_cluster_visible < 3:
                continue

            cluster_px = px[cluster_valid]
            cluster_py = py[cluster_valid]

            # Count overlap with each PCS mask
            label_overlaps = defaultdict(int)

            for m in labeled_masks:
                pcs_mask = m['mask']
                # Resize PCS mask if needed (SAM3 might return different resolution)
                if pcs_mask.shape[0] != img_h or pcs_mask.shape[1] != img_w:
                    pcs_mask = cv2.resize(
                        pcs_mask.astype(np.uint8), (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                # Count how many cluster points are inside this mask
                points_in_mask = pcs_mask[cluster_py, cluster_px].sum()
                if points_in_mask > 0:
                    label_overlaps[m['label']] += int(points_in_mask)

            # The label with most overlapping points wins for this view
            if label_overlaps:
                best_label = max(label_overlaps, key=label_overlaps.get)
                best_count = label_overlaps[best_label]
                coverage = best_count / n_cluster_visible

                # Only vote if significant overlap (>20% of visible cluster points)
                if coverage > 0.2:
                    cluster_votes[cluster_id].append({
                        'label': best_label,
                        'coverage': float(coverage),
                        'view': f"az{az_deg}_el{el_deg}",
                        'all_overlaps': dict(label_overlaps),
                    })

    ############################################################
    # STEP 5: Majority vote per cluster
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 5: Majority vote")
    print("=" * 60)

    cluster_results = {}

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        n_pts = int(cluster_mask.sum())
        if n_pts < 10:
            continue

        votes = cluster_votes.get(cluster_id, [])

        if votes:
            label_counts = Counter(v['label'] for v in votes)
            best_label = label_counts.most_common(1)[0][0]
            best_count = label_counts.most_common(1)[0][1]
            confidence = best_count / len(votes)

            avg_coverage = np.mean([v['coverage'] for v in votes if v['label'] == best_label])

            cluster_results[cluster_id] = {
                'label': best_label,
                'confidence': float(confidence),
                'votes': len(votes),
                'vote_breakdown': dict(label_counts),
                'avg_coverage': float(avg_coverage),
                'n_points': n_pts,
            }
            print(f"  Cluster {cluster_id:3d} ({n_pts:6d} pts): {best_label:<20s} "
                  f"({confidence:.0%}, {best_count}/{len(votes)} votes, coverage={avg_coverage:.0%})")
        else:
            cluster_results[cluster_id] = {
                'label': 'unknown', 'confidence': 0.0, 'votes': 0, 'n_points': n_pts,
            }
            print(f"  Cluster {cluster_id:3d} ({n_pts:6d} pts): unknown (no votes)")

    ############################################################
    # STEP 6: Save results
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 6: Saving results")
    print("=" * 60)

    with open(output_dir / "semantic_labels.json", 'w') as f:
        json.dump(cluster_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    label_summary = defaultdict(int)
    for cid, r in cluster_results.items():
        label_summary[r['label']] += r['n_points']

    print("Label Summary:")
    for label, count in sorted(label_summary.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(points)
        print(f"  {label:<25s}: {count:8d} points ({pct:.1f}%)")

    # Save semantically colored point cloud
    COLORS = {
        'window': [0.0, 0.8, 1.0],
        'wall': [0.7, 0.7, 0.6],
        'roof': [0.8, 0.2, 0.2],
        'door': [0.8, 0.4, 0.0],
        'entrance': [1.0, 0.6, 0.0],
        'column': [0.9, 0.9, 0.9],
        'parapet': [0.6, 0.3, 0.3],
        'balcony': [0.2, 0.8, 0.2],
        'staircase': [1.0, 1.0, 0.0],
        'HVAC equipment': [0.5, 0.0, 0.5],
        'ground': [0.4, 0.4, 0.4],
        'vegetation': [0.0, 0.6, 0.0],
        'mechanical equipment': [0.6, 0.0, 0.6],
        'facade panel': [0.8, 0.7, 0.5],
        'canopy': [0.3, 0.6, 0.6],
        'unknown': [0.3, 0.3, 0.3],
    }

    import open3d as o3d

    pc = np.zeros((len(points), 3))
    for cid, r in cluster_results.items():
        pc[labels == cid] = COLORS.get(r['label'], [0.3, 0.3, 0.3])
    pc[labels == -1] = [0.2, 0.2, 0.2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(str(output_dir / "semantic_pointcloud.ply"), pcd)
    print(f"\nSaved: {output_dir / 'semantic_pointcloud.ply'}")

    # Also save per-cluster IoU with SAM3 masks for evaluation
    print(f"Saved: {output_dir / 'semantic_labels.json'}")
    print(f"Renders: {renders_dir}")
    print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster Labeling via SAM 3 PCS")
    parser.add_argument("--config", type=Path,
                        default=Path("/home/eaghae1/outputs/PFTdrone/garfield-gauss/2026-05-03_101858/config.yml"),
                        help="Gaussian Splatting config (NOT garfield nerfacto)")
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/ortho_projection_cropped"))
    parser.add_argument("--labels-npy", type=Path, default=Path("outputs/ortho_projection_cropped/cluster_labels.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_labels_sam3"))
    parser.add_argument("--num-views", type=int, default=12)
    args = parser.parse_args()

    label_clusters(args.config, args.features_dir, args.labels_npy, args.output_dir, args.num_views)
