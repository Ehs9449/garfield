#!/usr/bin/env python3
"""
Semantic Cluster Labeling via Novel View CLIP Voting
Uses nerfstudio's Cameras for rendering (no custom ray bundles).

For each cluster:
1. Create nerfstudio Camera pointing at cluster centroid
2. Render NeRF novel view using get_outputs_for_camera
3. Project cluster points into the rendered view to find crop region
4. Run CLIP on the crop
5. Vote across views → semantic label

Usage:
    python label_clusters_nerf.py \
        --features-dir outputs/ortho_projection_cropped \
        --labels-npy outputs/ortho_projection_cropped/cluster_labels.npy \
        --output-dir outputs/semantic_labels
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


BUILDING_CATEGORIES = [
    "building roof",
    "building wall",
    "window",
    "door",
    "entrance",
    "column",
    "parapet",
    "balcony",
    "staircase",
    "HVAC equipment on roof",
    "ground or pavement",
    "vegetation or trees",
    "mechanical equipment",
    "facade panel",
    "overhang or canopy",
]


def load_clip_model(device):
    """Load CLIP model via transformers."""
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    print("✓ Loaded CLIP ViT-B-32")
    return model, processor


def classify_crop_clip(crop_image, categories, model, processor, device):
    """Classify a cropped image using CLIP."""
    from PIL import Image

    if crop_image.shape[0] < 10 or crop_image.shape[1] < 10:
        return []

    pil_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))

    inputs = processor(
        text=categories, images=pil_image,
        return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()

    results = [(cat, float(prob)) for cat, prob in zip(categories, probs)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def build_c2w_matrix(position, centroid):
    """
    Build camera-to-world matrix looking at centroid from position.
    Same convention as _generate_camera_path_for_cluster:
        col0=right, col1=up, col2=-forward, col3=position
    """
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
    """
    Project 3D points into a nerfstudio camera.
    c2w: 3x4 camera-to-world (col0=right, col1=up, col2=-forward, col3=pos)
    Nerfstudio looks along -Z. Points in front have negative camera Z.
    """
    c2w_np = c2w.numpy() if isinstance(c2w, torch.Tensor) else c2w
    c2w_4x4 = np.eye(4, dtype=np.float32)
    c2w_4x4[:3, :] = c2w_np
    w2c = np.linalg.inv(c2w_4x4)

    N = len(points_3d)
    pts_homo = np.hstack([points_3d, np.ones((N, 1), dtype=np.float32)])
    pts_cam = (w2c @ pts_homo.T).T[:, :3]

    # Camera looks along -Z, so depth = -Z
    depth = -pts_cam[:, 2]
    in_front = depth > 0.01

    px = np.full(N, -1, dtype=np.int32)
    py = np.full(N, -1, dtype=np.int32)

    if in_front.sum() > 0:
        px[in_front] = (fx * pts_cam[in_front, 0] / depth[in_front] + cx).astype(np.int32)
        py[in_front] = (fy * (-pts_cam[in_front, 1]) / depth[in_front] + cy).astype(np.int32)

    valid = in_front & (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    return px, py, valid


def label_clusters(config_path, features_dir, labels_path, output_dir, num_views_per_cluster=8):

    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    points = np.load(features_dir / "points.npy")
    labels = np.load(labels_path)
    n_clusters = int(labels.max()) + 1
    print(f"  Points: {len(points)}, Clusters: {n_clusters}")

    clip_model, clip_processor = load_clip_model(device)

    # Load NeRF
    print("\n" + "=" * 60)
    print("STEP 2: Loading NeRF model")
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
    print("✓ Ready")

    # View angles
    view_angles = [
        (0, 15), (45, 15), (90, 15), (135, 15),
        (180, 15), (225, 15), (270, 15), (315, 15),
        (0, 40), (90, 40), (180, 40), (270, 40),
    ]

    # Classify each cluster
    print("\n" + "=" * 60)
    print("STEP 3: Rendering and classifying")
    print("=" * 60)

    cluster_results = {}

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_points = points[cluster_mask].astype(np.float32)
        n_pts = len(cluster_points)
        if n_pts < 10:
            continue

        print(f"\n--- Cluster {cluster_id}/{n_clusters-1} ({n_pts} pts) ---")

        centroid = cluster_points.mean(axis=0)
        bbox_size = np.linalg.norm(cluster_points.max(axis=0) - cluster_points.min(axis=0))
        radius = max(0.3, bbox_size * 2.0)

        votes = []

        for view_idx, (az_deg, el_deg) in enumerate(view_angles[:num_views_per_cluster]):
            try:
                az = np.radians(az_deg)
                el = np.radians(el_deg)

                cam_pos = [
                    centroid[0] + radius * np.cos(az) * np.cos(el),
                    centroid[1] + radius * np.sin(az) * np.cos(el),
                    centroid[2] + radius * np.sin(el),
                ]

                c2w = build_c2w_matrix(cam_pos, centroid)

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

                # Render
                with torch.no_grad():
                    outputs = pipeline.model.get_outputs_for_camera(camera)

                rgb = outputs["rgb"].cpu().numpy()
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Project cluster points to find crop region
                ppx, ppy, valid = project_points_to_camera(
                    cluster_points, c2w, fx, fy, cx, cy, img_w, img_h
                )

                if valid.sum() < 3:
                    continue

                vis_px = ppx[valid]
                vis_py = ppy[valid]

                pad = 40
                x1 = max(0, int(vis_px.min()) - pad)
                x2 = min(img_w, int(vis_px.max()) + pad)
                y1 = max(0, int(vis_py.min()) - pad)
                y2 = min(img_h, int(vis_py.max()) + pad)

                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue

                crop = rgb_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Save first 3 views per cluster
                if view_idx < 3:
                    vis_img = rgb_bgr.copy()
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    step_p = max(1, valid.sum() // 50)
                    for px_i, py_i in zip(vis_px[::step_p], vis_py[::step_p]):
                        cv2.circle(vis_img, (int(px_i), int(py_i)), 3, (0, 0, 255), -1)
                    cv2.imwrite(str(renders_dir / f"c{cluster_id:03d}_az{az_deg:03d}_full.jpg"), vis_img)
                    cv2.imwrite(str(renders_dir / f"c{cluster_id:03d}_az{az_deg:03d}_crop.jpg"), crop)

                # CLIP classify
                results = classify_crop_clip(crop, BUILDING_CATEGORIES, clip_model, clip_processor, device)

                if results:
                    votes.append({
                        'label': results[0][0],
                        'score': results[0][1],
                        'azimuth': az_deg,
                        'top3': [(r[0], round(r[1], 3)) for r in results[:3]],
                    })

            except Exception as e:
                print(f"  az={az_deg}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue

        # Majority vote
        if votes:
            counts = Counter(v['label'] for v in votes)
            best = counts.most_common(1)[0]
            confidence = best[1] / len(votes)
            avg_score = np.mean([v['score'] for v in votes if v['label'] == best[0]])

            cluster_results[cluster_id] = {
                'label': best[0], 'confidence': float(confidence),
                'votes': len(votes), 'vote_breakdown': dict(counts),
                'avg_clip_score': float(avg_score), 'n_points': n_pts,
            }
            print(f"  → {best[0]} ({confidence:.0%}, {best[1]}/{len(votes)} votes)")
        else:
            cluster_results[cluster_id] = {
                'label': 'unknown', 'confidence': 0.0, 'votes': 0, 'n_points': n_pts,
            }
            print(f"  → unknown")

    # Save results
    print("\n" + "=" * 60)
    print("STEP 4: Saving results")
    print("=" * 60)

    with open(output_dir / "semantic_labels.json", 'w') as f:
        json.dump(cluster_results, f, indent=2)

    print(f"\n{'Cluster':>8s} {'Points':>8s} {'Label':<25s} {'Conf':>6s} {'Votes':>6s}")
    print("-" * 60)
    label_summary = defaultdict(int)
    for cid in sorted(cluster_results.keys()):
        r = cluster_results[cid]
        print(f"{cid:8d} {r['n_points']:8d} {r['label']:<25s} {r['confidence']:5.0%} {r['votes']:6d}")
        label_summary[r['label']] += r['n_points']

    print(f"\nSummary:")
    for label, count in sorted(label_summary.items(), key=lambda x: -x[1]):
        print(f"  {label:<25s}: {count:8d} points ({100*count/len(points):.1f}%)")

    # Save colored point cloud
    COLORS = {
        'building roof': [0.8, 0.2, 0.2], 'building wall': [0.7, 0.7, 0.6],
        'window': [0.0, 0.8, 1.0], 'door': [0.8, 0.4, 0.0],
        'entrance': [1.0, 0.6, 0.0], 'column': [0.9, 0.9, 0.9],
        'parapet': [0.6, 0.3, 0.3], 'balcony': [0.2, 0.8, 0.2],
        'staircase': [1.0, 1.0, 0.0], 'HVAC equipment on roof': [0.5, 0.0, 0.5],
        'ground or pavement': [0.4, 0.4, 0.4], 'vegetation or trees': [0.0, 0.6, 0.0],
        'mechanical equipment': [0.6, 0.0, 0.6], 'facade panel': [0.8, 0.7, 0.5],
        'overhang or canopy': [0.3, 0.6, 0.6], 'unknown': [0.3, 0.3, 0.3],
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
    print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("/home/eaghae1/outputs/unnamed/garfield/2026-04-30_092326/config.yml"))
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/ortho_projection_cropped"))
    parser.add_argument("--labels-npy", type=Path, default=Path("outputs/ortho_projection_cropped/cluster_labels.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_labels"))
    parser.add_argument("--num-views", type=int, default=8)
    args = parser.parse_args()

    label_clusters(args.config, args.features_dir, args.labels_npy, args.output_dir, args.num_views)
