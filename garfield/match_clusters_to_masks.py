#!/usr/bin/env python3
"""
SCRIPT 3: Match cluster projections to SAM 3 labeled masks
Environment: nerfstudio3

Fixed matching logic: uses IoU between each cluster's projected pixels
and each INDIVIDUAL mask, not summed by label.

Usage:
    conda activate nerfstudio3
    python match_clusters_to_masks.py \
        --features-dir outputs/ortho_projection_cropped \
        --labels-npy outputs/ortho_projection_cropped/cluster_labels.npy \
        --views-dir outputs/labeling_views \
        --masks-dir outputs/labeling_masks \
        --output-dir outputs/semantic_labels_final
"""

import numpy as np
from pathlib import Path
import argparse
import json
import cv2
from collections import Counter, defaultdict


def project_points_to_camera(points_3d, c2w, fx, fy, cx, cy, img_w, img_h):
    """Project 3D points into a camera defined by c2w matrix."""
    c2w = np.array(c2w, dtype=np.float32)
    c2w_4x4 = np.eye(4, dtype=np.float32)
    c2w_4x4[:3, :] = c2w
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


def load_masks_for_view(masks_dir, view_name, img_h, img_w):
    """Load all saved masks for a specific view."""
    mask_dir = masks_dir / view_name

    if not mask_dir.exists():
        return []

    info_path = mask_dir / "masks_info.json"
    if not info_path.exists():
        return []

    with open(info_path) as f:
        info = json.load(f)

    masks = []
    for i in range(info['n_masks']):
        label = info['labels'][i]
        confidence = info['confidences'][i]

        mask_files = list(mask_dir.glob(f"mask_{i:03d}_*.npy"))
        if not mask_files:
            continue

        mask_data = np.load(mask_files[0]).astype(bool)

        # Resize if needed
        if mask_data.shape[0] != img_h or mask_data.shape[1] != img_w:
            mask_data = cv2.resize(
                mask_data.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        masks.append({
            'label': label,
            'mask': mask_data,
            'confidence': confidence,
            'size': int(mask_data.sum()),
        })

    return masks


def compute_iou(cluster_mask_2d, sam_mask):
    """
    Compute IoU between cluster's projected pixels and a SAM mask.
    cluster_mask_2d: H x W boolean (True where cluster points project)
    sam_mask: H x W boolean (True where SAM detected something)
    """
    intersection = (cluster_mask_2d & sam_mask).sum()
    union = (cluster_mask_2d | sam_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def main():
    parser = argparse.ArgumentParser(description="Match clusters to SAM 3 masks")
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/ortho_projection_cropped"))
    parser.add_argument("--labels-npy", type=Path, default=Path("outputs/ortho_projection_cropped/cluster_labels.npy"))
    parser.add_argument("--views-dir", type=Path, default=Path("outputs/labeling_views"))
    parser.add_argument("--masks-dir", type=Path, default=Path("outputs/labeling_masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_labels_final"))
    parser.add_argument("--min-iou", type=float, default=0.05,
                        help="Minimum IoU to count as a vote")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ############################################################
    # STEP 1: Load data
    ############################################################
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    points = np.load(args.features_dir / "points.npy").astype(np.float32)
    labels = np.load(args.labels_npy)
    n_clusters = int(labels.max()) + 1

    print(f"  Points: {len(points)}")
    print(f"  Clusters: {n_clusters}")

    with open(args.views_dir / "view_params.json") as f:
        view_params = json.load(f)
    print(f"  Views: {len(view_params)}")

    ############################################################
    # STEP 2: For each view, load masks and match clusters
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 2: Matching clusters to masks (IoU-based)")
    print("=" * 60)

    cluster_votes = defaultdict(list)

    for vi, vp in enumerate(view_params):
        view_name = vp['view_name']
        c2w = np.array(vp['c2w'], dtype=np.float32)
        fx, fy = vp['fx'], vp['fy']
        cx, cy = vp['cx'], vp['cy']
        img_w, img_h = vp['img_w'], vp['img_h']

        # Load masks
        masks = load_masks_for_view(args.masks_dir, view_name, img_h, img_w)
        if len(masks) == 0:
            continue

        print(f"\n--- View {vi+1}/{len(view_params)}: {view_name} ({len(masks)} masks) ---")

        # Project ALL points
        px, py, valid = project_points_to_camera(points, c2w, fx, fy, cx, cy, img_w, img_h)
        n_visible = valid.sum()
        print(f"  Visible points: {n_visible} ({100*n_visible/len(points):.1f}%)")

        if n_visible < 100:
            continue

        # For each cluster, find the best matching INDIVIDUAL mask by IoU
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_valid = valid & cluster_mask

            n_cluster_visible = cluster_valid.sum()
            if n_cluster_visible < 3:
                continue

            # Create a 2D binary image of where this cluster projects
            cluster_2d = np.zeros((img_h, img_w), dtype=bool)
            cluster_px = px[cluster_valid]
            cluster_py = py[cluster_valid]
            cluster_2d[cluster_py, cluster_px] = True

            # Find the best matching individual mask by IoU
            best_iou = 0.0
            best_label = None
            best_mask_size = 0

            for m in masks:
                iou = compute_iou(cluster_2d, m['mask'])
                if iou > best_iou:
                    best_iou = iou
                    best_label = m['label']
                    best_mask_size = m['size']

            # Only vote if IoU is above threshold
            if best_iou > args.min_iou and best_label is not None:
                cluster_votes[cluster_id].append({
                    'label': best_label,
                    'iou': float(best_iou),
                    'view': view_name,
                    'n_visible': int(n_cluster_visible),
                    'mask_size': best_mask_size,
                })

    ############################################################
    # STEP 3: Majority vote per cluster
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 3: Majority vote")
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
            avg_iou = np.mean([v['iou'] for v in votes if v['label'] == best_label])

            cluster_results[cluster_id] = {
                'label': best_label,
                'confidence': float(confidence),
                'votes': len(votes),
                'vote_breakdown': dict(label_counts),
                'avg_iou': float(avg_iou),
                'n_points': n_pts,
            }
            print(f"  Cluster {cluster_id:3d} ({n_pts:6d} pts): {best_label:<20s} "
                  f"({confidence:.0%}, {best_count}/{len(votes)} votes, IoU={avg_iou:.3f})")
        else:
            cluster_results[cluster_id] = {
                'label': 'unknown', 'confidence': 0.0, 'votes': 0, 'n_points': n_pts,
            }
            print(f"  Cluster {cluster_id:3d} ({n_pts:6d} pts): unknown (no votes)")

    ############################################################
    # STEP 4: Save results
    ############################################################
    print("\n" + "=" * 60)
    print("STEP 4: Saving results")
    print("=" * 60)

    with open(args.output_dir / "semantic_labels.json", 'w') as f:
        json.dump(cluster_results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Cluster':>8s} {'Points':>8s} {'Label':<20s} {'Conf':>6s} {'Votes':>6s} {'IoU':>6s}")
    print(f"{'-'*70}")

    label_summary = defaultdict(int)
    for cid in sorted(cluster_results.keys()):
        r = cluster_results[cid]
        iou_str = f"{r.get('avg_iou', 0):.3f}"
        print(f"{cid:8d} {r['n_points']:8d} {r['label']:<20s} {r['confidence']:5.0%} {r['votes']:6d} {iou_str:>6s}")
        label_summary[r['label']] += r['n_points']

    print(f"\n{'='*60}")
    print("Label Summary:")
    for label, count in sorted(label_summary.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(points)
        print(f"  {label:<20s}: {count:8d} points ({pct:.1f}%)")

    # Save colored point cloud
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
    o3d.io.write_point_cloud(str(args.output_dir / "semantic_pointcloud.ply"), pcd)

    print(f"\nSaved: {args.output_dir / 'semantic_pointcloud.ply'}")
    print(f"Saved: {args.output_dir / 'semantic_labels.json'}")
    print("DONE!")


if __name__ == "__main__":
    main()
