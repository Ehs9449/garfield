#!/usr/bin/env python3
"""
Pipeline Visualization & Diagnostics

Checks each pipeline stage, shows statistics, and generates charts.
Run after any stage to see current progress.

Usage:
    conda activate nerfstudio3
    python pipeline/visualize_pipeline.py
    python pipeline/visualize_pipeline.py --stage 4    # Just stage 4
    python pipeline/visualize_pipeline.py --html       # Generate HTML report
"""

import numpy as np
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

# Load config
import yaml
CONFIG_PATH = Path("pipeline/config.yaml")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

REPORT_DIR = Path("outputs/pipeline_report")


def ensure_report_dir():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Stage 1: COLMAP
# ============================================================
def check_colmap():
    print("\n" + "=" * 60)
    print("  STAGE 1: COLMAP")
    print("=" * 60)

    dataset_path = Path(cfg.get("dataset", {}).get("path", "data/PFTdrone"))
    images_dir = dataset_path / "images"
    sparse_dir = dataset_path / "sparse" / "0"
    db_path = dataset_path / "database.db"

    # Check images
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) + \
                      list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
        print(f"  Images found: {len(image_files)}")
    else:
        print(f"  Images dir NOT FOUND: {images_dir}")
        return False

    # Check database
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  Database: {db_path.name} ({size_mb:.1f} MB)")
    else:
        print(f"  Database: NOT FOUND")

    # Check sparse reconstruction
    if sparse_dir.exists():
        files = list(sparse_dir.iterdir())
        print(f"  Sparse reconstruction: {sparse_dir}")
        print(f"  Files: {[f.name for f in files]}")

        # Try to read cameras/images/points
        cameras_bin = sparse_dir / "cameras.bin"
        images_bin = sparse_dir / "images.bin"
        points_bin = sparse_dir / "points3D.bin"

        for f in [cameras_bin, images_bin, points_bin]:
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    {f.name}: {size_mb:.1f} MB")

        print(f"  STATUS: COMPLETE ✓")
        return True
    else:
        print(f"  Sparse reconstruction: NOT FOUND")
        print(f"  STATUS: NOT STARTED ✗")
        return False


# ============================================================
# Stage 2: GARField Training
# ============================================================
def check_garfield():
    print("\n" + "=" * 60)
    print("  STAGE 2a: GARField (NeRF) Training")
    print("=" * 60)

    import glob
    configs = sorted(glob.glob("outputs/*/garfield/*/config.yml"),
                     key=os.path.getmtime, reverse=True)

    if configs:
        config_path = configs[0]
        print(f"  Config: {config_path}")
        print(f"  STATUS: COMPLETE ✓")
        return True
    else:
        print(f"  No GARField config found")
        print(f"  STATUS: NOT STARTED ✗")
        return False


def check_garfield_gauss():
    print("\n" + "=" * 60)
    print("  STAGE 2b: GARField-Gauss Training")
    print("=" * 60)

    import glob
    configs = sorted(glob.glob("outputs/*/garfield-gauss/*/config.yml"),
                     key=os.path.getmtime, reverse=True)

    if configs:
        config_path = configs[0]
        print(f"  Config: {config_path}")
        print(f"  STATUS: COMPLETE ✓")
        return True
    else:
        print(f"  No GARField-Gauss config found")
        print(f"  STATUS: NOT STARTED ✗")
        return False


# ============================================================
# Stage 3: Orthographic Projection
# ============================================================
def check_projection():
    print("\n" + "=" * 60)
    print("  STAGE 3: Orthographic Feature Projection")
    print("=" * 60)

    proj_dir = Path(cfg.get("projection", {}).get("output_dir", "outputs/ortho_projection_s005"))

    features_path = proj_dir / "avg_features.npy"
    points_path = proj_dir / "points.npy"

    if not features_path.exists() or not points_path.exists():
        # Try alternative location
        proj_dir = Path("outputs/ortho_projection_cropped")
        features_path = proj_dir / "avg_features.npy"
        points_path = proj_dir / "points.npy"

    if features_path.exists() and points_path.exists():
        features = np.load(features_path)
        points = np.load(points_path)

        print(f"  Points: {len(points)}")
        print(f"  Feature dim: {features.shape[1]}")
        print(f"  Point cloud bounds:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"  Feature stats:")
        print(f"    Mean: {features.mean():.4f}")
        print(f"    Std:  {features.std():.4f}")
        print(f"    Min:  {features.min():.4f}")
        print(f"    Max:  {features.max():.4f}")

        if HAS_MPL:
            ensure_report_dir()

            # Feature magnitude distribution
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Feature magnitude histogram
            magnitudes = np.linalg.norm(features, axis=1)
            axes[0].hist(magnitudes, bins=50, color='#534AB7', alpha=0.8, edgecolor='white')
            axes[0].set_title('Feature Magnitude Distribution')
            axes[0].set_xlabel('L2 Norm')
            axes[0].set_ylabel('Count')

            # Point cloud XY scatter
            axes[1].scatter(points[:, 0], points[:, 1], s=0.1, c=points[:, 2],
                           cmap='viridis', alpha=0.5)
            axes[1].set_title('Point Cloud (top view)')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].set_aspect('equal')

            # Point cloud XZ scatter
            axes[2].scatter(points[:, 0], points[:, 2], s=0.1, c=points[:, 1],
                           cmap='viridis', alpha=0.5)
            axes[2].set_title('Point Cloud (front view)')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Z')
            axes[2].set_aspect('equal')

            plt.tight_layout()
            plt.savefig(REPORT_DIR / "stage3_projection.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  Chart saved: {REPORT_DIR / 'stage3_projection.png'}")

        print(f"  STATUS: COMPLETE ✓")
        return True
    else:
        print(f"  Features/points NOT FOUND in {proj_dir}")
        print(f"  STATUS: NOT STARTED ✗")
        return False


# ============================================================
# Stage 4: Clustering
# ============================================================
def check_clustering():
    print("\n" + "=" * 60)
    print("  STAGE 4: HDBSCAN Clustering")
    print("=" * 60)

    features_dir = Path(cfg.get("matching", {}).get("features_dir", "outputs/ortho_projection_cropped"))
    labels_path = features_dir / "cluster_labels.npy"

    if not labels_path.exists():
        print(f"  Cluster labels NOT FOUND: {labels_path}")
        print(f"  STATUS: NOT STARTED ✗")
        return False

    labels = np.load(labels_path)
    n_clusters = int(labels.max()) + 1
    noise_count = (labels == -1).sum()

    print(f"  Total points: {len(labels)}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {noise_count} ({100 * noise_count / len(labels):.1f}%)")

    # Cluster size distribution
    cluster_sizes = []
    for i in range(n_clusters):
        size = (labels == i).sum()
        cluster_sizes.append(size)

    cluster_sizes = np.array(cluster_sizes)
    print(f"  Cluster sizes:")
    print(f"    Min: {cluster_sizes.min()}")
    print(f"    Max: {cluster_sizes.max()}")
    print(f"    Mean: {cluster_sizes.mean():.0f}")
    print(f"    Median: {np.median(cluster_sizes):.0f}")

    if HAS_MPL:
        ensure_report_dir()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Cluster size distribution
        axes[0].bar(range(n_clusters), sorted(cluster_sizes, reverse=True),
                    color='#534AB7', alpha=0.8, edgecolor='white')
        axes[0].set_title(f'Cluster Size Distribution ({n_clusters} clusters)')
        axes[0].set_xlabel('Cluster (sorted by size)')
        axes[0].set_ylabel('Number of Points')

        # Cluster size histogram
        axes[1].hist(cluster_sizes, bins=20, color='#1D9E75', alpha=0.8, edgecolor='white')
        axes[1].set_title('Cluster Size Histogram')
        axes[1].set_xlabel('Cluster Size')
        axes[1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(REPORT_DIR / "stage4_clustering.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved: {REPORT_DIR / 'stage4_clustering.png'}")

    print(f"  STATUS: COMPLETE ✓")
    return True


# ============================================================
# Stage 5: Render Views
# ============================================================
def check_render_views():
    print("\n" + "=" * 60)
    print("  STAGE 5: Render Labeling Views")
    print("=" * 60)

    views_dir = Path(cfg.get("labeling_views", {}).get("output_dir", "outputs/labeling_views"))
    params_path = views_dir / "view_params.json"

    if not params_path.exists():
        print(f"  View params NOT FOUND: {params_path}")
        print(f"  STATUS: NOT STARTED ✗")
        return False

    with open(params_path) as f:
        view_params = json.load(f)

    image_files = list(views_dir.glob("*.jpg")) + list(views_dir.glob("*.png"))

    print(f"  Views defined: {len(view_params)}")
    print(f"  Images rendered: {len(image_files)}")

    # Parse view angles
    elevations = defaultdict(int)
    for vp in view_params:
        name = vp['view_name']
        # Extract elevation from name like view_000_az000_el15
        parts = name.split('_')
        for p in parts:
            if p.startswith('el'):
                el = p[2:]
                elevations[f"{el}°"] += 1

    print(f"  Views by elevation:")
    for el, count in sorted(elevations.items()):
        print(f"    {el}: {count} views")

    if HAS_MPL and len(image_files) > 0:
        ensure_report_dir()

        # Show a grid of rendered views (up to 9)
        n_show = min(9, len(image_files))
        cols = 3
        rows = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes]
        axes = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

        for i in range(n_show):
            img = plt.imread(str(sorted(image_files)[i]))
            axes[i].imshow(img)
            axes[i].set_title(sorted(image_files)[i].stem, fontsize=8)
            axes[i].axis('off')

        for i in range(n_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Rendered Views ({len(image_files)} total)', fontsize=14)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "stage5_views.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved: {REPORT_DIR / 'stage5_views.png'}")

    print(f"  STATUS: COMPLETE ✓")
    return True


# ============================================================
# Stage 6: SAM 3 Inference
# ============================================================
def check_sam3_inference():
    print("\n" + "=" * 60)
    print("  STAGE 6: SAM 3 PCS Inference")
    print("=" * 60)

    masks_dir = Path("outputs/labeling_masks_finetuned")
    if not masks_dir.exists():
        masks_dir = Path("garfield/outputs/labeling_masks_finetuned")

    if not masks_dir.exists():
        print(f"  Masks directory NOT FOUND")
        print(f"  STATUS: NOT STARTED ✗")
        return False

    npz_files = sorted(masks_dir.glob("*.npz"))
    print(f"  Mask files: {len(npz_files)}")

    if len(npz_files) == 0:
        print(f"  STATUS: NOT STARTED ✗")
        return False

    # Aggregate detection counts
    total_detections = defaultdict(int)
    per_view_detections = defaultdict(lambda: defaultdict(int))

    for npz_path in npz_files:
        view_name = npz_path.stem.replace("_masks", "")
        data = np.load(npz_path)

        for key in data.files:
            if key.endswith("_masks"):
                label = key[:-6].replace("_", " ")
                count = len(data[key])
                total_detections[label] += count
                per_view_detections[view_name][label] = count

    print(f"\n  Total detections by class:")
    for label, count in sorted(total_detections.items(), key=lambda x: -x[1]):
        print(f"    {label:<20s}: {count}")

    print(f"\n  Total detections: {sum(total_detections.values())}")

    if HAS_MPL:
        ensure_report_dir()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Detection counts bar chart
        labels_sorted = sorted(total_detections.keys(), key=lambda x: -total_detections[x])
        counts = [total_detections[l] for l in labels_sorted]
        colors = ['#534AB7' if l in ['window', 'wall', 'opening', 'door', 'column']
                  else '#1D9E75' for l in labels_sorted]

        axes[0].barh(range(len(labels_sorted)), counts, color=colors, alpha=0.8, edgecolor='white')
        axes[0].set_yticks(range(len(labels_sorted)))
        axes[0].set_yticklabels(labels_sorted)
        axes[0].set_xlabel('Detection Count')
        axes[0].set_title('SAM 3 Detections by Class (all views)')
        axes[0].invert_yaxis()

        # Detections per view heatmap
        view_names = sorted(per_view_detections.keys())
        all_labels = sorted(total_detections.keys(), key=lambda x: -total_detections[x])[:8]
        heatmap_data = []
        for view in view_names:
            row = [per_view_detections[view].get(l, 0) for l in all_labels]
            heatmap_data.append(row)
        heatmap_data = np.array(heatmap_data)

        im = axes[1].imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd')
        axes[1].set_xticks(range(len(view_names)))
        axes[1].set_xticklabels([v.split('_')[1] for v in view_names], rotation=90, fontsize=6)
        axes[1].set_yticks(range(len(all_labels)))
        axes[1].set_yticklabels(all_labels, fontsize=8)
        axes[1].set_title('Detections per View (top 8 classes)')
        axes[1].set_xlabel('View')
        plt.colorbar(im, ax=axes[1], shrink=0.8)

        plt.tight_layout()
        plt.savefig(REPORT_DIR / "stage6_sam3.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved: {REPORT_DIR / 'stage6_sam3.png'}")

    print(f"  STATUS: COMPLETE ✓")
    return True


# ============================================================
# Stage 7: Semantic Labeling
# ============================================================
def check_semantic_labeling():
    print("\n" + "=" * 60)
    print("  STAGE 7: Semantic Labeling")
    print("=" * 60)

    output_dir = Path(cfg.get("matching", {}).get("output_dir", "outputs/semantic_labels_finetuned"))
    labels_path = output_dir / "semantic_labels.json"
    ply_path = output_dir / "semantic_pointcloud.ply"

    if not labels_path.exists():
        print(f"  Semantic labels NOT FOUND: {labels_path}")
        print(f"  STATUS: NOT STARTED ✗")
        return False

    with open(labels_path) as f:
        cluster_results = json.load(f)

    # Compute label summary
    label_summary = defaultdict(int)
    total_points = 0
    total_votes = 0
    clusters_with_votes = 0

    for cid, r in cluster_results.items():
        n_pts = r['n_points']
        total_points += n_pts
        label_summary[r['label']] += n_pts
        if r['votes'] > 0:
            clusters_with_votes += 1
            total_votes += r['votes']

    print(f"  Total clusters: {len(cluster_results)}")
    print(f"  Clusters with votes: {clusters_with_votes}")
    print(f"  Total points: {total_points}")
    print(f"  Total votes cast: {total_votes}")

    print(f"\n  Label distribution:")
    for label, count in sorted(label_summary.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_points if total_points > 0 else 0
        print(f"    {label:<20s}: {count:8d} points ({pct:.1f}%)")

    if ply_path.exists():
        size_mb = ply_path.stat().st_size / (1024 * 1024)
        print(f"\n  Semantic PLY: {ply_path} ({size_mb:.1f} MB)")

    # Per-label PLYs
    per_label_dir = output_dir / "per_label"
    if per_label_dir.exists():
        ply_files = list(per_label_dir.glob("*.ply"))
        print(f"  Per-label PLYs: {len(ply_files)} files")

    if HAS_MPL:
        ensure_report_dir()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Label distribution pie chart
        labels_sorted = sorted(label_summary.keys(), key=lambda x: -label_summary[x])
        sizes = [label_summary[l] for l in labels_sorted]
        colors_map = {
            'wall': '#B3B3A0', 'roof': '#CC3333', 'vegetation': '#009900',
            'opening': '#FF9900', 'ground': '#666666', 'unknown': '#4D4D4D',
            'window': '#00CCff', 'door': '#CC6600', 'column': '#E6E6E6',
            'beam': '#996600', 'ceiling': '#CCCCDD', 'sky': '#80B3FF',
        }
        colors = [colors_map.get(l, '#808080') for l in labels_sorted]

        wedges, texts, autotexts = axes[0].pie(
            sizes, labels=labels_sorted, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.85
        )
        for text in autotexts:
            text.set_fontsize(8)
        axes[0].set_title('Semantic Label Distribution')

        # Before vs after comparison (if we know baseline)
        baseline = {'roof': 62.1, 'ground': 1.2, 'unknown': 36.7}
        finetuned = {}
        for l, c in label_summary.items():
            finetuned[l] = 100 * c / total_points if total_points > 0 else 0

        all_labels = sorted(set(list(baseline.keys()) + list(finetuned.keys())),
                           key=lambda x: -finetuned.get(x, 0))

        x = range(len(all_labels))
        width = 0.35
        bars1 = axes[1].bar([i - width/2 for i in x],
                           [baseline.get(l, 0) for l in all_labels],
                           width, label='Off-the-shelf', color='#B4B2A9', alpha=0.8)
        bars2 = axes[1].bar([i + width/2 for i in x],
                           [finetuned.get(l, 0) for l in all_labels],
                           width, label='Fine-tuned', color='#534AB7', alpha=0.8)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('% of points')
        axes[1].set_title('Before vs After Fine-tuning')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(REPORT_DIR / "stage7_semantic.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved: {REPORT_DIR / 'stage7_semantic.png'}")

    print(f"  STATUS: COMPLETE ✓")
    return True


# ============================================================
# Summary
# ============================================================
def print_summary(results):
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)

    stages = [
        ("Stage 1", "COLMAP", results.get(1, False)),
        ("Stage 2a", "GARField (NeRF)", results.get("2a", False)),
        ("Stage 2b", "GARField-Gauss", results.get("2b", False)),
        ("Stage 3", "Ortho Projection", results.get(3, False)),
        ("Stage 4", "Clustering", results.get(4, False)),
        ("Stage 5", "Render Views", results.get(5, False)),
        ("Stage 6", "SAM 3 Inference", results.get(6, False)),
        ("Stage 7", "Semantic Labeling", results.get(7, False)),
    ]

    for stage_id, name, status in stages:
        icon = "✓" if status else "✗"
        color_status = "COMPLETE" if status else "PENDING"
        print(f"  {stage_id:<10s} {name:<25s} [{icon}] {color_status}")

    if REPORT_DIR.exists():
        charts = list(REPORT_DIR.glob("*.png"))
        if charts:
            print(f"\n  Charts saved to: {REPORT_DIR}/")
            for c in sorted(charts):
                print(f"    {c.name}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline Visualization")
    parser.add_argument("--stage", type=str, default="all",
                        help="Which stage to check (1-7, 2a, 2b, or 'all')")
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("  DRONE-TO-BIM PIPELINE DIAGNOSTICS")
    print("#" * 60)

    results = {}

    if args.stage in ["all", "1"]:
        results[1] = check_colmap()
    if args.stage in ["all", "2a"]:
        results["2a"] = check_garfield()
    if args.stage in ["all", "2b"]:
        results["2b"] = check_garfield_gauss()
    if args.stage in ["all", "3"]:
        results[3] = check_projection()
    if args.stage in ["all", "4"]:
        results[4] = check_clustering()
    if args.stage in ["all", "5"]:
        results[5] = check_render_views()
    if args.stage in ["all", "6"]:
        results[6] = check_sam3_inference()
    if args.stage in ["all", "7"]:
        results[7] = check_semantic_labeling()

    if args.stage == "all":
        print_summary(results)


if __name__ == "__main__":
    main()
