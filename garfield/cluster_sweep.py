#!/usr/bin/env python3
"""
HDBSCAN Parameter Sweep on saved GARField features.
Runs on HPC with cuml GPU HDBSCAN.

Usage:
    python cluster_sweep.py --input-dir /path/to/ortho_projection --output-dir /path/to/sweep_results
"""

import numpy as np
import argparse
import json
import time
import os
from pathlib import Path
from itertools import product

from cuml.cluster.hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import open3d as o3d


def run_sweep(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load saved features
    print("Loading features...")
    features = np.load(input_dir / "avg_features.npy")
    points = np.load(input_dir / "points.npy")
    colors_path = input_dir / "colors.npy"
    colors = np.load(colors_path) if colors_path.exists() else None

    print(f"  Points: {len(points)}")
    print(f"  Features: {features.shape}")
    print()

    # Parameter grid
    param_grid = {
        'min_cluster_size': [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000],
        'min_samples': [3, 5, 10, 15, 20, 30],
        'cluster_selection_epsilon': [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    }

    total_combos = len(param_grid['min_cluster_size']) * len(param_grid['min_samples']) * len(param_grid['cluster_selection_epsilon'])
    print(f"Total parameter combinations: {total_combos}")
    print("=" * 100)
    print(f"{'mcs':>6s} {'ms':>4s} {'eps':>5s} | {'n_clust':>8s} {'noise%':>7s} {'silhouette':>11s} {'calinski':>10s} {'davies_b':>10s} | {'time':>6s}")
    print("-" * 100)

    results = []
    best_silhouette = -1
    best_params_sil = None
    best_calinski = -1
    best_params_cal = None

    for mcs, ms, eps in product(
        param_grid['min_cluster_size'],
        param_grid['min_samples'],
        param_grid['cluster_selection_epsilon']
    ):
        try:
            start = time.time()

            clusterer = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=eps,
                allow_single_cluster=False,
            ).fit(features)

            labels = clusterer.labels_
            elapsed = time.time() - start

            nc = labels.max() + 1
            noise_pct = 100 * (labels == -1).sum() / len(labels)

            # Skip if too few or too many clusters
            if nc < 2 or nc > 500:
                continue

            valid = labels >= 0
            n_valid = int(valid.sum())
            n_valid_clusters = len(set(labels[valid]))

            if n_valid_clusters < 2 or n_valid < 100:
                continue

            # Compute metrics
            sample_size = min(5000, n_valid)

            sil_score = silhouette_score(
                features[valid], labels[valid],
                sample_size=sample_size
            )

            cal_score = calinski_harabasz_score(
                features[valid], labels[valid]
            )

            db_score = davies_bouldin_score(
                features[valid], labels[valid]
            )

            print(f"{mcs:6d} {ms:4d} {eps:5.2f} | {nc:8d} {noise_pct:6.1f}% {sil_score:11.4f} {cal_score:10.1f} {db_score:10.4f} | {elapsed:5.1f}s")

            result = {
                'min_cluster_size': int(mcs),
                'min_samples': int(ms),
                'cluster_selection_epsilon': float(eps),
                'n_clusters': int(nc),
                'noise_pct': float(noise_pct),
                'silhouette': float(sil_score),
                'calinski_harabasz': float(cal_score),
                'davies_bouldin': float(db_score),
                'n_valid_points': int(n_valid),
                'time_seconds': float(elapsed),
            }
            results.append(result)

            # Track best
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_params_sil = result.copy()

            if cal_score > best_calinski:
                best_calinski = cal_score
                best_params_cal = result.copy()

        except Exception as e:
            continue

    print("=" * 100)

    # Print best results
    print(f"\n{'=' * 60}")
    print("BEST RESULTS")
    print(f"{'=' * 60}")

    if best_params_sil:
        print(f"\nBest Silhouette ({best_params_sil['silhouette']:.4f}):")
        print(f"  mcs={best_params_sil['min_cluster_size']}, ms={best_params_sil['min_samples']}, eps={best_params_sil['cluster_selection_epsilon']}")
        print(f"  Clusters: {best_params_sil['n_clusters']}, Noise: {best_params_sil['noise_pct']:.1f}%")

    if best_params_cal:
        print(f"\nBest Calinski-Harabasz ({best_params_cal['calinski_harabasz']:.1f}):")
        print(f"  mcs={best_params_cal['min_cluster_size']}, ms={best_params_cal['min_samples']}, eps={best_params_cal['cluster_selection_epsilon']}")
        print(f"  Clusters: {best_params_cal['n_clusters']}, Noise: {best_params_cal['noise_pct']:.1f}%")

    # Sort results by different metrics
    results_by_sil = sorted(results, key=lambda x: x['silhouette'], reverse=True)
    results_by_cal = sorted(results, key=lambda x: x['calinski_harabasz'], reverse=True)
    results_by_db = sorted(results, key=lambda x: x['davies_bouldin'])  # lower is better

    print(f"\nTop 5 by Silhouette:")
    for r in results_by_sil[:5]:
        print(f"  mcs={r['min_cluster_size']:4d} ms={r['min_samples']:2d} eps={r['cluster_selection_epsilon']:.2f}: "
              f"{r['n_clusters']:3d} clusters, sil={r['silhouette']:.4f}, noise={r['noise_pct']:.1f}%")

    print(f"\nTop 5 by Calinski-Harabasz (higher=better):")
    for r in results_by_cal[:5]:
        print(f"  mcs={r['min_cluster_size']:4d} ms={r['min_samples']:2d} eps={r['cluster_selection_epsilon']:.2f}: "
              f"{r['n_clusters']:3d} clusters, cal={r['calinski_harabasz']:.1f}, noise={r['noise_pct']:.1f}%")

    print(f"\nTop 5 by Davies-Bouldin (lower=better):")
    for r in results_by_db[:5]:
        print(f"  mcs={r['min_cluster_size']:4d} ms={r['min_samples']:2d} eps={r['cluster_selection_epsilon']:.2f}: "
              f"{r['n_clusters']:3d} clusters, db={r['davies_bouldin']:.4f}, noise={r['noise_pct']:.1f}%")

    # Save all results
    with open(output_dir / "sweep_results.json", 'w') as f:
        json.dump({
            'total_combinations_tested': len(results),
            'best_silhouette': best_params_sil,
            'best_calinski': best_params_cal,
            'all_results': results,
        }, f, indent=2)
    print(f"\n✓ Saved results: {output_dir / 'sweep_results.json'}")

    # Save top 5 clustered point clouds for visual comparison
    print("\nSaving top 5 clustered point clouds...")
    for rank, r in enumerate(results_by_sil[:5]):
        c = HDBSCAN(
            min_cluster_size=r['min_cluster_size'],
            min_samples=r['min_samples'],
            cluster_selection_epsilon=r['cluster_selection_epsilon'],
            allow_single_cluster=False,
        ).fit(features)
        labels = c.labels_
        nc = labels.max() + 1

        np.random.seed(42)
        cmap = np.random.rand(max(nc, 1) + 1, 3)
        pc_colors = np.array([cmap[l % len(cmap)] if l >= 0 else [0.3, 0.3, 0.3] for l in labels])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)

        fname = f"rank{rank+1}_mcs{r['min_cluster_size']}_ms{r['min_samples']}_eps{r['cluster_selection_epsilon']:.2f}_{nc}clusters.ply"
        o3d.io.write_point_cloud(str(output_dir / fname), pcd)
        print(f"  Saved: {fname}")

    # Also save labels for best result
    best_c = HDBSCAN(
        min_cluster_size=best_params_sil['min_cluster_size'],
        min_samples=best_params_sil['min_samples'],
        cluster_selection_epsilon=best_params_sil['cluster_selection_epsilon'],
        allow_single_cluster=False,
    ).fit(features)
    np.save(output_dir / "best_labels.npy", best_c.labels_)

    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE: {len(results)} valid combinations tested")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN Parameter Sweep")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with avg_features.npy and points.npy")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save sweep results")
    args = parser.parse_args()
    run_sweep(args.input_dir, args.output_dir)
