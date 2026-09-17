#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import open3d as o3d
from cuml.cluster.hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score


def run_hdbscan(features, min_cluster_size, min_samples, epsilon):
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        allow_single_cluster=False,
    )
    return clusterer.fit(features).labels_


def objective_score(features, labels, min_clusters=50, max_clusters=120):
    n_clusters = int(labels.max() + 1)
    noise_pct = 100.0 * float((labels == -1).sum()) / len(labels)

    if n_clusters < min_clusters or n_clusters > max_clusters:
        return -9999.0, n_clusters, noise_pct

    valid = labels >= 0
    if valid.sum() < 1000:
        return -9999.0, n_clusters, noise_pct

    if len(set(labels[valid])) < 2:
        return -9999.0, n_clusters, noise_pct

    sample_size = min(5000, int(valid.sum()))
    sil = silhouette_score(
        features[valid],
        labels[valid],
        sample_size=sample_size,
        random_state=42,
    )

    penalty = 0.002 * noise_pct
    score = float(sil - penalty)

    return score, n_clusters, noise_pct


def save_clustered_pointcloud(points, labels, output_path):
    n_clusters = int(labels.max() + 1)

    np.random.seed(42)
    colors = np.random.rand(max(n_clusters, 1) + 1, 3)

    point_colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        if label >= 0:
            point_colors[i] = colors[label % len(colors)]
        else:
            point_colors[i] = [0.3, 0.3, 0.3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    o3d.io.write_point_cloud(str(output_path), pcd)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HDBSCAN optimization for GARField feature clustering"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-clusters", type=int, default=50,
                        help="Reject trials that produce fewer clusters than this")
    parser.add_argument("--max-clusters", type=int, default=120,
                        help="Reject trials that produce more clusters than this")

    args = parser.parse_args()

    if args.min_clusters > args.max_clusters:
        parser.error(
            f"--min-clusters ({args.min_clusters}) is larger than "
            f"--max-clusters ({args.max_clusters}); every trial would be rejected"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    features = np.load(args.input_dir / "avg_features.npy")
    points = np.load(args.input_dir / "points.npy")

    print(f"  Points: {len(points)}")
    print(f"  Features: {features.shape}")

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.sample_size, len(features))
    sample_idx = rng.choice(len(features), size=n_sample, replace=False)
    features_sample = features[sample_idx]

    print(f"Using sample size: {n_sample}")
    print(f"Running Optuna trials: {args.n_trials}")
    print(f"Accepted cluster count: {args.min_clusters} to {args.max_clusters}")

    trial_records = []

    def objective(trial):
        min_cluster_size = trial.suggest_int(
            "min_cluster_size", 20, 300, log=True
        )
        min_samples = trial.suggest_int("min_samples", 3, 20)
        epsilon = trial.suggest_float(
            "cluster_selection_epsilon", 0.0, 0.12
        )

        start = time.time()

        try:
            labels = run_hdbscan(
                features_sample,
                min_cluster_size,
                min_samples,
                epsilon,
            )

            score, n_clusters, noise_pct = objective_score(
                features_sample, labels,
                args.min_clusters, args.max_clusters
            )

        except Exception as e:
            score = -9999.0
            n_clusters = -1
            noise_pct = 100.0
            print(f"Trial {trial.number} failed: {e}")

        elapsed = time.time() - start

        record = {
            "trial": trial.number,
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "cluster_selection_epsilon": float(epsilon),
            "score": float(score),
            "n_clusters": int(n_clusters),
            "noise_pct": float(noise_pct),
            "time_seconds": float(elapsed),
        }
        trial_records.append(record)

        print(
            f"trial={trial.number:03d} "
            f"mcs={min_cluster_size} "
            f"ms={min_samples} "
            f"eps={epsilon:.3f} "
            f"score={score:.4f} "
            f"clusters={n_clusters} "
            f"noise={noise_pct:.1f}% "
            f"time={elapsed:.1f}s"
        )

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    if study.best_value <= -9999.0:
        raise SystemExit(
            f"No trial produced between {args.min_clusters} and "
            f"{args.max_clusters} clusters. Widen the range in "
            f"clustering.optimization_min_clusters / "
            f"optimization_max_clusters, or raise --n-trials."
        )

    best_params = study.best_params

    print("\nBest parameters:")
    print(best_params)

    print("\nRunning final HDBSCAN on full point cloud...")
    final_labels = run_hdbscan(
        features,
        best_params["min_cluster_size"],
        best_params["min_samples"],
        best_params["cluster_selection_epsilon"],
    )

    final_score, final_n_clusters, final_noise_pct = objective_score(
        features, final_labels,
        args.min_clusters, args.max_clusters
    )

    if final_score <= -9999.0:
        print(
            f"  Note: the full cloud gave {final_n_clusters} clusters, "
            f"outside the {args.min_clusters}-{args.max_clusters} range tuned "
            f"on the sample. The labels are still saved."
        )

    np.save(args.output_dir / "cluster_labels.npy", final_labels)
    np.save(args.output_dir / "best_labels.npy", final_labels)

    # Save/copy geometry files expected by the semantic_labeling stage.
    np.save(args.output_dir / "points.npy", points)

    colors_path = args.input_dir / "colors.npy"
    if colors_path.exists():
        colors = np.load(colors_path)
        np.save(args.output_dir / "colors.npy", colors)

    save_clustered_pointcloud(
        points,
        final_labels,
        args.output_dir / "clustered_pointcloud.ply",
    )

    results = {
        "method": "optuna_tpe_hdbscan",
        "n_trials": args.n_trials,
        "sample_size": int(n_sample),
        "min_clusters": int(args.min_clusters),
        "max_clusters": int(args.max_clusters),
        "best_params": best_params,
        "final_metrics": {
            "score": float(final_score),
            "n_clusters": int(final_n_clusters),
            "noise_pct": float(final_noise_pct),
        },
        "trials": trial_records,
    }

    with open(args.output_dir / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone.")
    print(f"Saved: {args.output_dir / 'cluster_labels.npy'}")
    print(f"Saved: {args.output_dir / 'clustered_pointcloud.ply'}")
    print(f"Saved: {args.output_dir / 'optimization_results.json'}")


if __name__ == "__main__":
    main()
