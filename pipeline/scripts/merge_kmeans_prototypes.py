#!/usr/bin/env python3
"""Hierarchical cosine merging of existing spherical K-means GARField clusters.

Input:  original GARField 256-D features, XYZ points (only for PLY output),
        and spherical K-means labels saved by experiment_suite.py.

For each starting K:
  1. L2-normalize point embeddings.
  2. Compute L2-normalized prototype mean of every original K-means cluster.
  3. Average-linkage hierarchical clustering with cosine distance among prototypes.
  4. Save a separate labeled PLY for each cosine-similarity cut.

This script does not train GARField, recompute K-means, modify Snakemake,
or overwrite existing experimental outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) != len(colors):
        raise ValueError("Points/colors have different lengths")
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices = np.empty(len(points), dtype=dt)
    vertices["x"], vertices["y"], vertices["z"] = (
        points[:, 0], points[:, 1], points[:, 2]
    )
    vertices["red"], vertices["green"], vertices["blue"] = (
        colors[:, 0], colors[:, 1], colors[:, 2]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def labels_stats(labels: np.ndarray, name: str, k: int, similarity: float | None):
    counts = np.bincount(labels)
    n = len(labels)
    return {
        "name": name,
        "initial_k": k,
        "cosine_threshold": "original" if similarity is None else similarity,
        "final_groups": int(len(counts)),
        "groups_merged": int(k - len(counts)),
        "points": n,
        "noise_fraction": 0.0,
        "median_group_size": float(np.median(counts)),
        "largest_group_size": int(counts.max()),
        "largest_group_fraction": float(counts.max() / n),
        "groups_under_100_points": int((counts < 100).sum()),
    }


def save_case(
    folder: Path,
    name: str,
    points: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    base_colors: np.ndarray,
    k: int,
    similarity: float | None,
):
    folder.mkdir(parents=True, exist_ok=True)
    # Merged groups retain the original K-means color of their lowest-ID member.
    # This makes unchanged groups more visually consistent across thresholds.
    colors = base_colors[anchors[labels]]
    ply = folder / f"{name}.ply"
    write_binary_ply(ply, points, colors)
    np.save(folder / f"{name}_labels.npy", labels.astype(np.int32))
    report = labels_stats(labels, name, k, similarity)
    (folder / f"{name}_metrics.json").write_text(json.dumps(report, indent=2))
    print(
        f"  {name:43s} "
        f"groups={report['final_groups']:4d}, "
        f"largest={report['largest_group_fraction']:.1%}, "
        f"small(<100)={report['groups_under_100_points']:3d}"
    )
    return report


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_base = repo_root / "outputs/towerlsu/ortho_projection_guided_plus_top"
    default_suite = repo_root / "outputs/towerlsu/feature_clustering_experiments"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=default_base / "points.npy")
    ap.add_argument("--features", type=Path, default=default_base / "avg_features.npy")
    ap.add_argument("--suite-root", type=Path, default=default_suite)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[200, 400, 600])
    ap.add_argument(
        "--cosine-thresholds", type=float, nargs="+",
        default=[0.995, 0.990, 0.980, 0.970, 0.960, 0.950, 0.930, 0.900],
    )
    args = ap.parse_args()

    out_root = args.output_root or args.suite_root / "14_kmeans_prototype_merge"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading points:   {args.points}", flush=True)
    p = np.load(args.points)
    print(f"Loading features: {args.features}", flush=True)
    f = np.load(args.features).astype(np.float32, copy=False)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected points Nx3, got {p.shape}")
    if f.ndim != 2 or len(f) != len(p):
        raise ValueError(f"Expected features NxD matching points, got {f.shape}")
    if not np.isfinite(f).all() or not np.isfinite(p).all():
        raise ValueError("Input contains NaN or infinity")
    if any(not (0.0 < t < 1.0) for t in args.cosine_thresholds):
        raise ValueError("All cosine thresholds must be strictly between 0 and 1")

    norms = np.linalg.norm(f, axis=1, keepdims=True)
    f = f / np.maximum(norms, 1e-12)
    print(f"Data: {len(p):,} points, {f.shape[1]} feature dimensions")

    records = []
    for k in args.k:
        source = args.suite_root / "03_spherical_kmeans" / f"spherical_kmeans_k{k:03d}_labels.npy"
        if not source.is_file():
            raise FileNotFoundError(f"Missing K={k} source labels: {source}")
        original_labels = np.load(source)
        if original_labels.shape != (len(p),):
            raise ValueError(f"Unexpected label shape for K={k}: {original_labels.shape}")
        if not np.issubdtype(original_labels.dtype, np.integer):
            raise ValueError(f"Labels for K={k} must be integers")
        if np.any(original_labels < 0):
            raise ValueError(f"K={k} contains negative labels; expected spherical K-means labels")

        original_ids, inverse = np.unique(original_labels, return_inverse=True)
        n_proto = len(original_ids)
        if n_proto != k:
            print(f"  Note: K={k} has {n_proto} nonempty clusters; using {n_proto} prototypes")
        group_root = out_root / f"k{k:03d}"
        group_root.mkdir(parents=True, exist_ok=True)
        np.save(group_root / "source_cluster_ids.npy", original_ids)

        print(f"\nK={k}: Computing {n_proto} mean prototypes...", flush=True)
        counts = np.bincount(inverse, minlength=n_proto)
        sums = np.zeros((n_proto, f.shape[1]), dtype=np.float64)
        # Only K x D accumulator; no N x N similarity matrix.
        np.add.at(sums, inverse, f)
        prototypes = sums / counts[:, None]
        prototype_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
        if (prototype_norm < 1e-12).any():
            raise ValueError(f"K={k} includes a near-zero prototype")
        prototypes = (prototypes / prototype_norm).astype(np.float32)
        np.save(group_root / "original_256d_normalized_prototypes.npy", prototypes)
        np.save(group_root / "prototype_point_counts.npy", counts)

        base_colors = np.random.default_rng(12345).integers(
            35, 240, size=(n_proto, 3), dtype=np.uint8
        )
        # Keep a copy of unmerged baseline in this folder for easy comparison.
        anchors = np.arange(n_proto, dtype=np.int32)
        record = save_case(
            group_root / "00_original",
            f"spherical_kmeans_k{k:03d}_original",
            p, inverse.astype(np.int32), anchors, base_colors, k, None,
        )
        records.append(record)

        if n_proto < 2:
            print("  Only one prototype, skipping hierarchical merging.")
            continue

        # Average linkage avoids simple single-edge transitive chaining.
        distances = pdist(prototypes, metric="cosine")
        tree = linkage(distances, method="average", optimal_ordering=False)
        np.save(group_root / "average_linkage_tree.npy", tree)

        previous_num = n_proto
        for threshold in sorted(set(args.cosine_thresholds), reverse=True):
            groups = fcluster(tree, t=1.0 - threshold, criterion="distance")
            # Renumber merged prototype groups consecutively from 0.
            _, proto_to_group = np.unique(groups, return_inverse=True)
            labels = proto_to_group[inverse].astype(np.int32)
            final_n = len(np.unique(proto_to_group))
            if final_n > previous_num:
                raise RuntimeError("Unexpected nonmonotone cluster count")
            previous_num = final_n

            group_anchors = np.full(final_n, n_proto, dtype=np.int32)
            np.minimum.at(group_anchors, proto_to_group, np.arange(n_proto))
            tag = f"cos_{threshold:.3f}".replace(".", "p")
            stem = f"spherical_kmeans_k{k:03d}_avg_{tag}"
            record = save_case(
                group_root / f"merge_{tag}", stem,
                p, labels, group_anchors, base_colors, k, threshold,
            )
            records.append(record)
            np.save(group_root / f"merge_{tag}" / "prototype_to_merged_group.npy", proto_to_group.astype(np.int32))

    csv_path = out_root / "prototype_merging_metrics.csv"
    if records:
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    print("\nDONE. Separate PLY folders:", out_root)
    print("Metrics:", csv_path)


if __name__ == "__main__":
    main()
