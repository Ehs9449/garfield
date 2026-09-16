#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze candidate cluster merges using "
            "SAM2 supporting-view evidence and "
            "GARField feature cosine similarity."
        )
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to avg_features.npy",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to cluster_labels.npy",
    )

    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Path to sam2_cluster_pair_evidence.csv",
    )

    parser.add_argument(
        "--min-support",
        type=int,
        default=50,
        help="Minimum number of supporting SAM2 views",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV for ranked candidates",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    features = np.load(
        args.features
    ).astype(np.float32)

    labels = np.load(
        args.labels
    )

    if len(features) != len(labels):
        raise ValueError(
            f"Feature/label mismatch: "
            f"{len(features)} features vs "
            f"{len(labels)} labels"
        )

    print("=" * 70)
    print("CLUSTER MERGE CANDIDATE ANALYSIS")
    print("=" * 70)

    print(f"Features:    {args.features}")
    print(f"Labels:      {args.labels}")
    print(f"Evidence:    {args.evidence}")
    print(f"Min support: {args.min_support}")

    # --------------------------------------------------------------
    # Normalize every point feature
    # --------------------------------------------------------------

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    features_normalized = (
        features / (norms + 1e-12)
    )

    # --------------------------------------------------------------
    # Compute one normalized mean feature per cluster
    # --------------------------------------------------------------

    cluster_means = {}

    for cluster_id in np.unique(labels):

        if cluster_id < 0:
            continue

        cluster_mask = (
            labels == cluster_id
        )

        mean_feature = (
            features_normalized[
                cluster_mask
            ].mean(axis=0)
        )

        mean_norm = np.linalg.norm(
            mean_feature
        )

        mean_feature = (
            mean_feature /
            (mean_norm + 1e-12)
        )

        cluster_means[
            int(cluster_id)
        ] = mean_feature

    print(
        f"Clusters with feature means: "
        f"{len(cluster_means)}"
    )

    # --------------------------------------------------------------
    # Read SAM2 pair evidence
    # --------------------------------------------------------------

    candidates = []

    with open(
        args.evidence,
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            cluster_a = int(
                row["cluster_a"]
            )

            cluster_b = int(
                row["cluster_b"]
            )

            support = int(
                row["supporting_views"]
            )

            if support < args.min_support:
                continue

            if (
                cluster_a not in cluster_means
                or
                cluster_b not in cluster_means
            ):
                continue

            cosine_similarity = float(
                np.dot(
                    cluster_means[
                        cluster_a
                    ],
                    cluster_means[
                        cluster_b
                    ],
                )
            )

            candidates.append(
                (
                    support,
                    cosine_similarity,
                    cluster_a,
                    cluster_b,
                )
            )

    # --------------------------------------------------------------
    # Rank primarily by SAM2 support
    # and secondarily by feature similarity
    # --------------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )

    print("\nRanked merge candidates:\n")

    for (
        support,
        similarity,
        cluster_a,
        cluster_b
    ) in candidates:

        print(
            f"{cluster_a:3d} + "
            f"{cluster_b:3d} | "
            f"views={support:3d} | "
            f"cosine={similarity:.4f}"
        )

    # --------------------------------------------------------------
    # Optional CSV output
    # --------------------------------------------------------------

    if args.output is not None:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            args.output,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "cluster_a",
                "cluster_b",
                "supporting_views",
                "cosine_similarity",
            ])

            for (
                support,
                similarity,
                cluster_a,
                cluster_b
            ) in candidates:

                writer.writerow([
                    cluster_a,
                    cluster_b,
                    support,
                    similarity,
                ])

        print(
            "\nSaved ranked candidates:",
            args.output
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
