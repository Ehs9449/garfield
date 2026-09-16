#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge fine GARField clusters using SAM2 supporting-view "
            "evidence and GARField cosine similarity, then save both "
            "merged labels and a colored clustered point cloud."
        )
    )

    parser.add_argument(
        "--points",
        type=Path,
        required=True,
        help="points.npy corresponding exactly to the cluster labels",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Original fine cluster_labels.npy",
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="merge_candidates.csv produced by analyze_cluster_merge_candidates.py",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where merged outputs will be saved",
    )

    parser.add_argument(
        "--min-support",
        type=int,
        default=50,
        help="Minimum number of SAM2 supporting views",
    )

    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.90,
        help="Minimum GARField feature cosine similarity",
    )

    return parser.parse_args()


class UnionFind:
    def __init__(self, items):
        self.parent = {
            int(item): int(item)
            for item in items
        }

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(
                self.parent[item]
            )
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def save_clustered_pointcloud(points, labels, output_path):
    """
    Save the point cloud using the same random-color convention
    used by cluster_sweep_fine.py.
    """

    valid_labels = labels[labels >= 0]

    if len(valid_labels) > 0:
        n_clusters = int(valid_labels.max() + 1)
    else:
        n_clusters = 0

    np.random.seed(42)

    colors = np.random.rand(
        max(n_clusters, 1) + 1,
        3
    )

    point_colors = np.zeros(
        (len(labels), 3),
        dtype=np.float64
    )

    for i, label in enumerate(labels):
        if label >= 0:
            point_colors[i] = colors[
                int(label) % len(colors)
            ]
        else:
            point_colors[i] = [
                0.3,
                0.3,
                0.3
            ]

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(points)
    )

    pcd.colors = (
        o3d.utility.Vector3dVector(
            point_colors
        )
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        pcd
    )

    if not success:
        raise RuntimeError(
            f"Failed to write point cloud: {output_path}"
        )


def main():
    args = parse_args()

    print("=" * 70)
    print("CLUSTER MERGING")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load points and fine cluster labels
    # ----------------------------------------------------------

    points = np.load(
        args.points
    )

    labels = np.load(
        args.labels
    )

    if len(points) != len(labels):
        raise ValueError(
            "Point/label mismatch: "
            f"{len(points)} points vs "
            f"{len(labels)} labels"
        )

    original_clusters = sorted(
        int(cluster_id)
        for cluster_id in np.unique(labels)
        if cluster_id >= 0
    )

    print(f"Points: {len(points)}")
    print(
        f"Original clusters: "
        f"{len(original_clusters)}"
    )
    print(
        f"Minimum SAM2 support: "
        f"{args.min_support}"
    )
    print(
        f"Minimum cosine similarity: "
        f"{args.min_cosine}"
    )

    # ----------------------------------------------------------
    # Initialize union-find
    # ----------------------------------------------------------

    union_find = UnionFind(
        original_clusters
    )

    accepted_pairs = []

    # ----------------------------------------------------------
    # Read candidate pairs
    # ----------------------------------------------------------

    with open(
        args.candidates,
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "cluster_a",
            "cluster_b",
            "supporting_views",
            "cosine_similarity",
        }

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Candidate CSV does not contain "
                "the required columns."
            )

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

            cosine = float(
                row["cosine_similarity"]
            )

            if (
                support >= args.min_support
                and
                cosine >= args.min_cosine
            ):
                if (
                    cluster_a in union_find.parent
                    and
                    cluster_b in union_find.parent
                ):
                    union_find.union(
                        cluster_a,
                        cluster_b
                    )

                    accepted_pairs.append(
                        (
                            cluster_a,
                            cluster_b,
                            support,
                            cosine,
                        )
                    )

    # ----------------------------------------------------------
    # Report accepted pairwise evidence
    # ----------------------------------------------------------

    print(
        f"\nAccepted merge pairs: "
        f"{len(accepted_pairs)}"
    )

    for (
        cluster_a,
        cluster_b,
        support,
        cosine
    ) in accepted_pairs:

        print(
            f"{cluster_a:3d} + "
            f"{cluster_b:3d} | "
            f"views={support:3d} | "
            f"cosine={cosine:.4f}"
        )

    # ----------------------------------------------------------
    # Determine connected merge groups
    # ----------------------------------------------------------

    groups = {}

    for cluster_id in original_clusters:
        root = union_find.find(
            cluster_id
        )

        groups.setdefault(
            root,
            []
        ).append(
            cluster_id
        )

    merged_groups = [
        sorted(group)
        for group in groups.values()
        if len(group) > 1
    ]

    merged_groups.sort(
        key=lambda group: group[0]
    )

    print(
        f"\nMerged groups: "
        f"{len(merged_groups)}"
    )

    for group in merged_groups:
        print(
            "  "
            + " + ".join(
                str(cluster_id)
                for cluster_id in group
            )
        )

    # ----------------------------------------------------------
    # Apply union-find result to labels
    # ----------------------------------------------------------

    merged_labels = labels.copy()

    for cluster_id in original_clusters:
        merged_labels[
            labels == cluster_id
        ] = union_find.find(
            cluster_id
        )

    # ----------------------------------------------------------
    # Renumber clusters consecutively
    #
    # Noise remains -1.
    # ----------------------------------------------------------

    remaining_ids = sorted(
        int(cluster_id)
        for cluster_id
        in np.unique(merged_labels)
        if cluster_id >= 0
    )

    id_map = {
        old_id: new_id
        for new_id, old_id
        in enumerate(remaining_ids)
    }

    final_labels = np.full(
        merged_labels.shape,
        -1,
        dtype=np.int32
    )

    for old_id, new_id in id_map.items():
        final_labels[
            merged_labels == old_id
        ] = new_id

    final_cluster_count = len(
        remaining_ids
    )

    # ----------------------------------------------------------
    # Create output directory
    # ----------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels_path = (
        args.output_dir
        / "merged_labels.npy"
    )

    pointcloud_path = (
        args.output_dir
        / "merged_clustered_pointcloud.ply"
    )

    groups_path = (
        args.output_dir
        / "merged_groups.csv"
    )

    # ----------------------------------------------------------
    # Save merged labels
    # ----------------------------------------------------------

    np.save(
        labels_path,
        final_labels
    )

    # ----------------------------------------------------------
    # Save colored clustered point cloud
    # ----------------------------------------------------------

    save_clustered_pointcloud(
        points,
        final_labels,
        pointcloud_path
    )

    # ----------------------------------------------------------
    # Save merge-group record
    # ----------------------------------------------------------

    with open(
        groups_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "merged_group",
            "original_cluster_ids",
        ])

        for group_number, group in enumerate(
            merged_groups
        ):
            writer.writerow([
                group_number,
                " ".join(
                    str(cluster_id)
                    for cluster_id in group
                ),
            ])

    # ----------------------------------------------------------
    # Final report
    # ----------------------------------------------------------

    print(
        f"\nFinal clusters: "
        f"{final_cluster_count}"
    )

    print(
        f"Clusters reduced by: "
        f"{len(original_clusters) - final_cluster_count}"
    )

    print(
        f"\nSaved labels: "
        f"{labels_path}"
    )

    print(
        f"Saved point cloud: "
        f"{pointcloud_path}"
    )

    print(
        f"Saved merge groups: "
        f"{groups_path}"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
