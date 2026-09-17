#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import csv

import numpy as np
from nerfstudio.utils.eval_utils import eval_setup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAM2 cluster-pair evidence for 3D cluster merging."
    )

    parser.add_argument(
        "--points",
        type=Path,
        required=True,
        help="Path to points.npy",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to cluster_labels.npy",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="GARField/Nerfstudio config.yml",
    )

    parser.add_argument(
        "--sam-cache",
        type=Path,
        default=Path("data/sam_cache"),
        help="Directory containing sam_XXXXXX.npz files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )

    parser.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=0.50,
        help="Minimum visible-cluster fraction inside the same SAM2 mask",
    )

    parser.add_argument(
        "--crop-center",
        type=float,
        nargs=3,
        default=[0.02, -0.05, -0.15],
        metavar=("X", "Y", "Z"),
        help="Building centre, used to select cameras that face the building",
    )

    parser.add_argument(
        "--max-elevation-deg",
        type=float,
        default=35.0,
        help="Keep only cameras that look at the building from at most this "
             "elevation angle (0 = horizontal side view, 90 = straight down)",
    )

    parser.add_argument(
        "--max-incidence-deg",
        type=float,
        default=30.0,
        help="Keep only cameras whose horizontal direction is within this "
             "angle of a facade normal (see --facade-azimuths)",
    )

    parser.add_argument(
        "--facade-azimuths",
        type=float,
        nargs="+",
        default=[0.0, 90.0, 180.0, 270.0],
        help="Azimuth angles in degrees of the facade view directions",
    )

    parser.add_argument(
        "--max-offaxis-deg",
        type=float,
        default=40.0,
        help="Keep only cameras pointing within this angle of the building centre",
    )

    parser.add_argument(
        "--no-view-filter",
        action="store_true",
        help="Use every training camera, as before",
    )

    return parser.parse_args()


def select_side_view_cameras(cams, center, max_elevation_deg,
                             max_incidence_deg, facade_azimuths,
                             max_offaxis_deg):
    """Indices of training cameras that look at the building from the side.

    Oblique and near-nadir drone shots make two different facades fall in
    the same SAM2 mask, which produces wrong merge evidence. Only cameras
    that face a facade roughly head-on are kept.
    """
    center = np.asarray(center, dtype=np.float32)

    facade_dirs = np.array([
        [np.cos(np.radians(a)), np.sin(np.radians(a))]
        for a in facade_azimuths
    ], dtype=np.float32)

    keep = []

    for i in range(len(cams)):
        c2w = cams.camera_to_worlds[i].cpu().numpy()

        position = c2w[:3, 3]
        # Nerfstudio/OpenGL cameras look along local -Z.
        look = -c2w[:3, 2]
        look = look / (np.linalg.norm(look) + 1e-8)

        # Elevation of the viewing direction. Looking down gives look_z < 0.
        elevation = np.degrees(np.arcsin(np.clip(-look[2], -1.0, 1.0)))
        if elevation > max_elevation_deg:
            continue

        # Is the camera actually pointing at the building?
        to_center = center - position
        norm = np.linalg.norm(to_center)
        if norm < 1e-8:
            continue
        offaxis = np.degrees(np.arccos(
            np.clip(float(look @ (to_center / norm)), -1.0, 1.0)
        ))
        if offaxis > max_offaxis_deg:
            continue

        # Angle between the horizontal view direction and the nearest facade.
        horizontal = look[:2]
        h_norm = np.linalg.norm(horizontal)
        if h_norm < 1e-8:
            continue
        horizontal = horizontal / h_norm

        cosines = facade_dirs @ horizontal
        incidence = np.degrees(np.arccos(
            np.clip(float(cosines.max()), -1.0, 1.0)
        ))
        if incidence > max_incidence_deg:
            continue

        keep.append(i)

    return keep


def main():
    args = parse_args()

    points = np.load(args.points).astype(np.float32)
    labels = np.load(args.labels)

    if len(points) != len(labels):
        raise ValueError(
            f"Point/label mismatch: {len(points)} points vs {len(labels)} labels"
        )

    print(f"Points: {len(points)}")
    print(f"Clusters: {len(np.unique(labels[labels >= 0]))}")

    print("Loading Nerfstudio pipeline...")
    _, pipeline, _, _ = eval_setup(args.config)

    cams = pipeline.datamanager.train_dataset.cameras

    if args.no_view_filter:
        camera_indices = list(range(len(cams)))
        print(f"Using all {len(camera_indices)} training cameras")
    else:
        camera_indices = select_side_view_cameras(
            cams,
            args.crop_center,
            args.max_elevation_deg,
            args.max_incidence_deg,
            args.facade_azimuths,
            args.max_offaxis_deg,
        )
        print(
            f"Side-view cameras: {len(camera_indices)} / {len(cams)} "
            f"(elevation <= {args.max_elevation_deg}deg, "
            f"incidence <= {args.max_incidence_deg}deg)"
        )

        if len(camera_indices) < 10:
            raise SystemExit(
                "Too few side-view cameras. Relax --max-elevation-deg / "
                "--max-incidence-deg, set --facade-azimuths to your real "
                "facade directions, or pass --no-view-filter."
            )

    pair_views = defaultdict(set)

    for n_done, i in enumerate(camera_indices):
        cache = args.sam_cache / f"sam_{i:06d}.npz"

        if not cache.exists():
            continue

        data = np.load(cache)

        if "pixel_level_keys" not in data:
            continue

        keys = data["pixel_level_keys"]

        c2w = cams.camera_to_worlds[i].cpu().numpy()

        T = np.eye(4, dtype=np.float32)
        T[:3, :4] = c2w

        w2c = np.linalg.inv(T)

        pts_h = np.hstack([
            points,
            np.ones((len(points), 1), dtype=np.float32)
        ])

        pc = (w2c @ pts_h.T).T[:, :3]

        depth = -pc[:, 2]
        front = depth > 0.01

        xu = np.zeros(len(points), dtype=np.float32)
        yu = np.zeros(len(points), dtype=np.float32)

        xu[front] = pc[front, 0] / depth[front]
        yu[front] = -pc[front, 1] / depth[front]

        k1 = float(cams.distortion_params[i, 0].item())

        r2 = xu * xu + yu * yu

        xd = xu * (1.0 + k1 * r2)
        yd = yu * (1.0 + k1 * r2)

        fx = float(cams.fx[i].item())
        fy = float(cams.fy[i].item())
        cx = float(cams.cx[i].item())
        cy = float(cams.cy[i].item())

        w = int(cams.width[i].item())
        h = int(cams.height[i].item())

        px = np.rint(fx * xd + cx).astype(np.int32)
        py = np.rint(fy * yd + cy).astype(np.int32)

        valid = (
            front
            & (px >= 0)
            & (px < w)
            & (py >= 0)
            & (py < h)
        )

        valid_ids = np.where(valid)[0]

        if len(valid_ids) == 0:
            continue

        order = valid_ids[np.argsort(depth[valid_ids])]

        zbuf = np.full((h, w), np.inf, dtype=np.float32)
        visible = np.zeros(len(points), dtype=bool)

        for idx in order:
            x = px[idx]
            y = py[idx]

            if depth[idx] < zbuf[y, x]:
                zbuf[y, x] = depth[idx]
                visible[idx] = True

        vis_ids = np.where(
            visible & (labels >= 0)
        )[0]

        if len(vis_ids) == 0:
            continue

        cluster_ids, counts = np.unique(
            labels[vis_ids],
            return_counts=True
        )

        visible_cluster_counts = {
            int(c): int(n)
            for c, n in zip(cluster_ids, counts)
        }

        for level in range(keys.shape[2]):
            vals = keys[
                py[vis_ids],
                px[vis_ids],
                level
            ]

            for group_id in np.unique(vals):
                if group_id < 0:
                    continue

                mask_ids = vis_ids[
                    vals == group_id
                ]

                clusters_in_mask, counts_in_mask = np.unique(
                    labels[mask_ids],
                    return_counts=True
                )

                supported = []

                for c, n in zip(
                    clusters_in_mask,
                    counts_in_mask
                ):
                    c = int(c)

                    frac = (
                        n /
                        visible_cluster_counts[c]
                    )

                    if frac >= args.min_cluster_fraction:
                        supported.append(c)

                for a_idx in range(len(supported)):
                    for b_idx in range(
                        a_idx + 1,
                        len(supported)
                    ):
                        a = supported[a_idx]
                        b = supported[b_idx]

                        pair = tuple(
                            sorted((a, b))
                        )

                        pair_views[pair].add(i)

        if (n_done + 1) % 25 == 0:
            print(
                f"Processed {n_done + 1}/{len(camera_indices)} cameras"
            )

    rows = sorted(
        [
            (a, b, len(views))
            for (a, b), views in pair_views.items()
        ],
        key=lambda x: x[2],
        reverse=True
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "cluster_a",
            "cluster_b",
            "supporting_views"
        ])

        writer.writerows(rows)

    print("\nSaved:", args.output)
    print("\nStrongest cluster-pair evidence:")

    for a, b, support in rows[:30]:
        print(
            f"clusters {a:3d} + {b:3d}: "
            f"{support:4d} supporting views"
        )


if __name__ == "__main__":
    main()
