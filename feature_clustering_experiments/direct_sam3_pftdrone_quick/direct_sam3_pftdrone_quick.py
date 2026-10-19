#!/usr/bin/env python3
"""
Direct SAM3 -> 3D semantic labeling for legacy PFTdrone assets.

Purpose
-------
Quickly reuse the existing PFTdrone:
  - points.npy
  - view_params.json
  - fine-tuned SAM3 *_masks.npz files

WITHOUT using:
  - avg_features.npy
  - cluster_labels.npy
  - K400/HDBSCAN
  - cluster-to-mask matching

This script supports the legacy SAM3 NPZ format used by
match_clusters_to_masks_finetuned.py:
    window_masks, window_scores, window_boxes, ...
    wall_masks, wall_scores, wall_boxes, ...

It performs:
  1) perspective 3D point -> image projection
  2) z-buffer visibility filtering
  3) direct pointwise multi-view SAM3 fusion
  4) component-aware refinement for window/door/opening/column/beam
  5) architecture-over-sky cleanup
  6) semantic PLY + per-class PLY export

Later the same code can accept a registered LiDAR PLY instead of points.npy.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion

COLORS = {
    "window": (0, 205, 255),
    "wall": (179, 179, 153),
    "roof": (204, 51, 51),
    "door": (204, 102, 0),
    "opening": (255, 153, 0),
    "column": (230, 230, 230),
    "beam": (153, 76, 0),
    "ceiling": (204, 204, 230),
    "floor": (128, 128, 102),
    "curtain wall": (77, 153, 204),
    "ground": (102, 102, 102),
    "vegetation": (0, 153, 0),
    "sky": (128, 179, 255),
    "staircase": (255, 255, 0),
    "railing": (179, 76, 179),
    "unknown": (75, 75, 75),
}

COMPONENTS = ("opening", "door", "window", "column", "beam")
ARCHITECTURE = (
    "wall", "roof", "curtain wall", "opening",
    "door", "window", "column", "beam"
)

# Component refinement should correct broad facade labels, but should not
# overwrite ground/vegetation/floor/ceiling/staircase/railing.
COMPONENT_ALLOWED_BASE = {
    "unknown", "wall", "roof", "curtain wall",
    "opening", "door", "window", "column", "beam", "sky"
}

EROSION = {
    "window": 2,
    "door": 2,
    "opening": 2,
    "column": 1,
    "beam": 1,
}


def load_xyz(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        xyz = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".ply":
        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError(
                "Open3D is required for PLY input. "
                "Use points.npy or install open3d."
            ) from exc
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
    else:
        raise ValueError("Point input must be .npy or .ply")

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 XYZ array, got {xyz.shape}")
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("Point cloud is empty or contains invalid values")
    return xyz


def project_points(xyz: np.ndarray, rec: dict):
    c2w = np.asarray(rec["c2w"], dtype=np.float64)
    if c2w.shape != (3, 4):
        raise ValueError(
            f"{rec.get('view_name','?')}: expected c2w 3x4, got {c2w.shape}"
        )

    T = np.eye(4, dtype=np.float64)
    T[:3, :] = c2w
    w2c = np.linalg.inv(T)

    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    depth = -cam[:, 2]
    ids = np.flatnonzero(depth > 0.01)

    fx = float(rec["fx"])
    fy = float(rec["fy"])
    cx = float(rec["cx"])
    cy = float(rec["cy"])
    W = int(rec["img_w"])
    H = int(rec["img_h"])

    u = np.rint(fx * cam[ids, 0] / depth[ids] + cx).astype(np.int32)
    v = np.rint(-fy * cam[ids, 1] / depth[ids] + cy).astype(np.int32)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ids = ids[valid]
    return ids, u[valid], v[valid], depth[ids], W, H


def zbuffer(ids, u, v, depth, W, H, tile=4, abs_tol=0.02, rel_tol=0.025):
    if len(ids) == 0:
        return ids, u, v

    nx = (W + tile - 1) // tile
    ny = (H + tile - 1) // tile
    bin_id = (v // tile) * nx + (u // tile)

    nearest = np.full(nx * ny, np.inf, dtype=np.float32)
    np.minimum.at(nearest, bin_id, depth.astype(np.float32))
    near = nearest[bin_id]

    keep = depth <= near + np.maximum(abs_tol, rel_tol * near)
    return ids[keep], u[keep], v[keep]


def resize_mask(mask: np.ndarray, W: int, H: int) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.shape == (H, W):
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def classes_from_masks(mask_files):
    classes = set()
    for p in mask_files[: min(5, len(mask_files))]:
        with np.load(p, allow_pickle=False) as data:
            for key in data.files:
                if key.endswith("_masks"):
                    classes.add(key[:-6].replace("_", " "))
    return sorted(classes)


def key_for_class(cls: str) -> str:
    return cls.replace(" ", "_")


def write_ply(path: Path, xyz: np.ndarray, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.empty((len(labels), 3), dtype=np.uint8)
    rgb[:] = COLORS["unknown"]
    for cls in np.unique(labels):
        rgb[labels == cls] = COLORS.get(str(cls), COLORS["unknown"])

    arr = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, required=True,
                   help="points.npy now; registered LiDAR .ply is also supported")
    p.add_argument("--views", type=Path, required=True,
                   help="directory containing view_params.json")
    p.add_argument("--masks", type=Path, required=True,
                   help="legacy SAM3 *_masks.npz directory")
    p.add_argument("--output", type=Path, required=True)

    p.add_argument("--min-sam3-score", type=float, default=0.30)

    # Initial direct pointwise fusion.
    p.add_argument("--min-evidence-views", type=int, default=2)
    p.add_argument("--min-point-ratio", type=float, default=0.15)
    p.add_argument("--min-weighted-score", type=float, default=0.12)

    # Component-aware refinement.
    p.add_argument("--component-min-views", type=int, default=3)
    p.add_argument("--component-min-ratio", type=float, default=0.25)
    p.add_argument("--component-min-evidence", type=float, default=0.12)

    p.add_argument("--tile", type=int, default=4)
    p.add_argument("--abs-tol", type=float, default=0.02)
    p.add_argument("--rel-tol", type=float, default=0.025)

    a = p.parse_args()
    t0 = time.perf_counter()

    view_json = a.views / "view_params.json"
    if not a.points.is_file():
        raise FileNotFoundError(a.points)
    if not view_json.is_file():
        raise FileNotFoundError(view_json)
    if not a.masks.is_dir():
        raise FileNotFoundError(a.masks)

    xyz = load_xyz(a.points)
    records = json.loads(view_json.read_text())
    mask_files = sorted(a.masks.glob("*_masks.npz"))

    if not records:
        raise RuntimeError("No views in view_params.json")
    if not mask_files:
        raise RuntimeError("No *_masks.npz files found")

    mask_lookup = {
        p.stem.replace("_masks", ""): p
        for p in mask_files
    }

    classes = classes_from_masks(mask_files)
    if not classes:
        raise RuntimeError("Could not infer semantic classes from NPZ keys")

    print("=" * 70)
    print("DIRECT SAM3 -> 3D POINT SEMANTICS")
    print("=" * 70)
    print(f"Points: {len(xyz):,}")
    print(f"Views in JSON: {len(records)}")
    print(f"SAM3 mask files: {len(mask_files)}")
    print("Classes:", ", ".join(classes))
    print()
    print("NOT USED: avg_features.npy, cluster_labels.npy, K400, HDBSCAN")
    print()

    n = len(xyz)
    C = len(classes)
    class_index = {c: i for i, c in enumerate(classes)}

    # Initial broad semantic evidence: opportunities = every visible view.
    global_opp = np.zeros(n, dtype=np.uint16)
    hits = np.zeros((C, n), dtype=np.uint16)
    conf_sum = np.zeros((C, n), dtype=np.float32)

    # Component-aware evidence: opportunities only in views where that class
    # actually has at least one accepted SAM3 mask.
    comp_hits = {
        c: np.zeros(n, dtype=np.uint16)
        for c in COMPONENTS if c in class_index
    }
    comp_opp = {
        c: np.zeros(n, dtype=np.uint16)
        for c in COMPONENTS if c in class_index
    }
    comp_conf = {
        c: np.zeros(n, dtype=np.float32)
        for c in COMPONENTS if c in class_index
    }

    rows = []
    used_views = 0

    for i, rec in enumerate(records, 1):
        view_name = rec["view_name"]
        npz_path = mask_lookup.get(view_name)
        if npz_path is None:
            print(f"[{i}/{len(records)}] {view_name}: no mask file, skip")
            continue

        ids, u, v, depth, W, H = project_points(xyz, rec)
        projected = len(ids)
        ids, u, v = zbuffer(
            ids, u, v, depth, W, H,
            tile=a.tile, abs_tol=a.abs_tol, rel_tol=a.rel_tol
        )
        visible = len(ids)
        global_opp[ids] += 1
        used_views += 1

        view_mask_counts = {}
        view_point_hits = {}

        with np.load(npz_path, allow_pickle=False) as data:
            for cls in classes:
                key = key_for_class(cls)
                mk = f"{key}_masks"
                sk = f"{key}_scores"
                if mk not in data:
                    view_mask_counts[cls] = 0
                    view_point_hits[cls] = 0
                    continue

                masks = data[mk]
                scores = (
                    data[sk]
                    if sk in data
                    else np.ones(len(masks), dtype=np.float32)
                )

                point_conf = np.zeros(visible, dtype=np.float32)
                accepted_masks = 0

                # Component-specific version uses erosion.
                component_point_conf = (
                    np.zeros(visible, dtype=np.float32)
                    if cls in comp_hits else None
                )

                for mask, score in zip(masks, scores):
                    score = float(score)
                    if score < a.min_sam3_score:
                        continue

                    mask = resize_mask(mask, W, H)
                    if not mask.any():
                        continue
                    accepted_masks += 1

                    inside = mask[v, u] if visible else np.zeros(0, bool)
                    np.maximum(
                        point_conf,
                        inside.astype(np.float32) * score,
                        out=point_conf,
                    )

                    if component_point_conf is not None:
                        er = EROSION.get(cls, 1)
                        cmask = (
                            binary_erosion(mask, iterations=er)
                            if er > 0 else mask
                        )
                        if cmask.any():
                            cinside = cmask[v, u] if visible else np.zeros(0, bool)
                            np.maximum(
                                component_point_conf,
                                cinside.astype(np.float32) * score,
                                out=component_point_conf,
                            )

                ci = class_index[cls]
                point_supported = point_conf > 0
                if visible:
                    hits[ci, ids[point_supported]] += 1
                    conf_sum[ci, ids] += point_conf

                if cls in comp_hits and accepted_masks > 0 and visible:
                    # Crucial: component opportunity exists only in a view where
                    # that component had an accepted SAM3 detection.
                    comp_opp[cls][ids] += 1
                    cs = component_point_conf > 0
                    comp_hits[cls][ids[cs]] += 1
                    comp_conf[cls][ids] += component_point_conf

                view_mask_counts[cls] = accepted_masks
                view_point_hits[cls] = int(point_supported.sum())

        rows.append({
            "view": view_name,
            "projected_points": projected,
            "zbuffer_visible_points": visible,
            "accepted_masks": json.dumps(view_mask_counts, sort_keys=True),
            "point_hits": json.dumps(view_point_hits, sort_keys=True),
        })

        print(
            f"[{i}/{len(records)}] {view_name}: "
            f"visible={visible:,}, "
            f"masks={sum(view_mask_counts.values())}",
            flush=True,
        )

    if used_views == 0:
        raise RuntimeError("No matching view/mask pairs were usable")

    # ------------------------------------------------------------
    # Stage 1: direct pointwise multi-class fusion
    # ------------------------------------------------------------
    denom = np.maximum(global_opp, 1).astype(np.float32)
    hit_ratio = hits.astype(np.float32) / denom[None, :]
    weighted = conf_sum / denom[None, :]

    eligible = (
        (hits >= a.min_evidence_views)
        & (hit_ratio >= a.min_point_ratio)
        & (weighted >= a.min_weighted_score)
    )

    score = weighted.copy()
    score[~eligible] = -1.0

    winner = np.argmax(score, axis=0)
    best = score[winner, np.arange(n)]
    has_label = best >= 0

    labels = np.full(n, "unknown", dtype="<U32")
    class_arr = np.asarray(classes, dtype="<U32")
    labels[has_label] = class_arr[winner[has_label]]

    stage1_labels = labels.copy()

    # ------------------------------------------------------------
    # Stage 2: component-aware refinement
    # ------------------------------------------------------------
    comp_score_matrix = []
    comp_names = []

    for cls in COMPONENTS:
        if cls not in comp_hits:
            continue

        h = comp_hits[cls].astype(np.float32)
        o = np.maximum(comp_opp[cls], 1).astype(np.float32)
        ratio = h / o
        mean_conf = comp_conf[cls] / np.maximum(h, 1.0)
        evidence = ratio * mean_conf

        supported = (
            (comp_hits[cls] >= a.component_min_views)
            & (ratio >= a.component_min_ratio)
            & (evidence >= a.component_min_evidence)
        )

        e = evidence.copy()
        e[~supported] = -1.0
        comp_score_matrix.append(e)
        comp_names.append(cls)

    if comp_score_matrix:
        cm = np.stack(comp_score_matrix, axis=0)
        cw = np.argmax(cm, axis=0)
        cbest = cm[cw, np.arange(n)]
        cvalid = cbest >= 0
        cclass = np.asarray(comp_names, dtype="<U32")[cw]

        base_allowed = np.isin(labels, list(COMPONENT_ALLOWED_BASE))
        change = cvalid & base_allowed
        labels[change] = cclass[change]

    stage2_labels = labels.copy()

    # ------------------------------------------------------------
    # Stage 3: architecture > sky cleanup
    # ------------------------------------------------------------
    if "sky" in class_index:
        sky_points = labels == "sky"
        arch_scores = []
        arch_names = []

        for cls in ARCHITECTURE:
            if cls not in class_index:
                continue
            ci = class_index[cls]

            # Any valid architectural hit can challenge sky; normalized
            # confidence chooses among architecture classes.
            s = weighted[ci].copy()
            s[hits[ci] == 0] = -1.0
            arch_scores.append(s)
            arch_names.append(cls)

        if arch_scores:
            am = np.stack(arch_scores, axis=0)
            aw = np.argmax(am, axis=0)
            abest = am[aw, np.arange(n)]
            aclass = np.asarray(arch_names, dtype="<U32")[aw]
            replace = sky_points & (abest >= 0)
            labels[replace] = aclass[replace]

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    out = a.output
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "semantic_labels_direct.npy", labels)
    np.save(out / "semantic_labels_stage1_pointwise.npy", stage1_labels)
    np.save(out / "semantic_labels_stage2_component_aware.npy", stage2_labels)
    np.save(out / "point_visibility_opportunities.npy", global_opp)

    write_ply(out / "semantic_pointcloud.ply", xyz, labels)

    per_class = out / "per_class"
    per_class.mkdir(exist_ok=True)
    counts = Counter(labels.tolist())
    for cls, count in sorted(counts.items()):
        sel = labels == cls
        if sel.any():
            write_ply(
                per_class / f"{cls.replace(' ', '_')}.ply",
                xyz[sel],
                labels[sel],
            )

    with (out / "class_counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["semantic_class", "points"])
        for cls, count in sorted(counts.items()):
            w.writerow([cls, count])

    if rows:
        with (out / "per_view_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    elapsed = time.perf_counter() - t0
    summary = {
        "method": "direct_sam3_pointwise_legacy_masks",
        "point_source": str(a.points),
        "points": int(n),
        "views_in_json": len(records),
        "views_used": int(used_views),
        "mask_files": len(mask_files),
        "classes": classes,
        "class_counts": dict(counts),
        "unknown_points": int((labels == "unknown").sum()),
        "elapsed_seconds": elapsed,
        "uses_garfield_features": False,
        "uses_clustering": False,
        "uses_cluster_semantic_matching": False,
        "note": (
            "This quick test may reuse point coordinates and rendered views "
            "that were generated by an earlier GARField/GARField-Gauss run. "
            "The semantic assignment itself uses only XYZ + cameras + SAM3 masks."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Elapsed: {elapsed:.1f} s")
    print("Final counts:", dict(counts))
    print("Semantic cloud:", out / "semantic_pointcloud.ply")
    print("Per-class clouds:", per_class)
    print("Summary:", out / "summary.json")


if __name__ == "__main__":
    main()
