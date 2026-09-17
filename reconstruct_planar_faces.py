#!/usr/bin/env python3
"""
Trimmed planar face extraction from a photogrammetry point cloud.

Replaces visualize_cropped_density_planes.py.

Four changes remove the redundant interior planes:

  1. Normal gating   - a point only votes for a direction if its own surface
                       normal is parallel to that direction. A wall point can
                       no longer create evidence for a horizontal plane.
  2. Peak detection  - plane offsets come from prominent peaks in the signed
                       distance histogram, not from a greedy loop that keeps
                       consuming points until the cloud is empty.
  3. Occupancy faces - a face is the set of grid cells that really hold points
                       (after a small morphological closing), not the bounding
                       rectangle of a connected region.
  4. Occlusion test  - a candidate face is dropped when the building shell
                       blocks the view on BOTH sides of it. Such a face is
                       buried inside the building and cannot be an exterior
                       surface.

Usage
-----
    python reconstruct_planar_faces.py --cloud lsu_tower.ply

    # directions from your own principal-direction script (3 lines, "x y z")
    python reconstruct_planar_faces.py --cloud lsu_tower.ply --dirs dirs.txt

Outputs
-------
    planar_faces.ply    coloured triangle mesh of the accepted faces
    planar_faces.json   one record per face (plane, normal, offset, corners,
                        area, point count) - input for the B-rep step
"""

import argparse
import json

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.signal import find_peaks


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

DEFAULT_DIRS = {
    "V":  [-0.20815405,  0.97774725,  0.02611914],
    "H1": [ 0.02266596, -0.02187495,  0.99950375],
    "H2": [ 0.97783339,  0.20864277, -0.01760823],
}

COLORS = {
    "V":  [1.00, 0.25, 0.25],
    "H1": [0.25, 1.00, 0.25],
    "H2": [0.25, 0.45, 1.00],
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_cloud(path, normal_radius, normal_nn, auto):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)

    if len(points) == 0:
        raise SystemExit(f"empty cloud: {path}")

    extent = points.max(axis=0) - points.min(axis=0)
    diagonal = float(np.linalg.norm(extent))

    if len(points) < 2_000_000:
        spacing = float(np.median(pcd.compute_nearest_neighbor_distance()))
    else:
        spacing = diagonal / 2000.0
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = diagonal / 1000.0

    had_normals = pcd.has_normals()
    if not had_normals:
        radius = 6.0 * spacing if auto else normal_radius
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=normal_nn
            )
        )

    normals = np.asarray(pcd.normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths

    print(f"loaded {len(points)} points from {path}")
    print(f"  bounding box : {extent[0]:.3f} x {extent[1]:.3f} "
          f"x {extent[2]:.3f}  (diagonal {diagonal:.3f})")
    print(f"  point spacing: {spacing:.4f}")
    print(f"  normals      : {'from file' if had_normals else 'estimated'}")

    return points, normals, diagonal, spacing


def auto_parameters(args, diagonal, spacing, count):
    """Scale the metre-based defaults to whatever units the cloud uses."""
    args.thickness = max(3.0 * spacing, diagonal / 700.0)
    args.cell = max(2.0 * spacing, diagonal / 400.0)
    args.voxel = max(4.0 * spacing, diagonal / 200.0)
    args.min_side = 4.0 * args.cell
    args.min_area = 60.0 * args.cell * args.cell
    args.max_hole = 60.0 * args.cell * args.cell
    args.min_inliers = max(50, count // 2000)
    args.normal_radius = 6.0 * spacing

    print("\nauto parameters")
    print(f"  --thickness   {args.thickness:.4f}")
    print(f"  --cell        {args.cell:.4f}")
    print(f"  --voxel       {args.voxel:.4f}")
    print(f"  --min-side    {args.min_side:.4f}")
    print(f"  --min-area    {args.min_area:.4f}")
    print(f"  --max-hole    {args.max_hole:.4f}")
    print(f"  --min-inliers {args.min_inliers}")
    return args


def read_directions(path):
    if path is None:
        return {k: np.array(v, float) for k, v in DEFAULT_DIRS.items()}

    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.replace(",", " ").split()])

    names = ["V", "H1", "H2"]
    return {names[i]: np.array(rows[i], float) for i in range(len(rows))}


# --- occupancy grid over the whole cloud, used by the occlusion test -------

class Occupancy:
    """Sparse 3D voxel occupancy with vectorised membership queries."""

    def __init__(self, points, voxel):
        self.voxel = voxel
        self.origin = points.min(axis=0) - voxel
        index = np.floor((points - self.origin) / voxel).astype(np.int64)
        self.dims = index.max(axis=0) + 2
        keys = (index[:, 0] * self.dims[1] + index[:, 1]) * self.dims[2] \
            + index[:, 2]
        self.keys = np.unique(keys)
        self.upper = self.origin + self.dims * voxel

    def occupied(self, query):
        index = np.floor((query - self.origin) / self.voxel).astype(np.int64)
        inside = np.all((index >= 0) & (index < self.dims), axis=1)
        index = np.clip(index, 0, self.dims - 1)
        keys = (index[:, 0] * self.dims[1] + index[:, 1]) * self.dims[2] \
            + index[:, 2]
        pos = np.searchsorted(self.keys, keys)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        return (self.keys[pos] == keys) & inside


def blocked_fraction(occupancy, samples, direction, skip, reach):
    """Fraction of samples whose ray hits an occupied voxel before leaving."""
    step = occupancy.voxel
    offsets = np.arange(skip, reach, step)
    hit = np.zeros(len(samples), dtype=bool)

    for t in offsets:
        pending = ~hit
        if not pending.any():
            break
        probe = samples[pending] + direction * t
        found = occupancy.occupied(probe)
        idx = np.flatnonzero(pending)
        hit[idx[found]] = True

    return hit.mean() if len(hit) else 0.0


# --- plane offsets ---------------------------------------------------------

def find_plane_offsets(values, threshold, min_inliers, peak_ratio, merge_gap):
    """Prominent peaks of the signed-distance histogram."""
    if len(values) < min_inliers:
        return []

    bin_width = threshold / 2.0
    # pad with empty bins so a plane at the very edge of the cloud
    # still looks like a peak to find_peaks
    pad = 4 * threshold
    low, high = values.min() - pad, values.max() + pad
    bins = max(int(np.ceil((high - low) / bin_width)), 8)
    counts, edges = np.histogram(values, bins=bins, range=(low, high))

    # window roughly as wide as the plane thickness
    width = max(int(round(threshold / bin_width)), 1)
    kernel = np.ones(2 * width + 1)
    smooth = np.convolve(counts.astype(float), kernel, mode="same")

    if smooth.max() <= 0:
        return []

    height = max(min_inliers, peak_ratio * smooth.max())
    peaks, _ = find_peaks(smooth, height=height, prominence=0.35 * height,
                          distance=max(int(round(merge_gap / bin_width)), 1))

    centers = 0.5 * (edges[:-1] + edges[1:])
    offsets = []

    for peak in peaks:
        offset = centers[peak]
        for _ in range(4):                       # snap onto the real points
            mask = np.abs(values - offset) <= threshold
            if np.count_nonzero(mask) < min_inliers:
                break
            new = np.median(values[mask])
            if abs(new - offset) < 1e-6:
                offset = new
                break
            offset = new
        mask = np.abs(values - offset) <= threshold
        if np.count_nonzero(mask) >= min_inliers:
            offsets.append((offset, np.count_nonzero(mask)))

    # drop near-duplicates, keep the better supported one
    offsets.sort(key=lambda item: -item[1])
    kept = []
    for offset, count in offsets:
        if all(abs(offset - other) > merge_gap for other, _ in kept):
            kept.append((offset, count))

    kept.sort(key=lambda item: item[0])
    return kept


# --- face outline ----------------------------------------------------------

def greedy_rectangles(mask):
    """Cover a boolean mask with a small number of axis-aligned rectangles."""
    work = mask.copy()
    rows, cols = work.shape
    rectangles = []

    for i in range(rows):
        j = 0
        while j < cols:
            if not work[i, j]:
                j += 1
                continue
            j2 = j
            while j2 + 1 < cols and work[i, j2 + 1]:
                j2 += 1
            i2 = i
            while i2 + 1 < rows and work[i2 + 1, j:j2 + 1].all():
                i2 += 1
            rectangles.append((i, j, i2, j2))
            work[i:i2 + 1, j:j2 + 1] = False
            j = j2 + 1

    return rectangles


def fill_small_holes(mask, max_cells):
    """Close windows and gaps, but keep real openings such as a courtyard."""
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask

    labels, total = ndimage.label(holes)
    if total == 0:
        return mask

    sizes = ndimage.sum(holes, labels, index=np.arange(1, total + 1))
    small = np.concatenate([[False], sizes <= max_cells])
    return mask | small[labels]


def build_mask(a, b, cell, min_hits, close_size, max_hole_cells):
    a0, b0 = a.min(), b.min()
    ia = np.floor((a - a0) / cell).astype(int)
    ib = np.floor((b - b0) / cell).astype(int)

    counts = np.zeros((ia.max() + 1, ib.max() + 1), dtype=np.int32)
    np.add.at(counts, (ia, ib), 1)

    mask = counts >= min_hits
    if close_size > 0:
        structure = np.ones((close_size, close_size), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure)
    if max_hole_cells > 0:
        mask = fill_small_holes(mask, max_hole_cells)

    return mask, counts, a0, b0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", default="lsu_tower.ply")
    parser.add_argument("--dirs", default=None,
                        help="text file with three principal directions")
    parser.add_argument("--out", default="planar_faces")

    parser.add_argument("--thickness", type=float, default=0.06,
                        help="half thickness of a plane slab, metres")
    parser.add_argument("--angle", type=float, default=25.0,
                        help="max angle between point normal and direction")
    parser.add_argument("--min-inliers", type=int, default=300)
    parser.add_argument("--peak-ratio", type=float, default=0.06,
                        help="peak height as a fraction of the strongest peak")

    parser.add_argument("--cell", type=float, default=0.10,
                        help="in-plane grid cell, metres")
    parser.add_argument("--min-hits", type=int, default=2,
                        help="points needed before a cell counts as filled")
    parser.add_argument("--close", type=int, default=3,
                        help="morphological closing size in cells")
    parser.add_argument("--max-hole", type=float, default=1.0,
                        help="holes smaller than this area are filled, "
                             "larger ones stay open (square metres)")
    parser.add_argument("--min-area", type=float, default=1.0,
                        help="smallest accepted face area, square metres")
    parser.add_argument("--min-side", type=float, default=0.4,
                        help="smallest accepted face side, metres")

    parser.add_argument("--voxel", type=float, default=0.20,
                        help="voxel size of the visibility grid")
    parser.add_argument("--occlusion", type=float, default=0.55,
                        help="drop a face when both sides are blocked above "
                             "this fraction of samples")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--no-occlusion-test", action="store_true")

    parser.add_argument("--normal-radius", type=float, default=0.30)
    parser.add_argument("--normal-nn", type=int, default=40)

    parser.add_argument("--auto", action="store_true",
                        help="scale all size parameters to the cloud, "
                             "use this when the cloud is not in metres")

    args = parser.parse_args()

    points, normals, diagonal, spacing = load_cloud(
        args.cloud, args.normal_radius, args.normal_nn, args.auto)

    if args.auto:
        args = auto_parameters(args, diagonal, spacing, len(points))

    directions = read_directions(args.dirs)

    occupancy = Occupancy(points, args.voxel)
    extent = points.max(axis=0) - points.min(axis=0)
    reach = float(np.linalg.norm(extent))
    skip = max(3 * args.voxel, 2 * args.thickness + args.voxel)

    cos_limit = np.cos(np.deg2rad(args.angle))
    merge_gap = 4 * args.thickness

    mesh = o3d.geometry.TriangleMesh()
    records = []
    report = []

    for name, raw in directions.items():
        normal = raw / np.linalg.norm(raw)

        reference = np.array([1.0, 0.0, 0.0])
        if abs(reference @ normal) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, reference)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)

        # 1. normal gating
        aligned = np.abs(normals @ normal) >= cos_limit
        subset = points[aligned]

        if len(subset) < args.min_inliers:
            report.append(f"{name}: only {len(subset)} aligned points, skipped")
            continue

        distance = subset @ normal

        # 2. peak detection
        planes = find_plane_offsets(distance, args.thickness,
                                    args.min_inliers, args.peak_ratio,
                                    merge_gap)

        kept = dropped_small = dropped_buried = 0

        for offset, _ in planes:
            inliers = subset[np.abs(distance - offset) <= args.thickness]
            projected = inliers + (offset - inliers @ normal)[:, None] * normal

            a = projected @ u
            b = projected @ v

            # 3. occupancy instead of a bounding rectangle
            max_hole_cells = args.max_hole / (args.cell * args.cell)
            mask, counts, a0, b0 = build_mask(a, b, args.cell, args.min_hits,
                                              args.close, max_hole_cells)
            labels, total = ndimage.label(mask, structure=np.ones((3, 3)))

            for region in range(1, total + 1):
                piece = labels == region
                cells = int(piece.sum())
                area = cells * args.cell * args.cell

                ri, ci = np.nonzero(piece)
                side_a = (ri.max() - ri.min() + 1) * args.cell
                side_b = (ci.max() - ci.min() + 1) * args.cell

                if (area < args.min_area
                        or min(side_a, side_b) < args.min_side):
                    dropped_small += 1
                    continue

                centers = (normal * offset
                           + np.outer((ri + 0.5) * args.cell + a0, u)
                           + np.outer((ci + 0.5) * args.cell + b0, v))

                # 4. is anything blocking the view from outside?
                if not args.no_occlusion_test:
                    if len(centers) > args.samples:
                        pick = np.random.default_rng(0).choice(
                            len(centers), args.samples, replace=False)
                        probe = centers[pick]
                    else:
                        probe = centers

                    front = blocked_fraction(occupancy, probe, normal,
                                             skip, reach)
                    back = blocked_fraction(occupancy, probe, -normal,
                                            skip, reach)

                    if min(front, back) > args.occlusion:
                        dropped_buried += 1
                        continue
                else:
                    front = back = 0.0

                # mesh for this face
                vertices = []
                triangles = []

                for (i0, j0, i1, j1) in greedy_rectangles(piece):
                    amin = a0 + i0 * args.cell
                    amax = a0 + (i1 + 1) * args.cell
                    bmin = b0 + j0 * args.cell
                    bmax = b0 + (j1 + 1) * args.cell

                    origin = normal * offset
                    quad = [
                        origin + u * amin + v * bmin,
                        origin + u * amax + v * bmin,
                        origin + u * amax + v * bmax,
                        origin + u * amin + v * bmax,
                    ]
                    k = len(vertices)
                    vertices.extend(quad)
                    triangles.append([k, k + 1, k + 2])
                    triangles.append([k, k + 2, k + 3])

                face = o3d.geometry.TriangleMesh()
                face.vertices = o3d.utility.Vector3dVector(np.array(vertices))
                face.triangles = o3d.utility.Vector3iVector(
                    np.array(triangles, dtype=np.int32))
                face.paint_uniform_color(COLORS.get(name, [0.8, 0.8, 0.8]))
                mesh += face

                amin = a0 + ri.min() * args.cell
                amax = a0 + (ri.max() + 1) * args.cell
                bmin = b0 + ci.min() * args.cell
                bmax = b0 + (ci.max() + 1) * args.cell
                origin = normal * offset
                corners = [
                    origin + u * amin + v * bmin,
                    origin + u * amax + v * bmin,
                    origin + u * amax + v * bmax,
                    origin + u * amin + v * bmax,
                ]

                records.append({
                    "direction": name,
                    "normal": normal.tolist(),
                    "u": u.tolist(),
                    "v": v.tolist(),
                    "offset": float(offset),
                    "area": float(area),
                    "cells": cells,
                    "points": int(counts[piece].sum()),
                    "blocked_front": float(front),
                    "blocked_back": float(back),
                    "bbox_corners": [c.tolist() for c in corners],
                })
                kept += 1

        report.append(
            f"{name}: {len(planes)} planes -> {kept} faces "
            f"(dropped {dropped_small} small, {dropped_buried} buried)"
        )

    print()
    for line in report:
        print(line)

    if len(mesh.triangles) == 0:
        print("\nno faces survived - loosen --min-area, --min-hits "
              "or --occlusion")
        return

    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(f"{args.out}.ply", mesh)

    with open(f"{args.out}.json", "w") as handle:
        json.dump(records, handle, indent=2)

    print(f"\nsaved {args.out}.ply and {args.out}.json")
    print(f"faces: {len(records)}   triangles: {len(mesh.triangles)}")


if __name__ == "__main__":
    main()
