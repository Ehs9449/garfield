#!/usr/bin/env python3
"""
Semantic point clouds  ->  closed EnergyPlus IDF.

Input
-----
Three PLY files from the SAM segmentation:

    roof - Cloud.ply      the building envelope (walls + roofs)
    window - Cloud.ply    glazing
    column - Cloud.ply    exterior columns

Method
------
1.  Rotate the cloud so the building's own horizontal axes become X and Y.
    The real orientation is written back as the IDF North Axis.
2.  Scale from normalized units to metres using a height you supply.
3.  Build a height map of the envelope on a coarse grid, quantize it into a
    few flat levels, and take the boundary of the resulting stack of prisms.
    That boundary is closed by construction: a floor, one roof patch per
    level, and a vertical wall wherever the height steps down.
4.  Cluster the window points, snap each cluster onto its host wall, and
    write it as FenestrationSurface:Detailed inside that wall.
5.  Cluster the column points and write each one as Shading:Building:Detailed.
6.  Verify closure with the divergence theorem before writing the file.

Usage
-----
    python cloud_to_idf.py \
        --envelope "cleaned_final_SAM3/roof - Cloud.ply" \
        --windows  "cleaned_final_SAM3/window - Cloud.ply" \
        --columns  "cleaned_final_SAM3/column - Cloud.ply" \
        --height 53.0 --out lsu_tower.idf
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.signal import find_peaks


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------

def load_points(path):
    if path is None or not os.path.exists(path):
        return np.zeros((0, 3))
    return np.asarray(o3d.io.read_point_cloud(path).points)


def read_directions(path):
    if path is None:
        return None
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.array(rows, float) if rows else None


def building_frame(directions, points):
    """Rotation whose Z is up and whose X follows the building's long axis."""
    if directions is None or len(directions) < 3:
        flat = points[:, :2] - points[:, :2].mean(axis=0)
        _, vectors = np.linalg.eigh(flat.T @ flat)
        x_axis = np.array([vectors[0, -1], vectors[1, -1], 0.0])
    else:
        unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        up = int(np.argmax(np.abs(unit[:, 2])))
        x_axis = np.array(
            [unit[i] for i in range(len(unit)) if i != up][0], float)
        x_axis[2] = 0.0

    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0])
    if x_axis[0] < 0 or (abs(x_axis[0]) < 1e-9 and x_axis[1] < 0):
        x_axis = -x_axis                     # keep the frame canonical
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.vstack([x_axis, y_axis, z_axis])
    north = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
    return rotation, north


def polygon_normal(vertices):
    normal = np.zeros(3)
    for i in range(len(vertices)):
        normal += np.cross(vertices[i], vertices[(i + 1) % len(vertices)])
    length = np.linalg.norm(normal)
    return normal / length if length else normal


def polygon_area(vertices):
    total = np.zeros(3)
    for i in range(len(vertices)):
        total += np.cross(vertices[i], vertices[(i + 1) % len(vertices)])
    return 0.5 * np.linalg.norm(total)


def orient(vertices, outward):
    """EnergyPlus wants vertices counter-clockwise seen from outside,
    starting at the upper-left corner."""
    v = list(vertices)
    if polygon_normal(v) @ outward < 0:
        v = v[::-1]

    left = np.cross(np.array([0.0, 0.0, 1.0]), outward)
    if np.linalg.norm(left) < 1e-9:
        left = np.array([1.0, 0.0, 0.0])
    left /= np.linalg.norm(left)

    keys = np.array([[-round(p[2], 4), round(float(p @ left), 4)] for p in v])
    start = int(np.lexsort((keys[:, 1], keys[:, 0]))[0])
    return v[start:] + v[:start]


def greedy_rectangles(mask):
    """Cover a boolean mask with few axis-aligned rectangles."""
    work = mask.copy()
    rows, cols = work.shape
    out = []
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
            out.append((i, j, i2, j2))
            work[i:i2 + 1, j:j2 + 1] = False
            j = j2 + 1
    return out


def merge_runs(entries):
    """entries: dict key -> list of consecutive integer indices."""
    for key, values in entries.items():
        values = sorted(set(values))
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
            else:
                yield key, (start, previous)
                start = previous = value
        yield key, (start, previous)


# --------------------------------------------------------------------------
# massing
# --------------------------------------------------------------------------

def height_map(points, cell, min_hits, percentile):
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    ix = np.floor((points[:, 0] - x0) / cell).astype(int)
    iy = np.floor((points[:, 1] - y0) / cell).astype(int)
    shape = (int(ix.max()) + 1, int(iy.max()) + 1)

    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, (ix, iy), 1)

    heights = np.zeros(shape)
    flat = ix * shape[1] + iy
    order = np.lexsort((points[:, 2], flat))
    flat, z = flat[order], points[order, 2]
    breaks = np.r_[0, np.flatnonzero(np.diff(flat)) + 1, len(flat)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        cell_id = flat[start]
        heights[cell_id // shape[1], cell_id % shape[1]] = \
            np.percentile(z[start:stop], percentile)

    mask = counts >= min_hits
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), bool))
    mask = ndimage.binary_fill_holes(mask)
    heights = np.where(mask, heights, 0.0)
    return mask, heights, x0, y0


def quantize(mask, heights, count, min_region):
    """Snap the height map to a few flat levels."""
    values = heights[mask]
    if values.size == 0:
        raise SystemExit("envelope cloud produced an empty footprint")

    hist, edges = np.histogram(values, bins=60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks, _ = find_peaks(np.r_[0, hist, 0], distance=2)
    peaks = [p - 1 for p in peaks if 0 <= p - 1 < len(centers)]
    peaks.sort(key=lambda p: -hist[p])

    levels = sorted(float(centers[p]) for p in peaks[:count])
    if not levels:
        levels = [float(np.median(values))]
    if values.max() - max(levels) > 0.10 * values.max():
        levels.append(float(values.max()))
    levels = sorted({round(v, 3) for v in levels})

    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.argmin(
        np.abs(values[:, None] - np.array(levels)[None, :]), axis=1)

    # drop speckle: give tiny islands the level of their neighbours
    for level in range(len(levels)):
        labels, total = ndimage.label(index == level,
                                      structure=np.ones((3, 3)))
        for region in range(1, total + 1):
            island = labels == region
            if island.sum() < min_region:
                index[island] = -2
    missing = index == -2
    if missing.any():
        source = np.where(index < 0, 0, index)
        grown = ndimage.grey_dilation(source, size=(5, 5))
        index[missing] = grown[missing]

    return np.array(levels), index


def envelope_surfaces(mask, index, levels, cell, x0, y0):
    """Floor, roof patches and walls of the stack of prisms."""
    surfaces = []

    def x_of(i):
        return x0 + i * cell

    def y_of(j):
        return y0 + j * cell

    for (i0, j0, i1, j1) in greedy_rectangles(mask):
        quad = [np.array([x_of(i0), y_of(j0), 0.0]),
                np.array([x_of(i1 + 1), y_of(j0), 0.0]),
                np.array([x_of(i1 + 1), y_of(j1 + 1), 0.0]),
                np.array([x_of(i0), y_of(j1 + 1), 0.0])]
        surfaces.append({"type": "Floor", "outward": np.array([0., 0., -1.]),
                         "vertices": orient(quad, np.array([0., 0., -1.]))})

    for level, height in enumerate(levels):
        piece = (index == level) & mask
        if not piece.any():
            continue
        for (i0, j0, i1, j1) in greedy_rectangles(piece):
            quad = [np.array([x_of(i0), y_of(j0), height]),
                    np.array([x_of(i1 + 1), y_of(j0), height]),
                    np.array([x_of(i1 + 1), y_of(j1 + 1), height]),
                    np.array([x_of(i0), y_of(j1 + 1), height])]
            surfaces.append({"type": "Roof",
                             "outward": np.array([0., 0., 1.]),
                             "vertices": orient(quad, np.array([0., 0., 1.]))})

    # height with a zero border, so the outside is just another neighbour
    nx, ny = mask.shape
    top = np.zeros((nx + 2, ny + 2))
    top[1:-1, 1:-1] = np.where(mask, levels[np.clip(index, 0, None)], 0.0)

    faces = {
        "+x": ((1, 0), np.array([1., 0., 0.])),
        "-x": ((-1, 0), np.array([-1., 0., 0.])),
        "+y": ((0, 1), np.array([0., 1., 0.])),
        "-y": ((0, -1), np.array([0., -1., 0.])),
    }

    for name, ((di, dj), outward) in faces.items():
        here = top[1:-1, 1:-1]
        there = top[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj]
        step = here > there + 1e-6

        entries = defaultdict(list)
        for i, j in zip(*np.nonzero(step)):
            low, high = round(there[i, j], 4), round(here[i, j], 4)
            if name in ("+x", "-x"):
                entries[(i, low, high)].append(j)      # run along y
            else:
                entries[(j, low, high)].append(i)      # run along x

        for (fixed, low, high), (first, last) in merge_runs(entries):
            if name == "+x":
                x = x_of(fixed + 1)
                a, b = y_of(first), y_of(last + 1)
                quad = [np.array([x, a, low]), np.array([x, b, low]),
                        np.array([x, b, high]), np.array([x, a, high])]
            elif name == "-x":
                x = x_of(fixed)
                a, b = y_of(first), y_of(last + 1)
                quad = [np.array([x, a, low]), np.array([x, b, low]),
                        np.array([x, b, high]), np.array([x, a, high])]
            elif name == "+y":
                y = y_of(fixed + 1)
                a, b = x_of(first), x_of(last + 1)
                quad = [np.array([a, y, low]), np.array([b, y, low]),
                        np.array([b, y, high]), np.array([a, y, high])]
            else:
                y = y_of(fixed)
                a, b = x_of(first), x_of(last + 1)
                quad = [np.array([a, y, low]), np.array([b, y, low]),
                        np.array([b, y, high]), np.array([a, y, high])]

            surfaces.append({"type": "Wall", "outward": outward,
                             "vertices": orient(quad, outward)})

    for number, surface in enumerate(surfaces, start=1):
        surface["name"] = f"{surface['type']}_{number:04d}"
        surface["area"] = polygon_area(surface["vertices"])
    return surfaces


def closure_report(surfaces):
    """Divergence theorem: a closed surface has sum(area * normal) == 0."""
    total_area = sum(s["area"] for s in surfaces)
    imbalance = np.zeros(3)
    volume = 0.0
    for s in surfaces:
        imbalance += s["area"] * s["outward"]
        centroid = np.mean(s["vertices"], axis=0)
        volume += (centroid @ s["outward"]) * s["area"] / 3.0
    return total_area, float(np.linalg.norm(imbalance)), volume


# --------------------------------------------------------------------------
# windows and columns
# --------------------------------------------------------------------------

def wall_frame(surface):
    outward = surface["outward"]
    horizontal = np.cross(np.array([0.0, 0.0, 1.0]), outward)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.array([0.0, 0.0, 1.0])
    origin = surface["vertices"][0]
    a = [float((p - origin) @ horizontal) for p in surface["vertices"]]
    b = [float((p - origin) @ vertical) for p in surface["vertices"]]
    return origin, horizontal, vertical, (min(a), max(a)), (min(b), max(b))


def cluster(points, eps, min_points):
    if len(points) < min_points:
        return []
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points,
                                           print_progress=False))
    return [points[labels == k] for k in range(labels.max() + 1)]


def place_windows(clusters, walls, margin, min_size, max_offset, cos_limit):
    placed = defaultdict(list)
    skipped = {"no_wall": 0, "too_small": 0, "overlap": 0}

    for group in clusters:
        centre = group.mean(axis=0)
        centred = group - centre
        _, _, basis = np.linalg.svd(centred, full_matrices=False)
        normal = basis[2]

        best, best_distance = None, max_offset
        for wall in walls:
            if abs(normal @ wall["outward"]) < cos_limit:
                continue
            origin, ha, va, arange, brange = wall["frame"]
            distance = abs((centre - origin) @ wall["outward"])
            if distance >= best_distance:
                continue
            a = (centre - origin) @ ha
            b = (centre - origin) @ va
            if not (arange[0] <= a <= arange[1] and
                    brange[0] <= b <= brange[1]):
                continue
            best, best_distance = wall, distance

        if best is None:
            skipped["no_wall"] += 1
            continue

        origin, ha, va, arange, brange = best["frame"]
        local = group - origin
        a = local @ ha
        b = local @ va
        amin, amax = np.percentile(a, [2, 98])
        bmin, bmax = np.percentile(b, [2, 98])

        amin = max(amin, arange[0] + margin)
        amax = min(amax, arange[1] - margin)
        bmin = max(bmin, brange[0] + margin)
        bmax = min(bmax, brange[1] - margin)

        if amax - amin < min_size or bmax - bmin < min_size:
            skipped["too_small"] += 1
            continue

        clash = False
        for (oa0, oa1, ob0, ob1) in placed[best["name"]]:
            if amin < oa1 and oa0 < amax and bmin < ob1 and ob0 < bmax:
                clash = True
                break
        if clash:
            skipped["overlap"] += 1
            continue

        placed[best["name"]].append((amin, amax, bmin, bmax))

    windows = []
    for wall in walls:
        origin, ha, va, _, _ = wall["frame"]
        for number, (amin, amax, bmin, bmax) in enumerate(
                placed[wall["name"]], start=1):
            quad = [origin + ha * amin + va * bmin,
                    origin + ha * amax + va * bmin,
                    origin + ha * amax + va * bmax,
                    origin + ha * amin + va * bmax]
            windows.append({
                "name": f"{wall['name']}_Win_{number:02d}",
                "wall": wall["name"],
                "vertices": orient(quad, wall["outward"]),
                "area": (amax - amin) * (bmax - bmin),
            })
    return windows, skipped


def column_shades(clusters, min_height):
    shades = []
    for number, group in enumerate(clusters, start=1):
        low = group.min(axis=0)
        high = group.max(axis=0)
        if high[2] - low[2] < min_height:
            continue
        corners = [(low[0], low[1]), (high[0], low[1]),
                   (high[0], high[1]), (low[0], high[1])]
        for side in range(4):
            (xa, ya) = corners[side]
            (xb, yb) = corners[(side + 1) % 4]
            quad = [np.array([xa, ya, low[2]]), np.array([xb, yb, low[2]]),
                    np.array([xb, yb, high[2]]), np.array([xa, ya, high[2]])]
            edge = np.array([xb - xa, yb - ya, 0.0])
            outward = np.cross(edge, np.array([0.0, 0.0, 1.0]))
            outward /= np.linalg.norm(outward)
            shades.append({
                "name": f"Column_{number:03d}_S{side + 1}",
                "vertices": orient(quad, outward),
            })
    return shades


# --------------------------------------------------------------------------
# IDF text
# --------------------------------------------------------------------------

def vertex_block(vertices, indent="    "):
    lines = []
    for number, point in enumerate(vertices, start=1):
        end = ";" if number == len(vertices) else ","
        lines.append(f"{indent}{point[0]:.4f}, {point[1]:.4f}, "
                     f"{point[2]:.4f}{end}  !- Vertex {number} X,Y,Z")
    return "\n".join(lines)


HEADER = """! Generated by cloud_to_idf.py from semantic point clouds.
! Geometry only - constructions, loads and schedules are placeholders.

Version, {version};

SimulationControl,
    No,                      !- Do Zone Sizing Calculation
    No,                      !- Do System Sizing Calculation
    No,                      !- Do Plant Sizing Calculation
    Yes,                     !- Run Simulation for Sizing Periods
    No,                      !- Run Simulation for Weather File Run Periods
    No,                      !- Do HVAC Sizing Simulation for Sizing Periods
    1;                       !- Maximum Number of HVAC Sizing Simulation Passes

Building,
    {name},
    {north:.2f},             !- North Axis {{deg}}
    City,                    !- Terrain
    0.04,                    !- Loads Convergence Tolerance Value
    0.4,                     !- Temperature Convergence Tolerance Value
    FullExterior,            !- Solar Distribution
    25,                      !- Maximum Number of Warmup Days
    6;                       !- Minimum Number of Warmup Days

Timestep, 6;

GlobalGeometryRules,
    UpperLeftCorner,         !- Starting Vertex Position
    Counterclockwise,        !- Vertex Entry Direction
    World;                   !- Coordinate System

Site:Location,
    Baton Rouge LA,
    30.53,                   !- Latitude {{deg}}
    -91.15,                  !- Longitude {{deg}}
    -6.0,                    !- Time Zone {{hr}}
    22.0;                    !- Elevation {{m}}

SizingPeriod:DesignDay,
    Baton Rouge Summer,      !- Name
    7, 21, SummerDesignDay,  !- Month, Day, Day Type
    34.4,                    !- Maximum Dry-Bulb Temperature {{C}}
    9.4,                     !- Daily Dry-Bulb Temperature Range {{deltaC}}
    ,                        !- Dry-Bulb Temperature Range Modifier Type
    ,                        !- Dry-Bulb Temperature Range Modifier Schedule
    Wetbulb,                 !- Humidity Condition Type
    25.6,                    !- Wetbulb at Maximum Dry-Bulb {{C}}
    , , , ,                  !- unused humidity fields
    101217.,                 !- Barometric Pressure {{Pa}}
    3.4,                     !- Wind Speed {{m/s}}
    200,                     !- Wind Direction {{deg}}
    No, No, No,              !- Rain, Snow, Daylight Saving
    ASHRAEClearSky,          !- Solar Model Indicator
    , , , ,                  !- unused solar fields
    1.00;                    !- Sky Clearness

SizingPeriod:DesignDay,
    Baton Rouge Winter,      !- Name
    1, 21, WinterDesignDay,  !- Month, Day, Day Type
    -2.2,                    !- Maximum Dry-Bulb Temperature {{C}}
    0.0,                     !- Daily Dry-Bulb Temperature Range {{deltaC}}
    ,                        !- Dry-Bulb Temperature Range Modifier Type
    ,                        !- Dry-Bulb Temperature Range Modifier Schedule
    Wetbulb,                 !- Humidity Condition Type
    -2.2,                    !- Wetbulb at Maximum Dry-Bulb {{C}}
    , , , ,                  !- unused humidity fields
    101217.,                 !- Barometric Pressure {{Pa}}
    4.9,                     !- Wind Speed {{m/s}}
    350,                     !- Wind Direction {{deg}}
    No, No, No,              !- Rain, Snow, Daylight Saving
    ASHRAEClearSky,          !- Solar Model Indicator
    , , , ,                  !- unused solar fields
    0.00;                    !- Sky Clearness

!  ---- placeholder constructions ----

Material, Brick 200mm, MediumRough, 0.2000, 0.890, 1920.0, 790.0;
Material, Insulation 80mm, MediumRough, 0.0800, 0.040, 40.0, 1200.0;
Material, Plaster 15mm, Smooth, 0.0150, 0.160, 600.0, 1000.0;
Material, Concrete 150mm, MediumRough, 0.1500, 1.730, 2243.0, 837.0;
Material, Roof Membrane, VeryRough, 0.0095, 0.160, 1121.0, 1460.0;

WindowMaterial:SimpleGlazingSystem,
    Simple Glazing, 2.500, 0.350, 0.700;

Construction, ExtWall,  Brick 200mm, Insulation 80mm, Plaster 15mm;
Construction, ExtRoof,  Roof Membrane, Insulation 80mm, Concrete 150mm;
Construction, ExtFloor, Concrete 150mm;
Construction, ExtWindow, Simple Glazing;

Zone,
    {zone},                  !- Name
    0.0,                     !- Direction of Relative North {{deg}}
    0.0, 0.0, 0.0,           !- X, Y, Z Origin {{m}}
    1,                       !- Type
    1,                       !- Multiplier
    autocalculate,           !- Ceiling Height {{m}}
    autocalculate,           !- Volume {{m3}}
    autocalculate,           !- Floor Area {{m2}}
    ,                        !- Zone Inside Convection Algorithm
    ,                        !- Zone Outside Convection Algorithm
    Yes;                     !- Part of Total Floor Area

Output:VariableDictionary, IDF;
Output:Surfaces:Drawing, DXF;
Output:Diagnostics, DisplayExtraWarnings;
"""

CONSTRUCTION = {"Wall": "ExtWall", "Roof": "ExtRoof", "Floor": "ExtFloor"}


def write_idf(path, surfaces, windows, shades, zone, north, version, name):
    blocks = [HEADER.format(version=version, north=north, zone=zone,
                            name=name)]

    for surface in surfaces:
        kind = surface["type"]
        if kind == "Floor":
            boundary, sun, wind = "Ground", "NoSun", "NoWind"
        else:
            boundary, sun, wind = "Outdoors", "SunExposed", "WindExposed"

        blocks.append(
            "BuildingSurface:Detailed,\n"
            f"    {surface['name']},\n"
            f"    {kind},\n"
            f"    {CONSTRUCTION[kind]},\n"
            f"    {zone},\n"
            "    ,                        !- Space Name\n"
            f"    {boundary},\n"
            "    ,                        !- Outside Boundary Condition Object\n"
            f"    {sun},\n"
            f"    {wind},\n"
            "    ,                        !- View Factor to Ground\n"
            f"    {len(surface['vertices'])},  !- Number of Vertices\n"
            + vertex_block(surface["vertices"]))

    for window in windows:
        blocks.append(
            "FenestrationSurface:Detailed,\n"
            f"    {window['name']},\n"
            "    Window,\n"
            "    ExtWindow,\n"
            f"    {window['wall']},\n"
            "    ,                        !- Outside Boundary Condition Object\n"
            "    ,                        !- View Factor to Ground\n"
            "    ,                        !- Frame and Divider Name\n"
            "    1,                       !- Multiplier\n"
            "    4,                       !- Number of Vertices\n"
            + vertex_block(window["vertices"]))

    for shade in shades:
        blocks.append(
            "Shading:Building:Detailed,\n"
            f"    {shade['name']},\n"
            "    ,                        !- Transmittance Schedule Name\n"
            "    4,                       !- Number of Vertices\n"
            + vertex_block(shade["vertices"]))

    with open(path, "w") as handle:
        handle.write("\n\n".join(blocks) + "\n")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", required=True)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--columns", default=None)
    parser.add_argument("--dirs", default=None)
    parser.add_argument("--out", default="building.idf")

    parser.add_argument("--height", type=float, required=True,
                        help="real height of the building in metres")
    parser.add_argument("--cell", type=float, default=1.5,
                        help="footprint grid size in metres")
    parser.add_argument("--levels", type=int, default=3,
                        help="how many flat height levels to fit")
    parser.add_argument("--min-hits", type=int, default=6,
                        help="points needed before a footprint cell counts")
    parser.add_argument("--top-percentile", type=float, default=97.0)
    parser.add_argument("--min-region", type=int, default=4,
                        help="smallest island of cells kept at one level")

    parser.add_argument("--window-eps", type=float, default=0.6)
    parser.add_argument("--window-min-points", type=int, default=60)
    parser.add_argument("--window-margin", type=float, default=0.15)
    parser.add_argument("--window-min-size", type=float, default=0.4)
    parser.add_argument("--window-max-offset", type=float, default=2.0)
    parser.add_argument("--window-angle", type=float, default=40.0)

    parser.add_argument("--column-eps", type=float, default=0.5)
    parser.add_argument("--column-min-points", type=int, default=120)
    parser.add_argument("--column-min-height", type=float, default=1.0)

    parser.add_argument("--zone", default="Zone_Building")
    parser.add_argument("--name", default="LSU Tower")
    parser.add_argument("--version", default="24.1")
    args = parser.parse_args()

    envelope = load_points(args.envelope)
    windows_raw = load_points(args.windows)
    columns_raw = load_points(args.columns)
    if len(envelope) == 0:
        raise SystemExit(f"no points in {args.envelope}")

    print(f"envelope {len(envelope)}  windows {len(windows_raw)}  "
          f"columns {len(columns_raw)}")

    rotation, north = building_frame(read_directions(args.dirs), envelope)
    envelope = envelope @ rotation.T
    windows_raw = windows_raw @ rotation.T if len(windows_raw) else windows_raw
    columns_raw = columns_raw @ rotation.T if len(columns_raw) else columns_raw

    # the facade points are part of the envelope surface, so they help
    span = envelope[:, 2].max() - envelope[:, 2].min()
    scale = args.height / span
    origin = np.array([envelope[:, 0].min(), envelope[:, 1].min(),
                       envelope[:, 2].min()])

    def place(points):
        return (points - origin) * scale if len(points) else points

    envelope = place(envelope)
    windows_raw = place(windows_raw)
    columns_raw = place(columns_raw)

    print(f"north axis {north:.2f} deg, scale {scale:.3f} "
          f"(height {args.height:.1f} m)")

    massing = envelope
    if len(windows_raw):
        massing = np.vstack([envelope, windows_raw])

    mask, heights, x0, y0 = height_map(massing, args.cell, args.min_hits,
                                       args.top_percentile)
    levels, index = quantize(mask, heights, args.levels, args.min_region)
    print(f"footprint {int(mask.sum())} cells, "
          f"levels {[round(float(v), 1) for v in levels]} m")

    surfaces = envelope_surfaces(mask, index, levels, args.cell, x0, y0)
    walls = [s for s in surfaces if s["type"] == "Wall"]
    for wall in walls:
        wall["frame"] = wall_frame(wall)

    area, imbalance, volume = closure_report(surfaces)
    print(f"surfaces {len(surfaces)} "
          f"({len(walls)} wall, "
          f"{sum(1 for s in surfaces if s['type'] == 'Roof')} roof, "
          f"{sum(1 for s in surfaces if s['type'] == 'Floor')} floor)")
    print(f"total area {area:.1f} m2, volume {volume:.1f} m3")
    print(f"closure error {imbalance:.6f} m2 "
          f"({'closed' if imbalance < 1e-6 * max(area, 1) else 'NOT CLOSED'})")

    window_groups = cluster(windows_raw, args.window_eps,
                            args.window_min_points)
    placed, skipped = place_windows(
        window_groups, walls, args.window_margin, args.window_min_size,
        args.window_max_offset, np.cos(np.deg2rad(args.window_angle)))
    print(f"windows {len(window_groups)} clusters -> {len(placed)} placed "
          f"(skipped {skipped['no_wall']} no host wall, "
          f"{skipped['too_small']} too small, "
          f"{skipped['overlap']} overlapping)")

    wall_area = {w["name"]: w["area"] for w in walls}
    glazed = defaultdict(float)
    for window in placed:
        glazed[window["wall"]] += window["area"]
    bad = [n for n, a in glazed.items() if a > 0.95 * wall_area[n]]
    if bad:
        print(f"  warning: {len(bad)} walls are over 95% glazed")

    column_groups = cluster(columns_raw, args.column_eps,
                            args.column_min_points)
    shades = column_shades(column_groups, args.column_min_height)
    print(f"columns {len(column_groups)} clusters -> {len(shades)} "
          f"shading surfaces")

    write_idf(args.out, surfaces, placed, shades, args.zone, north,
              args.version, args.name)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
