#!/usr/bin/env python3
"""
Geometry sanity check for an IDF written by cloud_to_idf.py.

Reads the file back and verifies, without EnergyPlus:
  - every surface polygon is planar
  - the zone surfaces form a closed volume (divergence theorem)
  - every window is coplanar with its base wall
  - every window lies fully inside its base wall
  - windows on the same wall do not overlap
  - the total glazing on a wall is smaller than the wall

Usage:  python check_idf_geometry.py building.idf
"""

import re
import sys

import numpy as np


def read_objects(text):
    text = re.sub(r"!.*", "", text)
    for chunk in text.split(";"):
        # keep blank fields: IDF is positional, dropping them shifts indices
        fields = [f.strip() for f in chunk.split(",")]
        while fields and fields[0] == "":
            fields.pop(0)
        if fields and fields[0]:
            yield fields[0].lower(), fields


def vertices_from(fields, start):
    numbers = [float(f) for f in fields[start:] if f != ""]
    return [np.array(numbers[i:i + 3]) for i in range(0, len(numbers) - 2, 3)]


def normal_of(points):
    total = np.zeros(3)
    for i in range(len(points)):
        total += np.cross(points[i], points[(i + 1) % len(points)])
    length = np.linalg.norm(total)
    return total / length, 0.5 * length


def planar(points, tolerance=1e-4):
    unit, _ = normal_of(points)
    offsets = [abs((p - points[0]) @ unit) for p in points]
    return max(offsets) <= tolerance


def main(path):
    text = open(path).read()
    surfaces, windows = {}, []

    for kind, fields in read_objects(text):
        if kind == "buildingsurface:detailed":
            points = vertices_from(fields, 12)
            surfaces[fields[1].lower()] = {"name": fields[1],
                                           "type": fields[2],
                                           "vertices": points}
        elif kind == "fenestrationsurface:detailed":
            windows.append({"name": fields[1], "base": fields[4].lower(),
                            "vertices": vertices_from(fields, 10)})

    problems = []
    print(f"surfaces {len(surfaces)}   windows {len(windows)}")

    # 1. planarity
    for surface in list(surfaces.values()) + windows:
        if not planar(surface["vertices"]):
            problems.append(f"not planar: {surface['name']}")

    # 2. closure
    imbalance = np.zeros(3)
    volume = area = 0.0
    for surface in surfaces.values():
        unit, face_area = normal_of(surface["vertices"])
        imbalance += unit * face_area
        area += face_area
        volume += (np.mean(surface["vertices"], axis=0) @ unit) * face_area / 3
    error = float(np.linalg.norm(imbalance))
    print(f"total area {area:.1f} m2   volume {volume:.1f} m3")
    print(f"closure error {error:.6e} m2")
    if error > 1e-6 * max(area, 1.0):
        problems.append(f"zone is not closed, residual {error:.4f} m2")
    if volume <= 0:
        problems.append("volume is not positive - normals point inward")

    # 3. windows against their base wall
    used = {}
    for window in windows:
        base = surfaces.get(window["base"])
        if base is None:
            problems.append(f"{window['name']}: base surface not found")
            continue

        unit, base_area = normal_of(base["vertices"])
        origin = base["vertices"][0]
        offsets = [abs((p - origin) @ unit) for p in window["vertices"]]
        if max(offsets) > 1e-3:
            problems.append(f"{window['name']}: not coplanar with "
                            f"{base['name']} ({max(offsets):.4f} m)")

        horizontal = np.cross([0, 0, 1], unit)
        if np.linalg.norm(horizontal) < 1e-9:
            horizontal = np.array([1.0, 0.0, 0.0])
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(unit, horizontal)

        def box(points):
            a = [float((p - origin) @ horizontal) for p in points]
            b = [float((p - origin) @ vertical) for p in points]
            return min(a), max(a), min(b), max(b)

        wa0, wa1, wb0, wb1 = box(base["vertices"])
        a0, a1, b0, b1 = box(window["vertices"])
        if a0 < wa0 - 1e-6 or a1 > wa1 + 1e-6 or \
           b0 < wb0 - 1e-6 or b1 > wb1 + 1e-6:
            problems.append(f"{window['name']}: sticks out of {base['name']}")

        window_normal, window_area = normal_of(window["vertices"])
        if window_normal @ unit < 0.99:
            problems.append(f"{window['name']}: normal disagrees with "
                            f"{base['name']}")

        others = used.setdefault(window["base"], [])
        for (name, oa0, oa1, ob0, ob1) in others:
            if a0 < oa1 - 1e-6 and oa0 < a1 - 1e-6 and \
               b0 < ob1 - 1e-6 and ob0 < b1 - 1e-6:
                problems.append(f"{window['name']} overlaps {name}")
        others.append((window["name"], a0, a1, b0, b1))

    # 4. glazing ratio
    glazed = {}
    for window in windows:
        _, window_area = normal_of(window["vertices"])
        glazed[window["base"]] = glazed.get(window["base"], 0.0) + window_area
    for key, total in glazed.items():
        _, base_area = normal_of(surfaces[key]["vertices"])
        if total >= base_area:
            problems.append(f"{surfaces[key]['name']}: glazing {total:.1f} m2 "
                            f">= wall {base_area:.1f} m2")

    print()
    if problems:
        print(f"FAILED  {len(problems)} problem(s):")
        for line in problems[:40]:
            print(f"  - {line}")
        sys.exit(1)

    print("PASSED  geometry is closed, windows sit inside their walls")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "building.idf")
