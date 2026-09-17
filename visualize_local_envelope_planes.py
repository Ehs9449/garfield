import numpy as np
import open3d as o3d

P = np.asarray(
    o3d.io.read_point_cloud("roof_corner_candidates.ply").points
)

V = np.array([-0.20815405, 0.97774725, 0.02611914])
H1 = np.array([0.02266596, -0.02187495, 0.99950375])
H2 = np.array([0.97783339, 0.20864277, -0.01760823])

directions = {
    "V": V,
    "H1": H1,
    "H2": H2,
}

def fit_planes(normal, threshold=0.01, min_inliers=30):
    s = P @ normal
    remaining = np.arange(len(P))
    result = []

    while len(remaining) >= min_inliers:
        values = s[remaining]

        counts = np.array([
            np.count_nonzero(np.abs(values - v) <= threshold)
            for v in values
        ])

        best = values[np.argmax(counts)]
        mask = np.abs(values - best) <= threshold

        if np.count_nonzero(mask) < min_inliers:
            break

        indices = remaining[mask]
        offset = np.median(s[indices])

        result.append((offset, indices))
        remaining = remaining[~mask]

    return result


# Dense cloud used for local two-sided occupancy tests.
dense = o3d.io.read_point_cloud(
    "data/towerlsu_idf_cleaned/cleaned_final_SAM3/roof - Cloud.ply"
)
dense_points = np.asarray(dense.points)

CELL_SIZE = 0.04
DENSITY_RADIUS = 0.10
SIDE_GAP = 0.015
SIDE_DEPTH = 0.20
MIN_SUPPORT = 3
MIN_SIDE_POINTS = 5

combined = o3d.geometry.TriangleMesh()
kept_cells = 0
rejected_cells = 0

colors = {
    "V": [1.0, 0.25, 0.25],
    "H1": [0.25, 1.0, 0.25],
    "H2": [0.25, 0.45, 1.0],
}

for name, normal in directions.items():

    normal = normal / np.linalg.norm(normal)

    # Two orthogonal directions spanning the plane.
    reference = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    fitted = fit_planes(normal)

    for offset, indices in fitted:

        points = P[indices]

        # Project the fitted inliers onto their exact principal plane.
        projected = points + (
            offset - points @ normal
        )[:, None] * normal

        a = projected @ u
        b = projected @ v

        # Finite extent from this plane's inlier points.
        amin, amax = a.min(), a.max()
        bmin, bmax = b.min(), b.max()

        # Divide the fitted plane into local cells.
        na = max(1, int(np.ceil((amax - amin) / CELL_SIZE)))
        nb = max(1, int(np.ceil((bmax - bmin) / CELL_SIZE)))

        # Dense points expressed in the plane coordinate system.
        dense_a = dense_points @ u
        dense_b = dense_points @ v
        dense_s = dense_points @ normal

        for ia in range(na):
            a0 = amin + ia * (amax - amin) / na
            a1 = amin + (ia + 1) * (amax - amin) / na

            for ib in range(nb):
                b0 = bmin + ib * (bmax - bmin) / nb
                b1 = bmin + (ib + 1) * (bmax - bmin) / nb

                # Examine a local prism around this surface cell.
                # Larger neighborhood for independent density analysis.
                ac = (a0 + a1) / 2
                bc = (b0 + b1) / 2

                local = (
                    (np.abs(dense_a - ac) <= DENSITY_RADIUS) &
                    (np.abs(dense_b - bc) <= DENSITY_RADIUS) &
                    (np.abs(dense_s - offset) <= SIDE_DEPTH)
                )

                distances = dense_s[local] - offset

                # Evidence close to the fitted surface.
                support = np.count_nonzero(
                    np.abs(distances) <= SIDE_GAP
                )

                if support < MIN_SUPPORT:
                    rejected_cells += 1
                    continue

                positive = np.count_nonzero(
                    (distances > SIDE_GAP) &
                    (distances <= SIDE_DEPTH)
                )

                negative = np.count_nonzero(
                    (distances < -SIDE_GAP) &
                    (distances >= -SIDE_DEPTH)
                )

                # Compare point density on both sides.
                total = positive + negative

                if total < MIN_SIDE_POINTS:
                    rejected_cells += 1
                    continue

                imbalance = abs(positive - negative) / total

                # Require at least 90% of side points on one side.
                if imbalance < 0.80:
                    rejected_cells += 1
                    continue

                corners = np.array([
                    normal * offset + u * a0 + v * b0,
                    normal * offset + u * a1 + v * b0,
                    normal * offset + u * a1 + v * b1,
                    normal * offset + u * a0 + v * b1,
                ])

                face = o3d.geometry.TriangleMesh()
                face.vertices = o3d.utility.Vector3dVector(corners)
                face.triangles = o3d.utility.Vector3iVector([
                    [0, 1, 2],
                    [0, 2, 3],
                ])
                face.paint_uniform_color(colors[name])

                combined += face
                kept_cells += 1

    print(f"{name}: {len(fitted)} fitted planes")

combined.compute_triangle_normals()

output = "roof_local_envelope_planes.ply"

o3d.io.write_triangle_mesh(output, combined)

print(f"Saved: {output}")
print(f"Total faces: {len(combined.triangles) // 2}")

print(f"Kept surface cells: {kept_cells}")
print(f"Rejected surface cells: {rejected_cells}")
