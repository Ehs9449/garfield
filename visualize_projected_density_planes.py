
def projected_density_maps(
    dense_points,
    normal,
    u,
    v,
    offset,
    amin,
    amax,
    bmin,
    bmax,
    cell_size=0.04,
    side_gap=0.015,
    side_depth=0.20,
):
    """Build aligned 2D density maps on opposite sides of a plane."""

    na = max(1, int(np.ceil((amax - amin) / cell_size)))
    nb = max(1, int(np.ceil((bmax - bmin) / cell_size)))

    a_edges = np.linspace(amin, amax, na + 1)
    b_edges = np.linspace(bmin, bmax, nb + 1)

    # Project ALL dense points into this plane's coordinate system.
    a = dense_points @ u
    b = dense_points @ v
    signed_distance = dense_points @ normal - offset

    footprint = (
        (a >= amin) & (a <= amax) &
        (b >= bmin) & (b <= bmax)
    )

    positive = footprint & (
        (signed_distance > side_gap) &
        (signed_distance <= side_depth)
    )

    negative = footprint & (
        (signed_distance < -side_gap) &
        (signed_distance >= -side_depth)
    )

    positive_map = np.histogram2d(
        a[positive], b[positive],
        bins=[a_edges, b_edges],
    )[0]

    negative_map = np.histogram2d(
        a[negative], b[negative],
        bins=[a_edges, b_edges],
    )[0]

    return positive_map, negative_map, a_edges, b_edges


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


dense_points = np.asarray(
    o3d.io.read_point_cloud(
        "data/towerlsu_idf_cleaned/cleaned_final_SAM3/roof - Cloud.ply"
    ).points
)

combined = o3d.geometry.TriangleMesh()

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

        positive_map, negative_map, a_edges, b_edges = (
            projected_density_maps(
                dense_points, normal, u, v,
                offset, amin, amax, bmin, bmax
            )
        )

        print(
            f"{name} offset={offset:+.4f} "
            f"positive={int(positive_map.sum())} "
            f"negative={int(negative_map.sum())} "
            f"kept_cells={int(((np.maximum(positive_map, negative_map) >= 10) & (np.minimum(positive_map, negative_map) <= 0.10 * np.maximum(positive_map, negative_map))).sum())}"
        )
        total = positive_map + negative_map
        dominant = np.maximum(positive_map, negative_map)
        opposite = np.minimum(positive_map, negative_map)

        # First reject planes with substantial density on both sides.
        plane_positive = positive_map.sum()
        plane_negative = negative_map.sum()

        plane_dominant = max(plane_positive, plane_negative)
        plane_opposite = min(plane_positive, plane_negative)

        if plane_dominant < 10 or plane_opposite > 0.10 * plane_dominant:
            print(f"  REJECTED entire plane: {name} offset={offset:+.4f}")
            continue

        # Use the SAME dominant side throughout the surviving plane.
        if plane_positive > plane_negative:
            keep = (
                (positive_map >= 10) &
                (negative_map <= 0.10 * positive_map)
            )
        else:
            keep = (
                (negative_map >= 10) &
                (positive_map <= 0.10 * negative_map)
            )

        for ia, ib in np.argwhere(keep):
            a0, a1 = a_edges[ia:ia + 2]
            b0, b1 = b_edges[ib:ib + 2]

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

    print(f"{name}: {len(fitted)} fitted planes")

combined.compute_triangle_normals()

output = "roof_projected_density_planes.ply"

o3d.io.write_triangle_mesh(output, combined)

print(f"Saved: {output}")
print(f"Total faces: {len(combined.triangles) // 2}")
