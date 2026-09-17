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



def crop_connected_regions(a, b, cell_size=0.04, min_points=5):
    """Group projected supporting points by connected occupied grid cells."""
    from scipy.ndimage import label

    amin, bmin = a.min(), b.min()
    ia = np.floor((a - amin) / cell_size).astype(int)
    ib = np.floor((b - bmin) / cell_size).astype(int)

    occupied = np.zeros((ia.max() + 1, ib.max() + 1), dtype=bool)
    occupied[ia, ib] = True

    # Eight-neighbor connectivity.
    structure = np.ones((3, 3), dtype=int)
    labels, count = label(occupied, structure=structure)

    regions = []
    for region_id in range(1, count + 1):
        member = labels[ia, ib] == region_id
        if np.count_nonzero(member) >= min_points:
            regions.append(member)

    return regions


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

        regions = crop_connected_regions(a, b)

        for member in regions:
            region_a = a[member]
            region_b = b[member]

            amin, amax = region_a.min(), region_a.max()
            bmin, bmax = region_b.min(), region_b.max()

            corners = np.array([
                normal * offset + u * amin + v * bmin,
                normal * offset + u * amax + v * bmin,
                normal * offset + u * amax + v * bmax,
                normal * offset + u * amin + v * bmax,
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

output = "roof_cropped_principal_planes.ply"

o3d.io.write_triangle_mesh(output, combined)

print(f"Saved: {output}")
print(f"Total faces: {len(combined.triangles) // 2}")
