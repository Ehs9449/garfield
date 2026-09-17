import open3d as o3d
import numpy as np

INPUT = "roof_corner_candidates.ply"

V = np.array([-0.20815405,  0.97774725,  0.02611914])
H1 = np.array([ 0.02266596, -0.02187495,  0.99950375])
H2 = np.array([ 0.97783339,  0.20864277, -0.01760823])

pcd = o3d.io.read_point_cloud(INPUT)
P = np.asarray(pcd.points)

print(f"Loaded candidate points: {len(P):,}")

def fit_principal_planes(points, normal, threshold=0.01, min_inliers=30):
    """
    Fit planes with a fixed principal normal.

    Plane equation:
        normal . x = offset
    """
    s = points @ normal
    remaining = np.arange(len(points))
    planes = []

    while len(remaining) >= min_inliers:
        values = s[remaining]

        # Each observed point proposes a possible plane offset.
        # Choose the offset having the largest consensus.
        best_count = 0
        best_mask = None

        for offset_candidate in values:
            mask = np.abs(values - offset_candidate) <= threshold
            count = np.count_nonzero(mask)

            if count > best_count:
                best_count = count
                best_mask = mask

        if best_count < min_inliers:
            break

        inlier_indices = remaining[best_mask]

        # Robustly refine the plane offset from its consensus points.
        offset = np.median(s[inlier_indices])

        planes.append({
            "offset": offset,
            "indices": inlier_indices,
        })

        remaining = remaining[~best_mask]

    return planes


for name, normal in [("V", V), ("H1", H1), ("H2", H2)]:
    planes = fit_principal_planes(P, normal)

    print(f"\n{name} planes:")

    for i, plane in enumerate(planes):
        print(
            f"  {name}{i}: "
            f"offset={plane['offset']: .4f} | "
            f"inliers={len(plane['indices']):4d}"
        )

# Refit and store the three principal-plane families.
principal_families = {
    "V": fit_principal_planes(P, V),
    "H1": fit_principal_planes(P, H1),
    "H2": fit_principal_planes(P, H2),
}

# Matrix whose rows are the three principal plane normals.
A = np.vstack([V, H1, H2])

candidate_vertices = []

for iv, pv in enumerate(principal_families["V"]):
    for i1, p1 in enumerate(principal_families["H1"]):
        for i2, p2 in enumerate(principal_families["H2"]):

            b = np.array([
                pv["offset"],
                p1["offset"],
                p2["offset"],
            ])

            xyz = np.linalg.solve(A, b)

            candidate_vertices.append({
                "point": xyz,
                "planes": (iv, i1, i2),
            })

print("\n3-PLANE INTERSECTIONS")
print(f"V planes:  {len(principal_families['V'])}")
print(f"H1 planes: {len(principal_families['H1'])}")
print(f"H2 planes: {len(principal_families['H2'])}")
print(f"Candidate 3D vertices: {len(candidate_vertices):,}")

# Measure support of each 3-plane intersection using the
# original Poisson sharp-edge/corner candidate points.
candidate_xyz = np.array([
    item["point"] for item in candidate_vertices
])

corner_tree = o3d.geometry.KDTreeFlann(pcd)

nearest_distances = []

for xyz in candidate_xyz:
    _, _, dist2 = corner_tree.search_knn_vector_3d(xyz, 1)
    nearest_distances.append(np.sqrt(dist2[0]))

nearest_distances = np.asarray(nearest_distances)

print("\nINTERSECTION SUPPORT BY CORNER POINTS")

for threshold in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
    count = np.count_nonzero(nearest_distances <= threshold)

    print(
        f"distance <= {threshold:.2f}: "
        f"{count:,} / {len(candidate_vertices):,}"
    )

print(
    f"\nNearest-distance median: "
    f"{np.median(nearest_distances):.4f}"
)

# Export tightly supported intersection candidates for inspection.
mask = nearest_distances <= 0.01
supported_xyz = candidate_xyz[mask]

vertices = o3d.geometry.PointCloud()
vertices.points = o3d.utility.Vector3dVector(supported_xyz)

output = "roof_intersections_001.ply"
o3d.io.write_point_cloud(output, vertices)

print(f"\nExported {len(supported_xyz):,} candidate vertices")
print(f"Saved: {output}")
