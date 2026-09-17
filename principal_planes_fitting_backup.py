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
