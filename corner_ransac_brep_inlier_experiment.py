import open3d as o3d
import numpy as np

INPUT = "roof_corner_candidates.ply"

# Building principal directions derived earlier.
V = np.array([
    -0.20815405,
     0.97774725,
     0.02611914,
])

H1 = np.array([
     0.02266596,
    -0.02187495,
     0.99950375,
])

H2 = np.array([
     0.97783339,
     0.20864277,
    -0.01760823,
])

pcd = o3d.io.read_point_cloud(INPUT)
points = np.asarray(pcd.points)

print(f"Loaded corner candidates: {len(points):,}")
print(f"V  = {V}")
print(f"H1 = {H1}")
print(f"H2 = {H2}")

# Repeated RANSAC on corner/edge candidates.
remaining = pcd
planes = []

o3d.utility.random.seed(42)

for plane_id in range(20):
    if len(remaining.points) < 30:
        break

    model, inliers = remaining.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=2000,
    )

    if len(inliers) < 30:
        break

    plane_cloud = remaining.select_by_index(inliers)

    planes.append({
        "id": plane_id,
        "model": np.asarray(model),
        "cloud": plane_cloud,
        "n_points": len(inliers),
    })

    remaining = remaining.select_by_index(
        inliers,
        invert=True,
    )

print("\nRANSAC PLANES FROM CORNER CANDIDATES")

for plane in planes:
    n = plane["model"][:3]
    n = n / np.linalg.norm(n)

    angles = {}

    for name, direction in [
        ("V", V),
        ("H1", H1),
        ("H2", H2),
    ]:
        angle = np.degrees(
            np.arccos(
                np.clip(
                    abs(np.dot(n, direction)),
                    0.0,
                    1.0,
                )
            )
        )
        angles[name] = angle

    family = min(angles, key=angles.get)

    print(
        f"Plane {plane['id']:2d}: "
        f"{plane['n_points']:4d} points | "
        f"closest={family:2s} | "
        f"deviation={angles[family]:5.2f} deg"
    )

print(f"\nDetected planes: {len(planes)}")
print(f"Remaining points: {len(remaining.points)}")

print("\nPRINCIPAL PLANES (<= 6 degrees)")

for plane in planes:
    n = plane["model"][:3]
    n = n / np.linalg.norm(n)

    candidates = {
        "V": V,
        "H1": H1,
        "H2": H2,
    }

    angles = {
        name: np.degrees(
            np.arccos(
                np.clip(abs(np.dot(n, direction)), 0.0, 1.0)
            )
        )
        for name, direction in candidates.items()
    }

    family = min(angles, key=angles.get)

    if angles[family] <= 6.0:
        direction = candidates[family]
        P_in = np.asarray(plane["cloud"].points)

        offset = np.median(P_in @ direction)

        print(
            f"Plane {plane['id']:2d}: "
            f"{family:2s} | "
            f"offset={offset: .4f} | "
            f"{plane['n_points']:4d} points | "
            f"{angles[family]:4.2f} deg"
        )

# Export only inlier points belonging to accepted principal planes.
principal_cloud = o3d.geometry.PointCloud()
accepted_count = 0

for plane in planes:
    n = plane["model"][:3]
    n /= np.linalg.norm(n)

    directions = {
        "V": V,
        "H1": H1,
        "H2": H2,
    }

    angles = {
        name: np.degrees(
            np.arccos(np.clip(abs(np.dot(n, d)), 0.0, 1.0))
        )
        for name, d in directions.items()
    }

    family = min(angles, key=angles.get)

    if angles[family] <= 6.0:
        principal_cloud += plane["cloud"]
        accepted_count += 1

o3d.io.write_point_cloud(
    "roof_principal_plane_candidates.ply",
    principal_cloud,
)

print(f"\nAccepted principal planes: {accepted_count}")
print(f"Exported points: {len(principal_cloud.points):,}")
print("Saved: roof_principal_plane_candidates.ply")
