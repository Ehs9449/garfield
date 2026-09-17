from pathlib import Path

import numpy as np
import open3d as o3d


# ------------------------------------------------------------
# Input data
# ------------------------------------------------------------

DATA_DIR = Path(
    "/home/eaghae1/garfield/data/towerlsu_idf_cleaned/cleaned_final_SAM3"
)

FILES = {
    "window": DATA_DIR / "window - Cloud.ply",
    "roof": DATA_DIR / "roof - Cloud.ply",
    "column": DATA_DIR / "column - Cloud.ply",
}

# Based on the PCA analysis of the column point cloud:
# X = horizontal
# Y = vertical
# Z = horizontal
VERTICAL_AXIS = 1


def load_cloud(path):
    """Load a point cloud and verify that it contains points."""
    cloud = o3d.io.read_point_cloud(str(path))

    if len(cloud.points) == 0:
        raise RuntimeError(f"No points loaded from: {path}")

    return cloud

def extract_ransac_planes(
    cloud,
    distance_threshold=0.005,
    ransac_n=3,
    num_iterations=2000,
    min_inliers=1000,
    max_planes=20,
):
    """
    Repeatedly extract dominant planes from a point cloud using RANSAC.

    Returns a list containing:
        plane equation [a, b, c, d]
        points belonging to the plane
        number of inlier points
    """

    remaining = cloud
    planes = []

    for plane_id in range(max_planes):

        if len(remaining.points) < min_inliers:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        if len(inliers) < min_inliers:
            break

        plane_cloud = remaining.select_by_index(inliers)

        planes.append({
            "id": plane_id,
            "model": np.asarray(plane_model),
            "cloud": plane_cloud,
            "n_points": len(inliers),
        })

        # Remove this plane before searching for the next one
        remaining = remaining.select_by_index(inliers, invert=True)

    return planes, remaining


def estimate_building_frame(column_cloud, roof_cloud):
    """
    Estimate an orthonormal building coordinate frame.

    V  : dominant tower/column direction
    H1 : dominant roof/building direction perpendicular to V
    H2 : direction perpendicular to both V and H1
    """
    column_points = np.asarray(column_cloud.points)

    centered_columns = (
        column_points - column_points.mean(axis=0)
    )

    column_cov = np.cov(centered_columns, rowvar=False)
    values, vectors = np.linalg.eigh(column_cov)

    V = vectors[:, np.argmax(values)]
    V /= np.linalg.norm(V)

    # Keep vertical direction pointing approximately toward +Y.
    if V[1] < 0:
        V *= -1.0

    roof_points = np.asarray(roof_cloud.points)
    centered_roof = roof_points - roof_points.mean(axis=0)

    # Project roof geometry onto plane perpendicular to V.
    horizontal_points = (
        centered_roof
        - np.outer(centered_roof @ V, V)
    )

    horizontal_cov = np.cov(
        horizontal_points,
        rowvar=False,
    )

    values, vectors = np.linalg.eigh(horizontal_cov)
    order = np.argsort(values)[::-1]

    H1 = vectors[:, order[0]]

    # Numerically enforce perpendicularity to V.
    H1 = H1 - np.dot(H1, V) * V
    H1 /= np.linalg.norm(H1)

    H2 = np.cross(V, H1)
    H2 /= np.linalg.norm(H2)

    # Recompute H1 to guarantee an orthonormal frame.
    H1 = np.cross(H2, V)
    H1 /= np.linalg.norm(H1)

    return V, H1, H2


def classify_planes_by_principal_normal(planes, V, H1, H2):
    """
    Classify each RANSAC plane by the closest building principal normal.
    This does not modify the fitted plane.
    """
    references = {
        "V": np.asarray(V, dtype=float),
        "H1": np.asarray(H1, dtype=float),
        "H2": np.asarray(H2, dtype=float),
    }

    classified = []

    for plane in planes:
        normal = np.asarray(plane["model"][:3], dtype=float)
        normal /= np.linalg.norm(normal)

        angles = {}

        for name, reference in references.items():
            reference = reference / np.linalg.norm(reference)

            angle = np.degrees(
                np.arccos(
                    np.clip(
                        abs(np.dot(normal, reference)),
                        0.0,
                        1.0,
                    )
                )
            )

            angles[name] = angle

        family = min(angles, key=angles.get)

        classified.append(
            {
                "plane_id": plane["id"],
                "family": family,
                "angle": angles[family],
                "angles": angles,
            }
        )

    return classified


def regularize_planes_to_principal_normals(
    planes,
    V,
    H1,
    H2,
    max_angle_deg=6.0,
):
    """
    Snap RANSAC plane normals to the building principal frame only
    when the angular deviation is small.

    Each plane keeps its own spatial location. After snapping the
    normal, d is refitted from that plane's original RANSAC inliers.
    """
    references = {
        "V": np.asarray(V, dtype=float),
        "H1": np.asarray(H1, dtype=float),
        "H2": np.asarray(H2, dtype=float),
    }

    regularized = []

    for plane in planes:
        old_model = np.asarray(plane["model"], dtype=float)
        old_normal = old_model[:3]
        old_normal /= np.linalg.norm(old_normal)

        best_name = None
        best_normal = None
        best_angle = float("inf")

        for name, reference in references.items():
            candidate = reference / np.linalg.norm(reference)

            if np.dot(old_normal, candidate) < 0:
                candidate = -candidate

            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(old_normal, candidate),
                        -1.0,
                        1.0,
                    )
                )
            )

            if angle < best_angle:
                best_name = name
                best_normal = candidate
                best_angle = angle

        if best_angle <= max_angle_deg:
            points = np.asarray(plane["cloud"].points)

            # Refit only the offset using the original RANSAC inliers.
            d = -np.median(points @ best_normal)

            model = np.array([
                best_normal[0],
                best_normal[1],
                best_normal[2],
                d,
            ])

            snapped = True
        else:
            model = old_model.copy()
            snapped = False

        regularized.append({
            **plane,
            "model": model,
            "original_model": old_model.copy(),
            "principal_family": best_name,
            "principal_angle": best_angle,
            "snapped": snapped,
        })

    return regularized


def group_duplicate_regularized_planes(
    planes,
    offset_threshold=0.01,
):
    """
    Group snapped RANSAC planes that likely represent the same
    physical plane.

    Only snapped planes belonging to the same principal-normal
    family can be grouped. Preserved non-principal planes remain
    independent.
    """
    groups = []
    used = set()

    for i, plane in enumerate(planes):
        if i in used:
            continue

        group = [i]
        used.add(i)

        if plane["snapped"]:
            family = plane["principal_family"]
            d0 = plane["model"][3]

            for j in range(i + 1, len(planes)):
                if j in used:
                    continue

                other = planes[j]

                if not other["snapped"]:
                    continue

                if other["principal_family"] != family:
                    continue

                d1 = other["model"][3]

                if abs(d1 - d0) <= offset_threshold:
                    group.append(j)
                    used.add(j)

        groups.append(group)

    return groups


def finite_plane_boundary_from_inliers(
    plane,
    margin=0.01,
):
    """
    Create a finite rectangular boundary for a plane using its
    original RANSAC inlier points.

    Steps:
      1. Project inliers onto the plane.
      2. Construct a local 2D (u, v) coordinate system.
      3. Find the 2D extent of the projected points.
      4. Expand that extent by a small margin.
      5. Convert the four corners back to 3D.
    """
    model = np.asarray(plane["model"], dtype=float)

    normal = model[:3]
    normal /= np.linalg.norm(normal)
    d = model[3]

    points = np.asarray(plane["cloud"].points)

    # Project every RANSAC inlier exactly onto the plane.
    signed_distance = points @ normal + d
    projected = points - signed_distance[:, None] * normal

    # Use the projected-point centroid as local origin.
    origin = projected.mean(axis=0)

    # Determine the dominant in-plane direction from PCA.
    centered = projected - origin
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)

    order = np.argsort(values)[::-1]

    u = vectors[:, order[0]]

    # Guarantee u lies exactly in the plane.
    u = u - np.dot(u, normal) * normal
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # Project the points into the plane's local 2D coordinates.
    local_u = centered @ u
    local_v = centered @ v

    u_min = local_u.min() - margin
    u_max = local_u.max() + margin
    v_min = local_v.min() - margin
    v_max = local_v.max() + margin

    # Rectangular finite boundary in local coordinates.
    corners_2d = [
        (u_min, v_min),
        (u_max, v_min),
        (u_max, v_max),
        (u_min, v_max),
    ]

    corners_3d = np.array([
        origin + uu * u + vv * v
        for uu, vv in corners_2d
    ])

    return corners_3d

def main():
    print("=" * 70)
    print("Semantic RANSAC-to-IDF Reconstruction")
    print("=" * 70)

    clouds = {}

    for semantic_class, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        cloud = load_cloud(path)
        points = np.asarray(cloud.points)

        clouds[semantic_class] = cloud

        print(f"\n{semantic_class.upper()}")
        print(f"  Points: {len(points):,}")
        print(f"  Min XYZ: {points.min(axis=0)}")
        print(f"  Max XYZ: {points.max(axis=0)}")
    print("\n" + "=" * 70)
    print("RANSAC ROOF PLANE EXTRACTION")
    print("=" * 70)
    # Fixed seed makes RANSAC reproducible during development.
    o3d.utility.random.seed(42)

    roof_planes, roof_remaining = extract_ransac_planes(clouds["roof"])

    for plane in roof_planes:
        a, b, c, d = plane["model"]

        print(
            f"Plane {plane['id']:2d}: "
            f"{plane['n_points']:7,d} points | "
            f"normal = [{a: .4f}, {b: .4f}, {c: .4f}] | "
            f"d = {d: .4f}"
        )

    print(f"\nDetected roof planes: {len(roof_planes)}")
    print(f"Unassigned roof points: {len(roof_remaining.points):,}")
    V, H1, H2 = estimate_building_frame(
        clouds["column"],
        clouds["roof"],
    )

    classifications = classify_planes_by_principal_normal(
        roof_planes,
        V,
        H1,
        H2,
    )

    regularized_planes = regularize_planes_to_principal_normals(
        roof_planes,
        V,
        H1,
        H2,
        max_angle_deg=6.0,
    )

    duplicate_groups = group_duplicate_regularized_planes(
        regularized_planes,
        offset_threshold=0.01,
    )

    print("\nDUPLICATE REGULARIZED PLANE GROUPS")
    for group_id, group in enumerate(duplicate_groups):
        plane_ids = [
            regularized_planes[i]["id"]
            for i in group
        ]

        family = regularized_planes[group[0]]["principal_family"]

        offsets = [
            regularized_planes[i]["model"][3]
            for i in group
        ]

        print(
            f"Group {group_id:2d}: "
            f"planes={plane_ids} | "
            f"family={family} | "
            f"d={[round(d, 4) for d in offsets]}"
        )

    # Export finite regularized plane boundaries for inspection.
    boundary_points = []
    boundary_lines = []

    for plane in regularized_planes:
        corners = finite_plane_boundary_from_inliers(
            plane,
            margin=0.01,
        )

        start = len(boundary_points)
        boundary_points.extend(corners)

        boundary_lines.extend([
            [start + 0, start + 1],
            [start + 1, start + 2],
            [start + 2, start + 3],
            [start + 3, start + 0],
        ])

    triangles = []

    for start in range(0, len(boundary_points), 4):
        triangles.append([start, start + 1, start + 2])
        triangles.append([start, start + 2, start + 3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(boundary_points)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(triangles, dtype=int)
    )
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        "regularized_plane_boundaries.ply",
        mesh,
    )

    print(
        "\nFinite plane boundaries written to "
        "regularized_plane_boundaries.ply"
    )

    print("\nPRINCIPAL-NORMAL REGULARIZATION")
    for plane in regularized_planes:
        status = "SNAPPED" if plane["snapped"] else "PRESERVED"
        model = plane["model"]

        print(
            f"Plane {plane['id']:2d}: "
            f"{status:9s} | "
            f"{plane['principal_family']:2s} | "
            f"{plane['principal_angle']:5.2f} deg | "
            f"normal=[{model[0]: .4f}, "
            f"{model[1]: .4f}, "
            f"{model[2]: .4f}] | "
            f"d={model[3]: .4f}"
        )

    print("\nRANSAC PLANE PRINCIPAL-NORMAL CLASSIFICATION")
    for item in classifications:
        print(
            f"Plane {item['plane_id']:2d}: "
            f"{item['family']:2s} | "
            f"deviation = {item['angle']:6.2f} deg"
        )

    print("\nBUILDING PRINCIPAL FRAME")
    print(f"V  = {V}")
    print(f"H1 = {H1}")
    print(f"H2 = {H2}")

    print("\nOrthogonality:")
    print(f"V·H1  = {np.dot(V, H1):.8f}")
    print(f"V·H2  = {np.dot(V, H2):.8f}")
    print(f"H1·H2 = {np.dot(H1, H2):.8f}")
    print("\nAll semantic point clouds loaded successfully.")


if __name__ == "__main__":
    main()
