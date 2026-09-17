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

# Based on PCA analysis of the column point cloud:
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

        planes.append(
            {
                "id": plane_id,
                "model": np.asarray(plane_model),
                "cloud": plane_cloud,
                "n_points": len(inliers),
            }
        )

        remaining = remaining.select_by_index(inliers, invert=True)

    return planes, remaining


def plane_rectangle_boundary(plane):
    """
    Estimate a finite rectangular boundary for a RANSAC plane.

    The inlier points are:
      1. projected exactly onto the RANSAC plane,
      2. represented in a local 2D coordinate system,
      3. fit with an oriented minimum-area rectangle,
      4. converted back into four 3D corner vertices.

    This produces a preliminary BRep-like face. Adjacent faces are not
    yet intersected or trimmed to form a watertight solid.
    """
    points = np.asarray(plane["cloud"].points)

    if len(points) < 3:
        raise RuntimeError("Plane contains too few points for a boundary.")

    model = np.asarray(plane["model"], dtype=float)
    normal = model[:3]
    d = float(model[3])

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise RuntimeError("Invalid RANSAC plane normal.")

    normal = normal / normal_norm
    d = d / normal_norm

    # Project all inlier points exactly onto the fitted plane.
    signed_distances = points @ normal + d
    projected = points - signed_distances[:, None] * normal[None, :]

    origin = projected.mean(axis=0)

    # Build a stable orthonormal coordinate system lying in the plane.
    reference_axes = np.eye(3)
    reference = reference_axes[np.argmin(np.abs(reference_axes @ normal))]

    axis_u = np.cross(normal, reference)
    axis_u = axis_u / np.linalg.norm(axis_u)

    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)

    centered = projected - origin

    local_2d = np.column_stack(
        (
            centered @ axis_u,
            centered @ axis_v,
        )
    )

    # Open3D computes the minimum-area oriented rectangle in 2D.
    points_2d_3d = np.column_stack(
        (
            local_2d,
            np.zeros(len(local_2d)),
        )
    )

    local_cloud = o3d.geometry.PointCloud()
    local_cloud.points = o3d.utility.Vector3dVector(points_2d_3d)

    obb = local_cloud.get_oriented_bounding_box()

    center = obb.center[:2]
    rotation = obb.R[:2, :2]
    half_extent = obb.extent[:2] / 2.0

    local_corners = np.array(
        [
            [-half_extent[0], -half_extent[1]],
            [ half_extent[0], -half_extent[1]],
            [ half_extent[0],  half_extent[1]],
            [-half_extent[0],  half_extent[1]],
        ]
    )

    local_corners = local_corners @ rotation.T + center

    corners_3d = (
        origin
        + local_corners[:, 0, None] * axis_u
        + local_corners[:, 1, None] * axis_v
    )

    edge_1 = np.linalg.norm(corners_3d[1] - corners_3d[0])
    edge_2 = np.linalg.norm(corners_3d[2] - corners_3d[1])
    area = edge_1 * edge_2

    return {
        "corners": corners_3d,
        "edge_lengths": (edge_1, edge_2),
        "area": area,
    }


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

    print("\n" + "=" * 70)
    print("FINITE PLANE BOUNDARIES")
    print("=" * 70)

    for plane in roof_planes:
        boundary = plane_rectangle_boundary(plane)
        plane["boundary"] = boundary

        print(
            f"\nPlane {plane['id']:2d} | "
            f"size = {boundary['edge_lengths'][0]:.4f} x "
            f"{boundary['edge_lengths'][1]:.4f} | "
            f"area = {boundary['area']:.6f}"
        )

        for corner_id, corner in enumerate(boundary["corners"], start=1):
            print(
                f"  P{corner_id}: "
                f"[{corner[0]: .6f}, "
                f"{corner[1]: .6f}, "
                f"{corner[2]: .6f}]"
            )

    print("\nVertical axis: Y")
    print("\nAll semantic point clouds loaded successfully.")
    print("Finite rectangular boundaries generated successfully.")


if __name__ == "__main__":
    main()
