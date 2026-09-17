from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


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
    """Repeatedly extract dominant planes from a point cloud using RANSAC."""
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


def minimum_area_rectangle_2d(points):
    """
    Find the minimum-area oriented rectangle enclosing a set of 2D points.

    Returns four rectangle corners in counterclockwise order.
    """
    if len(points) < 3:
        raise RuntimeError("At least three 2D points are required.")

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    edges = np.roll(hull_points, -1, axis=0) - hull_points

    best_area = np.inf
    best_corners = None

    for edge in edges:
        edge_length = np.linalg.norm(edge)

        if edge_length < 1e-12:
            continue

        # Rotate this hull edge onto the local x-axis.
        angle = np.arctan2(edge[1], edge[0])

        c = np.cos(angle)
        s = np.sin(angle)

        rotation = np.array(
            [
                [c, s],
                [-s, c],
            ]
        )

        rotated = hull_points @ rotation.T

        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)

        width = max_xy[0] - min_xy[0]
        height = max_xy[1] - min_xy[1]
        area = width * height

        if area < best_area:
            rectangle_rotated = np.array(
                [
                    [min_xy[0], min_xy[1]],
                    [max_xy[0], min_xy[1]],
                    [max_xy[0], max_xy[1]],
                    [min_xy[0], max_xy[1]],
                ]
            )

            # Transform rectangle back into the original 2D coordinates.
            best_corners = rectangle_rotated @ rotation
            best_area = area

    if best_corners is None:
        raise RuntimeError("Could not determine a minimum-area rectangle.")

    return best_corners


def plane_rectangle_boundary(plane):
    """
    Estimate a finite rectangular boundary for one RANSAC plane.

    Steps:
      1. Project RANSAC inlier points exactly onto the fitted plane.
      2. Construct a local 2D coordinate system on that plane.
      3. Compute the 2D convex hull.
      4. Find its minimum-area enclosing rectangle.
      5. Transform the four rectangle corners back into 3D.

    The resulting rectangle is a preliminary BRep-like face.
    Faces are not yet intersected, merged, or made watertight.
    """
    points = np.asarray(plane["cloud"].points)

    if len(points) < 3:
        raise RuntimeError("Plane contains too few points for a boundary.")

    model = np.asarray(plane["model"], dtype=float)

    normal = model[:3]
    d = float(model[3])

    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-12:
        raise RuntimeError("Invalid RANSAC plane normal.")

    normal = normal / normal_norm
    d = d / normal_norm

    # Project inliers exactly onto the mathematical RANSAC plane.
    signed_distances = points @ normal + d
    projected = points - signed_distances[:, None] * normal[None, :]

    origin = projected.mean(axis=0)

    # Choose the global axis least parallel to the plane normal.
    global_axes = np.eye(3)
    reference = global_axes[np.argmin(np.abs(global_axes @ normal))]

    # Construct two orthonormal axes lying on the plane.
    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u)

    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)

    centered = projected - origin

    local_2d = np.column_stack(
        (
            centered @ axis_u,
            centered @ axis_v,
        )
    )

    corners_2d = minimum_area_rectangle_2d(local_2d)

    # Transform the four finite boundary vertices back to XYZ.
    corners_3d = (
        origin
        + corners_2d[:, 0, None] * axis_u
        + corners_2d[:, 1, None] * axis_v
    )

    edge_1 = np.linalg.norm(corners_3d[1] - corners_3d[0])
    edge_2 = np.linalg.norm(corners_3d[2] - corners_3d[1])

    area = edge_1 * edge_2

    return {
        "corners": corners_3d,
        "edge_lengths": (edge_1, edge_2),
        "area": area,
        "normal": normal,
        "origin": origin,
        "axis_u": axis_u,
        "axis_v": axis_v,
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

        edge_1, edge_2 = boundary["edge_lengths"]

        print(
            f"\nPlane {plane['id']:2d} | "
            f"size = {edge_1:.4f} x {edge_2:.4f} | "
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
