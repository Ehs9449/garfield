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


def plane_plane_intersection(model1, model2, parallel_tolerance=1e-3):
    """
    Compute the infinite 3D intersection line of two planes.

    Returns:
        point     - one point on the intersection line
        direction - unit direction vector of the line

    Returns None when the planes are parallel or nearly parallel.
    """
    m1 = np.asarray(model1, dtype=float)
    m2 = np.asarray(model2, dtype=float)

    n1 = m1[:3]
    n2 = m2[:3]
    d1 = m1[3]
    d2 = m2[3]

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return None

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    d1 = d1 / n1_norm
    d2 = d2 / n2_norm

    direction = np.cross(n1, n2)
    direction_norm = np.linalg.norm(direction)

    if direction_norm < parallel_tolerance:
        return None

    direction = direction / direction_norm

    # Solve:
    # n1 . x = -d1
    # n2 . x = -d2
    # direction . x = 0
    A = np.vstack((n1, n2, direction))
    b = np.array([-d1, -d2, 0.0])

    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    return point, direction


def rectangle_line_interval(corners, point, direction, tolerance=0.02):
    """
    Estimate the portion of an infinite intersection line that passes
    through or sufficiently near a rectangular RANSAC face.

    Returns the min/max line parameters or None.
    """
    corners = np.asarray(corners)

    t = (corners - point) @ direction
    projected = point + t[:, None] * direction

    distances = np.linalg.norm(corners - projected, axis=1)

    if distances.min() > tolerance:
        # The line may still cross the interior, so use the rectangle
        # center as an additional proximity test.
        center = corners.mean(axis=0)
        tc = np.dot(center - point, direction)
        closest = point + tc * direction

        # Approximate rectangle radius.
        radius = 0.5 * max(
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
        )

        if np.linalg.norm(center - closest) > radius + tolerance:
            return None

    return float(t.min()), float(t.max())


def find_candidate_intersections(
    planes,
    support_distance=0.03,
    min_support_points=30,
    min_length=0.03,
):
    """
    Find plane-plane intersections supported by actual RANSAC inlier points.

    A pair is accepted only when BOTH planes contain enough points close
    to their mathematical intersection line.
    """
    candidates = []

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            result = plane_plane_intersection(
                planes[i]["model"],
                planes[j]["model"],
            )

            if result is None:
                continue

            point, direction = result

            points_i = np.asarray(planes[i]["cloud"].points)
            points_j = np.asarray(planes[j]["cloud"].points)

            # Perpendicular distance from every inlier point to the
            # infinite plane-plane intersection line.
            vec_i = points_i - point
            vec_j = points_j - point

            t_i = vec_i @ direction
            t_j = vec_j @ direction

            closest_i = point + t_i[:, None] * direction
            closest_j = point + t_j[:, None] * direction

            dist_i = np.linalg.norm(points_i - closest_i, axis=1)
            dist_j = np.linalg.norm(points_j - closest_j, axis=1)

            support_i = t_i[dist_i <= support_distance]
            support_j = t_j[dist_j <= support_distance]

            if len(support_i) < min_support_points:
                continue

            if len(support_j) < min_support_points:
                continue

            # Use only the line interval that is supported by points
            # from BOTH RANSAC planes.
            t_start = max(
                np.percentile(support_i, 5),
                np.percentile(support_j, 5),
            )
            t_end = min(
                np.percentile(support_i, 95),
                np.percentile(support_j, 95),
            )

            if t_end <= t_start:
                continue

            p_start = point + t_start * direction
            p_end = point + t_end * direction
            length = np.linalg.norm(p_end - p_start)

            if length < min_length:
                continue

            candidates.append(
                {
                    "plane_i": i,
                    "plane_j": j,
                    "start": p_start,
                    "end": p_end,
                    "length": length,
                    "support_i": len(support_i),
                    "support_j": len(support_j),
                }
            )

    return candidates

def export_intersection_lines(intersections, output_path):
    """Export candidate BRep edges as an Open3D LineSet PLY."""
    points = []
    lines = []

    for intersection in intersections:
        start_index = len(points)

        points.append(intersection["start"])
        points.append(intersection["end"])

        lines.append([start_index, start_index + 1])

    line_set = o3d.geometry.LineSet()

    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))

    if not o3d.io.write_line_set(str(output_path), line_set):
        raise RuntimeError(
            f"Failed to write intersection lines: {output_path}"
        )


def clip_polygon_with_line(
    polygon,
    line_point,
    line_direction,
    plane_normal,
    keep_point,
    tolerance=1e-9,
):
    """
    Clip a planar 3D polygon by one plane-plane intersection line.

    The intersection line divides the face plane into two half-planes.
    We keep the side containing keep_point, which is taken from the
    actual RANSAC inlier cloud.
    """
    polygon = np.asarray(polygon, dtype=float)

    if len(polygon) < 3:
        return polygon

    # Direction within the face plane and perpendicular to cutting line.
    side_normal = np.cross(plane_normal, line_direction)
    side_norm = np.linalg.norm(side_normal)

    if side_norm < 1e-12:
        return polygon

    side_normal /= side_norm

    keep_sign = np.dot(keep_point - line_point, side_normal)

    if abs(keep_sign) < tolerance:
        return polygon

    if keep_sign < 0:
        side_normal = -side_normal

    def signed_value(point):
        return np.dot(point - line_point, side_normal)

    output = []

    for k in range(len(polygon)):
        current = polygon[k]
        previous = polygon[k - 1]

        current_value = signed_value(current)
        previous_value = signed_value(previous)

        current_inside = current_value >= -tolerance
        previous_inside = previous_value >= -tolerance

        if current_inside:
            if not previous_inside:
                denom = previous_value - current_value
                if abs(denom) > 1e-12:
                    alpha = previous_value / denom
                    intersection = previous + alpha * (current - previous)
                    output.append(intersection)

            output.append(current)

        elif previous_inside:
            denom = previous_value - current_value
            if abs(denom) > 1e-12:
                alpha = previous_value / denom
                intersection = previous + alpha * (current - previous)
                output.append(intersection)

    return np.asarray(output, dtype=float)


def crop_faces_with_intersections(planes, intersections):
    """
    Experimentally crop every preliminary rectangular face using all
    candidate intersection lines associated with that plane.

    For each cut, retain the half-plane containing the median RANSAC
    inlier location. This is intentionally exploratory because the
    current candidate-intersection set is over-connected.
    """
    cropped = []

    for plane_id, plane in enumerate(planes):
        polygon = np.asarray(
            plane["boundary"]["corners"],
            dtype=float,
        ).copy()

        cloud_points = np.asarray(plane["cloud"].points)

        # Robust representative point from the real RANSAC support.
        keep_point = np.median(cloud_points, axis=0)

        plane_normal = np.asarray(
            plane["boundary"]["normal"],
            dtype=float,
        )

        cuts_used = 0

        for item in intersections:
            if item["plane_i"] != plane_id and item["plane_j"] != plane_id:
                continue

            line_start = np.asarray(item["start"], dtype=float)
            line_end = np.asarray(item["end"], dtype=float)

            line_direction = line_end - line_start
            line_norm = np.linalg.norm(line_direction)

            if line_norm < 1e-12:
                continue

            line_direction /= line_norm

            new_polygon = clip_polygon_with_line(
                polygon,
                line_start,
                line_direction,
                plane_normal,
                keep_point,
            )

            # Never destroy a face because of one bad experimental cut.
            if len(new_polygon) >= 3:
                polygon = new_polygon
                cuts_used += 1

        cropped.append(
            {
                "plane_id": plane_id,
                "vertices": polygon,
                "cuts_used": cuts_used,
            }
        )

    return cropped


def export_cropped_faces(cropped_faces, output_path):
    """
    Export cropped polygons as one triangulated PLY mesh.
    """
    vertices = []
    triangles = []

    for face in cropped_faces:
        polygon = np.asarray(face["vertices"], dtype=float)

        if len(polygon) < 3:
            continue

        start = len(vertices)
        vertices.extend(polygon.tolist())

        # Fan triangulation is valid because sequential half-plane
        # clipping preserves convexity.
        for k in range(1, len(polygon) - 1):
            triangles.append(
                [start, start + k, start + k + 1]
            )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=float)
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(triangles, dtype=np.int32)
    )

    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(
            f"Failed to write cropped-face mesh: {output_path}"
        )

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

    # ------------------------------------------------------------
    # Export rectangular RANSAC faces for CloudCompare inspection
    # ------------------------------------------------------------

    output_dir = Path("outputs/ransac_brep_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = []
    triangles = []

    for plane in roof_planes:
        corners = plane["boundary"]["corners"]
        start = len(vertices)

        vertices.extend(corners.tolist())

        # Two triangles form each rectangular face.
        triangles.append([start, start + 1, start + 2])
        triangles.append([start, start + 2, start + 3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))

    mesh.compute_vertex_normals()

    mesh_path = output_dir / "roof_ransac_rectangles.ply"

    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Failed to write visualization mesh: {mesh_path}")

    print("\nVertical axis: Y")
    print("\nAll semantic point clouds loaded successfully.")
    print("Finite rectangular boundaries generated successfully.")
    print(f"Visualization mesh written to: {mesh_path}")

    intersections = find_candidate_intersections(roof_planes)

    cropped_faces = crop_faces_with_intersections(
        roof_planes,
        intersections,
    )

    cropped_path = (
        output_dir / "roof_ransac_faces_cropped_experiment.ply"
    )

    export_cropped_faces(
        cropped_faces,
        cropped_path,
    )

    print("\nEXPERIMENTAL CROPPED FACES")

    for face in cropped_faces:
        print(
            f"  Plane {face['plane_id']:2d}: "
            f"{len(face['vertices'])} vertices | "
            f"{face['cuts_used']} cuts"
        )

    print(f"Cropped mesh written to: {cropped_path}")

    intersection_path = (
        output_dir / "roof_candidate_intersection_lines.ply"
    )

    export_intersection_lines(
        intersections,
        intersection_path,
    )

    print(f"Candidate plane intersections: {len(intersections)}")

    for item in intersections:
        print(
            f"  Plane {item['plane_i']:2d} <-> "
            f"Plane {item['plane_j']:2d} | "
            f"length = {item['length']:.4f}"
        )

    print(
        f"Intersection lines written to: "
        f"{intersection_path}"
    )


if __name__ == "__main__":
    main()
