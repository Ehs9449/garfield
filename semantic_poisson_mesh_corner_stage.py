import open3d as o3d
import numpy as np

INPUT = "data/towerlsu_idf_cleaned/cleaned_final_SAM3/roof - Cloud.ply"
OUTPUT = "roof_poisson_mesh.ply"

print("Loading point cloud...")
pcd = o3d.io.read_point_cloud(INPUT)
print(f"Points: {len(pcd.points):,}")

print("Estimating normals...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.03,
        max_nn=30,
    )
)

print("Orienting normals...")
pcd.orient_normals_consistent_tangent_plane(30)

print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=9,
)

print(
    f"Initial mesh: {len(mesh.vertices):,} vertices, "
    f"{len(mesh.triangles):,} triangles"
)

# Remove the least-supported Poisson-generated regions.
densities = np.asarray(densities)
threshold = np.quantile(densities, 0.02)
mesh.remove_vertices_by_mask(densities < threshold)

mesh.compute_vertex_normals()

print(
    f"Final mesh: {len(mesh.vertices):,} vertices, "
    f"{len(mesh.triangles):,} triangles"
)

o3d.io.write_triangle_mesh(OUTPUT, mesh)

print(f"Saved: {OUTPUT}")
