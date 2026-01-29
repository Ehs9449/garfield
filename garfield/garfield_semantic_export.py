"""
Direct integration with your GarfieldGaussianPipeline for semantic export
Add this to your existing GARFIELD code
"""

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
import json

def add_semantic_export_to_garfield_pipeline(pipeline):
    """
    Add semantic export functionality to your existing GarfieldGaussianPipeline
    Call this function after creating your pipeline
    """
    
    def export_semantic_pointcloud_simple(output_folder="semantic_export"):
        """
        Simple semantic point cloud export using existing GARFIELD data
        """
        print("Starting semantic point cloud export...")
        
        # Create output folder
        Path(output_folder).mkdir(exist_ok=True)
        
        # STEP 1: Get Gaussian data (positions, colors, etc.)
        model = pipeline.model
        
        with torch.no_grad():
            # Get 3D positions
            positions = model.means.detach().cpu().numpy()  # [N, 3]
            
            # Get RGB colors from spherical harmonics
            if hasattr(model, 'shs_0'):
                # Convert SH to RGB (simplified)
                colors = model.shs_0.detach().cpu().numpy().squeeze() + 0.5
                colors = np.clip(colors, 0, 1)
            else:
                colors = np.ones((len(positions), 3)) * 0.5
        
        print(f"Extracted {len(positions)} Gaussians")
        
        # STEP 2: Get semantic labels
        semantic_labels = get_semantic_labels_from_pipeline(pipeline, len(positions))
        
        # STEP 3: Create semantic colors
        semantic_colors = create_semantic_colors(semantic_labels)
        
        # STEP 4: Save point clouds
        save_semantic_pointclouds(positions, colors, semantic_colors, semantic_labels, output_folder)
        
        print(f"Export completed! Check folder: {output_folder}")
        
        return {
            'positions': positions,
            'colors': colors,
            'semantic_labels': semantic_labels,
            'num_points': len(positions)
        }
    
    # Add the export function to your pipeline
    pipeline.export_semantic_pointcloud = export_semantic_pointcloud_simple
    
    return pipeline

def get_semantic_labels_from_pipeline(pipeline, num_points):
    """
    Extract semantic labels from your GARFIELD pipeline
    """
    
    # METHOD 1: Use cluster_labels (from clustering)
    if hasattr(pipeline, 'cluster_labels') and pipeline.cluster_labels is not None:
        cluster_labels = pipeline.cluster_labels.detach().cpu().numpy()
        print(f"Found cluster labels: {len(cluster_labels)} labels")
        
        if len(cluster_labels) == num_points:
            # Map cluster IDs to building element semantics
            # You can customize this mapping based on your data
            semantic_labels = map_clusters_to_building_elements(cluster_labels)
            return semantic_labels
    
    # METHOD 2: Use GARFIELD grouping features to create semantic labels
    if hasattr(pipeline, 'garfield_pipeline') and len(pipeline.garfield_pipeline) > 0:
        try:
            grouping_model = pipeline.garfield_pipeline[0].model
            positions = torch.from_numpy(pipeline.model.means.detach().cpu().numpy()).to(pipeline.device)
            
            # Get grouping features at a reasonable scale
            scale = 0.5  # You can adjust this
            with torch.no_grad():
                group_features = grouping_model.get_grouping_at_points(positions, scale)
                
            # Simple clustering of features to get semantic groups
            semantic_labels = cluster_grouping_features_to_semantics(group_features.cpu().numpy())
            print(f"Generated semantic labels from grouping features")
            return semantic_labels
            
        except Exception as e:
            print(f"Warning: Could not use grouping features: {e}")
    
    # METHOD 3: Fallback - position-based labeling
    print("Using position-based semantic labeling as fallback")
    positions = pipeline.model.means.detach().cpu().numpy()
    return create_position_based_semantic_labels(positions)

def map_clusters_to_building_elements(cluster_labels):
    """
    Map your cluster IDs to building element categories
    Customize this based on your training data
    """
    # Define building elements
    building_elements = {
        0: 'background',
        1: 'window',
        2: 'door',
        3: 'column',
        4: 'wall', 
        5: 'roof',
        6: 'balcony',
        7: 'stair',
        8: 'facade'
    }
    
    # Simple mapping: map cluster IDs to semantic categories
    # You might need to adjust this based on your specific clustering results
    max_clusters = cluster_labels.max()
    num_elements = len(building_elements)
    
    # Create mapping from cluster ID to semantic label
    semantic_labels = np.zeros_like(cluster_labels, dtype=int)
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id >= 0:  # Skip noise (-1)
            # Map to building element (you can customize this logic)
            semantic_class = cluster_id % num_elements
            mask = cluster_labels == cluster_id
            semantic_labels[mask] = semantic_class
    
    return semantic_labels

def cluster_grouping_features_to_semantics(group_features):
    """
    Cluster grouping features into semantic categories
    """
    try:
        from sklearn.cluster import KMeans
        
        # Cluster into 9 semantic categories (background + 8 building elements)
        n_clusters = 9
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        semantic_labels = kmeans.fit_predict(group_features)
        
        return semantic_labels
        
    except ImportError:
        print("scikit-learn not available, using position-based fallback")
        return np.zeros(len(group_features), dtype=int)

def create_position_based_semantic_labels(positions):
    """
    Create semantic labels based on 3D position (simple fallback)
    """
    heights = positions[:, 2]  # Z coordinate
    
    # Normalize heights
    min_h, max_h = heights.min(), heights.max()
    norm_heights = (heights - min_h) / (max_h - min_h + 1e-8)
    
    # Simple height-based semantic assignment
    labels = np.zeros(len(positions), dtype=int)
    
    # Bottom 20% -> doors (2)
    labels[norm_heights <= 0.2] = 2  
    
    # 20-60% -> walls (4)  
    labels[(norm_heights > 0.2) & (norm_heights <= 0.6)] = 4
    
    # 60-80% -> windows (1)
    labels[(norm_heights > 0.6) & (norm_heights <= 0.8)] = 1
    
    # Top 20% -> roof (5)
    labels[norm_heights > 0.8] = 5
    
    return labels

def create_semantic_colors(semantic_labels):
    """
    Create colors for each semantic category
    """
    color_map = {
        0: [0.5, 0.5, 0.5],  # background - gray
        1: [0.0, 0.8, 1.0],  # window - light blue
        2: [0.8, 0.4, 0.0],  # door - brown
        3: [0.9, 0.9, 0.9],  # column - white
        4: [0.7, 0.7, 0.6],  # wall - light gray
        5: [0.8, 0.2, 0.2],  # roof - red
        6: [0.2, 0.8, 0.2],  # balcony - green
        7: [1.0, 1.0, 0.0],  # stair - yellow
        8: [0.6, 0.4, 0.2],  # facade - beige
    }
    
    semantic_colors = np.array([
        color_map.get(label, [0.5, 0.5, 0.5]) 
        for label in semantic_labels
    ])
    
    return semantic_colors

def save_semantic_pointclouds(positions, colors, semantic_colors, semantic_labels, output_folder):
    """
    Save point clouds in different formats
    """
    output_path = Path(output_folder)
    
    # 1. Save RGB point cloud
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(positions)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_path / "building_rgb.ply"), pcd_rgb)
    print("Saved: building_rgb.ply")
    
    # 2. Save semantic point cloud
    pcd_semantic = o3d.geometry.PointCloud() 
    pcd_semantic.points = o3d.utility.Vector3dVector(positions)
    pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
    o3d.io.write_point_cloud(str(output_path / "building_semantic.ply"), pcd_semantic)
    print("Saved: building_semantic.ply")
    
    # 3. Save individual building elements
    save_individual_building_elements(positions, colors, semantic_labels, output_path)
    
    # 4. Save metadata
    save_metadata(semantic_labels, output_path)

def save_individual_building_elements(positions, colors, semantic_labels, output_path):
    """
    Save each building element as separate PLY file
    """
    element_names = {
        0: 'background',
        1: 'windows',
        2: 'doors', 
        3: 'columns',
        4: 'walls',
        5: 'roof',
        6: 'balconies',
        7: 'stairs',
        8: 'facade'
    }
    
    elements_dir = output_path / "building_elements"
    elements_dir.mkdir(exist_ok=True)
    
    for label_id, element_name in element_names.items():
        mask = semantic_labels == label_id
        
        if not mask.any():
            continue
            
        element_positions = positions[mask]
        element_colors = colors[mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(element_positions)
        pcd.colors = o3d.utility.Vector3dVector(element_colors)
        
        filename = elements_dir / f"{element_name}.ply"
        o3d.io.write_point_cloud(str(filename), pcd)
        print(f"Saved: {element_name} ({len(element_positions)} points)")

def save_metadata(semantic_labels, output_path):
    """
    Save export metadata and statistics
    """
    element_names = {
        0: 'background', 1: 'windows', 2: 'doors', 3: 'columns',
        4: 'walls', 5: 'roof', 6: 'balconies', 7: 'stairs', 8: 'facade'
    }
    
    # Count points per element
    element_counts = {}
    total_points = len(semantic_labels)
    
    for label_id, element_name in element_names.items():
        count = np.sum(semantic_labels == label_id)
        if count > 0:
            element_counts[element_name] = {
                'count': int(count),
                'percentage': round((count / total_points) * 100, 2)
            }
    
    metadata = {
        'total_points': int(total_points),
        'building_elements': element_counts,
        'export_info': {
            'method': 'GARFIELD Gaussian Splatting Export',
            'semantic_source': 'cluster_labels or grouping_features'
        }
    }
    
    # Save JSON metadata
    with open(output_path / "export_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print statistics
    print("\nBuilding Elements Summary:")
    print("-" * 40)
    for element_name, info in element_counts.items():
        print(f"{element_name:12s}: {info['count']:6,} points ({info['percentage']:5.1f}%)")
    print("-" * 40)

# ==============================================================================
# INTEGRATION EXAMPLE
# ==============================================================================

def example_usage():
    """
    Example of how to integrate with your GARFIELD pipeline
    """
    
    # After you create your GarfieldGaussianPipeline
    # pipeline = GarfieldGaussianPipeline(config, device, ...)
    
    # Add semantic export functionality
    # pipeline = add_semantic_export_to_garfield_pipeline(pipeline)
    
    # Export semantic point cloud
    # results = pipeline.export_semantic_pointcloud("my_building_export")
    
    print("""
    INTEGRATION STEPS:
    
    1. Add this code to your GARFIELD project
    
    2. After creating your pipeline, add export functionality:
       
       pipeline = add_semantic_export_to_garfield_pipeline(pipeline)
    
    3. Export semantic point clouds:
       
       pipeline.export_semantic_pointcloud("output_folder")
    
    4. You'll get:
       - building_rgb.ply (original colors)
       - building_semantic.ply (colored by building element)
       - building_elements/ folder with individual elements
       - export_metadata.json with statistics
    """)

if __name__ == "__main__":
    example_usage()
