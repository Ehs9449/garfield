#!/usr/bin/env python3
"""
Extract orthogonal views from a trained GARFIELD model for each cluster.
"""

import torch
import yaml
from pathlib import Path
from typing import Optional
import tyro
from dataclasses import dataclass
import numpy as np
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType


@dataclass
class ExtractViewsConfig:
    """Configuration for extracting orthogonal views."""
    
    checkpoint_path: Path
    """Path to the trained model checkpoint."""
    
    config_path: Path
    """Path to the config.yml file."""
    
    output_dir: Path = Path("orthogonal_views")
    """Directory to save the extracted views."""
    
    image_size: int = 2048
    """Size of the rendered images (width and height)."""
    
    distance: float = 5.0
    """Distance of camera from the scene center."""


def create_orthogonal_camera(
    direction: str,
    image_size: int,
    distance: float,
    scene_center: torch.Tensor,
    device: str = "cuda"
) -> Cameras:
    """
    Create an orthogonal camera looking at the scene from a specific direction.
    
    Args:
        direction: One of 'front', 'back', 'left', 'right', 'top', 'bottom'
        image_size: Size of the output image
        distance: Distance from scene center
        scene_center: Center point of the scene
        device: Device to create tensors on
    
    Returns:
        Camera object
    """
    # Define camera positions for each direction
    positions = {
        'front': torch.tensor([0, 0, distance]),      # Looking along +Z
        'back': torch.tensor([0, 0, -distance]),      # Looking along -Z
        'left': torch.tensor([-distance, 0, 0]),      # Looking along -X
        'right': torch.tensor([distance, 0, 0]),      # Looking along +X
        'top': torch.tensor([0, distance, 0]),        # Looking along +Y (down)
        'bottom': torch.tensor([0, -distance, 0]),    # Looking along -Y (up)
    }
    
    # Define up vectors for each direction
    up_vectors = {
        'front': torch.tensor([0, 1, 0]),
        'back': torch.tensor([0, 1, 0]),
        'left': torch.tensor([0, 1, 0]),
        'right': torch.tensor([0, 1, 0]),
        'top': torch.tensor([0, 0, -1]),
        'bottom': torch.tensor([0, 0, 1]),
    }
    
    # Get camera position and up vector
    camera_pos = positions[direction].float() + scene_center.cpu()
    up = up_vectors[direction].float()
    
    # Calculate camera-to-world matrix
    # Forward direction (camera looks along -Z in camera space)
    forward = (scene_center.cpu() - camera_pos)
    forward = forward / torch.norm(forward)
    
    # Right direction
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    
    # Recalculate up to ensure orthogonality
    up = torch.cross(right, forward)
    up = up / torch.norm(up)
    
    # Build rotation matrix (camera to world)
    c2w_rotation = torch.stack([right, up, -forward], dim=1)
    
    # Build full camera-to-world matrix
    c2w = torch.eye(4)
    c2w[:3, :3] = c2w_rotation
    c2w[:3, 3] = camera_pos
    
    # Camera intrinsics (orthographic-like with large FOV)
    focal_length = image_size / 2.0
    cx, cy = image_size / 2.0, image_size / 2.0
    
    camera = Cameras(
        camera_to_worlds=c2w.unsqueeze(0).to(device),
        fx=torch.tensor([[focal_length]]).to(device),
        fy=torch.tensor([[focal_length]]).to(device),
        cx=torch.tensor([[cx]]).to(device),
        cy=torch.tensor([[cy]]).to(device),
        width=torch.tensor([[image_size]]).to(device),
        height=torch.tensor([[image_size]]).to(device),
        camera_type=CameraType.PERSPECTIVE,
    )
    
    return camera


def extract_views(config: ExtractViewsConfig):
    """Extract orthogonal views from trained GARFIELD model."""
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config from: {config.config_path}")
    print(f"Loading checkpoint from: {config.checkpoint_path}")
    
    # Load the trained pipeline using eval_setup
    _, pipeline, _, _ = eval_setup(
        config.config_path,
        eval_num_rays_per_chunk=None,
        test_mode="test",
    )
    
    device = pipeline.device
    pipeline.eval()
    
    # Get scene bounds to determine camera distance
    if hasattr(pipeline.datamanager, 'train_dataset'):
        # Get scene center from dataparser
        scene_box = pipeline.datamanager.train_dataset.scene_box
        scene_center = (scene_box.aabb[0] + scene_box.aabb[1]) / 2.0
        scene_scale = torch.norm(scene_box.aabb[1] - scene_box.aabb[0])
        distance = config.distance * scene_scale.item()
    else:
        scene_center = torch.tensor([0.0, 0.0, 0.0])
        distance = config.distance
    
    scene_center = scene_center.to(device)
    
    print(f"Scene center: {scene_center}")
    print(f"Camera distance: {distance}")
    
    # Get number of clusters if using GARFIELD
    num_clusters = getattr(pipeline.model, 'num_clusters', 1)
    print(f"Number of clusters: {num_clusters}")
    
    # Define orthogonal views to extract
    directions = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    # Extract views for each cluster
    for cluster_idx in range(num_clusters):
        print(f"\nProcessing cluster {cluster_idx + 1}/{num_clusters}")
        
        cluster_dir = config.output_dir / f"cluster_{cluster_idx:03d}"
        cluster_dir.mkdir(exist_ok=True)
        
        # Render each orthogonal view
        for direction in directions:
            print(f"  Rendering {direction} view...")
            
            # Create camera for this direction
            camera = create_orthogonal_camera(
                direction=direction,
                image_size=config.image_size,
                distance=distance,
                scene_center=scene_center,
                device=device
            )
            
            # Render the view
            with torch.no_grad():
                # Get camera ray bundle
                ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                
                # If GARFIELD model, set active cluster
                if hasattr(pipeline.model, 'set_active_cluster'):
                    pipeline.model.set_active_cluster(cluster_idx)
                
                # Render
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
                
                # Extract RGB image
                rgb = outputs["rgb"].cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)
                
                # Reshape if needed
                if len(rgb.shape) == 3:
                    # Already in correct shape
                    pass
                elif len(rgb.shape) == 2:
                    # Need to reshape
                    rgb = rgb.reshape(config.image_size, config.image_size, 3)
                
                # Save image
                output_path = cluster_dir / f"{direction}.png"
                Image.fromarray(rgb).save(output_path)
                print(f"    Saved: {output_path}")
                
                # Also save depth if available
                if "depth" in outputs:
                    depth = outputs["depth"].cpu().numpy()
                    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_img = (depth_normalized * 255).astype(np.uint8)
                    
                    # Reshape if needed
                    if len(depth_img.shape) == 1:
                        depth_img = depth_img.reshape(config.image_size, config.image_size)
                    elif len(depth_img.shape) == 3:
                        depth_img = depth_img.squeeze()
                    
                    depth_path = cluster_dir / f"{direction}_depth.png"
                    Image.fromarray(depth_img).save(depth_path)
    
    print(f"\n✓ Extraction complete! Views saved to: {config.output_dir}")


if __name__ == "__main__":
    config = tyro.cli(ExtractViewsConfig)
    extract_views(config)
