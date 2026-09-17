#!/usr/bin/env python3
"""
GARField Orthographic Feature Projection Pipeline

Renders GARField instance features in orthographic mode from multiple facade views,
projects a point cloud into those views, looks up smooth rendered features,
averages across views, and clusters with HDBSCAN.

Usage:
    python garfield_ortho_projection.py \
        --config /path/to/garfield/config.yml \
        --pointcloud /path/to/pointcloud.ply \
        --scale 0.1 \
        --output-dir outputs/ortho_projection
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os
import time
import json

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle
import open3d as o3d
from cuml.cluster.hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import cv2


class GarfieldOrthoProjector:
    """
    Renders GARField features orthographically and projects point clouds
    into those rendered views to collect smooth features.
    """

    def __init__(self, config_path: Path, data_path: Path = None,
                 crop_center=None, crop_scale=None):
        """Load the GARField model with crop enabled."""

        print("Loading GARField model...")
        config, self.pipeline, checkpoint_path, step = eval_setup(
            config_path, test_mode='test'
        )
        self.device = self.pipeline.device
        print("✓ Model loaded successfully")

        # Enable cropping to building bounding box
        if crop_center is None:
            crop_center = [0.02, -0.05, -0.15]
        if crop_scale is None:
            crop_scale = [1.0, 0.91, 0.19]

        self.crop_center = np.array(crop_center, dtype=np.float32)
        self.crop_scale = np.array(crop_scale, dtype=np.float32)

        crop_center_t = torch.tensor(self.crop_center, device=self.device)
        crop_scale_t = torch.tensor(self.crop_scale, device=self.device)
        crop_min = crop_center_t - crop_scale_t / 2
        crop_max = crop_center_t + crop_scale_t / 2

        self.pipeline.model.crop_enabled = True
        self.pipeline.model.crop_min = crop_min
        self.pipeline.model.crop_max = crop_max
        self.pipeline.model.crop_bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        print(f"✓ Crop enabled: min={crop_min.tolist()}, max={crop_max.tolist()}")


    def render_ortho_features(self, plane_center, look_dir, up_vec,
                               width, height, img_width, img_height,
                               near, far, scale):
        """
        Render an orthographic view and return raw 256-dim features + depth.

        Returns:
            features: numpy array [H, W, 256]
            depth: numpy array [H, W]
            view_params: dict with projection parameters for later use
        """
        # Set GARField scale
        if hasattr(self.pipeline.model, 'scale_slider'):
            self.pipeline.model.scale_slider.value = scale

        # Convert to tensors
        plane_center = torch.tensor(plane_center, dtype=torch.float32)
        look_dir = torch.tensor(look_dir, dtype=torch.float32)
        up_vec = torch.tensor(up_vec, dtype=torch.float32)

        # Normalize and orthogonalize
        look_dir = look_dir / torch.norm(look_dir)
        right = torch.cross(look_dir, up_vec, dim=0)
        right = right / torch.norm(right)
        up_vec = torch.cross(right, look_dir, dim=0)
        up_vec = up_vec / torch.norm(up_vec)

        # Pixel size in world units
        pixel_size_x = width / img_width
        pixel_size_y = height / img_height

        # Create grid of ray origins on the plane
        u = torch.linspace(-width / 2, width / 2, img_width)
        v = torch.linspace(-height / 2, height / 2, img_height)
        uu, vv = torch.meshgrid(u, v, indexing='xy')

        origins = (
            plane_center.unsqueeze(0).unsqueeze(0) +
            uu.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) +
            vv.unsqueeze(-1) * up_vec.unsqueeze(0).unsqueeze(0)
        )

        # origins is [img_height, img_width, 3]; directions must match it.
        directions = look_dir.unsqueeze(0).unsqueeze(0).expand(img_height, img_width, -1)

        num_rays = img_width * img_height
        origins = origins.reshape(num_rays, 3)
        directions = directions.reshape(num_rays, 3)

        # Render in chunks using model.forward() directly
        # (get_outputs_for_camera_ray_bundle strips instance features)
        chunk_size = 4096
        from collections import defaultdict
        outputs_lists = defaultdict(list)

        with torch.no_grad():
            for i in range(0, num_rays, chunk_size):
                end_idx = min(i + chunk_size, num_rays)
                cs = end_idx - i

                ray_bundle = RayBundle(
                    origins=origins[i:end_idx].to(self.device),
                    directions=directions[i:end_idx].to(self.device),
                    pixel_area=torch.ones((cs, 1), device=self.device) * (pixel_size_x * pixel_size_y),
                    camera_indices=torch.zeros((cs, 1), dtype=torch.int32, device=self.device),
                    nears=torch.ones((cs, 1), device=self.device) * near,
                    fars=torch.ones((cs, 1), device=self.device) * far,
                    metadata={},
                    times=None
                )

                outputs = self.pipeline.model.forward(ray_bundle=ray_bundle)

                for key, val in outputs.items():
                    if isinstance(val, torch.Tensor):
                        outputs_lists[key].append(val.cpu())

        # Combine chunks
        combined = {}
        for key, vals in outputs_lists.items():
            try:
                combined[key] = torch.cat(vals)
            except:
                pass

        print(f"  Available outputs: {list(combined.keys())}")

        # Extract features
        features = None
        if "instance" in combined:
            features = combined["instance"].reshape(img_height, img_width, -1).numpy()
        else:
            print("WARNING: 'instance' not in outputs. Keys:", list(combined.keys()))

        # Extract depth
        depth = None
        if "depth" in combined:
            depth = combined["depth"].reshape(img_height, img_width).numpy()

        # Extract RGB for visualization
        rgb = combined["rgb"].reshape(img_height, img_width, 3).numpy()
        rgb = np.clip(rgb, 0, 1)

        # Store view parameters for projection
        view_params = {
            'plane_center': plane_center.numpy(),
            'look_dir': look_dir.numpy(),
            'right': right.numpy(),
            'up': up_vec.numpy(),
            'width': width,
            'height': height,
            'img_width': img_width,
            'img_height': img_height,
            'pixel_size_x': pixel_size_x,
            'pixel_size_y': pixel_size_y,
            'near': near,
            'far': far,
        }

        return features, depth, rgb, view_params

    def project_points_ortho(self, points, view_params):
        """
        Project 3D points into an orthographic view.
        Returns pixel coordinates and depth along look direction.

        This is trivially simple for orthographic projection:
        - px = dot(point - center, right) / pixel_size + img_width/2
        - py = dot(point - center, up) / pixel_size + img_height/2
        - depth = dot(point - center, look_dir)
        """
        center = view_params['plane_center']
        right = view_params['right']
        up = view_params['up']
        look = view_params['look_dir']

        # Vector from plane center to each point
        diff = points - center  # N x 3

        # Project onto right, up, and look axes
        proj_right = diff @ right   # N
        proj_up = diff @ up         # N
        proj_depth = diff @ look    # N (depth along look direction)

        # Convert to pixel coordinates
        px = (proj_right / view_params['pixel_size_x'] + view_params['img_width'] / 2).astype(np.int32)
        py = (proj_up / view_params['pixel_size_y'] + view_params['img_height'] / 2).astype(np.int32)

        return px, py, proj_depth

    @staticmethod
    def _make_ortho_view(name, center, azimuth_deg, elevation_deg,
                         cam_dist, bbox_corners, padding):
        """Build one orthographic view that looks at the building centre.

        azimuth_deg is measured around +Z from the +X axis.
        elevation_deg is 0 for a horizontal (facade) view and 90 for top-down.
        """
        center = np.asarray(center, dtype=np.float32)

        az = np.radians(float(azimuth_deg))
        el = np.radians(float(elevation_deg))

        # Unit vector from the building centre out to the camera.
        offset = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ], dtype=np.float32)

        plane_center = center + cam_dist * offset
        look_dir = -offset

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(look_dir, world_up)

        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, look_dir)
        up = up / (np.linalg.norm(up) + 1e-8)

        relative = bbox_corners - plane_center
        projected_right = relative @ right
        projected_up = relative @ up

        view_width = (projected_right.max() - projected_right.min()) * padding
        view_height = (projected_up.max() - projected_up.min()) * padding

        return {
            'name': name,
            'plane_center': plane_center.tolist(),
            'look_dir': look_dir.tolist(),
            'up_vec': up.tolist(),
            'view_width': float(view_width),
            'view_height': float(view_height),
        }

    @staticmethod
    def _view_resolution(view_width, view_height, long_side):
        """Pick pixel dimensions that keep pixels close to square."""
        aspect = float(view_height) / max(float(view_width), 1e-8)

        if aspect <= 1.0:
            img_w = int(long_side)
            img_h = max(64, int(round(long_side * aspect)))
        else:
            img_h = int(long_side)
            img_w = max(64, int(round(long_side / aspect)))

        return img_w, img_h

    def run_pipeline(self, pointcloud_path, scale, output_dir,
                     img_resolution=1080,
                     n_side=4, side_elevations=(0.0,),
                     n_top_ring=4, top_elevations=(80.0,),
                     azimuth_offset=0.0, dist_factor=1.5,
                     square_pixels=True, strict_depth=True):
        """
        Full pipeline:
        1. Render orthographic feature maps from multiple views
        2. Project point cloud into each view
        3. Look up features
        4. Average across views
        5. Cluster with HDBSCAN
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ############################################################
        # STEP 1: Load and crop point cloud
        ############################################################
        print("\n" + "=" * 60)
        print("STEP 1: Loading and cropping point cloud")
        print("=" * 60)

        pcd = o3d.io.read_point_cloud(str(pointcloud_path))
        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        print(f"  Loaded {len(points)} points")
        print(f"  Bounds: {points.min(axis=0)} to {points.max(axis=0)}")

        # Crop to configured building bounding box
        crop_center = self.crop_center
        crop_scale = self.crop_scale
        crop_padding = 1.1  # 10% padding beyond crop box
        crop_min = crop_center - (crop_scale * crop_padding) / 2
        crop_max = crop_center + (crop_scale * crop_padding) / 2

        crop_mask = (
            (points[:, 0] >= crop_min[0]) & (points[:, 0] <= crop_max[0]) &
            (points[:, 1] >= crop_min[1]) & (points[:, 1] <= crop_max[1]) &
            (points[:, 2] >= crop_min[2]) & (points[:, 2] <= crop_max[2])
        )

        points = points[crop_mask]
        if colors is not None:
            colors = colors[crop_mask]

        print(f"  After crop: {len(points)} points ({100 * crop_mask.sum() / len(crop_mask):.1f}%)")
        print(f"  Crop box: {crop_min} to {crop_max}")
        print(f"  Cropped bounds: {points.min(axis=0)} to {points.max(axis=0)}")

        N_points = len(points)
        feature_dim = 256  # GARField feature dimension
        feature_sum = np.zeros((N_points, feature_dim), dtype=np.float32)
        feature_count = np.zeros(N_points, dtype=np.int32)

        ############################################################
        # STEP 2: Generate dense orthographic views around building
        ############################################################
        print("\n" + "=" * 60)
        print("STEP 2: Generating dense orthographic views")
        print("=" * 60)

        # Building crop bounds
        cx, cy, cz = self.crop_center
        sx, sy, sz = self.crop_scale
        padding = 1.3  # 30% padding to avoid clipping edges

        # Bounding box
        bbox_min = np.array([cx - sx/2, cy - sy/2, cz - sz/2])
        bbox_max = np.array([cx + sx/2, cy + sy/2, cz + sz/2])
        max_extent = max(sx, sy, sz)
        center = np.array([cx, cy, cz])
        dist = max_extent * 0.7  # distance from center to place camera plane

        print(f"  Crop center: [{cx}, {cy}, {cz}]")
        print(f"  Crop scale:  [{sx}, {sy}, {sz}]")
        print(f"  BBox: {bbox_min} to {bbox_max}")

        # Eight corners of the building crop box.
        bbox_corners = np.array([
            [x, y, z]
            for x in [bbox_min[0], bbox_max[0]]
            for y in [bbox_min[1], bbox_max[1]]
            for z in [bbox_min[2], bbox_max[2]]
        ], dtype=np.float32)

        # Distance from the building centre to the orthographic image plane.
        half_diag = 0.5 * float(np.linalg.norm(np.array([sx, sy, sz])))
        cam_dist = half_diag * dist_factor

        views = []

        # ------------------------------------------------------------
        # Side views: the camera looks horizontally straight at the
        # facade. Oblique drone poses smear features across depth, so
        # the facade-facing directions are used instead.
        # ------------------------------------------------------------
        for el_deg in side_elevations:
            for k in range(n_side):
                az_deg = (azimuth_offset + 360.0 * k / float(n_side)) % 360.0

                views.append(self._make_ortho_view(
                    name=f"side_az{int(round(az_deg)):03d}_el{int(round(el_deg)):02d}",
                    center=center,
                    azimuth_deg=az_deg,
                    elevation_deg=el_deg,
                    cam_dist=cam_dist,
                    bbox_corners=bbox_corners,
                    padding=padding,
                ))

        # ------------------------------------------------------------
        # Top views for the roof.
        # ------------------------------------------------------------
        top_center = np.array(
            [cx, cy, cz + cam_dist],
            dtype=np.float32
        )

        views.append({
            'name': 'top_down',
            'plane_center': top_center.tolist(),
            'look_dir': [0.0, 0.0, -1.0],
            'up_vec': [0.0, 1.0, 0.0],
            'view_width': float(sx * padding),
            'view_height': float(sy * padding),
        })

        for el_deg in top_elevations:
            for k in range(n_top_ring):
                az_deg = (azimuth_offset + 360.0 * k / float(n_top_ring)) % 360.0

                views.append(self._make_ortho_view(
                    name=f"roof_az{int(round(az_deg)):03d}_el{int(round(el_deg)):02d}",
                    center=center,
                    azimuth_deg=az_deg,
                    elevation_deg=el_deg,
                    cam_dist=cam_dist,
                    bbox_corners=bbox_corners,
                    padding=padding,
                ))

        print(f"  Generated {len(views)} orthographic views")
        print(f"    side: {n_side} azimuths x {len(side_elevations)} elevation(s) "
              f"{side_elevations}")
        print(f"    top:  1 top-down + {n_top_ring} x {len(top_elevations)} roof view(s)")
        print(f"    camera distance from centre: {cam_dist:.3f}")

        ############################################################
        # STEP 3: Render features and project points for each view
        ############################################################
        print("\n" + "=" * 60)
        print("STEP 3: Rendering features and projecting points")
        print("=" * 60)

        # Create views subfolder for all rendered images
        views_dir = output_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Saving rendered views to: {views_dir}")

        near_clip = 0.01
        # The image plane sits cam_dist from the centre, so the far side of
        # the building is at cam_dist + half_diag. Keep a margin on top.
        far_clip = float(cam_dist + 2.0 * half_diag)

        view_stats = []

        for view_idx, view in enumerate(views):
            print(f"\n--- View {view_idx + 1}/{len(views)}: {view['name']} ---")

            if square_pixels:
                img_w, img_h = self._view_resolution(
                    view['view_width'], view['view_height'], img_resolution
                )
            else:
                img_w, img_h = img_resolution, img_resolution

            # Render orthographic feature map
            start_time = time.time()
            features, depth, rgb, view_params = self.render_ortho_features(
                plane_center=view['plane_center'],
                look_dir=view['look_dir'],
                up_vec=view['up_vec'],
                width=view['view_width'],
                height=view['view_height'],
                img_width=img_w,
                img_height=img_h,
                near=near_clip,
                far=far_clip,
                scale=scale,
            )
            render_time = time.time() - start_time

            if features is None:
                print(f"  No features rendered, skipping")
                continue

            H, W, D = features.shape
            print(f"  Rendered: {H}x{W}x{D} features in {render_time:.1f}s")

            # Save RGB image
            rgb_path = views_dir / f"{view['name']}_rgb.png"
            cv2.imwrite(str(rgb_path), (rgb * 255).astype(np.uint8)[..., ::-1])

            # Save PCA visualization of features
            pca = PCA(n_components=3)
            feat_flat = features.reshape(-1, D)
            feat_3d = pca.fit_transform(feat_flat)
            feat_3d = (feat_3d - feat_3d.min()) / (feat_3d.max() - feat_3d.min() + 1e-6)
            pca_img = feat_3d.reshape(H, W, 3)
            pca_path = views_dir / f"{view['name']}_features.png"
            cv2.imwrite(str(pca_path), (pca_img * 255).astype(np.uint8)[..., ::-1])

            # Save depth map
            if depth is not None:
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                depth_path = views_dir / f"{view['name']}_depth.png"
                cv2.imwrite(str(depth_path), (depth_norm * 255).astype(np.uint8))

            # Project ALL point cloud points into this orthographic view
            px, py, proj_depth = self.project_points_ortho(points, view_params)

            # Check which points are within image bounds
            in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

            # Check which points are within the depth range
            in_depth = (proj_depth > near_clip) & (proj_depth < far_clip)

            valid = in_bounds & in_depth

            if valid.sum() == 0:
                print(f"  No valid points in this view")
                continue

            # Visibility check using rendered depth
            valid_px = px[valid]
            valid_py = py[valid]
            valid_proj_depth = proj_depth[valid]

            if depth is not None:
                rendered_depth_at_points = depth[valid_py, valid_px]
                # A point is visible when its depth matches the rendered
                # surface depth. The tolerance is a small fraction of the
                # building size, not of the (large) camera distance.
                depth_tolerance = np.full(
                    len(rendered_depth_at_points),
                    max(0.02 * half_diag, 1e-4),
                    dtype=np.float32,
                )
                depth_diff = np.abs(valid_proj_depth - rendered_depth_at_points)
                visible = depth_diff < depth_tolerance

                if visible.sum() < 100 and valid.sum() > 100:
                    if strict_depth:
                        # Falling back to all in-bounds points would give the
                        # far facade the features of the near facade.
                        print(f"  Only {visible.sum()} points pass the depth check, skipping view")
                        continue
                    print(f"  Depth check too strict ({visible.sum()} visible), using all {valid.sum()} in-bounds points")
                    visible = np.ones(valid.sum(), dtype=bool)
            else:
                # No depth map available, use all in-bounds points
                visible = np.ones(valid.sum(), dtype=bool)

            # Get indices of visible points in original array
            valid_indices = np.where(valid)[0]
            visible_indices = valid_indices[visible]
            visible_px = valid_px[visible]
            visible_py = valid_py[visible]

            # Look up features at projected pixel locations
            looked_up_features = features[visible_py, visible_px, :]  # N_visible x 256

            # Accumulate features
            feature_sum[visible_indices] += looked_up_features
            feature_count[visible_indices] += 1

            n_visible = int(visible.sum())
            print(f"  Visible points: {n_visible} ({100 * n_visible / N_points:.1f}%)")
            print(f"  Saved: {rgb_path.name}, {pca_path.name}")

            view_stats.append({
                'name': view['name'],
                'visible_points': n_visible,
                'visible_pct': round(100.0 * n_visible / N_points, 2),
                'img_width': int(img_w),
                'img_height': int(img_h),
                'view_width': float(view['view_width']),
                'view_height': float(view['view_height']),
                'render_seconds': round(float(render_time), 2),
                'plane_center': [float(v) for v in view['plane_center']],
                'look_dir': [float(v) for v in view['look_dir']],
            })

        ############################################################
        # STEP 4: Average features across views
        ############################################################
        print("\n" + "=" * 60)
        print("STEP 4: Averaging features across views")
        print("=" * 60)

        has_features = feature_count > 0
        print(f"  Points with features: {has_features.sum()} / {N_points} ({100 * has_features.sum() / N_points:.1f}%)")

        if has_features.sum() < 100:
            print("ERROR: Too few points got features!")
            print("  Check: are the view positions correct for your scene?")
            print("  Check: is the point cloud in the same coordinate system as GARField?")
            return

        # Average
        avg_features = np.zeros_like(feature_sum)
        avg_features[has_features] = feature_sum[has_features] / feature_count[has_features, np.newaxis]

        # For points without features, use nearest neighbor
        if (~has_features).sum() > 0:
            from sklearn.neighbors import NearestNeighbors
            print(f"  Filling {(~has_features).sum()} points without view coverage...")
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(points[has_features])
            _, fill_idx = nn.kneighbors(points[~has_features])
            avg_features[~has_features] = avg_features[has_features][fill_idx[:, 0]]

        print(f"  Feature stats: min={avg_features.min():.3f}, max={avg_features.max():.3f}")
        print(f"  Avg view count per point: {feature_count[has_features].mean():.1f}")

        # SAVE FEATURES IMMEDIATELY so we never have to re-render
        np.save(output_dir / "avg_features.npy", avg_features)
        np.save(output_dir / "points.npy", points)
        if colors is not None:
            np.save(output_dir / "colors.npy", colors)
        np.save(output_dir / "feature_count.npy", feature_count)

        # Per-view report, read by the control panel.
        side = sum(v['visible_points'] for v in view_stats
                   if v['name'].startswith('side'))
        top = sum(v['visible_points'] for v in view_stats
                  if not v['name'].startswith('side'))
        total_samples = side + top

        with open(output_dir / "view_stats.json", "w") as f:
            json.dump({
                'scale': float(scale),
                'n_side': int(n_side),
                'side_elevations': [float(e) for e in side_elevations],
                'n_top_ring': int(n_top_ring),
                'top_elevations': [float(e) for e in top_elevations],
                'cam_dist': float(cam_dist),
                'n_points': int(N_points),
                'points_with_features': int(has_features.sum()),
                'side_samples': int(side),
                'top_samples': int(top),
                'side_share_pct': round(100.0 * side / total_samples, 2)
                if total_samples else 0.0,
                'views': view_stats,
            }, f, indent=2)

        print(f"  Side views contributed {100.0 * side / total_samples:.1f}% "
              f"of all feature samples" if total_samples else "")
        print(f"  ✓ Saved features to {output_dir}/avg_features.npy")
        print(f"  ✓ To re-cluster without re-rendering, use --load-features flag")

        # DISABLED OLD CLUSTERING:         self._cluster_features(avg_features, points, colors, output_dir)

    def _cluster_features(self, avg_features, points, colors, output_dir):
        """Cluster the averaged features. Separated so it can be called independently."""
        output_dir = Path(output_dir)
        N_points = len(points)

        ############################################################
        # STEP 5: Cluster with GPU-accelerated HDBSCAN (cuml)
        ############################################################
        print("\n" + "=" * 60)
        print("STEP 5: Clustering (GPU HDBSCAN, full 256 dims)")
        print("=" * 60)

        # Parameter search using GPU-accelerated cuml HDBSCAN
        best_score = -1
        best_labels = None
        best_params = None

        for min_cluster_size in [50, 100, 200, 500]:
            for min_samples in [10, 20, 30]:
                try:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=0.1,
                        allow_single_cluster=False,
                    ).fit(avg_features)

                    labels = clusterer.labels_.copy()
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                    if n_clusters < 2:
                        print(f"  min_cluster={min_cluster_size}, min_samples={min_samples}: "
                              f"only {n_clusters} cluster(s), skipping")
                        continue

                    valid_labels = labels >= 0
                    if valid_labels.sum() < 100:
                        continue

                    score = silhouette_score(
                        avg_features[valid_labels],
                        labels[valid_labels],
                        sample_size=min(5000, int(valid_labels.sum()))
                    )

                    noise_pct = 100 * (labels == -1).sum() / len(labels)
                    print(f"  min_cluster={min_cluster_size}, min_samples={min_samples}: "
                          f"{n_clusters} clusters, silhouette={score:.3f}, noise={noise_pct:.1f}%")

                    if score > best_score:
                        best_score = score
                        best_labels = labels.copy()
                        best_params = (min_cluster_size, min_samples)

                except Exception as e:
                    print(f"  ERROR: {e}")

        if best_labels is None:
            print("No valid clustering found!")
            return

        labels = best_labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"\n  BEST: min_cluster={best_params[0]}, min_samples={best_params[1]}")
        print(f"  Clusters: {n_clusters}, Silhouette: {best_score:.3f}")

        ############################################################
        # STEP 6: Save results
        ############################################################
        print("\n" + "=" * 60)
        print("STEP 6: Saving results")
        print("=" * 60)

        # Generate random colors for clusters
        np.random.seed(42)
        cluster_colors = np.random.rand(n_clusters + 1, 3)
        cluster_colors[0] = [0.5, 0.5, 0.5]  # noise = gray

        # Assign colors to points
        point_colors = np.zeros((N_points, 3))
        for i in range(N_points):
            if labels[i] >= 0:
                point_colors[i] = cluster_colors[labels[i] % len(cluster_colors)]
            else:
                point_colors[i] = [0.3, 0.3, 0.3]

        # Save clustered point cloud
        pcd_clustered = o3d.geometry.PointCloud()
        pcd_clustered.points = o3d.utility.Vector3dVector(points)
        pcd_clustered.colors = o3d.utility.Vector3dVector(point_colors)

        clustered_path = output_dir / "clustered_pointcloud.ply"
        o3d.io.write_point_cloud(str(clustered_path), pcd_clustered)
        print(f"  Saved: {clustered_path}")

        # Save original colors version too
        if colors is not None:
            pcd_original = o3d.geometry.PointCloud()
            pcd_original.points = o3d.utility.Vector3dVector(points)
            pcd_original.colors = o3d.utility.Vector3dVector(colors)
            original_path = output_dir / "original_pointcloud.ply"
            o3d.io.write_point_cloud(str(original_path), pcd_original)

        # Save metadata
        metadata = {
            'n_points': int(N_points),
            'n_clusters': int(n_clusters),
            'silhouette_score': float(best_score),
            'best_params': {
                'min_cluster_size': int(best_params[0]),
                'min_samples': int(best_params[1]),
            },
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save labels
        np.save(output_dir / "cluster_labels.npy", labels)

        print(f"\n{'=' * 60}")
        print(f"PIPELINE COMPLETE")
        print(f"  Clusters: {n_clusters}")
        print(f"  Silhouette: {best_score:.3f}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="GARField Orthographic Feature Projection Pipeline"
    )

    parser.add_argument("--config", type=Path,
                        required=True,
                        help="Path to GARField config.yml")
    parser.add_argument("--pointcloud", type=Path, required=True,
                        help="Path to point cloud PLY file")
    parser.add_argument("--scale", type=float, default=0.1,
                        help="GARField grouping scale (0.1 = fine components)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ortho_projection"),
                        help="Output directory")
    parser.add_argument("--crop-center", type=float, nargs=3,
                        default=[0.02, -0.05, -0.15],
                        metavar=("X", "Y", "Z"),
                        help="Crop box center as X Y Z")
    parser.add_argument("--crop-scale", type=float, nargs=3,
                        default=[1.0, 0.91, 0.19],
                        metavar=("SX", "SY", "SZ"),
                        help="Crop box size as SX SY SZ")
    parser.add_argument("--resolution", type=int, default=1080,
                        help="Pixels on the long side of each rendered view")
    parser.add_argument("--n-side", type=int, default=4,
                        help="Number of horizontal (facade) views around the building")
    parser.add_argument("--side-elevations", type=float, nargs="+", default=[0.0],
                        help="Elevation angles in degrees for the side views (0 = horizontal)")
    parser.add_argument("--n-top-ring", type=int, default=4,
                        help="Number of steep roof views")
    parser.add_argument("--top-elevations", type=float, nargs="+", default=[80.0],
                        help="Elevation angles in degrees for the roof views")
    parser.add_argument("--azimuth-offset", type=float, default=0.0,
                        help="Rotate all azimuths, e.g. to align views with the real facades")
    parser.add_argument("--dist-factor", type=float, default=1.5,
                        help="Image-plane distance as a multiple of the crop box half-diagonal")
    parser.add_argument("--no-square-pixels", action="store_true",
                        help="Render every view at resolution x resolution instead of keeping square pixels")
    parser.add_argument("--allow-depth-fallback", action="store_true",
                        help="If the depth check rejects almost everything, use all in-bounds points anyway")
    parser.add_argument("--load-features", action="store_true",
                        help="Skip rendering, load saved features from output-dir and re-cluster")

    args = parser.parse_args()

    if args.load_features:
        # Load saved features and re-cluster (no rendering needed)
        print("Loading saved features (skipping rendering)...")
        avg_features = np.load(args.output_dir / "avg_features.npy")
        points = np.load(args.output_dir / "points.npy")
        colors_path = args.output_dir / "colors.npy"
        colors = np.load(colors_path) if colors_path.exists() else None
        print(f"  Loaded {len(points)} points with {avg_features.shape[1]}-dim features")

        # Create a dummy projector just for clustering
        projector = GarfieldOrthoProjector.__new__(GarfieldOrthoProjector)
        projector._cluster_features(avg_features, points, colors, args.output_dir)
    else:
        # Full pipeline: render + project + cluster
        projector = GarfieldOrthoProjector(
            config_path=args.config,
            data_path=None,
            crop_center=args.crop_center,
            crop_scale=args.crop_scale
        )

        projector.run_pipeline(
            pointcloud_path=args.pointcloud,
            scale=args.scale,
            output_dir=args.output_dir,
            img_resolution=args.resolution,
            n_side=args.n_side,
            side_elevations=tuple(args.side_elevations),
            n_top_ring=args.n_top_ring,
            top_elevations=tuple(args.top_elevations),
            azimuth_offset=args.azimuth_offset,
            dist_factor=args.dist_factor,
            square_pixels=not args.no_square_pixels,
            strict_depth=not args.allow_depth_fallback,
        )


if __name__ == "__main__":
    main()
