# Add this at the very top of garfield_gaussian_pipeline.py
# BEFORE any other imports

import torch

# Monkey patch torch.load to always use weights_only=False for backward compatibility
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    """Patched torch.load that defaults to weights_only=False for compatibility"""
    if weights_only is None:
        weights_only = False
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=weights_only, **kwargs)

torch.load = _patched_torch_load
import numpy as np
import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Mapping, Any, Optional, List, Dict
from torchtyping import TensorType
from pathlib import Path
import trimesh
import viser
import viser.transforms as vtf
import open3d as o3d
import cv2
import time
import json
import torch
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.models.splatfacto import SplatfactoModel

from cuml.cluster.hdbscan import HDBSCAN
from nerfstudio.models.splatfacto import RGB2SH

import tqdm

from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.transform import Rotation as Rot

from garfield.garfield_datamanager import GarfieldDataManagerConfig, GarfieldDataManager
from garfield.garfield_model import GarfieldModel, GarfieldModelConfig
from garfield.garfield_pipeline import GarfieldPipelineConfig, GarfieldPipeline

def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))

def generate_random_colors(N=5000) -> torch.Tensor:
    """Generate random colors for visualization"""
    hs = np.random.uniform(0, 1, size=(N, 1))
    ss = np.random.uniform(0.6, 0.61, size=(N, 1))
    vs = np.random.uniform(0.84, 0.95, size=(N, 1))
    hsv = np.concatenate([hs, ss, vs], axis=-1)
    # convert to rgb
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8)[None, ...], cv2.COLOR_HSV2RGB)
    return torch.Tensor(rgb.squeeze() / 255.0)


@dataclass
class GarfieldGaussianPipelineConfig(VanillaPipelineConfig):
    """Gaussian Splatting, but also loading GARField grouping field from ckpt."""
    _target: Type = field(default_factory=lambda: GarfieldGaussianPipeline)
    garfield_ckpt: Optional[Path] = None  # Need to specify this
    # LEGO conversion parameters
    lego_voxel_size: float = 0.002  # Size of each voxel in meters (LEGO unit size)
    lego_min_brick_size: int = 1  # Minimum brick size (1x1)
    lego_max_brick_size: int = 4  # Maximum brick size (4x4)
    lego_enable_merging: bool = True  # Whether to merge voxels into larger bricks

class GarfieldGaussianPipeline(VanillaPipeline):
    """
    Trains a Gaussian Splatting model, but also loads a GARField grouping field from ckpt.
    This grouping field allows you to:
     - interactive click-based group selection (you can drag it around)
     - scene clustering, then group selection (also can drag it around)

    Note that the pipeline training must be stopped before you can interact with the scene!!
    """
    model: SplatfactoModel
    garfield_pipeline: List[GarfieldPipeline]  # To avoid importing Viewer* from nerf pipeline
    state_stack: List[Dict[str, TensorType]]  # To revert to previous state
    click_location: Optional[TensorType]  # For storing click location
    click_handle: Optional[viser.GlbHandle]  # For storing click handle
    crop_group_list: List[TensorType]  # For storing gaussian crops (based on click point)
    crop_transform_handle: Optional[viser.TransformControlsHandle]  # For storing scene transform handle -- drag!
    cluster_labels: Optional[TensorType]  # For storing cluster labels

    def __init__(
        self,
        config: GarfieldGaussianPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        print("Loading instance feature model...")
        assert config.garfield_ckpt is not None, "Need to specify garfield checkpoint"
        from nerfstudio.utils.eval_utils import eval_setup
        _, garfield_pipeline, _, _ = eval_setup(
            config.garfield_ckpt, test_mode="inference"
        )
        # Choose device for grouping model:
        # tinycudann requires model params to be on CUDA for forward to work.
        grouping_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            garfield_pipeline.model.eval()
            garfield_pipeline.model.to(grouping_device)
        except Exception:
            # some models may not support `.to` fully; ignore and still keep reference
            pass
        self.garfield_pipeline = [garfield_pipeline]
        # store device so other methods know where to put inputs for grouping
        self._grouping_device = grouping_device

        self.state_stack = []

        self.colormap = generate_random_colors()

        self.viewer_control = ViewerControl()

        self.a_interaction_method = ViewerDropdown(
            "Interaction Method",
            default_value="Interactive",
            options=["Interactive", "Clustering"],
            cb_hook=self._update_interaction_method
        )

        self.click_gaussian = ViewerButton(name="Click", cb_hook=self._click_gaussian)
        self.click_location = None
        self.click_handle = None

        self.crop_to_click = ViewerButton(name="Crop to Click", cb_hook=self._crop_to_click, disabled=True)
        self.crop_to_group_level = ViewerSlider(name="Group Level", min_value=0, max_value=29, step=1, default_value=0, cb_hook=self._update_crop_vis, disabled=True)
        self.crop_group_list = []

        self.move_current_crop = ViewerButton(name="Drag Current Crop", cb_hook=self._drag_current_crop, disabled=True)
        self.crop_transform_handle = None

        self.cluster_scene = ViewerButton(name="Cluster Scene", cb_hook=self._cluster_scene, disabled=False, visible=False)
        self.cluster_scene_scale = ViewerSlider(name="Cluster Scale", min_value=0.0, max_value=2.0, step=0.01, default_value=0.0, disabled=False, visible=False)
        self.cluster_scene_shuffle_colors = ViewerButton(name="Reshuffle Cluster Colors", cb_hook=self._reshuffle_cluster_colors, disabled=False, visible=False)
        self.cluster_labels = None
        # Cluster selection and multi-view rendering
        self.select_cluster_by_click = ViewerButton(
            name="Select Cluster by Click", 
            cb_hook=self._select_cluster_by_click, 
            disabled=True, 
            visible=False
        )
        self.isolate_selected_cluster = ViewerButton(
            name="Isolate Selected Cluster",
            cb_hook=self._isolate_selected_cluster,
            disabled=True,
            visible=False
        )
        self.render_isolated_cluster_views = ViewerButton(
            name="Generate Camera Path for Cluster",
            cb_hook=self._generate_camera_path_for_cluster,  # NEW
            disabled=True,
            visible=False
        )
        self.num_render_views = ViewerSlider(
            name="Number of Views",
            min_value=4,
            max_value=12,
            step=1,
            default_value=6,
            visible=False
        )
        self.convert_cluster_to_mesh = ViewerButton(
            name="Convert Cluster to Mesh",
            cb_hook=self._convert_cluster_to_mesh,
            disabled=True,
            visible=False
        )
        self.convert_cluster_to_lego = ViewerButton(
            name="Convert Cluster to LEGO Mesh",
            cb_hook=self._convert_cluster_to_lego,
            disabled=True,
            visible=False
        )
        self.selected_cluster_id = None
        self.isolated_cluster_indices = None
        self.reset_state = ViewerButton(name="Reset State", cb_hook=self._reset_state, disabled=True)

        self.z_export_options = ViewerCheckbox(name="Export Options", default_value=False, cb_hook=self._update_export_options)
        self.z_export_options_visible_gaussians = ViewerButton(
            name="Export Visible Gaussians",
            visible=False,
            cb_hook=self._export_visible_gaussians
            )
        self.z_export_options_camera_path_filename = ViewerText("Camera Path Filename", "", visible=False)
        self.z_export_options_camera_path_render = ViewerButton("Render Current Pipeline", cb_hook=self.render_from_path, visible=False)
        self.z_export_semantic_pointcloud = ViewerButton("Export Semantic Pointcloud", visible=False, cb_hook=self._export_semantic_pointcloud)

    def _update_interaction_method(self, dropdown: ViewerDropdown):
        """Update the UI based on the interaction method"""
        hide_in_interactive = (not (dropdown.value == "Interactive")) # i.e., hide if in interactive mode

        self.cluster_scene.set_hidden((not hide_in_interactive))
        self.cluster_scene_scale.set_hidden((not hide_in_interactive))
        self.cluster_scene_shuffle_colors.set_hidden((not hide_in_interactive))

        self.click_gaussian.set_hidden(hide_in_interactive)
        self.crop_to_click.set_hidden(hide_in_interactive)
        self.crop_to_group_level.set_hidden(hide_in_interactive)
        self.move_current_crop.set_hidden(hide_in_interactive)

    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_options_camera_path_filename.set_hidden(not checkbox.value)
        self.z_export_options_camera_path_render.set_hidden(not checkbox.value)
        self.z_export_options_visible_gaussians.set_hidden(not checkbox.value)
        self.z_export_semantic_pointcloud.set_hidden(not checkbox.value)

    def _reset_state(self, button: ViewerButton):
        """Revert to previous saved state"""
        assert len(self.state_stack) > 0, "No previous state to revert to"
        state_entry = self.state_stack.pop()
        # state_entry is now a filename (str) created by _queue_state
        if isinstance(state_entry, (str, Path)):
            data = np.load(state_entry, allow_pickle=False)
            for name in self.model.gauss_params.keys():
                # load numpy, convert to tensor and register as parameter
                arr = data[name]
                tensor = torch.nn.Parameter(torch.from_numpy(arr).to(self.device))
                self.model.gauss_params[name] = tensor
            # optionally delete the file to free disk if you want
            try:
                os.remove(state_entry)
            except Exception:
                pass
        else:
            # backward compat: older code path where state stored in memory dict
            prev_state = state_entry
            for name in self.model.gauss_params.keys():
                self.model.gauss_params[name] = prev_state[name]

        if len(self.state_stack) == 0:
            self.reset_state.set_disabled(True)


        self.cluster_labels = None
        self.cluster_scene.set_disabled(False)

    def _queue_state(self):
        """Save current state to disk-backed stack to avoid huge RAM usage."""
        import tempfile
        output_dir = Path(f"outputs/{self.datamanager.config.dataparser.data.name}/gauss_state")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a filename based on stack length (or timestamp)
        idx = len(self.state_stack)
        fname = output_dir / f"state_{idx:04d}.npz"

        # Save only CPU numpy arrays (no gradients)
        to_save = {}
        for k, v in self.model.gauss_params.items():
            # detach -> cpu -> numpy
            arr = v.detach().cpu().numpy()
            to_save[k] = arr

        # write compressed file
        np.savez_compressed(str(fname), **to_save)

        # push the filename so we don't hold the tensors in RAM
        self.state_stack.append(str(fname))
        self.reset_state.set_disabled(False)

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()

        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)

        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )

    def _crop_to_click(self, button: ViewerButton):
        """Crop to click location"""
        assert self.click_location is not None, "Need to specify click location"

        self._queue_state()  # Save current state
        curr_means = self.model.gauss_params['means'].detach()
        self.model.eval()

        # The only way to reset is to reset the state using the reset button.
        self.click_gaussian.set_disabled(True)  # Disable user from changing click
        self.crop_to_click.set_disabled(True)  # Disable user from changing click

        # Get the 3D location of the click
        location = self.click_location
        location = torch.tensor(location).view(1, 3).to(self.device)

        # The list of positions to query for garfield features. The first one is the click location.
        positions = torch.cat([location, curr_means])  # N x 3

        # Create a kdtree, to get the closest gaussian to the click-point.
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(curr_means.cpu().numpy()))
        kdtree = o3d.geometry.KDTreeFlann(points)
        _, inds, _ = kdtree.search_knn_vector_3d(location.view(3, -1).float().detach().cpu().numpy(), 10)

        # get the closest point to the sphere, using kdtree
        sphere_inds = inds
        scales = torch.ones((positions.shape[0], 1)).to(self.device)

        keep_list = []
        prev_group = None

        # Iterate over different scales, to get the a range of possible groupings.
        grouping_model = self.garfield_pipeline[0].model
        for s in tqdm.tqdm(torch.linspace(0, 1.5, 30)):
            # Move positions to grouping device in small batch (positions could be on self.device)
            _positions = positions.to(self._grouping_device)
            with torch.no_grad():
                instances = grouping_model.get_grouping_at_points(_positions, s)
            # ensure CPU tensor for numpy/o3d ops below
            if isinstance(instances, torch.Tensor):
                instances = instances.cpu()
            click_instance = instances[0]
            affinity = torch.norm(click_instance - instances, dim=1)[1:]

            # Filter out points that have affinity < 0.5 (i.e., not likely to be in the same group)
            keeps = torch.where(affinity < 0.5)[0].cpu()
            keep_points = points.select_by_index(keeps.tolist())  # indices of gaussians

            # Here, we desire the gaussian groups to be grouped tightly together spatially. 
            # We use DBSCAN to group the gaussians together, and choose the cluster that contains the click point.
            # Note that there may be spuriously high affinity between points that are spatially far apart,
            #  possibly due two different groups being considered together at an odd angle / far viewpoint.

            # If there are too many points, we downsample them first before DBSCAN.
            # Then, we assign the filtered points to the cluster of the nearest downsampled point.
            if len(keeps) > 5000:
                curr_point_min = keep_points.get_min_bound()
                curr_point_max = keep_points.get_max_bound()

                downsample_size = 0.01 * s
                _, _, curr_points_ds_ids = keep_points.voxel_down_sample_and_trace(
                    voxel_size=max(downsample_size, 0.0001),
                    min_bound=curr_point_min,
                    max_bound=curr_point_max,
                )
                curr_points_ds_ids = np.array([points[0] for points in curr_points_ds_ids])
                curr_points_ds = keep_points.select_by_index(curr_points_ds_ids)
                curr_points_ds_selected = np.zeros(len(keep_points.points), dtype=bool)
                curr_points_ds_selected[curr_points_ds_ids] = True

                _clusters = np.asarray(curr_points_ds.cluster_dbscan(eps=0.02, min_points=5))
                nn_model = NearestNeighbors(
                    n_neighbors=1, algorithm="auto", metric="euclidean"
                ).fit(np.asarray(curr_points_ds.points))

                _, indices = nn_model.kneighbors(np.asarray(keep_points.points)[~curr_points_ds_selected])

                clusters = np.zeros(len(keep_points.points), dtype=int)
                clusters[curr_points_ds_selected] = _clusters
                clusters[~curr_points_ds_selected] = _clusters[indices[:, 0]]

            else:
                clusters = np.asarray(keep_points.cluster_dbscan(eps=0.02, min_points=5))

            # Choose the cluster that contains the click point. If there is none, move to the next scale.
            cluster_inds = clusters[np.isin(keeps, sphere_inds)]
            cluster_inds = cluster_inds[cluster_inds != -1]
            if len(cluster_inds) == 0:
                continue
            cluster_ind = cluster_inds[0]

            keeps = keeps[np.where(clusters == cluster_ind)]

            if prev_group is None:
                prev_group = keeps
                keep_list.append(keeps)
                continue

            keeps = torch.cat([prev_group, keeps])
            keeps = torch.unique(keeps)

            # # Deduplication, based on the # of current points included in the previous group.
            # overlap = torch.isin(keeps, prev_group).sum()
            # if overlap < 0.8 * len(keeps):
            #     prev_group = keeps
            keep_list.append(keeps)

        if len(keep_list) == 0:
            print("No gaussians within crop, aborting")
            # The only way to reset is to reset the state using the reset button.
            self.click_gaussian.set_disabled(False)
            self.crop_to_click.set_disabled(False)
            return

        # Remove the click handle + visualization
        self.click_location = None
        self.click_handle.remove()
        self.click_handle = None
        
        self.crop_group_list = keep_list
        self.crop_to_group_level.set_disabled(False)
        self.crop_to_group_level.value = 29
        self.move_current_crop.set_disabled(False)

    def _update_crop_vis(self, number: ViewerSlider):
        """Update which click-based crop to visualize -- this requires that _crop_to_click has been called."""
        # If there is no click-based crop or saved state to crop from, do nothing
        if len(self.crop_group_list) == 0:
            return
        if len(self.state_stack) == 0:
            return
        
        # Clamp the number to be within the range of possible crops
        if number.value > len(self.crop_group_list) - 1:
            number.value = len(self.crop_group_list) - 1
            return
        elif number.value < 0:
            number.value = 0
            return

        keep_inds = self.crop_group_list[number.value]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name][keep_inds]

    def _drag_current_crop(self, button: ViewerButton):
        """Add a transform control to the current scene, and update the model accordingly."""
        self.crop_to_group_level.set_disabled(True)  # Disable user from changing crop
        self.move_current_crop.set_disabled(True)  # Disable user from creating another drag handle
        
        scene_centroid = self.model.gauss_params['means'].detach().mean(dim=0)
        self.crop_transform_handle = self.viewer_control.viser_server.add_transform_controls(
            name=f"/scene_transform",
            position=(VISER_NERFSTUDIO_SCALE_RATIO*scene_centroid).cpu().numpy(),
        )

        # Visualize the whole scene -- the points corresponding to the crop will be controlled by the transform handle.
        crop_inds = self.crop_group_list[self.crop_to_group_level.value]
        prev_state = self.state_stack[-1]
        for name in self.model.gauss_params.keys():
            self.model.gauss_params[name] = prev_state[name].clone()

        curr_means = self.model.gauss_params['means'].clone().detach()
        curr_rotmats = quat_to_rotmat(self.model.gauss_params['quats'][crop_inds].detach())

        @self.crop_transform_handle.on_update
        def _(_):
            handle_position = torch.tensor(self.crop_transform_handle.position).to(self.device)
            handle_position = handle_position / VISER_NERFSTUDIO_SCALE_RATIO
            handle_rotmat = quat_to_rotmat(torch.tensor(self.crop_transform_handle.wxyz).to(self.device).float())

            means = self.model.gauss_params['means'].detach()
            quats = self.model.gauss_params['quats'].detach()

            means[crop_inds] = handle_position.float() + torch.matmul(
                handle_rotmat, (curr_means[crop_inds] - curr_means[crop_inds].mean(dim=0)).T
            ).T
            quats[crop_inds] = torch.Tensor(Rot.from_matrix(
                torch.matmul(handle_rotmat.float(), curr_rotmats.float()).cpu().numpy()
            ).as_quat()).to(self.device)  # this is in xyzw format
            quats[crop_inds] = quats[crop_inds][:, [3, 0, 1, 2]]  # convert to wxyz format

            self.model.gauss_params['means'] = torch.nn.Parameter(means.float())
            self.model.gauss_params['quats'] = torch.nn.Parameter(quats.float())

            self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender

    def _reshuffle_cluster_colors(self, button: ViewerButton):
        """Reshuffle the cluster colors, if clusters defined using `_cluster_scene`."""
        if self.cluster_labels is None:
            return
        self.cluster_scene_shuffle_colors.set_disabled(True)  # Disable user from reshuffling colors
        self.colormap = generate_random_colors()
        colormap = self.colormap

        labels = self.cluster_labels

        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max().int().item() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])
        self.cluster_scene_shuffle_colors.set_disabled(False)

    def _cluster_scene(self, button: ViewerButton):
        """Cluster the scene, and assign gaussian colors based on the clusters.
        Also populates self.crop_group_list with the clusters group indices."""

        self._queue_state()  # Save current state
        self.cluster_scene.set_disabled(True)  # Disable user from clustering, while clustering

        scale = self.cluster_scene_scale.value
        grouping_model = self.garfield_pipeline[0].model
        
        positions_t = self.model.gauss_params['means'].detach()  # tensor on model device (likely GPU)
        # move positions to CPU for grouping model (which we kept on CPU earlier)
        positions_cpu = positions_t.cpu()
        N = positions_cpu.shape[0]
        batch_size = 8192  # tune down if still high memory; smaller = less peak RAM
        feats_list = []
        for s_i in range(0, N, batch_size):
            p_batch = positions_cpu[s_i : s_i + batch_size]  # CPU tensor
            # move the batch to the grouping device (GPU if available)
            _p_in = p_batch.to(self._grouping_device)
            with torch.no_grad():
                f_batch = grouping_model.get_grouping_at_points(_p_in, scale)
            # move outputs back to CPU / numpy
            if isinstance(f_batch, torch.Tensor):
                f_batch = f_batch.cpu().numpy()
            feats_list.append(f_batch)

        group_feats = np.concatenate(feats_list, axis=0)
        positions = positions_cpu.numpy()


        start = time.time()

        # Cluster the gaussians using HDBSCAN.
        # We will first cluster the downsampled gaussians, then 
        #  assign the full gaussians to the spatially closest downsampled gaussian.

        vec_o3d = o3d.utility.Vector3dVector(positions)
        pc_o3d = o3d.geometry.PointCloud(vec_o3d)
        min_bound = np.clip(pc_o3d.get_min_bound(), -1, 1)
        max_bound = np.clip(pc_o3d.get_max_bound(), -1, 1)
        # downsample size to be a percent of the bounding box extent
        downsample_size = 0.01 * scale
        pc, _, ids = pc_o3d.voxel_down_sample_and_trace(
            max(downsample_size, 0.0001), min_bound, max_bound
        )
        if len(ids) > 1e6:
            print(f"Too many points ({len(ids)}) to cluster... aborting.")
            print( "Consider using interactive select to reduce points before clustering.")
            print( "Are you sure you want to cluster? Press y to continue, else return.")
            # wait for input to continue, if yes then continue, else return
            if input() != "y":
                self.cluster_scene.set_disabled(False)
                return

        id_vec = np.array([points[0] for points in ids])  # indices of gaussians kept after downsampling
        group_feats_downsampled = group_feats[id_vec]
        positions_downsampled = np.array(pc.points)

        print(f"Clustering {group_feats_downsampled.shape[0]} gaussians... ", end="", flush=True)

        # Run cuml-based HDBSCAN
        clusterer = HDBSCAN(
            cluster_selection_epsilon=0.1,
            min_samples=30,
            min_cluster_size=30,
            allow_single_cluster=True,
        ).fit(group_feats_downsampled)

        non_clustered = np.ones(positions.shape[0], dtype=bool)
        non_clustered[id_vec] = False
        labels = clusterer.labels_.copy()
        clusterer.labels_ = -np.ones(positions.shape[0], dtype=np.int32)
        clusterer.labels_[id_vec] = labels

        # Assign the full gaussians to the spatially closest downsampled gaussian, with scipy NearestNeighbors.
        positions_np = positions[non_clustered]
        if positions_np.shape[0] > 0:  # i.e., if there were points removed during downsampling
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(positions_downsampled)
            _, indices = nn_model.kneighbors(positions_np)
            clusterer.labels_[non_clustered] = labels[indices[:, 0]]

        labels = clusterer.labels_
        print(f"done. Took {time.time()-start} seconds. Found {labels.max() + 1} clusters.")

        noise_mask = labels == -1
        if noise_mask.sum() != 0 and (labels>=0).sum() > 0:
            # if there is noise, but not all of it is noise, relabel the noise
            valid_mask = labels >=0
            valid_positions = positions[valid_mask]
            k = 1
            nn_model = NearestNeighbors(
                n_neighbors=k, algorithm="auto", metric="euclidean"
            ).fit(valid_positions)
            noise_positions = positions[noise_mask]
            _, indices = nn_model.kneighbors(noise_positions)
            # for now just pick the closest cluster
            noise_relabels = labels[valid_mask][indices[:, 0]]
            labels[noise_mask] = noise_relabels
            clusterer.labels_ = labels

        labels = clusterer.labels_

        colormap = self.colormap

        opacities = self.model.gauss_params['opacities'].detach()
        opacities[labels < 0] = -100  # hide unclustered gaussians
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        self.cluster_labels = torch.Tensor(labels)
        features_dc = self.model.gauss_params['features_dc'].detach()
        features_rest = self.model.gauss_params['features_rest'].detach()
        for c_id in range(0, labels.max() + 1):
            # set the colors of the gaussians accordingly using colormap from matplotlib
            cluster_mask = np.where(labels == c_id)
            features_dc[cluster_mask] = RGB2SH(colormap[c_id, :3].to(self.model.gauss_params['features_dc']))
            features_rest[cluster_mask] = 0

        self.model.gauss_params['features_dc'] = torch.nn.Parameter(self.model.gauss_params['features_dc'])
        self.model.gauss_params['features_rest'] = torch.nn.Parameter(self.model.gauss_params['features_rest'])
        # Enable cluster selection after clustering completes
        self.select_cluster_by_click.set_disabled(False)
        self.select_cluster_by_click.set_hidden(False)
        self.num_render_views.set_hidden(False)
        self.cluster_scene.set_disabled(False)
        self.viewer_control.viewer._trigger_rerender()  # trigger viewer rerender

    def _export_visible_gaussians(self, button: ViewerButton):
        """Export the visible gaussians to a .ply file"""
        # location to save
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"gaussians.ply"

        # Copied from exporter.py
        from collections import OrderedDict
        map_to_tensors = OrderedDict()
        model=self.model

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)
        from nerfstudio.scripts.exporter import ExportGaussianSplat
        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


    def render_from_path(self, button: ViewerButton):
        from nerfstudio.cameras.camera_paths import get_path_from_json
        import json
        from nerfstudio.scripts.render import _render_trajectory_video

        assert self.z_export_options_camera_path_filename.value != ""
        camera_path_filename = Path(self.z_export_options_camera_path_filename.value)
        
        with open(camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        camera_path = get_path_from_json(camera_path)
        self.model.eval()
        with torch.no_grad():
            _render_trajectory_video(
                self,
                camera_path,
                output_filename=Path('render.mp4'),
                rendered_output_names=['rgb'],
                rendered_resolution_scaling_factor=1.0 ,
                seconds=seconds,
                output_format="video",
            )
        self.model.train()

    def _export_semantic_pointcloud(self, button: ViewerButton):
        """Export semantic point cloud with building element labels"""
        print("Starting semantic point cloud export...")
    
        output_folder = "semantic_export"
        Path(output_folder).mkdir(exist_ok=True)
    
        # Get Gaussian data
        with torch.no_grad():
            positions = self.model.means.detach().cpu().numpy()
        
        # Get colors
        if hasattr(self.model, 'shs_0'):
            colors = self.model.shs_0.detach().cpu().numpy().squeeze() + 0.5
            colors = np.clip(colors, 0, 1)
        else:
            colors = np.ones((len(positions), 3)) * 0.5
    
        # Get semantic labels
        semantic_labels = self._get_semantic_labels(len(positions))
        semantic_colors = self._create_semantic_colors(semantic_labels)
    
        # Save point clouds
        self._save_semantic_pointclouds(positions, colors, semantic_colors, semantic_labels, output_folder)
    
        print(f"Export completed! Check folder: {output_folder}")

    def _get_semantic_labels(self, num_points):
        """Get semantic labels for each Gaussian"""
    
        # Use cluster labels if available
        if self.cluster_labels is not None:
            cluster_labels = self.cluster_labels.detach().cpu().numpy()
            if len(cluster_labels) == num_points:
                # Map clusters to building elements
                semantic_labels = cluster_labels % 9  # Map to 9 categories
                print(f"Using cluster labels: {len(np.unique(semantic_labels))} semantic classes")
                return semantic_labels
    
        # Fallback: position-based labeling
        print("Using position-based semantic labeling")
        positions = self.model.means.detach().cpu().numpy()
        heights = positions[:, 2]
    
        # Normalize heights
        min_h, max_h = heights.min(), heights.max()
        norm_heights = (heights - min_h) / (max_h - min_h + 1e-8)
    
        # Simple semantic assignment
        labels = np.zeros(len(positions), dtype=int)
        labels[norm_heights <= 0.2] = 2  # doors
        labels[(norm_heights > 0.2) & (norm_heights <= 0.6)] = 4  # walls
        labels[(norm_heights > 0.6) & (norm_heights <= 0.8)] = 1  # windows
        labels[norm_heights > 0.8] = 5  # roof
    
        return labels

    def _create_semantic_colors(self, semantic_labels):
        """Create colors for semantic visualization"""
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
    
        return np.array([color_map.get(label, [0.5, 0.5, 0.5]) for label in semantic_labels])

    def _save_semantic_pointclouds(self, positions, colors, semantic_colors, semantic_labels, output_folder):
        """Save point clouds in different formats"""
        output_path = Path(output_folder)
    
        # RGB point cloud
        pcd_rgb = o3d.geometry.PointCloud()
        pcd_rgb.points = o3d.utility.Vector3dVector(positions)
        pcd_rgb.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(output_path / "building_rgb.ply"), pcd_rgb)
    
        # Semantic point cloud
        pcd_semantic = o3d.geometry.PointCloud()
        pcd_semantic.points = o3d.utility.Vector3dVector(positions)
        pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
        o3d.io.write_point_cloud(str(output_path / "building_semantic.ply"), pcd_semantic)
    
        # Individual elements
        element_names = {0: 'background', 1: 'windows', 2: 'doors', 3: 'columns', 
                        4: 'walls', 5: 'roof', 6: 'balconies', 7: 'stairs', 8: 'facade'}
    
        elements_dir = output_path / "building_elements"
        elements_dir.mkdir(exist_ok=True)
    
        for label_id, element_name in element_names.items():
            mask = semantic_labels == label_id
            if not mask.any():
                continue
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions[mask])
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            o3d.io.write_point_cloud(str(elements_dir / f"{element_name}.ply"), pcd)
            print(f"Saved {element_name}: {mask.sum()} points")

    def _select_cluster_by_click(self, button: ViewerButton):
        """Start listening for click to select a specific cluster"""
        if self.cluster_labels is None:
            print("No clusters available. Run clustering first.")
            return
    
        def on_cluster_click(click: ViewerClick):
            self._identify_and_select_cluster(click)
            self.select_cluster_by_click.set_disabled(False)
            self.viewer_control.unregister_click_cb(on_cluster_click)
    
        self.select_cluster_by_click.set_disabled(True)
        print("Click on a cluster to select it...")
        self.viewer_control.register_click_cb(on_cluster_click)

    def _identify_and_select_cluster(self, click: ViewerClick):
        """Identify which cluster was clicked"""
        # Get 3D click position using ray casting
        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf
    
        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
    
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
    
        self.model.eval()
        outputs = self.model.get_outputs(cam.to(self.device))
        self.model.train()
    
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()
    
        click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)
    
        # Find nearest Gaussian to click location
        curr_means = self.model.gauss_params['means'].detach().cpu().numpy()
        distances = np.linalg.norm(curr_means - click_location, axis=1)
        nearest_idx = np.argmin(distances)
    
        # Get cluster ID at that Gaussian
        cluster_id = int(self.cluster_labels[nearest_idx].item())
    
        if cluster_id < 0:
            print("Clicked on unclustered/noise point. Please click on a colored cluster.")
            return
    
        self.selected_cluster_id = cluster_id
    
        # Count points in this cluster
        cluster_mask = (self.cluster_labels == cluster_id).cpu().numpy()
        num_points = cluster_mask.sum()
    
        print(f"Selected Cluster ID: {cluster_id} with {num_points} points")
    
        # Enable isolation button
        self.isolate_selected_cluster.set_disabled(False)
        self.isolate_selected_cluster.set_hidden(False)

    def _isolate_selected_cluster(self, button: ViewerButton):
        """Isolate the selected cluster - hide all other clusters"""
        if self.selected_cluster_id is None:
            print("No cluster selected. Use 'Select Cluster by Click' first.")
            return

        self._queue_state()  # Save current state before isolation

        # Get indices of selected cluster
        cluster_mask = (self.cluster_labels == self.selected_cluster_id).cpu().numpy()
        self.isolated_cluster_indices = np.where(cluster_mask)[0]

        # RESTORE ORIGINAL RGB COLORS for the selected cluster
        # Load from the saved state (before clustering changed colors)
        if len(self.state_stack) > 0:
            original_state = np.load(self.state_stack[0], allow_pickle=False)
            original_features_dc = torch.from_numpy(original_state['features_dc']).to(self.device)
            original_features_rest = torch.from_numpy(original_state['features_rest']).to(self.device)
        
            # CORRECT - create new tensors
            features_dc = self.model.gauss_params['features_dc'].detach().clone()
            features_rest = self.model.gauss_params['features_rest'].detach().clone()

            features_dc[cluster_mask] = original_features_dc[cluster_mask]
            features_rest[cluster_mask] = original_features_rest[cluster_mask]

            self.model.gauss_params['features_dc'] = torch.nn.Parameter(features_dc)
            self.model.gauss_params['features_rest'] = torch.nn.Parameter(features_rest)

        # Hide all non-selected clusters by setting their opacity to -100
        opacities = self.model.gauss_params['opacities'].detach().clone()
        opacities[~cluster_mask] = -100
        self.model.gauss_params['opacities'] = torch.nn.Parameter(opacities.float())

        print(f"Isolated cluster {self.selected_cluster_id} ({len(self.isolated_cluster_indices)} points)")
        print("Restored original RGB colors for rendering")

        # Enable render button
        self.render_isolated_cluster_views.set_disabled(False)
        self.render_isolated_cluster_views.set_hidden(False)
        self.isolate_selected_cluster.set_disabled(True)
        self.convert_cluster_to_lego.set_disabled(False)
        self.convert_cluster_to_lego.set_visible(True)

        self.viewer_control.viewer._trigger_rerender()

    def _generate_camera_path_for_cluster(self, button: ViewerButton):
        """Generate camera path and render views for isolated cluster"""
        if self.isolated_cluster_indices is None or self.selected_cluster_id is None:
            print("No isolated cluster.")
            return

        # Get cluster info
        cluster_positions = self.model.gauss_params['means'].detach().cpu().numpy()[self.isolated_cluster_indices]
        centroid = cluster_positions.mean(axis=0)
        bbox_size = np.linalg.norm(cluster_positions.max(axis=0) - cluster_positions.min(axis=0))

        num_views = int(self.num_render_views.value)
        radius = bbox_size * 1.8

        print(f"Generating and rendering {num_views} views for cluster {self.selected_cluster_id}")
        print(f"Centroid: {centroid}, Radius: {radius}, BBox: {bbox_size}")

        # Create camera path
        from nerfstudio.cameras.cameras import Cameras

        # Generate camera positions
        camera_to_worlds = []
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views
            x = centroid[0] + radius * np.cos(angle)
            y = centroid[1] + radius * np.sin(angle)
            z = centroid[2] + radius * 0.3  # Add elevation

            # Look at centroid
            position = np.array([x, y, z], dtype=np.float32)
            forward = centroid - position
            forward = forward / np.linalg.norm(forward)

            world_up = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(world_up, forward)
            right_norm = np.linalg.norm(right)
        
            if right_norm < 1e-6:
                right = np.array([1, 0, 0], dtype=np.float32)
            else:
                right = right / right_norm
        
            up = np.cross(forward, right)
            up = up / np.linalg.norm(up)

            # Build transform (with -forward for proper camera orientation)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward  # Negative for camera convention
            c2w[:3, 3] = position

            camera_to_worlds.append(torch.from_numpy(c2w[:3, :4]))

        # Create camera path object
        camera_path = Cameras(
            camera_to_worlds=torch.stack(camera_to_worlds),
            fx=self.datamanager.train_dataset.cameras.fx[0].repeat(num_views),
            fy=self.datamanager.train_dataset.cameras.fy[0].repeat(num_views),
            cx=self.datamanager.train_dataset.cameras.cx[0].repeat(num_views),
            cy=self.datamanager.train_dataset.cameras.cy[0].repeat(num_views),
            width=self.datamanager.train_dataset.cameras.width[0].repeat(num_views),
            height=self.datamanager.train_dataset.cameras.height[0].repeat(num_views),
        )

        # Output directory
        output_dir = Path(f"outputs/{self.datamanager.config.dataparser.data.name}/cluster_{self.selected_cluster_id}_views")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render each view
        self.model.eval()
        print("Rendering views...")
        with torch.no_grad():
            for i in range(num_views):
                # Get single camera
                camera = camera_path[i:i+1].to(self.device)

                # Render
                outputs = self.model.get_outputs_for_camera(camera)

                # Save RGB
                rgb = outputs["rgb"].cpu().numpy()
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(str(output_dir / f"view_{i:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                print(f"  Rendered view {i+1}/{num_views}")

        self.model.train()

        # Save metadata
        metadata = {
            "cluster_id": int(self.selected_cluster_id),
            "num_views": num_views,
            "centroid": centroid.tolist(),
            "radius": float(radius),
            "bbox_size": float(bbox_size)
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Rendered {num_views} views to: {output_dir}")
        print(f"✓ Ready for diffusion completion")
    def _export_isolated_cluster_gaussians(self, button: ViewerButton):
        """Export only the isolated cluster gaussians to PLY file"""
        if self.isolated_cluster_indices is None or self.selected_cluster_id is None:
            print("No isolated cluster to export.")
            return None
    
        from collections import OrderedDict
    
        # Output directory
        output_dir = Path(f"outputs/{self.datamanager.config.dataparser.data.name}/cluster_{self.selected_cluster_id}_export")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"cluster_{self.selected_cluster_id}.ply"
    
        map_to_tensors = OrderedDict()
    
        with torch.no_grad():
            # Get only the isolated cluster indices
            positions = self.model.means[self.isolated_cluster_indices].cpu().numpy()
            count = positions.shape[0]
        
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(count, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(count, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(count, dtype=np.float32)
        
            if self.model.config.sh_degree > 0:
                shs_0 = self.model.shs_0[self.isolated_cluster_indices].contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
            
                shs_rest = self.model.shs_rest[self.isolated_cluster_indices].transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((count, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(self.model.colors[self.isolated_cluster_indices].clone(), 0.0, 1.0).cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
        
                map_to_tensors["opacity"] = self.model.opacities[self.isolated_cluster_indices].cpu().numpy()
        
            scales = self.model.scales[self.isolated_cluster_indices].cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]
        
            quats = self.model.quats[self.isolated_cluster_indices].cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]
    
        # Check for NaN/Inf
        select = np.ones(count, dtype=bool)
        for k, t in map_to_tensors.items():
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
    
        if np.sum(select) < count:
            print(f"Removed {count - np.sum(select)} NaN/Inf gaussians")
            for k in map_to_tensors.keys():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)
    
        # Write PLY
        from nerfstudio.scripts.exporter import ExportGaussianSplat
        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)
    
        print(f"✓ Exported {count} gaussians to: {filename}")
        return filename

    def _convert_cluster_to_mesh(self, button: ViewerButton):
        """Convert isolated cluster gaussians to mesh using Poisson reconstruction"""
        if self.isolated_cluster_indices is None or self.selected_cluster_id is None:
            print("No isolated cluster to convert.")
            return
    
        # First export gaussians
        print("Exporting cluster gaussians...")
        ply_file = self._export_isolated_cluster_gaussians(button)
    
        if ply_file is None:
            return
    
        print("Converting to mesh using Poisson reconstruction...")
    
        # Load gaussian positions as point cloud
        positions = self.model.means[self.isolated_cluster_indices].detach().cpu().numpy()
        colors = None
    
        if hasattr(self.model, 'shs_0'):
            # Convert SH to RGB (simplified - just use DC component)
            colors = self.model.shs_0[self.isolated_cluster_indices].detach().cpu().numpy().squeeze() + 0.5
            colors = np.clip(colors, 0, 1)
    
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    
        # Estimate normals
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
        # Poisson reconstruction
        print("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )
    
        # Remove low density vertices (outliers)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
        # Save mesh
        output_dir = Path(f"outputs/{self.datamanager.config.dataparser.data.name}/cluster_{self.selected_cluster_id}_export")
        mesh_file = output_dir / f"cluster_{self.selected_cluster_id}_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_file), mesh)
    
        print(f"✓ Mesh saved to: {mesh_file}")
        print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.triangles)}")
    
        # Also save as OBJ for easier viewing
        obj_file = output_dir / f"cluster_{self.selected_cluster_id}_mesh.obj"
        o3d.io.write_triangle_mesh(str(obj_file), mesh)
        print(f"✓ Also saved as OBJ: {obj_file}")
    def _voxelize_cluster_gaussians(self, voxel_size=0.08):
        """
        Convert isolated cluster gaussians to voxel grid.
    
        Args:
            voxel_size: Size of each voxel cube in world units
        
        Returns:
            voxel_grid: 3D numpy array of occupied voxels (1=occupied, 0=empty)
            origin: The world-space origin of the voxel grid
            grid_shape: Shape of the voxel grid (x, y, z)
        """
        if self.isolated_cluster_indices is None:
            print("No isolated cluster available for voxelization")
            return None, None, None
    
        print(f"Voxelizing cluster with voxel size: {voxel_size}")
    
        # Get gaussian positions for isolated cluster
        positions = self.model.means[self.isolated_cluster_indices].detach().cpu().numpy()
    
        # Get colors if available
        colors = None
        if hasattr(self.model, 'shs_0'):
            colors = self.model.shs_0[self.isolated_cluster_indices].detach().cpu().numpy().squeeze() + 0.5
            colors = np.clip(colors, 0, 1)
    
        # Calculate bounding box
        min_bound = positions.min(axis=0)
        max_bound = positions.max(axis=0)
    
        # Add padding
        padding = voxel_size * 2
        min_bound -= padding
        max_bound += padding
    
        # Calculate grid dimensions
        grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    
        print(f"  Grid size: {grid_size} voxels")
        print(f"  World bounds: {min_bound} to {max_bound}")
    
        # Initialize voxel grid and color grid
        voxel_grid = np.zeros(grid_size, dtype=bool)
        color_grid = np.zeros((*grid_size, 3), dtype=np.float32)
        color_count = np.zeros(grid_size, dtype=int)
    
        # Fill voxel grid
        for i, pos in enumerate(positions):
            voxel_idx = ((pos - min_bound) / voxel_size).astype(int)
        
            # Clamp to grid bounds
            voxel_idx = np.clip(voxel_idx, 0, grid_size - 1)
        
            voxel_grid[tuple(voxel_idx)] = True
        
            if colors is not None:
                color_grid[tuple(voxel_idx)] += colors[i]
                color_count[tuple(voxel_idx)] += 1
    
        # Average colors for voxels with multiple gaussians
        mask = color_count > 0
        color_grid[mask] /= color_count[mask][..., None]
    
        occupied_voxels = np.sum(voxel_grid)
        print(f"  Occupied voxels: {occupied_voxels}")
    
        return voxel_grid, color_grid, min_bound, grid_size
    def _place_lego_bricks(self, voxel_grid, color_grid, min_brick_size=1, max_brick_size=4, enable_merging=True):
        """
        Place LEGO bricks using greedy algorithm to fill occupied voxels.
    
        Args:
            voxel_grid: 3D boolean array of occupied voxels
            color_grid: 3D color array (x, y, z, 3)
            min_brick_size: Minimum brick dimension
            max_brick_size: Maximum brick dimension
            enable_merging: Whether to merge voxels into larger bricks
        
        Returns:
            bricks: List of dicts with keys: 'position', 'size', 'color'
        """
        print("Placing LEGO bricks...")
    
        # Copy voxel grid so we can mark placed voxels
        remaining_voxels = voxel_grid.copy()
        bricks = []
    
        if not enable_merging:
            # Simple 1x1x1 bricks for each occupied voxel
            occupied_positions = np.argwhere(voxel_grid)
            for pos in occupied_positions:
                bricks.append({
                    'position': pos,
                    'size': np.array([1, 1, 1]),
                    'color': color_grid[tuple(pos)]
                })
            print(f"  Placed {len(bricks)} 1x1x1 bricks (no merging)")
            return bricks
    
        # Greedy brick placement - start with largest bricks
        # Define brick sizes to try (prioritize common LEGO brick sizes)
        brick_sizes = [
            # Large bricks first
            [4, 2, 1], [4, 1, 1], [2, 4, 1],
            [3, 2, 1], [3, 1, 1], [2, 3, 1],
            [2, 2, 1], [2, 1, 1],
            [1, 1, 1]  # Single studs last
        ]
    
        # Filter by max_brick_size
        brick_sizes = [s for s in brick_sizes if all(d <= max_brick_size for d in s)]
    
        total_voxels = np.sum(voxel_grid)
        placed_voxels = 0
    
        # Try to place bricks
        for brick_size in brick_sizes:
            brick_size = np.array(brick_size)
        
            # Scan through grid
            for x in range(voxel_grid.shape[0] - brick_size[0] + 1):
                for y in range(voxel_grid.shape[1] - brick_size[1] + 1):
                    for z in range(voxel_grid.shape[2] - brick_size[2] + 1):
                        # Check if brick fits
                        region = remaining_voxels[
                            x:x+brick_size[0],
                            y:y+brick_size[1],
                            z:z+brick_size[2]
                        ]
                    
                        if region.shape != tuple(brick_size):
                            continue
                    
                        # Check if all voxels in region are occupied
                        if np.all(region):
                            # Place brick
                            position = np.array([x, y, z])
                        
                            # Get average color for this brick
                            color_region = color_grid[
                                x:x+brick_size[0],
                                y:y+brick_size[1],
                                z:z+brick_size[2]
                            ]
                            avg_color = np.mean(color_region.reshape(-1, 3), axis=0)
                        
                            bricks.append({
                                'position': position,
                                'size': brick_size.copy(),
                                'color': avg_color
                            })
                        
                            # Mark voxels as placed
                            remaining_voxels[
                                x:x+brick_size[0],
                                y:y+brick_size[1],
                                z:z+brick_size[2]
                            ] = False
                        
                            placed_voxels += np.prod(brick_size)
    
        print(f"  Placed {len(bricks)} bricks")
        print(f"  Coverage: {placed_voxels}/{total_voxels} voxels ({100*placed_voxels/total_voxels:.1f}%)")
    
        return bricks
    def _generate_lego_mesh(self, bricks, voxel_size, origin):
        """
        Generate Open3D mesh from LEGO bricks.
    
        Args:
            bricks: List of brick dictionaries from _place_lego_bricks
            voxel_size: Size of each voxel in world units
            origin: World-space origin of the voxel grid
        
        Returns:
            mesh: Combined Open3D TriangleMesh
        """
        print("Generating LEGO mesh...")
    
        all_vertices = []
        all_triangles = []
        all_colors = []
        vertex_offset = 0
    
        for i, brick in enumerate(bricks):
            pos = brick['position']
            size = brick['size']
            color = brick['color']
        
            # Calculate world-space position and dimensions
            world_pos = origin + pos * voxel_size
            world_size = size * voxel_size
        
            # Create box mesh for this brick
            brick_mesh = o3d.geometry.TriangleMesh.create_box(
                width=world_size[0],
                height=world_size[1],
                depth=world_size[2]
            )
        
            # Translate to correct position
            brick_mesh.translate(world_pos)
        
            # Set color
            brick_mesh.paint_uniform_color(color)
        
            # Get vertices, triangles, and colors
            vertices = np.asarray(brick_mesh.vertices)
            triangles = np.asarray(brick_mesh.triangles)
            colors = np.asarray(brick_mesh.vertex_colors)
        
            # Add to combined mesh
            all_vertices.append(vertices)
            all_triangles.append(triangles + vertex_offset)
            all_colors.append(colors)
        
            vertex_offset += len(vertices)
        
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{len(bricks)} bricks...")
    
        # Combine all meshes
        combined_mesh = o3d.geometry.TriangleMesh()
        combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
        combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))
        combined_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    
        # Compute normals for better visualization
        combined_mesh.compute_vertex_normals()
    
        print(f"  Final mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.triangles)} triangles")
    
        return combined_mesh
    def _convert_cluster_to_lego(self, button: ViewerButton):
        """
        Convert isolated cluster gaussians to LEGO brick mesh.
        This is the main callback function for the "Convert Cluster to LEGO Mesh" button.
        """
        if self.isolated_cluster_indices is None or self.selected_cluster_id is None:
            print("No isolated cluster to convert.")
            return
    
        print("\n" + "="*60)
        print("STARTING LEGO CONVERSION")
        print("="*60)
    
        # Get configuration
        voxel_size = self.config.lego_voxel_size
        min_brick_size = self.config.lego_min_brick_size
        max_brick_size = self.config.lego_max_brick_size
        enable_merging = self.config.lego_enable_merging
    
        print(f"Configuration:")
        print(f"  Voxel size: {voxel_size}")
        print(f"  Brick size range: {min_brick_size}x{min_brick_size}x1 to {max_brick_size}x{max_brick_size}x1")
        print(f"  Merging enabled: {enable_merging}")
        print()
    
        # Step 1: Voxelize
        start_time = time.time()
        voxel_grid, color_grid, origin, grid_shape = self._voxelize_cluster_gaussians(voxel_size)
    
        if voxel_grid is None:
            return
    
        voxel_time = time.time() - start_time
        print(f"✓ Voxelization completed in {voxel_time:.2f}s\n")
    
        # Step 2: Place LEGO bricks
        start_time = time.time()
        bricks = self._place_lego_bricks(
            voxel_grid, 
            color_grid,
            min_brick_size=min_brick_size,
            max_brick_size=max_brick_size,
            enable_merging=enable_merging
        )
        brick_time = time.time() - start_time
        print(f"✓ Brick placement completed in {brick_time:.2f}s\n")
    
        # Step 3: Generate mesh
        start_time = time.time()
        lego_mesh = self._generate_lego_mesh(bricks, voxel_size, origin)
        mesh_time = time.time() - start_time
        print(f"✓ Mesh generation completed in {mesh_time:.2f}s\n")
    
        # Step 4: Save outputs
        output_dir = Path(f"outputs/{self.datamanager.config.dataparser.data.name}/cluster_{self.selected_cluster_id}_export")
        output_dir.mkdir(parents=True, exist_ok=True)
    
        # Save as PLY
        ply_file = output_dir / f"cluster_{self.selected_cluster_id}_lego_mesh.ply"
        o3d.io.write_triangle_mesh(str(ply_file), lego_mesh)
        print(f"✓ Saved LEGO mesh (PLY): {ply_file}")
    
        # Save as OBJ
        obj_file = output_dir / f"cluster_{self.selected_cluster_id}_lego_mesh.obj"
        o3d.io.write_triangle_mesh(str(obj_file), lego_mesh)
        print(f"✓ Saved LEGO mesh (OBJ): {obj_file}")
    
        # Save metadata
        metadata = {
            "cluster_id": int(self.selected_cluster_id),
            "num_bricks": len(bricks),
            "voxel_size": float(voxel_size),
            "grid_shape": grid_shape.tolist(),
            "origin": origin.tolist(),
            "vertices": len(lego_mesh.vertices),
            "triangles": len(lego_mesh.triangles),
            "processing_time": {
                "voxelization": float(voxel_time),
                "brick_placement": float(brick_time),
                "mesh_generation": float(mesh_time),
                "total": float(voxel_time + brick_time + mesh_time)
            }
        }
    
        metadata_file = output_dir / f"cluster_{self.selected_cluster_id}_lego_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")
    
        print("\n" + "="*60)
        print("LEGO CONVERSION COMPLETED")
        print("="*60)
        print(f"Total bricks: {len(bricks)}")
        print(f"Total vertices: {len(lego_mesh.vertices)}")
        print(f"Total triangles: {len(lego_mesh.triangles)}")
        print(f"Output directory: {output_dir}")
        print("="*60 + "\n")
