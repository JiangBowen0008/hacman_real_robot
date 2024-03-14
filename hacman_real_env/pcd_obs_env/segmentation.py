'''
Background is segmented by fitting the point cloud to a plane with two bins.
'''

import numpy as np
import os
import open3d as o3d
import pickle
from scipy.spatial.transform import Rotation
from copy import deepcopy

from hacman.utils.transformations import to_pose_mat

from hacman_real_env.pcd_obs_env.utils import display_inlier_outlier
from hacman_real_env.pcd_obs_env.pcd_obs_env import PCDObsEnv

seg_param_dir = os.path.join(os.path.dirname(__file__), 'segmentation_params')

class BackgroundGeometry():
    def __init__(
            self,
            param_path="background_params.pkl",
            bg_pcd_path="background.pcd",
            bin_tolerance=0.007,
            **params) -> None:
        # Load the background pcd
        bg_pcd_path = os.path.join(seg_param_dir, bg_pcd_path)
        if os.path.exists(bg_pcd_path):
            self.bg_pcd = o3d.io.read_point_cloud(bg_pcd_path)
            print(f":: Background point cloud loaded from {bg_pcd_path}")
        else:
            print(f"WARNING: Background point cloud not found.")

        # Try loading params from the path
        self.load_params(param_path)

        # Use the default params if no param pickle file is found
        if self.params is None:
            self.params = dict(
                scene_translation=np.zeros(3),
                scene_rotation=np.zeros(3), # Euler angles
                bin_distance=[0.01],
                # bin_half_size=np.array([0.4, 0.24, 0.06]),
                # bin_half_size=np.array([0.38, 0.28, 0.076]),      # Black bin
                bin_half_size=np.array([0.35, 0.249, 0.06]),        # Grey bin
                # bin_half_size=np.array([0.33, 0.22, 0.04]),       # Dark grey bin
                plane_tolerance=0.02,
                aabb_min_bound=np.zeros(3),                     # Axis aligned bounding box of the bins
                aabb_max_bound=np.zeros(3),
            )
        
        # Override with the custom params
        self.params.update(params)

        # Set the bin tolerance
        self.bin_tolerance = bin_tolerance
    
    def get_obs2real_transform(self):
        '''
        Get the transformation from the observation space to the real space.
        '''
        # Get the bin space to observation space
        bin2obs_translation = self.params['scene_translation']
        bin2obs_quat = Rotation.from_euler('xyz', self.params['scene_rotation']).as_quat()
        bin2obs_transform = to_pose_mat(bin2obs_translation, bin2obs_quat, input_wxyz=False)

        obs2bin_transform = np.linalg.inv(bin2obs_transform)
        return obs2bin_transform
    
    def get_real2obs_transform(self):
        '''
        Get the transformation from the real space to the observation space.
        '''
        # Get the bin space to observation space
        bin2obs_translation = self.params['scene_translation']
        bin2obs_quat = Rotation.from_euler('xyz', self.params['scene_rotation']).as_quat()
        bin2obs_transform = to_pose_mat(bin2obs_translation, bin2obs_quat, input_wxyz=False)

        return bin2obs_transform
    
    def transform2obs(self, pcd):
        '''
        Transform the point cloud from the real space to the observation space.
        '''
        pcd = deepcopy(pcd)
        real2obs_transform = self.get_real2obs_transform()
        pcd.transform(real2obs_transform)
        return pcd
    
    def process_pcd(self, pcd, replace_bg=False, debug=False):
        """
        Process the point cloud.
        3. Transform the entire pcd such that the center of the bin bottom is at the origin
        4. (if replace_bg) Replace the background points with the syns background points
        """
        assert len(self.bg_pcd.points) > 0, "Background point cloud is not loaded!"
        pcd = deepcopy(pcd)

        # Segment the object and background points
        obj_pcd, bg_pcd = self._segment_pcd(pcd, debug=debug)

        # Transform the entire pcd such that the center of the bin bottom is at the origin
        obj_pcd = self.transform2obs(obj_pcd)
        bg_pcd = self.transform2obs(bg_pcd)

        # Replace the background points with the syns background points
        if replace_bg:
            syns_bg_pcd = self.generate_pcd()
            bg_pcd = syns_bg_pcd     
        
        # Visualize the background points
        if debug:
            bg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            if not bg_pcd.has_colors():
                bg_pcd.paint_uniform_color([0, 0.651, 0.929])
            obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            if not obj_pcd.has_colors():
                obj_pcd.paint_uniform_color([1, 0.706, 0])
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([bg_pcd, obj_pcd, origin])
        
        # Combine the background and object points
        pcd = bg_pcd + obj_pcd
        bg_mask = np.arange(len(bg_pcd.points))
        
        return pcd, bg_mask
    
    def _segment_pcd(self, pcd, debug=False):
        '''
        Segment out background, object points.

        Preprocess the point cloud.
        1. Crop with the bin AABB
        2. Compare with the existing background pcd and label the background points
        3. Segment the object pcd by clustering and removing the outliers
        '''
        # Crop with the bin AABB
        min_bound, max_bound = self.params['aabb_min_bound'], self.params['aabb_max_bound']
        min_bound[2] = -self.bin_tolerance   # Hack the min bound to allow some tolerance
        max_bound[2] = 1
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd = pcd.crop(aabb)

        # Compare with the existing background pcd and label the background points
        dists = np.asarray(pcd.compute_point_cloud_distance(self.bg_pcd))
        bg_mask = np.where(dists <= self.bin_tolerance)[0]
        bg_pcd = pcd.select_by_index(bg_mask)
        non_bg_pcd = pcd.select_by_index(bg_mask, invert=True)

        # All object pcd should be above the bin bottom
        min_bound, max_bound = self.params['aabb_min_bound'], self.params['aabb_max_bound']
        min_bound[2] = self.bin_tolerance
        aabb.min_bound = min_bound
        non_bg_pcd = non_bg_pcd.crop(aabb)

        # Removing the outliers
        _, idx = non_bg_pcd.remove_radius_outlier(nb_points=20, radius=0.02)
        # _, idx = obj_pcd.remove_statistical_outlier(nb_neighbors=64, std_ratio=0.3)
        # display_inlier_outlier(obj_pcd, idx)
        non_bg_pcd = non_bg_pcd.select_by_index(idx)

        # Cluster the non_bg points
        cluster_idx = non_bg_pcd.cluster_dbscan(eps=0.03, min_points=100, print_progress=False)
        cluster_idx = np.asarray(cluster_idx)   # The result might contain both the objects and the gripper

        # Remove the cluster noise
        non_bg_idx = np.where(cluster_idx!= -1)[0]
        # display_inlier_outlier(obj_pcd, non_bg_idx)
        non_bg_pcd = non_bg_pcd.select_by_index(non_bg_idx)
        cluster_idx = cluster_idx[non_bg_idx]

        # The object pcd should be the cluster with the lowest z
        non_bg_min_z = np.asarray(non_bg_pcd.points)[:, 2].min()
        obj_bottom_mask = np.asarray(non_bg_pcd.points)[:, 2] <= non_bg_min_z + 0.02
        obj_idx_candidates, counts = np.unique(cluster_idx[obj_bottom_mask], return_counts=True)

        obj_idx = obj_idx_candidates[np.argmax(counts)]
        obj_idx = np.where(cluster_idx == obj_idx)[0]
        # display_inlier_outlier(non_bg_pcd, obj_idx)
        obj_pcd = non_bg_pcd.select_by_index(obj_idx)

        return obj_pcd, bg_pcd
    
    def estimate_params(self, pcd, debug=False):
        '''
        Estimate the background geometry parameters from the point cloud.
        '''
        # Preprocess the point cloud
        pcd_down = pcd.voxel_down_sample(voxel_size=0.002)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Segment the bin top surface
        pcd_z = np.asarray(pcd_down.points)[:, 2]
        bin_top_z = pcd_z.max()
        bin_top_mask = np.where(pcd_z >= bin_top_z - 0.025)[0]
        bin_top = pcd_down.select_by_index(bin_top_mask)

        # Find the bbox of the bin top surface and estimate the full box bbox
        aabb = bin_top.get_axis_aligned_bounding_box()
        min_bound, max_bound = aabb.get_min_bound(), aabb.get_max_bound()
        min_bound[2] = -0.05    # Hack the min bound to be below the table
        max_bound[2] = 0.4
        aabb.min_bound = min_bound
        aabb.max_bound = max_bound

        # Update the params
        self.params['aabb_min_bound'] = min_bound
        self.params['aabb_max_bound'] = max_bound

        # Crop out the rest of the points
        bins = pcd_down.crop(aabb)

        if debug:
            dists = np.asarray(pcd_down.compute_point_cloud_distance(bins))
            indices = np.where(dists > 0.00001)[0]
            rest_scene = pcd_down.select_by_index(indices)
            bin_top.paint_uniform_color([0, 0, 1])
            bins.paint_uniform_color([0, 1, 0])
            rest_scene.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([bin_top, bins, rest_scene, aabb])

        # Find geometric param of the arena plane: scene_translation, scene_rotation, bin_distance, bin_half_sizes
        # This is done by fitting the geometry model to the bin_bottom point cloud (hence bin_half_sizes do not matter)
        params = self.fit_bin_model(bins, debug=debug)
        self.params.update(params)
        
        return
    
    def fit_bin_model(self, bins, debug=False):
        '''
        Fit the scene model to the point cloud.
        '''
        # Segment the table plane
        plane_model, inliers = bins.segment_plane(distance_threshold=self.params['plane_tolerance'], ransac_n=3, num_iterations=1000)
        bin_bottom = bins.select_by_index(inliers)
        bin_walls = bins.select_by_index(inliers, invert=True)
        if debug:
            bin_bottom.paint_uniform_color([0, 0, 1])
            bin_walls.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([bin_bottom, bin_walls])
        
        # Find the center of the bin bottom
        bin_bottom_aabb = bin_bottom.get_axis_aligned_bounding_box()
        bin_bottom_center = bin_bottom_aabb.get_center()
        self.params['scene_translation'] = -bin_bottom_center
        
        # Define the optimization targets
        optim_targets = ['scene_translation', 'scene_rotation', 'bin_distance']
        
        # Define the objective function
        def compute_model_err(model_params, obs_pcd, visualize=False):
            params = self.params.copy()
            updated_params = self.unflatten_params(model_params, optim_targets)
            params.update(updated_params)

            # Translate and rotate the observed point cloud
            translated_pcd = deepcopy(obs_pcd)
            translated_pcd.translate(params['scene_translation'])
            rot_mat = Rotation.from_euler('xyz', params['scene_rotation']).as_matrix()
            translated_pcd.rotate(rot_mat)

            # Compare to the synthetic point cloud
            syns_pcd = self.generate_pcd(params)
            dists = np.asarray(translated_pcd.compute_point_cloud_distance(syns_pcd))

            if visualize:
                syns_pcd.paint_uniform_color([0, 0, 1])
                translated_pcd.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([syns_pcd, translated_pcd])

            # Scale the dists
            dists = ((dists * 100) ** 2).mean() 
            return dists
        
        # Optimize the model parameters
        from scipy.optimize import minimize
        init_params = self.flatten_params(self.params, optim_targets)
        err = compute_model_err(init_params, bins, visualize=True)
        print(f":: Performing optimization ::")
        print(f"Initial error: {err}")
        res = minimize(
            compute_model_err, init_params, args=(bins, False),
            tol=0.5, options={'maxiter': 1000}
        )

        print(f"Optimization success: {res.success}")
        print(f"Optimization message: {res.message}")
        print(f"Optimization error: {res.fun}")
        params = self.unflatten_params(res.x, optim_targets)
        from pprint import pprint
        print(f"Optimization result:")
        pprint(params)
        err = compute_model_err(res.x, bins, visualize=True)
    
        return params
    
    def flatten_params(self, params, targets):
        '''
        Flatten the parameters into a vector.
        '''
        flat_params = []
        for key in targets:
            flat_params.append(params[key])
        return np.concatenate(flat_params)

    def unflatten_params(self, flat_params, targets):
        '''
        Unflatten the parameters from a vector.
        '''
        params = {}
        idx = 0
        for key in targets:
            param_len = len(self.params[key])
            params[key] = flat_params[idx:idx+param_len]
            idx += param_len
        return params
    
    def generate_pcd(self, params=None):
        '''
        Generate a point cloud of the background geometry.
        '''
        if params is None:
            params = self.params
        
        def generate_bin_pcd(bin_sizes):
            '''
            Generate the point cloud of a single bin.
            '''
            bin = o3d.geometry.TriangleMesh.create_box(width=bin_sizes[0], height=bin_sizes[1], depth=bin_sizes[2])
            bin_pcd = bin.sample_points_uniformly(number_of_points=40000)

            # Remove the top
            bin_pcd = bin_pcd.select_by_index(np.where(np.asarray(bin_pcd.points)[:, 2] < bin_sizes[2] - 0.005)[0])

            bin_pcd = bin_pcd.voxel_down_sample(voxel_size=0.002)
            return bin_pcd    

        bin_sizes = np.array(params['bin_half_size']) * 2.
        bin_pcd = generate_bin_pcd(params['bin_half_size'])
        bin_pcds = [deepcopy(bin_pcd) for _ in range(2)]

        # Separate the two bins
        y_offset = params['bin_distance'][0] / 2
        x_offset = -bin_sizes[0] / 4
        bin_pcds[0].translate([x_offset, -y_offset - bin_sizes[1] / 2, 0])
        bin_pcds[1].translate([x_offset, y_offset, 0])

        # Combine
        pcd = o3d.geometry.PointCloud()
        for bin_pcd in bin_pcds:
            pcd += bin_pcd
        
        return pcd
    
    def save_params(self, path=None):
        '''
        Save the parameters to a pickle file.
        '''
        if path is None:
            curr_dir = os.path.dirname(__file__)
            path = os.path.join(curr_dir, 'segmentation_params', 'background_params.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, path=None):
        '''
        Load the parameters from a pickle file.
        '''
        if path is None:
            self.params = None
        else:
            path = os.path.join(seg_param_dir, path)
            with open(path, 'rb') as f:
                self.params = pickle.load(f)
                print(f":: Background params loaded from {path}")
    
    def get_scene_bounds(self):
        '''
        Get the scene bounds.
        '''
        min_bound = self.params['aabb_min_bound']
        max_bound = self.params['aabb_max_bound']
        return min_bound, max_bound

# Test the segmentation
if __name__ == "__main__":
    obs_env = PCDObsEnv(
        voxel_size=0.002,
    )
    bg = BackgroundGeometry()
    pcd = obs_env.get_pcd(
        return_numpy=False,
        color=False,
        )
    bg.process_pcd(
        pcd,
        replace_bg=False,
        debug=True)