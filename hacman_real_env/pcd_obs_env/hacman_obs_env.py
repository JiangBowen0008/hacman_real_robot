"""
pcd_obs_env with:
1. object/background segmentation
2. object registration
3. goal sampling
4. reward calculation
"""

import numpy as np
import open3d as o3d
import os
import pickle
import gym

from hacman_real_env.pcd_obs_env.utils import *
from hacman_real_env.pcd_obs_env.pcd_obs_env import PCDObsEnv
from hacman_real_env.pcd_obs_env.segmentation import BackgroundGeometry
from hacman_real_env.pcd_obs_env.object_registration import ObjectRegistration

class HACManObsEnv(PCDObsEnv):
    def __init__(
            self, 
            object_name,
            n_objects=1,
            voxel_downsample_size=0.01,
            registration=True,
            allow_manual_registration=False,
            allow_full_pcd=False,
            symmetric_object=False,
            **args) -> None:
        super().__init__(**args)
        self.object_name = object_name
        self.n_objects = n_objects
        self.voxel_downsample_size = voxel_downsample_size
        self.registration = registration
        self.bg = BackgroundGeometry()
        self.obj_reg = ObjectRegistration(
            object_name, 
            allow_manual_registration=allow_manual_registration,
            allow_full_pcd=allow_full_pcd,
            symmetric_object=symmetric_object)
    
    def resample_goal(self):
        self.obj_reg.resample_goal()
    
    def get_obs(self, 
                color=True,
                clip_gripper_pcd=None,
                visualize=False):
        # Get the pcd
        pcd = self.get_pcd(return_numpy=False, color=color)

        # Clip the pcd
        if clip_gripper_pcd is not None:
            pcd = self._clip_pcd_z(pcd, clip_gripper_pcd)
        
        # Process the pcd
        pcd, bg_mask = self.bg.process_pcd(
                            pcd,
                            replace_bg=True,
                            n_objects=self.n_objects,
                            debug=False)
        bg_pcd = pcd.select_by_index(bg_mask)
        obj_pcd = pcd.select_by_index(bg_mask, invert=True)

        # Estimate the normals
        if not bg_pcd.has_normals():
            bg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if not obj_pcd.has_normals():
            obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # o3d.visualization.draw_geometries([obj_pcd, bg_pcd], point_show_normal=True)
        
        # Register the object pcd
        if self.registration:
            goal_pcd, goal_pose = self.obj_reg.get_transformed_goal_pcd(obj_pcd, debug=False)
            obj_pose = np.eye(4)    # Consider the object pose to be identity
        else:
            goal_pcd = self.obj_reg.get_goal_pcd()
            goal_pose = np.eye(4)
            obj_pose = np.eye(4)

        # Voxel downsample the point cloud
        obj_pcd = obj_pcd.voxel_down_sample(voxel_size=self.voxel_downsample_size)
        bg_pcd = bg_pcd.voxel_down_sample(voxel_size=self.voxel_downsample_size * 2)
        
        if visualize:
            obj_pcd.paint_uniform_color([1, 0.706, 0])
            goal_pcd.paint_uniform_color([0.651, 0.929, 0])
            bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            gt_goal_pcd = self.obj_reg.get_goal_pcd()
            gt_goal_pcd.paint_uniform_color([0.2, 0.3, 0.2])
            o3d.visualization.draw_geometries([obj_pcd, bg_pcd, goal_pcd, gt_goal_pcd])
        
        return {
            'object_pose': obj_pose,
            'goal_pose': goal_pose,
            'object_pcd_o3d': obj_pcd,
            'background_pcd_o3d': bg_pcd,
        }

    def _clip_pcd_z(self, pcd, z_min, debug=False):
        pcd_z = np.asarray(pcd.points)[:, 2]
        clip_mask = np.where(pcd_z > z_min)[0]
        if debug:
            display_inlier_outlier(pcd, clip_mask)
        pcd = pcd.select_by_index(clip_mask, invert=True)

        return pcd
    
    def get_obs2real_transform(self):
        return self.bg.get_obs2real_transform()
    
    def get_real2obs_transform(self):
        return self.bg.get_real2obs_transform()
    
    def transform2obs(self, points):
        return self.bg.transform2obs(points)

    def get_scene_bounds(self, offset=0.0):
        min_bound, max_bound = self.bg.get_scene_bounds()
        # Transform to the obs frame
        real2obs = self.get_real2obs_transform()
        min_bound = transform_point(min_bound, real2obs)
        max_bound = transform_point(max_bound, real2obs)
        min_bound += offset
        max_bound -= offset
        return min_bound, max_bound
        
if __name__ == '__main__':
    env = HACManObsEnv('white_box')
    for _ in range(3):
        env.resample_goal()
        for _ in range(10):
            env.get_obs(visualize=True)