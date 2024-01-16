import numpy as np
import open3d as o3d
import os
import pickle
import gym

from pcd_obs_env import PCDObsEnv
from segmentation import BackgroundGeometry
from object_registration import ObjectRegistration

class HACManObsEnv():
    def __init__(
            self, 
            object_name) -> None:
        super().__init__()
        self.object_name = object_name
        self.pcd_obs = PCDObsEnv()
        self.bg = BackgroundGeometry()
        self.obj_reg = ObjectRegistration(object_name)
    
    def resample_goal(self):
        self.obj_reg.resample_goal()
    
    def get_obs(self, return_numpy=True, visualize=False):
        # Get the pcd
        pcd = self.pcd_obs.get_pcd(return_numpy=False)
        
        # Process the pcd
        pcd, bg_mask = self.bg.process_pcd(
                            pcd,
                            replace_bg=True,
                            debug=False)
        bg_pcd = pcd.select_by_index(bg_mask)
        obj_pcd = pcd.select_by_index(bg_mask, invert=True)
        
        # Register the object pcd
        goal_pcd = self.obj_reg.get_transformed_goal_pcd(obj_pcd)
        
        if visualize:
            obj_pcd.paint_uniform_color([1, 0.706, 0])
            goal_pcd.paint_uniform_color([0.651, 0.929, 0])
            bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            gt_goal_pcd = self.obj_reg.get_goal_pcd()
            gt_goal_pcd.paint_uniform_color([0.2, 0.3, 0.2])
            o3d.visualization.draw_geometries([obj_pcd, bg_pcd, goal_pcd, gt_goal_pcd])

        # Calculate obj pcd flow
        obj_pcd_np = np.asarray(obj_pcd.points)
        goal_pcd_np = np.asarray(goal_pcd.points)
        flow = goal_pcd_np - obj_pcd_np
        obj_pcd_np = np.concatenate([obj_pcd_np, flow], axis=-1)

        # Calculate background pcd flow
        bg_pcd_np = np.asarray(bg_pcd.points)
        bg_flow = np.zeros_like(bg_pcd_np)
        bg_pcd_np = np.concatenate([bg_pcd_np, bg_flow], axis=-1)

        # Add segmentation mask
        obj_mask = np.ones(obj_pcd_np.shape[0], dtype=np.float32)
        bg_mask = np.zeros(bg_pcd_np.shape[0], dtype=np.float32)
        obj_pcd_np = np.concatenate([obj_pcd_np, obj_mask[:, None]], axis=-1)
        bg_pcd_np = np.concatenate([bg_pcd_np, bg_mask[:, None]], axis=-1)
        
        # Return the observation
        pcd_np = np.concatenate([obj_pcd_np, bg_pcd_np], axis=0)
        if not return_numpy:
            return obj_pcd, goal_pcd, bg_pcd
        else:
            return pcd_np


if __name__ == '__main__':
    env = HACManObsEnv('blue_box')
    for _ in range(3):
        env.resample_goal()
        for _ in range(10):
            env.get_obs(visualize=True)