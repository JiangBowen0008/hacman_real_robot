import open3d as o3d
import numpy as np
import os
import pickle

from hacman_real_env.pcd_obs_env.pcd_obs_env import PCDObsEnv
from hacman_real_env.pcd_obs_env.segmentation import BackgroundGeometry

seg_param_dir = os.path.join(os.path.dirname(__file__), 'segmentation_params')

def record_background_points():
    '''
    Record the background point cloud.
    '''
    # Load the environment
    obs_env = PCDObsEnv()

    pcd = o3d.geometry.PointCloud()

    # Get the pcd multiple times
    for _ in range(5):
        pcd += obs_env.get_pcd(return_numpy=False)
    
    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    
    # Remove the gripper (z > 0.12)
    pcd_z = np.asarray(pcd.points)[:, 2]
    gripper_mask = np.where(pcd_z > 0.12)[0]
    pcd = pcd.select_by_index(gripper_mask, invert=True)
    
    o3d.visualization.draw_geometries([pcd])
    
    # Save the pcd
    pcd_path = os.path.join(seg_param_dir, 'background.pcd')
    o3d.io.write_point_cloud(pcd_path, pcd)

def parse_background_geometry(pcd, debug=False):
    background = BackgroundGeometry(
        param_path=None,
    )
    background.estimate_params(pcd, debug=debug)
    background.save_params()

if __name__ == '__main__':
    record_background_points()

    # Load the pcd
    pcd_path = os.path.join(seg_param_dir, 'background.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Parse the background geometry
    parse_background_geometry(pcd, debug=True)