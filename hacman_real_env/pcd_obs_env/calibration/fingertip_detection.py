import cv2
import os, time
import numpy as np
from copy import deepcopy
from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType
import matplotlib.pyplot as plt

import open3d as o3d
from hacman_real_env.pcd_obs_env.utils import *

def get_kinect_depth_pcd(device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    # Capture an IR frame
    for _ in range(20):
        # try:
        device.get_capture()
        capture = device.get_capture()
        if capture is not None:
            pcd = capture.depth_point_cloud
            colors = capture.transformed_color
            pcd = pcd.reshape(-1, 3)
            pcd = pcd.astype(np.float32) / 1000.0
            colors = colors.reshape(-1, 4)

            # Clip points 1m away
            distances = np.linalg.norm(pcd, axis=1)
            mask = distances < 1.0
            pcd, colors = pcd[mask], colors[mask]
            
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
            # cv2.imshow('IR', ir_frame)
            if visualize:
                o3d.visualization.draw_geometries([pcd_o3d])
            return pcd_o3d
    #     except:
    #         time.sleep(0.1)
    #         print("Failed to capture PCD.")
    # else:
    #     print("Failed to capture PCD after 20 attempts.")
    #     return None

def detect_fingertip(pcd, init_transformation, debug=False):
    pcd = deepcopy(pcd)
    pcd.transform(init_transformation)

    # Remove outliers
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    if debug:
        display_inlier_outlier(pcd, idx)
    pcd = pcd.select_by_index(idx)
    
    # Clip the pcd to remove the table
    # The gripper is above 2cm of the table
    table_x_boundary = [0.1, 0.85]
    table_y_boundary = [-0.85, 0.25]
    table_z_boundary = [0.04, 0.5]
    aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([table_x_boundary[0], table_y_boundary[0], table_z_boundary[0]]),
            max_bound=np.array([table_x_boundary[1], table_y_boundary[1], table_z_boundary[1]])
        )
    gripper_pcd = pcd.crop(aabb)

    # Click on the visualizer to select the gripper tip
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(gripper_pcd)
    vis.run()
    vis.destroy_window()
    picked_points = vis.get_picked_points()
    
    if len(picked_points) == 0:
        print("No point selected!")
        return None
    else:
        print(f"Selected {len(picked_points)} points.")
        gripper_pcd.transform(np.linalg.inv(init_transformation))
        fingertip_points = np.asarray(gripper_pcd.points)[picked_points]
        fingertip_pos = np.asarray(fingertip_points).mean(axis=0)
    
        if debug:
            fingertip_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=fingertip_pos)
            o3d.visualization.draw_geometries([gripper_pcd, fingertip_marker])
    # The fingertip should be at the bottom of the gripper
    # gripper_points = np.asarray(gripper_pcd.points)
    # gripper_z = gripper_points[:, 2]
    # fingertip_z = gripper_z.min()
    # fingertip_mask = np.where(gripper_z <= (fingertip_z + 0.005))[0]
    # fingertip_pcd = gripper_pcd.select_by_index(fingertip_mask)
    
    # Calculate the fingertip position
    # fingertip_pos = np.asarray(fingertip_pcd.points).mean(axis=0)
    
    # Transform things back

    return fingertip_pos

def main():
    cam_id = 0

    # Load the initial transformation (calibration result)
    curr_dir = os.path.dirname(__file__)
    camera_param_dir = os.path.join(curr_dir, 'calibration_results')
    camera_params_file = {
        0: 'cam0_calibration.npz',
        1: 'cam1_calibration.npz',
        2: 'cam2_calibration.npz',
        3: 'cam3_calibration.npz',
    }[cam_id]
    
    content = np.load(os.path.join(camera_param_dir, camera_params_file))
    camera_extrinsic = content['T']

    # Initialize the camera
    k4a = PyK4A(device_id=cam_id)
    k4a.start()

    # Capture a single frame (implement error checking in real code)
    for _ in range(100):
        pcd = get_kinect_depth_pcd(k4a, visualize=False)
        detect_fingertip(pcd, camera_extrinsic, debug=True)


if __name__ == '__main__':
    main()


    