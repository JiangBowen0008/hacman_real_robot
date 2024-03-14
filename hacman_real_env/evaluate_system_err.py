import os
import pickle
import numpy as np
import open3d as o3d
from copy import deepcopy
from tqdm import tqdm
import itertools

from scipy.spatial.transform import Rotation

from hacman_real_env.utils import *
from hacman_real_env.real_env import RealEnv
from hacman_real_env.pcd_obs_env.hacman_obs_env import HACManObsEnv

from hacman_real_env.pcd_obs_env.calibration.marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation
from hacman_real_env.pcd_obs_env.calibration.solve_calibration import estimate_tag_pose

def visualize():
    env = RealEnv(
        object_name="white_box",
        background_pcd_size=5000,
        object_pcd_size=2000,
        voxel_downsample_size=0.001,
        obs_args=dict(voxel_size=0.001,),
        robot_args={"controller_type": "OSC_POSE"}
    )
    pos_ubound = [0.1, 0.1, 0.15]
    pos_lbound = [-0.1, -0.1, 0.1]
    rot_bounds = 0.6
    
    for _ in range(10):
        # Move the robot to a random pose
        env.robot.reset()
        random_delta_axis_angle = [np.random.uniform(-rot_bounds, rot_bounds, size=(3,))]
        random_pos = [np.random.uniform(pos_lbound, pos_ubound, size=(3,))]
        env.robot.move_to(random_pos, target_delta_axis_angle=random_delta_axis_angle)

        # obs pcd
        pcd = env.get_pcd(
            color=True, return_numpy=False,
            clip_table=False)
        pcd = env.bg.transform2obs(pcd)

        # gripper eef base location
        eef_pose = env.robot.eef_pose
        eef_base_pose = env.robot.eef_base_pose
        print("eef_pose", np.round(eef_pose, 2))
        eef_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eef_coord.transform(eef_pose)
        eef_base_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eef_base_coord.transform(eef_base_pose)

        # Estimate ArUco marker pose from forward kinematics
        hand_pose, tag_pose = estimate_tag_pose(eef_base_pose)
        aruco_coord_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        aruco_coord_robot.transform(tag_pose)

        # Calculate ArUco marker pose from main camera
        aruco_coord_cam = get_aruco_coord(env)

        # Visualize
        o3d.visualization.draw_geometries([
            pcd, eef_coord, eef_base_coord, 
            aruco_coord_cam, aruco_coord_robot
            ])

def get_aruco_coord(env: RealEnv, cam_id=1):
    # Obtain the main camera IR image
    camera = env.k4as[cam_id]
    ir_frame = get_kinect_ir_frame(camera)
    corners, ids = detect_aruco_markers(ir_frame, debug=False)
    if len(corners) == 0:
        print("No ArUco markers detected.")
        return None
    else:
        # Estimate the transformation matrix
        from pyk4a.calibration import CalibrationType
        camera_matrix = camera.calibration.get_camera_matrix(CalibrationType.DEPTH)
        dist_coeffs = camera.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
        pose_in_cam = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
    
        # Convert the pose to the world frame
        aruco_coord_in_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        aruco_coord_in_cam.transform(pose_in_cam)
        aruco_coord = env.transform_raw_pcd(aruco_coord_in_cam, cam_id=cam_id)

        # Convert the pose to the observation frame
        aruco_coord = env.bg.transform2obs(aruco_coord_in_cam)

        return aruco_coord


if __name__ == "__main__":
    visualize()
    
    