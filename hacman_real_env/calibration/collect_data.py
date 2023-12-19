"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import os, pickle
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType

from robot_controller import FrankaOSCController
from marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation


def move_robot_and_record_data(
        cam_id, 
        num_movements=3, 
        debug=False,
        initial_joint_positions=None):
    """
    Move the robot to random poses and record the necessary data.
    """
    
    # Initialize the robot
    robot = FrankaOSCController()

    # Initialize the camera
    k4a = PyK4A(device_id=cam_id)
    k4a.start()
    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
    dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)

    data = []
    for _ in tqdm(range(num_movements)):
        # Generate a random target delta pose
        random_delta_pos = [np.random.uniform(-0.08, 0.08, size=(3,))]
        random_delta_axis_angle = [np.random.uniform(-0.6, 0.6, size=(3,))]
        robot.reset(joint_positions=initial_joint_positions)
        robot.move_by(random_delta_pos, random_delta_axis_angle, num_steps=40, num_additional_steps=30)

        # Get current pose of the robot 
        gripper_pose = robot.eef_pose
        print(f"Gripper pos: {gripper_pose[:3, 3]}")

        # Capture IR frame from Kinect
        ir_frame = get_kinect_ir_frame(k4a)
        if ir_frame is not None:
            # Detect ArUco markers and get visualization
            corners, ids = detect_aruco_markers(ir_frame, debug=debug)

            # Estimate transformation if marker is detected
            if ids is not None and len(ids) > 0:
                transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
                if transform_matrix is not None:
                    data.append((
                        gripper_pose,       # gripper pose in base
                        transform_matrix    # tag pose in camera
                    ))
                
    
    print(f"Recorded {len(data)} data points.")
    # Save data
    os.makedirs("pcd_env/calibration/data", exist_ok=True)
    filepath = f"pcd_env/calibration/data/cam{cam_id}_data.pkl"
    with open(f"pcd_env/calibration/data/cam{cam_id}_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return filepath

def main():
    cam_id = 0
    # 0: left -     000059793712
    # 1: right -    000003493812
    # 2: front -    000180921812
    # 3: back -     000263392612
    initial_joint_positions = {
        0: [-0.68299696, 0.65603606, 0.07339937, -1.45441668, -0.06963243, 2.11292397, 1.73479704],
        1: [-0.74697406, 0.15221428, 0.47367525, -2.34519478, 0.14010332, 2.45179711, -1.2359939 ],
        2: [-0.57259571, 0.54167994, 0.07167276, -1.70355534, -0.01052658, 2.23024466, 0.28936683],
        3: [-0.57346419, 0.39241199, 0.04834748, -2.25460585, 0.61730919, 3.71824636, 1.5602955]
    }[cam_id]
    
    # Perform the movements and record data
    move_robot_and_record_data(
        cam_id=cam_id, num_movements=50, debug=False, 
        initial_joint_positions=initial_joint_positions)
    

if __name__ == "__main__":
    main()