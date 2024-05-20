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

from hacman_real_env.robot_controller import FrankaOSCController
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
    robot = FrankaOSCController(
        tip_offset=np.zeros(3),     # Set the default to 0 to disable accounting for the tip
    )

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
        else:
            print("\033[91m" + "No IR frame captured." + "\033[0m")
    
    print(f"Recorded {len(data)} data points.")
    # Save data
    os.makedirs("hacman_real_env/pcd_obs_env/calibration/data", exist_ok=True)
    filepath = f"hacman_real_env/pcd_obs_env/calibration/data/cam{cam_id}_data.pkl"
    with open(f"hacman_real_env/pcd_obs_env/calibration/data/cam{cam_id}_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return filepath

def main():
    cam_id = 3
    # 0: right -     000880595012
    # 2: left -     000059793721
    # 1: front -    000180921812
    # 3: back -     000263392612
    initial_joint_positions = {
        0: [-0.86812917, 0.36391594, 0.25352557, -1.92162717, 0.12602475, 2.1308299, -1.50102163],
        1: [-0.83424677, 0.42084166, 0.2774182, -1.97982254, -0.1749291, 2.40231471, 0.27310384],
        2: [-0.81592058, 0.39429853, 0.29050235, -1.88333403, -0.17686262, 2.28619198, 1.98916667],
        3: [-0.85456277, 0.36942704, 0.38232294, -1.88742087, -0.45677587, 2.19400042, -2.88310376]
    }[cam_id]
    
    # Perform the movements and record data
    move_robot_and_record_data(
        cam_id=cam_id, num_movements=50, debug=False, 
        initial_joint_positions=initial_joint_positions)
    

if __name__ == "__main__":
    main()