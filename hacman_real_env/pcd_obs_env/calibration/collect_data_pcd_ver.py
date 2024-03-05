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
from fingertip_detection import get_kinect_depth_pcd, detect_fingertip


def move_robot_and_record_data(
        cam_id, 
        num_movements=3, 
        debug=False,
        initial_yaw=None):
    """
    Move the robot to random poses and record the necessary data.
    """
    # Initialize the robot
    robot = FrankaOSCController()

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

    data = []
    for _ in tqdm(range(num_movements)):
        # Generate a random target delta pose
        lower_bound = np.array([0.4, -0.5, 0.08])
        upper_bound = np.array([0.7, -0.2, 0.12])
        random_pos = np.random.uniform(lower_bound, upper_bound)
        random_yaw = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi) + initial_yaw
        target_quat = Rotation.from_euler('xyz', [np.pi, 0, random_yaw]).as_quat()
        robot.reset()
        robot.move_to(
            random_pos, 
            target_quat=target_quat,
            grasp=True,
            num_steps=50, num_additional_steps=20)

        # Get current pose of the robot 
        gripper_pose = robot.eef_pose
        gripper_pos = gripper_pose[:3, 3]
        print(f"Gripper pos: {gripper_pos}")

        # Capture depth pcd from kinect
        pcd = get_kinect_depth_pcd(k4a, visualize=False)
        if pcd is not None:
            # Detect the fingertip pos
            fingertip_pos = detect_fingertip(pcd, camera_extrinsic, debug=debug)

            if fingertip_pos is not None:
                data.append((
                    gripper_pose,       # gripper pose in base
                    fingertip_pos       # fingertip pos in camera
                ))
        else:
            print("\033[91m" + "No PCD captured." + "\033[0m")
    
    print(f"Recorded {len(data)} data points.")
    # Save data
    datadir = "hacman_real_env/pcd_obs_env/calibration/data"
    os.makedirs(datadir, exist_ok=True)
    filepath = f"{datadir}/cam{cam_id}_pcd_data.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return filepath

def main():
    cam_id = 2
    # 0: right -     000003493812
    # 2: left -     000880595012
    # 1: front -    000180921812
    # 3: back -     000263392612
    initial_yaw = {
        0: 0.5 * np.pi,
        1: -0.5 * np.pi,
        2: 0.0,
        3: 0.0
    }[cam_id]
    
    # Perform the movements and record data
    move_robot_and_record_data(
        cam_id=cam_id, num_movements=20, debug=True, 
        initial_yaw=initial_yaw)
    

if __name__ == "__main__":
    main()