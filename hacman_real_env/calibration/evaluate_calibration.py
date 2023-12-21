"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import os, pickle
import cv2

from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType

from hacman_real_env.robot_controller import FrankaOSCController
from marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation


def move_robot_and_record_data(cam_id, num_movements=3, visualize=False):
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
    for _ in range(num_movements):
        # Generate a random target delta pose
        random_delta_pos = [np.random.uniform(-0.08, 0.08, size=(3,))]
        random_delta_axis_angle = [np.random.uniform(-0.6, 0.6, size=(3,))]
        robot.reset(joint_positions=[
            -0.57259571, 0.54167994, 0.07167276, -1.70355534, -0.01052658, 2.23024466, 0.28936683])
        robot.move_by(random_delta_pos, random_delta_axis_angle, num_steps=40, num_additional_steps=30)

        # Get current pose of the robot 
        gripper_pose = robot.eef_pose

        # Capture IR frame from Kinect
        ir_frame = get_kinect_ir_frame(k4a)
        if ir_frame is not None:
            # Detect ArUco markers and get visualization
            corners, ids = detect_aruco_markers(ir_frame, visualize=visualize)

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

def solve_hand_eye_calibration(data_filepath, eye_to_hand=True):
    # Load data
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    gripper_poses, target_poses = zip(*data)

    # Convert to arrays of rotation vectors and translation vectors
    R_gripper, t_gripper = [], []
    R_target, t_target = [], []

    for pose in gripper_poses:
        # r, _ = cv2.Rodrigues(pose[:3, :3])
        r = pose[:3, :3]
        t = pose[:3, 3]
        R_gripper.append(r)
        t_gripper.append(t)

    for pose in target_poses:
        # r, _ = cv2.Rodrigues(pose[:3, :3])
        r = pose[:3, :3]
        t = pose[:3, 3]
        R_target.append(r)
        t_target.append(t)

    # Solve Hand-Eye Calibration
    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper, t_gripper):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        # change parameters values
        R_gripper = R_base2gripper
        t_gripper = t_base2gripper

    # calibrate
    R, T = cv2.calibrateHandEye(
        R_gripper2base=R_gripper,
        t_gripper2base=t_gripper,
        R_target2cam=R_target,
        t_target2cam=t_target,
    )
    return R, T    


def main():
    # Perform the movements and record data
    cam_id = 1
    # data_filepath = move_robot_and_record_data(
    #     cam_id=cam_id, num_movements=50, visualize=False)
    
    data_filepath = f"pcd_env/calibration/data/cam{cam_id}_data.pkl"
    R, T = solve_hand_eye_calibration(data_filepath)
    print(f"R: {R}")
    print(f"T: {T}")

    # Save the calibration
    os.makedirs("pcd_env/calibration/calibration_results", exist_ok=True)
    filepath = f"pcd_env/calibration/calibration_results/cam{cam_id}_calibration.npz"
    np.savez(filepath, R=R, T=T)

if __name__ == "__main__":
    main()