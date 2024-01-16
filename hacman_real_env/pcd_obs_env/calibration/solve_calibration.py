import numpy as np
import cv2
import pickle, os
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation

def estimate_tag_pose(finger_pose):
    """
    Estimate the tag pose given the gripper pose by applying the gripper-to-tag transformation.

    Args:
        finger_pose (eef_pose): 4x4 transformation matrix from gripper to robot base
    Returns:
        hand_pose: 4x4 transformation matrix from hand to robot base
        tag_pose: 4x4 transformation matrix from tag to robot base
    """
    from scipy.spatial.transform import Rotation

    # Estimate the hand pose
    # finger_to_hand obtained from the product manual: 
    # [https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf]
    finger_to_hand = np.array([
        [0.707,  0.707, 0, 0],
        [-0.707, 0.707, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1],
    ])
    finger_to_hand = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1],
    ])
    hand_to_finger = np.linalg.inv(finger_to_hand)
    print("hand to finger", hand_to_finger)
    hand_pose = np.dot(finger_pose, hand_to_finger)

    t_tag_to_hand = np.array([0.048914, 0.0275, 0.00753])
    # R_tag_to_hand = Rotation.from_quat([0.5, -0.5, 0.5, -0.5])
    R_tag_to_hand = Rotation.from_quat([0, 0, 0, 1])
    tag_to_hand = np.eye(4)
    tag_to_hand[:3, :3] = R_tag_to_hand.as_matrix()
    tag_to_hand[:3, 3] = t_tag_to_hand

    tag_pose = np.dot(hand_pose, tag_to_hand)
    
    return hand_pose, tag_pose

def solve_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.
    """
    assert inpts.shape == outpts.shape
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def calculate_reprojection_error(tag_poses, target_poses, T_matrix):
    errors = []
    for tag_pose, target_pose in zip(tag_poses, target_poses):
        # Transform target pose using T_matrix
        transformed_target = np.dot(T_matrix, target_pose)
        transformed_pos = transformed_target[:3, 3]

        # Compare with tag pos
        tag_pos = tag_pose[:3, 3]
        error = np.linalg.norm(tag_pos - transformed_pos)
        errors.append(error)

    # Compute average error
    avg_error = np.mean(errors)
    return avg_error

def solve_extrinsic(gripper_poses, target_poses_in_camera, eye_to_hand=True):
    """
    Solve the extrinsic calibration between the camera and the base.
    """
    if eye_to_hand:
        # Calculate the transformation matrix from gripper to tag
        tag_poses = [estimate_tag_pose(pose)[1] for pose in gripper_poses]
    # T1, T = solve_hand_eye_calibration(
    #     gripper_poses, target_poses_in_camera)
    # print(f"Transformation matrix T1:\n{T1}")
    # print(f"Transformation matrix T:\n{T}")

    # origin_pose = np.eye(4)
    # transformed_origin = T @ origin_pose
    # print(f"Transformed origin:\n{transformed_origin}")
    
    gripper_pos = np.array([pose[:3, 3] for pose in tag_poses])
    target_pos = np.array([pose[:3, 3] for pose in target_poses_in_camera])
    T = solve_rigid_transformation(target_pos, gripper_pos)
    print(f"Transformation matrix T:\n{T}")

    # Calculate the reprojection error
    avg_error = calculate_reprojection_error(
        tag_poses, target_poses_in_camera, T)
    print(f"Average reprojection error: {avg_error}")

    # # Calculate tag pose in base
    # target_poses = [T @ p for p in gripper_poses]

    # # Solve the rigid transformation
    # T = solve_rigid_transformation(
    #     target_poses_in_camera, target_poses)

    return T


if __name__ == "__main__":
    filepath = os.path.abspath(__file__)
    dirpath = os.path.dirname(filepath)

    # Load data
    cam_id = 0
    data_dirname = os.path.join(dirpath, "data")
    data_filepath = os.path.join(data_dirname, f"cam{cam_id}_data.pkl")
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    gripper_poses, target_poses_in_camera = zip(*data) 
    
    # Solve the extrinsic calibration
    T = solve_extrinsic(gripper_poses, target_poses_in_camera)

    # Save the calibration
    calib_dirname = os.path.join(dirpath, "calibration_results")
    os.makedirs(calib_dirname, exist_ok=True)
    filepath = os.path.join(calib_dirname, f"cam{cam_id}_calibration.npz")
    np.savez(filepath, T=T)
