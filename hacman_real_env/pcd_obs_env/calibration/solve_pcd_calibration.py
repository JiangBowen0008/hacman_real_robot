import numpy as np
import cv2
import pickle, os
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation

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

def calculate_reprojection_error(tip_pos_in_controller, tip_pos_in_camera, T_matrix):
    errors = []
    for pos_ctrl, pos_cam in zip(tip_pos_in_controller, tip_pos_in_camera):
        # Transform target pose using T_matrix
        transformed_pos_cam = transform_point(pos_cam, T_matrix)

        # Compare with tag pos
        error = np.linalg.norm(pos_ctrl - transformed_pos_cam)
        errors.append(error)

    # Compute average error
    avg_error = np.mean(errors)
    return avg_error

def transform_point(point, T):
    """
    Transform a point using the transformation matrix T.
    """
    point = np.asarray(point)
    point = np.hstack((point, 1))
    transformed_point = np.dot(T, point)
    return transformed_point[:3]

def solve_extrinsic(tip_pose_in_controller, tip_pos_in_camera):
    """
    Solve the extrinsic calibration between the camera and the base.
    """
    tip_pose_in_controller = np.asarray(tip_pose_in_controller)
    tip_pos_in_camera = np.asarray(tip_pos_in_camera)

    tip_pos_in_controller = np.array([pose[:3, 3] for pose in tip_pose_in_controller])
    T = solve_rigid_transformation(tip_pos_in_camera, tip_pos_in_controller)
    print(f"Transformation matrix T:\n{T}")

    # Calculate the reprojection error
    avg_error = calculate_reprojection_error(
        tip_pos_in_controller, tip_pos_in_camera, T)
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
    cam_id = 2
    data_dirname = os.path.join(dirpath, "data")
    data_filepath = os.path.join(data_dirname, f"cam{cam_id}_pcd_data.pkl")
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    gripper_poses, target_poses_in_camera = zip(*data) 
    
    # Solve the extrinsic calibration
    T = solve_extrinsic(gripper_poses, target_poses_in_camera)

    # Save the calibration
    calib_dirname = os.path.join(dirpath, "calibration_results")
    os.makedirs(calib_dirname, exist_ok=True)
    filepath = os.path.join(calib_dirname, f"cam{cam_id}_pcd_calibration.npz")
    np.savez(filepath, T=T)
