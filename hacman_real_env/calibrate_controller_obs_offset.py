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

calibration_dir = os.path.join(os.path.dirname(__file__), 'data')

def record_gripper_pos(num_steps=100, visualize=False, T=None):
    # Initialize the env
    env = RealEnv(
        object_name="white_box",
        background_pcd_size=3000,
        object_pcd_size=2000,
        voxel_size=0.002,
        obs_args=dict(voxel_size=0.002,),
    )

    env.robot.reset()
    lower_bound, upper_bound = env.get_scene_bounds(offset=0.05)
    lower_bound[2] = 0.06 # Always above the surface
    upper_bound[2] = 0.1
    pos_pcd, pos_controller = [], []
    for _ in tqdm.tqdm(range(num_steps)):
        # env.robot.reset()
        target_pos = np.random.uniform(lower_bound, upper_bound)
        target_yaw = np.random.uniform(-np.pi/2, np.pi/2)
        # target_yaw = 0.0
        target_quat = Rotation.from_euler('xyz', [np.pi, 0, target_yaw]).as_quat()
        # target_pos = np.array([0.0, 0.0, 0.15])
        action = np.hstack((target_pos, target_quat))
        reached_pose = env.step(action=action)

        # Estimated reached pos with correction
        if T is not None:
            reached_pose = T @ reached_pose

        reached_pos = reached_pose[:3, 3]

        # Get the pcd
        pcd = env.get_pcd(return_numpy=False)
        pcd = env.transform2obs(pcd)    # Transform to obs frame
        tip_pcd, gripper_pcd = segment_tip(env, pcd)
        tip_pos = np.asarray(tip_pcd.points).mean(axis=0)

        # Report the error
        err = np.asarray(tip_pos) - np.asarray(reached_pos)
        err_norm = np.linalg.norm(err)
        print(f"Error norm: {err_norm:.3f}. Error: {err}")

        if visualize:
            reached_pos_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            reached_pos_o3d.transform(reached_pose)
            tip_pos_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=tip_pos)
            tip_pcd.paint_uniform_color([1, 0, 0])
            gripper_pcd.paint_uniform_color([0, 1, 0])
            pcd.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([tip_pcd, gripper_pcd, pcd, reached_pos_o3d, tip_pos_o3d])

        # Check if it is outlier
        if err_norm > 0.03:
            print("Outlier!")
            continue
        
        pos_pcd.append(tip_pos)
        pos_controller.append(reached_pos)

    # Save the pos
    os.makedirs(calibration_dir, exist_ok=True)
    filepath = os.path.join(calibration_dir, f'offset_calibration_data.pkl')
    with open(filepath, 'wb') as f:
        content = {
            'pos_pcd': pos_pcd,
            'pos_controller': pos_controller,
        }
        pickle.dump(content, f)
    
    return

def segment_tip(env, pcd):
    pcd = deepcopy(pcd)

    # Get gripper pcd (any point not in the background)
    # bg_pcd = load_background()
    # bg_pcd = env.transform2obs(bg_pcd)
    # distances = pcd.compute_point_cloud_distance(bg_pcd)
    # bg_mask = np.where(np.asarray(distances) <= 0.03)[0]
    # gripper_pcd = pcd.select_by_index(bg_mask, invert=True)

    # Get the gripper pcd (points above 4cm)
    pcd_z = np.asarray(pcd.points)[:, 2]
    gripper_mask = np.where(pcd_z >= 0.04)[0]
    gripper_pcd = pcd.select_by_index(gripper_mask)

    # Get the tip points
    gripper_z = np.asarray(gripper_pcd.points)[:, 2]
    tip_z = gripper_z.min()
    tip_mask = np.where(gripper_z <= (tip_z + 0.005))[0]
    tip_pcd = gripper_pcd.select_by_index(tip_mask)

    return tip_pcd, gripper_pcd

def load_calibration_data():
    filepath = os.path.join(calibration_dir, f'offset_calibration_data.pkl')
    with open(filepath, 'rb') as f:
        content = pickle.load(f)
        pos_pcd = content['pos_pcd']
        pos_controller = content['pos_controller']
    
    return pos_pcd, pos_controller

def compute_correspondence(pos_pcd, pos_controller):
    from hacman_real_env.pcd_obs_env.calibration.solve_calibration import solve_rigid_transformation, calculate_reprojection_error
    pos_pcd = np.asarray(pos_pcd)
    pos_controller = np.asarray(pos_controller)

    T = solve_rigid_transformation(pos_controller, pos_pcd)
    print(f"Transformation matrix T:\n{T}")

    # Calculate the reprojection error
    transfomed_pos_controller = transform_points(pos_controller, T)
    err = np.linalg.norm(transfomed_pos_controller - pos_pcd, axis=1)
    print(f"Reprojection error: {err.mean():.3f} +- {err.std():.3f}")

    return T, transfomed_pos_controller

def visualize_offset(pos_pcd, pos_controller):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(0, 0.4)

    # Assuming pos_pcd and pos_controller are lists of numpy arrays
    for pos, ctrl in zip(pos_pcd, pos_controller):
        delta = ctrl - pos
        ax.arrow3D(
              *ctrl, 
              *delta,
              mutation_scale=10,
              arrowstyle="-|>",
              linestyle='dashed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()
    plt.title('Offset Visualization')
    plt.show()

def visualize_pos_distribution(pos_pcd, pos_controller):
    pos_pcd = np.asarray(pos_pcd)
    pos_controller = np.asarray(pos_controller)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(0, 0.3)

    ax.scatter(*pos_pcd.T, s=5, label='Tip Position')
    ax.scatter(*pos_controller.T, s=5, label='Controller Position')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()
    plt.legend()
    plt.title('Offset Visualization')
    plt.show()

def visualize_offset_distribution(pos_pcd, pos_controller):
    """
    Visualize the offset distribution along each axis
    """
    axes_titles = ['X', 'Y', 'Z']
    pos_pcd = np.asarray(pos_pcd)
    pos_controller = np.asarray(pos_controller)
    err = pos_pcd - pos_controller

    fig = plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.hist(err[:, i], bins=40)
        plt.ylabel(axes_titles[i])
    
    plt.tight_layout()
    plt.show()

def visualize_pairwise_distribution(pos_pcd, pos_controller):
    """
    Visualize the pairwise offset distribution as heatmaps
    """
    axes_titles = ['X', 'Y', 'Z']
    pos_pcd = np.asarray(pos_pcd)
    pos_controller = np.asarray(pos_controller)
    err = pos_pcd - pos_controller

    fig = plt.figure()
    plt_idx = 1
    for i, j in itertools.combinations(range(3), 2):
        plt.subplot(1, 3, plt_idx)
        plt_idx += 1
        plt.hist2d(
            err[:, i],
            err[:, j],
            range=[[-0.05, 0.05], [-0.05, 0.05]],
            bins=40)
        plt.xlabel(axes_titles[i])
        plt.ylabel(axes_titles[j])
        plt.tight_layout()
    
    plt.show()

def save_calibration_param(T):
    os.makedirs(calibration_dir, exist_ok=True)
    filepath = os.path.join(calibration_dir, f'offset_calibration_params.npz')
    np.savez(filepath, T=T)

def load_calibration_param(filepath=None):
    if filepath is None:
        filepath = os.path.join(calibration_dir, f'offset_calibration_params.npz')
    content = np.load(filepath)
    T = content['T']
    return T

def record_background():
    env = RealEnv(
        object_name="white_box",
        background_pcd_size=3000,
        object_pcd_size=2000,
        voxel_size=0.002,
        obs_args=dict(voxel_size=0.002,),
    )
    bg_pcd = env.get_pcd(return_numpy=False)
    o3d.visualization.draw_geometries([bg_pcd])

    # Save the bg pcd
    os.makedirs(calibration_dir, exist_ok=True)
    filepath = os.path.join(calibration_dir, f'empty_background.pcd')
    o3d.io.write_point_cloud(filepath, bg_pcd)

def load_background():
    filepath = os.path.join(calibration_dir, f'empty_background.pcd')
    bg_pcd = o3d.io.read_point_cloud(filepath)
    return bg_pcd

if __name__ == "__main__":
    # record_background()
    # record_gripper_pos(
    #     num_steps=1000,
    #     # visualize=True,
    #     )
    
    pos_pcd, pos_controller = load_calibration_data()
    # visualize_offset_distribution(pos_pcd, pos_controller)
    # visualize_offset(pos_pcd, pos_controller)
    # visualize_pos_distribution(pos_pcd, pos_controller)
    # visualize_pairwise_distribution(pos_pcd, pos_controller)

    # Compute the correspondence
    T, transformed_pos_ctrl = compute_correspondence(pos_pcd, pos_controller)
    # T = load_calibration_param()
    # T[2, 3] -= -0.0766
    # save_calibration_param(T)
    record_gripper_pos(
        num_steps=50,
        visualize=True,
        # T=T,
        )
    # visualize_offset(pos_pcd, transformed_pos_ctrl)
    # visualize_offset_distribution(pos_pcd, transformed_pos_ctrl)
    # visualize_pairwise_distribution(pos_pcd, transformed_pos_ctrl)


    

