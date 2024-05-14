import numpy as np
import copy
import os
import itertools
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from hacman_real_env.robot_controller.robot_controller import FrankaOSCController

def benchmark_controller_err(kp_pos_list, kp_rot_list, residual_mass_vec_list, num_movements=20, existing_results=None):
    # Initialize the controller
    controller = FrankaOSCController(controller_type="OSC_YAW")
    initial_joint_positions = [
        -0.5493463,
        0.18639661,
        0.04967389,
        -1.92004654,
        -0.01182675,
        2.10698001,
        0.27106661]
    
    # Estimate the total number of experiments
    num_exps = len(kp_pos_list) * len(kp_rot_list) * len(residual_mass_vec_list) * num_movements
    print(f"Total number of experiments: {num_exps}")
    pg_bar = tqdm(total=num_exps)
    
    # Run the benchmark
    results = {}
    for kp_pos, kp_rot, residual_mass_vec in itertools.product(kp_pos_list, kp_rot_list, residual_mass_vec_list):
        # Skip if the results already exist
        param_key = (kp_pos, kp_rot, tuple(residual_mass_vec))
        if existing_results is not None and param_key in existing_results:
            print(f"Skipping {param_key}...")
            pg_bar.update(num_movements)
            continue

        print(f"kp_pos: {kp_pos}, kp_rot: {kp_rot}, residual_mass_vec: {residual_mass_vec}")
        controller_cfg = controller.controller_cfg 
        controller_cfg.Kp.translation = kp_pos
        controller_cfg.Kp.rotation = kp_rot
        controller_cfg.residual_mass_vec = residual_mass_vec
        # breakpoint()

        initial_pos = np.array([0.52, -0.3, 0.2])  # initial pos is slightly different from pos obtained from initial_joint_positions
        controller.move_to(initial_pos, target_delta_axis_angle=np.array([0, 0, 0]),
                            num_steps=100, num_additional_steps=50)
        
        """
        Alternative benchmarking method: move to a set of fixed target positions
        """
        # breakpoint()
        # delta_xs = [-0.1, 0.1]
        # delta_ys = [-0.1, 0.1]
        # delta_zs = [-0.1, 0.1]

        # target_poses, reached_poses = [], []
        # for _ in range(4):
        #     for dx, dy, dz in itertools.product(delta_xs, delta_ys, delta_zs):
        #         target_pos = initial_pos + np.array([dx, dy, dz])
        #         print("--------------------")
        #         print(f"Target pos: {target_pos}")
        #         controller.reset(joint_positions=initial_joint_positions)
        #         controller.move_to(target_pos, target_delta_axis_angle=np.array([0, 0, 0]),
        #                         num_steps=40, num_additional_steps=20)
        #         err = controller.eef_pose[:3, 3] - target_pos
        #         print(f"Error: {err}, norm: {np.linalg.norm(err)}")
        #         target_poses.append(target_pos)
        #         reached_poses.append(controller.eef_pose[:3, 3])

        x_bounds = [-0.05, 0.05]
        y_bounds = [-0.05, 0.05]
        z_bounds = [-0.05, 0.05]
        target_poses, reached_poses = [], []
        for _ in range(num_movements):
            delta_pos = np.random.uniform(low=[x_bounds[0], y_bounds[0], z_bounds[0]], high=[x_bounds[1], y_bounds[1], z_bounds[1]])
            curr_target_pos = initial_pos + delta_pos
            print("--------------------")
            print(f"Target pos: {curr_target_pos}")
            # controller.reset(joint_positions=initial_joint_positions)
            controller.move_to(curr_target_pos, target_delta_axis_angle=np.array([0, 0, 0]),
                            num_steps=40, num_additional_steps=20)
            err = controller.eef_pose[:3, 3] - curr_target_pos
            print(f"Error: {err}, norm: {np.linalg.norm(err)}")
            target_poses.append(curr_target_pos)
            reached_poses.append(controller.eef_pose[:3, 3])
            pg_bar.update(1)
        
        target_poses = np.array(target_poses)
        reached_poses = np.array(reached_poses)
        controller.reset(joint_positions=initial_joint_positions)

        # Record the results
        param_key = (kp_pos, kp_rot, tuple(residual_mass_vec))
        results[param_key] = (target_poses, reached_poses)
    
    return results

def save_benchmark_results(results):
    save_dir = os.path.join(os.path.dirname(__file__), "benchmark_results_small_motions")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "benchmark_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

def load_benchmark_results():
    save_dir = os.path.join(os.path.dirname(__file__), "benchmark_results_small_motions")
    save_path = os.path.join(save_dir, "benchmark_results.pkl")
    if not os.path.exists(save_path):
        return None
    with open(save_path, "rb") as f:
        results = pickle.load(f)
    return results

def parse_benchmark_results(results):
    """
    Calculate the mean error and error along each axis for each set of parameters. Sort the results
    by mean error, and print the results.
    """
    report = []

    # Calcaulte the mean error for each set of parameters
    for param_key, (target_poses, reached_poses) in results.items():
        err = abs(reached_poses - target_poses)
        err_along_axis = np.mean(err, axis=0)
        err_norm = np.linalg.norm(err, axis=-1)
        mean_err = np.mean(err_norm, axis=0)
        std_err = np.std(err_norm, axis=0)
        report.append((param_key, mean_err, std_err, *err_along_axis))
    
    # Sort the results by mean error
    report.sort(key=lambda x: x[1])

    # Print the results as a table (with each entry equal length)
    print("Mean error\tStd error\tError in x\tError in y\tError in z\tKp pos\tKp rot\tResidual mass vec")
    for param_key, mean_err, std_err, err_x, err_y, err_z in report:
        kp_pos, kp_rot, residual_mass_vec = param_key
        print("{:10.3f}\t{:10.3f}\t{:10.3f}\t{:10.3f}\t{:10.3f}\t{}\t{}\t{}".format(mean_err, std_err, err_x, err_y, err_z, kp_pos, kp_rot, residual_mass_vec))
        

def visualize_err(all_target_pos, all_final_pos):
    # Report the mean
    err = abs(all_final_pos - all_target_pos)
    err_norm = np.linalg.norm(err, axis=-1)
    print(f"Mean error: {np.mean(err_norm, axis=0)}")

    # Draw a histogram of the errors
    axis_names = ["x", "y", "z"]
    fig = plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.hist(err[:, i], bins=20)

        # Draw where the mean is
        plt.axvline(x=np.mean(err[:, i]), color='r', linestyle='dashed', linewidth=1)
        plt.ylabel(f"Error in axis {axis_names[i]}")
    plt.show()

    # Draw a 3D plot with both target and final positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_target_pos[:, 0], all_target_pos[:, 1], all_target_pos[:, 2], c='r', marker='o')
    ax.scatter(all_final_pos[:, 0], all_final_pos[:, 1], all_final_pos[:, 2], c='b', marker='^')
    plt.show()

if __name__ == "__main__":
    kp_pos_list = [
        # 150.0,
        # 170.0,
        # (170.0, 170.0, 150.0),
        # 200.0,
        # 220.0,
        # (200, 180, 180),
        # (240, 200, 200),
        # (200, 180, 180)
        # (240, 220, 220),
        # (260, 220, 220)
        # (250, 200, 200),
        # 250.0,
        # (300, 250, 250),
        # 300.0,
        # (350, 300, 300),
        # 350.0,
        # (400, 350, 350),
        # 400.0,
        # (450, 400, 400),
        # 450.0,
        (500, 450, 450),
        500.0,
        (550, 500, 500),
        550.0,
        (600, 550, 550),
        # 700.0,
        ]
    kp_rot_list = [
        # 50.0,
        # 150.0,
        250.0
        ]
    residual_mass_vec_list = [
        # [0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5],
        # [0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.3],
        # [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
        # [0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.5],
        # [0.0, 0.0, 0.0, 0.0, 0.5, 0.7, 0.7],
    ]

    # results = load_benchmark_results()
    new_results = benchmark_controller_err(
        kp_pos_list, kp_rot_list, residual_mass_vec_list,
        num_movements=40,
        # existing_results=results,     # Skip parameters in existing results
        existing_results=None           # Set to None to force collect new results
        )
    # results.update(new_results)
    results = new_results
    save_benchmark_results(results)
    parse_benchmark_results(results)

    # all_target_pos, all_final_pos = load_benchmark_results()
    # visualize_err(all_target_pos, all_final_pos)