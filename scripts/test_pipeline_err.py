import numpy as np
import open3d as o3d
import json

from hacman_real_env.real_env import RealEnv
from hacman.utils.primitive_utils import get_primitive_class, GroundingTypes
import hacman_real_env.primitives

result_dir = "scripts/pipeline_results"

def poke(env, pos):
    # Move to precontact, then to contact
    gripper_target_pos = pos + np.array([0, 0, 0.04])
    env.step(gripper_target_pos)
    contact_pos = pos + np.array([0, 0, -0.015])
    env.step(contact_pos)
    final_pos = env.robot.eef_rot_and_pos[1]

    # Move up
    env.step(gripper_target_pos)
    
    # Check the object center after poking
    return final_pos


def test_pipeline_err(num_pokes=100):
    object_name = "cardboard"
    env = RealEnv(
        voxel_size=0.001,
        voxel_downsample_size=0.001,
        registration=False,
        background_pcd_size=10000,
        object_pcd_size=5000,
        object_name=object_name,
        n_objects=1)

    obs = env.reset()
    object_pcd_points = obs['object_pcd_points']
    background_pcd_points = obs['background_pcd_points']

    # Calculate the target positions
    # Remove the points too close to the edge
    min_bounds, max_bounds = np.min(object_pcd_points, axis=0), np.max(object_pcd_points, axis=0)
    object_pcd_points = object_pcd_points[
        (object_pcd_points[:, 0] > min_bounds[0] + 0.03) &
        (object_pcd_points[:, 0] < max_bounds[0] - 0.03) &
        (object_pcd_points[:, 1] > min_bounds[1] + 0.03) &
        (object_pcd_points[:, 1] < max_bounds[1] - 0.03)
    ]
    target_indices = np.random.choice(len(object_pcd_points), num_pokes)
    targets = object_pcd_points[target_indices]

    # Visualize
    object_pcd = obs['object_pcd_o3d']
    background_pcd = obs['background_pcd_o3d']
    # object_pcd_down = object_pcd.voxel_down_sample(voxel_size=0.05)
    target_coords = []
    for target in targets:
        target_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        target_coord.translate(target)
        target_coords.append(target_coord)
    o3d.visualization.draw_geometries([object_pcd, background_pcd, *target_coords])

    # Poke at the target positions
    finals = []
    for target in targets:
        final_pos = poke(env, target)
        err = np.linalg.norm(final_pos - target)
        print(f"Error: {np.round(err, 3)}")
        finals.append(final_pos.tolist())
    
    # Save the results
    with open(f"{result_dir}/poke_results.json", "w") as f:
        results = {
            "targets": targets.tolist(),
            "finals": finals
        }
        json.dump(results, f)

def visualize_results(file):
    with open(file, "r") as f:
        results = json.load(f)
    targets = results["targets"]
    finals = results["finals"]

    # Visualize
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(targets))
    target_pcd.paint_uniform_color([1, 0.706, 0])
    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(finals))
    final_pcd.paint_uniform_color([0, 0.651, 0.929])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([target_pcd, final_pcd, origin])

def compute_z_error(file):
    with open(file, "r") as f:
        results = json.load(f)
    targets = results["targets"]
    finals = results["finals"]

    z_errors = []
    for target, final in zip(targets, finals):
        z_errors.append(target[2] - final[2])
    
    return z_errors

if __name__ == "__main__":
    # test_pipeline_err()
    visualize_results(f"{result_dir}/poke_results.json")