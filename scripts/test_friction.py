import numpy as np
import open3d as o3d

from hacman_real_env.real_env import RealEnv
from hacman.utils.primitive_utils import get_primitive_class, GroundingTypes
import hacman_real_env.primitives

result_dir = "scripts/test_results"

def poke(poke_env, angle=30, x_value=0.5, visualize=True):
    obs = poke_env.env.reset()
    object_pcd_points = obs['object_pcd_points']
    background_pcd_points = obs['background_pcd_points']

    # Get the target position
    target_pos = detect_center(object_pcd_points)

    # Calculate the delta value
    z = -x_value * np.tan(np.deg2rad(angle))
    motion = np.array([x_value, 0, z])

    # Visualize in open3d
    action_vec = poke_env.visualize(motion)
    if visualize:
        object_pcd = obs['object_pcd_o3d']
        # object_pcd.paint_uniform_color([1, 0.706, 0])
        background_pcd = obs['background_pcd_o3d']
        target_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        target_coord.translate(target_pos)
        action_o3d = np.linspace(target_pos, target_pos + action_vec, 100)
        action_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(action_o3d))
        action_o3d.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([object_pcd, background_pcd, target_coord, action_o3d])

    # Perform the poke action
    obs, _, _, _ = poke_env.execute(target_pos, motion)
    
    # Check the object center after poking
    center_pos = detect_center(obs['object_pcd_points'])
    delta_x = center_pos[0] - target_pos[0]
    target_delta_x = action_vec[0]
    print(f"Target delta_x: {np.round(target_delta_x, 3)}, Actual delta_x: {np.round(delta_x, 3)}")
    return delta_x, target_delta_x


def detect_center(pcd):
    # Selects the point at the center of the object pcd
    corner_min, corner_max = np.min(pcd, axis=0), np.max(pcd, axis=0)
    center = (corner_min + corner_max) / 2
    top_z = corner_max[2]
    center[2] = top_z
    distances = np.linalg.norm(pcd - center, axis=1)
    idx = np.argmin(distances)
    target_pos = pcd[idx]

    return target_pos

def test_main():
    object_name = "cardboard"
    env = RealEnv(
        voxel_size=0.001,
        voxel_downsample_size=0.001,
        registration=False,
        background_pcd_size=10000,
        object_pcd_size=5000,
        object_name=object_name,
        n_objects=1)
    poke_primitive = get_primitive_class('real-poke')
    poke_env = poke_primitive(
        env=env,
        grounding_type=GroundingTypes.OBJECT_ONLY,
        use_oracle_rotation=True)
    
    # Perform the poking action
    delta_xs, target_delta_xs, angles = [], [], []
    for angle in np.arange(10, 45.001, 5):
        for i in range(5):
            delta_x, target_delta_x = poke(poke_env, angle=angle)
            delta_xs.append(delta_x)
            target_delta_xs.append(target_delta_x)
            angles.append(angle)
    
    # Save the results
    import json
    with open(f"{result_dir}/poke_{object_name}_result.json", "w") as f:
        json.dump({"delta_x": delta_xs, "target_delta_x": target_delta_xs, "angle": angles}, f)

if __name__ == "__main__":
    test_main()