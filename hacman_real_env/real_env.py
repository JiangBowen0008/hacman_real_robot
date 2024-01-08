import numpy as np
import open3d as o3d
import gym
from scipy.spatial.transform import Rotation

from hacman_real_env.pcd_obs_env import PCDObsEnv
from hacman_real_env.robot_controller import FrankaOSCController

from hacman.utils.transformations import to_pose_mat, transform_point_cloud, decompose_pose_mat, sample_idx

class RealEnv(gym.Env, PCDObsEnv):
    def __init__(self,
                 object_pcd_size=400,
                 background_pcd_size=1200,
                 obs_args={}, 
                 robot_args={"controller_type": "OSC_YAW"}
                 ) -> None:
        gym.Env.__init__(self)
        
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        self.pcd_size = object_pcd_size + background_pcd_size
        self.observation_space = self.init_observation_space()
        
        self.robot = FrankaOSCController(**robot_args)
        PCDObsEnv.__init__(self, **obs_args)

        # Primitive states
        self.grasped = False
    
    def init_observation_space(self):
        obs_space = gym.spaces.Dict(
            spaces={
                # "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                # "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                # "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                "object_pcd_normals": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                "background_pcd_normals": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
            }
        )
        return obs_space
    
    def reset(self, **kwargs):
        self.robot.reset()
        return self.get_obs()
    
    def get_obs(self, return_o3d=True):
        # TODO
        pcd = PCDObsEnv.get_pcd(self, return_numpy=False)
        
        # Mask out the non-scene pcd (gripper, undesired table area)
        table_x_boundary = [0.15, 0.8]
        table_y_boundary = [-0.85, -0.2]
        table_z_boundary = [0.0, 0.15]
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([table_x_boundary[0], table_y_boundary[0], table_z_boundary[0]]),
                max_bound=np.array([table_x_boundary[1], table_y_boundary[1], table_z_boundary[1]])
            )
        )

        # objects pcds are all above 0.02 but below 0.2
        pcd_z = np.asarray(pcd.points)[:, 2]
        object_z_bounds = (0.035, 0.15)
        object_mask = (pcd_z > object_z_bounds[0])
        object_pcd = pcd.select_by_index(np.where(object_mask)[0])
        bg_pcd = pcd.select_by_index(np.where(~object_mask)[0])

        # Process the object pcd
        object_pcd.estimate_normals()
        object_pcd.orient_normals_consistent_tangent_plane(10)
        curr_pcd_size = len(object_pcd.points)
        pcd_down_mask = np.random.choice(curr_pcd_size, int(self.object_pcd_size))
        object_pcd = object_pcd.select_by_index(pcd_down_mask)
        
        # Process the background pcd
        bg_pcd.estimate_normals()
        bg_pcd.orient_normals_consistent_tangent_plane(10)
        curr_pcd_size = len(bg_pcd.points)
        pcd_down_mask = np.random.choice(curr_pcd_size, int(self.background_pcd_size))
        background_pcd = bg_pcd.select_by_index(pcd_down_mask)

        obs = {
            'object_pcd_points': np.asarray(object_pcd.points),
            'object_pcd_normals': np.asarray(object_pcd.normals),
            'background_pcd_points': np.asarray(background_pcd.points),
            'background_pcd_normals': np.asarray(background_pcd.normals),
        }
        if return_o3d:
            obs.update({
                'object_pcd_o3d': object_pcd,
                'background_pcd_o3d': background_pcd,
            })
        return obs
    
    def get_step_return(self, info):
        # TODO
        reward = 0.
        obs = self.get_obs()
        success = False

        info["success"] = success
        return obs, reward, False, info
    
    def get_primitive_states(self):
        # TODO
        return {"is_grasped": self.grasped}
    
    def get_object_pose(self, format='mat'):
        # TODO
        p, q = np.array([0.55, -0.35, 0.12]), np.array([0, 0, 0, 1])
        if format == 'mat':
            return to_pose_mat(p, q, input_wxyz=False)
        elif format == 'vector':
            return np.concatenate([p, q])
    
    def get_goal_pose(self, format='mat'):
        # TODO
        p, q = np.array([0.55, -0.35, 0.12]), np.array([0, 0, 0, 1])
        if format == 'mat':
            return to_pose_mat(p, q, input_wxyz=False)
        elif format == 'vector':
            return np.concatenate([p, q])
    
    def get_gripper_pose(self, format='mat'):
        if format == 'mat':
            return self.robot.eef_pose
        elif format == 'vector':
            rot, p = self.robot.eef_rot_and_pos
            q = Rotation.from_matrix(rot).as_quat()
            return np.concatenate([p, q])
    
    def get_object_dim(self):
        # TODO
        return 0.05

    def get_segmentation_ids(self):
        return {"object_ids": [1], "background_ids": [0]}
    
    def step(self, action):
        target_pos = action[:3]
        # target_quat = Rotation.from_euler('xyz', action[3:]).as_quat()
        # target_quat = np.array([0, 0, -0.8509035, 0.525322])
        target_delta_axis_angle = np.zeros(3)
        # target_quat = np.array([0, 0, 0, 1])
        self.move_to(target_pos, target_delta_axis_angle=target_delta_axis_angle, num_steps=40, num_additional_steps=20)
        final_pos = self.eef_pose[:3, 3]
        return self.get_obs()

if __name__ == "__main__":
    env = RealEnv()
    pcd = env.reset()

    # objects pcds are all above 0.08 but below 0.25
    mask = (pcd[:, 2] > 0.085) * (pcd[:, 2] < 0.25)
    object_pcd = pcd[mask]
    background_pcd = pcd[~mask]

    # Selects the point at the center of the object pcd
    center = np.mean(object_pcd, axis=0)
    distances = np.linalg.norm(object_pcd - center, axis=1)
    idx = np.argmin(distances)
    target_pos = object_pcd[idx]
    gripper_target_pos = target_pos + np.array([0, 0, 0.04])

    # Visualize in open3d
    object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_pcd))
    object_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    object_pcd.paint_uniform_color([1, 0.706, 0])
    background_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(background_pcd))
    background_pcd.paint_uniform_color([0, 0.651, 0.929])
    background_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pos.reshape(1, 3)))
    target_pcd.paint_uniform_color([1, 0, 0])
    gripper_target_pos_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_target_pos.reshape(1, 3)))
    gripper_target_pos_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([object_pcd, background_pcd, target_pcd, gripper_target_pos_pcd])

    # Move to the target position
    env.step(gripper_target_pos)
    env.step(target_pos)

