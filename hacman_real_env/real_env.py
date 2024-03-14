import os
import numpy as np
import open3d as o3d
import gym
from copy import deepcopy
from scipy.spatial.transform import Rotation
import cv2

from hacman_real_env.pcd_obs_env.hacman_obs_env import HACManObsEnv
from hacman_real_env.robot_controller import FrankaOSCController
from hacman_real_env.utils import *

from hacman.utils.transformations import to_pose_mat, transform_point_cloud, decompose_pose_mat, sample_idx

filedir = os.path.dirname(__file__)

class RealEnv(gym.Env, HACManObsEnv):
    def __init__(self,
                 object_name,
                 object_pcd_size=400,
                 background_pcd_size=1000,
                 voxel_downsample_size=0.005,
                 obs_args={}, 
                 robot_args={"controller_type": "OSC_YAW"},
                 eef_offset_path="data/offset_calibration_params.npz",
                 video_style="overlay",   # overlay or zoomed
                 transform_to_obs_frame=True,
                 allow_manual_registration=False,
                 allow_full_pcd=False,
                 symmetric_object=False,
                 record_video=False,
                 save_dir=None,
                 ) -> None:
        gym.Env.__init__(self)
        
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        self.pcd_size = object_pcd_size + background_pcd_size
        self.observation_space = self.init_observation_space()
        self.action_space = gym.spaces.Box(-1, 1, (5,))
        
        HACManObsEnv.__init__(self, object_name,
                              voxel_downsample_size=voxel_downsample_size,
                              allow_manual_registration=allow_manual_registration,
                              allow_full_pcd=allow_full_pcd,
                              symmetric_object=symmetric_object,
                              **obs_args)

        if transform_to_obs_frame:
            real2obs_transform = self.get_real2obs_transform()
        else:
            real2obs_transform = np.eye(4)
        
        # eef_offset = load_calibration_param(eef_offset_path)
        eef_offset = np.eye(4)
        # eef_offset[0, 3] += 0.015
        # eef_offset[2, 3] += 0.003
        self.robot = FrankaOSCController(
            frame_transform=real2obs_transform,    # All the poses sent to the robot need to be transformed to the real poses
            controller_offset=eef_offset,
            **robot_args)
        
        # Pre-defined segmentation ids
        self.object_seg_ids = [1]
        self.background_seg_ids = [0]

        # Primitive states
        self.lifted = False
        self.grasped = False
        self.current_obs = None
        self.last_reward = None

        # Video recording
        self.video_style = video_style
        self.video_count = 0
        self.video_info = {"elapsed_steps": 0}
        self.record_video = record_video
        self.save_dir = save_dir
        self.frames = []
    
    def init_observation_space(self):
        obs_space = gym.spaces.Dict(
            spaces={
                # "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                # "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                # "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                # "object_pcd_normals": gym.spaces.Box(-np.inf, np.inf, (self.object_pcd_size, 3)),
                "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                # "background_pcd_normals": gym.spaces.Box(-np.inf, np.inf, (self.background_pcd_size, 3)),
                # "object_ids": gym.spaces.Box(-np.inf, np.inf, (5,)),
                # "background_ids": gym.spaces.Box(-np.inf, np.inf, (5,)),
                # "object_dim": gym.spaces.Box(-np.inf, np.inf, (1,)),
            }
        )
        return obs_space
    
    def reset(self, **kwargs):
        self.lifted = False
        self.grasped = False
        self.current_obs = None
        self.robot.reset()
        if self.record_video and len(self.frames) > 0:
            success = "success" if self.video_info["success"] else "failure"
            step_count = self.video_info["elapsed_steps"]
            video_name = f"video_{self.video_count}_{success}_{step_count}_{np.round(self.last_reward, 3)}"
            images_to_video(self.frames, self.save_dir, video_name, fps=60)    # 2 * speed

            # Also save the goal image separately
            if self.video_style == "zoomed":
                goal_image = self.obj_reg.get_goal_img()
                goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
                goal_image_name = f"goal_image_{self.video_count}.png"
                cv2.imwrite(os.path.join(self.save_dir, goal_image_name), goal_image)

            self.video_count += 1
            self.video_info = {"elapsed_steps": 0}
            self.frames = []
        self.resample_goal()
        self.last_reward = None
        return self.get_obs()
    
    def get_obs(self):
        # Process the point cloud
        obs = HACManObsEnv.get_obs(self)
        object_pcd = obs['object_pcd_o3d']
        background_pcd = obs['background_pcd_o3d']

        # Additionally clip the object pcd using the gripper pose
        object_pcd = self._clip_gripper_pcd(object_pcd)

        # Convert to numpys
        object_pcd_points = np.asarray(object_pcd.points)
        object_pcd_normals = np.asarray(object_pcd.normals)
        background_pcd_points = np.asarray(background_pcd.points)
        background_pcd_normals = np.asarray(background_pcd.normals)        

        # Update the object pose and goal pose as the env state
        self.object_pose = obs['object_pose']
        self.goal_pose = obs['goal_pose']

        # Update the primitive states based on pcd obs
        object_lowest_z = np.asarray(object_pcd.points)[:,2].min()
        self.lifted = (object_lowest_z > 0.035)

        # State change:
        # 1. Grasped only when the object is lifted
        # 2. Not grasped if the robot is not grasping
        if self.lifted:
            self.grasped = True
        else:
            if not self.robot.is_grasped:
                self.grasped = False

        # Downsample the pcd points & normals
        down_sample_idx = sample_idx(len(object_pcd_points), self.object_pcd_size)
        object_pcd_points = object_pcd_points[down_sample_idx]
        object_pcd_normals = object_pcd_normals[down_sample_idx]

        down_sample_idx = sample_idx(len(background_pcd_points), self.background_pcd_size)
        background_pcd_points = background_pcd_points[down_sample_idx]
        background_pcd_normals = background_pcd_normals[down_sample_idx]

        # Save the object and background o3d
        self.object_pcd_o3d = object_pcd
        self.object_pcd_normals = object_pcd_normals
        self.background_pcd_o3d = background_pcd
        self.background_pcd_normals = background_pcd_normals

        processed_obs = {
            'object_pcd_points': object_pcd_points,
            'object_pcd_normals': object_pcd_normals,
            'background_pcd_points': background_pcd_points,
            'background_pcd_normals': background_pcd_normals,
            'object_pcd_o3d': object_pcd,            # kept for debugging, no downsample
            'background_pcd_o3d': background_pcd,   # kept for debugging, no downsample
        }
        return processed_obs
    
    def get_step_return(self, info):
        obs = self.get_obs()
        self.current_obs = deepcopy(obs)

        # Compute the flow reward
        object_pcd = deepcopy(obs['object_pcd_o3d'])
        object_pcd_points = np.asarray(object_pcd.points)
        goal_pose, object_pose = self.goal_pose, self.object_pose
        goal_pcd_points = transform_point_cloud(object_pose, goal_pose, object_pcd_points)
        flow = goal_pcd_points - object_pcd_points
        reward = -np.linalg.norm(flow, axis=1).mean()
        self.last_reward = reward

        # Compute the success (success if the mean flow is within 3cm)
        success = (reward > -0.03) or (reward > -0.05 and self.is_close_to_goal(tol=0.02))
        info["success"] = success
        self.video_info["success"] = success
        self.video_info["elapsed_steps"] += 1

        return obs, reward, success, info
    
    def _clip_gripper_pcd(self, pcd):
        # Clip the object pcd using the gripper pose
        gripper_pos = self.robot.eef_pose[:3, 3]
        gripper_base_z = gripper_pos[2] + 0.075
        
        # Any points above the gripper base z is clipped
        pcd_points_z = np.asarray(pcd.points)[:, 2]
        clip_mask = np.where(pcd_points_z > gripper_base_z)[0]
        pcd = pcd.select_by_index(clip_mask, invert=True)

        return pcd
    
    def get_primitive_states(self):
        return {
            "is_lifted": self.lifted,
            "is_grasped": self.grasped,
            "is_close_to_goal": self.is_close_to_goal()}
    
    def is_close_to_goal(self, tol=0.06):
        # if self.last_reward is None:
        #     return False
        # flow_distance = -self.last_reward
        if self.current_obs is None:
            return False
        object_pcd_points = deepcopy(self.current_obs['object_pcd_points'])
        object_centroid = object_pcd_points.mean(axis=0)
        goal_points = np.asarray(self.obj_reg.get_goal_pcd().points)
        goal_centroid = goal_points.mean(axis=0)

        centroid_dist = np.linalg.norm(object_centroid - goal_centroid)
        return centroid_dist < tol
    
    def get_object_pose(self, format='mat'):
        # Object pose is identity (since we only care about the transform to the goal pose)
        if format == 'mat':
            return self.object_pose
        else:
            p, q = decompose_pose_mat(self.object_pose)
            return np.concatenate([p, q])
    
    def get_goal_pose(self, format='mat'):
        # Goal pose is updated during get obs
        if format == 'mat':
            return self.goal_pose
        else:
            p, q = decompose_pose_mat(self.goal_pose)
            return np.concatenate([p, q])
    
    def get_gripper_pose(self, format='mat'):
        if format == 'mat':
            return self.robot.eef_pose
        elif format == 'vector':
            rot, p = self.robot.eef_rot_and_pos
            p = p.flatten()
            q = Rotation.from_matrix(rot).as_quat()
            return np.concatenate([p, q])
    
    def get_object_dim(self):
        # TODO
        return 0.05

    def get_segmentation_ids(self):
        return {"object_ids": [1], "background_ids": [0]}
    
    def start_video_record(self, cam_id=2, video_info={}):
        if self.record_video:
            super().start_video_record(cam_id)
            self.video_info.update(video_info)
    
    def end_video_record(self):
        if self.record_video:
            goal_image = self.obj_reg.get_goal_img()
            frames = super().end_video_record()
            for i, frame in enumerate(frames):
                # Change from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Add the goal image
                if self.video_style == "overlay":
                    frame = overlay_image(frame, goal_image, resize=0.3, corner="left bottom")
                    frame = put_info_on_image(frame, self.video_info)
                elif self.video_style == "zoomed":
                    frame = crop_image(frame, crop_size=0.6)

                # Add relevant info to the video frames
                frames[i] = frame
            return frames
    
    def step(self, action):
        """ Only used for testing purpose """
        target_pos = action[:3]
        # target_quat = Rotation.from_euler('xyz', action[3:]).as_quat()
        # target_quat = np.array([0, 0, -0.8509035, 0.525322])
        # target_delta_axis_angle = action[3:6]
        if len(action) == 3:
            target_quat = None
            target_delta_axis_angle = np.zeros(3)
        else:
            target_quat = action[3:7]
            target_delta_axis_angle = None
        # target_quat = np.array([0, 0, 0, 1])
        self.robot.move_to(
            target_pos,
            target_quat=target_quat,
            target_delta_axis_angle=target_delta_axis_angle,
            # num_steps=40,
            # num_additional_steps=20,
            # max_delta_pos=0.05,
            grasp=True)
        final_pose = self.robot.eef_pose
        return final_pose
    
def load_calibration_param(path):
    path = os.path.join(filedir, path)
    content = np.load(path)
    T = content['T']
    return T


'''
Test scripts
'''
def test_move_to_point(object_name):
    env = RealEnv(
        object_name=object_name,
        transform_to_obs_frame=False)
    env.reset()

    # Create interactive visualizer
    pcd = env.get_pcd(return_numpy=False, color=True)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked_points = vis.get_picked_points()
    print(picked_points)

    for i in picked_points:
        target_pos = np.asarray(pcd.points)[i]
        # Move to the target position
        gripper_target_pos = target_pos + np.array([0, 0, 0.04])
        env.step(gripper_target_pos)
        env.step(target_pos)

def test_main(object_name):
    env = RealEnv(object_name=object_name)
    obs = env.reset()

    object_pcd_points = obs['object_pcd_points']
    background_pcd_points = obs['background_pcd_points']

    # # Selects the point at the center of the object pcd
    # center = np.mean(object_pcd_points, axis=0)
    # distances = np.linalg.norm(object_pcd_points - center, axis=1)
    # idx = np.argmin(distances)
    # target_pos = object_pcd_points[idx]
    # gripper_target_pos = target_pos + np.array([0, 0, 0.04])

    # # Selects the corner of the object
    # object_y, object_z = object_pcd_points[:, 1], object_pcd_points[:, 2]
    # object_right, object_top = object_y.max(), object_z.max()
    # corner = np.array([object_right, object_top])
    # distances = np.linalg.norm(object_pcd_points[:, 1:3] - corner, axis=1)
    # idx = np.argmin(distances)
    # target_pos = object_pcd_points[idx]
    # gripper_target_pos = target_pos + np.array([0, 0, 0.04])

    # Visualize in open3d
    object_pcd = obs['object_pcd_o3d']
    # object_pcd.paint_uniform_color([1, 0.706, 0])
    background_pcd = obs['background_pcd_o3d']
    # background_pcd.paint_uniform_color([0, 0.651, 0.929])

    # Create interactive visualizer
    pcd = object_pcd + background_pcd
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked_points = vis.get_picked_points()
    print(picked_points)

    for i in picked_points:
        target_pos = np.asarray(pcd.points)[i]
        # Move to the target position
        gripper_target_pos = target_pos + np.array([0, 0, 0.08])
        env.step(gripper_target_pos)
        env.step(target_pos)

def test_poke(object_name):
    env = RealEnv(object_name=object_name)
    obs = env.reset()

    object_pcd_points = obs['object_pcd_points']
    background_pcd_points = obs['background_pcd_points']

    # Visualize in open3d
    object_pcd = obs['object_pcd_o3d']
    object_pcd.paint_uniform_color([1, 0.706, 0])
    background_pcd = obs['background_pcd_o3d']
    background_pcd.paint_uniform_color([0, 0.651, 0.929])

    # Create interactive visualizer
    pcd = object_pcd + background_pcd
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked_points = vis.get_picked_points()
    print(picked_points)

    for i in picked_points:
        target_pos = np.asarray(pcd.points)[i]
        # Move to the target position
        gripper_target_pos = target_pos + np.array([0, 0, 0.04])
        # env.step(gripper_target_pos)
        env.step(target_pos)

if __name__ == "__main__":
    object_name = "white_box"
    # test_move_to_point(object_name)
    test_main(object_name)
    # test_poke(object_name)

