
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation

from hacman.utils.primitive_utils import GroundingTypes, Primitive, register_primitive

class RealEnvPrimitive(Primitive):
    def reset(self, **kwargs):
        self.env.robot.reset(**kwargs)
    
    def get_step_return(self, info):
        return self.env.get_step_return(info)
    
    def get_location_bounds(self):
        return self.env.get_scene_bounds(offset=0.015)

    def move_to(self, location, quat, grasp=True, **kwargs):
        # Bound the x, y location
        location_bounds = self.get_location_bounds()
        location[:2] = np.clip(location[:2], location_bounds[0][:2], location_bounds[1][2:])
        self.env.robot.move_to(location, target_quat=quat, grasp=grasp, **kwargs)
    
    def move_by(self, target_delta_pos, grasp=True, **kwargs):
        self.env.robot.move_by(target_delta_pos, grasp=grasp, **kwargs)
    
    def close_gripper(self):
        self.env.robot.move_by(grasp=True, num_steps=20, num_additional_steps=0, end_on_reached=False)
    
    def open_gripper(self):
        self.env.robot.move_by(grasp=False, num_steps=20, num_additional_steps=0, end_on_reached=False)
        self.env.robot.move_by(
            target_delta_pos=[0, 0, 0.15],
            grasp=False, num_steps=20, num_additional_steps=0)

    def move_to_contact(self, location, normal, quat, close_gripper=True):
        # Move to precontact
        precontact = location + normal * 0.04
        self.move_to(location=precontact, quat=quat, grasp=close_gripper)

        # Move to location
        offset = np.array([0, 0, +0.000])
        self.move_to(
            location=location+offset,
            quat=quat,
            grasp=close_gripper,
            num_steps=10,
            num_additional_steps=10,
            max_delta_pos=0.05,)
        return True
    
    def move_to_from_top(self, location, quat, close_gripper=True):
        # Move to precontact
        precontact = location.copy()
        precontact[2] = 0.25
        self.move_to(location=precontact, quat=quat, grasp=close_gripper)

        # Move to location
        offset = np.array([0, 0, -0.00])
        self.move_to(location=location+offset, quat=quat, grasp=close_gripper)
        return True
    
    def convert_yaw_to_quat(self, yaw):
        yaw = yaw * 0.5 # - 0.5 * np.pi
        if yaw > 0.5 * np.pi:
            yaw -= np.pi
        elif yaw < -0.5 * np.pi:
            yaw += np.pi
        quat = Rotation.from_euler('xyz', [np.pi, 0, yaw]).as_quat()
        return quat
    
    def start_video_record(self):
        self.env.unwrapped.start_video_record()
    
    def end_video_record(self):
        frames = self.env.unwrapped.end_video_record()
        if frames is not None:
            self.env.unwrapped.frames.extend(frames)

@register_primitive('real-open_gripper', GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class OpenGripper(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        self.start_video_record()
        self.open_gripper()
        self.end_video_record()
        info = {}
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.15
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_lifted'] or states['is_grasped'] 
        # return states['is_grasped'] 

@register_primitive("real-poke", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Poke(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        self.start_video_record()
        # Save original object pose
        # obj_pose = self.env.get_object_pose()
        normal = kwargs.get('normal', None)

        # Calculate the angle
        # z_rot = np.arctan2(motion[-1], motion[-2])
        ## type 1 rotation
        # z_rot = motion[-2] * np.pi/2
        ## type 3 rotation
        z_rot = motion[-2] * np.pi
        if self.use_oracle_rotation:
            z_rot = 0.0
        quat = self.convert_yaw_to_quat(z_rot)

        # Calculate the corner of the object
        # obj_pcd = self.env.prev_obs["object_pcd_points"]
        # target_corner = np.array([0, 0, 0.2])
        # distance = np.linalg.norm(obj_pcd - target_corner, axis=-1)
        # corner_idx = np.argmin(distance)
        # location = obj_pcd[corner_idx]

        # self.env._visualize_action(location, motion, self)
        
        # Regress locations and actions
        # self.reset()
        if normal is None:
            # Approach from start location (estimated based the motion param)
            normal = -motion.copy()[:3]
            normal /= np.linalg.norm(normal)
            success = self.move_to_contact(location=location, quat=quat, normal=normal, close_gripper=True)
            # success = self.move_to_from_top(location=location, quat=quat, close_gripper=True)

        # Per-point actions
        else:
            # assert self.collision_check(location)
            success = self.move_to_contact(location=location, quat=quat, normal=normal, close_gripper=True)

        # Post-contact movements
        if success:
            action_repeat = 3
            # for _ in range(action_repeat):  # Action repeat
            #     # delta_pos = motion[:3] * 0.02 * 1.15   # For poke
            #     delta_pos = motion[:3] * 0.02 * 1.2
            #     self.move_by(
            #         target_delta_pos=delta_pos,
            #         num_steps=10,
            #         # step_repeat=10,
            #         # Hack to reproduce the original behavior
            #         num_additional_steps=0,
            #         end_on_reached=True,
            #         grasp=True)
            
            # for _ in range(action_repeat):  # Action repeat
            #     delta_pos = motion[:3] * 0.02 * 1.3
            #     self.move_by(
            #         target_delta_pos=delta_pos,
            #         num_steps=16,
            #         num_additional_steps=0,
            #         end_on_reached=True,
            #         grasp=True)
            
            delta_pos = motion[:3] * 0.02 * 1.3 * action_repeat
            self.move_by(
                target_delta_pos=delta_pos,
                num_steps=15 * action_repeat,
                num_additional_steps=0,
                # max_delta_pos=0.03,
                end_on_reached=True,
                grasp=True)
            
            
        # Reset the robot to move the gripper out of the way
        self.move_by(target_delta_pos=[0, 0, 0.2], grasp=True)
        # self.reset()
        
        # Calculate the step outcome
        self.end_video_record()
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = motion[..., :3] * 0.02 * 1.2 * 3
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return not (states['is_lifted'] or states['is_grasped']) 

@register_primitive("real-pick_n_lift", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class PickNLift(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        self.start_video_record()
        # Calculate the angle
        # z_rot = np.arctan2(motion[-1], motion[-2])
        ## rotation type 1
        # z_rot = motion[-2] * np.pi/2
        ## rotation type 3
        z_rot = motion[-2] * np.pi
        if self.use_oracle_rotation:
            z_rot = 0.0
        quat = self.convert_yaw_to_quat(z_rot)

        # Move to contact
        contact_location = location + np.array([0, 0, -0.03])
        success = self.move_to_from_top(location=contact_location, quat=quat, close_gripper=False)

        # Grasp
        if success:
            # self.close_gripper()
            self.move_to(location=contact_location, quat=quat, grasp=True)
        
        # Lift
        lifted_location = location + np.array([0, 0, 0.15])
        self.move_to(location=lifted_location, quat=quat, grasp=True)

        # Calculate the step outcome
        self.end_video_record()
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.15
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return not (states['is_lifted'] or states['is_grasped']) and (not states['is_close_to_goal'])
        # return False

@register_primitive("real-place", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class Place(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        self.start_video_record()
        # Calculate the angle
        # z_rot = np.arctan2(motion[-1], motion[-2])
        ## rotation type 1
        # z_rot = motion[-2] * np.pi/2
        ## rotation type 3  
        z_rot = motion[-2] * np.pi
        if self.use_oracle_rotation:
            z_rot = 0.0
        quat = self.convert_yaw_to_quat(z_rot)

        # Calculate the target location
        target_location = location + motion[:3] * 0.1
        success = self.move_to_from_top(location=target_location, quat=quat)

        # Drop
        # if success:
        #     self.open_gripper()
        
        # Reset the robot to move the gripper out of the way
        # self.reset()

        # Calculate the step outcome
        self.end_video_record()
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.15
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_lifted'] or states['is_grasped']

@register_primitive("real-move", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Move(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        self.start_video_record()
        # Calculate the angle
        # z_rot = np.arctan2(motion[-1], motion[-2])
        ## rotation type 1
        # z_rot = motion[-2] * np.pi/2
        ## rotation type 3
        z_rot = motion[-2] * np.pi
        if self.use_oracle_rotation:
            z_rot = 0.0
        quat = self.convert_yaw_to_quat(z_rot)

        # Calculate the target location
        current_location = self.env.get_gripper_pose().p
        target_location = current_location + motion[:3] * 0.2
        success = self.move_to(location=target_location, quat=quat)

        # Calculate the step outcome
        self.end_video_record()
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = motion[..., :3] * 0.2
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_lifted'] or states['is_grasped']

