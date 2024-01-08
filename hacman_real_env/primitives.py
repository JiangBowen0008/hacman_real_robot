
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation

from hacman.utils.primitive_utils import GroundingTypes, Primitive, register_primitive

class RealEnvPrimitive(Primitive):
    def reset(self):
        self.env.robot.reset()
    
    def get_step_return(self, info):
        return self.env.get_step_return(info)

    def move_to(self, location, quat, grasp=True, **kwargs):
        self.env.robot.move_to(location, target_quat=quat, grasp=grasp, **kwargs)
    
    def close_gripper(self):
        self.env.robot.move_by(grasp=True, num_steps=20, num_additional_steps=0)
        self.env.unwrapped.grasped = True
    
    def open_gripper(self):
        self.env.robot.move_by(grasp=False, num_steps=20, num_additional_steps=0)
        self.env.unwrapped.grasped = False

    def move_to_contact(self, location, normal, quat):
        # Move to precontact
        precontact = location + normal * 0.04
        self.move_to(location=precontact, quat=quat)

        # Move to location
        self.move_to(location=location, quat=quat)
        return True
    
    def move_to_from_top(self, location, quat, close_gripper=True):
        # Move to precontact
        precontact = location + np.array([0, 0, 0.15])
        self.move_to(location=precontact, quat=quat, grasp=close_gripper)

        # Move to location
        self.move_to(location=location, quat=quat, grasp=close_gripper)
        return True
    
    def convert_yaw_to_quat(self, yaw):
        quat = Rotation.from_euler('xyz', [np.pi, 0, yaw]).as_quat()
        return quat

@register_primitive("real-poke", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Poke(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        # Save original object pose
        # obj_pose = self.env.get_object_pose()
        normal = kwargs.get('normal', None)

        # Calculate the angle
        z_rot = np.arctan2(motion[-1], motion[-2])
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
        self.reset()
        if normal is None:
            # Approach from start location (should be top)
            success = self.move_to_from_top(location=location, quat=quat, close_gripper=True)

        # Per-point actions
        else:
            # assert self.collision_check(location)
            success = self.move_to_contact(location=location, quat=quat, normal=normal)

        # Post-contact movements
        if success:
            target_location = location + motion[:3] * 0.1
            self.move_to(location=target_location, quat=quat)
            
    
        # Reset the robot to move the gripper out of the way
        self.reset()
        
        # Calculate the step outcome
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = motion[..., :3] * 0.1
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']

@register_primitive("real-pick_n_lift", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class PickNLift(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        # Calculate the angle
        z_rot = np.arctan2(motion[-1], motion[-2])
        if self.use_oracle_rotation:
            z_rot = 0.0
        quat = self.convert_yaw_to_quat(z_rot)

        # Calculate the position
        obj_pcd = self.env.prev_obs["object_pcd_points"]
        obj_center = obj_pcd.mean(axis=0)
        location = obj_center

        # Move to precontact
        success = self.move_to_from_top(location=location, quat=quat, close_gripper=False)

        # Grasp
        if success:
            self.close_gripper()
        
        # Lift
        lifted_location = location + np.array([0, 0, 0.15])
        self.move_to(location=lifted_location, quat=quat)

        # Calculate the step outcome
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.15
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']

@register_primitive("real-place", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class Place(RealEnvPrimitive):
    def execute(self, location, motion, **kwargs):
        # Calculate the angle
        z_rot = np.arctan2(motion[-1], motion[-2])
        if self.use_oracle_rotation:
            z_rot = 0.0
        quat = self.convert_yaw_to_quat(z_rot)

        # Calculate the target location
        # TODO
        target_location = location + np.array([0, 0, 0.15])
        success = self.move_to_from_top(location=target_location, quat=quat)

        # Drop
        if success:
            self.open_gripper()
        
        # Reset the robot to move the gripper out of the way
        self.reset()

        # Calculate the step outcome
        info = {"poke_success": success} 
        obs, reward, done, info = self.get_step_return(info)
        all_rewards = [[reward,],]
        return obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.15
        return motion_
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']

