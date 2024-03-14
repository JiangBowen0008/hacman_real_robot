"""
OSC Controller adapted from deoxys/examples/osc_control.py
"""

import argparse
import pickle
import threading
import time
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_project_logger

logger = get_project_logger()

def compute_errors(pose_1, pose_2):

    pose_a = (
        pose_1[:3]
        + transform_utils.quat2axisangle(np.array(pose_1[3:]).flatten()).tolist()
    )
    pose_b = (
        pose_2[:3]
        + transform_utils.quat2axisangle(np.array(pose_2[3:]).flatten()).tolist()
    )
    return np.abs(np.array(pose_a) - np.array(pose_b))

class FrankaOSCController():
    def __init__(self,
                 interface_cfg="charmander.yml",
                 controller_type="OSC_POSE",
                 controller_cfg="hacman_real_env/robot_controller/tuned-osc-yaw-controller.yml",
                 controller_offset=np.eye(4),
                 frame_transform=np.eye(4),
                 tip_offset=np.array([0, 0, 0.0766]),
                 visualizer=False):
        self.robot_interface = FrankaInterface(
            config_root + f"/{interface_cfg}",
            control_freq=20, 
            use_visualizer=visualizer)
        
        # Load controller config
        self.controller_type = controller_type
        if controller_cfg is not None:
            controller_cfg = YamlConfig(controller_cfg).as_easydict()
            verify_controller_config(controller_cfg)
        else:
            controller_cfg = get_default_controller_config(controller_type)
        self.controller_cfg = controller_cfg

        # Set the offset (for control osc pose in a different frame)
        self.frame_transform = frame_transform
        self.controller_offset = controller_offset
        self.tip_transform = np.eye(4)
        self.tip_transform[:3, 3:] = tip_offset.reshape(3, 1)

        self.reset_joint_positions = [
            -0.5493463,
            0.18639661,
            0.04967389,
            -1.92004654,
            -0.01182675,
            2.10698001,
            0.27106661]
    
    def reset(self, joint_positions=None):
        joint_positions = joint_positions if joint_positions is not None else self.reset_joint_positions
        reset_joints_to(self.robot_interface, joint_positions)
        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

    def move_to(self, 
                target_pos,
                target_quat=None,
                target_delta_axis_angle=None,
                grasp=True,
                num_steps=50,
                num_additional_steps=10,
                pos_tolerance=0.004,
                rot_tolerance=0.05,
                end_on_reached=True,
                **kwargs):
        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)
        
        # Compute target rotation
        if target_quat is not None:
            pass
        elif target_delta_axis_angle is not None:
            current_axis_angle = self.eef_axis_angle
            target_axis_angle = current_axis_angle + target_delta_axis_angle
            target_quat = transform_utils.axisangle2quat(target_axis_angle)
        else:
            raise ValueError("Either target_quat or target_delta_axis_angle must be specified")        
        
        target_pos = np.array(target_pos).reshape(3, 1)
        try:
            pos_diff, angle_diff = self._osc_move(
                (target_pos, target_quat),
                num_steps,
                grasp=grasp,
                pos_tolerance=pos_tolerance,
                rot_tolerance=rot_tolerance,
                end_on_reached=end_on_reached,
                **kwargs
            )
            if num_additional_steps > 0:
                pos_diff, angle_diff = self._osc_move(
                    (target_pos, target_quat),
                    num_additional_steps,
                    grasp=grasp,
                    end_on_reached=False,
                    **kwargs
                )
            
                if np.linalg.norm(pos_diff) > pos_tolerance:
                    logger.warn(f"Position not reached. Error: {pos_diff}")
                if np.linalg.norm(angle_diff) > rot_tolerance:
                    logger.warn(f"Rotation not reached. Error: {angle_diff}")
        except:
            logger.error("Error while moving")

    def move_by(self, 
                target_delta_pos=np.zeros(3), 
                target_delta_axis_angle=np.zeros(3),
                grasp=True,
                num_steps=50,
                num_additional_steps=10,
                pos_tolerance=0.004,
                rot_tolerance=0.05,
                end_on_reached=True,
                **kwargs):
        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

        current_ee_pose = self.eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos

        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle

        # logger.info(f"Before conversion {target_axis_angle}")
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        # logger.info(f"After conversion {target_axis_angle}")
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

        self.move_to(target_pos, target_quat, target_delta_axis_angle=None, 
                     grasp=grasp, num_steps=num_steps, 
                     num_additional_steps=num_additional_steps,
                     pos_tolerance=pos_tolerance,
                     rot_tolerance=rot_tolerance,
                     end_on_reached=end_on_reached,
                     **kwargs)
    
    def _osc_move(
            self, target_pose, num_steps,
            grasp=True,
            max_delta_pos=0.055,
            pos_tolerance=0.004,
            rot_tolerance=0.05,
            end_on_reached=True,
            step_repeat=1,  # A hack to reproduce the control in HACMan
            ):
        # Apply the offset transform
        target_pos, target_quat = self._apply_offset(target_pose)
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        grasp = {
            None: 0.0,
            True: 1.0,
            False: -1.0
        }[grasp]

        step_count = 0
        while step_count < num_steps:
            current_pose = self.robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            current_axis_angle = transform_utils.quat2axisangle(current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

            reached_pos, reached_rot = self.check_reached(
                (target_pos, target_quat), pos_tolerance, rot_tolerance)
            if end_on_reached:
                if reached_pos and reached_rot:
                    break

            if max_delta_pos is not None:
                target_pos, target_quat = self._apply_offset(target_pose)
                delta_pos = target_pos - current_pos
                if np.linalg.norm(delta_pos) > max_delta_pos:
                    target_pos = current_pos + (delta_pos / np.linalg.norm(delta_pos)) * max_delta_pos
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            # logger.info(f"Action pos {action_pos.tolist()}. Current pos {current_pos.flatten().tolist()}. Target pos {target_pos.flatten().tolist()}")
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [grasp]
            # logger.info(f"Axis angle action {action_axis_angle.tolist()}")
            # print(np.round(action, 2))
            for _ in range(step_repeat):
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=action,
                    controller_cfg=self.controller_cfg,)
                step_count += 1
        return np.linalg.norm(delta_pos), np.linalg.norm(axis_angle_diff)
    
    def _apply_offset(self, target_pose):
        target_pos, target_quat = target_pose
        target_mat = transform_utils.pose2mat((target_pos.flatten(), target_quat))

        target_mat = np.linalg.inv(self.frame_transform) @ target_mat   # Apply the scene transform
        target_mat = np.linalg.inv(self.controller_offset) @ target_mat # Apply the offset transform
        target_mat = target_mat @ np.linalg.inv(self.tip_transform)      # Apply the tip transform
        
        # target_pos = target_pos.reshape(3, 1)
        # target_pos = inverse_transform[:3, :3] @ target_pos + inverse_transform[:3, 3:]

        # target_rot = Rotation.from_quat(target_quat)
        # target_rot = inverse_transform[:3, :3] @ target_rot.as_matrix()
        # target_quat = transform_utils.mat2quat(target_rot)
        target_pos = target_mat[:3, 3:]
        target_quat = transform_utils.mat2quat(target_mat[:3, :3])
        return target_pos, target_quat
    
    def check_reached(self, target_pose, pos_tolerance=0.005, rot_tolerance=0.1):
        target_pos, target_quat = target_pose

        current_pose = self.robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

        norm_pos_diff = np.linalg.norm(target_pos - current_pos)
        norm_angle_diff = np.linalg.norm(axis_angle_diff)
        return norm_pos_diff < pos_tolerance, norm_angle_diff < rot_tolerance

    
    def update_controller_config(self, controller_cfg):
        self.controller_cfg = controller_cfg
    
    @property
    def eef_axis_angle(self):
        # rot = self.eef_rot_and_pos[0]
        # # Apply the inverse offset transform
        # rot = np.linalg.inv(self.frame_transform[:3, :3]) @ rot
        rot = self.robot_interface.last_eef_rot_and_pos[0]
        quat = transform_utils.mat2quat(rot)
        return transform_utils.quat2axisangle(quat)

    @property
    def eef_pose(self):
        # pose of the gripper tip
        last_eef_pose = self.robot_interface.last_eef_pose
        last_eef_pose = last_eef_pose @ self.tip_transform      # Apply the tip transform
        last_eef_pose = self.frame_transform @ last_eef_pose    # Apply the scene transform
        last_eef_pose = self.controller_offset @ last_eef_pose  # Apply the offset transform
        return last_eef_pose
    
    @property
    def eef_base_pose(self):
        # pose of the gripper tip base
        last_eef_pose = self.robot_interface.last_eef_pose
        last_eef_pose = self.frame_transform @ last_eef_pose    # Apply the scene transform
        return last_eef_pose
    
    @property
    def eef_rot_and_pos(self):
        pose = self.eef_pose
        rot, pos = pose[:3, :3], pose[:3, 3:]
        return rot, pos

    @property
    def joint_positions(self):
        return self.robot_interface.last_q
    
    @property
    def is_grasped(self):
        gripper_q = self.robot_interface.last_gripper_q
        last_gripper_action = self.robot_interface.last_gripper_action
        # return (gripper_q > 0.01) and (last_gripper_action >= 0.0)
        return (last_gripper_action >= 0.0)

'''
Test program
'''

if __name__ == "__main__":
    controller = FrankaOSCController(
        controller_type="OSC_POSE",
        visualizer=False)
    # controller.reset()
    # controller.move_by(np.array([0, 0, -0.01]), np.array([0, 0, 0]), num_steps=40, num_additional_steps=10)
    # initial_joint_positions = [
    #     -0.55118707,
    #     -0.2420445,
    #     0.01447328,
    #     -2.28358781,
    #     -0.0136721,
    #     2.03815885,
    #     0.25261351]
    # reset_joint_positions = [
    #     0.09162008114028396,
    #     -0.19826458111314524,
    #     -0.01990020486871322,
    #     -2.4732269941140346,
    #     -0.01307073642274261,
    #     2.30396583422025,
    #     0.8480939705504309,
    # ]
    # controller.reset(joint_positions=reset_joint_positions)
    # controller.move_to(np.array([0.45, -0.3, 0.25]), 
    #                    target_quat=np.array([ 0.7071068, -0.7071068, 0, 0 ]),
    #                    target_delta_axis_angle=np.array([0, 0, 0]),
    #                    grasp=False,
    #                    num_steps=40, num_additional_steps=10)
    controller.move_by(np.array([0, 0, -0.0]),
                       np.array([0, 0, 0]),
                       grasp=True,
                       num_steps=40, num_additional_steps=10)
    logger.debug("Final movement finished")
    # print(controller.robot_interface.last_q)
    eef_pose = controller.eef_pose
    joint_positions = controller.robot_interface.last_q
    # print(eef_pose)
    # hand_pose, tag_pose = estimate_tag_pose(eef_pose)
    print(f"eef pos: {eef_pose[:3, 3]}")
    # print(f"hand pos: {hand_pose[:3, 3]}")
    # print(f"Tag pos: {tag_pose[:3, 3]}")

    print(f"Joint positions: {joint_positions}")
    # Visualize the poses

    # print(controller.robot_interface.last_eef_quat_and_pos)