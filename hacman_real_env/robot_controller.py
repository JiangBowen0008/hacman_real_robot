"""
OSC Controller adapted from deoxys/examples/osc_control.py
"""

import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

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
                 visualizer=False):
        self.robot_interface = FrankaInterface(
            config_root + f"/{interface_cfg}", use_visualizer=visualizer)
        self.controller_type = controller_type
        self.controller_cfg = get_default_controller_config(controller_type)
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

    def move_to(self, 
                target_pos,
                target_quat,
                num_steps=80,
                num_additional_steps=40):
        self._osc_move(
            (target_pos, target_quat),
            num_steps,
        )
        self._osc_move(
            (target_pos, target_quat),
            num_additional_steps,
        )
        print(f'Target_quat: {target_quat}, target_pos: {target_pos}')

    def move_by(self, 
                target_delta_pos=np.zeros(3), 
                target_delta_axis_angle=np.zeros(3),
                num_steps=80,
                num_additional_steps=40):
        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

        current_ee_pose = self.robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos

        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle

        logger.info(f"Before conversion {target_axis_angle}")
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        logger.info(f"After conversion {target_axis_angle}")
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

        self.move_to(target_pos, target_quat, num_steps, num_additional_steps)
    
    def _osc_move(self, target_pose, num_steps):
        target_pos, target_quat = target_pose
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        current_rot, current_pos = self.robot_interface.last_eef_rot_and_pos

        for _ in range(num_steps):
            current_pose = self.robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            current_axis_angle = transform_utils.quat2axisangle(current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
            logger.info(f"Axis angle action {action_axis_angle.tolist()}")
            # print(np.round(action, 2))
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,)
        return action

    @property
    def eef_pose(self):
        return self.robot_interface.last_eef_pose
    
    @property
    def eef_rot_and_pos(self):
        return self.robot_interface.last_eef_rot_and_pos

    @property
    def joint_positions(self):
        return self.robot_interface.last_q

'''
Test program
'''
def estimate_tag_pose(finger_pose):
    """
    Estimate the tag pose given the gripper pose by applying the gripper-to-tag transformation.

    Args:
        finger_pose (eef_pose): 4x4 transformation matrix from gripper to robot base
    Returns:
        hand_pose: 4x4 transformation matrix from hand to robot base
        tag_pose: 4x4 transformation matrix from tag to robot base
    """
    from scipy.spatial.transform import Rotation

    # Estimate the hand pose
    # finger_to_hand obtained from the product manual: 
    # [https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf]
    finger_to_hand = np.array([
        [0.707,  0.707, 0, 0],
        [-0.707, 0.707, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1],
    ])
    finger_to_hand = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.1034],
        [0, 0, 0, 1],
    ])
    hand_to_finger = np.linalg.inv(finger_to_hand)
    print("hand to finger", hand_to_finger)
    hand_pose = np.dot(finger_pose, hand_to_finger)

    t_tag_to_hand = np.array([0.048914, 0.0275, 0.00753])
    # R_tag_to_hand = Rotation.from_quat([0.5, -0.5, 0.5, -0.5])
    R_tag_to_hand = Rotation.from_quat([0, 0, 0, 1])
    tag_to_hand = np.eye(4)
    tag_to_hand[:3, :3] = R_tag_to_hand.as_matrix()
    tag_to_hand[:3, 3] = t_tag_to_hand

    tag_pose = np.dot(hand_pose, tag_to_hand)
    
    return hand_pose, tag_pose

if __name__ == "__main__":
    controller = FrankaOSCController(visualizer=False)
    # controller.reset()
    controller.move_by(np.array([0, 0, 0]), np.array([0, 0, 0]), num_steps=10, num_additional_steps=10)
    # init_pos = np.array([0.52560095, -0.28889169, 0.29223859])
    # # init_quat = np.array([1, 0, 0, 0])
    # init_quat = np.array([9.9994910e-01,  1.0066551e-02, -3.4788260e-04,  5.9928966e-04])
    # controller.move_to(init_pos, init_quat)
    # controller.reset()
    logger.debug("Final movement finished")
    # print(controller.robot_interface.last_q)
    eef_pose = controller.eef_pose
    # print(eef_pose)
    hand_pose, tag_pose = estimate_tag_pose(eef_pose)
    print(f"eef pos: {eef_pose[:3, 3]}")
    print(f"hand pos: {hand_pose[:3, 3]}")
    print(f"Tag pos: {tag_pose[:3, 3]}")

    # Visualize the poses

    # print(controller.robot_interface.last_eef_quat_and_pos)