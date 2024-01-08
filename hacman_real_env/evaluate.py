from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import numpy as np
from copy import deepcopy
import open3d as o3d

import gym
from gym import Wrapper, Env
from hacman.envs.env_wrappers import HACManActionWrapper, FlatActionWrapper, RegressedActionWrapper

from real_env import RealEnv
import primitives

class EnvwithLocationPolicy(Wrapper):
    def __init__(self,
                 env: Env,
                 observation_space: Optional[gym.spaces.Space] = None,
                 action_space: Optional[gym.spaces.Space] = None,
                 location_model: Callable[[Dict], Dict] = None):
        
        self.location_model = location_model
        super().__init__(env)
        if observation_space is not None:
            observation_space = self.update_observation_space(deepcopy(observation_space))
    
    def update_observation_space(self, obs_space: gym.spaces.Dict):
        """
        Updates the observation space with additional keys for action scores.
        """
        space_dict = obs_space.spaces
        # Add the HACMan spaces
        obj_pcd_size = extra_spaces['object_pcd_points'].shape[0]
        bg_pcd_size = extra_spaces['bg_pcd_points'].shape[0]
        pcd_size = obj_pcd_size + bg_pcd_size
        extra_spaces = {
            "action_location_score": gym.spaces.Box(-np.inf, np.inf, (pcd_size,)),
            "action_params": gym.spaces.Box(-np.inf, np.inf, (pcd_size, 3,)),}
        space_dict.update(extra_spaces)

        return gym.spaces.Dict(space_dict)
    
    def process_obs(self, obs: Dict[str, np.ndarray]):
        location_infos = self.location_model.get_action(obs)
        obs.update(location_infos)
        assert self.env.set_prev_obs(obs)
        return obs
    
    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        obs = self.process_obs(obs) # New
        return obs, rew, done, info
    
    def reset(self):
        obs = self.env.reset()
        self.process_obs(obs) # New
        return obs

def test_env():
    from hacman.algos.location_policy import RandomLocation

    primitives = [
        # "real-poke",
        "real-pick_n_lift",
        "real-place"
    ]
    env = RealEnv()
    env = HACManActionWrapper(
        env, primitives=primitives,
        use_oracle_rotation=True
        )
    env = EnvwithLocationPolicy(env, location_model=RandomLocation())

    for _ in range(10):
        obs = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            # action *= 0
            obs, reward, done, info = env.step(
                action, 
                debug=True
                )
            print(reward)
            if done:
                break

def visualize_obs(obs, action=None):
    object_pcd = obs['object_pcd_points']
    object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_pcd))
    object_pcd.paint_uniform_color([1, 0.706, 0])
    bg_pcd = obs['background_pcd_points']
    bg_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg_pcd))
    bg_pcd.paint_uniform_color([0, 0.651, 0.929])
    
    o3d.visualization.draw_geometries([object_pcd, bg_pcd])

if __name__ == "__main__":
    test_env()