from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import os
import numpy as np
from copy import deepcopy
import open3d as o3d
from functools import partial
import wandb

import gym
from gym import Wrapper, Env
from gym.wrappers.time_limit import TimeLimit
from hacman.envs.env_wrappers import HACManActionWrapper, FlatActionWrapper, RegressedActionWrapper
from hacman.envs.vec_env_wrappers import WandbPointCloudRecorder, PCDDummyVecEnv
from hacman.algos.location_policy import LocationPolicyWithArgmaxQ
from hacman.envs.vec_env_wrappers import VecEnvwithLocationPolicy
from stable_baselines3.common.env_util import make_vec_env as make_sb3_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from hacman_real_env.real_env import RealEnv
import hacman_real_env.primitives

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
        # Retrieve the first element of the dict
        location_infos = {k: v[0] for k, v in location_infos.items()}
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

def test_random_env():
    from hacman.algos.location_policy import RandomLocation
    object_name = "pink_box"
    primitives = [
        "real-poke",
        # "real-pick_n_lift",
        # "real-place"
    ]
    
    env = RealEnv(object_name=object_name)
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

def test_ckpt(ckpt_path, object_name, n_evals = 2, env_args={}):
    from hacman.algos import MultiTD3
    import datetime
    max_steps = 20
    # record_video = True

    datetime_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"RealRobot-{datetime_str}-{object_name}"
    run_id = wandb.util.generate_id()   
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)
    run = wandb.init(
        name=run_name, config={"object_name": object_name}, id=run_id, 
        dir=run_dir, sync_tensorboard=False)
    
    env = make_env(
        save_dir=run_dir,
        object_name=object_name,
        record_video=True,
        seed=0,
        max_episode_steps=max_steps,
        **env_args,
    )

    # Load the model
    model = MultiTD3.load(path=ckpt_path, env=env)
    model.policy.eval()
    env.location_model.load_model(model)

    # Run the model
    
    from tqdm import tqdm
    pbar = tqdm(total=n_evals)

    from hacman.sb3_utils.evaluation import evaluate_policy
    mean_reward, std_reward, succ, verbose_buffer, prim_perc = evaluate_policy(
            model, env, n_eval_episodes=n_evals, deterministic=True,
            save_path=os.path.join(run_dir, f'obs_list_{0}.pkl'), pbar=pbar,
            return_success_rate=True, verbose=True, return_episode_primitiv_perc=True)

    uncertainty = 1.96 * np.sqrt(succ * (1 - succ) / n_evals)
    print(f"succ={succ:.3f} +/- {uncertainty:.3f}. mean_reward={mean_reward:.2f} +/- {std_reward}")
    print(f"primitive usage:")
    for p, v in prim_perc.items():
        print(f"{p}: {v:.1f}%")

def make_env(save_dir,
             object_name,
             record_video=False,
             seed=0,
             max_episode_steps=10,
             **env_args,
             ):
    
    primitives = [
        "real-poke",
        "real-pick_n_lift",
        "real-place",
        "real-move",
        "real-open_gripper"
    ]

    # env kewargs   
    env_kwargs = env_args
    env_kwargs.update({
        "object_name": object_name,
        "record_video": record_video,
        "save_dir": save_dir,
    })
    vecenv_kwargs = {
        "primitives": primitives,
        "background_pcd_size": 1000,
        "object_pcd_size": 400,
        "voxel_downsample_size": 0.01,
        "skip_processing": True,
    }

    # Define the env wrappers
    wrappers = [
        partial(HACManActionWrapper, primitives=primitives),
        partial(TimeLimit, max_episode_steps=max_episode_steps),]
    def wrapper_class(env, **kwargs):
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    
    vec_env_cls = PCDDummyVecEnv
    venv = make_sb3_vec_env(
        RealEnv, n_envs=1,
        seed=seed,
        vec_env_cls=vec_env_cls,
        env_kwargs=env_kwargs,
        wrapper_class=wrapper_class,
        vec_env_kwargs=vecenv_kwargs,
    )
    location_model = LocationPolicyWithArgmaxQ(
        temperature=0.05,
        egreedy=0.0,
        # deterministic=True,
        )
    venv = VecMonitor(venv)
    venv = VecEnvwithLocationPolicy(venv, location_model=location_model)
    venv = WandbPointCloudRecorder(venv,
                                  real_robot=True, save_plotly=True,
                                  foldername=save_dir, log_plotly_once=False)
    
    return venv


def visualize_obs(obs, action=None):
    object_pcd = obs['object_pcd_points']
    object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_pcd))
    object_pcd.paint_uniform_color([1, 0.706, 0])
    bg_pcd = obs['background_pcd_points']
    bg_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg_pcd))
    bg_pcd.paint_uniform_color([0, 0.651, 0.929])
    
    o3d.visualization.draw_geometries([object_pcd, bg_pcd])

if __name__ == "__main__":
    # test_random_env()
    test_ckpt(
        # "ckpts/Exp2142-0-2-finetune/rl_model_latest.zip",
        "ckpts/Exp2142-0-2-realworld-finetune_panda_finger_low_friction/rl_model_latest.zip",
        # "ckpts/Exp2142-0-2-finetune-festo/rl_model_latest.zip",
        "pink_box",
        n_evals=30,
        env_args=dict(
            # allow_manual_registration=True,
            # allow_full_pcd=True,
            # symmetric_object=True,
        )
    )

    # test_ckpt(
    #     "ckpts/Exp2142-0-2-finetune/rl_model_latest.zip",
    #     # "ckpts/Exp2142-0-1-double_bin_all_6d/Exp2142-0-1-double_bin_all_6d/rl_model_latest.zip",
    #     # "ckpts/Exp2142-0-2-finetune-festo/rl_model_latest.zip",
    #     "white_box",
    #     n_evals=40,
    #     env_args=dict(
    #         # allow_manual_registration=True,
    #         allow_full_pcd=True,
    #         # symmetric_object=True,
    #     )
    # )