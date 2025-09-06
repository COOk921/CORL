# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import time

from scipy.stats import kendalltau, spearmanr
import logging
from tqdm import tqdm
import gym

import numpy as np
import torch

import time
import warnings
warnings.filterwarnings("ignore")

from utils import calculation_metrics
from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics

import pdb

if __name__ == "__main__":
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = './runs/container-v0__ppo_or__2025-09-06_14_25/ckpt/20.pt'
    agent = Agent(device=device, name='container').to(device)
    agent.load_state_dict(torch.load(ckpt_path))
    env_id = 'container-v0'
    num_steps = 21
    num_envs = 50
    n_traj = 50

    env_entry_point = 'envs.container_vector_env:ContainerVectorEnv'
    seed = 0

    gym.envs.register(
        id=env_id,
        entry_point=env_entry_point,
    )

    def make_env(env_id, seed, cfg={}):
        def thunk():
            env = gym.make(env_id, **cfg)
            env = RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = SyncVectorEnv(  [make_env(env_id, seed, dict(n_traj=n_traj)) for i in range(num_envs) ]  )

    
    trajectories = []
    agent.eval()
    obs = envs.reset()
    for step in range(0, num_steps):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logits = agent(obs)
        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())

    # resulting_traj = np.array(trajectories)[:,0,0]
    # print(trajectories)
   
    # 
    
    """
    trajectories : Step,(env,traj)
    episode_returns : (env,traj)
    test_obs['observations'] : (env, node,obs_dim)

    """
    
    resulting_traj = np.array(trajectories).transpose(1, 2, 0)  # (env,traj,step)
    # rehandle_rate = calculation_metrics(resulting_traj, test_obs['observations'][0])
   
    
    target  = np.concatenate([  np.arange(resulting_traj.shape[-1] -1), [0]  ])
    target = np.tile(target, (resulting_traj.shape[0], resulting_traj.shape[1], 1))  #(env,traj,step)

    # tau, _ = kendalltau(target, resulting_traj)
    tau_mean = 0
    for i in range(resulting_traj.shape[0]):
        max_traj = -1
        for j in range(resulting_traj.shape[1]):
            tau, _ = kendalltau(target[i,j], resulting_traj[i,j])
            max_traj = max(max_traj, tau)
        tau_mean += max_traj
    tau_mean /= resulting_traj.shape[0]
    print("tau_mean:", tau_mean)

    rho_mean = 0
    for i in range(resulting_traj.shape[0]):
        max_traj = -1
        for j in range(resulting_traj.shape[1]):
            rho, _ = spearmanr(target[i,j], resulting_traj[i,j])
            max_traj = max(max_traj, rho)
        rho_mean += max_traj
    rho_mean /= resulting_traj.shape[0]

    print("rho_mean:", rho_mean)
    # avg_episodic_return = np.mean(np.mean(episode_returns, axis=1))
    # max_episodic_return = np.mean(np.max(episode_returns, axis=1))
    # avg_episodic_length = np.mean(episode_lengths)
    pdb.set_trace()
    # logging.info(
    #     "--------------------------------------------"
    #     f"[test] episodic_return={max_episodic_return}\n"
    #     f"avg_episodic_return={avg_episodic_return}\n"
    #     f"max_episodic_return={max_episodic_return}\n"
    #     f"avg_episodic_length={avg_episodic_length}\n"
    #     f"rehandle_rate={rehandle_rate}\n"
    #     f"tau={tau}\n"
    #     f"rho={rho_mean}\n"
    #     "--------------------------------------------"
    # )
    # logging.info("")


    # writer.add_scalar("test/episodic_return_mean", avg_episodic_return, global_step)
    # writer.add_scalar("test/episodic_return_max", max_episodic_return, global_step)
    # writer.add_scalar("test/episodic_length", avg_episodic_length, global_step)

    # writer.add_scalar("test/rehandle_rate", rehandle_rate, global_step)
    # writer.add_scalar("test/tau", tau, global_step)
    # writer.add_scalar("test/rho", rho_mean, global_step)


    # envs.close()
    # writer.close()
