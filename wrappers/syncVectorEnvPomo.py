from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
from gym.vector.utils import concatenate, create_empty_array, iterate
from gym.vector.vector_env import VectorEnv
from envs.container_vector_env import _MODEL_CACHE,get_discriminator_reward,similarity_reward
from discriminator.config import Config

import pdb


__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata
        self.n_traj = self.envs[0].n_traj
        self.dim = self.envs[0].dim
        self.device = self.envs[0].device
        

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs, self.n_traj), dtype=np.float64)
        self._dones = np.zeros((self.num_envs, self.n_traj), dtype=np.bool_)
        self._actions = None
        

    def seed(self, seed=None):
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._dones[:] = False
        observations = []
        data_list = []
        for env, single_seed in zip(self.envs, seed):

            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            if return_info == True:
                kwargs["return_info"] = return_info

            if not return_info:
                observation = env.reset(**kwargs)
                observations.append(observation)
            else:
                observation, data = env.reset(**kwargs)
                observations.append(observation)
                data_list.append(data)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        if not return_info:
            return deepcopy(self.observations) if self.copy else self.observations
        else:
            return (deepcopy(self.observations) if self.copy else self.observations), data_list

    def step_async(self, actions):
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
    
        observations, infos = [], []
        dest_node = np.zeros((self.num_envs, self.n_traj, self.dim), dtype=np.float32)
        prev_node = np.zeros((self.num_envs, self.n_traj, self.dim), dtype=np.float32)

        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            
            observation, _ , self._dones[i], info  = env.step(action) # 
            dest_node[i] = env.dest_node
            prev_node[i] =  env.prev_node
            num_steps = self.envs[i].num_steps
          
            observations.append(observation)
            infos.append(info)
        
        """ 
        余弦相似度: [0, 1] 值越大 相似度越高
        判别器奖励: [0, 1] 值越大 奖励越高

        缩放奖励到[-1, 0]
        """
       
        # [512,20]
        if num_steps - 1 != 0:
            discriminator_reward = get_discriminator_reward(dest_node,prev_node, Config.input_dim, Config.hidden_dim, self.device) - 1 
            sim_reward = similarity_reward(dest_node,prev_node) - 1 
          
            self._rewards = (sim_reward + discriminator_reward)/ 2 

        else:
            self._rewards = np.zeros((self.num_envs, self.n_traj), dtype=np.float64)
       
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
      
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        else:
            return True
