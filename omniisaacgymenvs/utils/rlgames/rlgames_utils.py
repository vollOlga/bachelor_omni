# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Callable

import numpy as np
import torch
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver


class RLGPUAlgoObserver(AlgoObserver):
    """
    Class that acts as an observer for reinforcement learning algorithms,
    capable of tracking and logging environment and algorithm statistics during training.
    Allows us to log stats from the env along with the algorithm running stats.
    """

    def __init__(self):
        """
        Initializes the observer without any initial configuration.
        """
        pass

    def after_init(self, algo):
        """
        Sets up necessary variables after the algorithm initialization.
        
        Parameters:
            algo: The algorithm instance to observe.
        """
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        """
        Processes information from the environment used for logging purposes.
        
        Parameters:
            infos (dict): Dictionary of information returned from the environment.
            done_indices: Indices indicating which environments are done.
        """

        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if (
                        isinstance(v, float)
                        or isinstance(v, int)
                        or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                    ):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        """
        Clears statistical counters after each training iteration.
        """
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        """
        Aggregates and logs statistics after each epoch to the tensorboard writer.
        
        Parameters:
            frame: The current training frame.
            epoch_num: The current epoch number.
            total_time: Total elapsed training time.
        """
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    """
    Vectorized environment wrapper for GPU-based reinforcement learning environments.
    """
    def __init__(self, config_name, num_actors, **kwargs):
        """
        Initializes the environment by loading configuration based on the provided name.
        
        Parameters:
            config_name (str): Name of the configuration for environment creation.
            num_actors (int): Number of simultaneous agents or actors in the environment.
            kwargs: Additional keyword arguments for environment creation.
        """
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):
        """
        Takes an action in the environment and returns the result.
        
        Parameters:
            action: The action to be executed in the environment.
        
        Returns:
            Tuple containing the new state, reward, done flag, and additional info.
        """
        return self.env.step(action)

    def reset(self):
        """
        Resets the environment to its initial state.
        
        Returns:
            Initial state of the environment.
        """
        return self.env.reset()

    def get_number_of_agents(self):
        """
        Retrieves the number of agents active in the environment.
        
        Returns:
            Integer representing the number of agents.
        """
        return self.env.get_number_of_agents()

    def get_env_info(self):
        """
        Collects and returns information about the environment such as action and observation spaces.
        
        Returns:
            A dictionary containing details about the environment's spaces and other configurations.
        """
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space

        if self.env.num_states > 0:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info
