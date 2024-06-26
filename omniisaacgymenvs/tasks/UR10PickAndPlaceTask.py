from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.robots.articulations.UR10.ur10_view import UR10View
from omniisaacgymenvs.robots.articulations.UR10.UR10_pap import UR10
from abc import abstractmethod
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.shared.pick_and_place_new import PickAndPlaceTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
import pandas as pd
import random

import numpy as np
import torch
import math

class UR10PickAndPlaceTask(PickAndPlaceTask):
    def __init__(
        self,
        name: str,
        sim_config: SimConfig,
        env: VecEnvBase,
        offset=None
    ) -> None:
        '''
        Initializes the UR10ReacherTask instance with the necessary configuration.
        Parameters:
            name (str): The name of the task.
            sim_config (SimConfig): The simulation configuration object containing both simulation and task-specific configurations.
            env (VecEnvBase): The environment in which the task is operating.
            offset (Optional): Additional parameter that can be used to adjust configurations or initial states.
        Return:
            None
        '''
        self._device = "cuda:0" 
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [full]")
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "full": 40,
        }

        self.object_scale = torch.tensor([1.0] * 3)
        self.goal_scale = torch.tensor([2.0] * 3)

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 6
        self._num_states = 0

        self.obs_buf = None  # Initialize obs_buf to None

        self._num_envs = self._task_cfg["env"]["numEnvs"]

        pi = math.pi
        if self._task_cfg['safety']['enabled']:
            self._dof_limits = torch.tensor([[
                [np.deg2rad(-135), np.deg2rad(135)],
                [np.deg2rad(-180), np.deg2rad(-60)],
                [np.deg2rad(0), np.deg2rad(180)],
                [np.deg2rad(-180), np.deg2rad(0)],
                [np.deg2rad(-180), np.deg2rad(0)],
                [np.deg2rad(-180), np.deg2rad(180)],
            ]], dtype=torch.float32, device=self._cfg["sim_device"])
        else:
            self._dof_limits = torch.tensor([[
                [-2*pi, 2*pi],
                [-pi + pi/8, 0 - pi/8],
                [-pi + pi/8, pi - pi/8],
                [-pi, 0],
                [-pi, pi],
                [-2*pi, 2*pi],
            ]], dtype=torch.float32, device=self._cfg["sim_device"])

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float32, device=self._device).repeat((self._num_envs, 1))

        PickAndPlaceTask.__init__(self, name=name, env=env)

        sim2real_config = self._task_cfg['sim2real']
        if sim2real_config['enabled'] and self.test and self.num_envs == 1:
            self.act_moving_average /= 5
            self.real_world_ur10 = RealWorldUR10(
                sim2real_config['fail_quietely'],
                sim2real_config['verbose']
            )
        return

    def get_num_dof(self):
        '''
        Retrieves the number of degrees of freedom (DOF) for the robot arm.
        Parameters:
            None
        Return:
            int: The number of degrees of freedom of the robot arm.
        '''
        print(f'the number of degrees of freedom (DOF) for the robot arm: {self._arms.num_dof}')
        return self._arms.num_dof

    def get_arm(self):
        '''
        Configures and retrieves an instance of the UR10 robot arm.
        Parameters:
            None
        Return:
            None: The function sets up the UR10 robot within the simulation environment but does not return anything.
        '''
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="UR10", gripper_usd=None, attach_gripper=True)
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )
        return ur10

    def get_arm_view(self, scene):
        '''
        Creates a view of the UR10 robot arm within the given scene context.
        Parameters:
            scene: The simulation scene in which the robot is visualized or managed.
        Return:
            UR10View: An instance of UR10View that provides an interface to visualize or interact with the robot's configuration.
        '''
        arm_view = UR10View(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view")
        scene.add(arm_view._end_effectors)
        return arm_view

    def get_object_displacement_tensor(self):
        '''
        Generates a tensor representing the displacement of objects in the environment, used for computations in simulation.
        Parameters:
            None
        Return:
            torch.Tensor: A tensor indicating displacement values for objects in each environment instance.
        '''
        return torch.tensor([0.0, 0.05, 0.0], device=self.device).repeat((self.num_envs, 1))

    def get_observations(self):
        '''
        Retrieves observations from the simulation, depending on the observation type defined in the task configuration.
        Parameters:
            None
        Return:
            dict: A dictionary containing observation buffers for the robot arms.
        '''
        self.arm_dof_pos = self._arms.get_joint_positions()
        #print(self.arm_dof_pos)
        self.arm_dof_vel = self._arms.get_joint_velocities()
        #print(self.arm_dof_vel)
        self.gripper_pos = self._arms.get_gripper_positions()
        

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unkown observations type!")

        observations = {
            self._arms.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def get_reset_target_new_pos(self, n_reset_envs):
        '''
        Computes new target positions for reset environments when resetting part of the simulation environments.
        Parameters:
            n_reset_envs (int): The number of environments to reset.
        Return:
            torch.Tensor: A tensor containing new target positions for each reset environment.
        '''
        # Randomly generate goal positions, although the resulting goal may still not be reachable.
        new_pos = torch_rand_float(-1, 1, (n_reset_envs, 3), device=self.device)
        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            # Depends on your real robot setup
            new_pos[:, 0] = torch.abs(new_pos[:, 0] * 0.1) + 0.35
            new_pos[:, 1] = torch.abs(new_pos[:, 1] * 0.1) + 0.35
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.5) + 0.3
        else:
            new_pos[:, 0] = new_pos[:, 0] * 0.4 + 0.5 * torch.sign(new_pos[:, 0])
            #print(f'new_pos[:, 0]: {new_pos[:, 0]}')
            new_pos[:, 1] = new_pos[:, 1] * 0.4 + 0.5 * torch.sign(new_pos[:, 1])
            #print(f'new_pos[:, 1]: {new_pos[:, 1]}')
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.8) + 0.1
            #print(f'new_pos[:, 2]: {new_pos[:, 2]}')
        if self._task_cfg['safety']['enabled']:
            new_pos[:, 0] = torch.abs(new_pos[:, 0]) / 1.25
            new_pos[:, 1] = torch.abs(new_pos[:, 1]) / 1.25
        return new_pos

    def compute_full_observations(self, no_vel=False):
        '''
        Computes and updates the observation buffer with all required observations including joint positions, velocities, and goal information.
        Parameters:
            no_vel (bool): If true, skips the velocity calculations.
        Return:
            None: The method updates the observation buffer in-place and does not return anything.
        '''
        if no_vel:
            raise NotImplementedError()
        else:
            if self.obs_buf is None:
                self.obs_buf = torch.zeros((self.num_envs, self._num_observations), device=self._device)
            # There are many redundant information for the simple Reacher task, but we'll keep them for now.
            self.obs_buf[:, 0:self.num_arm_dofs] = unscale(self.arm_dof_pos[:, :self.num_arm_dofs],
                self.arm_dof_lower_limits, self.arm_dof_upper_limits)
            self.obs_buf[:, self.num_arm_dofs:2*self.num_arm_dofs] = self.vel_obs_scale * self.arm_dof_vel[:, :self.num_arm_dofs]
            base = 2 * self.num_arm_dofs
            self.obs_buf[:, base+0:base+3] = self.goal_pos
            self.obs_buf[:, base+3:base+7] = self.goal_rot
            self.obs_buf[:, base+7:base+11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, base+11:base+17] = self.actions

            # Include gripper position and velocity
            self.obs_buf[:, base+17:base+20] = self.gripper_pos  # Match the correct dimension
            #self.obs_buf[:, base+20:base+21] = self.vel_obs_scale * self.gripper_vel.unsqueeze(1)


    def send_joint_pos(self, joint_pos):
        '''
        Sends the calculated joint positions to the real UR10 robot if operating in a sim-to-real scenario.
        Parameters:
            joint_pos (torch.Tensor): The joint positions to be sent to the real robot.
        Return:
            None: The function communicates with the real robot but does not return any value.
        '''
        self.real_world_ur10.send_joint_pos(joint_pos)
        gripper_pos = self.obs_buf[:, -1]  # Assuming gripper position is the second last element
        self.real_world_ur10.send_gripper_pos(gripper_pos)
        #print(f'joint positions: {joint_pos}')