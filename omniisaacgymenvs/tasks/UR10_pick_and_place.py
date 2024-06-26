from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.tasks.shared.pickandplace import PickAndPlaceTask
from omniisaacgymenvs.robots.articulations.views.ur10_view import UR10View
from omniisaacgymenvs.robots.articulations.UR10 import UR10

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *
from omni.isaac.core.utils.stage import get_current_stage  # Import get_current_stage
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.scenes.scene import Scene  # Import Scene class

import numpy as np
import torch
import math

class UR10PickAndPlaceTask(PickAndPlaceTask):
    def __init__(self, name: str, sim_config: SimConfig, env: VecEnvBase, offset=None) -> None:
        self._device = "cuda:0"
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full"]):
            raise Exception("Unknown type of observations!\nobservationType should be one of: [full]")
        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full": 43,
        }

        self.object_scale = torch.tensor([1.0] * 3)
        self.goal_scale = torch.tensor([2.0] * 3)

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 6
        self._num_states = 0

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

        PickAndPlaceTask.__init__(self, name=name, env=env)

        sim2real_config = self._task_cfg['sim2real']
        if sim2real_config['enabled'] and self.test and self.num_envs == 1:
            self.act_moving_average /= 5
            self.real_world_ur10 = RealWorldUR10(
                sim2real_config['fail_quietely'],
                sim2real_config['verbose']
            )

        # Initialize episode lengths and other attributes
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self.episode_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_count = 0

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()  # Get the current USD stage
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1'  # Path to assets

        # Retrieve and set up arm, object, and goal elements in the scene
        self.get_arm()
        self.get_object()
        self.get_goal()
        self.get_place_goal()

        super().set_up_scene(scene)  # Call to superclass method to complete scene setup

        # Create views for arms, objects, and goals
        self._arms = self.get_arm_view(scene)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
        )
        scene.add(self._objects)
        
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object",
            name="goal_view",
            reset_xform_properties=False,
        )
        scene.add(self._goals)

        # Initialize grippers
        self.grippers = [arm.suction_gripper for arm in self._arms]

    def get_arm_view(self, scene):
        arm_view = UR10View(prim_paths_expr="/World/envs/.*/ur10", name=f"ur10_view_{id(self)}")
        scene.add(arm_view._end_effectors)
        return arm_view

    def get_num_dof(self):
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
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="UR10")
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

    def get_object_displacement_tensor(self):
        return torch.tensor([0.0, 0.05, 0.0], device=self.device).repeat((self.num_envs, 1))

    def get_observations(self):
        self.arm_dof_pos = self._arms.get_joint_positions()
        self.arm_dof_vel = self._arms.get_joint_velocities()

        if self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unknown observations type!")

        observations = {
            self._arms.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def get_reset_target_new_pos(self, n_reset_envs):
        new_pos = torch_rand_float(-1, 1, (n_reset_envs, 3), device=self.device)
        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            new_pos[:, 0] = torch.abs(new_pos[:, 0] * 0.1) + 0.35
            new_pos[:, 1] = torch.abs(new_pos[:, 1] * 0.1) + 0.35
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.5) + 0.3
        else:
            new_pos[:, 0] = new_pos[:, 0] * 0.4 + 0.5 * torch.sign(new_pos[:, 0])
            new_pos[:, 1] = new_pos[:, 1] * 0.4 + 0.5 * torch.sign(new_pos[:, 1])
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.8) + 0.1
        if self._task_cfg['safety']['enabled']:
            new_pos[:, 0] = torch.abs(new_pos[:, 0]) / 1.25
            new_pos[:, 1] = torch.abs(new_pos[:, 1]) / 1.25
        return new_pos

    def compute_full_observations(self):
        self.obs_buf[:, 0:self.num_arm_dofs] = unscale(self.arm_dof_pos[:, :self.num_arm_dofs],
            self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        self.obs_buf[:, self.num_arm_dofs:2*self.num_arm_dofs] = self.vel_obs_scale * self.arm_dof_vel[:, :self.num_arm_dofs]
        base = 2 * self.num_arm_dofs
        self.obs_buf[:, base+0:base+3] = self.goal_pos
        self.obs_buf[:, base+3:base+7] = self.goal_rot
        self.obs_buf[:, base+7:base+11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        self.obs_buf[:, base+11:base+14] = self.place_goal_pos
        self.obs_buf[:, base+14:base+18] = self.place_goal_rot
        self.obs_buf[:, base+18:base+22] = quat_mul(self.object_rot, quat_conjugate(self.place_goal_rot))
        self.obs_buf[:, base+22:base+28] = self.actions

    def send_joint_pos(self, joint_pos):
        self.real_world_ur10.send_joint_pos(joint_pos)

    def turn_on_suction(self):
        for gripper in self.grippers:
            gripper.suction_on()

    def turn_off_suction(self):
        for gripper in self.grippers:
            gripper.suction_off()

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        end_effectors_pos, end_effectors_rot = self._arms._end_effectors.get_world_poses()

        self.object_pos = end_effectors_pos + quat_rotate(end_effectors_rot, quat_rotate_inverse(self.end_effectors_init_rot, self.get_object_displacement_tensor()))
        self.object_pos -= self._env_pos
        self.object_rot = end_effectors_rot
        object_pos = self.object_pos + self._env_pos
        object_rot = self.object_rot
        self._objects.set_world_poses(object_pos, object_rot)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        self.actions[:, 4] = 0.0

        gripper_action = actions[:, 5]  # Gripper action

        # Apply gripper action to the object
        for idx, action in enumerate(gripper_action):
            if action > 0.5:
                self.grippers[idx].suction_on()  # Turn on suction
            else:
                self.grippers[idx].suction_off()

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.arm_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self._arms.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )
        
        # Suction control logic
        if self.reached_goal():
            self.turn_on_suction()
        elif self.reached_place_goal():
            self.turn_off_suction()

    def reached_goal(self):
        goal_distance = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        return torch.all(goal_distance < self.success_tolerance)

    def reached_place_goal(self):
        place_goal_distance = torch.norm(self.object_pos - self.place_goal_pos, p=2, dim=-1)
        return torch.all(place_goal_distance < self.success_tolerance)
