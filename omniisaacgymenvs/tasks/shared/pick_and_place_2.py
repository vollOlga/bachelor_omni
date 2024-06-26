from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path, set_prim_parent
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from pxr import Usd, UsdGeom, Sdf, UsdPhysics, PhysxSchema, UsdShade, Gf
import omni.usd
import omni

import numpy as np
import torch
import random

class PickAndPlaceTask(RLTask):
    def __init__(self, name: str, env: VecEnvBase, offset=None) -> None:
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.gripper = None  # Initialize the gripper attribute to None

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]

        self.episode_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self.episode_count = 0
        self.object_placed_correctly = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.is_object_held = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.is_object_held_tensor = self.is_object_held.clone().detach().to(self.device)
        self.object_placed_correctly_tensor = self.object_placed_correctly.clone().detach().to(self.device)
        self.rew_buf = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.grasp_goal_bonus = self._task_cfg["env"]["graspGoalBonus"]
        self.place_goal_bonus = self._task_cfg["env"]["placeGoalBonus"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]

        self.arm_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)
        self._num_gripper_dof = 1

        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        self.cumulative_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.goal_distances = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_rewards = []
        self.episode_goal_distances = []
        self.task_state = 'reaching'
        self.reacher_success = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.gripper_success = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.place_success = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_success = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)

        self.target_rot = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self.device)

        self.place_goal_pos = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float, device=self.device)
        self.place_goal_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device)

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1'

        self.get_arm()
        self.get_object()
        self.get_goal()
        self.get_place_goal()

        super().set_up_scene(scene)

        self._arms = self.get_arm_view(scene)
        scene.add(self._arms)
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
        self._place_goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/place_goal/object",
            name="place_goal_view",
            reset_xform_properties=False,
        )
        scene.add(self._place_goals)

    def close_gripper(self, env_ids):
        for env_id in env_ids:
            object_prim_path = f"/World/envs/env_{env_id}/object/object"
            end_effector_path = f"/World/envs/env_{env_id}/ur10_short_suction/ee_link"
            set_prim_parent(object_prim_path, end_effector_path)
            self.is_object_held[env_id] = True

    def open_gripper(self, env_ids):
        for env_id in env_ids:
            object_prim_path = f"/World/envs/env_{env_id}/object/object"
            original_parent_path = f"/World/envs/env_{env_id}"
            set_prim_parent(object_prim_path, original_parent_path)
            self.is_object_held[env_id] = False

    def is_gripper_closed(self, env_id):
        return self.is_object_held[env_id]

    def is_holding_object(self):
        return self.is_object_held.any().item()

    @abstractmethod
    def get_num_dof(self):
        pass

    @abstractmethod
    def get_arm(self):
        pass

    @abstractmethod
    def get_arm_view(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass

    @abstractmethod
    def get_reset_target_new_pos(self, n_reset_envs):
        pass

    @abstractmethod
    def send_joint_pos(self, joint_pos):
        pass

    def get_object(self):
        self.object_start_translation = torch.tensor([0.1585, 0.0, 0.0], device=self.device)
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.object_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float).to(self.device)
        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"

        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale
        )
        self._sim_config.apply_articulation_settings("object", get_prim_at_path(obj.prim_path), self._sim_config.parse_actor_config("object"))

    def get_goal(self):
        possible_goals = [
            f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd",
            f"{self._assets_root_path}/Isaac/Props/Blocks/tomato_soup.usd",
            f"{self._assets_root_path}/Isaac/Props/Blocks/block.usd"
        ]
        self.goal_usd_path = random.choice(possible_goals)
        self.goal_scale = torch.tensor([random.uniform(0.5, 4.0), random.uniform(0.5, 4.0), random.uniform(0.5, 4.0)], device=self.device)

        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal/object",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self._sim_config.apply_articulation_settings("goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object"))

    def get_place_goal(self):
        possible_goals = [
            f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        ]
        self.goal_usd_path = random.choice(possible_goals)
        self.goal_scale = torch.tensor([random.uniform(0.01, 0.02), random.uniform(0.01, 0.02), random.uniform(0.01, 0.02)], device=self.device)

        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/place_goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/place_goal/object",
            name="place_goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self._sim_config.apply_articulation_settings("place_goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("place_goal_object"))

    def post_reset(self):
        self.num_arm_dofs = self.get_num_dof()
        self.actuated_dof_indices = torch.arange(self.num_arm_dofs, dtype=torch.long, device=self.device)

        self.arm_dof_targets = torch.zeros((self.num_envs, self._arms.num_dof), dtype=torch.float, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)

        dof_limits = self._dof_limits
        self.arm_dof_lower_limits, self.arm_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        self.arm_dof_default_pos = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device)
        self.arm_dof_default_vel = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device)

        self.end_effectors_init_pos, self.end_effectors_init_rot = self._arms._end_effectors.get_world_poses()

        self.goal_pos, self.goal_rot = self._goals.get_world_poses()
        self.goal_pos -= self._env_pos

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        distance_to_goal = torch.norm(self.goal_pos - self.place_goal_pos, p=2, dim=-1)
        self.object_placed_correctly = distance_to_goal < 0.1

        self.fall_dist = 0
        self.fall_penalty = 0

        is_object_held_tensor = self.is_object_held.clone().detach().to(self.rew_buf.device)
        object_placed_correctly_tensor = self.object_placed_correctly.clone().detach().to(self.rew_buf.device)

        if not isinstance(self.place_goal_pos, torch.Tensor):
            self.place_goal_pos = torch.tensor(self.place_goal_pos, dtype=torch.float, device=self.device)
        if not isinstance(self.place_goal_rot, torch.Tensor):
            self.place_goal_rot = torch.tensor(self.place_goal_rot, dtype=torch.float, device=self.device)

        if self.object_rot.shape != self.target_rot.shape:
            self.target_rot = self.target_rot.repeat(self.object_rot.shape[0], 1)
        if self.object_rot.shape != self.place_goal_rot.shape:
            self.place_goal_rot = self.place_goal_rot.repeat(self.object_rot.shape[0], 1)

        rewards, resets, goal_resets, progress, successes, cons_successes, reacher_success, gripper_success, place_success = compute_pick_and_place_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.target_pos, self.target_rot,
            self.place_goal_pos, self.place_goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.grasp_goal_bonus, self.place_goal_bonus,
            self.fall_dist, self.fall_penalty, int(self.max_consecutive_successes), self.av_factor,
            is_object_held_tensor, object_placed_correctly_tensor, self.reacher_success, self.gripper_success, self.place_success
        )

        self.reacher_success = reacher_success
        self.gripper_success = gripper_success
        self.place_success = place_success

        self.cumulative_rewards += rewards
        self.extras['cumulative reward'] = self.cumulative_rewards
        self.goal_distances += torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)

        self.episode_lengths += 1

        resets_indices = torch.nonzero(resets).squeeze(-1)
        if len(resets_indices) > 0:
            average_rewards = self.cumulative_rewards[resets_indices] / self.episode_lengths[resets_indices]
            average_distances = self.goal_distances[resets_indices] / self.episode_lengths[resets_indices]

            self.extras['Average reward'] = average_rewards
            self.extras['Average goal distances'] = average_distances

            for idx, avg_reward, avg_dist in zip(resets_indices, average_rewards, average_distances):
                self.episode_count += 1

            self.cumulative_rewards[resets_indices] = 0
            self.goal_distances[resets_indices] = 0
            self.episode_lengths[resets_indices] = 0

        self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes = rewards, resets, goal_resets, progress, successes, cons_successes

        self.extras['consecutive_successes'] = cons_successes.mean()

        if self.print_success_stat:
            self.total_resets += resets.sum()
            direct_average_successes = successes.sum()
            self.total_successes += (successes * resets).sum()
            if self.total_resets > 0:
                print("Direct average consecutive successes = {:.1f}".format(direct_average_successes / self.total_resets))
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets))

    def print_episode_stats(self, resets_indices):
        for idx in resets_indices:
            average_reward = self.episode_rewards[idx] / self.episode_lengths[idx]
            print(f'Episode {self.episode_count}: Environment {idx} - Average Reward: {average_reward.item()}')
            self.episode_count += 1

    def get_object_displacement_tensor(self):
        desired_offset = torch.tensor([0.1585, 0.0, 0.0], device=self.device)
        return desired_offset

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        end_effectors_pos, end_effectors_rot = self._arms._end_effectors.get_world_poses()

        if self.task_state == 'reaching' or self.task_state == 'transport':
            self.object_pos = end_effectors_pos + quat_rotate(end_effectors_rot, quat_rotate_inverse(self.end_effectors_init_rot, self.get_object_displacement_tensor()))
            self.object_pos -= self._env_pos
            self.object_rot = end_effectors_rot
            object_pos = self.object_pos + self._env_pos
            object_rot = self.object_rot
            self._objects.set_world_poses(object_pos, object_rot)

            if self.task_state == 'reaching':
                self.target_pos = self.object_pos
                dist_to_object = torch.norm(end_effectors_pos - object_pos, p=2, dim=-1)

                if torch.all(dist_to_object < self.success_tolerance):
                    self.task_state = 'grasping'
                    self.grasping_timer = 0

            elif self.task_state == 'transport':
                self.place_goal_pos, self.place_goal_rot = self._place_goals.get_world_poses()
                self.place_goal_pos -= self._env_pos
                self.goal_pos = self.place_goal_pos
                self.goal_rot = self.place_goal_rot
                goal_pos = self.goal_pos + self._env_pos
                self._goals.set_world_poses(goal_pos, self.goal_rot)

                dist_to_goal = torch.norm(self.object_pos - self.place_goal_pos, p=2, dim=-1)

                if torch.all(dist_to_goal < self.success_tolerance):
                    self.task_state = 'placing'
                    self.open_gripper(env_ids)

        elif self.task_state == 'grasping':
            self.close_gripper(env_ids)

            if self.is_gripper_closed():
                self.grasping_timer += self.dt
                if self.grasping_timer >= 4.0:
                    self.task_state = 'transport'

        elif self.task_state == 'placing':
            self.open_gripper(env_ids)

            if self.object_placed_correctly:
                self.task_state = 'reaching'

        self._arms.set_joint_position_targets(actions[:, :self._num_arm_dof])
        self._grippers.set_joint_efforts(actions[:, self._num_arm_dof:self._num_arm_dof + self._num_gripper_dof])

        if len(env_ids) > 0:
            self.reset(env_ids)
        if len(goal_env_ids) > 0:
            self.reset_goal(goal_env_ids)

    def is_done(self):
        pass

    def reset_target_pose(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_pos = self.get_reset_target_new_pos(len(env_ids))
        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids]

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        delta_max = self.arm_dof_upper_limits - self.arm_dof_default_pos
        delta_min = self.arm_dof_lower_limits - self.arm_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5 + self.num_arm_dofs] + 1.0) * 0.5

        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos

        dof_vel = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_arm_dofs:5 + self.num_arm_dofs * 2]

        self.prev_targets[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_dofs] = pos
        self.arm_dof_targets[env_ids, :self.num_arm_dofs] = pos

        self._arms.set_joint_position_targets(self.arm_dof_targets[env_ids], indices)
        self._arms.set_joint_positions(dof_pos[env_ids], indices)
        self._arms.set_joint_velocities(dof_vel[env_ids], indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        self.cumulative_rewards[env_ids] = 0
        self.goal_distances[env_ids] = 0

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@torch.jit.script
def compute_pick_and_place_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    place_pos, place_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, grasp_goal_bonus: float, place_goal_bonus: float,
    fall_dist: float, fall_penalty: float, max_consecutive_successes: int, av_factor: float, 
    is_object_held: torch.Tensor, object_placed_correctly: torch.Tensor,
    reacher_success: torch.Tensor, gripper_success: torch.Tensor, place_success: torch.Tensor
):
    assert object_rot.shape == target_rot.shape, "Shape mismatch between object_rot and target_rot"
    assert object_rot.shape == place_rot.shape, "Shape mismatch between object_rot and place_rot"

    object_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    object_dist_rew = object_dist * dist_reward_scale
    
    quat_diff_object = quat_mul(object_rot, quat_conjugate(target_rot))
    object_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff_object[:, 1:4], p=2, dim=-1), max=1.0))
    object_rot_rew = 1.0 / (torch.abs(object_rot_dist) + rot_eps) * rot_reward_scale

    reach_reward = - (object_dist_rew + object_rot_rew)
    reach_reward += torch.where(object_dist < success_tolerance, reach_goal_bonus, torch.zeros_like(reach_reward))

    place_dist = torch.norm(object_pos - place_pos, p=2, dim=-1)
    place_dist_rew = place_dist * dist_reward_scale
    
    quat_diff_place = quat_mul(object_rot, quat_conjugate(place_rot))
    place_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff_place[:, 1:4], p=2, dim=-1), max=1.0))
    place_rot_rew = 1.0 / (torch.abs(place_rot_dist) + rot_eps) * rot_reward_scale

    place_reward = - (place_dist_rew + place_rot_rew)
    place_reward += torch.where(object_placed_correctly, place_goal_bonus, torch.zeros_like(place_reward))

    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale

    reward = reach_reward + place_reward - action_penalty

    goal_resets = torch.where(object_placed_correctly, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes += goal_resets

    reacher_success = torch.where(object_dist < success_tolerance, torch.ones_like(reacher_success), reacher_success)
    gripper_success = torch.where(is_object_held, torch.ones_like(gripper_success), gripper_success)
    place_success = torch.where(object_placed_correctly, torch.ones_like(place_success), place_success)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, reacher_success, gripper_success, place_success
