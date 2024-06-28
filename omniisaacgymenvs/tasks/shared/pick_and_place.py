from abc import abstractmethod
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.utils import nucleus
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omniisaacgymenvs.robots.articulations.UR10.UR10_pap import UR10  # Adjust the import path as needed

import numpy as np
import pandas as pd
import torch
import random


class PickAndPlaceTask(RLTask):
    def __init__(self, name: str, env: VecEnvBase, offset=None) -> None:
        # Initialization of environment parameters from configuration
        self._num_envs = self._task_cfg["env"]["numEnvs"]  # Number of parallel environments
        self._env_spacing = self._task_cfg["env"]["envSpacing"]  # Spacing between environments

        # Reward scaling factors
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]  # Distance reward scaling factor
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]  # Rotation reward scaling factor
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]  # Action penalty scaling factor

        self.episode_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self.joint_velocity_data = pd.DataFrame(columns=['episode', 'environment', 'joint', 'velocity'])
        self.current_episode = 0
        self.episode_count = 0

        # Performance and success metrics
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]  # Tolerance for success in reaching the goal
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]  # Bonus for reaching the goal
        self.rot_eps = self._task_cfg["env"]["rotEps"]  # Small epsilon value for rotation calculation stability
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]  # Velocity observation scaling factor

        # Noise configurations for reset conditions
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]  # Position noise on reset
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"]  # Rotation noise on reset
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]  # Degree of freedom position noise interval
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]  # Degree of freedom velocity noise interval

        # Miscellaneous configuration
        self.arm_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]  # Speed scale for degrees of freedom
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]  # Flag to use relative control
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]  # Moving average factor for actions

        # Episode configuration
        self.max_episode_length = self._task_cfg["env"]["episodeLength"]  # Maximum length of an episode
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)  # Time after which the environment resets
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]  # Flag to print number of successes
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]  # Maximum number of consecutive successes before reset
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)  # Averaging factor for success computation

        self.dt = 1.0 / 60  # Time step for simulation
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)  # Inverse of control frequency

        # Adjust episode length based on reset time and control frequency
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        RLTask.__init__(self, name, env)

        # Tensor configurations for environment management
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Indicates which environments should be reset
        self.reset_goal_buf = self.reset_buf.clone()  # Buffer to manage goal resets
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Success tracking
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Tracking consecutive successes

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)  # Tensor for averaging factor
        self.total_successes = 0
        self.total_resets = 0
        self.phase = torch.zeros(self._num_envs, dtype=torch.int, device=self.device)  # 0: reach_to_pick, 1: pick, 2: reach_to_place, 3: place

        # Metrics
        self.cumulative_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.goal_distances = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_rewards = []
        self.episode_goal_distances = []
        self.episode_successes = []

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1'
        self._ur10 = scene.add(self.get_arm())

        self.get_arm()
        self.get_object()
        self.get_goal()
        self.get_place()
        super().set_up_scene(scene)
        self._arms = self.get_arm_view(scene)
        scene.add(self._arms)

        self._ur10._gripper.set_translate(value=0.162)
        self._ur10._gripper.set_direction(value="x")
        self._ur10._gripper.set_force_limit(value=8.0e1)
        self._ur10._gripper.set_torque_limit(value=5.0e0)

        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/.*/object/object",
            name="object_view",
            reset_xform_properties=True,
        )
        scene.add(self._objects)
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/.*/goal/object",
            name="goal_view",
            reset_xform_properties=True,
        )
        scene.add(self._goals)

        self._place_goals = RigidPrimView(
            prim_paths_expr="/World/envs/.*/place_goal/object",
            name="place_goal_view",
            reset_xform_properties=True,
        )
        scene.add(self._place_goals)

        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[3, 3, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

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

    def get_gripper_position(self):
        end_effectors_pos, _ = self._arms._end_effectors.get_world_poses()
        return end_effectors_pos

    def get_object(self):
        self.object_start_translation = torch.tensor([0.1585, 0.0, 0.0], device=self.device)
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
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
        self.goal_start_translation = torch.tensor([0.5, 0.0, 0.5], device=self.device)
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal/object",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self._sim_config.apply_articulation_settings("goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object"))

    def get_place(self):
        random_position = torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1), 0.5], device=self.device)
        self.goal_start_translation = random_position
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/place_goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/place_goal/object",
            name="place_goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self._sim_config.apply_articulation_settings("place_goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("place_goal"))

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

        self.object_pos, self.object_rot = self._objects.get_world_poses()
        self.goal_pos, self.goal_rot = self._goals.get_world_poses()
        self.place_pos, self.place_rot = self._place_goals.get_world_poses()

        self.goal_pos -= self._env_pos
        self.place_pos -= self._env_pos

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def reach_the_target(self, actions):
        """
        Move the arm to reach and manipulate the object, following the phases:
        0: Reach to pick
        1: Pick
        2: Reach to place
        3: Place
        
        Parameters:
        - actions: Actions to be taken by the robot
        
        Returns:
        - rewards: Tensor of rewards for each environment
        - resets: Tensor indicating which environments need a reset
        """
        gripper_pos = self.get_gripper_position()
        rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        resets = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        for i in range(self._num_envs):
            action = actions[i]

            if self.phase[i] == 0:  # Reach to pick
                self._arms.set_joint_position_targets(action.unsqueeze(0))
                if torch.norm(gripper_pos[i] - self.object_pos[i]) < self.success_tolerance:
                    self.close_gripper()
                    self.phase[i] = 1
                    rewards[i] += 100  # Bonus for reaching the object

            elif self.phase[i] == 1:  # Pick
                self.close_gripper()
                self.phase[i] = 2

            elif self.phase[i] == 2:  # Reach to place
                self._arms.set_joint_position_targets(action.unsqueeze(0))
                if torch.norm(gripper_pos[i] - self.place_pos[i]) < self.success_tolerance:
                    self.open_gripper()
                    self.phase[i] = 3

            elif self.phase[i] == 3:  # Place
                self.open_gripper()
                rot_dist = self.calculate_rotation_distance(self.goal_rot[i], self.place_rot[i])
                if rot_dist < self.success_tolerance:
                    self.extras['rot_dist'] = rot_dist
                    rewards[i] += 500  # Bonus for correctly placing the object
                    self.episode_successes.append(1)  # Log successful placement
                else:
                    self.episode_successes.append(0)  # Log unsuccessful placement
                resets[i] = True
                self.phase[i] = 0  # Reset to initial phase

        return rewards, resets

    def calculate_rotation_distance(self, rot1, rot2):
        """
        Calculate the rotational distance between two quaternions.
        
        Parameters:
        - rot1: First quaternion
        - rot2: Second quaternion
        
        Returns:
        - The angular difference between the two rotations
        """
        quat_diff = quat_mul(quat_conjugate(rot1), rot2)
        angle = 2 * torch.acos(quat_diff[..., 0].clamp(-1, 1))
        return angle

    def pre_physics_step(self, actions):
        """
        Pre-physics step to handle actions and resets.
        
        Parameters:
        - actions: Actions to be taken by the robot
        """
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

        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.arm_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.arm_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.arm_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.arm_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.arm_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                   self.actuated_dof_indices] + \
                                                             (1.0 - self.act_moving_average) * self.prev_targets[:,
                                                                                         self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:,
                                                                     self.actuated_dof_indices],
                                                                          self.arm_dof_lower_limits[self.actuated_dof_indices],
                                                                          self.arm_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self._arms.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )

        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            cur_joint_pos = self._arms.get_joint_positions(indices=[0], joint_indices=self.actuated_dof_indices)
            joint_pos = cur_joint_pos[0]
            if torch.any(joint_pos < self.arm_dof_lower_limits) or torch.any(joint_pos > self.arm_dof_upper_limits):
                print("get_joint_positions out of bound, send_joint_pos skipped")
            else:
                self.send_joint_pos(joint_pos)

        self.rewards, self.resets = self.reach_the_target(self.actions)
        self.rew_buf[:] = self.rewards
        self.reset_buf[:] = self.resets

    def calculate_metrics(self):
        """
        Calculate various metrics to evaluate the performance of the task.
        """
        gripper_pos = self.get_gripper_position()
        gripper_dist = torch.norm(self.object_pos - gripper_pos, p=2, dim=-1)
        place_dist = torch.norm(self.object_pos - self.place_pos, p=2, dim=-1)

        rewards, resets, goal_resets, progress_buf, successes, cons_successes, gripper_dist, place_dist = compute_arm_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, 0.0, 0.0, self.max_consecutive_successes, self.av_factor, gripper_pos,
            self.place_pos
        )

        # Update cumulative rewards and other metrics
        self.cumulative_rewards += rewards
        self.extras['cumulative reward'] = self.cumulative_rewards

        self.goal_distances += torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        self.goal_place_distances = place_dist  # Save place_dist
        self.extras['Average distance between goal and place'] = self.goal_place_distances / 512
        self.extras['Average rewards'] = self.cumulative_rewards / 512

        self.episode_lengths += 1

        resets_indices = torch.nonzero(resets).squeeze(-1)
        if len(resets_indices) > 0:
            average_rewards = self.cumulative_rewards[resets_indices] / 512
            average_goal_distances = self.goal_distances[resets_indices] / 512
            average_goal_place_distances = self.goal_place_distances[resets_indices] / 512

            self.extras['Average reward'] = average_rewards
            self.extras['Average goal distances'] = average_goal_distances
            # self.extras['Average goal place distances'] = average_goal_place_distances  # Track goal_place_dist

            for idx, avg_reward, avg_goal_dist, avg_goal_place_dist in zip(resets_indices, average_rewards,
                                                                           average_goal_distances,
                                                                           average_goal_place_distances):
                print(
                    f'Episode {self.episode_count}: Environment {idx} - Average Reward: {avg_reward.item()}, Average Goal Distance: {avg_goal_dist.item()}, Average Goal Place Distance: {avg_goal_place_dist.item()}')

                self.episode_count += 1

            self.cumulative_rewards[resets_indices] = 0
            self.goal_distances[resets_indices] = 0
            self.goal_place_distances[resets_indices] = 0
            self.episode_lengths[resets_indices] = 0

        self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes = rewards, resets, goal_resets, progress_buf, successes, cons_successes

        self.extras['consecutive_successes'] = cons_successes.mean()
        self.extras['successes'] = successes.sum()
        self.extras['goal_resets'] = goal_resets.sum()
        self.extras['Average distance between goal and place'] = place_dist.mean()
        self.extras['Average rewards'] = rewards.mean()
        self.extras['Best reward'] = rewards.max()

        if self.print_success_stat:
            self.total_resets += resets.sum()
            direct_average_successes = successes.sum()
            self.total_successes += (successes * resets).sum()
            if self.total_resets > 0:
                print("Direct average consecutive successes = {:.1f}".format(direct_average_successes / self.total_resets))
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets))

    def is_done(self):
        """
        Check if the current task is done.
        """
        pass

    def reset_target_pose(self, env_ids):
        """
        Reset the target pose for the specified environments.
        
        Parameters:
        - env_ids: Indices of the environments to reset
        """
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        new_pos = self.get_reset_target_new_pos(len(env_ids))
        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids],
                                     self.y_unit_tensor[env_ids])
        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot
        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids]
        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        """
        Reset the specified environments.
        
        Parameters:
        - env_ids: Indices of the environments to reset
        """
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch.rand((len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device) * 2 - 1
        pos = torch.clamp(
            self.arm_dof_default_pos.unsqueeze(0) + self.reset_dof_pos_noise * rand_floats[:, 5: 5 + self.num_arm_dofs],
            self.arm_dof_lower_limits, self.arm_dof_upper_limits,
        )
        dof_pos = torch.zeros((len(env_ids), self._arms.num_dof), device=self.device)
        dof_vel = torch.zeros((len(env_ids), self._arms.num_dof), device=self.device)
        dof_pos[:, : self.num_arm_dofs] = pos
        dof_pos[:, self.num_arm_dofs:] = self._arms.get_joint_positions(indices=indices)[:, self.num_arm_dofs:]
        self.cur_targets[env_ids, : self.num_arm_dofs] = dof_pos[:, : self.num_arm_dofs]
        self.prev_targets[env_ids, : self.num_arm_dofs] = dof_pos[:, : self.num_arm_dofs]
        self.arm_dof_targets[env_ids, : self.num_arm_dofs] = dof_pos[:, : self.num_arm_dofs]
        self._arms.set_joint_positions(dof_pos, indices)
        self._arms.set_joint_velocities(dof_vel, indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.cumulative_rewards[env_ids] = 0
        self.goal_distances[env_ids] = 0
        self.episode_lengths[env_ids] = 0

        self.extras["consecutive_successes"] = 0
        self.reset_goal_buf[env_ids] = 1
        self.episode_rewards.append(self.cumulative_rewards.mean().item())
        self.episode_goal_distances.append(self.goal_distances.mean().item())

    def close_gripper(self):
        """
        Close the gripper.
        """
        self._ur10._gripper.close()

    def open_gripper(self):
        """
        Open the gripper.
        """
        self._ur10._gripper.open()

    def check_gripper_holding(self, gripper_pos, goal_pos):
        """
        Check if the gripper is holding the goal object.
        
        Parameters:
        - gripper_pos: Position of the gripper
        - goal_pos: Position of the goal object
        
        Returns:
        - True if the gripper is holding the object, False otherwise
        """
        return torch.norm(gripper_pos - goal_pos, p=2, dim=-1) < 0.05

    def calculate_accuracy(self):
        """
        Calculate the accuracy of the task based on successful placements.
        
        Returns:
        - Accuracy as a percentage
        """
        total_episodes = len(self.episode_successes)
        successful_episodes = sum(self.episode_successes)
        if total_episodes == 0:
            return 0.0
        accuracy = (successful_episodes / total_episodes) * 100
        return accuracy

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """
    Generate a randomized rotation quaternion.
    
    Parameters:
    - rand0: Random value for x-axis rotation
    - rand1: Random value for y-axis rotation
    - x_unit_tensor: Unit vector for x-axis
    - y_unit_tensor: Unit vector for y-axis
    
    Returns:
    - Quaternion representing the random rotation
    """
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def compute_arm_reward(
        rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
        max_episode_length: float, object_pos, object_rot, goal_pos, goal_rot,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, gripper_position,
        place_pos):
    """
    Compute the reward for the pick and place task.

    Parameters:
    - rew_buf: Buffer for storing rewards
    - reset_buf: Buffer for storing environment reset flags
    - reset_goal_buf: Buffer for storing goal reset flags
    - progress_buf: Buffer for tracking progress in the episode
    - successes: Tensor tracking successes
    - consecutive_successes: Tensor tracking consecutive successes
    - max_episode_length: Maximum length of an episode
    - object_pos: Positions of the objects
    - object_rot: Rotations of the objects
    - goal_pos: Positions of the goals
    - goal_rot: Rotations of the goals
    - dist_reward_scale: Scaling factor for distance rewards
    - rot_reward_scale: Scaling factor for rotation rewards
    - rot_eps: Small epsilon value for numerical stability in rotation calculations
    - actions: Actions taken by the agent
    - action_penalty_scale: Scaling factor for action penalties
    - success_tolerance: Tolerance for considering a task as successful
    - reach_goal_bonus: Bonus for reaching the goal
    - fall_dist: Distance threshold for fall penalty
    - fall_penalty: Penalty for falling
    - max_consecutive_successes: Maximum number of consecutive successes before reset
    - av_factor: Averaging factor for computing consecutive successes
    - gripper_position: Positions of the grippers
    - place_pos: Positions of the place targets

    Returns:
    - reward: Computed rewards
    - resets: Flags indicating environments to reset
    - goal_resets: Flags indicating goals to reset
    - progress_buf: Updated progress buffer
    - successes: Updated successes tensor
    - cons_successes: Updated consecutive successes tensor
    - gripper_dist: Distance between the gripper and the goal
    - place_dist: Distance between the object and the place target
    """
    # Calculate the Euclidean distance from the gripper to the goal
    gripper_dist = torch.norm(goal_pos - gripper_position, p=2, dim=-1)
    
    # Calculate the Euclidean distance from the object to the place position
    place_dist = torch.norm(goal_pos - place_pos, p=2, dim=-1)
    
    # Calculate the rotational distance between the object's orientation and the goal's orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # Compute angular difference
    
    # Calculate distance and rotation rewards
    gripper_rew = gripper_dist * dist_reward_scale
    dist_rew = place_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    
    # Sum the distance and rotation rewards
    reward = gripper_rew + dist_rew + rot_rew
    
    # Apply action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale
    reward -= action_penalty
    
    # Check if the gripper is holding the goal object and add a reward if true
    holding = torch.norm(gripper_position - goal_pos, p=2, dim=-1) < 0.05
    reward += holding * 50.0
    
    # Check if the object is correctly placed and add a reward if true
    correct_placement = torch.norm(object_pos - place_pos, p=2, dim=-1) < success_tolerance
    reward += correct_placement * 250.0
    
    # Apply a penalty for the correct placement based on the distance between the goal and the place position
    placement_penalty = correct_placement * torch.norm(goal_pos - place_pos, p=2, dim=-1)
    reward -= placement_penalty
    
    # Determine if a goal reset is required based on correct placement
    goal_resets = torch.where(correct_placement, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    
    # Determine if an environment reset is required based on correct placement or maximum episode length
    resets = torch.where(correct_placement | (progress_buf >= max_episode_length), torch.ones_like(reset_buf), reset_buf)
    
    # Calculate the number of resets
    num_resets = torch.sum(resets)
    
    # Calculate the number of finished consecutive successes
    finished_cons_successes = torch.sum(successes * resets.float())
    
    # Update the consecutive successes with an averaging factor
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes
    )
    
    return reward, resets, goal_resets, progress_buf, successes, cons_successes, gripper_dist, place_dist
