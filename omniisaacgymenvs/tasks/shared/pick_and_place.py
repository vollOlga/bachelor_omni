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
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.episode_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self.joint_velocity_data = pd.DataFrame(columns=['episode', 'environment', 'joint', 'velocity'])
        self.current_episode = 0
        self.episode_count = 0
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
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
        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
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
        self.phase = torch.zeros(self._num_envs, dtype=torch.int, device=self.device) # 0: reach_to_pick, 1: pick, 2: reach_to_place, 3: place


        # Define the target point (place_goal) where the goal object should be placed
        self.place_pose = torch.tensor([0.7, 0.7, 0.0515 / 2.0], device=self.device)  # Example target position

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1'
        self._ur10 = scene.add(self.get_arm())

        self.get_arm()
        self.get_object()
        self.get_goal()
        super().set_up_scene(scene)
        self._arms = self.get_arm_view(scene)
        scene.add(self._arms)

        #function is used to set the location of the gripper in relation to the UR10 robot.
        self._ur10._gripper.set_translate(value=0.162) 
        #function is used to set the direction of the gripper.
        self._ur10._gripper.set_direction(value="x")
        #functions are used to set the force and torque limits of the gripper.
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
        
        # Add the target point visualization
        self.add_place_goal_visualization()

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[3, 3, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def reset_place_goal_pose(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch.rand((len(env_ids), 4), device=self.device)
        new_pos = torch.zeros((len(env_ids), 3), device=self.device)

        for i in range(len(env_ids)):
            while True:
                # Generate a random offset within a circle of radius 1.2 meters
                angle = rand_floats[i, 0] * 2 * np.pi
                radius = 0.5 + rand_floats[i, 1] * 0.7  # Ensure minimum distance is 0.5m and max is 1.2m

                # Calculate potential new position
                random_offset = torch.tensor([radius * np.cos(angle), radius * np.sin(angle)], device=self.device)
                robot_base_pos = self._arms.get_world_poses()[0][env_ids[i], :2]
                potential_pos = robot_base_pos + random_offset

                # Calculate distance
                distance = torch.norm(potential_pos - robot_base_pos)

                # Ensure the new position is within the allowed range
                if 0.5 < distance < 1.2:
                    new_pos[i, :2] = potential_pos
                    new_pos[i, 2] = self.place_pos[2]  # Keep the z-axis position the same
                    break

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot
        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids]
        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def add_place_goal_visualization(self):
        place_goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(place_goal_usd_path, self.default_zero_env_path + "/place_goal")
        
        place_goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/place_goal/object",
            name="place_goal",
            translation=self.place_pose.clone(),  # Ensure it's updated dynamically
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
            scale=torch.tensor([0.1, 0.1, 0.1], device=self.device)  # Scale down for visualization
        )
        self._sim_config.apply_articulation_settings("place_goal", get_prim_at_path(place_goal.prim_path), self._sim_config.parse_actor_config("place_goal"))
    
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
        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.5, 0.0, 0.5], device=self.device)  # Fixed position in the environment
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
        self.fall_dist = 0
        self.fall_penalty = 0
        gripper_pos = self.get_gripper_position()
        rewards, resets, goal_resets, progress, successes, cons_successes = compute_arm_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes, self.max_episode_length, self.object_pos, self.object_rot,
            self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale, self.success_tolerance, self.reach_goal_bonus,
            self.fall_dist, self.fall_penalty, self.max_consecutive_successes, self.av_factor, gripper_pos
        )
        self.mean_rewards = rewards.mean()
        self.extras['mean reward'] = self.mean_rewards
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
                print(f'Episode {self.episode_count}: Environment {idx} - Average Reward: {avg_reward.item()}, Average Goal Distance: {avg_dist.item()}')
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

    def pre_physics_step(self, actions):
        """
        Prepare and process the environment before the physics simulation step.

        Args:
            actions (torch.Tensor): Tensor of actions to be performed.

        Steps:
        1. Identify environments that need resetting:
            - `env_ids` are environments that require a reset.
            - `goal_env_ids` are environments that need their target goal reset.

        2. Compute object positions and rotations:
            - Obtain the end effectors' world positions and rotations.
            - Calculate the object position relative to the end effector's position and rotation.
            - Adjust the object position by subtracting the environment position.

        3. Set the world poses for objects:
            - Update the object's world positions and rotations.

        4. Reset target poses if needed:
            - If there are environments needing goal resets, reset their target poses.
            - If there are environments requiring a reset, reset their indices.

        5. Process actions:
            - Clone and move actions to the appropriate device.
            - Zero out specific actions for stability.

        6. Control target calculations:
            - If relative control is used, compute targets based on previous targets and action deltas.
            - If absolute control is used, compute targets directly from actions, applying scaling and clamping as needed.

        7. Set joint position targets:
            - Update current targets and set them for the arm's joints.

        8. Sim-to-real adjustment (if enabled):
            - For a single environment, check joint positions against limits and send joint positions if within bounds.

        """

        # Identify environments needing resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # Compute object positions and rotations
        end_effectors_pos, end_effectors_rot = self._arms._end_effectors.get_world_poses()
        self.object_pos = end_effectors_pos + quat_rotate(
            end_effectors_rot, 
            quat_rotate_inverse(self.end_effectors_init_rot, self.get_object_displacement_tensor())
        )
        self.object_pos -= self._env_pos
        self.object_rot = end_effectors_rot

        # Set the world poses for objects
        object_pos = self.object_pos + self._env_pos
        object_rot = self.object_rot
        self._objects.set_world_poses(object_pos, object_rot)

        # Reset target poses if needed
        if len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Process actions
        self.actions = actions.clone().to(self.device)
        # Remove or comment out the line below to enable gripper actions
        # self.actions[:, 5] = 0.0

        # Control target calculations
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + \
                    self.arm_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.arm_dof_lower_limits[self.actuated_dof_indices],
                self.arm_dof_upper_limits[self.actuated_dof_indices]
            )
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.arm_dof_lower_limits[self.actuated_dof_indices],
                self.arm_dof_upper_limits[self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * \
                self.cur_targets[:, self.actuated_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.arm_dof_lower_limits[self.actuated_dof_indices],
                self.arm_dof_upper_limits[self.actuated_dof_indices]
            )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # Set joint position targets
        self._arms.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], 
            indices=None, 
            joint_indices=self.actuated_dof_indices
        )

        # Sim-to-real adjustment (if enabled)
        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            cur_joint_pos = self._arms.get_joint_positions(
                indices=[0], joint_indices=self.actuated_dof_indices
            )
            joint_pos = cur_joint_pos[0]
            if torch.any(joint_pos < self.arm_dof_lower_limits) or torch.any(joint_pos > self.arm_dof_upper_limits):
                print("get_joint_positions out of bound, send_joint_pos skipped")
            else:
                self.send_joint_pos(joint_pos)


    def is_done(self):
        pass

    def reset_target_pose(self, env_ids):
        """
        Reset the target pose for specified environments.

        Args:
            env_ids (torch.Tensor): Tensor containing the indices of environments that need their target pose reset.

        Steps:
        1. Convert environment IDs to int32 type.
        2. Generate random floats for rotation randomization.
        3. Get new target positions for the specified environments.
        4. Compute new rotations using randomization.
        5. Update the goal positions and rotations for the specified environments.
        6. Adjust the goal positions by adding the environment positions.
        7. Set the world poses for the goals in the simulation environment.
        8. Reset the goal buffer for the specified environments.

        """

        # Convert environment IDs to int32 type
        indices = env_ids.to(dtype=torch.int32)
        
        # Generate random floats for rotation randomization
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        
        # Get new target positions for the specified environments
        new_pos = self.get_reset_target_new_pos(len(env_ids))
        
        # Compute new rotations using randomization
        new_rot = randomize_rotation(
            rand_floats[:, 0], 
            rand_floats[:, 1], 
            self.x_unit_tensor[env_ids], 
            self.y_unit_tensor[env_ids]
        )
        
        # Update the goal positions and rotations for the specified environments
        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot
        
        # Adjust the goal positions by adding the environment positions
        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids]
        
        # Set the world poses for the goals in the simulation environment
        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        
        # Reset the goal buffer for the specified environments
        self.reset_goal_buf[env_ids] = 0


    def reset_idx(self, env_ids):
        """
        Reset the state for specified environments.

        Args:
            env_ids (torch.Tensor): Tensor containing the indices of environments that need resetting.

        Steps:
        1. Convert environment IDs to int32 type.
        2. Generate random floats for position noise and reset calculations.
        3. Calculate new positions with noise and clamp them within joint limits.
        4. Initialize position and velocity tensors for degrees of freedom (DOF).
        5. Set the arm DOF positions to the calculated positions.
        6. Update the current and previous targets, and arm DOF targets.
        7. Set the joint positions and velocities for the specified environments.
        8. Reset various buffers and counters for the specified environments.
        9. Update additional tracking variables and configurations.

        """

        # Convert environment IDs to int32 type
        indices = env_ids.to(dtype=torch.int32)
        
        # Generate random floats for position noise and reset calculations
        rand_floats = torch.rand((len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device) * 2 - 1
        
        # Calculate new positions with noise and clamp them within joint limits
        pos = torch.clamp(
            self.arm_dof_default_pos.unsqueeze(0) + self.reset_dof_pos_noise * rand_floats[:, 5 : 5 + self.num_arm_dofs],
            self.arm_dof_lower_limits, self.arm_dof_upper_limits,
        )
        
        # Initialize position and velocity tensors for degrees of freedom (DOF)
        dof_pos = torch.zeros((len(env_ids), self._arms.num_dof), device=self.device)
        dof_vel = torch.zeros((len(env_ids), self._arms.num_dof), device=self.device)
        
        # Set the arm DOF positions to the calculated positions
        dof_pos[:, : self.num_arm_dofs] = pos
        dof_pos[:, self.num_arm_dofs :] = self._arms.get_joint_positions(indices=indices)[:, self.num_arm_dofs :]
        
        # Update the current and previous targets, and arm DOF targets
        self.cur_targets[env_ids, : self.num_arm_dofs] = dof_pos[:, : self.num_arm_dofs]
        self.prev_targets[env_ids, : self.num_arm_dofs] = dof_pos[:, : self.num_arm_dofs]
        self.arm_dof_targets[env_ids, : self.num_arm_dofs] = dof_pos[:, : self.num_arm_dofs]
        
        # Set the joint positions and velocities for the specified environments
        self._arms.set_joint_positions(dof_pos, indices)
        self._arms.set_joint_velocities(dof_vel, indices)
        
        # Reset various buffers and counters for the specified environments
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.cumulative_rewards[env_ids] = 0
        self.goal_distances[env_ids] = 0
        self.episode_lengths[env_ids] = 0
        
        # Update additional tracking variables and configurations
        self.extras["consecutive_successes"] = 0
        self.reset_goal_buf[env_ids] = 1
        self.episode_rewards.append(self.cumulative_rewards.mean().item())
        self.episode_goal_distances.append(self.goal_distances.mean().item())
        if self._task_cfg["env"]["episodeLength"]:
            self.episode_lengths[env_ids] = self._task_cfg["env"]["episodeLength"]



    def get_object_displacement_tensor(self):
        return self.goal_displacement_tensor
    def close_gripper(self):
        self._ur10._gripper.close_gripper()
    
    def open_gripper(self):
        self._ur10._gripper.open_gripper()

    def check_gripper_holding(self, gripper_pos, goal_pos):
        # Simple distance-based check to see if the gripper is holding the goal object
        return torch.norm(gripper_pos - goal_pos, p=2, dim=-1) < 0.05  # Threshold distance for holding

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@torch.jit.script
def compute_arm_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, gripper_position
):
    """
    Compute the reward for the robotic arm in the current simulation step.

    Args:
        rew_buf (torch.Tensor): Buffer to store the rewards.
        reset_buf (torch.Tensor): Buffer indicating which environments need a reset.
        reset_goal_buf (torch.Tensor): Buffer indicating which goals need resetting.
        progress_buf (torch.Tensor): Buffer tracking the progress of each environment.
        successes (torch.Tensor): Buffer tracking the number of successes.
        consecutive_successes (torch.Tensor): Buffer tracking consecutive successes.
        max_episode_length (float): Maximum length of an episode.
        object_pos (torch.Tensor): Positions of the objects.
        object_rot (torch.Tensor): Rotations of the objects.
        target_pos (torch.Tensor): Target positions for the objects.
        target_rot (torch.Tensor): Target rotations for the objects.
        dist_reward_scale (float): Scaling factor for distance reward.
        rot_reward_scale (float): Scaling factor for rotation reward.
        rot_eps (float): Small value to avoid division by zero in rotation calculations.
        actions (torch.Tensor): Actions taken by the robotic arm.
        action_penalty_scale (float): Scaling factor for action penalty.
        success_tolerance (float): Distance threshold for considering the task a success.
        reach_goal_bonus (float): Bonus reward for reaching the goal.
        fall_dist (float): Distance threshold for considering the object has fallen.
        fall_penalty (float): Penalty for the object falling.
        max_consecutive_successes (int): Maximum number of consecutive successes allowed.
        av_factor (float): Averaging factor for calculating consecutive successes.
        gripper_position (torch.Tensor): Positions of the gripper.

    Returns:
        tuple: A tuple containing updated rewards, resets, goal resets, progress buffer, successes, and consecutive successes.

    Steps:
    1. Compute the distance between the object and the target, and between the object and the gripper.
    2. Initialize the reward based on these distances.
    3. Apply scaling to the distance reward and add penalties for actions taken.
    4. Check if the gripper is holding the object and apply a bonus if true.
    5. Check if the object is correctly placed at the target and apply a bonus if true.
    6. Apply a penalty for the precision of the placement.
    7. Determine if the goal needs resetting based on the success tolerance.
    8. Update the successes buffer and apply the reach goal bonus if needed.
    9. Handle resets based on episode length and consecutive successes.
    10. Calculate and return the updated reward, resets, goal resets, progress buffer, successes, and consecutive successes.

    """

    # Compute the distance between the object and the target, and between the object and the gripper
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    gripper_dist = torch.norm(object_pos - gripper_position, p=2, dim=-1)
    
    # Initialize the reward based on these distances
    reward = goal_dist + gripper_dist
    
    # Apply scaling to the distance reward and add penalties for actions taken
    reward = reward * dist_reward_scale
    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale
    reward -= action_penalty

    # Check if the gripper is holding the object and apply a bonus if true
    holding = torch.norm(gripper_position - object_pos, p=2, dim=-1) < 0.05  # Threshold distance for holding
    reward += holding * 50.0  # Bonus for picking up the goal object

    # Check if the object is correctly placed at the target and apply a bonus if true
    correct_placement = torch.norm(object_pos - target_pos, p=2, dim=-1) < success_tolerance
    reward += correct_placement * 250.0  # Bonus for correct placement

    # Apply a penalty for the precision of the placement
    placement_penalty = correct_placement * torch.norm(object_pos - target_pos, p=2, dim=-1)
    reward -= placement_penalty

    # Determine if the goal needs resetting based on the success tolerance
    goal_resets = torch.where(correct_placement, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    
    # Handle resets based on episode length and correct placement
    resets = torch.where(correct_placement | (progress_buf >= max_episode_length), torch.ones_like(reset_buf), reset_buf)
    
    # Calculate the number of resets and update the consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0, 
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, 
        consecutive_successes
    )
    
    return reward, resets, goal_resets, progress_buf, successes, cons_successes


