from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
import omni
import omniisaacgymenvs.scripts.math_util as math_util
from omni.isaac.core.objects.capsule import VisualCapsule
from omni.isaac.core.objects.sphere import VisualSphere
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.scripts.cortex_rigid_prim import CortexRigidPrim
from omniisaacgymenvs.scripts.cortex_utils import get_assets_root_path
from omniisaacgymenvs.scripts.robot import CortexUr10
#from omni.isaac.cortex.sample_behaviors.ur10 import bin_stacking_behavior as behavior
from omni.isaac.examples.cortex.cortex_base import CortexBase
# `scale` maps [-1, 1] to [L, U]; `unscale` maps [L, U] to [-1, 1]
from omni.isaac.core.utils.torch import scale, unscale
from omni.isaac.gym.vec_env import VecEnvBase

from omniisaacgymenvs.scripts.df import (
    DfDecider,
    DfDecision,
    DfNetwork,
    DfSetLockState,
    DfState,
    DfStateMachineDecider,
    DfStateSequence,
    DfTimedDeciderState,
    DfWaitState,
    DfWriteContextState,
)

from pxr import Sdf, UsdPhysics, UsdShade 

import numpy as np
import torch
import random

class CloseSuctionGripper(DfState):
    """
    Represents a state to close the suction gripper.

    Args:
        None
    """
    def enter(self):
        """
        Initiates the closing of the suction gripper.

        Returns:
            None
        """
        print("<close gripper>")
        self.context.robot.suction_gripper.close()

    def step(self):
        """
        Completes the state.

        Returns:
            None
        """
        return None


class OpenSuctionGripper(DfState):
    """
    Represents a state to open the suction gripper.

    Args:
        None
    """
    def enter(self):
        """
        Initiates the opening of the suction gripper.

        Returns:
            None
        """
        print("<open gripper>")
        self.context.robot.suction_gripper.open()

    def step(self):
        """
        Completes the state.

        Returns:
            Noneß
        """
        return None
    

class PlaceObject(DfStateMachineDecider):
    """
    Represents a state machine for placing a bin.

    Args:
        None
    """
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPlace(),  # Step 1: Reach the target place location.
                    DfWaitState(wait_time=0.5),  # Step 2: Wait for 0.5 seconds.
                    DfSetLockState(set_locked_to=True, decider=self),  # Step 3: Lock the state machine.
                    OpenSuctionGripper(),  # Step 4: Open the suction gripper to release the bin.
                    DfTimedDeciderState(DfLift(0.1), activity_duration=0.25),  # Step 5: Lift the gripper slightly.
                    DfWriteContextState(lambda ctx: ctx.mark_active_bin_as_complete()),  # Step 6: Mark the bin as placed.
                    DfSetLockState(set_locked_to=False, decider=self),  # Step 7: Unlock the state machine.
                ]
            )
        )


class PickObject(DfStateMachineDecider):
    """
    Represents a state machine for picking a bin.

    Args:
        None
    """
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPick(),  # Step 1: Reach the bin to pick it up.
                    DfWaitState(wait_time=0.5),  # Step 2: Wait for 0.5 seconds.
                    DfSetLockState(set_locked_to=True, decider=self),  # Step 3: Lock the state machine.
                    CloseSuctionGripper(),  # Step 4: Close the suction gripper to pick the bin.
                    DfTimedDeciderState(DfLift(0.3), activity_duration=0.4),  # Step 5: Lift the bin for 0.4 seconds.
                    DfSetLockState(set_locked_to=False, decider=self),  # Step 6: Unlock the state machine.
                ]
            )
        )



class PickAndPlace(RLTask):
    """
    Implements a Reacher task for Reinforcement Learning using the Omni Isaac Gym environment.
    This class is designed to manage the environment setup, interaction, and rewards specific to a robotic reaching task.
    """
    def __init__(
        self,
        name: str,
        env: VecEnvBase,
        offset=None
    ) -> None:
        """
        Initializes the Reacher task with configuration and environment settings.
        
        Args:
            name (str): The name of the task, used for identification.
            env (VecEnvBase): The vectorized environment instance this task will operate within.
            offset (optional): An optional offset parameter used for environment setup. Default is None.
        """
        # Initialization of environment parameters from configuration
        self._num_envs = self._task_cfg["env"]["numEnvs"] # Number of parallel environments
        self._env_spacing = self._task_cfg["env"]["envSpacing"] # Spacing between environments

         # Reward scaling factors
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"] # Distance reward scaling factor
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"] # Rotation reward scaling factor
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"] # Action penalty scaling factor
        
        self.episode_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self.episode_count = 0

        
        # Performance and success metrics
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]  #Tolerance for success in reaching the goal
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"] # Bonus for reaching the goal
        self.rot_eps = self._task_cfg["env"]["rotEps"]  # Small epsilon value for rotation calculation stability
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"] # Velocity observation scaling factor
        
        # Noise configurations for reset conditions
        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"] # Position noise on reset
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"] # Rotation noise on reset
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"] # Degree of freedom position noise interval
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"] # Degree of freedom velocity noise interval

        # Miscellaneous configuration
        self.arm_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"] # Speed scale for degrees of freedom
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"] # Flag to use relative control
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"] # Moving average factor for actions

        # Episode configuration
        self.max_episode_length = self._task_cfg["env"]["episodeLength"] # Maximum length of an episode
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0) # Time after which the environment resets
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"] # Flag to print number of successes
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"] # Maximum number of consecutive successes before reset
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1) # Averaging factor for success computation

        self.dt = 1.0 / 60 # Time step for simulation
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1) # Inverse of control frequency

        # Adjust episode length based on reset time and control frequency
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # Call to superclass initializer
        RLTask.__init__(self, name, env)

        # Tensor configurations for environment management
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Indicates which environments should be reset
        self.reset_goal_buf = self.reset_buf.clone()  # Buffer to manage goal resets
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # Success tracking
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device) # Tracking consecutive successes

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device) # Tensor for averaging factor
        self.total_successes = 0
        self.total_resets = 0

        # Metriks
        self.cumulative_rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.goal_distances = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        self.episode_rewards = []
        self.episode_goal_distances = []

        self.timesteps_since_start = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)  
        self.success_timesteps = [] 
        self.accuracy = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        return
    

    def set_up_scene(self, scene: Scene) -> None:
        """
        Sets up the scene for the Reacher task by adding necessary assets and configuring the environment.
        
        Args:
            scene (Scene): The scene to which the task elements will be added.
        """
        self._stage = get_current_stage() # Get the current USD stage
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1' # Path to assets

        # Retrieve and set up arm, object, and goal elements in the scene
        self.get_arm()
        self.get_object()
        self.get_goal()
        self.get_place()

        super().set_up_scene(scene) # Call to superclass method to complete scene setup

        # Create views for arms, objects, and goals
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

        self._goal_places = RigidPrimView(
        prim_paths_expr="/World/envs/env_.*/goal_place/object",
        name="goal_place_view",
        reset_xform_properties=False,
        )
        scene.add(self._goal_places)

    @abstractmethod
    def get_num_dof(self):
        """
        Abstract method to retrieve the number of degrees of freedom for the robotic arm.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_arm(self):
        """
        Abstract method to retrieve or set up the robotic arm in the environment.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_arm_view(self):
        """
        Abstract method to create a view for the robotic arm.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_observations(self):
        """
        Abstract method to obtain observations from the environment.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_reset_target_new_pos(self, n_reset_envs):
        """
        Abstract method to compute new positions for targets upon environment reset.
        Must be implemented by subclasses.
        
        Args:
            n_reset_envs (int): Number of environments that need resetting.
        """
        pass

    @abstractmethod
    def send_joint_pos(self, joint_pos):
        """
        Abstract method to send joint positions, possibly to real hardware if simulating real-world scenarios.
        Must be implemented by subclasses.
        
        Args:
            joint_pos: The positions of the joints to be set.
        """
        pass

    def get_object(self):
        """
        Retrieves and sets up the object in the environment, applying necessary transformations and references.
        """
        self.object_start_translation = torch.tensor([0.1585, 0.0, 0.0], device=self.device)  # Initial position
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device) # Initial orientation
        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd" # USD path for the object

        # Add object to the stage
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale
        )
        # Apply configuration settings from simulation to the object
        self._sim_config.apply_articulation_settings("object", get_prim_at_path(obj.prim_path), self._sim_config.parse_actor_config("object"))

    def get_goal(self):
        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
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
        self.goal_scale = torch.tensor([random.uniform(0.5, 4.0), random.uniform(0.5, 4.0), random.uniform(0.5, 4.0)], device=self.device)
        print(f"Goal scale: {self.goal_scale}")

        self._sim_config.apply_articulation_settings("goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object"))

    def get_place(self):
        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal_place")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal_place/object",
            name="goal_place",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self.goal_scale = torch.tensor([random.uniform(0.01, 0.5), random.uniform(0.01, 0.5), random.uniform(0.01, 0.5)], device=self.device)
        print(f"Place goal scale: {self.goal_scale}")

        self._sim_config.apply_articulation_settings("goal_place", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_place_object"))

    def calculate_accuracy(self, env_ids):
        """
        Calculate the accuracy of the placement by measuring the distance between the object's center and the goal's center.
        Args:
            env_ids: IDs of environments to calculate accuracy for.
        """
        object_centers = self._objects.get_world_poses()[0][env_ids]  # Get object positions for the specified environments
        goal_place_centers = self._goal_places.get_world_poses()[0][env_ids]  # Get goal_place positions for the specified environments

        # Calculate the Euclidean distance between the object centers and the goal_place centers
        distances = torch.norm(object_centers - goal_place_centers, p=2, dim=-1)
    
        # Convert distances to millimeters (assuming the distances are in meters)
        distances_mm = distances * 1000


        return distances_mm
        
    def post_reset(self):
        """
        Actions to perform after environment reset, including setting initial poses and calculating new targets.
        """
        self.num_arm_dofs = self.get_num_dof() # Retrieve the number of degrees of freedom for the arm
        self.actuated_dof_indices = torch.arange(self.num_arm_dofs, dtype=torch.long, device=self.device) # Indices of actuated degrees of freedom

        # Initialize targets and limits for arm degrees of freedom
        self.arm_dof_targets = torch.zeros((self.num_envs, self._arms.num_dof), dtype=torch.float, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)

        dof_limits = self._dof_limits
        self.arm_dof_lower_limits, self.arm_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        self.arm_dof_default_pos = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device) # Default position for arm degrees of freedom
        self.arm_dof_default_vel = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device) # Default velocity for arm degrees of freedom

        # Retrieve initial poses for end effectors and goals
        self.end_effectors_init_pos, self.end_effectors_init_rot = self._arms._end_effectors.get_world_poses()

        self.goal_pos, self.goal_rot = self._goals.get_world_poses()
        self.goal_pos -= self._env_pos # Adjust goal position relative to environment position

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        """
        Calculate and update metrics after each step, including rewards and success tracking.
        """
        self.fall_dist = 0  # Distance fallen, used for calculating fall penalty
        self.fall_penalty = 0  # Penalty for falling, if applicable

        # Compute rewards and update buffers and success counts
        rewards, resets, goal_resets, progress, successes, cons_successes = compute_arm_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes, self.max_episode_length, self.object_pos, self.object_rot,
            self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale, self.success_tolerance, self.reach_goal_bonus,
            self.fall_dist, self.fall_penalty, self.max_consecutive_successes, self.av_factor
        )

        # Update the accumulated rewards and steps
        self.cumulative_rewards += rewards
        self.mean_rewards = self.cumulative_rewards / self.episode_lengths
        self.extras['cumulative reward'] = self.cumulative_rewards
        self.goal_distances += torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)

        self.episode_lengths += 1

        # Handle resets: calculate average rewards and reset counters
        resets_indices = torch.nonzero(resets).squeeze(-1)
        if len(resets_indices) > 0:
            average_rewards = self.cumulative_rewards[resets_indices] / self.episode_lengths[resets_indices]
            average_distances = self.goal_distances[resets_indices] / self.episode_lengths[resets_indices]

            self.extras['Average reward'] = average_rewards
            self.extras['Average goal distances'] = average_distances

            # Logging and printing for debugging or monitoring
            for idx, avg_reward, avg_dist in zip(resets_indices, average_rewards, average_distances):
                #print(f'Episode {self.episode_count}: Environment {idx} - Average Reward: {avg_reward.item()}, Average Goal Distance: {avg_dist.item()}')
                self.extras[f'Average Reward environment{idx}'] = avg_reward.item()
                self.extras[f'Average Goal Distance environment{idx}'] = avg_dist.item()

                #self.episode_count += 1
            
            # Calculate and log accuracy for successful episodes
            success_indices = resets_indices[successes[resets_indices] > 0]
            if len(success_indices) > 0:
                accuracy_mm = self.calculate_accuracy(success_indices)
                min_accuracy = torch.min(accuracy_mm).item()
                max_accuracy = torch.max(accuracy_mm).item()
                self.extras['min_accuracy_mm'] = min_accuracy  
                self.extras['max_accuracy_mm'] = max_accuracy

                print(f'Episode {self.episode_count}: Min Accuracy: {min_accuracy} mm')


            # Reset the cumulative counters for the next episode
            self.cumulative_rewards[resets_indices] = 0
            self.goal_distances[resets_indices] = 0
            self.episode_lengths[resets_indices] = 0

            # Log the timesteps for successful episodes
            for idx in resets_indices:
                if self.successes[idx] > 0:
                    self.success_timesteps.append(self.timesteps_since_start[idx].item())
                    min_timesteps = torch.min(torch.tensor(self.success_timesteps))
                    self.extras[f'timesteps_to_success'] = min_timesteps
                    self.timesteps_since_start[idx] = 0 

        # Update buffers
        self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes = rewards, resets, goal_resets, progress, successes, cons_successes

        # Update extras with average consecutive successes
        self.extras['consecutive_successes'] = cons_successes.mean()
        self.extras['successes'] = successes.sum()
        self.extras['mean_reward'] = rewards.mean()

        # Print success statistics if enabled
        if self.print_success_stat:
            self.total_resets += resets.sum()
            direct_average_successes = successes.sum()
            self.total_successes += (successes * resets).sum()
            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            if self.total_resets > 0:
                print("Direct average consecutive successes = {:.1f}".format(direct_average_successes / self.total_resets))
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets))

    
    def print_episode_stats(self, resets_indices):
        for idx in resets_indices:
            average_reward = self.episode_rewards[idx] / self.episode_lengths[idx]
            print(f'Episode {self.episode_count}: Environment {idx} - Average Reward: {average_reward.item()}')
            self.episode_count += 1

    def get_object_displacement_tensor(self):
        """
        Returns the displacement tensor for the object relative to the end effector.
        """
        # Define the desired offset from the end effector
        desired_offset = torch.tensor([0.1585, 0.0, 0.0], device=self.device)  # example offset of 0.1 meters in x-direction
        return desired_offset


    def pre_physics_step(self, actions):
        """
        Actions to perform before each physics simulation step, including resetting targets and applying actions.
        
        Args:
            actions: The actions to apply to the environment.
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)  # IDs of environments needing reset
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1) # IDs of environments where goals need resetting

        # Retrieve current positions and orientations for end effectors
        end_effectors_pos, end_effectors_rot = self._arms._end_effectors.get_world_poses()
        
        #self.pick_box(env_ids)
        
        # Reverse the default rotation and rotate the displacement tensor according to the current rotation
        # Update object position and orientation based on end effector states
        self.object_pos = end_effectors_pos + quat_rotate(end_effectors_rot, quat_rotate_inverse(self.end_effectors_init_rot, self.get_object_displacement_tensor()))
        self.object_pos -= self._env_pos # subtract world env pos # Adjust object position relative to environment position
        self.object_rot = end_effectors_rot
        object_pos = self.object_pos + self._env_pos
        object_rot = self.object_rot
        self._objects.set_world_poses(object_pos, object_rot)

        # if only goals need reset, then call set API
        # Reset target poses if necessary
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device) # Clone actions to device
        # Reacher tasks don't require gripper actions, disable it.
        self.actions[:, 5] = 0.0

        if self.use_relative_control:
            # Calculate new target positions for joints based on actions and speed scale
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.arm_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
        else:
            # Scale and apply moving average to actions to calculate new targets
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices] # Update previous targets
        
        # Apply new joint position targets to the robotic arm
        self._arms.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )
        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            # Only retrieve the 0-th joint position even when multiple envs are used
            # Retrieve current joint positions and send them to the real robot if sim2real is enabled and in test mode
            cur_joint_pos = self._arms.get_joint_positions(indices=[0], joint_indices=self.actuated_dof_indices)
            # Send the current joint positions to the real robot
            # Check for joint position bounds and skip sending if out of bounds
            joint_pos = cur_joint_pos[0]
            if torch.any(joint_pos < self.arm_dof_lower_limits) or torch.any(joint_pos > self.arm_dof_upper_limits):
                print("get_joint_positions out of bound, send_joint_pos skipped")
            else:
                self.send_joint_pos(joint_pos)
        self.timesteps_since_start += 1


    def is_done(self):
        """
        Method to determine if the task is complete.
        Must be implemented by subclasses to define specific completion conditions.
        """
        pass

    def reset_target_pose(self, env_ids):
        """
        Resets the target pose for specified environments based on randomization.
        
        Args:
            env_ids: IDs of environments where targets need resetting.
        """
        # reset goal
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device) # Generate random floats for new position and orientation

        new_pos = self.get_reset_target_new_pos(len(env_ids)) # Calculate new position for targets
        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]) # Calculate new orientation

        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids] # add world env pos  # Adjust goal position for world coordinates

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices) # Update world poses for goals
        self.reset_goal_buf[env_ids] = 0 # Reset goal buffer for these environments

    def reset_idx(self, env_ids):
        """
        Resets specified environments to initial states, including arm and target poses.
        
        Args:
            env_ids: IDs of environments to reset.
        """
         
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device) # Generate random floats for resetting

        self.reset_target_pose(env_ids) # Reset target pose based on environment IDs

        # Calculate new positions and velocities for arm degrees of freedom based on randomization
        delta_max = self.arm_dof_upper_limits - self.arm_dof_default_pos
        delta_min = self.arm_dof_lower_limits - self.arm_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5+self.num_arm_dofs] + 1.0) * 0.5

        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos

        dof_vel = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_dofs:5+self.num_arm_dofs*2]

        self.prev_targets[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_dofs] = pos
        self.arm_dof_targets[env_ids, :self.num_arm_dofs] = pos

        self._arms.set_joint_position_targets(self.arm_dof_targets[env_ids], indices)  # Apply new joint position targets
        self._arms.set_joint_positions(dof_pos[env_ids], indices)  # Set joint positions directly
        self._arms.set_joint_velocities(dof_vel[env_ids], indices)  # Set joint velocities
        
        self.progress_buf[env_ids] = 0  # Reset progress buffer
        self.reset_buf[env_ids] = 0  # Reset environment buffer
        self.successes[env_ids] = 0  # Reset successes count

        self.cumulative_rewards[env_ids] = 0
        self.goal_distances[env_ids] = 0

        self.timesteps_since_start[env_ids] = 0  # Reset timesteps counter for reset environments
        self.accuracy[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """
    Randomizes rotation based on input random floats and unit tensors for the x and y axes.
    
    Args:
        rand0: Random float for rotation around the x-axis.
        rand1: Random float for rotation around the y-axis.
        x_unit_tensor: Unit tensor for the x-axis.
        y_unit_tensor: Unit tensor for the y-axis.
        
    Returns:
        A quaternion representing the randomized rotation.
    """
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def compute_arm_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float
):
    """
    Computes rewards for the Reacher task based on distances, orientations, actions, and success conditions.
    
    Args:
        rew_buf (Tensor): Buffer to store computed rewards.
        reset_buf (Tensor): Buffer indicating which environments need to be reset.
        reset_goal_buf (Tensor): Buffer for resetting goals.
        progress_buf (Tensor): Buffer to track progress of the environments.
        successes (Tensor): Tensor tracking the number of successes.
        consecutive_successes (Tensor): Tensor tracking the number of consecutive successes.
        max_episode_length (float): Maximum length of an episode.
        object_pos (Tensor): Positions of objects in the environments.
        object_rot (Tensor): Orientations of objects in the environments.
        target_pos (Tensor): Target positions in the environments.
        target_rot (Tensor): Target orientations in the environments.
        dist_reward_scale (float): Scaling factor for distance-based rewards.
        rot_reward_scale (float): Scaling factor for rotation-based rewards.
        rot_eps (float): Small epsilon value for rotation calculations to improve stability.
        actions (Tensor): Actions taken by the agents.
        action_penalty_scale (float): Scaling factor for penalties based on the magnitude of actions.
        success_tolerance (float): Tolerance for considering an action successful.
        reach_goal_bonus (float): Bonus given for reaching the goal.
        fall_dist (float): Distance fallen, used for calculating penalties.
        fall_penalty (float): Penalty for falling.
        max_consecutive_successes (int): Maximum number of consecutive successes allowed before a reset.
        av_factor (float): Averaging factor used in computing the rolling average of successes.
        
    Returns:
        Tuple containing updated rewards, reset states, goal resets, progress states, success counts, and consecutive success counts.
    """

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1) # Calculate Euclidean distance from object to target
    #print(f'Goal distance: {goal_dist}')

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # Compute angular difference

    # Calculate distance and rotation rewards

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # Compute penalty for actions taken
    action_penalty = torch.sum(actions ** 2, dim=-1)
    #print(f'action penalty: {action_penalty}')

    # Calculate total reward combining all components
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + action_penalty * action_penalty_scale

    # Check and update success conditions based on tolerance thresholds
    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    #print(f'Successes: {successes}')

    # Apply bonus for reaching the goal
    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    #print(f'Reward: {reward}')

    # Determine which environments need resetting based on progress and success conditions
    resets = reset_buf
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    #print(f'Resets: {resets}')

    # Calculate and update statistics for consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)
    #print(f'Cons success: {cons_successes}')
    return reward, resets, goal_resets, progress_buf, successes, cons_successes