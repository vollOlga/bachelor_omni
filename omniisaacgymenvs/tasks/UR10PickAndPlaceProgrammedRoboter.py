from pxr import Usd, UsdGeom, UsdPhysics, Gf
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.robots.robot import Robot
from typing import Optional
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import torch
import math
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.scenes.scene import Scene
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
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

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1'
        self.initialize_robot(scene)
        self.get_object()
        self.get_goal()
        super().set_up_scene(scene)
        scene.add(self._arms)
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

    def initialize_robot(self, scene):
        try:
            # UR10 Robot Base
            robot_prim_path = self.default_zero_env_path + "/ur10"
            robot_stage_path = "/World/envs/env_0/ur10"

            UsdGeom.Xform.Define(self._stage, robot_prim_path)
            robot_prim = self._stage.GetPrimAtPath(robot_prim_path)

            ur10 = UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

            # Add links and joints to the UR10 robot
            links = []
            joints = []

            link_paths = [
                robot_prim_path + "/base_link",
                robot_prim_path + "/shoulder_link",
                robot_prim_path + "/upper_arm_link",
                robot_prim_path + "/forearm_link",
                robot_prim_path + "/wrist_1_link",
                robot_prim_path + "/wrist_2_link",
                robot_prim_path + "/wrist_3_link"
            ]

            for link_path in link_paths:
                links.append(UsdGeom.Xform.Define(self._stage, link_path))

            joint_paths = [
                (links[0], links[1], "shoulder_pan_joint"),
                (links[1], links[2], "shoulder_lift_joint"),
                (links[2], links[3], "elbow_joint"),
                (links[3], links[4], "wrist_1_joint"),
                (links[4], links[5], "wrist_2_joint"),
                (links[5], links[6], "wrist_3_joint")
            ]

            for parent_link, child_link, joint_name in joint_paths:
                joint = UsdPhysics.RevoluteJoint.Define(self._stage, parent_link.GetPath().AppendChild(joint_name))
                joint.GetBody0Rel().SetTargets([parent_link.GetPath()])
                joint.GetBody1Rel().SetTargets([child_link.GetPath()])
                joints.append(joint)

            # Define Gripper
            gripper = ParallelGripper(
                end_effector_prim_path=robot_prim_path + "/wrist_3_link",
                joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
                joint_opened_positions=np.array([0, 0]),
                joint_closed_positions=np.array([0.628, -0.628]),
                action_deltas=np.array([-0.628, 0.628]),
            )

            ur10_robot = SingleManipulator(
                prim_path=robot_prim_path,
                name="UR10",
                end_effector_prim_name="wrist_3_link",
                gripper=gripper
            )

            joints_default_positions = np.zeros(12)
            joints_default_positions[7] = 0.628
            joints_default_positions[8] = 0.628
            joints_default_positions_tensor = torch.from_numpy(joints_default_positions).float().to("cuda:0")
            ur10_robot.set_joints_default_state(positions=joints_default_positions_tensor)

            self._arms = ur10_robot

            if self._arms and self._arms._end_effectors:
                try:
                    scene.add(self._arms._end_effectors)
                except Exception as e:
                    print(f"Failed to add end effector to the scene: {e}")
            else:
                print("Failed to initialize the UR10 robot or end effectors.")
        except Exception as e:
            print(f"UR10 initialization failed: {e}")

    def get_arm_view(self, scene):
        try:
            arm_view = UR10View(prim_paths_expr="/World/envs/.*/ur10/ur10", name="ur10_view")
            scene.add(arm_view._end_effectors)
            return arm_view
        except Exception as e:
            print(f"Error in getting arm view: {e}")
            return None

    def get_observations(self):
        # Retrieve the current state of the environment and robot
        end_effector_pos, end_effector_rot = self._arms._end_effectors.get_world_poses()
        object_pos, object_rot = self._objects.get_world_poses()
        goal_pos, goal_rot = self._goals.get_world_poses()

        # Concatenate and normalize observations
        observations = torch.cat(
            (
                end_effector_pos,
                end_effector_rot,
                object_pos,
                object_rot,
                goal_pos,
                goal_rot,
                self._arms.get_joint_positions(),
                self._arms.get_joint_velocities(),
            ),
            dim=-1,
        )
        return observations

    def get_reset_target_new_pos(self, n_reset_envs):
        # Generate new positions for resetting the target
        return torch.rand((n_reset_envs, 3), device=self.device) * 2 - 1

    def send_joint_pos(self, joint_pos):
        # Send the joint positions to the real robot if needed
        if self._task_cfg['sim2real']['enabled']:
            self.real_world_ur10.send_joint_positions(joint_pos)

    def is_done(self):
        # Determine if the episode is done
        return self.progress_buf >= self.max_episode_length

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
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5+self.num_arm_dofs] + 1.0) * 0.5
        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos
        dof_vel = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_dofs:5+self.num_arm_dofs*2]
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
        self.current_episode += 1

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
    fall_penalty: float, max_consecutive_successes: int, av_factor: float
):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale
    action_penalty = torch.sum(actions ** 2, dim=-1)
    reward = dist_rew + action_penalty * action_penalty_scale
    goal_resets = torch.where(torch.abs(goal_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    resets = reset_buf
    if max_consecutive_successes > 0:
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)
    return reward, resets, goal_resets, progress_buf, successes, cons_successes

class UR10PickAndPlaceTask(PickAndPlaceTask):
    def __init__(self, name: str, sim_config: SimConfig, env: VecEnvBase, offset=None) -> None:
        self._device = "cuda:0" 
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self.end_effectors_init_rot = torch.tensor([1, 0, 0, 0], device=self._device)  # w, x, y, z
        self._gripper = None

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full"]):
            raise Exception("Unknown type of observations!\nobservationType should be one of: [full]")
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "full": 43,
            # Observations
        }

        self.object_scale = torch.tensor([1.0] * 3)
        self.goal_scale = torch.tensor([2.0] * 3)
        self.place_goal_scale = torch.tensor([2.0] * 3)

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
                [-2 * pi, 2 * pi],
                [-pi + pi / 8, 0 - pi / 8],
                [-pi + pi / 8, pi - pi / 8],
                [-pi, 0],
                [-pi, pi],
                [-2 * pi, 2 * pi],
            ]], dtype=torch.float32, device=self._cfg["sim_device"])

        PickAndPlaceTask.__init__(self, name=name, env=env)

        # Setup Sim2Real
        sim2real_config = self._task_cfg['sim2real']
        if sim2real_config['enabled'] and self.test and self.num_envs == 1:
            self.act_moving_average /= 5  # Reduce moving speed
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
        Configures and retrieves an instance of the UR10 robot arm with a gripper.
        Parameters:
            None
        Return:
            None: The function sets up the UR10 robot within the simulation environment but does not return anything.
        '''
        # Path to the robot USD on the Omniverse server
        robot_prim_path = self.default_zero_env_path + "/ur10"
        robot_stage_path = "/World/envs/env_0/ur10"

        UsdGeom.Xform.Define(self._stage, robot_prim_path)
        robot_prim = self._stage.GetPrimAtPath(robot_prim_path)

        ur10 = UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

        # Add links and joints to the UR10 robot
        links = []
        joints = []

        link_paths = [
            robot_prim_path + "/base_link",
            robot_prim_path + "/shoulder_link",
            robot_prim_path + "/upper_arm_link",
            robot_prim_path + "/forearm_link",
            robot_prim_path + "/wrist_1_link",
            robot_prim_path + "/wrist_2_link",
            robot_prim_path + "/wrist_3_link"
        ]

        for link_path in link_paths:
            links.append(UsdGeom.Xform.Define(self._stage, link_path))

        joint_paths = [
            (links[0], links[1], "shoulder_pan_joint"),
            (links[1], links[2], "shoulder_lift_joint"),
            (links[2], links[3], "elbow_joint"),
            (links[3], links[4], "wrist_1_joint"),
            (links[4], links[5], "wrist_2_joint"),
            (links[5], links[6], "wrist_3_joint")
        ]

        for parent_link, child_link, joint_name in joint_paths:
            joint = UsdPhysics.RevoluteJoint.Define(self._stage, parent_link.GetPath().AppendChild(joint_name))
            joint.GetBody0Rel().SetTargets([parent_link.GetPath()])
            joint.GetBody1Rel().SetTargets([child_link.GetPath()])
            joints.append(joint)

        # Define Gripper
        gripper = ParallelGripper(
            end_effector_prim_path=robot_prim_path + "/wrist_3_link",
            joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0.628, -0.628]),
            action_deltas=np.array([-0.628, 0.628]),
        )

        ur10_robot = SingleManipulator(
            prim_path=robot_prim_path,
            name="UR10",
            end_effector_prim_name="wrist_3_link",
            gripper=gripper
        )

        # Set default joint positions if necessary
        joints_default_positions = np.zeros(12)  # Assuming 12 joints including the gripper
        joints_default_positions[7] = 0.628  # Assuming these indices are for the gripper
        joints_default_positions[8] = 0.628
        joints_default_positions_tensor = torch.from_numpy(joints_default_positions).float()
        joints_default_positions_tensor = joints_default_positions_tensor.to("cuda:0")

        ur10_robot.set_joints_default_state(positions=joints_default_positions_tensor)

        # Setting this manipulator as the primary manipulator if needed
        self._arms = ur10_robot
        if self._arms and self._arms._end_effectors:
            try:
                scene.add(self._arms._end_effectors)
            except Exception as e:
                print(f"Failed to add end effector to the scene: {e}")
        else:
            print("Failed to initialize the UR10 robot or end effectors.")
        except Exception as e:
            print(f"UR10 initialization failed: {e}")

    def get_arm_view(self, scene):
        try:
            arm_view = UR10View(prim_paths_expr="/World/envs/.*/ur10/ur10", name="ur10_view")
            scene.add(arm_view._end_effectors)
            return arm_view
        except Exception as e:
            print(f"Error in getting arm view: {e}")
            return None

    def get_observations(self):
        # Retrieve the current state of the environment and robot
        end_effector_pos, end_effector_rot = self._arms._end_effectors.get_world_poses()
        object_pos, object_rot = self._objects.get_world_poses()
        goal_pos, goal_rot = self._goals.get_world_poses()

        # Concatenate and normalize observations
        observations = torch.cat(
            (
                end_effector_pos,
                end_effector_rot,
                object_pos,
                object_rot,
                goal_pos,
                goal_rot,
                self._arms.get_joint_positions(),
                self._arms.get_joint_velocities(),
            ),
            dim=-1,
        )
        return observations

    def get_reset_target_new_pos(self, n_reset_envs):
        # Generate new positions for resetting the target
        return torch.rand((n_reset_envs, 3), device=self.device) * 2 - 1

    def send_joint_pos(self, joint_pos):
        # Send the joint positions to the real robot if needed
        if self._task_cfg['sim2real']['enabled']:
            self.real_world_ur10.send_joint_positions(joint_pos)

    def is_done(self):
        # Determine if the episode is done
        return self.progress_buf >= self.max_episode_length
