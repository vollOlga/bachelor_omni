from abc import abstractmethod
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.core.utils.torch import scale, unscale
from omni.isaac.gym.vec_env import VecEnvBase

import numpy as np
import torch

class GripTask(RLTask):
    """
    Task to control a robot with 6 joints and a gripper to reach, orient, and grasp a target object.
    """
    def __init__(self, name: str, env: VecEnvBase, offset=None):
        """
        Initializes the GripTask with the robot's configuration and the environment settings.

        Args:
            name (str): Name of the task for identification.
            env (VecEnvBase): Vectorized environment instance this task will operate within.
            offset (optional): Offset parameter used for environment setup. Default is None.
        """
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # Reward and movement parameters
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.grip_reward_scale = self._task_cfg["env"]["gripRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]

        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.grip_success_bonus = self._task_cfg["env"]["gripSuccessBonus"]

        # Robot joint configuration
        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        
        # Initialize task with super class
        RLTask.__init__(self, name, env)

        # Initializing tensors for position and orientation
        self.reset_buffers()

        self.initialize_robots_and_env()

    def reset_buffers(self):
        """
        Initialize or reset buffers and tensors used for task computations.
        """
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def initialize_robots_and_env(self):
        """
        Setup robot initial configuration and environment references.
        """
        self.robot_start_positions = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)  # 6 joints + 1 gripper
        self.robot_start_orientations = torch.full((self.num_envs, 7), np.pi, dtype=torch.float, device=self.device)  # Assume all start with pi orientation

    def set_up_scene(self, scene: Scene) -> None:
        """
        Set up the scene for the task by configuring the robot and adding necessary assets.
        """
        self._stage = get_current_stage()
        self._assets_root_path = 'omniverse://localhost/Projects/YourProjectPath'

        # Loading robot and target objects
        self.load_robot()
        self.load_target_object()

        super().set_up_scene(scene)

    def load_robot(self):
        """
        Load and configure the robot model in the simulation.
        """
        # Example path and setup, modify as needed
        robot_path = f"{self._assets_root_path}/Robots/robot_with_gripper.usd"
        add_reference_to_stage(robot_path, "/World/robot")
        self.robot_prim = XFormPrim(
            prim_path="/World/robot",
            name="robot",
            translation=self.robot_start_positions[0],
            orientation=torch.tensor([0, 0, 0, 1], dtype=torch.float, device=self.device)  # Identity quaternion
        )

    def load_target_object(self):
        """
        Load and position the target object in the simulation.
        """
        object_path = f"{self._assets_root_path}/Props/object_to_grasp.usd"
        add_reference_to_stage(object_path, "/World/target_object")
        self.object_prim = XFormPrim(
            prim_path="/World/target_object",
            name="target_object",
            translation=torch.tensor([1, 1, 0], dtype=torch.float, device=self.device),  # Example position
            orientation=torch.tensor([0, 0, 0, 1], dtype=torch.float, device=self.device)  # Identity quaternion
        )

    def compute_rewards(self):
        current_distance = torch.norm(self.robot_end_effector_pos - self.object_pos, p=2, dim=-1)
        angle_difference = self.calculate_angle_difference(self.robot_gripper_orientation, self.object_orientation)
        is_object_gripped = self.check_if_gripped()  # This should return a boolean tensor

        distance_reward = -self.dist_reward_scale * current_distance
        orientation_reward = -self.rot_reward_scale * angle_difference
        grip_bonus = self.grip_success_bonus * is_object_gripped.float()
        action_penalty = -self.action_penalty_scale * torch.sum(self.last_actions**2, dim=-1)

        total_reward = distance_reward + orientation_reward + grip_bonus + action_penalty
        return total_reward
    
    def calculate_angle_difference(self, quaternion1, quaternion2):
        """
        Calculate the angular difference between two quaternions. This function computes the angle required to rotate
        from quaternion1 to quaternion2. The result is the smallest angle between two orientations, which is useful
        for tasks that involve aligning objects, such as gripping.

        Args:
            quaternion1 (torch.Tensor): A tensor of quaternions representing the first orientation.
            quaternion2 (torch.Tensor): A tensor of quaternions representing the second orientation.

        Returns:
            torch.Tensor: A tensor containing the angular differences in radians between each pair of quaternions.
        """
        # Ensure the quaternions are normalized
        quaternion1 = quaternion1 / quaternion1.norm(dim=-1, keepdim=True)
        quaternion2 = quaternion2 / quaternion2.norm(dim=-1, keepdim=True)

        # Calculate the relative quaternion from quaternion1 to quaternion2
        q_rel = quaternion_multiply(quaternion_conjugate(quaternion1), quaternion2)

        # Ensure the scalar part is non-negative. If negative, invert the quaternion (equivalent rotation)
        q_rel[...,:1] = torch.abs(q_rel[...,:1])

        # Calculate the angle using the scalar part of the quaternion (w = cos(theta/2))
        angle = 2 * torch.acos(torch.clamp(q_rel[..., 0], -1.0, 1.0))

        return angle
    
    def quaternion_multiply(q1, q2):
        """ Multiply two quaternions. """
        # Extract parts of each quaternion
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Compute product
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((w, x, y, z), dim=-1)

    def quaternion_conjugate(q):
        """ Return the conjugate of a quaternion. """
        q_conj = q.clone()
        q_conj[..., 1:] = -q_conj[..., 1:]  # negate the vector part
        return q_conj



    @abstractmethod
    def get_robot_dofs(self):
        """
        Returns the degrees of freedom for the robot, including the gripper.
        """
        return 7  # 6 joints + 1 gripper

    @abstractmethod
    def apply_actions(self, actions):
        """
        Apply the provided actions to the robot joints and gripper.
        """
        pass

    def check_if_gripped(self):
        """
        Determine if the gripper has successfully gripped the target object.
        This could involve checking conditions such as the distance between the gripper and the object,
        the state of the gripper (open or closed), and whether the object is being held stationary
        relative to the gripper's movement.

        Returns:
            torch.Tensor: A boolean tensor indicating whether each instance in the environment
            has successfully gripped the object.
        """
        gripped = (self.gripper_state == 'closed') & (torch.norm(self.gripper_position - self.object_position, dim=-1) < self.grip_tolerance)
        return gripped

    @abstractmethod
    def check_success(self):
        """
        Check if the robot successfully reached and gripped the target object.
        """
        pass

    @abstractmethod
    def compute_rewards(self):
        """
        Compute rewards based on the robot's performance in reaching and gripping the target object.
        """
        pass

    @abstractmethod
    def reset_simulation(self):
        """
        Reset the simulation to start a new episode. This includes resetting the robot's position and orientation,
        possibly repositioning the target object, and resetting any simulation state variables.
        """
        # Reset robot and object to initial conditions
        pass

    @abstractmethod
    def get_observations(self):
        """
        Gather and return the current observations from the environment. This may include the positions and velocities
        of the robot's joints, the position and orientation of the gripper, and the relative position of the object to the gripper.

        Returns:
            dict: A dictionary containing tensors of all relevant observations.
        """
        observations = {
            'joint_positions': self.robot_joint_positions,
            'joint_velocities': self.robot_joint_velocities,
            'gripper_position': self.gripper_position,
            'gripper_state': self.gripper_state,
            'object_position': self.object_position,
            'relative_position': self.object_position - self.gripper_position
        }
        return observations


