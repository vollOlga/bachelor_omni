# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import time
import numpy as np
import omni
import omni.isaac.cortex.math_util as math_util
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.math import normalized
from omni.isaac.cortex.cortex_world import CortexWorld
from omni.isaac.cortex.df import (
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
from omni.isaac.cortex.dfb import DfDiagnosticsMonitor, DfLift, make_go_home
from omni.isaac.cortex.motion_commander import ApproachParams, MotionCommand, PosePq
from omni.isaac.cortex.obstacle_monitor_context import ObstacleMonitor, ObstacleMonitorContext


class BinState:
    """
    Represents the state of a bin in the bin stacking task.

    Args:
        bin_obj (Object): The bin object.

    Attributes:
        bin_obj (Object): The bin object.
        bin_base (XFormPrim): The base of the bin.
        grasp_T (np.ndarray): The transformation matrix for grasping.
        is_grasp_reached (bool): Indicates if the grasp is reached.
        is_attached (bool): Indicates if the bin is attached.
        needs_flip (bool): Indicates if the bin needs to be flipped.
    """
    def __init__(self, bin_obj):
        self.bin_obj = bin_obj
        self.bin_base = XFormPrim(self.bin_obj.prim_path + "/Collision/Cube_03")
        self.grasp_T = None
        self.is_grasp_reached = None
        self.is_attached = None
        self.needs_flip = None


class FlipStationObstacleMonitor(ObstacleMonitor):
    """
    Monitors obstacles at the flip station.

    Args:
        context (ObstacleMonitorContext): The context for the obstacle monitor.

    Attributes:
        context (ObstacleMonitorContext): The context for the obstacle monitor.
    """
    def __init__(self, context):
        super().__init__([context.world.scene.get_object("flip_station_sphere")])
        self.context = context

    def is_obstacle_required(self):
        """
        Determines if the obstacle is required based on the current position and orientation.

        Returns:
            bool: True if the obstacle is required, False otherwise.
        """
        eff_T = self.context.robot.arm.get_fk_T()
        eff_R, eff_p = math_util.unpack_T(eff_T)
        eff_ax, _, _ = math_util.unpack_R(eff_R)

        grasp_p = self.context.active_bin.grasp_T[:3, 3]
        grasp_ax = self.context.active_bin.grasp_T[:3, 0]
        v = eff_p - grasp_p
        dist = v.dot(grasp_ax)
        orth_dist = np.linalg.norm(v - dist * grasp_ax)
        return not (dist < 0.02 and grasp_ax.dot(eff_ax) > 0.75 and orth_dist < 0.03)


class NavigationObstacleMonitor(ObstacleMonitor):
    """
    Monitors navigation obstacles.

    Args:
        context (ObstacleMonitorContext): The context for the obstacle monitor.

    Attributes:
        context (ObstacleMonitorContext): The context for the obstacle monitor.
    """
    def __init__(self, context):
        obstacles = [
            context.world.scene.get_object("navigation_dome_obs"),
            context.world.scene.get_object("navigation_barrier_obs"),
            context.world.scene.get_object("navigation_flip_station_obs"),
        ]
        super().__init__(obstacles)
        self.context = context

    def is_obstacle_required(self):
        """
        Determines if the obstacle is required based on the current target position.

        Returns:
            bool: True if the obstacle is required, False otherwise.
        """
        target_p, _ = self.context.robot.arm.target_prim.get_world_pose()

        ref_p = np.array([0.6, 0.37, -0.99])
        eff_p = self.context.robot.arm.get_fk_p()

        ref_p[2] = 0.0
        eff_p[2] = 0.0
        target_p[2] = 0.0

        s_target = np.sign(np.cross(target_p, ref_p)[2])
        s_eff = np.sign(np.cross(eff_p, ref_p)[2])
        is_required = s_target * s_eff < 0.0
        return is_required


class BinStackingDiagnostic:
    """
    Represents the diagnostic information for bin stacking.

    Args:
        bin_name (str): The name of the bin.
        bin_base (XFormPrim): The base of the bin.
        grasp (np.ndarray): The grasp transformation matrix.
        grasp_reached (bool): Indicates if the grasp is reached.
        attached (bool): Indicates if the bin is attached.
        needs_flip (bool): Indicates if the bin needs to be flipped.
    """
    def __init__(self, bin_name=None, bin_base=None, grasp=None, grasp_reached=None, attached=None, needs_flip=None):
        self.bin_name = bin_name
        self.bin_base = bin_base
        self.grasp = grasp
        self.grasp_reached = grasp_reached
        self.attached = attached
        self.needs_flip = needs_flip


class BinStackingDiagnosticsMonitor(DfDiagnosticsMonitor):
    """
    Monitors and prints diagnostics for bin stacking.

    Args:
        print_dt (float): The time interval for printing diagnostics.
        diagnostic_fn (function): The function to call for diagnostics.

    Attributes:
        diagnostic_fn (function): The function to call for diagnostics.
    """
    def __init__(self, print_dt=1.0, diagnostic_fn=None):
        super().__init__(print_dt=print_dt)
        self.diagnostic_fn = diagnostic_fn

    def print_diagnostics(self, context):
        """
        Prints the diagnostic information.

        Args:
            context (ObstacleMonitorContext): The context for the diagnostics.
        """
        if context.has_active_bin:
            diagnostic = BinStackingDiagnostic(
                context.active_bin.bin_obj.name,
                context.active_bin.bin_base,
                context.active_bin.grasp_T,
                context.active_bin.is_grasp_reached,
                context.active_bin.is_attached,
                context.active_bin.needs_flip,
            )
        else:
            diagnostic = BinStackingDiagnostic()
        if self.diagnostic_fn:
            self.diagnostic_fn(diagnostic)
        # print("=========== logical state ==========")
        # if context.has_active_bin:
        #     print("active bin info:")
        #     print("- bin_obj.name: {}".format(context.active_bin.bin_obj.name))
        #     print("- bin_base: {}".format(context.active_bin.bin_base))
        #     print("- grasp_T:\n{}".format(context.active_bin.grasp_T))
        #     print("- is_grasp_reached: {}".format(context.active_bin.is_grasp_reached))
        #     print("- is_attached:  {}".format(context.active_bin.is_attached))
        #     print("- needs_flip:  {}".format(context.active_bin.needs_flip))
        # else:
        #     print("<no active bin>")

        # print("------------------------------------")


def get_bin_under(p, stacked_bins):
    """
    Gets the bin under the given position.

    Args:
        p (np.ndarray): The position to check.
        stacked_bins (list): The list of stacked bins.

    Returns:
        BinState: The bin under the given position, or None if not found.
    """
    x, y, z = p
    xy = np.array([x, y])

    for b in reversed(stacked_bins):
        (bin_x, bin_y, bin_z), _ = b.bin_obj.get_world_pose()
        bin_xy = np.array([bin_x, bin_y])
        if np.linalg.norm(bin_xy - xy) < 0.05:
            # We're searching in reverse. Return the first valid candidate.
            return b

    return None


class BinStackingContext(ObstacleMonitorContext):
    """
    Represents the context for bin stacking.

    Args:
        robot (Robot): The robot performing the bin stacking.
        monitor_fn (function): The function to call for monitoring diagnostics.

    Attributes:
        robot (Robot): The robot performing the bin stacking.
        world (CortexWorld): The Cortex world instance.
        diagnostics_monitor (BinStackingDiagnosticsMonitor): The diagnostics monitor.
        flip_station_obs_monitor (FlipStationObstacleMonitor): The flip station obstacle monitor.
        navigation_obs_monitor (NavigationObstacleMonitor): The navigation obstacle monitor.
        stack_xs (np.ndarray): The x coordinates for stacking.
        stack_ys (np.ndarray): The y coordinates for stacking.
        stack_zs (list): The z coordinates for stacking.
        stack_coordinates (list): The list of stack coordinates.
        bins (list): The list of bins.
        active_bin (BinState): The currently active bin.
        stacked_bins (list): The list of stacked bins.
    """
    def __init__(self, robot, monitor_fn=None):
        super().__init__(robot.arm)
        self.robot = robot
        self.world = CortexWorld.instance()
        self.diagnostics_monitor = BinStackingDiagnosticsMonitor(print_dt=1.0, diagnostic_fn=monitor_fn)

        self.flip_station_obs_monitor = FlipStationObstacleMonitor(self)
        self.navigation_obs_monitor = NavigationObstacleMonitor(self)
        self.add_obstacle_monitors([self.flip_station_obs_monitor, self.navigation_obs_monitor])

        h = 0.135
        e = 0.0075
        x_shift = 0.05

        full_stack = True
        if full_stack:
            self.stack_xs = np.array([1.00, 0.79, 0.58]) + x_shift
            self.stack_ys = [-0.62, -0.31, 0]
            self.stack_zs = [-0.59374 + (i * h) + h / 2 + e for i in range(4)]
        else:
            self.stack_xs = np.array([1.00, 0.79]) + x_shift
            self.stack_ys = [-0.62, -0.31]
            self.stack_zs = [-0.59374 + (i * h) + h / 2 + e for i in range(3)]

        self.stack_coordinates = []
        for zi in range(len(self.stack_zs)):
            for yi in range(len(self.stack_ys)):
                for xi in range(len(self.stack_xs)):
                    coords = np.array([self.stack_xs[xi], self.stack_ys[yi], self.stack_zs[zi]])
                    self.stack_coordinates.append(coords)

        self.bins = []
        self.active_bin = None
        self.stacked_bins = []

        self.add_monitors(
            [
                BinStackingContext.monitor_bins,
                BinStackingContext.monitor_active_bin,
                BinStackingContext.monitor_active_bin_grasp_T,
                BinStackingContext.monitor_active_bin_grasp_reached,
                self.diagnostics_monitor.monitor,
            ]
        )

    def reset(self):
        """
        Resets the context by clearing the bins and stacked bins.

        Returns:
            None
        """
        super().reset()
        self.bins.clear()
        self.active_bin = None
        self.stacked_bins.clear()

    @property
    def stack_complete(self):
        """
        Checks if the stack is complete.

        Returns:
            bool: True if the stack is complete, False otherwise.
        """
        return len(self.stacked_bins) == len(self.stack_coordinates)

    @property
    def elapse_time(self):
        """
        Gets the elapsed time since the start.

        Returns:
            float: The elapsed time.
        """
        return time.time() - self.start_time

    @property
    def has_active_bin(self):
        """
        Checks if there is an active bin.

        Returns:
            bool: True if there is an active bin, False otherwise.
        """
        return self.active_bin is not None

    def monitor_bins(self):
        """
        Monitors and updates the list of bins in the context.

        Returns:
            None
        """
        if self.active_bin is None:
            self.conveyor_bin = None
            min_y = None

            # Check whether there's a new bin in the world.
            bin_obj = self.world.scene.get_object("bin_{}".format(len(self.bins)))
            if bin_obj is not None:
                self.bins.append(BinState(bin_obj))

            # Cycle through all bins and find the bin in the active region with smallest y value.
            for bin_state in self.bins:
                p, _ = bin_state.bin_obj.get_world_pose()

                # Check whether it's on the conveyor in the active region.
                x, y, z = p
                if 0.0 < y and y < 0.7 and -0.4 < x and x < 0.4:
                    if self.active_bin is None or y < min_y:
                        self.active_bin = bin_state
                        min_y = y

    def monitor_active_bin(self):
        """
        Monitors and updates the active bin status.

        Returns:
            None
        """
        if self.active_bin is not None:
            p, _ = self.active_bin.bin_obj.get_world_pose()
            if p[2] < -1.0:
                self.active_bin = None

    def monitor_active_bin_grasp_T(self):
        """
        Monitors and updates the grasp transformation for the active bin.

        Returns:
            None
        """
        if self.active_bin is not None:
            bin_T = math_util.pq2T(*self.active_bin.bin_base.get_world_pose())
            bin_R, bin_p = math_util.unpack_T(bin_T)
            bin_ax, bin_ay, bin_az = math_util.unpack_R(bin_R)

            self.active_bin.is_rightside_up = False
            up_vec = np.array([0.0, 0.0, 1.0])
            if self.active_bin.is_attached:
                fk_R = self.robot.arm.get_fk_R()
                fk_x, _, _ = math_util.unpack_R(fk_R)
                up_vec = -fk_x

            margin = 0.0
            base_width = 0.00233

            self.active_bin.needs_flip = up_vec.dot(bin_az) > 0.0
            if self.active_bin.needs_flip:
                # The bin is right side up (opens upward)
                target_ax = -bin_az
                margin = 0.0025
            else:
                # The bin is upside down (opens downward)
                target_ax = bin_az
                margin = -0.0025
            if bin_ax[1] < 0.0:
                # x axis is pointing toward the robot
                target_ay = -bin_ax
            else:
                target_ay = bin_ax
            target_az = np.cross(target_ax, target_ay)
            target_p = bin_p + margin * bin_az

            target_T = math_util.pack_Rp(math_util.pack_R(target_ax, target_ay, target_az), target_p)
            self.active_bin.grasp_T = target_T

    def monitor_active_bin_grasp_reached(self):
        """
        Monitors and updates if the grasp for the active bin is reached.

        Returns:
            None
        """
        if self.has_active_bin:
            fk_T = self.robot.arm.get_fk_T()
            self.active_bin.is_grasp_reached = math_util.transforms_are_close(
                self.active_bin.grasp_T, fk_T, p_thresh=0.005, R_thresh=0.01
            )
            # We can be looser with this proximity check.
            self.active_bin.is_attached = (
                math_util.transforms_are_close(self.active_bin.grasp_T, fk_T, p_thresh=0.1, R_thresh=1.0)
                and self.robot.suction_gripper.is_closed()
            )

    def mark_active_bin_as_complete(self):
        """
        Marks the active bin as complete and adds it to the list of stacked bins.

        Returns:
            None
        """
        self.stacked_bins.append(self.active_bin)
        self.active_bin = None


class Move(DfState):
    """
    Represents a movement state for the robot arm.

    Args:
        p_thresh (float): The position threshold for completion.
        R_thresh (float): The rotation threshold for completion.

    Attributes:
        p_thresh (float): The position threshold for completion.
        R_thresh (float): The rotation threshold for completion.
        command (MotionCommand): The motion command to execute.
    """
    def __init__(self, p_thresh, R_thresh):
        self.p_thresh = p_thresh
        self.R_thresh = R_thresh
        self.command = None

    def update_command(self, command):
        """
        Updates the motion command.

        Args:
            command (MotionCommand): The new motion command.

        Returns:
            None
        """
        self.command = command

    def step(self):
        """
        Executes the motion command and checks if the movement is complete.

        Returns:
            DfState: The next state or None if complete.
        """
        self.context.robot.arm.send(self.command)

        fk_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(
            self.command.target_pose.to_T(), fk_T, p_thresh=self.p_thresh, R_thresh=self.R_thresh
        ):
            return None
        return self


class MoveWithNavObs(Move):
    """
    Represents a movement state for the robot arm with navigation obstacle monitoring.

    Args:
        p_thresh (float): The position threshold for completion.
        R_thresh (float): The rotation threshold for completion.
    """
    def enter(self):
        """
        Activates the navigation obstacle monitor.

        Returns:
            None
        """
        super().enter()
        self.context.navigation_obs_monitor.activate_autotoggle()

    def exit(self):
        """
        Deactivates the navigation obstacle monitor.

        Returns:
            None
        """
        super().exit()
        self.context.navigation_obs_monitor.deactivate_autotoggle()


class ReachToPick(MoveWithNavObs):
    """
    Represents a state to reach and pick a bin.

    Args:
        p_thresh (float): The position threshold for completion.
        R_thresh (float): The rotation threshold for completion.
    """
    def __init__(self):
        super().__init__(p_thresh=0.001, R_thresh=2.0)

    def enter(self):
        """
        Activates the flip station obstacle monitor and updates the motion command for reaching.

        Returns:
            None
        """
        super().enter()
        self.context.flip_station_obs_monitor.activate_autotoggle()

    def step(self):
        """
        Updates the motion command and checks if the reach movement is complete.

        Returns:
            DfState: The next state or None if complete.
        """
        R, p = math_util.unpack_T(self.context.active_bin.grasp_T)
        ax, ay, az = math_util.unpack_R(R)

        posture_config = np.array([-1.2654234, -2.9708025, -2.219733, 0.6445836, 1.5186214, 0.30098662])
        if self.context.active_bin.needs_flip:
            approach_length = 0.3
        else:
            approach_length = 0.1

        self.update_command(
            MotionCommand(
                target_pose=PosePq(p, math_util.matrix_to_quat(R)),
                approach_params=ApproachParams(direction=approach_length * ax, std_dev=0.005),
                posture_config=posture_config,
            )
        )

        return super().step()

    def exit(self):
        """
        Deactivates the flip station obstacle monitor.

        Returns:
            None
        """
        super().exit()
        self.context.flip_station_obs_monitor.deactivate_autotoggle()


class ReachToPlace(MoveWithNavObs):
    """
    Represents a state to reach and place a bin.

    Args:
        p_thresh (float): The position threshold for completion.
        R_thresh (float): The rotation threshold for completion.
    """
    def __init__(self):
        super().__init__(p_thresh=0.005, R_thresh=2.0)

    def enter(self):
        """
        Updates the target position and orientation for placing the bin.

        Returns:
            None
        """
        super().enter()

        self.target_p = self.context.stack_coordinates[len(self.context.stacked_bins)]
        self.bin_under = get_bin_under(self.target_p, self.context.stacked_bins)

        target_ax = np.array([0.0, 0.0, -1.0])
        target_az = np.array([0.0, -1.0, 0.0])
        target_ay = np.cross(target_az, target_ax)
        self.target_R = math_util.pack_R(target_ax, target_ay, target_az)

    def step(self):
        """
        Updates the motion command and checks if the place movement is complete.

        Returns:
            DfState: The next state or None if complete.
        """
        if self.bin_under is not None:
            bin_under_p, _ = self.bin_under.bin_obj.get_world_pose()
            bin_grasped_p, _ = self.context.active_bin.bin_obj.get_world_pose()
            xy_err = bin_under_p[:2] - bin_grasped_p[:2]
            if np.linalg.norm(xy_err) < 0.02:
                self.target_p[:2] += 0.1 * (bin_under_p[:2] - bin_grasped_p[:2])

        target_pose = PosePq(self.target_p, math_util.matrix_to_quat(self.target_R))

        approach_params = ApproachParams(direction=0.15 * np.array([0.0, 0.0, -1.0]), std_dev=0.005)
        posture_config = self.context.robot.default_config
        self.update_command(
            MotionCommand(target_pose=target_pose, approach_params=approach_params, posture_config=posture_config)
        )

        return super().step()


class CloseSuctionGripperWithRetries(DfState):
    """
    Represents a state to close the suction gripper with retries.

    Args:
        None
    """
    def enter(self):
        """
        Initiates the closing of the suction gripper.

        Returns:
            None
        """
        pass

    def step(self):
        """
        Checks if the suction gripper is closed, retries if not.

        Returns:
            DfState: The next state or None if complete.
        """
        gripper = self.context.robot.suction_gripper
        gripper.close()
        if gripper.is_closed():
            return None
        return self


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
            None
        """
        return None


class DoNothing(DfState):
    """
    Represents a state to do nothing.

    Args:
        None
    """
    def enter(self):
        """
        Clears the robot arm commands.

        Returns:
            None
        """
        self.context.robot.arm.clear()

    def step(self):
        """
        Prints the current pose of the target prim.

        Returns:
            None
        """
        print(self.context.robot.arm.target_prim.get_world_pose())
        return self


class LiftAndTurn(Move):
    """
    Represents a state to lift and turn the robot arm.

    Args:
        None
    """
    def __init__(self):
        super().__init__(p_thresh=0.05, R_thresh=0.1)

    def step(self):
        """
        Updates the motion command for lifting and turning.

        Returns:
            DfState: The next state or None if complete.
        """
        home_config = self.context.robot.default_config
        home_T = self.context.robot.arm.get_fk_T(config=home_config)

        p, q = math_util.T2pq(home_T)
        p += 0.5 * normalized(np.array([0.0, -0.5, -1.0]))
        self.target_pose = PosePq(p, q)
        self.update_command(MotionCommand(self.target_pose, posture_config=home_config))

        return super().step()


class PickBin(DfStateMachineDecider):
    """
    Represents a state machine for picking a bin.

    Args:
        None
    """
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPick(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    CloseSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.3), activity_duration=0.4),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class FlipBin(DfStateMachineDecider):
    """
    Represents a state machine for flipping a bin.

    Args:
        None
    """
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    LiftAndTurn(),
                    MoveToFlipStation(),
                    DfSetLockState(set_locked_to=True, decider=self),
                    OpenSuctionGripper(),
                    ReleaseFlipStationBin(),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class PlaceBin(DfStateMachineDecider):
    """
    Represents a state machine for placing a bin.

    Args:
        None
    """
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPlace(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    OpenSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.1), activity_duration=0.25),
                    DfWriteContextState(lambda ctx: ctx.mark_active_bin_as_complete()),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class MoveToFlipStation(DfState):
    """
    Represents a state to move to the flip station.

    Args:
        None
    """
    def __init__(self):
        self.target_pose = PosePq(
            np.array([0.7916634, 0.73902607, -0.02897218]), np.array([0.52239186, 0.6296602, -0.5042411, 0.27636158])
        )

        self.approach_params = ApproachParams(direction=0.4 * normalized(np.array([0.5, -0.3, -0.75])), std_dev=0.05)

        self.posture_config = np.array(
            [
                -2.1273114681243896,
                -3.004627227783203,
                -1.0576069355010986,
                -0.5193580389022827,
                -1.0809129476547241,
                2.0418107509613037,
            ]
        )

    def enter(self):
        """
        Sends the motion command to move to the flip station.

        Returns:
            None
        """
        motion_command = MotionCommand(
            target_pose=self.target_pose, approach_params=self.approach_params, posture_config=self.posture_config
        )
        self.context.robot.arm.send(motion_command)

    def step(self):
        """
        Checks if the movement to the flip station is complete.

        Returns:
            DfState: The next state or None if complete.
        """
        fk_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(self.target_pose.to_T(), fk_T, p_thresh=0.005, R_thresh=2.0, verbose=False):
            return None
        return self


class ReleaseFlipStationBin(DfState):
    """
    Represents a state to release the bin at the flip station.

    Args:
        None
    """
    def enter(self):
        """
        Initiates the motion command to release the bin at the flip station.

        Returns:
            None
        """
        self.entry_time = time.time()

        # Get some info about the current end-effector transform.
        fk_T = self.context.robot.arm.get_fk_T()
        fk_R, fk_p = math_util.unpack_T(fk_T)
        ax, ay, az = math_util.unpack_R(fk_R)

        v = normalized(np.array([-1.0, -0.3, 0.0]))  # Hard-coded vector pointing approx toward base.
        toward_base_alpha = 0.2
        target_p = fk_p - 0.3 * ax + toward_base_alpha * v
        self.target_p = target_p
        self.ax = ax
        self.v = v

        target_ax = normalized(np.array([1.0, -0.0, 0.0]))
        target_ay = normalized(np.array([0.0, -1.0, 0.0]))
        target_az = np.cross(target_ax, target_ay)
        target_R = math_util.pack_R(target_ax, target_ay, target_az)

        # This target pose is a little below the bin, but off to the side with the end
        # effector angled horizontally. It gets the end-effector out of the flip station
        # collision region and allows us to turn on that obstacle for now moving to pick
        # the bin.
        self.target_pose = PosePq(target_p, math_util.matrix_to_quat(target_R))
        motion_command = MotionCommand(
            target_pose=self.target_pose,
            approach_params=ApproachParams(direction=toward_base_alpha * v, std_dev=0.1),
            posture_config=self.context.robot.get_joint_positions().astype(float),
        )
        self.context.robot.arm.send(motion_command)

    def step(self):
        """
        Checks if the release movement is complete.

        Returns:
            DfState: The next state or None if complete.
        """
        # Exit (return None) when the end-effector is within 15cm of the target position.
        fk_p = self.context.robot.arm.get_fk_p()
        dist_to_target = np.linalg.norm(self.target_pose.p - fk_p)
        if dist_to_target < 0.15:
            return None
        return self


class Dispatch(DfDecider):
    """
    Represents the main decision-making state machine for the bin stacking task.

    Args:
        None
    """
    def __init__(self):
        super().__init__()

        self.add_child("flip_bin", FlipBin())
        self.add_child("pick_bin", PickBin())
        self.add_child("place_bin", PlaceBin())
        self.add_child("go_home", make_go_home())
        self.add_child("do_nothing", DfStateMachineDecider(DoNothing()))

    def decide(self):
        """
        Decides the next state based on the current context.

        Returns:
            DfDecision: The decision for the next state.
        """
        if self.context.stack_complete:
            return DfDecision("go_home")

        if self.context.has_active_bin:
            if not self.context.active_bin.is_attached:
                return DfDecision("pick_bin")
            elif self.context.active_bin.needs_flip:
                return DfDecision("flip_bin")
            else:
                return DfDecision("place_bin")
        else:
            return DfDecision("go_home")


def make_decider_network(robot, monitor_fn):
    """
    Creates the decision-making network for the bin stacking task.

    Args:
        robot (Robot): The robot performing the bin stacking.
        monitor_fn (function): The function to call for monitoring diagnostics.

    Returns:
        DfNetwork: The decision-making network.
    """
    return DfNetwork(Dispatch(), context=BinStackingContext(robot, monitor_fn))
