# Copyright (c) 2018-2022, NVIDIA Corporation
# Copyright (c) 2022-2023, Johnson Sun
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

from omniisaacgymenvs.tasks.UR10_reacher_task import UR10ReacherTask

from omni.isaac.core.utils.torch.rotations import *

import torch

import omni
import carb


class UR10ReacherDemo(UR10ReacherTask):
    """
    A demonstration subclass of UR10ReacherTask for controlling the UR10 robotic arm 
    in a simulated environment with customized camera settings and keyboard interactions.

    Attributes:
        add_noise (bool): Indicates if noise should be added to the observations.
        _current_command (list[float]): Stores the current motion command for the robot arm.
        _prim_selection (omni.usd.PrimSelection): Handles the selection of primitives in the scene.
        _selected_id (int or None): ID of the currently selected environment primitive, if any.
        _previous_selected_id (int or None): Stores the previously selected environment primitive ID.

    Parameters:
        name (str): The name of the task instance.
        sim_config (Config): Configuration object that includes simulation and environment settings.
        env (Environment): The simulation environment.
        offset (list[float] or None): Offset values for the camera position, defaults to None.

    Methods:
        create_camera(): Sets up the camera for the simulation environment.
        set_up_keyboard(): Configures keyboard shortcuts for robot control.
        _on_keyboard_event(event): Handles keyboard input events for robot motion and camera control.
        update_selected_object(): Updates the selected object based on user selection.
        _update_camera(): Adjusts the camera position based on the selected object's position and orientation.
        pre_physics_step(actions): Prepares and processes actions before the physics simulation step.
        post_physics_step(): Processes data after the physics simulation step, updates states and observations.
    """
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        """
        Initializes the UR10ReacherDemo class with specific settings and configurations.

        Parameters:
            name (str): The name of the demo task instance.
            sim_config (Config): Configuration for simulation settings.
            env (Environment): The active simulation environment.
            offset (list[float] or None): Optional offset for camera positioning.
        """
        max_num_envs = 128
        if sim_config.task_config["env"]["numEnvs"] >= max_num_envs:
            print(f"num_envs reduced to {max_num_envs} for this demo.")
            sim_config.task_config["env"]["numEnvs"] = max_num_envs
        UR10ReacherTask.__init__(self, name, sim_config, env)
        self.add_noise = False

        self.create_camera()
        self._current_command = [0.0] * 6
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        return

    def create_camera(self):
        """
        Sets up the default camera and an alternative perspective camera in the simulation environment.
        """
        stage = omni.usd.get_context().get_stage()
        from omni.kit.viewport.utility import get_active_viewport
        #self.view_port = omni.kit.viewport_legacy.get_default_viewport()
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        self.view_port = get_active_viewport()
        self.view_port.set_active_camera(self.camera_path)
        #self.view_port.set_active_camera(self.camera_path)
        camera_prim.GetAttribute("focalLength").Set(8.5)
        self.view_port.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """
        Configures keyboard inputs to control the UR10 robot joints and camera switching.
        """
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._key_to_control = {
            # Joint 0
            "Q": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Joint 1
            "W": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "S": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            # Joint 2
            "E": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            "D": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            # Joint 3
            "R": [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            "F": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            # Joint 4
            "T": [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "G": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            # Joint 5
            "Y": [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            "H": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        """
        Processes keyboard events to control the robot's joints or switch cameras.

        Parameters:
            event (KeyboardEvent): The keyboard event containing the key pressed.
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                self._current_command = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.view_port.get_active_camera() == self.camera_path:
                        self.view_port.set_active_camera(self.perspective_path)
                    else:
                        self.view_port.set_active_camera(self.camera_path)
        else:
            self._current_command = [0.0] * 6

    def update_selected_object(self):
        """
        Updates the currently selected object in the simulation environment, providing feedback if the selection is invalid.
        """
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.view_port.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
            else:
                print("The selected prim was not a UR10")

    def _update_camera(self):
        """
        Adjusts the camera position based on the currently selected object's position and orientation in the simulation.
        """
        base_pos = self.base_pos[self._selected_id, :].clone()
        base_quat = self.base_quat[self._selected_id, :].clone()

        camera_local_transform = torch.tensor([-1.8, 0.0, 0.6], device=self.device)
        camera_pos = quat_apply(base_quat, camera_local_transform) + base_pos

        self.view_port.set_camera_position(self.camera_path, camera_pos[0], camera_pos[1], camera_pos[2], True)
        self.view_port.set_camera_target(self.camera_path, base_pos[0], base_pos[1], base_pos[2]+0.6, True)

    def pre_physics_step(self, actions):
        """
        Applies the current commands to the selected robot arm prior to the physics update step.

        Parameters:
            actions (Tensor): The actions to be applied to the robot arm.

        Returns:
            Tensor: The result after processing actions in the base class's method.
        """
        if self._selected_id is not None:
            actions[self._selected_id, :] = torch.tensor(self._current_command, device=self.device)
        result = super().pre_physics_step(actions)
        if self._selected_id is not None:
            print('selected ur10 id:', self._selected_id)
            print('self.rew_buf[idx]:', self.rew_buf[self._selected_id])
            print('self.object_pos[idx]:', self.object_pos[self._selected_id])
            print('self.goal_pos[idx]:', self.goal_pos[self._selected_id])
        return result

    def post_physics_step(self):
        """
        Updates internal states, calculates rewards, and handles environment resets after the physics simulation step.

        Returns:
            tuple: Contains the updated observation buffer, reward buffer, reset buffer, and any extra information.
        """
        self.progress_buf[:] += 1

        self.update_selected_object()

        if self._selected_id is not None:
            self.reset_buf[self._selected_id] = 0

        self.get_states()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.get_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # Calculate rewards
        self.calculate_metrics()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras