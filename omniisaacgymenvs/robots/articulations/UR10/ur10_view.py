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


from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch

class UR10View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "UR10View",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        # Use RigidPrimView instead of XFormPrimView, since the XForm is not updated when running
        self._end_effectors = RigidPrimView(prim_paths_expr="/World/envs/.*/ur10/ee_link", name="end_effector_view", reset_xform_properties=False)

    def get_gripper_positions(self):
        """
        Retrieves the positions of the gripper.
        
        Returns:
            torch.Tensor: The positions of the gripper.
        """
        end_effectors_pos, _ = self._end_effectors.get_world_poses()
        return end_effectors_pos

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)