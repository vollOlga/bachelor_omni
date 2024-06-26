from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots.robot import Robot
import carb
from omni.isaac.core.articulations import ArticulationView  # Notwendiger Import
from omni.isaac.core.prims import RigidPrimView, XFormPrim

from typing import Optional
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
        self._end_effectors = RigidPrimView(prim_paths_expr="/World/envs/.*/ur10/ee_link", name="end_effector_view", reset_xform_properties=False)

    
        
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
 
