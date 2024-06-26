from typing import Optional
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.prims import get_prim_at_path

import carb

class UR10(Robot):
    def __init__(
        self,
        prim_path: str, #  prim_path=self.default_zero_env_path + "/ur10"
        name: Optional[str] = "UR10",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,

        ####################
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
        gripper_usd: Optional[str] = "default",
        ####################

    ) -> None:

        self._usd_path = usd_path
        self._name = name
        
        ####################
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        ####################

        if self._usd_path is None: # Default no usd path specified --> imports ur10_short_suction_instanceable.usd
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_long_suction.usd" #omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Robots/UR10/ur10_instanceable.usd
        
        ####################
        if self._end_effector_prim_name is None: #default --> Attaches the gripper
            self._end_effector_prim_path = prim_path + "/ee_link"
        else:
            self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        ####################

        # Depends on your real robot setup
        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )
        
        ####################
        self._gripper_usd = gripper_usd #default
        
        print('gripper_usd: ' + str(gripper_usd))
        print('attach_gripper: '+ str(attach_gripper) )
        if attach_gripper:
            if gripper_usd == "default":
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/long_gripper.usd"
                add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
            elif gripper_usd is None:
                print('Not adding a gripper usd, the gripper already exists in the ur10 asset')
                carb.log_warn("Not adding a gripper usd, the gripper already exists in the ur10 asset")
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
            else:
                print('NotImplementedError')
                raise NotImplementedError
        self._attach_gripper = attach_gripper
        return
        ####################

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> SurfaceGripper:
        """[summary]

        Returns:
            SurfaceGripper: [description]
        """
        self._gripper = SurfaceGripper()
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self.disable_gravity()
        self._end_effector.initialize(physics_sim_view)
        print("Attached gripper is initialized")
        return

    def post_reset(self) -> None:
        """[summary]
        """
        Robot.post_reset(self)
        self._end_effector.post_reset()
        self._gripper.post_reset()
        return

    def close_gripper(self):
        if self._gripper:
            self._gripper.close()
    
    def open_gripper(self):
        if self._gripper:
            self._gripper.open()
