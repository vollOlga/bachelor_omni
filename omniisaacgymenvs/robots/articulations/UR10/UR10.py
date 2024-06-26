from typing import Optional
import torch
from omni.isaac.core.robots.robot import Robot
#from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb

class UR10RobotArm(Robot):
    """
    Represents a UR10 robotic arm in a simulation environment. This class is used to instantiate a UR10 robot
    with specific configurations and add it to the simulation stage.

    Attributes:
        _usd_path (str): The path to the USD file that defines the visual and physical properties of the robot.
        _name (str): The name of the robot instance in the simulation.
        _position (torch.tensor): The initial position of the robot in the simulation environment.
        _orientation (torch.tensor): The initial orientation of the robot in the simulation environment.

    Args:
        prim_path (str): The path in the simulation scene graph where the robot will be added.
        name (Optional[str]): The name assigned to the robot instance. Defaults to "UR10".
        usd_path (Optional[str]): The path to a USD file for the robot. If None, a default path is set.
        translation (Optional[torch.tensor]): The initial position of the robot. If None, defaults to [0.0, 0.0, 0.0].
        orientation (Optional[torch.tensor]): The initial orientation of the robot. If None, defaults to [1.0, 0.0, 0.0, 0.0].
    """
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "UR10",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        attach_gripper: Optional[bool] = True,
        #gripper_usd_path: Optional[str] = None,
    ) -> None:
        """
        Initializes a new instance of the UR10 robot with the specified parameters and adds it to the simulation stage.

        The robot is positioned and oriented according to the provided translation and orientation tensors.
        If no USD path is provided, the default asset path is set. This is based on the assumption that there is a
        pre-configured USD path set up in the simulation environment.

        Raises:
            RuntimeError: If the assets root path cannot be determined when no specific USD path is provided.
        """

        self._usd_path = usd_path
        self.assets_root_path = get_assets_root_path()
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            #self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_long_suction.usd"
            #self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_short_suction.usd"
            #self._usd_path = (self.assets_root_path + "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/ur10_bin_stacking_short_suction.usd")
            #self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_with_hand_e.usd"
            #self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_with_2f_140_gripper.usd'
            #self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_instanceable.usd'
            self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_short_suction_gripper.usd'
            #self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_gripper_140_instanceable.usd'


        # Depends on your real robot setup
        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        add_reference_to_stage(self._usd_path, prim_path)

        if attach_gripper:
            self.attach_gripper()

        super().__init__(
            prim_path=prim_path,
            name=name,
            position =self._position,
            orientation=self._orientation,
            #articulation_controller=None,
            attach_gripper=attach_gripper,
            #gripper_usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/Props/short_gripper.usd"
        )

    
    def attach_gripper(self):
        gripper_usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/Props/short_gripper.usd"
        gripper_prim_path = f"{self.prim_path}/short_gripper"
        add_reference_to_stage(gripper_usd_path, gripper_prim_path)
        #self.attach_link(gripper_prim_path, "ee_link")

