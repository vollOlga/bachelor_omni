from omni.isaac.robot_assembler import RobotAssembler,AssembledRobot 
from omni.isaac.core.articulations import Articulation
import numpy as np

base_robot_path = "/World/ur10"
attach_robot_path = "/World/_fc2_instanceable"
base_robot_mount_frame = "/ee_link"
attach_robot_mount_frame = "/robotiq_85_adapter_link"
fixed_joint_offset = np.array([0.0,0.0,0.0])
fixed_joint_orient = np.array([1.0,0.0,0.0,0.0])
single_robot = False

robot_assembler = RobotAssembler()
assembled_robot = robot_assembler.assemble_articulations(
	base_robot_path,
	attach_robot_path,
	base_robot_mount_frame,
	attach_robot_mount_frame,
	fixed_joint_offset,
	fixed_joint_orient,
	mask_all_collisions = True,
	single_robot=single_robot
)

# The fixed joint in a assembled robot is editable after the fact:
# offset,orient = assembled_robot.get_fixed_joint_transform()
# assembled_robot.set_fixed_joint_transform(np.array([.3,0,0]),np.array([1,0,0,0]))

# And the assembled robot can be disassembled, after which point the AssembledRobot object will no longer function.
# assembled_robot.disassemble()

# Controlling the resulting assembled robot is different depending on the single_robot flag
if single_robot:
	# The robots will be considered to be part of a single Articulation at the base robot path
	controllable_single_robot = Articulation(base_robot_path)
else:
	# The robots are controlled independently from each other
	base_robot = Articulation(base_robot_path)
	attach_robot = Articulation(attach_robot_path)
	

# def add_parallel_gripper(self):
#         stage = get_current_stage()
#         ee_prim_path = self.prim_path + "/ee_link"
#         gripper_prim_path = ee_prim_path

#         gripper_prim = UsdGeom.Xform.Define(stage, gripper_prim_path)
#         xform_api = UsdGeom.XformCommonAPI(gripper_prim)
#         xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.1))
#         xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

#         block = UsdGeom.Cube.Define(stage, gripper_prim_path + "/visual")
#         block.GetSizeAttr().Set(0.05)

#         collision_prim = UsdPhysics.CollisionAPI.Apply(block.GetPrim())
#         collision_prim.GetCollisionEnabledAttr().Set(True)

#         if not gripper_prim.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
#             gripper_prim.GetPrim().ApplyAPI(UsdPhysics.RigidBodyAPI)

#         self.add_finger(gripper_prim_path, "left")
#         self.add_finger(gripper_prim_path, "right")

#     def add_finger(self, gripper_prim_path, side):
#         stage = get_current_stage()
#         finger_prim_path = gripper_prim_path + f"/{side}_finger"
        
#         finger_prim = UsdGeom.Xform.Define(stage, finger_prim_path)
#         xform_api = UsdGeom.XformCommonAPI(finger_prim)
        
#         if side == "left":
#             xform_api.SetTranslate(Gf.Vec3d(-0.025, 0.0, 0.0))
#         elif side == "right":
#             xform_api.SetTranslate(Gf.Vec3d(0.025, 0.0, 0.0))

#         xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

#         finger_block = UsdGeom.Cube.Define(stage, finger_prim_path + "/visual")
#         finger_block.GetSizeAttr().Set(0.01)

#         finger_collision_prim = UsdPhysics.CollisionAPI.Apply(finger_block.GetPrim())
#         finger_collision_prim.GetCollisionEnabledAttr().Set(True)

#         if not finger_prim.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
#             finger_prim.GetPrim().ApplyAPI(UsdPhysics.RigidBodyAPI)


