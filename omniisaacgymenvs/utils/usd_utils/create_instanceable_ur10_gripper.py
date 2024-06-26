
import omni.usd
import omni.client

from pxr import UsdGeom, Sdf, UsdPhysics, UsdShade


def create_ur10_mesh(asset_usd_path, ur10_mesh_usd_path):
    # Create ur10_mesh.usd file
    omni.client.copy(asset_usd_path, ur10_mesh_usd_path)
    omni.usd.get_context().open_stage(ur10_mesh_usd_path)
    stage = omni.usd.get_context().get_stage()
    edits = Sdf.BatchNamespaceEdit()
    # Create parent Xforms
    reparent_tasks = [
        ['/pp/Cube', 'geoms_xform'],
    ] # [prim_path, parent_xform_name]
    for task in reparent_tasks:
        prim_path, parent_xform_name = task
        old_parent_path = '/'.join(prim_path.split('/')[:-1])
        new_parent_path = f'{old_parent_path}/{parent_xform_name}'
        UsdGeom.Xform.Define(stage, new_parent_path)
        edits.Add(Sdf.NamespaceEdit.Reparent(prim_path, new_parent_path, -1))
    stage.GetRootLayer().Apply(edits)
    # Save to file
    omni.usd.get_context().save_stage()


def create_ur10_instanceable(ur10_mesh_usd_path, ur10_instanceable_usd_path):
    omni.client.copy(ur10_mesh_usd_path, ur10_instanceable_usd_path)
    omni.usd.get_context().open_stage(ur10_instanceable_usd_path)
    stage = omni.usd.get_context().get_stage()
    # Set up references and instanceables
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Xform':
            continue
        # Add reference to visuals_xform, collisions_xform, geoms_xform, and make them instanceable
        path = str(prim.GetPath())
        if path.endswith('visuals_xform') or path.endswith('collisions_xform') or path.endswith('geoms_xform'):
            ref = prim.GetReferences()
            ref.ClearReferences()
            ref.AddReference('./mesh.usd', path)
            prim.SetInstanceable(True)
    # Save to file
    omni.usd.get_context().save_stage()

if __name__ == '__main__':
    ur10_usd_path = '/root/RLrepo/OmniIsaacGymEnvs-UR10Reacher/Isaac/2022.1/Isaac/Props/Blocks/pp.usda'
    ur10_mesh_usd_path = '/root/RLrepo/OmniIsaacGymEnvs-UR10Reacher/Isaac/2022.1/Isaac/Props/Blocks/pp_mesh.usda'
    ur10_instanceable_usd_path = '/root/RLrepo/OmniIsaacGymEnvs-UR10Reacher/Isaac/2022.1/Isaac/Props/Blocks/pp_instanceable.usd'
    
    create_ur10_mesh(ur10_usd_path, ur10_mesh_usd_path)
    create_ur10_instanceable(ur10_mesh_usd_path, ur10_instanceable_usd_path)
    


import omni.usd
import omni.client

from pxr import UsdGeom, Sdf

def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):
    """ Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
        Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

        Args:
            asset_usd_path (str): USD file path for asset
            source_prim_path (str): USD path of root prim
            save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()
    print(stage)
    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box", "Cube"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        print("Hier war ich")
        omni.usd.get_context().save_as_stage(save_as_path)


if __name__ == '__main__':
    create_parent_xforms("/home/willi/Documents/platform_ready_for_instan.usda","/World/platform/platform_Xform" , "/home/willi/Documents/instantiable_items/")