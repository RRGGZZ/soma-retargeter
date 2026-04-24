import os
import xml.etree.ElementTree as ET

import bpy
import mathutils


XML_PATH = "assets/robot/adam_pro/adam_pro.xml"
CLEAR_SCENE = True


def parse_vec(value, size, default=None):
    if not value:
        if default is not None:
            return default
        return [0.0] * size
    return [float(x) for x in value.split()]


def make_material(name, rgba, specular, shininess):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Specular"].default_value = max(0.0, min(1.0, specular))
    roughness = 1.0 - max(0.0, min(1.0, shininess))
    principled.inputs["Roughness"].default_value = roughness
    if rgba[3] < 1.0:
        mat.blend_method = "BLEND"
        mat.shadow_method = "HASHED"
    return mat


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)


def import_obj(filepath):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.obj(
            filepath=filepath,
            axis_forward="X",
            axis_up="Z",
            use_materials=False,
        )
    except TypeError:
        bpy.ops.import_scene.obj(
            filepath=filepath,
            axis_forward="X",
            axis_up="Z",
        )
    return [obj for obj in bpy.data.objects if obj not in before]


def ensure_collection(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj, collection):
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    collection.objects.link(obj)


def get_collection_bounds(collection):
    min_corner = mathutils.Vector((float("inf"),) * 3)
    max_corner = mathutils.Vector((float("-inf"),) * 3)
    has_mesh = False
    for obj in collection.objects:
        if obj.type != "MESH":
            continue
        has_mesh = True
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)
    if not has_mesh:
        return None, None
    return min_corner, max_corner


def add_camera_and_light(collection):
    bounds = get_collection_bounds(collection)
    if bounds[0] is None:
        return
    min_corner, max_corner = bounds
    center = (min_corner + max_corner) * 0.5
    size = max(max_corner - min_corner)
    distance = max(1.0, size * 2.5)

    cam_data = bpy.data.cameras.new("AdamPro_Camera")
    cam = bpy.data.objects.new("AdamPro_Camera", cam_data)
    cam.location = center + mathutils.Vector((distance, -distance, distance * 0.7))
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("AdamPro_Light", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("AdamPro_Light", light_data)
    light.location = center + mathutils.Vector((distance, distance, distance * 1.5))
    light.rotation_euler = mathutils.Euler((0.8, 0.0, 0.8))
    bpy.context.scene.collection.objects.link(light)


def main():
    if CLEAR_SCENE:
        clear_scene()

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else ""
    base_dir = os.path.dirname(XML_PATH)
    asset_dir = os.path.normpath(os.path.join(base_dir, meshdir))

    material_map = {}
    mesh_map = {}

    asset = root.find("asset")
    if asset is not None:
        for mat in asset.findall("material"):
            name = mat.get("name")
            rgba = parse_vec(mat.get("rgba"), 4, default=[0.8, 0.8, 0.8, 1.0])
            specular = float(mat.get("specular", "0.0"))
            shininess = float(mat.get("shininess", "0.5"))
            material_map[name] = make_material(name, rgba, specular, shininess)

        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            filename = mesh.get("file")
            if name and filename:
                mesh_map[name] = os.path.join(asset_dir, filename)

    collection = ensure_collection("AdamPro")

    def traverse_body(body_elem, parent_matrix):
        body_name = body_elem.get("name", "body")
        pos = parse_vec(body_elem.get("pos"), 3)
        quat = parse_vec(body_elem.get("quat"), 4, default=[1.0, 0.0, 0.0, 0.0])
        local_matrix = (
            mathutils.Matrix.Translation(pos)
            @ mathutils.Quaternion(quat).to_matrix().to_4x4()
        )
        world_matrix = parent_matrix @ local_matrix

        for geom in body_elem.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue
            geom_class = geom.get("class", "")
            if geom_class == "collision":
                continue
            if geom_class and geom_class != "visual":
                continue
            mesh_path = mesh_map.get(mesh_name)
            if not mesh_path or not os.path.exists(mesh_path):
                print(f"Missing mesh file for {mesh_name}: {mesh_path}")
                continue

            geom_pos = parse_vec(geom.get("pos"), 3)
            geom_quat = parse_vec(geom.get("quat"), 4, default=[1.0, 0.0, 0.0, 0.0])
            geom_matrix = (
                mathutils.Matrix.Translation(geom_pos)
                @ mathutils.Quaternion(geom_quat).to_matrix().to_4x4()
            )
            obj_matrix = world_matrix @ geom_matrix

            imported = import_obj(mesh_path)
            for obj in imported:
                if obj.type != "MESH":
                    continue
                obj.name = f"{body_name}_{mesh_name}"
                obj.matrix_world = obj_matrix
                obj.data = obj.data.copy()
                mat_name = geom.get("material")
                if mat_name and mat_name in material_map:
                    obj.data.materials.clear()
                    obj.data.materials.append(material_map[mat_name])
                    for poly in obj.data.polygons:
                        poly.material_index = 0
                link_to_collection(obj, collection)

        for child in body_elem.findall("body"):
            traverse_body(child, world_matrix)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            traverse_body(body, mathutils.Matrix.Identity(4))

    add_camera_and_light(collection)

    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0


if __name__ == "__main__":
    main()