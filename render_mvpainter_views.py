"""render_mvpainter_views.py — render the 6 orthographic conditioning views
that MVPainter expects (color RGBA + geometric normal + depth EXR), from a
single GLB/OBJ mesh.

This is a *cleaned-up* port of upstream MVPainter's `scripts/blender_render_ortho.py`
(`third_party/MV-Painter/MVPainter/scripts/blender_render_ortho.py`). The original
mixes the 6-ortho path with random training-time views and several unused
branches; this file keeps only the inference-time render config.

What it produces
----------------
For an input mesh at <input_mesh>, with `--output_dir <out>`:

    <out>/<uid>/image/{000..005}.png   # 512x512 RGBA shaded render (Cycles, 1 sample)
    <out>/<uid>/normal/{000..005}.png  # uint16 RGB; encodes 0.5 + 0.5*N (world-space)
    <out>/<uid>/depth/{000..005}.exr   # OpenEXR depth, mapped to [distance-1, distance+1]
    <out>/<uid>/camera/{000..005}.npy  # dict with intrinsic/extrinsic/azimuth/elevation/...
    <out>/<uid>/blender.obj            # the loaded mesh, exported via Blender
    <out>/<uid>/<uid>.obj              # the merged-doubles mesh

The 6 views are MVPainter's canonical orthographic camera layout:

    view  azimuth  elevation
     000     0        0        front
     001    90        0        right
     002   180        0        back
     003   270        0        left
     004     0      -90        top
     005     0      +90        bottom

`--geo_rotation` is added to each azimuth — set 0 for TripoSG / our gt.glb
convention, -90 for TRELLIS / Hunyuan-2 / Hi3DGen meshes.

Scene normalization: longest bbox dim is scaled to 0.95 and centered at origin
*before* rendering, so the ortho camera at distance 3 with ortho_scale=1.0
sees the full mesh.

How to run
----------
Requires Blender 4.5+ (only blender's bundled python — no extra env needed):

    /path/to/blender --background -Y --python render_mvpainter_views.py -- \\
        --object_path /path/to/input_mesh.glb \\
        --output_dir  /path/to/out_dir \\
        --geo_rotation 0

If your Blender python doesn't have OpenCV, install it once:

    /path/to/blender/4.5/python/bin/python3.11 -m pip install opencv-python

The script also requires `scripts/overcast_soil_puresky_4k.exr` from the
upstream MVPainter repo for the HDRI lighting. Copy that file next to this
script (or pass `--hdri /path/to/hdri.exr`). Without it the script falls back
to a plain off-white world background — looks flatter but still works for
geometry/normal/depth conditioning.

Output layout matches what MVPainter's `infer_multiview.py` reads in
`render_temp/<uid>/`. If you point `infer_multiview.py` at this output, you
can skip its internal call to blender_render_ortho.py (useful for debugging
or for using a different renderer).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Generator, Optional, Tuple

import bpy
import cv2
import numpy as np
from mathutils import Matrix, Vector


# Blender doesn't slice argv at `--` for us.
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True,
                    help="Input mesh path (.glb / .obj / .fbx / .blend).")
parser.add_argument("--output_dir", type=str, default="render_temp",
                    help="Output root; one <uid>/ subdir is written per mesh.")
parser.add_argument("--geo_rotation", type=int, default=0,
                    help="Added to every view's azimuth. 0 for TripoSG / our "
                         "datasets/eval_assets_e2e meshes; -90 for TRELLIS / "
                         "Hunyuan-2 / Hi3DGen meshes.")
parser.add_argument("--resolution", type=int, default=512,
                    help="Output image resolution (square).")
parser.add_argument("--engine", type=str, default="CYCLES",
                    choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
parser.add_argument("--samples", type=int, default=1,
                    help="Cycles samples (upstream uses 1 since the geometry views "
                         "don't need shading detail).")
parser.add_argument("--hdri", type=str, default=None,
                    help="Path to an HDRI .exr for environment lighting (matches "
                         "upstream's overcast_soil_puresky_4k.exr). Optional.")
parser.add_argument("--distance", type=float, default=3.0,
                    help="Camera distance from origin (units). Default matches "
                         "MVPainter's depth_map_node which uses [distance-1, "
                         "distance+1] as the near/far for depth normalization.")
parser.add_argument("--ortho_scale", type=float, default=1.0,
                    help="Orthographic camera scale. Combined with the post-normalize "
                         "0.95 bbox fit, ortho_scale=1.0 leaves a small margin around "
                         "the mesh in each view.")
args = parser.parse_args(argv)

# ----------------------------------------------------------------------------
# Scene + camera setup (matches blender_render_ortho.py)
# ----------------------------------------------------------------------------
context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.data.type = "ORTHO"
cam.data.ortho_scale = args.ortho_scale
cam.data.lens = 35
cam.data.sensor_height = 32
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

if args.engine == "CYCLES":
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.use_denoising = True
render.film_transparent = True

# Enable depth + normal passes
context.view_layer.use_pass_normal = True
context.view_layer.use_pass_z = True
context.scene.use_nodes = True

# Wire up compositor: normal -> [0,1]-biased -> EXR; depth -> EXR
nodes = scene.node_tree.nodes
links = scene.node_tree.links
for n in list(nodes):
    nodes.remove(n)

render_layers = nodes.new("CompositorNodeRLayers")

# Normal: 0.5*N + 0.5 so it can be saved as positive values.
scale_normal = nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = "MULTIPLY"
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs["Normal"], scale_normal.inputs[1])
bias_normal = nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = "ADD"
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
normal_file_output.format.file_format = "OPEN_EXR"
normal_file_output.format.color_mode = "RGB"
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])


def prepare_depth_outputs():
    tree = scene.node_tree
    depth_out = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_map = tree.nodes.new(type="CompositorNodeMapRange")
    depth_out.base_path = ""
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "16"
    tree.links.new(render_layers.outputs["Depth"], depth_map.inputs[0])
    tree.links.new(depth_map.outputs[0], depth_out.inputs[0])
    return depth_out, depth_map


depth_file_output, depth_map_node = prepare_depth_outputs()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def reset_scene() -> None:
    """Clear everything except the camera + lights."""
    for obj in list(bpy.data.objects):
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)


def load_object(path: str) -> None:
    if path.endswith(".glb") or path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    elif path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif path.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=path)
    elif path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported input format: {path}")


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    for obj in scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    for obj in scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def scene_bbox(single_obj: Optional[bpy.types.Object] = None,
               ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no mesh in scene")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Scale so the longest bbox dim is 0.95 and center on origin. Matches
    blender_render_ortho.py exactly so downstream MVPainter code sees the
    same canonical pose."""
    if len(list(get_scene_root_objects())) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        scene.collection.objects.link(parent_empty)
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty
    box_scale = 0.95
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Camera"].parent = None
    return scale, offset


def set_camera_mvdream(azimuth: float, elevation: float, distance: float) -> bpy.types.Object:
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    point = (
        distance * math.cos(az_rad) * math.cos(el_rad),
        distance * math.sin(az_rad) * math.cos(el_rad),
        distance * math.sin(el_rad),
    )
    camera = bpy.data.objects["Camera"]
    camera.location = point
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def get_K_ortho(camd, ortho_scale: float) -> Matrix:
    rx, ry = render.resolution_x, render.resolution_y
    s = render.resolution_percentage / 100
    fx = rx / ortho_scale
    fy = ry / ortho_scale
    cx = rx / 2
    cy = ry / 2
    return Matrix(((fx, 0, cx), (0, fy, cy), (0, 0, 1)))


def get_RT(camera) -> np.ndarray:
    bpy.context.view_layer.update()
    location, rotation = camera.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t
    return np.concatenate([R_world2cv, t_world2cv[:, None]], 1)


def add_hdri_lighting(hdri_path: str) -> None:
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    env_node = nodes.get("Environment Texture") or nodes.new(type="ShaderNodeTexEnvironment")
    env_node.image = bpy.data.images.load(hdri_path)
    world.node_tree.links.new(env_node.outputs["Color"], nodes["Background"].inputs["Color"])
    world.node_tree.links.new(env_node.outputs["Color"], nodes["World Output"].inputs["Surface"])


def render_view(view_id: int, uid: str, azimuth: float, elevation: float,
                distance: float) -> None:
    out_root = os.path.join(args.output_dir, uid)
    scene.render.filepath = os.path.join(out_root, "image", f"{view_id:03d}.png")

    depth_map_node.inputs[1].default_value = distance - 1
    depth_map_node.inputs[2].default_value = distance + 1
    depth_file_output.base_path = os.path.join(out_root, "depth")
    depth_file_output.file_slots[0].path = f"{view_id:03d}"
    normal_file_output.base_path = os.path.join(out_root, "normal")
    normal_file_output.file_slots[0].path = f"{view_id:03d}"

    bpy.ops.render.render(write_still=True)

    # Blender writes the file with a frame suffix (e.g. 0000010001.exr) —
    # rename to match upstream's expected `{view_id:03d}.exr` / `.png` layout.
    depth_with_frame = os.path.join(out_root, "depth", f"{view_id:03d}0001.exr")
    depth_target = os.path.join(out_root, "depth", f"{view_id:03d}.exr")
    if os.path.exists(depth_with_frame):
        os.rename(depth_with_frame, depth_target)

    normal_with_frame = os.path.join(out_root, "normal", f"{view_id:03d}0001.exr")
    normal_png = os.path.join(out_root, "normal", f"{view_id:03d}.png")
    if os.path.exists(normal_with_frame):
        # Upstream: convert EXR normal to uint16 PNG (preserves more precision than 8-bit).
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        nimg = cv2.imread(normal_with_frame, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(normal_png, (nimg * 65535).astype(np.uint16))
        os.remove(normal_with_frame)

    # Camera params alongside.
    K = get_K_ortho(cam.data, args.ortho_scale)
    RT = get_RT(cam)
    paras = {
        "intrinsic": np.array(K, np.float32),
        "extrinsic": np.array(RT, np.float32),
        "fov": cam.data.angle,
        "azimuth": azimuth,
        "elevation": elevation,
        "distance": distance,
        "focal": cam.data.lens,
        "sensor_width": cam.data.sensor_width,
        "near": distance - 1,
        "far": distance + 1,
        "camera": "ortho",
    }
    cam_dir = os.path.join(out_root, "camera")
    os.makedirs(cam_dir, exist_ok=True)
    np.save(os.path.join(cam_dir, f"{view_id:03d}.npy"), paras)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    uid = os.path.basename(args.object_path).rsplit(".", 1)[0]
    out_root = os.path.join(args.output_dir, uid)
    os.makedirs(out_root, exist_ok=True)
    for sub in ("image", "normal", "depth", "camera"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    reset_scene()
    load_object(args.object_path)

    # Strip any imported lights so HDRI is the only light source.
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    if args.hdri and os.path.exists(args.hdri):
        add_hdri_lighting(args.hdri)
    else:
        if args.hdri:
            print(f"[warn] HDRI not found at {args.hdri}; using default world")

    normalize_scene()

    # Export the normalized OBJ next to the renders (upstream keeps this for the
    # downstream bake stage which reuses the mesh in normalized pose).
    bpy.ops.object.select_all(action="DESELECT")
    for obj in get_scene_meshes():
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        break
    bpy.ops.wm.obj_export(filepath=os.path.join(out_root, "blender.obj"))

    # Empty as the camera's TRACK_TO target.
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    azimuth_list = [0, 90, 180, 270, 0, 0]
    elevation_list = [0, 0, 0, 0, -90, 90]

    cam.data.type = "ORTHO"
    cam.data.ortho_scale = args.ortho_scale

    for i, (az, el) in enumerate(zip(azimuth_list, elevation_list)):
        az_final = az + args.geo_rotation
        bpy.context.view_layer.update()
        set_camera_mvdream(az_final, el, args.distance)
        render_view(i, uid, az_final, el, args.distance)
        print(f"  rendered view {i}: az={az_final} el={el}")

    print(f"\n[done] {uid} → {out_root}")


if __name__ == "__main__":
    main()
