import os

try:
    import bpy
except ModuleNotFoundError:
    bpy = None


def update_obj_mtl(obj_path, new_mtl_file="test.mtl"):
    """
    修改指定文件夹下所有 .obj 文件，使其引用的 .mtl 文件改为指定的文件名。

    :param folder_path: 文件夹路径
    :param new_mtl_file: 新的 .mtl 文件名（默认 'test.mtl'）
    """

    try:
        # 打开并读取 .obj 文件内容
        with open(obj_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # 修改引用的 .mtl 文件
        with open(obj_path, "w", encoding="utf-8") as file:
            for line in lines:
                # 如果是 mtllib 行，则替换为新的 .mtl 文件
                if line.lower().startswith("mtllib "):
                    file.write(f"mtllib {new_mtl_file}\n")
                else:
                    file.write(line)

        print(f"{obj_path} 更新成功")

    except Exception as e:
        print(f"处理 {obj_path} 时发生错误: {e}")


def reduce_mesh(fname, decimate_type='DECIMATE', target_vertices=5000, smooth_type=1, output_path=None):
    # Precondition: this function only works when run inside the Blender binary
    # (see scripts/blender_reduce_mesh.py). Fail loudly instead of silently
    # no-op'ing if bpy didn't load.
    assert bpy is not None, "bpy not available - reduce_mesh must be run via the Blender binary (see blender_reduce_mesh.py)"

    ### Load ###
    reset_scene()
    name = fname.split('/')[-1]
    input_obj_path = fname #tmppath or fname
    if fname.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=input_obj_path)
    elif fname.endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=input_obj_path)

    ### Decimate ###
    for obj in bpy.context.scene.objects:
            # ratio = target_vertices / current_poly_count
            if obj.type =='MESH':
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj  # 设置为活动对象

                bpy.ops.object.mode_set(mode='EDIT')  # 进入编辑模式
                # 合并多余顶点（按距离合并）
                bpy.ops.mesh.remove_doubles(threshold=0.0001)  # 可修改 threshold 来调整合并距离
                bpy.ops.object.mode_set(mode='OBJECT')

                current_poly_count = len(obj.data.vertices)  # 获取当前顶点数
                print(f"[reduce_mesh] {name}/{obj.name}: after remove_doubles -> "
                      f"verts={current_poly_count} faces={len(obj.data.polygons)} "
                      f"(target_vertices={target_vertices})")

                n_iter = 0
                while current_poly_count > target_vertices:

                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(state=True)
                    modifier = obj.modifiers.new(name=decimate_type, type='DECIMATE')
                    # modifier.ratio = decimate_ratio
                    modifier.ratio = 0.7
                    bpy.context.view_layer.update()
                    bpy.ops.object.mode_set(mode='OBJECT')
                    bpy.ops.object.modifier_apply(modifier=decimate_type)
                    new_poly_count = len(obj.data.vertices)  # 获取当前顶点数
                    n_iter += 1
                    print(f"[reduce_mesh] {name}/{obj.name}: decimate iter {n_iter} -> "
                          f"verts={new_poly_count} faces={len(obj.data.polygons)}")

                    # Some topologies (e.g. thin disconnected hair strands) hit a
                    # structural floor where DECIMATE can't reduce further even
                    # though we're still above target_vertices - without this,
                    # the loop spins forever reapplying a no-op modifier.
                    if new_poly_count >= current_poly_count:
                        print(f"[reduce_mesh] {name}/{obj.name}: decimate stalled "
                              f"(no progress past {new_poly_count} verts, target was "
                              f"{target_vertices}) - stopping early")
                        current_poly_count = new_poly_count
                        break
                    current_poly_count = new_poly_count

                print(f"[reduce_mesh] {name}/{obj.name}: post-decimate final -> "
                      f"verts={current_poly_count} faces={len(obj.data.polygons)}")

    # Debug: dump the decimated-but-unsmoothed mesh so it's possible to tell whether
    # fragmentation comes from the DECIMATE modifier itself or from the smoothing step
    # below.
    if output_path is not None:
        root, ext = os.path.splitext(output_path)
        debug_path = f"{root}_postdecimate_debug{ext}"
        bpy.ops.wm.obj_export(filepath=debug_path, export_selected_objects=False)
        print(f"[reduce_mesh] {name}: wrote post-decimate debug snapshot -> {debug_path}")

    # Smooth #
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj

            if smooth_type==1:
                bpy.ops.object.shade_smooth()
            elif smooth_type==2:
                bpy.ops.object.shade_smooth()
                obj.data.use_auto_smooth = True  # Start Auto Smooth
                obj.data.auto_smooth_angle = 30.0 * 3.14159 / 180.0
            else:
                bpy.ops.object.shade_flat()

    # Export obj #
    bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=False)
    print(f"[reduce_mesh] {name}: exported decimated+smoothed mesh -> {output_path}")



def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    if bpy is None:
        return
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def remesh_and_replace(fname, voxel_size=0.0005,output_path = None):
    ### Load ###
    reset_scene()
    name = fname.split('/')[-1]
    input_obj_path = fname #tmppath or fname
    bpy.ops.wm.obj_import(filepath=input_obj_path)
    model_name = os.path.splitext(os.path.basename(input_obj_path))[0]
    for obj in bpy.context.scene.objects:
        if obj.type =='MESH':
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj  # 设置为活动对象

            bpy.ops.object.mode_set(mode='EDIT')  # 进入编辑模式
            # 合并多余顶点（按距离合并）
            bpy.ops.mesh.remove_doubles(threshold=0.0001)  # 可修改 threshold 来调整合并距离

            print("before remesh vertices:",len(obj.data.vertices))


            bpy.ops.object.mode_set(mode='OBJECT')  # 进入编辑模式


            ### Remesh ###
            bpy.ops.object.modifier_add(type='REMESH')
            remesh_modifier = obj.modifiers["Remesh"]
            remesh_modifier.mode = 'VOXEL'
            remesh_modifier.voxel_size = voxel_size
            bpy.ops.object.modifier_apply(modifier="Remesh")


    if output_path is None:

        bpy.ops.wm.obj_export(filepath=fname.replace('.obj','_new.obj'), export_selected_objects=False)
    else:
        bpy.ops.wm.obj_export(filepath=(output_path), export_selected_objects=False)
