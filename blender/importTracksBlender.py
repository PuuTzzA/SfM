import bpy
import os
import mathutils

# ================= CONFIGURATION =================
# RELATIVE_PATH = "../../Data/test0.txt" 
RELATIVE_PATH = "../../Data/test0_8point.txt" 
# RELATIVE_PATH = "../../Data/test0_bundle.txt" 
# =================================================

def get_absolute_path(rel_path):
    """Safely resolves path relative to the current blend file."""
    if not bpy.data.filepath:
        return os.path.abspath(rel_path)
    
    if not rel_path.startswith("//"):
        rel_path = "//" + rel_path
        
    return bpy.path.abspath(rel_path)

def read_sfm_file(filepath):
    cameras = []
    points = []
    image_dir_path = None
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return [], [], None
    
    with open(filepath, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
    if not lines:
        return [], [], None

    # 1. Parse Header
    try:
        header = lines[0].split()
        num_cams = int(header[0])
        num_points = int(header[1])
    except (ValueError, IndexError):
        print("Error: Invalid header format.")
        return [], [], None
    
    current_line = 1
    
    # 2. Check for optional Image Path
    # We look at the next line. If it doesn't look like 16 floats, it's a path.
    if current_line < len(lines):
        next_line_tokens = lines[current_line].split()
        if len(next_line_tokens) != 16:
            # It's not a matrix, so it must be the path
            image_dir_path = lines[current_line]
            current_line += 1
    
    # 3. Parse Camera Matrices
    for _ in range(num_cams):
        if current_line >= len(lines): break
        vals = [float(x) for x in lines[current_line].split()]
        mat = (
            (vals[0], vals[1], vals[2], vals[3]),
            (vals[4], vals[5], vals[6], vals[7]),
            (vals[8], vals[9], vals[10], vals[11]),
            (vals[12], vals[13], vals[14], vals[15])
        )
        cameras.append(mat)
        current_line += 1
        
    # 4. Parse Points
    for _ in range(num_points):
        if current_line >= len(lines): break
        vals = [float(x) for x in lines[current_line].split()]
        points.append((vals[0], vals[1], vals[2]))
        current_line += 1
        
    return cameras, points, image_dir_path

def setup_background_images(cam_obj, txt_file_path, relative_img_path):
    """
    Resolves the image path, looks for the first image in that folder,
    and sets up the camera background sequence.
    """
    if not relative_img_path or not cam_obj:
        return

    # 1. Resolve Path: Combine txt file directory with the relative image path
    base_dir = os.path.dirname(txt_file_path)
    abs_img_dir = os.path.normpath(os.path.join(base_dir, relative_img_path))

    if not os.path.exists(abs_img_dir):
        print(f"Warning: Image directory not found at: {abs_img_dir}")
        return

    # 2. Find the first valid image file in the directory
    valid_exts = {'.png', '.jpg', '.jpeg', '.tga', '.exr', '.tif', '.tiff'}
    files = sorted([f for f in os.listdir(abs_img_dir) if os.path.splitext(f)[1].lower() in valid_exts])

    if not files:
        print(f"Warning: No image files found in {abs_img_dir}")
        return

    first_image_path = os.path.join(abs_img_dir, files[0])
    print(f"Loading image sequence starting from: {first_image_path}")

    # 3. Load the Image into Blender
    try:
        img_block = bpy.data.images.load(first_image_path)
        img_block.source = 'SEQUENCE'
        # Auto-detect sequence length is usually automatic in Blender if files are numbered
        # But we can try to force it to match camera frames if needed, though Blender handles this well.
    except RuntimeError as e:
        print(f"Error loading image: {e}")
        return

    # 4. Attach to Camera
    cam = cam_obj.data
    cam.show_background_images = True
    
    # Remove existing backgrounds if any
    cam.background_images.clear()
    
    bg = cam.background_images.new()
    bg.image = img_block
    bg.frame_method = 'FIT' # or 'CROP' depending on preference
    bg.alpha = 0.8  # Visibility strength

    num_frames = len(files)
    # These settings live on the 'image_user', not the image block itself
    if hasattr(bg, "image_user"):
        bg.image_user.frame_duration = num_frames
        bg.image_user.frame_start = 1
        bg.image_user.frame_offset = 0 
        bg.image_user.use_auto_refresh = True
        print(f"Set background sequence length to {num_frames} frames.")

def create_point_cloud_object(name, coords, collection):
    if not coords: return None
    mesh = bpy.data.meshes.new(name=f"{name}_TempMesh")
    mesh.from_pydata(coords, [], [])
    temp_obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(temp_obj)
    bpy.context.view_layer.objects.active = temp_obj
    temp_obj.select_set(True)
    try:
        bpy.ops.object.convert(target='POINTCLOUD')
        pc_obj = bpy.context.active_object
        bpy.data.meshes.remove(mesh)
        return pc_obj
    except RuntimeError:
        print("WARNING: Could not convert to native Point Cloud. Keeping as Mesh.")
        return temp_obj

def main():
    # 1. Resolve Path
    full_path = get_absolute_path(RELATIVE_PATH)
    print(f"Importing from: {full_path}")

    # 2. Read Data
    cam_matrices, point_coords, img_rel_path = read_sfm_file(full_path)
    
    if not cam_matrices and not point_coords:
        return

    # 3. Setup Collection
    filename = os.path.basename(full_path)
    col_name = f"SfM_{os.path.splitext(filename)[0]}"
    
    if col_name in bpy.data.collections:
        existing_col = bpy.data.collections[col_name]
        for obj in existing_col.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(existing_col)
        
    new_col = bpy.data.collections.new(col_name)
    bpy.context.scene.collection.children.link(new_col)

    cam_obj = None

    # 4. Create Camera
    if cam_matrices:
        cam_data = bpy.data.cameras.new(name="ImportedCam_Data")
        cam_obj = bpy.data.objects.new(name="ImportedCamera", object_data=cam_data)
        new_col.objects.link(cam_obj)
        
        # Set Framerate/Range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = len(cam_matrices)
        
        print(f"Creating Camera Animation ({len(cam_matrices)} frames)...")
        for i, mat_vals in enumerate(cam_matrices):
            frame_num = i + 1
            cam_obj.matrix_world = mathutils.Matrix(mat_vals)
            cam_obj.keyframe_insert(data_path="location", frame=frame_num)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

        bpy.ops.object.select_all(action='DESELECT')
        cam_obj.select_set(True)
        bpy.context.view_layer.objects.active = cam_obj
        print("Calculating Motion Path...")
        bpy.ops.object.paths_calculate(range='MANUAL')

        # --- NEW: SETUP BACKGROUND IMAGES ---
        if img_rel_path:
            setup_background_images(cam_obj, full_path, img_rel_path)
        # ------------------------------------

    # 5. Create Point Cloud
    if point_coords:
        print(f"Creating Point Cloud ({len(point_coords)} points)...")
        pc_obj = create_point_cloud_object("PointCloud", point_coords, new_col)
        
        if pc_obj:
            mod = pc_obj.modifiers.new(name="GeoNodes", type='NODES')
            node_group_name = "points"
            if node_group_name in bpy.data.node_groups:
                mod.node_group = bpy.data.node_groups[node_group_name]

    bpy.ops.object.select_all(action='DESELECT')
    print("Import Finished Successfully.")

if __name__ == "__main__":
    main()