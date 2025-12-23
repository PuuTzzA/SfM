import bpy
import os
import mathutils

# ================= CONFIGURATION =================
#RELATIVE_PATH = "../../Data/test0.txt" 
RELATIVE_PATH = "../../Data/test0_8point.txt" 
# =================================================

def get_absolute_path(rel_path):
    """Safely resolves path relative to the current blend file."""
    if not bpy.data.filepath:
        # If file not saved, assume path is absolute or relative to CWD
        return os.path.abspath(rel_path)
    
    # '//' is Blender's internal prefix for relative paths
    if not rel_path.startswith("//"):
        # If user provided "../file", convert to "//../file"
        rel_path = "//" + rel_path
        
    return bpy.path.abspath(rel_path)

def read_sfm_file(filepath):
    cameras = []
    points = []
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return [], []
    
    with open(filepath, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments
            if line and not line.startswith('#'):
                lines.append(line)
        
    if not lines:
        print("File is empty or valid data not found.")
        return [], []

    # 1. Parse Header
    try:
        header = lines[0].split()
        num_cams = int(header[0])
        num_points = int(header[1])
    except (ValueError, IndexError):
        print("Error: Invalid header format.")
        return [], []
    
    current_line = 1
    
    # 2. Parse Camera Matrices
    for _ in range(num_cams):
        if current_line >= len(lines): break
        vals = [float(x) for x in lines[current_line].split()]
        # Create 4x4 Tuple
        mat = (
            (vals[0], vals[1], vals[2], vals[3]),
            (vals[4], vals[5], vals[6], vals[7]),
            (vals[8], vals[9], vals[10], vals[11]),
            (vals[12], vals[13], vals[14], vals[15])
        )
        cameras.append(mat)
        current_line += 1
        
    # 3. Parse Points
    for _ in range(num_points):
        if current_line >= len(lines): break
        vals = [float(x) for x in lines[current_line].split()]
        points.append((vals[0], vals[1], vals[2]))
        current_line += 1
        
    return cameras, points

def create_point_cloud_object(name, coords, collection):
    """
    Creates a native PointCloud object by creating a mesh first and converting it.
    This works around the missing 'resize' API for PointClouds in Python.
    """
    if not coords:
        return None

    # Step A: Create a temporary Mesh (vertices only)
    # This is extremely fast (C++ backend)
    mesh = bpy.data.meshes.new(name=f"{name}_TempMesh")
    mesh.from_pydata(coords, [], []) # Verts, Edges, Faces
    
    # Step B: Create Object and Link
    temp_obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(temp_obj)
    
    # Step C: Convert to Native Point Cloud
    # We must make the object active and selected to use the Operator
    bpy.context.view_layer.objects.active = temp_obj
    temp_obj.select_set(True)
    
    try:
        # This operator converts the active Mesh object into a PointCloud object
        # and replaces the data block automatically.
        bpy.ops.object.convert(target='POINTCLOUD')
        
        # The object wrapper remains, but its data is now a PointCloud
        pc_obj = bpy.context.active_object
        
        # Cleanup: The Convert operator might leave the old mesh data block orphaned
        bpy.data.meshes.remove(mesh)
        
        return pc_obj
        
    except RuntimeError:
        print("WARNING: Could not convert to native Point Cloud (Blender version too old?). Keeping as Mesh.")
        return temp_obj

def main():
    # 1. Resolve Path
    full_path = get_absolute_path(RELATIVE_PATH)
    print(f"Importing from: {full_path}")

    # 2. Read Data
    cam_matrices, point_coords = read_sfm_file(full_path)
    if not cam_matrices and not point_coords:
        return

    # 3. Setup Collection
    filename = os.path.basename(full_path)
    col_name = f"SfM_{os.path.splitext(filename)[0]}"
    
    # Remove existing collection if exists
    if col_name in bpy.data.collections:
        existing_col = bpy.data.collections[col_name]
        # Remove objects
        for obj in existing_col.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        # Remove collection
        bpy.data.collections.remove(existing_col)
        
    new_col = bpy.data.collections.new(col_name)
    bpy.context.scene.collection.children.link(new_col)

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

        # -----------------------------------------------------------
        # AUTOMATIC MOTION PATH CALCULATION
        # -----------------------------------------------------------
        # Deselect everything first
        bpy.ops.object.select_all(action='DESELECT')
        
        # Select Camera and make active
        cam_obj.select_set(True)
        bpy.context.view_layer.objects.active = cam_obj
        
        # Calculate Paths
        # range='SCENE' uses the scene start/end we just set
        print("Calculating Motion Path...")
        bpy.ops.object.paths_calculate(range='MANUAL')
        # -----------------------------------------------------------

    # 5. Create Point Cloud
    if point_coords:
        print(f"Creating Point Cloud ({len(point_coords)} points)...")
        pc_obj = create_point_cloud_object("PointCloud", point_coords, new_col)
        
        if pc_obj:
            # 6. Add Geometry Nodes
            mod = pc_obj.modifiers.new(name="GeoNodes", type='NODES')
            
            # Assign "points" node group if it exists
            node_group_name = "points"
            if node_group_name in bpy.data.node_groups:
                mod.node_group = bpy.data.node_groups[node_group_name]
                print(f"Assigned Geometry Node group: '{node_group_name}'")
            else:
                print(f"WARNING: Node group '{node_group_name}' not found.")
                print("         Please create a Geometry Node group named 'points' for rendering.")

    bpy.ops.object.select_all(action='DESELECT')
    print("Import Finished Successfully.")

if __name__ == "__main__":
    main()