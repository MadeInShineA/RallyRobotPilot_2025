import json
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

def parse_obj_vertices(obj_path):
    """Parse vertices and faces from OBJ file"""
    vertices = []
    faces = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        v = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(v)
                    except ValueError:
                        pass
            elif line.startswith("f "):
                parts = line.split()
                if len(parts) >= 4:  # At least a triangle
                    face = []
                    for p in parts[1:]:
                        # Parse vertex index (ignore texture and normal indices)
                        v_idx = int(p.split("/")[0]) - 1  # OBJ uses 1-based indexing
                        face.append(v_idx)
                    faces.append(face)
    return vertices, faces

def load_obstacle_vertices(track_name):
    """Load and transform obstacle vertices for track boundary plotting"""
    obstacle_data = []
    
    # Load track metadata for obstacles
    metadata_path = f"assets/{track_name}/track_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Warning: track_metadata.json not found at {metadata_path}")
        return obstacle_data
    
    with open(metadata_path) as f:
        track_metadata = json.load(f)
    
    if "obstacles" not in track_metadata:
        return obstacle_data
    
    origin_pos = np.array(track_metadata.get("origin_position", [0, 0, 0]))
    origin_rot_y = track_metadata.get("origin_rotation", [0, 0, 0])[1]
    origin_scale = track_metadata.get("origin_scale", [1, 1, 1])
    scale = origin_scale[1] if len(origin_scale) > 1 else 1.0
    
    for obstacle in track_metadata["obstacles"]:
        model_path = f"assets/{track_name}/{obstacle['model']}"
        if os.path.exists(model_path):
            obs_vertices, obs_faces = parse_obj_vertices(model_path)
            transformed_vertices = []
            for v in obs_vertices:
                scaled_v = np.array(v) * scale
                rot_rad = math.radians(origin_rot_y)
                cos_r = math.cos(rot_rad)
                sin_r = math.sin(rot_rad)
                rotated_v = np.array([
                    scaled_v[0] * cos_r - scaled_v[2] * sin_r,
                    scaled_v[1],
                    scaled_v[0] * sin_r + scaled_v[2] * cos_r,
                ])
                transformed_v = rotated_v + origin_pos
                transformed_vertices.append(transformed_v)
            
            # Store both transformed vertices and faces
            obstacle_data.append({
                "vertices": transformed_vertices, 
                "faces": obs_faces
            })
    
    return obstacle_data

def plot_collision_objects(track_name):
    """Plot all collision objects for the given track"""
    obstacle_vertices = load_obstacle_vertices(track_name)
    
    if not obstacle_vertices:
        print(f"No collision objects found for track {track_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, obstacle in enumerate(obstacle_vertices):
        vertices = obstacle["vertices"]
        faces = obstacle["faces"]
        
        # Plot each face as a separate polygon
        for face in faces:
            # Get the vertices for this face
            face_vertices = [vertices[idx] for idx in face]
            
            # Extract X and Z coordinates (flip Z vertically for typical top-down view)
            xs = [v[0] for v in face_vertices]
            zs = [-v[2] for v in face_vertices]
            
            # Close the polygon by repeating the first point at the end
            xs.append(xs[0])
            zs.append(zs[0])
            
            # Plot this face
            ax.plot(
                xs,
                zs,
                color="black",
                linewidth=2,
                alpha=1.0,
                label="Collision Objects" if i == 0 and face == faces[0] else "",
            )
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.set_title(f"Collision Objects for Track: {track_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')  # Ensure equal scaling
    
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_collision_objects.py <track_name>")
        sys.exit(1)
    
    track_name = sys.argv[1]
    plot_collision_objects(track_name)

if __name__ == "__main__":
    main()