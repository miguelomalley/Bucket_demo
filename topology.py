import numpy as np
import gudhi
import matplotlib.pyplot as plt
import torch

def get_filtration_directions() -> list[np.ndarray]:
    """
    Generates 8 direction vectors: 6 evenly spaced around the XY plane,
    1 pointing straight up, and 1 pointing straight down.
    """
    directions = []
    
    # 6 evenly spaced angles in the XY plane
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for angle in angles:
        directions.append(np.array([np.cos(angle), np.sin(angle), 0.0]))
        
    # 1 from above and 1 from below
    directions.append(np.array([0.0, 0.0, 1.0]))
    directions.append(np.array([0.0, 0.0, -1.0]))
    
    return directions


def compute_directional_persistence(mesh, direction: np.ndarray):
    """
    Computes H0 and H1 persistence barcodes for a mesh along a given direction vector
    using lower-star filtration.
    """
    vertices = np.array(mesh.vertices)
    
    # project for direction
    vertex_values = np.dot(vertices, direction)
    
    # get simplex tree
    st = gudhi.SimplexTree()
    
    # give it the vertices
    for v_idx in range(len(vertices)):
        st.insert([v_idx], filtration=vertex_values[v_idx])
        
    # edges
    for edge in mesh.edges:
        v_val = max(vertex_values[edge[0]], vertex_values[edge[1]])
        st.insert(list(edge), filtration=v_val)
        
    # faces
    for face in mesh.faces:
        v_val = max([vertex_values[v] for v in face])
        #print(v_val)
        st.insert(list(face), filtration=v_val)
        
    # persistence
    st.compute_persistence()
    
    # barcode intervals for H0 and H1
    intervals_h0 = st.persistence_intervals_in_dimension(0)
    intervals_h1 = st.persistence_intervals_in_dimension(1)
    
    return intervals_h0, intervals_h1, st


def plot_mesh_barcodes(mesh):
    """
    Loops through all 8 directions, calculates persistence, and plots the results.
    """
    directions = get_filtration_directions()
    labels = [
        "XY-0°", "XY-60°", "XY-120°", "XY-180°", "XY-240°", "XY-300°", 
        "Above (+Z)", "Below (-Z)"
    ]
    
    # 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (dir_vec, label) in enumerate(zip(directions, labels)):
        print(f"Computing persistence barcode for direction: {label} {dir_vec.tolist()}")
        
        intervals_h0, intervals_h1, st = compute_directional_persistence(mesh, dir_vec)
        #print(intervals_h0)
        #print(intervals_h1)
        
        # gudhi built in works here
        gudhi.plot_persistence_barcode(st.persistence(), axes=axes[i])
        axes[i].set_title(f"Direction: {label}")
        
    plt.tight_layout()
    plt.show()

def get_barcodes(mesh):
    directions = get_filtration_directions()
    barcode_dict = dict()
    for dir_vec in directions:
        intervals_h0, intervals_h1, st = compute_directional_persistence(mesh, dir_vec)
        barcode_dict[dir_vec] = [intervals_h0, intervals_h1, st]
    return barcode_dict


def _intervals_to_points(intervals, is_h0: bool, drop_first_inf: bool, inf_fill: float):
    """Convert a gudhi interval array into a list of [birth, death, is_H0, is_H1]."""
    points = []
    dropped_universal = False
    for birth, death in intervals:
        if drop_first_inf and (not dropped_universal) and np.isinf(death):
            # There is one H0 element representing the presence of at least one object. We don't need it for this.
            dropped_universal = True
            continue
        d = inf_fill if np.isinf(death) else float(death)
        points.append([float(birth), d, 1.0 if is_h0 else 0.0, 0.0 if is_h0 else 1.0])
    return points
 
 
def barcodes_to_feature(intervals_h0, intervals_h1,
                                   max_points=32, 
                                   inf_fill=10.0):
    """
    Combine H0 and H1 intervals for a single direction into a padded 
    (max_points, POINT_DIM) float32 array.
    """
    pts = []
    pts += _intervals_to_points(intervals_h0, is_h0=True, drop_first_inf=True, inf_fill=inf_fill)
    pts += _intervals_to_points(intervals_h1, is_h0=False, drop_first_inf=False, inf_fill=inf_fill)
 
    # sort by persistence (death - birth) descending -- keep the most prominent features
    pts.sort(key=lambda p: (p[1] - p[0]), reverse=True)
 
    arr = np.zeros((max_points, 4), dtype=np.float32)
    n = min(len(pts), max_points)
    if n > 0:
        arr[:n, :] = np.array(pts[:n], dtype=np.float32)
    return arr
 
 
def mesh_to_feature_tensor(mesh, max_points=32, inf_fill=10.0):
    """
    Runs all 8 directional filtrations on mesh and returns a tensor of shape
    (8, max_points, point_dim)
    """
    directions = get_filtration_directions()
    out = np.zeros((len(directions), max_points, 4), dtype=np.float32)
    for i, dir_vec in enumerate(directions):
        intervals_h0, intervals_h1, _ = compute_directional_persistence(mesh, dir_vec)
        out[i] = barcodes_to_feature(intervals_h0, intervals_h1, max_points, inf_fill)
    return torch.from_numpy(out)  # (8, max_points, point_dim)


if __name__ == '__main__':
    from Mesh import *
    from mesh_deformations import *

    file_path = "train/Screws and bolts with hexagonal head/00005000.obj"
    base_mesh = Mesh(file_path)

    print("--- Initial Mesh ---")
    print(f"Vertices: {len(base_mesh.vertices)}, Faces: {len(base_mesh.faces)}")
    plot_mesh_barcodes(base_mesh)