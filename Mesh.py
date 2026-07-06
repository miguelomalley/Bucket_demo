import numpy as np
import os
import pyvista as pv
import trimesh

def load_mesh(filename: str) -> str:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist.")
        
    with open(filename, 'r') as file:
        return file.read()

def parse_mesh(meshtxt: str) -> tuple[list[list[float]], list[list[int]]]:
    vertices = []
    faces = []
    
    for line in meshtxt.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        parts = line.split()
        prefix = parts[0]
        
        if prefix == 'v':
            vertex = [float(x) for x in parts[1:4]]
            vertices.append(vertex)
            
        elif prefix == 'f':
            # OBJ faces can look like 'f 1/1/1 2/2/2' or 'f 1 2 3'
            face = []
            for vertex_info in parts[1:]:
                clean_vertex_info = vertex_info.replace('\x00', '').strip()

                if clean_vertex_info: 
                    vertex_index = int(clean_vertex_info.split('/')[0])
                else:
                    continue
                face.append(vertex_index - 1)
            faces.append(face)
            
    return vertices, faces

def repair_mesh(v, f):
    """
    Basically a container for trimesh repair
    """
    v = np.array(v, dtype=np.float64)
    
    cleaned_faces = []
    for face in f:
        if len(face) == 3:
            cleaned_faces.append(face)
        elif len(face) == 4:
            cleaned_faces.append([face[0], face[1], face[2]])
            cleaned_faces.append([face[0], face[2], face[3]])
        else:
            continue

    if len(cleaned_faces) == 0:
        # print out for truly empty meshes
        print("Warning: A mesh had no valid triangular faces! Returning blank arrays.")
        return v.tolist(), []

    f = np.array(cleaned_faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    _ = trimesh.repair.fill_holes(mesh)
    
    return mesh.vertices.tolist(), mesh.faces.tolist()

def get_edges(faces: list[list[int]]) -> list[tuple[int, int]]:
    """
    Goes through the faces and gets the edges
    """
    unique_edges = set()
    
    for face in faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            v1 = face[i]
            v2 = face[(i + 1) % num_vertices] 
            
            edge = (min(v1, v2), max(v1, v2))
            unique_edges.add(edge)
            
    return list(unique_edges)


def visualize_mesh(mesh):
    vertices = np.array(mesh.vertices)
    
    # PyVista expects a flat faces array where each face is prefixed by its number of padding vertices.
    # e.g., a triangle [0, 1, 2] becomes [3, 0, 1, 2]
    pv_faces = []
    for face in mesh.faces:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    pv_faces = np.array(pv_faces)

    poly_data = pv.PolyData(vertices, pv_faces)
    
    plotter = pv.Plotter()
    
    plotter.add_mesh(poly_data, color="lightblue", show_edges=True, edge_color="black", line_width=1)

    plotter.add_text("Part Mesh", font_size=10, position='upper_left')
    plotter.show()

class Mesh:
    """
    Bespoke mesh class. Probably we could have just gone with the trimesh.Trimesh class 
    but that would have required stapling on edges every time so this is easier.
    """
    def __init__(self, filename=None, repair = True):
        self.vertices = []
        self.faces = []
        self.edges = []
        
        if filename is not None:
            meshtxt = load_mesh(filename)
            self.vertices, self.faces = parse_mesh(meshtxt)

            # Some meshes are broken on intake (missing faces, etc.) so we use this relatively unobtrusive fix.
            if repair:
                self.vertices, self.faces = repair_mesh(self.vertices, self.faces)

            self.edges = get_edges(self.faces)


    def inherit(self, mesh):
        self.vertices = mesh.vertices
        self.edges = mesh.edges
        self.faces = mesh.faces

if __name__ == "__main__":
    file_path = "train/Screws and bolts with hexagonal head/00005000.obj"
    
    print(f"Loading {file_path}...")
    
    try:
        # Load and parse the mesh
        mesh = Mesh(file_path)
        
        # Simple counts check
        v_count = len(mesh.vertices)
        f_count = len(mesh.faces)
        e_count = len(mesh.edges)
        
        print("\n--- Mesh Summary ---")
        print(f"Vertices (V): {v_count}")
        print(f"Edges (E)   : {e_count}")
        print(f"Faces (F)   : {f_count}")
        print("--------------------")

    except FileNotFoundError:
        print(f"\nError: File not found '{file_path}'")