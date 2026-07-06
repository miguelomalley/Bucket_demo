import numpy as np
from Mesh import *

def dent_mesh(mesh: Mesh, face_indices: list[int], push_distance: float = 0.2) -> Mesh:
    """
    Creates and returns a new Mesh with a dent.
    """
    # copies to work with safely
    temp_vertices = list(mesh.vertices)
    temp_faces = list(mesh.faces)
    
    vertices_np = np.array(temp_vertices)

    # we are denting these
    deleted_faces = [temp_faces[idx] for idx in face_indices]
    
    # center and average normal of the target faces
    normals = []
    all_v_indices = set()
    for f in deleted_faces:
        all_v_indices.update(f)
        if len(f) >= 3:
            p0, p1, p2 = vertices_np[f[0]], vertices_np[f[1]], vertices_np[f[2]]
            n = np.cross(p1 - p0, p2 - p0)
            n_len = np.linalg.norm(n)
            if n_len > 1e-6:
                normals.append(n / n_len)
                
    avg_normal = np.mean(normals, axis=0) if normals else np.array([0.0, 0.0, 1.0])
    avg_normal /= np.linalg.norm(avg_normal)
    center = np.mean([vertices_np[v] for v in all_v_indices], axis=0)
    
    # add a new vertex pushed inside
    new_vertex = center - (avg_normal * push_distance)
    temp_vertices.append(new_vertex.tolist())
    new_v_idx = len(temp_vertices) - 1
    
    # find boundary edges to keep mesh tight
    edge_counts = {}
    directed_edges = []
    for f in deleted_faces:
        for i in range(len(f)):
            u, v = f[i], f[(i + 1) % len(f)]
            undirected = (min(u, v), max(u, v))
            edge_counts[undirected] = edge_counts.get(undirected, 0) + 1
            directed_edges.append((u, v))
            
    boundary_edges = [e for e in directed_edges if edge_counts[(min(e[0], e[1]), max(e[0], e[1]))] == 1]
    
    # remove faces to dent it, add new dent faces
    for idx in sorted(face_indices, reverse=True):
        temp_faces.pop(idx)

    for u, v in boundary_edges:
        temp_faces.append([u, v, new_v_idx])
        
    # retrieve new edges
    temp_edges = get_edges(temp_faces)
    
    # inherit to new Mesh
    container = Mesh()
    container.vertices = temp_vertices
    container.faces = temp_faces
    container.edges = temp_edges
    
    # out: new mesh
    new_mesh = Mesh()
    new_mesh.inherit(container)
    return new_mesh


def create_hole(mesh: Mesh, face_idx1: int, face_idx2: int) -> Mesh:
    """
    Creates and returns a new Mesh with a through-hole.
    """
    # mostly the same as dent
    temp_vertices = list(mesh.vertices)
    temp_faces = list(mesh.faces)
    
    # two faces are hit
    vertices_np = np.array(temp_vertices)
    f1 = temp_faces[face_idx1]
    f2 = temp_faces[face_idx2]
    
    # find points on the edges of the hole which are closest to pair
    # We need to make the hole realistic, so the sides of the hole need to be filled in to keep the mesh tight.
    # I think probably this is not the best way to do it but I went with pairing edges which were closest by midpoint and drawing faces between them.
    # In particular, this is an issue for triangle winding for renderers, but we don't care about that for persistence.
    def get_face_edges(face):
        return [(face[i], face[(i + 1) % len(face)]) for i in range(len(face))]
        
    edges1 = get_face_edges(f1)
    edges2 = get_face_edges(f2)
    
    def get_midpoint(edge):
        return (vertices_np[edge[0]] + vertices_np[edge[1]]) / 2.0
        
    midpoints1 = [get_midpoint(e) for e in edges1]
    midpoints2 = [get_midpoint(e) for e in edges2]
    
    matched_pairs = []
    used_f2_indices = set()
    
    for i, m1 in enumerate(midpoints1):
        best_j = -1
        min_dist = float('inf')
        for j, m2 in enumerate(midpoints2):
            if j in used_f2_indices:
                continue
            dist = np.linalg.norm(m1 - m2)
            if dist < min_dist:
                min_dist = dist
                best_j = j
        matched_pairs.append((edges1[i], edges2[best_j]))
        used_f2_indices.add(best_j)
        
    for idx in sorted([face_idx1, face_idx2], reverse=True):
        temp_faces.pop(idx)
        
    for e1, e2 in matched_pairs:
        a, b = e1
        v_c1, v_c2 = e2
        
        dist_straight = np.linalg.norm(vertices_np[a] - vertices_np[v_c1]) + np.linalg.norm(vertices_np[b] - vertices_np[v_c2])
        dist_flipped = np.linalg.norm(vertices_np[a] - vertices_np[v_c2]) + np.linalg.norm(vertices_np[b] - vertices_np[v_c1])
        
        if dist_straight <= dist_flipped:
            c, d = v_c1, v_c2
        else:
            c, d = v_c2, v_c1
            
        temp_faces.append([a, b, c])
        temp_faces.append([c, d, b])
        
    temp_edges = get_edges(temp_faces)
    
    # Bundle into container and apply inherit
    container = Mesh()
    container.vertices = temp_vertices
    container.faces = temp_faces
    container.edges = temp_edges
    
    new_mesh = Mesh()
    new_mesh.inherit(container)
    return new_mesh

def ray_mesh_intersect(ray_origin, ray_direction, mesh):
    """
    Moller-Trumbore implementation for this problem
    """
    O = np.array(ray_origin)
    D = np.array(ray_direction)
    D = D / np.linalg.norm(D)  # ensure direction is normalized
    
    vertices = np.array(mesh.vertices)
    intersections = []
    
    for idx, face in enumerate(mesh.faces):
        if len(face) != 3:
            continue
            
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(D, edge2)
        a = np.dot(edge1, h)
        
        if -1e-6 < a < 1e-6:
            continue  # ray is parallel to the triangle
            
        f = 1.0 / a
        s = O - v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            continue
            
        q = np.cross(s, edge1)
        v = f * np.dot(D, q)
        
        if v < 0.0 or u + v > 1.0:
            continue
            
        t = f * np.dot(edge2, q)
        
        if t > 1e-6:  # intersection is in the forward ray direction
            intersections.append((t, idx))
            
    # Sort intersections by distance t (closest first)
    intersections.sort(key=lambda x: x[0])
    return intersections


def ray_dent_mesh(mesh: Mesh, ray_origin, ray_direction, depth_fraction: float = 0.5) -> Mesh:
    """
    Finds the first face hit by a ray, calculates the distance to the next face 
    along that same ray, and dents the mesh inward by a fraction of that distance.
    """
    intersections = ray_mesh_intersect(ray_origin, ray_direction, mesh)
    
    if len(intersections) < 2:
        #print("Ray missed the mesh! No dent applied.")
        return None
        
    # ray strike travel distance between faces
    t1, hit_face_idx = intersections[0]
    t2, _ = intersections[1]
    
    material_thickness = t2 - t1
    
    # depth frac determines dentyness
    calculated_push = material_thickness * depth_fraction
    #print(f"Ray hit entry face {hit_face_idx} (t={t1:.4f}) and exit face {next_face_idx} (t={t2:.4f}).")
    #print(f"Local thickness: {material_thickness:.4f} -> Dynamic dent depth ({depth_fraction*100}%): {calculated_push:.4f}")
    
    return dent_mesh(mesh, face_indices=[hit_face_idx], push_distance=calculated_push)

def ray_hole_mesh(mesh: Mesh, ray_origin, ray_direction) -> Mesh:
    """
    Finds the first two faces hit by a ray passing through the object 
    and punches a hole between them.
    """
    intersections = ray_mesh_intersect(ray_origin, ray_direction, mesh)
    
    if len(intersections) < 2:
        #print("Ray didn't pass through two surfaces! Cannot punch a through-hole.")
        return None
        
    # The first two faces the ray hit
    _, entry_face_idx = intersections[0]
    _, exit_face_idx = intersections[1]
    
    #print(f"Hole Ray tunneling from face {entry_face_idx} directly out of face {exit_face_idx}")
    
    return create_hole(mesh, entry_face_idx, exit_face_idx)


def generate_random_ray(mesh, target_jitter: float = 0.1, ray_distance_factor: float = 2.0):
    """
    Generates a random ray targeting the jitter shifted center of mass of the mesh from the outside.
    """
    vertices = np.array(mesh.vertices)
    if len(vertices) == 0:
        raise ValueError("Cannot generate ray for an empty mesh.")
        
    # center of Mass
    com = np.mean(vertices, axis=0)
    
    # add small random jitter to the target
    random_dir = np.random.normal(0, 1, 3)
    random_dir /= np.linalg.norm(random_dir)
    jitter = random_dir * np.random.uniform(0, target_jitter)
    target_point = com + jitter
    
    # determine outside distance using the mesh's bounding box bounding sphere radius
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    bbox_diagonal = np.linalg.norm(max_bounds - min_bounds)
    outside_distance = bbox_diagonal * ray_distance_factor
    
    # pick a random starting direction on a sphere for the ray origin
    ray_dir_sphere = np.random.normal(0, 1, 3)
    ray_dir_sphere /= np.linalg.norm(ray_dir_sphere)
    
    # place the origin far outside
    ray_origin = target_point + (ray_dir_sphere * outside_distance)
    ray_direction = target_point - ray_origin
    ray_direction /= np.linalg.norm(ray_direction)
    
    return ray_origin.tolist(), ray_direction.tolist()



def mesh_to_voxel_grid(mesh, resolution=32):
    """
    3D mesh => 3D occupancy grid.
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    if len(vertices) == 0:
        return np.zeros((1, resolution, resolution, resolution), dtype=np.float32)

    # center and normalize
    center = vertices.mean(axis=0)
    vertices -= center
    max_dist = np.max(np.linalg.norm(vertices, axis=1))
    if max_dist > 0:
        vertices /= max_dist

    # extract shell elements
    points = [vertices]
    if len(faces) > 0:
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0
        edge_mid1 = (v0 + v1) / 2.0
        edge_mid2 = (v1 + v2) / 2.0
        edge_mid3 = (v2 + v0) / 2.0
        points.extend([face_centers, edge_mid1, edge_mid2, edge_mid3])
        
    all_structural_points = np.concatenate(points, axis=0)
    vox_coords = ((all_structural_points + 1.0) * 0.5 * (resolution - 1)).astype(np.int32)
    vox_coords = np.clip(vox_coords, 0, resolution - 1)

    # make voxel array
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    grid[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]] = 1.0

    # needs a channel to do conv3D
    return np.expand_dims(grid, axis=0)