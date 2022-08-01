import numpy as np
import sys, os
import trimesh
# conda install -c anaconda networkx
import networkx as nx
from tqdm import tqdm

FileDirPath = os.path.dirname(__file__)
sys.path.append(os.path.join(FileDirPath, '..'))

import rasterization

jcd_path = os.path.join(".", "jumping_cubes_dictionary_final.npy")
jcd = np.load(jcd_path, allow_pickle=True).item()


VISUAL = True

# set VISUAL to false if you don't want to use the open3d library. 
# If you do this you won't be able to visualize the mesh.
if VISUAL:
    import open3d as o3d

    def make_line_set(verts, lines, colors=None):
        '''
        Returns an open3d line set given vertices, line indices, and optional color
        '''
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(verts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        if colors is not None:
            line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def make_point_cloud(points, colors=None):
        '''
        Returns an open3d point cloud given a list of points and optional colors
        '''
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        if colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        return point_cloud

    def make_mesh(verts, faces, color=None):
        '''
        Returns an open3d triangle mesh given vertices, mesh indices, and optional color
        '''
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        if color is not None:
            if len(color.shape) == 1:
                mesh.paint_uniform_color(color)
            else:
                mesh.vertex_colors = o3d.utility.Vector3dVector(color)
        return mesh



def voxelized_implicit(volume, spacing, bottom_left_loc=np.array([0.,0.,0.])):
    '''
    Takes lattice grid depths output from the Jumping Cubes Algorithm and returns
    voxel occupancies based on the parity of the # of surfaces outside of that voxel.
    '''
    _, x_dim, y_dim, z_dim = volume.shape

    
    x_vals = np.linspace(0.0, spacing*(x_dim-1), num=x_dim) + bottom_left_loc[0]
    y_vals = np.linspace(0.0, spacing*(y_dim-1), num=y_dim) + bottom_left_loc[1]
    z_vals = np.linspace(0.0, spacing*(z_dim-1), num=z_dim) + bottom_left_loc[2]
    coords = np.stack(np.meshgrid(x_vals, y_vals, z_vals, indexing='ij'), axis=-1)

    surfaces = (volume < spacing).astype(int)
    x_interiors = np.cumsum(surfaces[0], axis=0) % 2
    y_interiors = np.cumsum(surfaces[1], axis=1) % 2
    z_interiors = np.cumsum(surfaces[2], axis=2) % 2

    interiors = np.logical_and(x_interiors, y_interiors) 
    interiors = np.logical_and(interiors, z_interiors)
    # print(interiors.shape)
    # print(coords.shape)
    full_voxels = coords[interiors==1]
    empty_voxels = coords[interiors==0]
    return full_voxels, empty_voxels    

def jumping_cubes(volume, spacing, bottom_left_loc=np.array([0.,0.,0.]), cc_threshold=-1):
    '''
    Applies the jumping cubes algorithm and returns 
    volume - array of shape [3, resolution, resolution, resolution] where the first dimension is for view directions of +x, +y, +z
    spacing - the spatial distance between neighboring points on the cuberille grid 
    bottom_left_loc - the location of the most negative corner of the cube
    cc_threshold - A threshold for connected component size, all mesh components with size less than this threshold will be removed. Pass -1 for no threshold
    '''

    # Get the depth value at each of the 12 cube edges (for all cubes in the grid)
    depth_0 = volume[0,:-1,:-1,:-1]
    depth_1 = volume[1,1:,:-1,:-1]
    depth_2 = volume[0,:-1,1:,:-1]
    depth_3 = volume[1,:-1,:-1,:-1]
    depth_4 = volume[0,:-1,:-1,1:]
    depth_5 = volume[1,1:,:-1,1:]
    depth_6 = volume[0,:-1,1:,1:]
    depth_7 = volume[1,:-1,:-1,1:]
    depth_8 = volume[2,:-1,:-1,:-1]
    depth_9 = volume[2,1:,:-1,:-1]
    depth_10 = volume[2,1:,1:,:-1]
    depth_11 = volume[2,:-1,1:,:-1]


    _, x_dim, y_dim, z_dim = volume.shape

    # assign an index to each vertex in the grid
    indices = np.arange(x_dim*y_dim*z_dim).reshape((x_dim,y_dim,z_dim))
    
    # calculate the xyz positions for each vertex in the grid
    x_vals = np.linspace(0.0, spacing*(x_dim-1), num=x_dim) + bottom_left_loc[0]
    y_vals = np.linspace(0.0, spacing*(y_dim-1), num=y_dim) + bottom_left_loc[1]
    z_vals = np.linspace(0.0, spacing*(z_dim-1), num=z_dim) + bottom_left_loc[2]
    x_view_coords = np.stack(np.meshgrid(x_vals+0.5*spacing, y_vals, z_vals, indexing='ij'), axis=-1)
    y_view_coords = np.stack(np.meshgrid(x_vals, y_vals+0.5*spacing, z_vals, indexing='ij'), axis=-1)
    z_view_coords = np.stack(np.meshgrid(x_vals, y_vals, z_vals+0.5*spacing, indexing='ij'), axis=-1)
    all_coords = np.concatenate([x_view_coords.reshape((-1,3)), y_view_coords.reshape((-1,3)), z_view_coords.reshape((-1,3))])



    # get the appropriate vertex index for each of the 12 cube edges (for all cubes in the grid)
    n_points = indices.shape[0]*indices.shape[1]*indices.shape[2]

    # get vertices
    intersections = volume < spacing
    x_inds = intersections[0].flatten()
    y_inds = intersections[1].flatten()
    z_inds = intersections[2].flatten()

    intersections = np.concatenate([x_inds, y_inds, z_inds])
    intersection_coords = all_coords[intersections]


    
    inds_0 = indices[:-1,:-1,:-1]
    inds_1 = indices[1:,:-1,:-1] + n_points
    inds_2 = indices[:-1,1:,:-1]
    inds_3 = indices[:-1,:-1,:-1] + n_points
    inds_4 = indices[:-1,:-1,1:]
    inds_5 = indices[1:,:-1,1:] + n_points
    inds_6 = indices[:-1,1:,1:]
    inds_7 = indices[:-1,:-1,1:] + n_points
    inds_8 = indices[:-1,:-1,:-1] + 2*n_points
    inds_9 = indices[1:,:-1,:-1] + 2*n_points
    inds_10 = indices[1:,1:,:-1] + 2*n_points
    inds_11 = indices[:-1,1:,:-1] + 2*n_points

    # stack indices for each cube so that the 0-12 face index can be converted to a global index
    # use the global index to get the vertex coordinates (x_view, y_view, z_view)
    # concatenate the views together so each vertex has a global coordinate^^


    # stack all of the indices and depth values into a single array
    inds_list = [inds_0, inds_1, inds_2, inds_3, inds_4, inds_5, inds_6, inds_7, inds_8, inds_9, inds_10, inds_11]
    inds_list = [i.flatten() for i in inds_list]
    indices = np.hstack([i[:,None] for i in inds_list])

    depth_list = [depth_0, depth_1, depth_2, depth_3, depth_4, depth_5, depth_6, depth_7, depth_8, depth_9, depth_10, depth_11]
    depth_list = [d.flatten() for d in depth_list]
    depths = np.hstack([d[:,None] for d in depth_list])


    # calculate which one of the 4096 jumping cubes cases we have (based on which cube edges contain intersections)
    coefficients = np.array([2**(11-x) for x in range(12)])
    coefficients = np.tile(coefficients, (depths.shape[0], 1))
    cases = np.sum(coefficients * (depths > spacing), axis=-1)

    # base_cases = np.array([base_mapping[case] for case in cases]).reshape((x_dim-1, y_dim-1, z_dim-1))
    faces = [jcd[case] for case in cases] #use the jumping cubes dictionary to get the faces for each case

    # convert the faces from local (0-11) indices into global (entire grid) indices
    global_faces = []

    vert_counts = {}
    for i in range(len(faces)):
        for face in faces[i]:
            global_index_face = [indices[i][j] for j in face]
            global_faces.append(global_index_face)
            for ind in global_index_face:
                if ind in vert_counts:
                    vert_counts[ind] += 1
                else:
                    vert_counts[ind] = 1

    
    vert_set = set(vert_counts.keys())


    # build graph and find connected components
    # Remove connected components below a certain size if desired
    if cc_threshold > 0:
        edge_list = []
        for face in global_faces:
            edge_list.append((face[0], face[1]))
            edge_list.append((face[1], face[2]))
            edge_list.append((face[2], face[0]))
        G = nx.from_edgelist(edge_list)
        ccs = nx.connected_components(G)

        verts_to_remove = []
        for vs in ccs:
            if len(vs) < cc_threshold:
                verts_to_remove += vs
        verts_to_remove = set(verts_to_remove)
        
        temp_faces = []
        for face in global_faces:
            if face[0] not in verts_to_remove:
                temp_faces.append(face)
        global_faces = temp_faces
        vert_set -= verts_to_remove

   
    # re-index the vertices to account for the fact that not all grid vertices exist in the final mesh
    used_vert_indices = list(vert_set)
    new_indices = {used_vert_indices[i]: i for i in range(len(used_vert_indices))}
    if len(used_vert_indices) > 0:
        final_vertices = np.stack([all_coords[i] for i in used_vert_indices], axis=0)
    else:
        final_vertices = np.array([])

    # rebuild the faces using the new global indices
    final_faces = []
    for face in global_faces:
        final_faces.append([new_indices[i] for i in face])

    # apply laplacian smoothing
    mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
    # mesh = trimesh.smoothing.filter_humphrey(mesh, iterations=60, alpha=0.1)
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=15, volume_constraint=False)
    # trimesh.repair.fill_holes(mesh)
    export= trimesh.exchange.obj.export_obj(mesh)

    final_vertices = mesh.vertices
    final_faces = list(mesh.faces)
    final_faces = [list(x) for x in final_faces]
    return final_vertices, final_faces, intersection_coords, export

def show_mesh(vertices, faces, other_geoms=[]):
    geometries = other_geoms
    if vertices.shape[0] > 0:
        geometries += [make_mesh(vertices, faces)]
        o3d.visualization.draw_geometries(geometries)



# Load object and apply jumping cubes
def load_object(obj_name, data_path):
    '''
    Loads .obj file and returns the vertices, faces, and Trimesh object
    '''
    obj_file = os.path.join(data_path, f"{obj_name}.obj")

    obj_mesh = trimesh.load(obj_file)
    # obj_mesh.show()

    ## deepsdf normalization
    mesh_vertices = obj_mesh.vertices
    mesh_faces = obj_mesh.faces
    center = (mesh_vertices.max(axis=0) + mesh_vertices.min(axis=0))/2.0
    max_dist = np.linalg.norm(mesh_vertices - center, axis=1).max()
    max_dist = max_dist * 1.03
    mesh_vertices = (mesh_vertices - center) / max_dist
    obj_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    return mesh_vertices, mesh_faces, obj_mesh

def get_depths(vertices, faces, pts, look):
    '''
    Returns the depths from points to a mesh surface in a specific viewing direction
    vertices - mesh vertices
    faces    - mesh faces
    pts      - query points
    look     - a single viewing direction for all points
    '''
    all_depths = []
    for i in tqdm(range(pts.shape[0])):
        rot_verts = rasterization.rotate_mesh(vertices, pts[i], pts[i]+look)
        _, depth = rasterization.ray_occ_depth(faces, rot_verts, ray_start_depth=np.linalg.norm(look), near_face_threshold=0.08, v=None)
        all_depths.append(depth)
    all_depths = np.array(all_depths).flatten()
    return all_depths

def make_odf_queries(vertices, faces, resolution=20):
    '''
    Build a 3D lattice of points and query the ground truth ODF (from mesh) for each point in each of the 3 axis-aligned viewing directions
    vertices - vertices of the object mesh
    faces    - faces of the object mesh
    '''
    lin_pts = np.linspace(-1.,1.,num=resolution)
    xs,ys,zs = np.meshgrid(lin_pts,lin_pts,lin_pts,indexing='ij')
    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()
    query_pts = np.hstack([xs[:,None], ys[:,None], zs[:,None]])
    half_scale = 2.0/(resolution-1) * 0.5
    x_look = np.array([half_scale, 0., 0.])
    y_look = np.array([0., half_scale, 0.])
    z_look = np.array([0., 0., half_scale])
    x_depths = get_depths(vertices, faces, query_pts, x_look).reshape((resolution, resolution, resolution))
    y_depths = get_depths(vertices, faces, query_pts, y_look).reshape((resolution, resolution, resolution))
    z_depths = get_depths(vertices, faces, query_pts, z_look).reshape((resolution, resolution, resolution))
    volume = np.stack([x_depths, y_depths, z_depths], axis=0)
    return volume, query_pts

def show_grid(grid_points):
    '''
    Makes an Open3D point cloud of the grid points
    '''
    pts = make_point_cloud(grid_points)
    return [pts]

def main():
    data_path = os.path.join(".", "meshes")
    object = "bunny_watertight"

    # Change the resolution of the 3D lattice grid
    resolution = 15


    mesh_verts, mesh_faces, _ = load_object(object, data_path)
    volume, grid_points = make_odf_queries(mesh_verts, mesh_faces, resolution=resolution)

    scale = 2.0/(resolution-1)
    vertices, faces, _, _ = jumping_cubes(volume, scale, bottom_left_loc=np.array([-1.,-1.,-1.]))

    # other_geoms = show_grid(grid_points)
    other_geoms = []
    show_mesh(vertices, faces, other_geoms=other_geoms)

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()