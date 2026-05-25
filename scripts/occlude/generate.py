import argparse
import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
from open3d.t.geometry import RaycastingScene
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
import torch
import multiprocessing as mp
from functools import partial
import gc
# Reference
# https://sketchfab.com/3d-models/lowpoly-people-waldo-9ec7a14729aa490fa712e51c217db0f5
# https://sketchfab.com/3d-models/traffic-set-3051e3677ef14e9dbe4490a640aa11fd
# https://sketchfab.com/3d-models/speed-bump-bump-the-traffic-avenue-13mb-5cab02a4935044218cec674e29f2df0c

SCALES = {'44_person': 0.035, '122_person': 0.035, '127_person': 0.035, '135_person': 0.035, '280_person': 0.035,
          'bump': 0.1, 'direction_1_9': 0.15, 'lamp_2': 0.2, 'pontoon1_5': 0.15, 'pontoon2_0': 0.15, 'stop_3': 0.2}

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    if pcd.shape[1] == n_points:
        return pcd
    elif pcd.shape[1] < n_points:
        raise ValueError(
            'FPS subsampling receives a larger n_points: {:d} > {:d}'.format(
                n_points, pcd.shape[1]))
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(),
        furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


def save_points(points, out_path):
    """
    Save points to a .npy file.
    Args:
        points: (N, 3) numpy array of points
        out_path: path to save the points
    """
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    np.save(out_path, points.astype(np.float32))


    

def process_single_occ(
                 mesh,
                 occ,
                 mesh_id,
                 i,
                 out_dir,
                 occ_shift=-0.06,
                 bbox_rescale_factor=1.2,
                 n_points=2048,
                 n_rays=300000,
                 noise_std=0.003,
                 occ_num_lower_bound=20,
                 vis=False,
                 id_bias=0
                 ):

    # ----------------- Prepare the mesh and occ ---------------


    scene = RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    
    # Define LiDAR origin and rays
    x = np.random.uniform(-1.0, 1.0)
    y = np.sqrt(1 - x**2) * np.random.choice([-1, 1])
    z = (mesh.get_max_bound()[1] + mesh.get_min_bound()[1]) / 2
    origin = np.array([[x, z, y]])

    # let occ locate between the mesh and the lidar origin
    mesh_center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    direction = mesh_center - origin[0]
    direction /= np.linalg.norm(direction)
    ray = np.hstack([origin, direction.reshape(1, 3)])
    ray = o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32)
    #find the intersection point of the line and the mesh
    hit = scene.cast_rays(ray)
    if hit['t_hit'].numpy()[0] == np.inf:
        #print("No intersection found, skipping this sample.")
        return False
    intersection_point = origin + hit['t_hit'].numpy()[0] * direction
    # rotate occ randomly around z axis
    angle = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, angle, 0))
    occ.rotate(rotation_matrix, center=occ.get_center())
    # Move the occ mesh to the intersection point
    occ.translate((intersection_point - occ.get_center()).flatten())
    # move intersection point a little bit away from the mesh

    x_shift = occ_shift * direction[0]
    y_shift = occ_shift * direction[2]
    occ.translate((x_shift, 0, y_shift))

    # make occ z min same with mesh z min
    occ_z_min = occ.get_min_bound()[1]
    mesh_z_min = mesh.get_min_bound()[1]
    occ.translate((0, mesh_z_min - occ_z_min, 0))

    # ----------------- Cast rays and sample points ---------------

    scene_mesh = RaycastingScene()
    _ = scene_mesh.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occ))
    # Define LiDAR origin and rays
    x = np.random.uniform(-0.7, 0.7)
    y = np.sqrt(0.7 - x**2) * np.random.choice([-1, 1])
    z = np.random.uniform(0.05, 0.12)
    lidar_origin = np.array([[x, z, y]])

    # Uniformly sample directions on a sphere using the Fibonacci lattice
    indices = np.arange(0, n_rays, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_rays)
    theta = np.pi * (1 + 5**0.5) * indices

    directions = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    # filter out the backward directions in the direction of the mesh
    directions = directions[(directions @ direction) > 0]
    n_rays = directions.shape[0]

    rays = np.hstack([np.tile(lidar_origin, (n_rays, 1)), directions])
    rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    # Cast rays
    hits = scene.cast_rays(rays_o3d)
    # Filter valid hits
    hit_points = hits['t_hit'].numpy()
    #print(f"Number of all hits: {np.sum(hit_points != np.inf)}")
    mask = hit_points != np.inf
    points = rays[mask][:, :3] + hit_points[mask, np.newaxis] * directions[mask]
    

    # Cast rays
    hits = scene_mesh.cast_rays(rays_o3d)
    # Filter valid hits
    hit_points = hits['t_hit'].numpy()
    mask = hit_points != np.inf
    points_mesh = rays[mask][:, :3] + hit_points[mask, np.newaxis] * directions[mask]
    #print(f"Occ num: {len(points) - len(points_mesh)}")
    if (len(points) - len(points_mesh)) < occ_num_lower_bound:
        #print(f"[{i}] Occ num: {len(points) - len(points_mesh)}. Not enough points sampled from occlusion, resample.")
        return False


    # filter out points that are outside the mesh bounding box
    bbox = mesh.get_oriented_bounding_box()
    bbox.scale(bbox_rescale_factor, center=bbox.get_center())
    points = points[(bbox.get_min_bound() <= points).all(axis=1) & 
                   (points <= bbox.get_max_bound()).all(axis=1)]

    # Subsample points using farthest point sampling
    points = torch.from_numpy(points).float().unsqueeze(0).to('cuda')
    try:
        fps_points = fps_subsample(points, n_points).squeeze(0).cpu().numpy()
    except Exception as e:
        #print(f"[{i}] Error during subsampling: {e}")
        return False
    

    # add gaussian noise to points
    noise = np.random.normal(0, noise_std, fps_points.shape)
    fps_points += noise

    if vis:
        # Create point cloud from hit points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(fps_points)
        # Visualize lidar scan lines
        origin_line_set = o3d.geometry.LineSet()
        origin_line_set.points = o3d.utility.Vector3dVector(np.concatenate([lidar_origin, fps_points], axis=0))
        origin_line_set.lines = o3d.utility.Vector2iVector([[0, i+1] for i in range(len(fps_points))])
        origin_line_set.paint_uniform_color([1, 0.8, 0.8])  # Red color for origin lines

        intersection = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        intersection.translate(intersection_point.flatten())

        origin_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        origin_vis.translate(origin.flatten())


        o3d.visualization.draw_geometries([point_cloud, intersection, origin_vis], window_name="Raycasting Scene", width=800, height=600)

    # save the point cloud as .npy file
    out_path = os.path.join(out_dir, mesh_id, 'models', f'{i+id_bias}.npy')
    save_points(fps_points, out_path)
    return True


def process_single_model(
                 obj_path,
                 occ_paths,
                 out_dir,
                 samples_per_model=64,
                 occ_shift=-0.06,
                 bbox_rescale_factor=1.2,
                 n_points=2048,
                 n_rays=300000,
                 noise_std=0.003,
                 occ_num_lower_bound=20,
                 vis=False,
                 id_bias=0
                 ):
    # Load car mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    for i in range(samples_per_model):
        occ_path = occ_paths[np.random.randint(0, len(occ_paths))]
        occ = o3d.io.read_triangle_mesh(occ_path)
        if not occ.has_vertex_normals():
            occ.compute_vertex_normals()
        occ_scale = SCALES.get(os.path.basename(occ_path).split('.')[0])
        occ.scale(occ_scale, center=occ.get_center())
        if_success = False
        cnt = 0
        while not if_success:
            if_success = process_single_occ(
                mesh,
                occ,
                mesh_id=obj_path.split('/')[-3],
                i=i,
                out_dir=out_dir,
                occ_shift=occ_shift,
                bbox_rescale_factor=bbox_rescale_factor,
                n_points=n_points,
                n_rays=n_rays,
                noise_std=noise_std,
                occ_num_lower_bound=occ_num_lower_bound,
                vis=vis,
                id_bias=id_bias
            )
            cnt += 1
            if cnt > 20:
                print(f"Failed to process {obj_path} with {occ_path} after 20 attempts, skipping this model.")
                return


def process_model_wrapper(obj_path, occ_paths, args):
    try:
        model_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
        #print(f"Started processing {model_id}")
        process_single_model(
            obj_path,
            occ_paths,
            out_dir=args.output_dir,
            occ_shift=args.occ_shift,
            samples_per_model=args.samples_per_model,
            bbox_rescale_factor=args.bbox_rescale_factor,
            n_points=args.n_points,
            n_rays=args.n_rays,
            noise_std=args.noise_std,
            occ_num_lower_bound=args.occ_num_lower_bound,
            vis=args.vis,
            id_bias=args.id_bias
        )
        return True
    except Exception as e:
        #print(f"Error processing {obj_path}:\n{e}")
        return False


def main():
    # This function is just a placeholder to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/dataset',
                        help='Root directory containing ShapeNet55-34/')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Input occlusion mesh directory. Defaults to dataset_dir/ShapeNet55-34/shapenet_occlusion')
    parser.add_argument('--category', type=str, default='02958343')  # car
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. Defaults to dataset_dir/ShapeNet55-34/occ_partial_noise/02958343')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='Split directory. Defaults to dataset_dir/ShapeNet55-34/Occ_ShapeNet_Car_Noise')
    parser.add_argument('--n_points', type=int, default=2048)
    parser.add_argument('--n_rays', type=int, default=150000)
    parser.add_argument('--samples_per_model', type=int, default=32)
    parser.add_argument('--bbox_rescale_factor', type=float, default=1.2) #1.1 1.2 1.3
    parser.add_argument('--occ_shift', type=float, default=-0.06)
    parser.add_argument('--noise_std', type=float, default=0.004) #0.003 0.004 0.005
    parser.add_argument('--occ_num_lower_bound', type=int, default=20) #10 20 30
    parser.add_argument('--vis',type=bool, default=False, help="Visualize the results")
    parser.add_argument('--id_bias', type=int, default=32, help="ID bias for the output files") #0 32 64
    parser.add_argument('--workers', type=int, default=30, help="Number of worker processes")
    parser.add_argument('--repair', type=bool, default=False, help="Repair mode: check if the models name in model_path already exists in the output directory")
    parser.add_argument('--gen_split', type=bool, default=False, help="Generate train/test split files")
    args = parser.parse_args()

    shapenet_root = os.path.join(args.dataset_dir, 'ShapeNet55-34')
    if args.data_dir is None:
        args.data_dir = os.path.join(shapenet_root, 'shapenet_occlusion')
    if args.output_dir is None:
        args.output_dir = os.path.join(shapenet_root, 'occ_partial_noise', args.category)
    if args.split_dir is None:
        args.split_dir = os.path.join(shapenet_root, 'Occ_ShapeNet_Car_Noise')

    model_dir = os.path.join(args.data_dir, args.category)
    print(f"Data directory: {model_dir}")
    occ_dir = os.path.join(args.data_dir, 'occ')
    #get all models list
    model_paths = [f for f in os.listdir(model_dir)]
    if args.id_bias > 0:
        model_paths = [f for f in model_paths if os.path.exists(os.path.join(args.output_dir, os.path.basename(f)))]
    if args.gen_split:
        ratio = 0.8
        out_ids = [f for f in os.listdir(args.output_dir)]
        n_train = int(len(model_paths) * ratio)
        # shuffle and split the out_paths
        np.random.shuffle(out_ids)
        train_paths = out_ids[:n_train]
        test_paths = out_ids[n_train:]
        print(f"Train set: {len(train_paths)} models, Test set: {len(test_paths)} models")
        # save the train and test paths to files
        if not os.path.exists(args.split_dir):
            os.makedirs(args.split_dir)
        with open(os.path.join(args.split_dir, 'train.txt'), 'w') as f:
            for path in train_paths:
                f.write(args.category + '-' + path + '.npy' + '\n')
        with open(os.path.join(args.split_dir, 'test.txt'), 'w') as f:
            for path in test_paths:
                f.write(args.category + '-' + path + '.npy' '\n')
        
        
    if args.repair:
        #check if the models name in model_path already exists in the output directory
        #model_paths = [f for f in model_paths if not os.path.exists(os.path.join(args.output_dir, os.path.basename(f)))]
        #check if all model folder in output_dir have samples_per_model files
        model_paths = []
        out_paths = [f for f in os.listdir(args.output_dir)]
        for f in out_paths:
            if len(os.listdir(os.path.join(args.output_dir, f, 'models'))) < args.samples_per_model:
                model_paths.append(f)
                print(f"Model {f} has less than {args.samples_per_model} samples, adding to the repair list.")
        print(f"Repair mode: {len(model_paths)} models to process.")

    obj_paths = [os.path.join(model_dir, f, 'models/model_normalized.obj') for f in model_paths]
    for obj_path in obj_paths:
        if not os.path.exists(obj_path):
            print(f"Model {obj_path} does not exist, skipping.")
            obj_paths.remove(obj_path)

    occ_paths = [os.path.join(occ_dir, f) for f in os.listdir(occ_dir) if f.endswith('.obj')]

    print(f"Processing {len(obj_paths)} models using {args.workers} workers...")

    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(partial(process_model_wrapper, occ_paths=occ_paths, args=args), obj_paths),
                      total=len(obj_paths), desc="Processing models"):
            pass


if __name__ == '__main__':
    main()

# easy config
# python scripts/occlude/generate.py --bbox_rescale_factor 1.1 --noise_std 0.003 --occ_num_lower_bound 10 --id_bias 0
# medium config
# python scripts/occlude/generate.py --bbox_rescale_factor 1.2 --noise_std 0.004 --occ_num_lower_bound 20 --id_bias 32
# hard config
# python scripts/occlude/generate.py --bbox_rescale_factor 1.3 --noise_std 0.005 --occ_num_lower_bound 30 --id_bias 64
