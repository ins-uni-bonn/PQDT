import numpy as np
import open3d as o3d
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ---------- Helpers ----------
def fibonacci_sphere(n):
    i = np.arange(n)
    g = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / (g**2)
    z = 1 - 2 * i / max(n - 1, 1)
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

def look_at(eye, center=np.array([0.,0.,0.], dtype=np.float32), up=np.array([0.,1.,0.], dtype=np.float32)):
    # returns 3x3 rotation (world->cam) and translation so that R*(X - eye) is in camera frame
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], axis=0)  # cam axes rows
    return R

def make_rays_pinhole(eye, R_cam, w, h, vfov_deg):
    # Camera looks along -Z in its local frame; image plane at z=-1
    vfov = np.deg2rad(vfov_deg)
    aspect = w / h
    # y ranges from -tan(vfov/2) .. +tan(vfov/2); x scaled by aspect
    y_extent = np.tan(vfov * 0.5)
    x_extent = aspect * y_extent

    # pixel centers in NDC -> image plane coords
    xs = np.linspace(-x_extent, x_extent, w, dtype=np.float32)
    ys = np.linspace(-y_extent, y_extent, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # (h, w)

    # directions in camera space
    dirs_cam = np.stack([xx, -yy, -np.ones_like(xx)], axis=-1)  # (h,w,3)
    dirs_cam = dirs_cam.reshape(-1, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-12

    # to world: dir_world = R_cam^T * dir_cam; origins are all at eye
    dirs_world = (R_cam.T @ dirs_cam.T).T
    origins = np.repeat(eye[None, :], dirs_world.shape[0], axis=0)

    rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)  # (N,6)
    return rays

def sample_one_object(mesh_path, out_path, target_N, num_views, img_w, img_h, vfov_deg):
    """
    Sample points on the surface of a mesh using raycasting from multiple viewpoints.
    Saves outputs to out_path as gt.ply and gt.npy.
    Returns (True, message) on success, (False, error_message) on failure.
    """
    try:
        # ---------- Load & normalize ----------
        if not os.path.isfile(mesh_path):
            return False, f"Missing mesh: {mesh_path}"
        os.makedirs(out_path, exist_ok=True)

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_triangles():
            return False, "Mesh has no triangles."
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.compute_triangle_normals()

        bbox = mesh.get_axis_aligned_bounding_box()
        orig_center = bbox.get_center()
        orig_extent = np.max(bbox.get_extent())

        # Normalize to stable scale
        mesh.translate(-orig_center)
        if orig_extent > 0:
            mesh.scale(1.0 / orig_extent, center=(0, 0, 0))

        # ---------- RaycastingScene ----------
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        F = np.asarray(mesh.triangles, dtype=np.int32)
        num_faces = F.shape[0]

        # ---------- Shoot rays from many viewpoints ----------
        radius = 3.0  # camera distance (mesh is unit-ish)
        views = fibonacci_sphere(num_views) * radius

        all_hits_pts = []
        all_hits_nrms = []

        for eye in views:
            # stable up
            up = np.array([0., 0., 1.], dtype=np.float32)
            if np.abs(np.dot(up, (0 - eye) / (np.linalg.norm(eye) + 1e-12))) > 0.9:
                up = np.array([0., 1., 0.], dtype=np.float32)

            R_cam = look_at(eye.astype(np.float32), np.zeros(3, dtype=np.float32), up)
            rays = make_rays_pinhole(eye.astype(np.float32), R_cam, img_w, img_h, vfov_deg)

            rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            ans = scene.cast_rays(rays_t)

            t_hit = ans["t_hit"].numpy()  # inf on miss
            pid = ans["primitive_ids"].numpy()

            # Fix dtype if pid is uint32 (no-hit encoded as 2^32-1)
            if pid.dtype == np.uint32:
                pid = pid.view(np.int32)
            pid = pid.astype(np.int64)

            valid = np.isfinite(t_hit) & (pid >= 0) & (pid < num_faces)
            if not np.any(valid):
                continue

            o = rays[:, :3][valid]
            d = rays[:, 3:][valid]
            t = t_hit[valid][:, None]
            pts = o + t * d  # (M,3)

            tri_idx = pid[valid]
            tri_normals = np.asarray(mesh.triangle_normals)[tri_idx]

            all_hits_pts.append(pts)
            all_hits_nrms.append(tri_normals)

        # ---------- Merge & lightly clean ----------
        if not all_hits_pts:
            return False, "No ray hits found. Increase num_views/img resolution or check mesh scale."

        P = np.concatenate(all_hits_pts, axis=0)
        Nrm = np.concatenate(all_hits_nrms, axis=0)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pcd.normals = o3d.utility.Vector3dVector(Nrm)

        # Deduplicate with a small voxel to remove near-duplicates
        pcd = pcd.voxel_down_sample(voxel_size=0.0025)  # tune as needed

        # ---------- Downsample to EXACT target_N ----------
        M = np.asarray(pcd.points).shape[0]
        need = int(target_N)
        if M == 0:
            return False, "No points after voxel downsample; relax voxel_size or increase rays."

        if M >= need and hasattr(o3d.geometry.PointCloud, "farthest_point_down_sample"):
            pcd = pcd.farthest_point_down_sample(need)
        elif M >= need:
            idx = np.random.choice(M, size=need, replace=False)
            pcd = pcd.select_by_index(idx.tolist(), cleanup=False)
        # else: keep all if M < need

        # Optional smoothing of normals
        pcd.estimate_normals()

        # ---------- Transform back & save ----------
        pts = np.asarray(pcd.points)
        pts = pts * orig_extent + orig_center
        pcd.points = o3d.utility.Vector3dVector(pts)

        if pcd.has_normals():
            norms = np.asarray(pcd.normals)
            norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-12)
            pcd.normals = o3d.utility.Vector3dVector(norms)

        # Save
        o3d.io.write_point_cloud(os.path.join(out_path, "gt.ply"), pcd)
        np.save(os.path.join(out_path, "gt.npy"), pts)

        return True, f"Saved {pts.shape[0]} points."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"



def _worker_task(model_id, mesh_root, out_root, target_N, num_views, img_w, img_h, vfov_deg):
    """
    Worker wrapper to build paths from IDs and call sample_one_object.
    Returns (model_id, success_bool, message)
    """
    mesh_path = os.path.join(mesh_root, model_id, "models", "model_normalized.obj")
    out_path = os.path.join(out_root, model_id, "models")
    ok, msg = sample_one_object(mesh_path, out_path, target_N, num_views, img_w, img_h, vfov_deg)
    return model_id, ok, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="/home/ubuntu/dataset",
                        help='Root directory containing ShapeNet55-34/')
    parser.add_argument('--mesh_path', type=str, default=None,
                        help='Path to input meshes. Defaults to dataset_dir/ShapeNet55-34/shapenet_occlusion/02958343')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Directory to save output point clouds. Defaults to dataset_dir/ShapeNet55-34/occ_partial_noise/02958343')
    parser.add_argument('--target_N', type=int, default=8192, help='Target number of points to sample.')
    parser.add_argument('--num_views', type=int, default=24, help='Number of viewpoints to sample from.')
    parser.add_argument('--img_w', type=int, default=512, help='Width of the image plane for raycasting.')
    parser.add_argument('--img_h', type=int, default=512, help='Height of the image plane for raycasting.')
    parser.add_argument('--vfov_deg', type=float, default=50, help='Vertical field of view in degrees.')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel worker processes.')
    parser.add_argument('--limit', type=int, default=0, help='Optional: limit number of models processed (0 = all).')
    
    args = parser.parse_args()

    shapenet_root = os.path.join(args.dataset_dir, "ShapeNet55-34")
    if args.mesh_path is None:
        args.mesh_path = os.path.join(shapenet_root, "shapenet_occlusion", "02958343")
    if args.out_path is None:
        args.out_path = os.path.join(shapenet_root, "occ_partial_noise", "02958343")

    model_ids = sorted([f for f in os.listdir(args.out_path) if os.path.isdir(os.path.join(args.out_path, f))])
    if args.limit > 0:
        model_ids = model_ids[:args.limit]

    if not model_ids:
        print("No model IDs found under out_path.", file=sys.stderr)
        sys.exit(1)

    # Ensure spawn start method (safer with Open3D + PyTorch/NumPy)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # already set
        pass

    # Optional: reduce BLAS thread contention across workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    task = partial(
        _worker_task,
        mesh_root=args.mesh_path,
        out_root=args.out_path,
        target_N=args.target_N,
        num_views=args.num_views,
        img_w=args.img_w,
        img_h=args.img_h,
        vfov_deg=args.vfov_deg,
    )

    successes = 0
    failures = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, mid): mid for mid in model_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing models"):
            mid = futures[fut]
            try:
                model_id, ok, msg = fut.result()
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {e}"
                model_id = mid
            if ok:
                successes += 1
            else:
                failures += 1
            # Optional per-item logging:
            # print(f"[{model_id}] {'OK' if ok else 'ERR'}: {msg}")

    print(f"Done. Success: {successes} | Fail: {failures} | Total: {len(model_ids)}")
if __name__ == "__main__":
    main()
