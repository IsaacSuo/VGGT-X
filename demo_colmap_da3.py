# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT-X with DA3 (Depth Anything V3) as the backbone.
Replaces VGGT model with DA3 for depth and pose estimation.
"""

import sys
from pathlib import Path
# Add Depth-Anything-3/src to path (relative to this script)
_DA3_PATH = Path(__file__).parent / "Depth-Anything-3" / "src"
sys.path.insert(0, str(_DA3_PATH))

import random
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
import trimesh
import utils.opt as opt_utils
import utils.colmap as colmap_utils
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from utils.metric_torch import evaluate_auc, evaluate_pcd, write_evaluation_results

from depth_anything_3.api import DepthAnything3
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


torch._dynamo.config.accumulated_cache_size_limit = 512


def run_DA3(image_paths, device, dtype, model_name='da3-giant', process_res=504):
    """
    Run DA3 for camera and depth estimation.

    Args:
        image_paths: List of image file paths
        device: torch device
        dtype: torch dtype (not used by DA3, it auto-selects)
        model_name: DA3 model variant
        process_res: Processing resolution

    Returns:
        extrinsic: [N, 3, 4] numpy array, w2c
        intrinsic: [N, 3, 3] numpy array
        depth_map: [N, H, W] numpy array
        depth_conf: [N, H, W] numpy array
    """
    # Load DA3 model
    model = DepthAnything3(model_name=model_name)
    model = model.to(device).eval()
    print(f"DA3 model ({model_name}) loaded")

    # Run inference
    prediction = model.inference(
        image_paths,
        process_res=process_res,
        process_res_method='upper_bound_resize',
    )

    # Extract outputs (already numpy arrays)
    extrinsic = prediction.extrinsics  # [N, 3, 4]
    intrinsic = prediction.intrinsics  # [N, 3, 3]
    depth_map = prediction.depth[..., np.newaxis]  # [N, H, W] -> [N, H, W, 1] to match VGGT format
    depth_conf = prediction.conf       # [N, H, W]

    # DA3 returns processed image size, store for later use
    processed_size = (prediction.depth.shape[1], prediction.depth.shape[2])  # (H, W)

    # Debug: check for NaN/Inf in outputs
    print(f"[DA3 Debug] extrinsic shape: {extrinsic.shape}, has NaN: {np.isnan(extrinsic).any()}, has Inf: {np.isinf(extrinsic).any()}")
    print(f"[DA3 Debug] intrinsic shape: {intrinsic.shape}, has NaN: {np.isnan(intrinsic).any()}, has Inf: {np.isinf(intrinsic).any()}")
    print(f"[DA3 Debug] depth_map range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
    print(f"[DA3 Debug] extrinsic sample (first frame):\n{extrinsic[0]}")

    return extrinsic, intrinsic, depth_map, depth_conf, processed_size


def load_images_for_ga(image_paths, target_size):
    """
    Load images for Global Alignment (GA) feature matching.
    Returns images as tensor [N, 3, H, W] normalized to [0, 1].
    """
    from PIL import Image
    from torchvision import transforms as TF

    images = []
    original_coords = []
    to_tensor = TF.ToTensor()

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        # Resize to target size while maintaining aspect ratio
        scale = target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)

        # Pad to square
        max_dim = max(new_width, new_height)
        padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        left = (max_dim - new_width) // 2
        top = (max_dim - new_height) // 2
        padded.paste(img, (left, top))

        # Final resize to target_size x target_size
        padded = padded.resize((target_size, target_size), Image.BILINEAR)

        images.append(to_tensor(padded))
        # Store original coords: [x1, y1, x2, y2, orig_w, orig_h]
        original_coords.append([left, top, left + new_width, top + new_height, width, height])

    images = torch.stack(images, dim=0)  # [N, 3, H, W]
    original_coords = torch.tensor(original_coords, dtype=torch.float32)

    return images, original_coords


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-X with DA3 Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--post_fix", type=str, default="_da3_x", help="Post fix for the output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ga", action="store_true", default=False, help="Whether to apply global alignment for better reconstruction")
    parser.add_argument("--save_depth", action="store_true", default=False, help="If save depth")
    parser.add_argument("--model_name", type=str, default="da3-giant", help="DA3 model variant")
    parser.add_argument("--process_res", type=int, default=504, help="Processing resolution for DA3")
    parser.add_argument("--total_frame_num", type=int, default=None, help="Number of frames to reconstruct")
    ######### GA parameters #########
    parser.add_argument("--max_query_pts", type=int, default=None, help="Maximum number of query points")
    parser.add_argument("--max_points_for_colmap", type=int, default=500000, help="Maximum number for colmap point cloud")
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    return parser.parse_args()


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    target_scene_dir = os.path.join(f"{os.path.dirname(args.scene_dir)}{args.post_fix}", os.path.basename(args.scene_dir))
    os.makedirs(target_scene_dir, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Get image paths
    image_dir = os.path.join(args.scene_dir, "images")
    if args.total_frame_num is None:
        args.total_frame_num = len(os.listdir(image_dir))

    images_gt_updated = None
    if os.path.exists(os.path.join(args.scene_dir, "sparse/0/images.bin")):
        print("Using order of ground truth images from COLMAP sparse reconstruction")
        images_gt = colmap_utils.read_images_binary(os.path.join(args.scene_dir, "sparse/0/images.bin"))
        assert args.total_frame_num <= len(images_gt), f"Requested total_frame_num {args.total_frame_num} exceeds available images {len(images_gt)}"

        images_gt = dict(list(images_gt.items())[:args.total_frame_num])
        images_gt_keys = list(images_gt.keys())

        random.shuffle(images_gt_keys)
        images_gt_updated = {id: images_gt[id] for id in list(images_gt_keys)}
        image_path_list = [os.path.join(image_dir, images_gt_updated[id].name) for id in images_gt_updated.keys()]

        inverse_idx = [images_gt_keys.index(key) for key in sorted(list(images_gt.keys()))]
    else:
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))[:args.total_frame_num]
        if not image_path_list:
            raise ValueError(f"No images found in {image_dir}")
        inverse_idx = list(range(len(image_path_list)))

    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    base_image_path_list_inv = [base_image_path_list[i] for i in inverse_idx]

    print(f"Found {len(image_path_list)} images in {image_dir}")

    torch.cuda.reset_peak_memory_stats()
    start_time = datetime.now()

    # Run DA3 to estimate camera and depth
    extrinsic, intrinsic, depth_map, depth_conf, processed_size = run_DA3(
        image_path_list, device, dtype, args.model_name, args.process_res
    )

    # Load images for GA and evaluation (if needed)
    # Use DA3's actual output size, not process_res (which may differ due to aspect ratio)
    img_load_resolution = max(processed_size)  # Use the larger dimension
    images, original_coords = load_images_for_ga(image_path_list, img_load_resolution)

    # Resize images to match DA3's depth output size (H, W)
    images = F.interpolate(images, size=processed_size, mode="bilinear", align_corners=False)
    original_coords = original_coords.to(device)
    images = images.to(device)

    if args.use_ga:
        if os.path.exists(os.path.join(target_scene_dir, "matches.pt")):
            print(f"Found existing matches at {os.path.join(target_scene_dir, 'matches.pt')}, loading it")
            match_outputs = torch.load(os.path.join(target_scene_dir, "matches.pt"), weights_only=False)
        else:
            print("Extracting matches for global alignment")
            if args.max_query_pts is None:
                args.max_query_pts = 4096 if len(images) < 500 else 2048
            match_outputs = opt_utils.extract_matches(extrinsic, intrinsic, images, depth_conf, base_image_path_list, args.max_query_pts)
            match_outputs["original_width"] = images.shape[-1]
            match_outputs["original_height"] = images.shape[-2]
            torch.save(match_outputs, os.path.join(target_scene_dir, "matches.pt"))
            print(f"Saved matches to {os.path.join(target_scene_dir, 'matches.pt')}")
        extrinsic, intrinsic = opt_utils.pose_optimization(
            match_outputs, extrinsic, intrinsic, images, depth_map, depth_conf,
            base_image_path_list, target_scene_dir=target_scene_dir, shared_intrinsics=args.shared_camera,
        )

    end_time = datetime.now()
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    conf_thres_value = np.percentile(depth_conf, 0.5)
    print(f"Using confidence threshold: {conf_thres_value}")
    shared_camera = False
    camera_type = "PINHOLE"

    c = 2.5  # scale factor for better reconstruction
    extrinsic[:, :3, 3] *= c
    depth_map *= c

    if os.path.exists(os.path.join(args.scene_dir, "sparse/0/points3D.bin")):
        print("Found ground truth colmap results, evaluating reconstruction quality")
        pcd_gt = colmap_utils.read_points3D_binary(os.path.join(args.scene_dir, "sparse/0/points3D.bin"))

        if images_gt_updated is not None:
            translation_gt = torch.tensor([image.tvec for image in images_gt_updated.values()], device=device)
            rotation_gt = torch.tensor([colmap_utils.qvec2rotmat(image.qvec) for image in images_gt_updated.values()], device=device)

            gt_se3 = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
            gt_se3[:, :3, :3] = rotation_gt
            gt_se3[:, 3, :3] = translation_gt

            pred_se3 = torch.eye(4, device=device).unsqueeze(0).repeat(len(images_gt_updated), 1, 1)
            pred_se3[:, :3, :3] = torch.tensor(extrinsic[:, :3, :3], device=device)
            pred_se3[:, 3, :3] = torch.tensor(extrinsic[:, :3, 3], device=device)

            auc_results, pred_se3_aligned, c, R, t = evaluate_auc(pred_se3, gt_se3, device, return_aligned=True)

            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            points_3d_transformed = c * (points_3d @ R.T) + t.T

            pcd_results = evaluate_pcd(
                pcd_gt, points_3d_transformed, depth_conf, images,
                images_gt_updated, original_coords,
                img_load_resolution, conf_thresh=1.5 if depth_conf.max() > 1.5 else conf_thres_value,
            )

            write_evaluation_results(target_scene_dir, len(images_gt_updated), auc_results, pcd_results,
                                     (end_time - start_time).total_seconds(), peak_mem_mb)

    else:
        print("No ground truth points3D.bin found, using random sampling")
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.save_depth:
        target_depth_dir = os.path.join(target_scene_dir, "estimated_depths")
        target_conf_dir = os.path.join(target_scene_dir, "estimated_confs")
        os.makedirs(target_depth_dir, exist_ok=True)
        os.makedirs(target_conf_dir, exist_ok=True)

        for idx, image_path in tqdm(enumerate(image_path_list), desc="Saving depth maps and confidences"):
            inverse_depth_map = 1 / (depth_map[idx] + 1e-8)
            normalized_inverse_depth_map = (inverse_depth_map - inverse_depth_map.min()) / (inverse_depth_map.max() - inverse_depth_map.min())
            depth_map_path = os.path.join(target_depth_dir, f"{os.path.basename(image_path)}.npy")
            depth_conf_path = os.path.join(target_conf_dir, f"{os.path.basename(image_path)}.npy")
            np.save(depth_map_path, normalized_inverse_depth_map.squeeze())
            np.save(depth_conf_path, depth_conf[idx].squeeze())

    image_size = np.array([depth_map.shape[1], depth_map.shape[2]])
    num_frames, height, width = depth_map.shape[:3]  # depth_map is [N, H, W, 1]

    points_rgb = F.interpolate(
        images, size=(depth_map.shape[1], depth_map.shape[2]), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    if args.use_ga:
        conf_mask = opt_utils.extract_conf_mask(match_outputs, depth_conf, base_image_path_list)
        conf_mask = conf_mask & (depth_conf >= conf_thres_value)
        conf_mask = randomly_limit_trues(conf_mask, args.max_points_for_colmap)
    else:
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, args.max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic[inverse_idx],
        intrinsic[inverse_idx],
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction_resolution = (depth_map.shape[2], depth_map.shape[1])

    reconstruction = colmap_utils.rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list_inv,
        original_coords.cpu().numpy()[inverse_idx],
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # Symlink original files
    for item in os.listdir(args.scene_dir):
        if item != "sparse" and not item.endswith("results.txt"):
            src = os.path.join(args.scene_dir, item)
            dst = os.path.join(target_scene_dir, item)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                for file in os.listdir(src):
                    if not os.path.exists(os.path.join(dst, file)):
                        os.symlink(os.path.abspath(os.path.join(src, file)), os.path.abspath(os.path.join(dst, file)))
            else:
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), os.path.abspath(dst))

    print(f"Saving reconstruction to {target_scene_dir}/sparse/0")
    sparse_reconstruction_dir = os.path.join(target_scene_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(target_scene_dir, "sparse/points.ply"))

    return True


if __name__ == "__main__":
    args = parse_args()
    demo_fn(args)
