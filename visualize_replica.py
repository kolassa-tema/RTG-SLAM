#!/usr/bin/env python3
# visualize_replica_rerun.py
import argparse
import json
import os
import re
import glob
import numpy as np
import cv2
import rerun as rr
import warnings
from math import sqrt


def rotation_matrix_to_quaternion(R: np.ndarray):
	"""Minimal, stable 3x3 -> (w,x,y,z)."""
	m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
	trace = m00 + m11 + m22
	if trace > 0.0:
		s = sqrt(trace + 1.0) * 2.0
		w = 0.25 * s
		x = (m21 - m12) / s
		y = (m02 - m20) / s
		z = (m10 - m01) / s
	else:
		if (m00 > m11) and (m00 > m22):
			s = sqrt(1.0 + m00 - m11 - m22) * 2.0
			w = (m21 - m12) / s
			x = 0.25 * s
			y = (m01 + m10) / s
			z = (m02 + m20) / s
		elif m11 > m22:
			s = sqrt(1.0 + m11 - m00 - m22) * 2.0
			w = (m02 - m20) / s
			x = (m01 + m10) / s
			y = 0.25 * s
			z = (m12 + m21) / s
		else:
			s = sqrt(1.0 + m22 - m00 - m11) * 2.0
			w = (m10 - m01) / s
			x = (m02 + m20) / s
			y = (m12 + m21) / s
			z = 0.25 * s
	q = np.array([w, x, y, z], dtype=np.float32)
	q /= max(1e-12, np.linalg.norm(q))
	return q

def read_cam_params(path):
	with open(path, "r") as f:
		data = json.load(f)
	cam = data["camera"]
	w = int(cam["w"])
	h = int(cam["h"])
	fx, fy = float(cam["fx"]), float(cam["fy"])
	cx, cy = float(cam["cx"]), float(cam["cy"])
	scale = float(cam.get("scale", 1.0))
	return w, h, fx*1.5, fy*1.5, cx, cy, scale

def read_traj(path):
	"""
	Expects each line to contain 16 numbers (row-major) forming a 4x4 camera-to-world matrix.
	Returns a list of (4,4) float32 arrays.
	"""
	mats = []
	with open(path, "r") as f:
		for line in f:
			vals = [float(x) for x in line.strip().split()]
			if len(vals) != 16:
				raise ValueError("Each line of traj.txt must have 16 numbers.")
			M = np.array(vals, dtype=np.float32).reshape(4, 4)
			mats.append(M)
	return mats

def depth_to_meters(depth_raw, scale_hint=1.0):
	d = depth_raw.astype(np.float32)
	return d / scale_hint
		

def backproject(depth_m, fx, fy, cx, cy, step=4):
	"""
	Backproject to camera space using a grid with stride 'step'.
	Returns Nx3 points and pixel indices used.
	"""
	# Get image dimensions
	img_height, img_width = depth_m.shape

	# Generate grid of pixel coordinates with stride 'step'
	pixel_rows = np.arange(0, img_height, step, dtype=np.int32)
	pixel_cols = np.arange(0, img_width, step, dtype=np.int32)
	grid_cols, grid_rows = np.meshgrid(pixel_cols, pixel_rows, indexing="xy")

	# Sample depth values at grid locations
	sampled_depth = depth_m[grid_rows, grid_cols]

	# Mask out invalid depth values (<= 0)
	valid_mask = sampled_depth > 0.0
	valid_cols = grid_cols[valid_mask].astype(np.float32)
	valid_rows = grid_rows[valid_mask].astype(np.float32)
	valid_depth = sampled_depth[valid_mask]

	# Backproject pixel coordinates to camera space (x, y, z)
	r_cam = (valid_cols - cx) * valid_depth / fx
	d_cam =(valid_rows - cy) * valid_depth / fy
	f_cam = valid_depth

	# Stack into Nx3 array of 3D points
	points_camera_space = np.stack([r_cam,d_cam,f_cam], axis=1)

	# Return 3D points and corresponding pixel indices
	return points_camera_space, (valid_cols.astype(np.int32), valid_rows.astype(np.int32))

def decode_depth_image(arr):
	if arr is None:
		return None
	if arr.ndim == 2:
		return arr.astype(np.float32)
	if arr.ndim == 3 and arr.shape[2] == 4:
		# Check if only R varies (common for your dataset's depth PNGs)
		b, g, r, a = cv2.split(arr)
		if len(np.unique(r)) > 2 and len(np.unique(b)) == 1 and len(np.unique(g)) == 1:
			return r.astype(np.float32)
		# Fallback grayscale
		return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY).astype(np.float32)
	if arr.ndim == 3 and arr.shape[2] == 3:
		return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32)
	raise ValueError("Unsupported depth image format")

def flip_rgb(rgb: np.ndarray) -> np.ndarray:
	"""
	Flip RGB horizontally.
	"""
	if rgb is None:
		return rgb
	return cv2.flip(rgb, 1)

def main():
	ap = argparse.ArgumentParser(description="Visualize Replica RGB-D with Rerun (camera pose, depth, point cloud).")
	ap.add_argument("dataset_root", type=str, help="Path containing cam_params.json, traj.txt, results/")
	ap.add_argument("--max_frames", type=int, default=10, help="How many frames to stream")
	ap.add_argument("--start_idx", type=int, default=0, help="Starting frame index (matched number in filenames)")
	ap.add_argument("--step", type=int, default=4, help="Subsample factor for point cloud")
	ap.add_argument("--entity", type=str, default="world", help="Rerun entity root")
	ap.add_argument("--pose_offset", type=int, default=0, help="Offset added to frame index when selecting pose (may be negative)")
	ap.add_argument("--flip_rgb", action="store_true", help="Flip RGB horizontally")
	ap.add_argument(
		"--accumulate_points",
		action="store_true",
		help="Accumulate point clouds over all frames and display them at once",
	)
	args = ap.parse_args()

	# Required files / dirs
	cam_json = os.path.join(args.dataset_root, "..", "cam_params.json")
	traj_txt = os.path.join(args.dataset_root, "traj.txt")
	results_dir = os.path.join(args.dataset_root, "results")
	if not (os.path.isfile(cam_json) and os.path.isfile(traj_txt) and os.path.isdir(results_dir)):
		raise SystemExit("Expected cam_params.json, traj.txt, and a results/ folder inside dataset_root.")

	# Load camera params & trajectory
	w, h, fx, fy, cx, cy, scale = read_cam_params(cam_json)
	# scale comes from cam_params.json; no override complexity

	poses = read_traj(traj_txt)  # list of (4,4) c2w matrices

	# Gather frames
	# Accept both with and without underscore, both jpg/png for color
	color_patterns = ["frame_*.png", "frame*.png", "frame_*.jpg", "frame*.jpg"]
	color_paths = []
	for pat in color_patterns:
		color_paths.extend(glob.glob(os.path.join(results_dir, pat)))
	color_paths = sorted(set(color_paths))
	depth_patterns = ["depth_*.png", "depth*.png"]
	depth_paths = []
	for pat in depth_patterns:
		depth_paths.extend(glob.glob(os.path.join(results_dir, pat)))
	depth_paths = sorted(set(depth_paths))

	# Match by numeric index suffix
	def idx_from_name(p, prefix):
		# Accept frame000123.jpg, frame_000123.png, depth000123.png, depth_000123.png
		basename = os.path.basename(p)
		m = re.search(rf"{prefix}_?(\d+)\.(png|jpg)$", basename, re.IGNORECASE)
		return int(m.group(1)) if m else -1

	color_by_idx = {idx_from_name(p, "frame"): p for p in color_paths}
	depth_by_idx = {idx_from_name(p, "depth"): p for p in depth_paths}

	# Build triplets: (index, color_path, depth_path, pose)
	indices = sorted(set(color_by_idx.keys()) & set(depth_by_idx.keys()))
	if not indices:
		raise SystemExit("No matching frame/depth pairs found in results/.")

	rr.init("Replica RGB-D Viewer")
	rr.serve_web(open_browser=True)

	# Coordinate convention for world
	rr.log(args.entity, rr.ViewCoordinates.RFU, static=True)
	#

	# Log the pinhole (intrinsics + resolution)
	rr.log(
		f"{args.entity}/camera",
		rr.Pinhole(
			focal_length=np.array([fx, fy], dtype=np.float32),
			principal_point=np.array([cx, cy], dtype=np.float32),
			resolution=[w, h],
			camera_xyz=rr.ViewCoordinates.RDF
		),
		static=True,
	)

	# Optional accumulators
	all_pts_world = []
	all_colors = []

	# Loop frames
	# Apply start index filter
	sel_indices = [i for i in indices if i >= args.start_idx]
	if not sel_indices:
		raise SystemExit("No frames found at or after start_idx.")
	for k, idx in enumerate(sel_indices[:args.max_frames]):
		# Set a time sequence so frames don't overwrite each other
		try:
			rr.set_time_sequence("frame", k)
		except Exception:
			pass
		cpath = color_by_idx[idx]
		dpath = depth_by_idx[idx]

		
		pose_idx = idx + args.pose_offset
		if pose_idx < 0:
			pose_idx = 0
		elif pose_idx >= len(poses):
			pose_idx = len(poses) - 1
		pose = poses[pose_idx]
		c2w = pose.astype(np.float32)
		R_unity = c2w[:3, :3]
		t = c2w[:3, 3]
		
		S_y = np.array([
			[1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0],
			[0.0, 1.0, 0.0]
		], dtype=np.float32)
		S_y = np.diag([1.0, 0.0, 0.0]).astype(np.float32)
		R = c2w[:3, :3] #

		# Unity conversion
		# R = S_y @ R_unity @ S_y  
		# t = S_y @ t
		# R to identity
		# R = np.eye(3, dtype=np.float32)

		# Images
		color_bgr = cv2.imread(cpath, cv2.IMREAD_UNCHANGED)
		if color_bgr is None:
			continue
		color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

		depth_raw = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
		if depth_raw is None:
			continue
		depth_scalar = decode_depth_image(depth_raw)
		# uniq_vals = np.unique(depth_scalar)
		# if uniq_vals.size <= 2 and 255 in uniq_vals and depth_scalar.dtype != np.float32:
		# 	depth_scalar[depth_scalar == 255] = 0
		depth_m = depth_to_meters(depth_scalar, scale_hint=scale)
		depth_m[(depth_m <= 0) | (depth_m > 50.0)] = 0

		# Optionally flip RGB to match depth image resolution
		if args.flip_rgb:
			color = flip_rgb(color)

		# Pose transform (convert to quaternion because passing raw 3x3 now errors in newer Rerun)
		quat = rotation_matrix_to_quaternion(R)
		rr.log(f"{args.entity}/camera", rr.Transform3D(mat3x3=R, translation=t))


		# Raw imagery
		try:
			rr.log(f"{args.entity}/camera/rgb",rr.Image(color))
		except Exception as e:
			print(f"[warn] Failed to log RGB image: {e}")
		try:
			rr.log(f"{args.entity}/camera/depth_m",  rr.DepthImage(depth_m, meter=1.0))
		except Exception as e:
			print(f"[warn] Failed to log depth image: {e}")

		pts_cam, (uu, vv) = backproject(depth_m, fx, fy, cx, cy, step=args.step)
		uu = np.clip(uu, 0, color.shape[1] - 1)
		vv = np.clip(vv, 0, color.shape[0] - 1)
		colors = color[vv, uu].reshape(-1, 3)
		pts_world = ((R @ pts_cam.T).T + t)

		if pts_world.size == 0:
			print("[info] No valid depth points for frame", idx)
		else:
			if args.accumulate_points:
				all_pts_world.append(pts_world)
				all_colors.append(colors)
			else:
				try:
					rr.log(
						f"{args.entity}/points",
						rr.Points3D(pts_world, colors=colors, radii=0.005),
					)
				except Exception as e:
					print(f"[warn] Failed to log point cloud: {e}")

	# After all frames, log accumulated point cloud once
	if args.accumulate_points and all_pts_world:
		pts_world_cat = np.concatenate(all_pts_world, axis=0)
		colors_cat = np.concatenate(all_colors, axis=0)
		try:
			# Use a final time step so it doesn't overwrite per-frame logs (if any)
			try:
				rr.set_time_sequence("frame", len(sel_indices))
			except Exception:
				pass
			rr.log(
				f"{args.entity}/points_accumulated",
				rr.Points3D(pts_world_cat, colors=colors_cat, radii=0.005),
			)
		except Exception as e:
			print(f"[warn] Failed to log accumulated point cloud: {e}")

	print("Done. Inspect the Rerun viewer timeline and entities.")

if __name__ == "__main__":
	main()
