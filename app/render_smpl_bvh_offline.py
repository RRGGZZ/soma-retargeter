#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import trimesh
import warp as wp

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils

from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.renderers.mesh_renderer import skinning_kernel, update_skinned_transform_kernel


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Zero-length vector is not allowed.")
    return vec / norm


def _camera_transform_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Build camera-to-world look-at transform for trimesh."""
    direction = _normalize(target - eye)
    up = _normalize(up)
    z_axis = -direction
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 0] = x_axis
    cam_to_world[:3, 1] = y_axis
    cam_to_world[:3, 2] = z_axis
    cam_to_world[:3, 3] = eye
    return cam_to_world


def _create_ffmpeg_pipe(output: pathlib.Path, fps: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg first.") from exc


def render_bvh_to_mp4(
    bvh_file: pathlib.Path,
    output_mp4: pathlib.Path,
    retarget_source: str,
    width: int,
    height: int,
    camera_pos: np.ndarray,
    look_at: np.ndarray,
    camera_up: np.ndarray,
    fov_y_deg: float,
    device: str,
):
    if not bvh_file.exists():
        raise FileNotFoundError(f"BVH file not found: {bvh_file}")

    skeleton, animation = bvh_utils.load_bvh(str(bvh_file), source_type=retarget_source)
    source_type = pipeline_utils.get_source_type_from_str(retarget_source)
    skeletal_mesh = pipeline_utils.get_source_model_mesh(source_type, skeleton)
    if skeletal_mesh is None:
        raise RuntimeError("Failed to load source skeletal mesh for rendering.")

    fps = max(1, int(round(float(animation.sample_rate))))
    num_frames = int(animation.num_frames)

    skeleton_instance = SkeletonInstance(
        skeleton=skeleton,
        color=wp.vec3(1.0, 1.0, 1.0),
        xform=wp.transform_identity(),
    )

    parent_indices = wp.array(skeleton_instance.parent_indices, dtype=wp.int32)
    skinned_transforms = wp.zeros((1, skeleton.num_joints), dtype=wp.transform)

    skinned_points_buffers = []
    trimesh_meshes = []
    for skinned_mesh in skeletal_mesh.skinned_meshes:
        points = skinned_mesh.points.numpy()
        faces = skinned_mesh.indices.numpy().reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        mesh.visual.vertex_colors = np.array([235, 245, 112, 255], dtype=np.uint8)
        trimesh_meshes.append(mesh)
        skinned_points_buffers.append(wp.zeros(skinned_mesh.num_points, dtype=wp.vec3))

    scene = trimesh.Scene()
    for mesh in trimesh_meshes:
        scene.add_geometry(mesh)

    scene.camera = trimesh.scene.Camera(
        resolution=(int(width), int(height)),
        fov=(float(fov_y_deg), float(fov_y_deg)),
    )
    scene.camera_transform = _camera_transform_look_at(camera_pos, look_at, camera_up)
    scene.background = np.array([0, 0, 0, 255], dtype=np.uint8)

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _create_ffmpeg_pipe(output_mp4, fps=fps)
    if ffmpeg.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin pipe.")

    print(f"[INFO]: Rendering {bvh_file} -> {output_mp4}")
    print(f"[INFO]: Frames={num_frames}, FPS={fps}, Resolution={width}x{height}")

    start_time = time.time()
    interrupted = False
    try:
        for frame_idx in range(num_frames):
            local_transforms = animation.get_local_transforms(frame_idx)
            skeleton_instance.set_local_transforms(local_transforms)

            wp.launch(
                update_skinned_transform_kernel,
                dim=1,
                inputs=[
                    skeleton_instance.num_joints,
                    wp.array(skeleton_instance.get_local_transforms(), dtype=wp.transform),
                    parent_indices,
                    skeletal_mesh.bind_transforms,
                    skeleton_instance.xform,
                ],
                outputs=[skinned_transforms],
            )

            for mesh_idx, skinned_mesh in enumerate(skeletal_mesh.skinned_meshes):
                if skinned_mesh.num_points == 0:
                    continue

                wp.launch(
                    skinning_kernel,
                    dim=skinned_mesh.num_points,
                    inputs=[
                        skinned_mesh.points,
                        skinned_mesh.joint_indices,
                        skinned_mesh.joint_weights,
                        int(skinned_mesh.num_influences),
                        wp.array(skinned_transforms[0], dtype=wp.transform),
                    ],
                    outputs=[skinned_points_buffers[mesh_idx]],
                )

                trimesh_meshes[mesh_idx].vertices = skinned_points_buffers[mesh_idx].numpy()

            png_bytes = scene.save_image(resolution=(int(width), int(height)), visible=False)
            if png_bytes is None:
                raise RuntimeError("trimesh failed to render frame to PNG bytes.")
            ffmpeg.stdin.write(png_bytes)

            if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
                elapsed = time.time() - start_time
                done = frame_idx + 1
                speed = done / max(elapsed, 1e-6)
                eta = (num_frames - done) / max(speed, 1e-6)
                print(
                    f"[INFO]: frame {done}/{num_frames} | "
                    f"{speed:.2f} fps render | ETA {eta:.1f}s"
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO]: Rendering interrupted by user.")
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        ret = ffmpeg.wait()
        if (ret != 0) and (not interrupted):
            raise RuntimeError(f"ffmpeg exited with code {ret}")

    if not interrupted:
        print(f"[INFO]: Done. Video saved to {output_mp4.resolve()}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Offline render SMPL mesh video from a BVH motion."
    )
    parser.add_argument("--bvh", required=True, type=str, help="Input BVH file path")
    parser.add_argument("--output", required=True, type=str, help="Output MP4 file path")
    parser.add_argument(
        "--retarget-source",
        type=str,
        default="soma",
        choices=["soma", "ours", "npz"],
        help="Source type for BVH loading and SMPL mesh selection",
    )
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=[0.0, 5.0, 3.0],
        metavar=("X", "Y", "Z"),
        help="Camera position in world coordinates",
    )
    parser.add_argument(
        "--look-at",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Look-at target in world coordinates",
    )
    parser.add_argument(
        "--camera-up",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Camera up direction",
    )
    parser.add_argument("--fov-y", type=float, default=45.0, help="Vertical field-of-view in degrees")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Warp runtime device, e.g. cuda or cpu"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    bvh_file = pathlib.Path(args.bvh)
    output = pathlib.Path(args.output)

    with wp.ScopedDevice(args.device):
        render_bvh_to_mp4(
            bvh_file=bvh_file,
            output_mp4=output,
            retarget_source=args.retarget_source,
            width=args.width,
            height=args.height,
            camera_pos=np.array(args.camera_pos, dtype=np.float64),
            look_at=np.array(args.look_at, dtype=np.float64),
            camera_up=np.array(args.camera_up, dtype=np.float64),
            fov_y_deg=float(args.fov_y),
            device=args.device,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO]: Interrupted by user.")
        sys.exit(130)
