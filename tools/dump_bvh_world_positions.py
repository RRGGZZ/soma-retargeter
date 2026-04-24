#!/usr/bin/env python3
"""
Dump BVH joint world positions after converting from Y-up to Z-up.

This script follows the same path as the main retargeter:
1. Parse BVH.
2. Convert positions from cm to m via BVHImporter.
3. Apply the MUJOCO facing-direction conversion (Y-up -> Z-up).
4. Run forward kinematics to obtain world-space joint transforms.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import warp as wp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soma_retargeter.assets.bvh import BVHImporter
from soma_retargeter.utils.pose_utils import compute_global_pose
from soma_retargeter.utils.space_conversion_utils import (
    FacingDirectionType,
    SpaceConverter,
)


DEFAULT_BVH = ROOT / "assets/motions/bvh/soma_zero_frame0.bvh"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print BVH joint world positions after Y-up -> Z-up conversion."
    )
    parser.add_argument(
        "--bvh",
        type=pathlib.Path,
        default=DEFAULT_BVH,
        help="Path to the BVH file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to inspect.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for printed numbers.",
    )
    return parser.parse_args()


def load_world_positions_zup(bvh_path: pathlib.Path, frame_idx: int):
    importer = BVHImporter()
    skeleton, root_joint = importer.create_skeleton(str(bvh_path))
    animation = importer.create_animation(root_joint, skeleton)

    if frame_idx < 0 or frame_idx >= animation.num_frames:
        raise ValueError(
            f"Frame index [{frame_idx}] out of range [0, {animation.num_frames})."
        )

    local_transforms = animation.get_local_transforms(frame_idx)
    root_tx = SpaceConverter(FacingDirectionType.MUJOCO).transform(
        wp.transform_identity()
    )
    global_tx = compute_global_pose(skeleton, local_transforms, root_tx)

    rows = []
    for idx, joint_name in enumerate(skeleton.joint_names):
        parent_idx = skeleton.parent_indices[idx]
        parent_name = "" if parent_idx == -1 else skeleton.joint_names[parent_idx]
        pos = np.asarray(global_tx[idx][:3], dtype=np.float64)
        quat = np.asarray(global_tx[idx][3:7], dtype=np.float64)
        rows.append(
            {
                "index": idx,
                "joint": joint_name,
                "parent": parent_name,
                "world_pos_zup_m": pos.tolist(),
                "world_quat_xyzw": quat.tolist(),
            }
        )

    return rows, animation.num_frames


def print_table(rows: list[dict], precision: int):
    fmt = f".{precision}f"
    header = (
        f"{'Idx':>4} {'Joint':<24} {'Parent':<24} "
        f"{'X':>12} {'Y':>12} {'Z':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        x, y, z = row["world_pos_zup_m"]
        print(
            f"{row['index']:>4} "
            f"{row['joint']:<24} "
            f"{row['parent']:<24} "
            f"{format(x, fmt):>12} "
            f"{format(y, fmt):>12} "
            f"{format(z, fmt):>12}"
        )


def main():
    args = parse_args()
    wp.init()

    if not args.bvh.exists():
        raise FileNotFoundError(f"BVH file not found: {args.bvh}")

    rows, num_frames = load_world_positions_zup(args.bvh, args.frame)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "bvh": str(args.bvh),
                    "frame": args.frame,
                    "num_frames": num_frames,
                    "space": "z-up",
                    "units": "meters",
                    "joints": rows,
                },
                indent=2,
            )
        )
    else:
        print(f"# BVH: {args.bvh}")
        print(f"# frame: {args.frame} / {num_frames - 1}")
        print("# space: z-up")
        print("# units: meters")
        print_table(rows, args.precision)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
