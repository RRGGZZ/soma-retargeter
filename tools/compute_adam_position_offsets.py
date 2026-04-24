#!/usr/bin/env python3
"""
Solve adam_pro joint position offsets from the zero-pose BVH.

Method
------
1. Load `adam_set_zero.bvh`.
2. Convert BVH positions from cm -> m through BVHImporter.
3. Rotate from Y-up into the repo's standard Z-up space.
4. Compute world-space human joint transforms with FK.
5. Apply the configured per-joint scales to obtain the scaled base effector
   positions used by HumanToRobotScaler.
6. Solve translation offsets so that each zero-pose effector lands on the
   robot body's default world position from `ik_map.t_body`.

For each mapped joint j, the runtime uses:

    t_j = scaled_base_j + rotate(q_human_j * q_offset_j, offset_t_j)

Therefore:

    offset_t_j = rotate_inv(q_human_j * q_offset_j, target_robot_pos_j - scaled_base_j)

This script solves `offset_t_j` while preserving the existing quaternion
offsets from the scaler config.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import warp as wp
import newton

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soma_retargeter.assets.bvh import BVHImporter
from soma_retargeter.utils.io_utils import load_json
from soma_retargeter.utils.pose_utils import compute_global_pose
from soma_retargeter.utils.space_conversion_utils import (
    FacingDirectionType,
    SpaceConverter,
)
import soma_retargeter.utils.newton_utils as newton_utils


DEFAULT_BVH = ROOT / "assets/motions/bvh/adam_set_zero.bvh"
DEFAULT_MJCF = ROOT / "assets/robot/adam_pro/adam_pro.xml"
DEFAULT_RETARGET_CFG = ROOT / "soma_retargeter/configs/adam_pro/soma_to_adam_retargeter_config.json"
DEFAULT_SCALER_CFG = ROOT / "soma_retargeter/configs/adam_pro/soma_to_adam_scaler_config.json"

EXTRA_BODY_MAP = {
    "Neck1": "neckYaw_link",
    "LeftToe": "toeLeft",
    "RightToe": "toeRight",
    "LeftToeBase": "toeLeft",
    "RightToeBase": "toeRight",
}

HUMAN_JOINT_ALIASES = {
    "LeftToe": ["LeftToe", "LeftToeEnd", "LeftToeBase"],
    "RightToe": ["RightToe", "RightToeEnd", "RightToeBase"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute adam_pro translation offsets from adam_set_zero.bvh."
    )
    parser.add_argument("--bvh", type=pathlib.Path, default=DEFAULT_BVH, help="Zero-pose BVH path.")
    parser.add_argument("--mjcf", type=pathlib.Path, default=DEFAULT_MJCF, help="adam_pro MJCF path.")
    parser.add_argument(
        "--retarget-config",
        type=pathlib.Path,
        default=DEFAULT_RETARGET_CFG,
        help="Retarget config path.",
    )
    parser.add_argument(
        "--scaler-config",
        type=pathlib.Path,
        default=DEFAULT_SCALER_CFG,
        help="Scaler config path.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Printed decimal precision.",
    )
    parser.add_argument(
        "--extra-yaw-deg",
        type=float,
        default=0.0,
        help="Optional extra yaw applied after the standard MUJOCO Y-up->Z-up conversion.",
    )
    return parser.parse_args()


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else q


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    xyz = q[:3]
    w = q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate(quat_conj(quat_normalize(q)), v)


def resolve_human_joint_name(joint_name: str, human_pose: dict[str, dict[str, np.ndarray]]) -> str | None:
    if joint_name in human_pose:
        return joint_name
    for candidate in HUMAN_JOINT_ALIASES.get(joint_name, []):
        if candidate in human_pose:
            return candidate
    return None


def quat_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(angle_deg) * 0.5
    s = np.sin(half)
    c = np.cos(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)


def make_root_tx(extra_yaw_deg: float) -> wp.transform:
    converter = SpaceConverter(FacingDirectionType.MUJOCO)
    base_q = np.array(
        [
            float(converter.converter[0]),
            float(converter.converter[1]),
            float(converter.converter[2]),
            float(converter.converter[3]),
        ],
        dtype=np.float64,
    )
    if abs(extra_yaw_deg) > 1e-8:
        yaw_q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), extra_yaw_deg)
        base_q = quat_normalize(quat_mul(yaw_q, base_q))
    return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*base_q.astype(np.float32)))


def load_human_world_pose(bvh_path: pathlib.Path, extra_yaw_deg: float) -> dict[str, dict[str, np.ndarray]]:
    importer = BVHImporter()
    skeleton, root_joint = importer.create_skeleton(str(bvh_path))
    animation = importer.create_animation(root_joint, skeleton)
    frame0_local = animation.get_local_transforms(0)

    root_tx = make_root_tx(extra_yaw_deg)
    global_tx = compute_global_pose(skeleton, frame0_local, root_tx)

    pose = {}
    for idx, joint_name in enumerate(skeleton.joint_names):
        pose[joint_name] = {
            "p": np.asarray(global_tx[idx][:3], dtype=np.float64),
            "q": quat_normalize(np.asarray(global_tx[idx][3:7], dtype=np.float64)),
        }
    return pose


def load_robot_body_positions(mjcf_path: pathlib.Path) -> dict[str, np.ndarray]:
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(mjcf_path))
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()
    positions = {}
    for idx, label in enumerate(model.body_label):
        name = newton_utils.get_name_from_label(label)
        positions[name] = np.asarray(body_q[idx][:3], dtype=np.float64)
    return positions


def build_joint_body_map(retarget_cfg: dict, scaler_cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    joint_body_map = {}
    source = {}

    for joint_name, entry in retarget_cfg["ik_map"].items():
        body = entry.get("t_body")
        if body:
            joint_body_map[joint_name] = body
            source[joint_name] = "ik_map.t_body"

    for joint_name in scaler_cfg["joint_offsets"].keys():
        if joint_name not in joint_body_map and joint_name in EXTRA_BODY_MAP:
            joint_body_map[joint_name] = EXTRA_BODY_MAP[joint_name]
            source[joint_name] = "heuristic"

    return joint_body_map, source


def compute_scaled_base_positions(
    human_pose: dict[str, dict[str, np.ndarray]],
    scaler_cfg: dict,
) -> dict[str, np.ndarray]:
    joint_scales = scaler_cfg["joint_scales"]
    root_name = scaler_cfg["human_root_name"]
    root_human_name = resolve_human_joint_name(root_name, human_pose)
    if root_human_name is None:
        raise RuntimeError(f"Root joint '{root_name}' not found in BVH pose.")

    root_p = human_pose[root_human_name]["p"]
    scaled_root = root_p * float(joint_scales[root_name])

    base_positions = {}
    for joint_name, scale in joint_scales.items():
        human_joint_name = resolve_human_joint_name(joint_name, human_pose)
        if human_joint_name is None:
            continue
        p = human_pose[human_joint_name]["p"]
        base_positions[joint_name] = scaled_root + (p - root_p) * float(scale)
    return base_positions


def solve_translation_offsets(
    human_pose: dict[str, dict[str, np.ndarray]],
    robot_targets: dict[str, np.ndarray],
    scaler_cfg: dict,
    base_positions: dict[str, np.ndarray],
) -> tuple[dict[str, list], dict[str, float]]:
    offsets_out = {}
    residuals = {}

    joint_offsets_cfg = scaler_cfg["joint_offsets"]
    for joint_name, entry in joint_offsets_cfg.items():
        current_t = np.asarray(entry[0], dtype=np.float64)
        current_q = quat_normalize(np.asarray(entry[1], dtype=np.float64))

        human_joint_name = resolve_human_joint_name(joint_name, human_pose)
        if joint_name not in robot_targets or joint_name not in base_positions or human_joint_name is None:
            offsets_out[joint_name] = [current_t.tolist(), current_q.tolist()]
            continue

        human_q = human_pose[human_joint_name]["q"]
        eff_q = quat_normalize(quat_mul(human_q, current_q))
        target_world = robot_targets[joint_name]
        base_world = base_positions[joint_name]

        solved_t = quat_rotate_inv(eff_q, target_world - base_world)
        reconstructed = base_world + quat_rotate(eff_q, solved_t)
        residuals[joint_name] = float(np.linalg.norm(reconstructed - target_world))

        offsets_out[joint_name] = [solved_t.tolist(), current_q.tolist()]

    # Match runtime toe-base override.
    if "LeftToe" in offsets_out:
        offsets_out["LeftToeBase"] = offsets_out["LeftToe"]
    if "RightToe" in offsets_out:
        offsets_out["RightToeBase"] = offsets_out["RightToe"]

    return offsets_out, residuals


def print_report(
    scaler_cfg: dict,
    joint_body_map: dict[str, str],
    body_sources: dict[str, str],
    robot_targets: dict[str, np.ndarray],
    base_positions: dict[str, np.ndarray],
    offsets_out: dict[str, list],
    residuals: dict[str, float],
    precision: int,
):
    fmt = f".{precision}f"
    print("=" * 120)
    print("ADAM POSITION OFFSET SOLVE")
    print("=" * 120)
    header = (
        f"{'Joint':<14} {'TargetBody':<18} {'Src':<14} "
        f"{'BasePos':<34} {'Solved offset_t':<34} {'Residual':>10}"
    )
    print(header)
    print("-" * len(header))

    def vec_text(v):
        return "[" + ", ".join(format(float(x), fmt) for x in v) + "]"

    for joint_name in scaler_cfg["joint_offsets"].keys():
        body = joint_body_map.get(joint_name, "-")
        src = body_sources.get(joint_name, "-")
        base = base_positions.get(joint_name)
        solved_t = offsets_out[joint_name][0]
        residual = residuals.get(joint_name, float("nan"))
        base_text = vec_text(base) if base is not None else "N/A"
        print(
            f"{joint_name:<14} {body:<18} {src:<14} "
            f"{base_text:<34} {vec_text(solved_t):<34} "
            f"{format(residual, fmt) if residual == residual else 'N/A':>10}"
        )

    print()
    print("=" * 120)
    print("READY-TO-PASTE joint_offsets")
    print("=" * 120)
    print('    "joint_offsets": {')
    keys = list(scaler_cfg["joint_offsets"].keys())
    for idx, joint_name in enumerate(keys):
        t_out, q_out = offsets_out[joint_name]
        t_out = [round(float(v), precision) for v in t_out]
        q_out = [round(float(v), precision) for v in q_out]
        comma = "," if idx < len(keys) - 1 else ""
        print(f'        "{joint_name}": [{t_out}, {q_out}]{comma}')
    print("    }")


def main():
    args = parse_args()
    wp.init()

    if not args.bvh.exists():
        raise FileNotFoundError(f"BVH file not found: {args.bvh}")
    if not args.mjcf.exists():
        raise FileNotFoundError(f"MJCF file not found: {args.mjcf}")

    retarget_cfg = load_json(args.retarget_config)
    scaler_cfg = load_json(args.scaler_config)

    human_pose = load_human_world_pose(args.bvh, args.extra_yaw_deg)
    robot_body_positions = load_robot_body_positions(args.mjcf)
    joint_body_map, body_sources = build_joint_body_map(retarget_cfg, scaler_cfg)

    robot_targets = {}
    for joint_name, body_name in joint_body_map.items():
        if body_name in robot_body_positions:
            robot_targets[joint_name] = robot_body_positions[body_name]

    base_positions = compute_scaled_base_positions(human_pose, scaler_cfg)
    offsets_out, residuals = solve_translation_offsets(
        human_pose,
        robot_targets,
        scaler_cfg,
        base_positions,
    )

    print_report(
        scaler_cfg,
        joint_body_map,
        body_sources,
        robot_targets,
        base_positions,
        offsets_out,
        residuals,
        args.precision,
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
