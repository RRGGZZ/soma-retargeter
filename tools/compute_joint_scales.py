#!/usr/bin/env python3
"""
Estimate soma_to_g1_scaler_config.json joint_scales from the zero-pose BVH.

Pipeline
--------
1. Parse the BVH zero pose.
2. Convert the BVH from Y-up to Z-up with the same SpaceConverter used by the app.
3. Use the BVH loader's built-in cm -> m conversion.
4. Load the robot MJCF in its default pose and read global body positions.
5. Solve the best yaw-only alignment from human root-relative points to robot root-relative points.
6. Compute per-joint scale from the matched body map.

By default this script computes the same kind of root-relative scale that the
runtime scaler applies to non-root joints:

    robot_j ~= scaled_root + scale_j * Rz(theta) * (human_j - human_root)

So for j != root:

    scale_j = ||robot_j - robot_root|| / ||human_j - human_root||

The root joint ("Hips") uses a world-space distance ratio because its
root-relative vector is zero.
"""

from __future__ import annotations

import argparse
import math
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


DEFAULT_BVH = ROOT / "assets/motions/bvh/soma_zero_frame0.bvh"
DEFAULT_MJCF = ROOT / "assets/robot/unitree_g1/g1_mocap_29dof.xml"
DEFAULT_RETARGET_CFG = ROOT / "soma_retargeter/configs/unitree_g1/soma_to_g1_retargeter_config.json"
DEFAULT_SCALER_CFG = ROOT / "soma_retargeter/configs/unitree_g1/soma_to_g1_scaler_config.json"

# Extra mappings for joints that are present in joint_scales but not in ik_map.
# These are heuristic choices so the script can produce a full candidate config.
EXTRA_BODY_MAP = {
    "Neck1": "head_mocap",
    "LeftToe": "left_toe_link",
    "RightToe": "right_toe_link",
    "LeftToeBase": "left_toe_link",
    "RightToeBase": "right_toe_link",
}

# Config joint names do not always exactly match the BVH joint names.
# This table lets the script resolve config-space joints to BVH-space joints.
HUMAN_JOINT_ALIASES = {
    "LeftToe": ["LeftToe", "LeftToeEnd", "LeftToeBase"],
    "RightToe": ["RightToe", "RightToeEnd", "RightToeBase"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute candidate joint_scales from soma zero pose.")
    parser.add_argument("--bvh", type=pathlib.Path, default=DEFAULT_BVH, help="Zero-pose BVH path.")
    parser.add_argument("--mjcf", type=pathlib.Path, default=DEFAULT_MJCF, help="Robot MJCF/XML path.")
    parser.add_argument(
        "--retarget-config",
        type=pathlib.Path,
        default=DEFAULT_RETARGET_CFG,
        help="Retarget config path with ik_map.",
    )
    parser.add_argument(
        "--scaler-config",
        type=pathlib.Path,
        default=DEFAULT_SCALER_CFG,
        help="Scaler config path with joint_scales order.",
    )
    parser.add_argument(
        "--root-joint",
        default="Hips",
        help="Human root joint used by the scaler.",
    )
    parser.add_argument(
        "--verbose-points",
        action="store_true",
        help="Print aligned human and robot vectors for every matched joint.",
    )
    return parser.parse_args()


def load_human_global_positions(bvh_path: pathlib.Path) -> dict[str, np.ndarray]:
    importer = BVHImporter()
    skeleton, root_joint = importer.create_skeleton(str(bvh_path))
    animation = importer.create_animation(root_joint, skeleton)

    # BVHImporter already converts positions from cm to m.
    frame0_local = animation.get_local_transforms(0)
    converter = SpaceConverter(FacingDirectionType.MUJOCO)
    root_tx = converter.transform(wp.transform_identity())
    global_tx = compute_global_pose(skeleton, frame0_local, root_tx)

    positions = {}
    for idx, joint_name in enumerate(skeleton.joint_names):
        # Warp transform arrays come back from numpy() as plain rows:
        # [tx, ty, tz, qx, qy, qz, qw]
        positions[joint_name] = np.asarray(global_tx[idx][:3], dtype=np.float64)
    return positions


def resolve_human_joint_name(joint_name: str, human_positions: dict[str, np.ndarray]) -> str | None:
    if joint_name in human_positions:
        return joint_name
    for candidate in HUMAN_JOINT_ALIASES.get(joint_name, []):
        if candidate in human_positions:
            return candidate
    return None


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    xyz = q[:3]
    w = q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def load_robot_body_poses(mjcf_path: pathlib.Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(mjcf_path))
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()
    positions = {}
    rotations = {}
    for idx, label in enumerate(model.body_label):
        name = newton_utils.get_name_from_label(label)
        positions[name] = np.asarray(body_q[idx][:3], dtype=np.float64)
        rotations[name] = quat_normalize(np.asarray(body_q[idx][3:7], dtype=np.float64))
    return positions, rotations


def compute_robot_effector_positions(
    scaler_cfg: dict,
    joint_body_map: dict[str, str],
    robot_body_positions: dict[str, np.ndarray],
    robot_body_rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    joint_offsets = {
        joint_name: (
            np.asarray(entry[0], dtype=np.float64),
            quat_normalize(np.asarray(entry[1], dtype=np.float64)),
        )
        for joint_name, entry in scaler_cfg["joint_offsets"].items()
    }

    # Match runtime behavior in HumanToRobotScaler.__init__().
    if "LeftToe" in joint_offsets:
        joint_offsets["LeftToeBase"] = joint_offsets["LeftToe"]
    if "RightToe" in joint_offsets:
        joint_offsets["RightToeBase"] = joint_offsets["RightToe"]

    effector_positions = {}
    for joint_name, body_name in joint_body_map.items():
        if body_name not in robot_body_positions or body_name not in robot_body_rotations:
            continue
        if joint_name not in joint_offsets:
            continue

        body_p = robot_body_positions[body_name]
        body_q = robot_body_rotations[body_name]
        offset_t, offset_q = joint_offsets[joint_name]
        effector_q = quat_mul(body_q, offset_q)
        effector_positions[joint_name] = body_p + quat_rotate(effector_q, offset_t)

    return effector_positions


def build_joint_body_map(retarget_cfg: dict, scaler_cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    joint_body_map = {}
    body_source = {}

    for joint_name, entry in retarget_cfg["ik_map"].items():
        body = entry.get("t_body") or entry.get("r_body")
        if body:
            joint_body_map[joint_name] = body
            body_source[joint_name] = "ik_map"

    for joint_name in scaler_cfg["joint_scales"].keys():
        if joint_name not in joint_body_map and joint_name in EXTRA_BODY_MAP:
            joint_body_map[joint_name] = EXTRA_BODY_MAP[joint_name]
            body_source[joint_name] = "heuristic"

    return joint_body_map, body_source


def rotation_about_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def solve_best_yaw(
    human_positions: dict[str, np.ndarray],
    robot_effector_positions: dict[str, np.ndarray],
    joint_body_map: dict[str, str],
    root_joint: str,
) -> float:
    human_root = human_positions[root_joint]
    robot_root = robot_effector_positions[root_joint]

    cross_sum = 0.0
    dot_sum = 0.0
    used = 0

    for joint_name, body_name in joint_body_map.items():
        if joint_name == root_joint:
            continue
        human_joint_name = resolve_human_joint_name(joint_name, human_positions)
        if human_joint_name is None or joint_name not in robot_effector_positions:
            continue

        h = human_positions[human_joint_name] - human_root
        r = robot_effector_positions[joint_name] - robot_root
        h_xy = h[:2]
        r_xy = r[:2]
        if np.linalg.norm(h_xy) < 1e-8 or np.linalg.norm(r_xy) < 1e-8:
            continue

        dot_sum += float(np.dot(h_xy, r_xy))
        cross_sum += float(h_xy[0] * r_xy[1] - h_xy[1] * r_xy[0])
        used += 1

    if used == 0:
        raise RuntimeError("No matched joints were usable for yaw fitting.")

    return math.atan2(cross_sum, dot_sum)


def angle_deg_between(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    dot = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def compute_scales(
    human_positions: dict[str, np.ndarray],
    robot_effector_positions: dict[str, np.ndarray],
    joint_body_map: dict[str, str],
    root_joint: str,
    theta: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    rot_z = rotation_about_z(theta)
    scales = {}
    direction_errors = {}
    length_errors = {}

    human_root = human_positions[root_joint]
    robot_root = robot_effector_positions[root_joint]

    root_h_norm = np.linalg.norm(human_root)
    root_r_norm = np.linalg.norm(robot_root)
    if root_h_norm < 1e-8:
        raise RuntimeError("Human root world position is near zero; cannot compute root scale.")
    scales[root_joint] = root_r_norm / root_h_norm
    direction_errors[root_joint] = angle_deg_between(rot_z @ human_root, robot_root)
    length_errors[root_joint] = abs(root_r_norm - scales[root_joint] * root_h_norm)

    for joint_name, body_name in joint_body_map.items():
        if joint_name == root_joint:
            continue
        human_joint_name = resolve_human_joint_name(joint_name, human_positions)
        if human_joint_name is None or joint_name not in robot_effector_positions:
            continue

        human_vec = rot_z @ (human_positions[human_joint_name] - human_root)
        robot_vec = robot_effector_positions[joint_name] - robot_root
        human_len = np.linalg.norm(human_vec)
        robot_len = np.linalg.norm(robot_vec)
        if human_len < 1e-8:
            continue

        scales[joint_name] = robot_len / human_len
        direction_errors[joint_name] = angle_deg_between(human_vec, robot_vec)
        length_errors[joint_name] = abs(robot_len - scales[joint_name] * human_len)

    return scales, direction_errors, length_errors


def print_summary(
    scaler_cfg: dict,
    joint_body_map: dict[str, str],
    body_source: dict[str, str],
    scales: dict[str, float],
    direction_errors: dict[str, float],
    length_errors: dict[str, float],
    yaw_rad: float,
):
    cfg_scales = scaler_cfg["joint_scales"]

    print("=" * 96)
    print("BEST YAW ALIGNMENT")
    print("=" * 96)
    print(f"  yaw_z = {math.degrees(yaw_rad):+.4f} deg")
    print()

    print("=" * 96)
    print("JOINT SCALE COMPARISON")
    print("=" * 96)
    header = (
        f"{'Joint':<14} {'Body':<26} {'Src':<10} "
        f"{'Estimated':>10} {'Config':>10} {'Delta':>10} {'DirErr(deg)':>12}"
    )
    print(header)
    print("-" * len(header))

    for joint_name in cfg_scales.keys():
        body_name = joint_body_map.get(joint_name, "-")
        source = body_source.get(joint_name, "-")
        estimated = scales.get(joint_name)
        config_value = cfg_scales[joint_name]
        if estimated is None:
            print(
                f"{joint_name:<14} {body_name:<26} {source:<10} "
                f"{'N/A':>10} {config_value:>10.4f} {'N/A':>10} {'N/A':>12}"
            )
            continue

        delta = estimated - config_value
        dir_err = direction_errors.get(joint_name, float("nan"))
        print(
            f"{joint_name:<14} {body_name:<26} {source:<10} "
            f"{estimated:>10.4f} {config_value:>10.4f} {delta:>10.4f} {dir_err:>12.4f}"
        )

    print()
    print("=" * 96)
    print("READY-TO-PASTE JSON SNIPPET")
    print("=" * 96)
    print('  "joint_scales": {')
    items = list(cfg_scales.keys())
    for idx, joint_name in enumerate(items):
        value = scales.get(joint_name, cfg_scales[joint_name])
        comma = "," if idx < len(items) - 1 else ""
        print(f'    "{joint_name}": {value:.4f}{comma}')
    print("  }")
    print()

    mean_dir_err = np.mean([v for k, v in direction_errors.items() if k in scales and k != "Hips"])
    max_dir_err = np.max([v for k, v in direction_errors.items() if k in scales and k != "Hips"])
    mean_len_err = np.mean([v for k, v in length_errors.items() if k in scales])
    print("=" * 96)
    print("FIT SUMMARY")
    print("=" * 96)
    print(f"  matched_joints: {len(scales)}")
    print(f"  mean_direction_error_deg: {mean_dir_err:.4f}")
    print(f"  max_direction_error_deg : {max_dir_err:.4f}")
    print(f"  mean_length_residual_m  : {mean_len_err:.6f}")


def print_verbose_points(
    human_positions: dict[str, np.ndarray],
    robot_effector_positions: dict[str, np.ndarray],
    joint_body_map: dict[str, str],
    scales: dict[str, float],
    root_joint: str,
    theta: float,
):
    rot_z = rotation_about_z(theta)
    human_root = human_positions[root_joint]
    robot_root = robot_effector_positions[root_joint]

    print()
    print("=" * 96)
    print("ALIGNED ROOT-RELATIVE POINTS")
    print("=" * 96)
    for joint_name, body_name in joint_body_map.items():
        if joint_name not in scales:
            continue
        human_joint_name = resolve_human_joint_name(joint_name, human_positions)
        if human_joint_name is None:
            continue
        if joint_name == root_joint:
            human_vec = rot_z @ human_positions[human_joint_name]
            robot_vec = robot_effector_positions[joint_name]
        else:
            human_vec = rot_z @ (human_positions[human_joint_name] - human_root)
            robot_vec = robot_effector_positions[joint_name] - robot_root
        print(
            f"{joint_name:<14} "
            f"h={np.array2string(human_vec, precision=5, suppress_small=True)} "
            f"r={np.array2string(robot_vec, precision=5, suppress_small=True)} "
            f"s={scales[joint_name]:.4f}"
        )


def main():
    args = parse_args()

    wp.init()

    if not args.bvh.exists():
        raise FileNotFoundError(f"BVH file not found: {args.bvh}")
    if not args.mjcf.exists():
        raise FileNotFoundError(f"MJCF file not found: {args.mjcf}")

    retarget_cfg = load_json(args.retarget_config)
    scaler_cfg = load_json(args.scaler_config)

    human_positions = load_human_global_positions(args.bvh)
    robot_body_positions, robot_body_rotations = load_robot_body_poses(args.mjcf)
    joint_body_map, body_source = build_joint_body_map(retarget_cfg, scaler_cfg)
    robot_effector_positions = compute_robot_effector_positions(
        scaler_cfg,
        joint_body_map,
        robot_body_positions,
        robot_body_rotations,
    )

    missing_human = [j for j in joint_body_map if resolve_human_joint_name(j, human_positions) is None]
    missing_robot = [j for j in joint_body_map if j not in robot_effector_positions]
    if missing_human:
        raise RuntimeError(f"Missing human joints in BVH skeleton: {missing_human}")
    if missing_robot:
        raise RuntimeError(f"Missing robot effectors from MJCF/config: {sorted(set(missing_robot))}")
    if args.root_joint not in joint_body_map:
        raise RuntimeError(f"Root joint '{args.root_joint}' has no mapped robot body.")

    yaw_rad = solve_best_yaw(human_positions, robot_effector_positions, joint_body_map, args.root_joint)
    scales, direction_errors, length_errors = compute_scales(
        human_positions,
        robot_effector_positions,
        joint_body_map,
        args.root_joint,
        yaw_rad,
    )

    print_summary(
        scaler_cfg,
        joint_body_map,
        body_source,
        scales,
        direction_errors,
        length_errors,
        yaw_rad,
    )

    if args.verbose_points:
        print_verbose_points(
            human_positions,
            robot_effector_positions,
            joint_body_map,
            scales,
            args.root_joint,
            yaw_rad,
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
