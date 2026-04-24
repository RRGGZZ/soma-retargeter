"""
Compute joint_offsets for the adam_pro scaler config relative to G1.

Strategy
--------
The G1 joint_offsets are empirically validated ground truth.  Instead of
deriving absolute offsets from scratch (which requires knowing the exact
reference frame used during manual tuning), we compute only the *relative*
rotation between the two robots for every mapped link:

    q_rel  = conj(q_g1_link_rest) * q_adam_link_rest
    q_offset_adam = q_offset_g1 * q_rel

If a link has the same rest-pose orientation in both robots (q_rel ≈ identity),
the G1 offset is reused unchanged.  Only links with a measurable difference
need a new value.

Output
------
  • Per-joint diff table (G1 rest vs adam_pro rest).
  • Forearm / hand summary.
  • Ready-to-paste JSON snippet for soma_to_adam_scaler_config.json.

Usage:
    python app/compute_scaler_offsets.py
"""

import os
import sys
import pathlib
import numpy as np

sys.path.append(os.getcwd())

import warp as wp
import newton

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.newton_utils as newton_utils

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT            = pathlib.Path(__file__).resolve().parent.parent
_G1_XML          = _ROOT / "assets/robot/unitree_g1/g1_mocap_29dof.xml"
_ADAM_XML        = _ROOT / "assets/robot/adam_pro/adam_pro.xml"
_G1_CFG          = io_utils.get_config_file("unitree_g1/soma_to_g1_retargeter_config.json")
_ADAM_CFG        = io_utils.get_config_file("adam_pro/soma_to_adam_retargeter_config.json")
_G1_SCALER_CFG   = io_utils.get_config_file("unitree_g1/soma_to_g1_scaler_config.json")
_ADAM_SCALER_CFG = io_utils.get_config_file("adam_pro/soma_to_adam_scaler_config.json")

# SOMA joints we care about (subset that appear in ik_map)
_IK_JOINT_ORDER = [
    "Hips", "Chest",
    "LeftLeg", "LeftShin", "LeftFoot",
    "RightLeg", "RightShin", "RightFoot",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
]

# ── quaternion helpers (x, y, z, w convention) ─────────────────────────────────

def quat_conj(q):
    """Conjugate (= inverse for unit quaternion). q = [x,y,z,w]"""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul(q1, q2):
    """Hamilton product q1 * q2. Both in [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_angle_deg(q1, q2):
    """Angular distance in degrees between two unit quaternions."""
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def fmt_quat(q, precision=4):
    return f"[{q[0]:.{precision}f}, {q[1]:.{precision}f}, {q[2]:.{precision}f}, {q[3]:.{precision}f}]"


# ── Robot rest-pose link global rotations via Newton FK ───────────────────────

def get_robot_global_quats(mjcf_path: str) -> dict[str, np.ndarray]:
    """
    Run FK at Newton's default joint_q (rest pose) and return
    {body_name: q_global (x,y,z,w)} for every body.
    """
    print(f"[INFO] Loading robot: {mjcf_path}")
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(mjcf_path))
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()          # (N, 7): [x,y,z, qx,qy,qz,qw]
    result = {}
    for i, label in enumerate(model.body_label):
        name = newton_utils.get_name_from_label(label)
        result[name] = quat_normalize(body_q[i][3:7])
    return result


# ── Relative offset computation ─────────────────────────────────────────────────

def compute_adam_offsets_from_g1(
    g1_scaler_cfg:   dict,
    g1_ik_map:       dict,
    adam_ik_map:     dict,
    g1_quats:        dict[str, np.ndarray],
    adam_quats:      dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    For every joint that appears in both ik_maps, compute adam_pro's q_offset as:

        q_rel          = conj(q_g1_link) * q_adam_link
        q_offset_adam  = q_offset_g1 * q_rel

    Joints not in the ik_map are left unchanged (kept from the existing config).
    Returns {soma_joint: q_offset_adam}.
    """
    g1_cfg_offsets = g1_scaler_cfg["joint_offsets"]
    result = {}

    for soma_joint in _IK_JOINT_ORDER:
        if soma_joint not in g1_ik_map or soma_joint not in adam_ik_map:
            continue

        g1_body   = g1_ik_map[soma_joint]["r_body"]
        adam_body = adam_ik_map[soma_joint]["r_body"]

        q_g1   = g1_quats.get(g1_body)
        q_adam = adam_quats.get(adam_body)
        if q_g1 is None:
            print(f"  [WARN] G1 body '{g1_body}' not found")
            continue
        if q_adam is None:
            print(f"  [WARN] adam body '{adam_body}' not found")
            continue
        if soma_joint not in g1_cfg_offsets:
            print(f"  [WARN] '{soma_joint}' not in G1 scaler config")
            continue

        q_offset_g1 = quat_normalize(np.array(g1_cfg_offsets[soma_joint][1], dtype=np.float64))
        q_rel       = quat_normalize(quat_mul(quat_conj(q_g1), q_adam))
        result[soma_joint] = quat_normalize(quat_mul(q_offset_g1, q_rel))

    return result


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_link_diff_table(g1_ik_map, adam_ik_map, g1_quats, adam_quats):
    """Show G1 vs adam_pro rest-pose rotations and their angular diff."""
    print("\n" + "="*72)
    print("REST-POSE LINK ORIENTATION DIFF  (G1 vs adam_pro)")
    print("="*72)
    print(f"  {'SOMA joint':<16} {'G1 body':<28} {'adam body':<28}  Δ°")
    print(f"  {'-'*16} {'-'*28} {'-'*28}  ---")
    for j in _IK_JOINT_ORDER:
        if j not in g1_ik_map or j not in adam_ik_map:
            continue
        g1b   = g1_ik_map[j]["r_body"]
        adab  = adam_ik_map[j]["r_body"]
        qg1   = g1_quats.get(g1b)
        qadam = adam_quats.get(adab)
        if qg1 is None or qadam is None:
            continue
        diff = quat_angle_deg(qg1, qadam)
        flag = "  ← DIFF" if diff > 3.0 else ""
        print(f"  {j:<16} {g1b:<28} {adab:<28}  {diff:5.1f}°{flag}")


def print_offset_comparison(g1_cfg_offsets, adam_offsets):
    """Side-by-side: G1 config offset vs new adam_pro offset."""
    print("\n" + "="*72)
    print("OFFSET COMPARISON  (G1 config → adam_pro new)")
    print("="*72)
    arm_joints = ["LeftArm", "LeftForeArm", "LeftHand",
                  "RightArm", "RightForeArm", "RightHand"]
    print(f"  {'Joint':<16} {'G1 config':^44} {'adam_pro new':^44}  Δ°")
    print(f"  {'-'*16} {'-'*44} {'-'*44}  ---")
    for j in arm_joints:
        if j not in g1_cfg_offsets or j not in adam_offsets:
            continue
        q_g1   = quat_normalize(np.array(g1_cfg_offsets[j][1], dtype=np.float64))
        q_adam = adam_offsets[j]
        diff   = quat_angle_deg(q_g1, q_adam)
        flag   = "  ← changed" if diff > 3.0 else ""
        print(f"  {j:<16} {fmt_quat(q_g1):44}  {fmt_quat(q_adam):44}  {diff:5.1f}°{flag}")


def generate_json_snippet(adam_offsets):
    """Print a ready-to-paste joint_offsets block for soma_to_adam_scaler_config.json."""
    existing  = io_utils.load_json(_ADAM_SCALER_CFG)["joint_offsets"]
    g1_cfg    = io_utils.load_json(_G1_SCALER_CFG)["joint_offsets"]

    print(f"\n{'='*72}")
    print("ADAM_PRO soma_to_adam_scaler_config.json  →  joint_offsets")
    print(f"{'='*72}")
    print('    "joint_offsets": {')
    keys = list(existing.keys())
    for i, joint in enumerate(keys):
        t = existing[joint][0]
        if joint in adam_offsets:
            q_out = [round(float(v), 4) for v in adam_offsets[joint]]
            note  = ""
        else:
            # Not in ik_map → keep G1 config value (best known reference)
            q_out = g1_cfg[joint][1] if joint in g1_cfg else existing[joint][1]
            note  = "  // (kept from G1 config)"
        comma = "," if i < len(keys) - 1 else ""
        print(f'        "{joint}": [{t}, {q_out}]{comma}{note}')
    print('    }')


# ── Main ────────────────────────────────────────────────────────────────────────

def validate_formula(g1_scaler_cfg, g1_ik_map, g1_quats):
    """
    Self-consistency check: treat G1 as both source and target.
    q_rel = conj(q_g1) * q_g1 = identity  →  q_offset_out = q_offset_g1 * identity = q_offset_g1.
    Every joint should show 0.0° diff. Any non-zero diff means a bug in the formula.
    """
    g1_self_offsets = compute_adam_offsets_from_g1(
        g1_scaler_cfg, g1_ik_map, g1_ik_map, g1_quats, g1_quats)

    g1_cfg_offsets = g1_scaler_cfg["joint_offsets"]
    all_ok = True
    print("\n" + "="*72)
    print("FORMULA VALIDATION  (G1 vs G1 self-check, all diffs must be 0.0°)")
    print("="*72)
    print(f"  {'Joint':<16} {'Config':^44} {'Recomputed':^44}  Δ°")
    print(f"  {'-'*16} {'-'*44} {'-'*44}  ---")
    for j in _IK_JOINT_ORDER:
        if j not in g1_cfg_offsets or j not in g1_self_offsets:
            continue
        q_cfg  = quat_normalize(np.array(g1_cfg_offsets[j][1], dtype=np.float64))
        q_self = g1_self_offsets[j]
        diff   = quat_angle_deg(q_cfg, q_self)
        flag   = "  ← BUG" if diff > 0.1 else ""
        if diff > 0.1:
            all_ok = False
        print(f"  {j:<16} {fmt_quat(q_cfg):44}  {fmt_quat(q_self):44}  {diff:5.1f}°{flag}")
    print()
    if all_ok:
        print("  ✓ All diffs < 0.1° — formula is correct.")
    else:
        print("  ✗ Diffs detected — formula has a bug!")


def main():
    wp.init()

    g1_cfg   = io_utils.load_json(_G1_CFG)
    adam_cfg = io_utils.load_json(_ADAM_CFG)
    g1_scaler_cfg = io_utils.load_json(_G1_SCALER_CFG)

    g1_ik_map   = g1_cfg["ik_map"]
    adam_ik_map = adam_cfg["ik_map"]

    g1_quats   = get_robot_global_quats(str(_G1_XML))
    adam_quats = get_robot_global_quats(str(_ADAM_XML))

    # 0. Formula self-check: G1 vs G1 must give back the exact config values
    validate_formula(g1_scaler_cfg, g1_ik_map, g1_quats)

    # 1. Show raw link orientation diff between the two robots
    print_link_diff_table(g1_ik_map, adam_ik_map, g1_quats, adam_quats)

    # 2. Compute adam offsets via relative rotation from G1 config
    adam_offsets = compute_adam_offsets_from_g1(
        g1_scaler_cfg, g1_ik_map, adam_ik_map, g1_quats, adam_quats)

    # 3. Compare arm offsets G1 config vs new adam values
    print_offset_comparison(g1_scaler_cfg["joint_offsets"], adam_offsets)

    # 4. Generate final JSON snippet
    generate_json_snippet(adam_offsets)


if __name__ == "__main__":
    main()
