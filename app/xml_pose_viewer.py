#!/usr/bin/env python3
import argparse
import time

import mujoco
import mujoco.viewer


def _joint_nq(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    # slide / hinge
    return 1


def _build_actuator_qpos_map(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """
    Build actuator->qpos mapping for 1-DoF joints only.
    Returns a list of (actuator_id, qpos_address).
    """
    mapping: list[tuple[int, int]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue

        joint_type = int(model.jnt_type[joint_id])
        if _joint_nq(joint_type) != 1:
            # Skip multi-DoF joints (ball/free). Typical robot motors are 1-DoF.
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        mapping.append((actuator_id, qpos_adr))
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Load MJCF/XML and interactively adjust pose.")
    parser.add_argument("xml", type=str, help="Path to robot XML (MJCF)")
    parser.add_argument("--fps", type=float, default=60.0, help="Viewer refresh rate")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    actuator_qpos_map = _build_actuator_qpos_map(model)

    # Disable gravity to avoid falling/jitter while posing.
    model.opt.gravity[:] = 0.0

    # Keep the initial XML pose by default.
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"[INFO] Loaded: {args.xml}")
    print(f"[INFO] nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(
        f"[INFO] 1-DoF actuator->joint mappings: {len(actuator_qpos_map)}/{model.nu}. "
        "Drag ctrl sliders in the right panel to pose the robot."
    )

    dt = 1.0 / max(1.0, args.fps)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()

            # Kinematic posing: ctrl directly drives joint qpos.
            for actuator_id, qpos_adr in actuator_qpos_map:
                data.qpos[qpos_adr] = data.ctrl[actuator_id]

            # Prevent residual dynamics from introducing oscillations.
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            mujoco.mj_forward(model, data)

            viewer.sync()
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()