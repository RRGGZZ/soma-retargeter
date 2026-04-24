# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
import joblib

import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


# Batch size for FK evaluation to avoid excessive GPU memory usage
_FK_BATCH_SIZE = 256


# def _compute_local_body_pos(buffer: CSVAnimationBuffer, robot_builder) -> np.ndarray:
#     """
#     Compute world-space body link positions for every frame via batched FK.

#     Args:
#         buffer: Retargeted animation buffer.  Each frame contains
#                 [tx, ty, tz, qx, qy, qz, qw, dof0, ...] (same layout as
#                 Newton's joint_q for one robot).
#         robot_builder: Single-robot ``newton.ModelBuilder`` used to build the
#                        FK model.  Its ``body_count`` must match the number of
#                        links in the robot.

#     Returns:
#         np.ndarray of shape ``(num_frames, num_bodies, 3)``, dtype float32.
#     """
#     import newton

#     num_frames = buffer.num_frames
#     num_bodies = robot_builder.body_count
#     joint_q_size = buffer.data[0].shape[0]

#     local_body_pos = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

#     # Process in batches to cap GPU memory usage
#     batch_size = _FK_BATCH_SIZE
#     for batch_start in range(0, num_frames, batch_size):
#         batch_end = min(batch_start + batch_size, num_frames)
#         n = batch_end - batch_start

#         # Build a temporary model with n robot copies for this batch
#         builder = newton.ModelBuilder()
#         for _ in range(n):
#             builder.add_builder(robot_builder, xform=wp.transform_identity())
#         model = builder.finalize()
#         state = model.state()

#         # Stack frame data -> (n * joint_q_size,) and copy to device
#         batch_data = np.stack([buffer.data[f] for f in range(batch_start, batch_end)], axis=0)
#         wp.copy(model.joint_q, wp.array(batch_data.flatten(), dtype=wp.float32))

#         # Run FK for all n environments at once
#         newton.eval_fk(model, model.joint_q, model.joint_qd, state)

#         # body_q shape: (n * num_bodies, 7); extract xyz positions
#         body_q = state.body_q.numpy()
#         local_body_pos[batch_start:batch_end] = body_q[:, 0:3].reshape(n, num_bodies, 3)

#     return local_body_pos


def save_pkl(
    file_path: str,
    buffer: CSVAnimationBuffer,
    robot_builder,
    dof_names: list[str],
    ik_targets: np.ndarray | None = None,
    ik_joint_names: list[str] | None = None,
) -> None:
    """
    Save a retargeted ``CSVAnimationBuffer`` as a joblib-compressed pickle file.

    The pickle contains a single ``motion_data`` dictionary with the fields
    required for downstream RL/motion-retargeting pipelines:

    .. code-block:: python

        {
            "fps"           : float,
            "root_pos"      : np.ndarray,  # (T, 3) – root translation (m)
            "root_rot"      : np.ndarray,  # (T, 4) – root quaternion (x, y, z, w)
            "dof_pos"       : np.ndarray,  # (T, D) – joint angles (rad)
            "local_body_pos": np.ndarray,  # (T, B, 3) – world-space link positions (m)
            "link_body_list": list[str],   # B body link names
            "dof_names"     : list[str],   # D DOF names
        }

    Args:
        file_path:    Destination ``.pkl`` path.
        buffer:       Retargeted animation buffer.
        robot_builder: Single-robot ``newton.ModelBuilder`` used to compute FK
                       and to derive the ordered body-link name list.
        dof_names:    Ordered list of DOF names matching ``buffer.data[:, 7:]``.
        ik_targets:   Optional scaler-computed IK targets with shape
                      ``(T, J, 7)`` in ``[tx, ty, tz, qx, qy, qz, qw]``.
        ik_joint_names: Optional effector joint names aligned with ``ik_targets``.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    num_frames = buffer.num_frames

    # Stack all frame data into a contiguous array: (num_frames, 7 + num_dofs)
    all_data = np.stack([buffer.data[f] for f in range(num_frames)], axis=0).astype(np.float32)

    root_pos_seq = all_data[:, 0:3]   # (T, 3)
    root_rot_seq = all_data[:, 3:7]   # (T, 4)  x, y, z, w
    dof_pos_seq  = all_data[:, 7:]    # (T, D)

    # Ordered body link names from the robot builder
    body_names = [newton_utils.get_name_from_label(label) for label in robot_builder.body_label]

    # print(f"[INFO]: Computing FK for {num_frames} frames (batch_size={_FK_BATCH_SIZE})...")
    local_body_pos_seq = None # _compute_local_body_pos(buffer, robot_builder)

    motion_data = {
        "fps"           : float(buffer.sample_rate),
        "root_pos"      : np.asarray(root_pos_seq, dtype=np.float32),
        "root_rot"      : np.asarray(root_rot_seq, dtype=np.float32),   # (x, y, z, w)
        "dof_pos"       : np.asarray(dof_pos_seq, dtype=np.float32),
        "local_body_pos": local_body_pos_seq, # np.asarray(local_body_pos_seq, dtype=np.float32),
        "link_body_list": body_names,
        "dof_names"     : list(dof_names),
    }

    if ik_targets is not None:
        if ik_targets.shape[0] != num_frames:
            raise ValueError(
                f"[ERROR]: IK target frame count mismatch. "
                f"Expected {num_frames}, got {ik_targets.shape[0]}."
            )
        ik_positions = np.asarray(ik_targets[:, :, 0:3], dtype=np.float32)
        ik_rot_xyzw = np.asarray(ik_targets[:, :, 3:7], dtype=np.float32)
        ik_rot_wxyz = np.concatenate([ik_rot_xyzw[:, :, 3:4], ik_rot_xyzw[:, :, 0:3]], axis=2)
        motion_data["ik_targets"] = {
            "joint_names": list(ik_joint_names) if ik_joint_names is not None else [],
            "positions": ik_positions,
            "rotations_wxyz": ik_rot_wxyz,
        }

    joblib.dump(motion_data, file_path)
    print(f"[INFO]: Saved PKL [{file_path}]")
