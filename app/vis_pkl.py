"""
Visualize retargeted motion PKL files with MuJoCo.

Each PKL file contains a single motion dict:
    {
        "fps"           : float,
        "root_pos"      : np.ndarray  (T, 3),  # root translation (m)
        "root_rot"      : np.ndarray  (T, 4),  # quaternion (x, y, z, w)
        "dof_pos"       : np.ndarray  (T, D),  # joint angles (rad)
        "local_body_pos": None | ndarray,
        "link_body_list": list[str],
        "dof_names"     : list[str],
        "ik_targets"    : {
            "joint_names"    : list[str],
            "positions"      : np.ndarray (T, J, 3),
            "rotations_wxyz" : np.ndarray (T, J, 4),
        } | optional,
    }

Usage examples:
    # Interactive viewer – single file
    python app/vis_pkl.py motion.pkl assets/robot/unitree_g1/g1_mocap_29dof.xml

    # Interactive viewer – whole directory (press N/P to switch motions, 9 to delete current file)
    python app/vis_pkl.py assets/motions/test-export/ assets/robot/unitree_g1/g1_mocap_29dof.xml

    # Offline render all motions to MP4
    python app/vis_pkl.py assets/motions/test-export/ assets/robot/unitree_g1/g1_mocap_29dof.xml --offline
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(os.getcwd())

import numpy as np
import mujoco
import mujoco.viewer
import joblib
import typer

# ──────────────────────────────────────────────────────────────────────────────
# Global state used by the keyboard callback
# ──────────────────────────────────────────────────────────────────────────────
_state = {
    "motion_idx": 0,
    "frame_idx": 0,
    "paused": False,
    "show_ik_targets": True,
}


def _load_pkl_files(path: str) -> list[tuple[Path, dict]]:
    """Return [(file_path, motion_dict), …] from a file or directory."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.pkl"))
    elif p.suffix in (".pkl", ".joblib"):
        files = [p]
    else:
        raise ValueError(f"Unsupported path: {path}")

    motions = []
    for f in files:
        data = joblib.load(f)
        motions.append((f, data))

    if not motions:
        raise FileNotFoundError(f"No PKL files found at: {path}")
    return motions


def _apply_frame(mj_data, motion: dict, frame_idx: int):
    """Write root pose + DOF angles into mj_data.qpos."""
    root_pos = motion["root_pos"][frame_idx]          # (3,)
    root_rot_xyzw = motion["root_rot"][frame_idx]     # (4,) x y z w
    dof_pos = motion["dof_pos"][frame_idx]            # (D,)

    mj_data.qpos[:3] = root_pos
    # MuJoCo qpos[3:7] expects (w, x, y, z)
    mj_data.qpos[3] = root_rot_xyzw[3]   # w
    mj_data.qpos[4] = root_rot_xyzw[0]   # x
    mj_data.qpos[5] = root_rot_xyzw[1]   # y
    mj_data.qpos[6] = root_rot_xyzw[2]   # z
    mj_data.qpos[7: 7 + len(dof_pos)] = dof_pos


def _quat_rotate_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by quaternion in wxyz order."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w, x, y, z = q
    u = np.array([x, y, z], dtype=np.float64)
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def _draw_ik_targets(scene, motion: dict, frame_idx: int, clear_scene: bool = False):
    """Draw IK target points (sphere) and orientations (RGB axes)."""
    if clear_scene:
        scene.ngeom = 0
    ik_targets = motion.get("ik_targets", None)
    if not ik_targets:
        return

    positions = ik_targets.get("positions", None)
    rotations = ik_targets.get("rotations_wxyz", None)
    if positions is None or rotations is None:
        return
    if frame_idx < 0 or frame_idx >= positions.shape[0]:
        return

    frame_positions = np.asarray(positions[frame_idx], dtype=np.float32)
    frame_rotations = np.asarray(rotations[frame_idx], dtype=np.float32)
    num_targets = min(frame_positions.shape[0], frame_rotations.shape[0])

    sphere_radius = 0.012
    orient_radius = 0.0025
    orient_length = 0.06
    base_color = np.array([0.95, 0.35, 0.15, 0.95], dtype=np.float32)
    axis_colors = (
        np.array([0.95, 0.25, 0.25, 0.95], dtype=np.float32),  # X
        np.array([0.25, 0.95, 0.25, 0.95], dtype=np.float32),  # Y
        np.array([0.25, 0.45, 0.95, 0.95], dtype=np.float32),  # Z
    )
    local_axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    for i in range(num_targets):
        # sphere + 3 axis capsules
        if scene.ngeom + 4 > scene.maxgeom:
            break

        pos = frame_positions[i]
        quat_wxyz = frame_rotations[i]

        # target point
        geom_point = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            geom_point,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([sphere_radius, 0.0, 0.0], dtype=np.float64),
            pos=np.array(pos, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=base_color,
        )

        # orientation axes (local XYZ)
        for local_axis, axis_color in zip(local_axes, axis_colors):
            axis_dir = _quat_rotate_wxyz(quat_wxyz, local_axis)
            norm = np.linalg.norm(axis_dir)
            if norm < 1e-8:
                continue
            axis_dir /= norm
            tip = np.asarray(pos, dtype=np.float64) + orient_length * axis_dir

            geom_orient = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mujoco.mjv_initGeom(
                geom_orient,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.zeros(3, dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=axis_color,
            )
            mujoco.mjv_connector(
                geom_orient,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                orient_radius,
                np.asarray(pos, dtype=np.float64),
                np.asarray(tip, dtype=np.float64),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Online (interactive) visualisation
# ──────────────────────────────────────────────────────────────────────────────

def online_visualize(motions: list[tuple[Path, dict]], humanoid_xml: str):
    """Interactive MuJoCo viewer.  Keyboard shortcuts:
        Space – pause / resume
        R     – restart current motion
        N     – next motion
        P     – previous motion
        9     – delete current motion file from disk and remove it from the playlist
    """
    _state["motion_idx"] = 0
    _state["frame_idx"] = 0
    _state["paused"] = False
    _state["show_ik_targets"] = True

    def key_callback(keycode):
        ch = chr(keycode)
        if ch == " ":
            _state["paused"] = not _state["paused"]
            print("Paused" if _state["paused"] else "Resumed")
        elif ch == "R":
            _state["frame_idx"] = 0
            print("Reset")
        elif ch == "N":
            if not motions:
                return
            _state["motion_idx"] = (_state["motion_idx"] + 1) % len(motions)
            _state["frame_idx"] = 0
            path, _ = motions[_state["motion_idx"]]
            print(f"Next  → [{_state['motion_idx'] + 1}/{len(motions)}] {path.stem}")
        elif ch == "P":
            if not motions:
                return
            _state["motion_idx"] = (_state["motion_idx"] - 1) % len(motions)
            _state["frame_idx"] = 0
            path, _ = motions[_state["motion_idx"]]
            print(f"Prev  → [{_state['motion_idx'] + 1}/{len(motions)}] {path.stem}")
        elif ch == "]":
            if not motions:
                return
            _state["motion_idx"] = (_state["motion_idx"] + 1) % len(motions)
            _state["frame_idx"] = 0
            path, _ = motions[_state["motion_idx"]]
            print(f"Next  → [{_state['motion_idx'] + 1}/{len(motions)}] {path.stem}")
        elif ch == "[":
            if not motions:
                return
            _state["motion_idx"] = (_state["motion_idx"] - 1) % len(motions)
            _state["frame_idx"] = 0
            path, _ = motions[_state["motion_idx"]]
            print(f"Prev  → [{_state['motion_idx'] + 1}/{len(motions)}] {path.stem}")
        elif ch == "9":
            if not motions:
                print("No motions loaded.")
                return
            idx = _state["motion_idx"]
            path, _ = motions[idx]
            try:
                path.unlink()
                print(f"Deleted file: {path}")
            except OSError as e:
                print(f"Failed to delete {path}: {e}")
                return
            motions.pop(idx)
            if not motions:
                print("No motions left; close the viewer window to quit.")
                return
            _state["motion_idx"] = min(idx, len(motions) - 1)
            _state["frame_idx"] = 0
            p2, _ = motions[_state["motion_idx"]]
            print(f"Now playing [{_state['motion_idx'] + 1}/{len(motions)}] {p2.stem}")
        elif ch == "I":
            _state["show_ik_targets"] = not _state["show_ik_targets"]
            print(f"IK targets {'ON' if _state['show_ik_targets'] else 'OFF'}")

    first_path, first_motion = motions[0]
    fps = float(first_motion["fps"])
    dt = 1.0 / fps
    n0 = len(motions)
    print(f"Playing [{1}/{n0}] {first_path.stem}  |  {first_motion['dof_pos'].shape[0]} frames @ {fps:.0f} fps")
    print("  Space=pause  R=restart  N/P or [/]=switch  9=delete  I=toggle IK targets")

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if not motions:
                viewer.sync()
                time.sleep(0.05)
                continue

            _, motion = motions[_state["motion_idx"]]
            n_frames = motion["dof_pos"].shape[0]
            fps = float(motion["fps"])
            dt = 1.0 / fps
            mj_model.opt.timestep = dt

            frame = _state["frame_idx"]
            if frame >= n_frames:
                n_total = len(motions)
                _state["motion_idx"] = (_state["motion_idx"] + 1) % n_total
                _state["frame_idx"] = 0
                frame = 0
                path, _ = motions[_state["motion_idx"]]
                print(f"Auto  → [{_state['motion_idx'] + 1}/{n_total}] {path.stem}")

            _apply_frame(mj_data, motion, frame)
            mujoco.mj_forward(mj_model, mj_data)
            if _state["show_ik_targets"]:
                _draw_ik_targets(viewer.user_scn, motion, frame, clear_scene=True)
            else:
                viewer.user_scn.ngeom = 0

            if not _state["paused"]:
                _state["frame_idx"] += 1

            viewer.sync()
            sleep_t = dt - (time.time() - step_start)
            if sleep_t > 0:
                time.sleep(sleep_t)


# ──────────────────────────────────────────────────────────────────────────────
# Offline rendering
# ──────────────────────────────────────────────────────────────────────────────

def _open_video_writer(video_file: Path, fps: int, width: int, height: int):
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            str(video_file), fps=fps, format="FFMPEG",
            codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_log_level="error",
        )
        return "imageio", writer
    except Exception:
        pass

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_file), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {video_file}")
    return "cv2", writer


def _append_frame(backend, writer, frame: np.ndarray):
    if backend == "imageio":
        writer.append_data(frame)
    else:
        writer.write(frame[:, :, ::-1])


def _close_writer(backend, writer):
    if backend == "imageio":
        writer.close()
    else:
        writer.release()


def offline_render(
    motions: list[tuple[Path, dict]],
    humanoid_xml: str,
    output_dir: str,
    width: int,
    height: int,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = mj_model.stat.extent * 2.2

    for i, (src_path, motion) in enumerate(motions, start=1):
        name = src_path.stem
        fps = max(1, int(round(float(motion["fps"]))))
        n_frames = motion["dof_pos"].shape[0]
        video_file = out / f"{name}.mp4"
        print(f"[{i}/{len(motions)}] Rendering {name}  ({n_frames} frames @ {fps} fps) → {video_file}")

        backend, writer = _open_video_writer(video_file, fps=fps, width=width, height=height)
        try:
            for f in range(n_frames):
                _apply_frame(mj_data, motion, f)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam)
                _draw_ik_targets(renderer.scene, motion, f, clear_scene=False)
                _append_frame(backend, writer, renderer.render())
        finally:
            _close_writer(backend, writer)

    renderer.close()
    print(f"\nDone. {len(motions)} video(s) saved to: {out.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(
    motion_path: str = typer.Argument(..., help="PKL file or directory containing PKL files"),
    asset_xml: str = typer.Argument(..., help="Path to MuJoCo XML robot asset"),
    offline: bool = typer.Option(False, "--offline", help="Render all motions to MP4 videos"),
    output_dir: str = typer.Option("video_pkl", "--output-dir", help="Output directory for MP4 files"),
    width: int = typer.Option(1280, help="Video width (offline mode)"),
    height: int = typer.Option(720, help="Video height (offline mode)"),
):
    motions = _load_pkl_files(motion_path)
    print(f"Loaded {len(motions)} motion(s) from: {motion_path}")

    if offline:
        offline_render(motions, asset_xml, output_dir, width, height)
    else:
        online_visualize(motions, asset_xml)


if __name__ == "__main__":
    typer.run(main)
