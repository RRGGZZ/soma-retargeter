# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import newton
import subprocess

import pathlib
import time
import warp as wp

import soma_retargeter.utils.math_utils as math_utils
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.assets.pkl as pkl_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils

from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

from tqdm import trange

_UI_NEWTON_PANEL_WIDTH  = 320
_UI_NEWTON_PANEL_MARGIN = 10
_UI_NEWTON_PANEL_ALPHA  = 0.9
_DEFAULT_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)


def _resolve_runtime_settings(config: dict, fallback_device: str):
    """Resolve runtime device and execution mode from config."""
    execution_mode = str(config.get("execution_mode", "gpu_parallel")).strip().lower()
    runtime_device = str(config.get("runtime_device", fallback_device)).strip()

    if execution_mode == "cpu_serial":
        runtime_device = "cpu"
    elif execution_mode not in ("gpu_parallel", "cpu_serial"):
        print(
            f"[WARNING]: Unknown execution_mode '{execution_mode}'. "
            "Falling back to 'gpu_parallel'."
        )
        execution_mode = "gpu_parallel"

    return runtime_device, execution_mode


def _resolve_viewer_robot_offset(config: dict):
    """Resolve fixed robot visualization offset from config."""
    default = [1.0, 0.0, 0.0]
    raw = config.get("viewer_robot_offset", default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        print(
            f"[WARNING]: Invalid viewer_robot_offset={raw}. "
            f"Expected [x, y, z], fallback to {default}."
        )
        raw = default
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    except (TypeError, ValueError):
        print(
            f"[WARNING]: Invalid viewer_robot_offset={raw}. "
            f"Expected numeric [x, y, z], fallback to {default}."
        )
        return default


def _apply_shard_filter(motion_files: list, config: dict) -> list:
    """Filter motion_files to a single shard using alternating-index assignment.

    If ``shard_index`` and ``shard_count`` are present in *config*, returns
    only the files whose position in the sorted list satisfies
    ``index % shard_count == shard_index``.  When either key is absent or
    ``shard_count <= 1``, the original list is returned unchanged (backward
    compatible with non-sharded runs).
    """
    shard_index = config.get("shard_index")
    shard_count = config.get("shard_count")
    if shard_index is None or shard_count is None:
        return motion_files
    shard_index = int(shard_index)
    shard_count = int(shard_count)
    if shard_count <= 1:
        return motion_files
    filtered = [f for i, f in enumerate(motion_files) if i % shard_count == shard_index]
    print(
        f"[INFO]: Shard {shard_index}/{shard_count}: "
        f"selected {len(filtered)} of {len(motion_files)} files."
    )
    return filtered


def _resolve_export_settings(config: dict):
    """Resolve export root and whether CSV export should be skipped."""
    export_folder = str(config.get("export_folder", "")).strip()
    if len(export_folder) == 0:
        raise ValueError("[ERROR]: No export folder specified.")

    export_path = pathlib.Path(export_folder).expanduser()
    export_pkl_only = export_path.name.endswith("_pkl")
    return export_path, export_pkl_only


class Viewer:
    def __init__(self, viewer, config):
        self.viewer = viewer
        self.config = config
        self.converter = SpaceConverter(get_facing_direction_type_from_str(self.config['retarget_source_facing_direction']))

        if isinstance(self.viewer, newton.viewer.ViewerNull):
            # Headless mode for batch processing
            return
        
        self.fps      = int(self.config.get("viewer_fps", 60))
        if self.fps <= 0:
            print(f"[WARNING]: Invalid viewer_fps={self.fps}, fallback to 60.")
            self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.time     = 0.0
        self.viewer.vsync = bool(self.config.get("viewer_vsync", True))

        self.is_playing          = True
        self.playback_time       = 0.0
        self.playback_speed      = 1.0
        self.playback_loop       = True
        self.playback_total_time = 0.0
        self.record_capture_ui   = bool(self.config.get("viewer_record_capture_ui", False))
        self.record_output_dir   = pathlib.Path(self.config.get("viewer_record_output_dir", "assets/video"))
        self.is_recording        = False
        self.record_process      = None
        self.record_output_file  = None
        self.record_frame_count  = 0
        self._r_prev_down        = False

        self.retarget_source_options = ['soma', 'ours']
        self.retarget_target_options = ['unitree_g1', 'adam_pro', 'kai']
        self.retarget_solver_options = ['Newton']
        self.retarget_solver_idx     = 0
        _default_target = self.config.get('retarget_target', 'unitree_g1')
        self.retarget_target_idx     = self.retarget_target_options.index(_default_target) if _default_target in self.retarget_target_options else 0
        _default_source = self.config.get('retarget_source', 'soma')
        self.retarget_source_idx     = self.retarget_source_options.index(_default_source) if _default_source in self.retarget_source_options else 0

        self.show_skeleton_mesh = True
        self.show_skeleton = False
        self.show_skeleton_joint_axes = False
        self.show_robot = True
        self.show_gizmos = True
        self.robot_base_offset = _resolve_viewer_robot_offset(self.config)

        self.viewer.renderer.set_title("BVH to CSV Converter")
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

        _ROBOT_MJCF_MAP = {
            'unitree_g1': "assets/robot/unitree_g1/g1_mocap_29dof.xml",
            'adam_pro':   "assets/robot/adam_pro/adam_pro.xml",
            'kai':        "assets/robot/kai/kai.xml",
        }
        _retarget_target = self.config.get('retarget_target', 'unitree_g1')
        _mjcf_rel_path = _ROBOT_MJCF_MAP.get(_retarget_target)
        if _mjcf_rel_path is None:
            raise ValueError(f"[ERROR]: Unknown retarget_target '{_retarget_target}'. Supported: {list(_ROBOT_MJCF_MAP.keys())}")

        robot_builder = newton.ModelBuilder()
        robot_mjcf_path = str(pathlib.Path(__file__).parent.parent / _mjcf_rel_path)
        robot_builder.add_mjcf(robot_mjcf_path)

        self.num_robots = 1
        self.robot_offsets = [
            wp.transform(
                wp.vec3(
                    self.robot_base_offset[0],
                    self.robot_base_offset[1] + i - (self.num_robots - 1) / 2.0,
                    self.robot_base_offset[2]),
                wp.quat_identity())
            for i in range(self.num_robots)
        ]
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for _ in range(self.num_robots):
            builder.add_builder(robot_builder, wp.transform_identity())
        self.model = builder.finalize()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.state = self.model.state()

        self.g1_num_joint_q = self.model.joint_coord_count // self.model.articulation_count
        self.g1_joint_q_offsets = [int(i * self.g1_num_joint_q) for i in range(self.model.articulation_count)]
        self.g1_default_joint_q_values = self.model.joint_q.numpy()

        self.coordinate_renderer = CoordinateRenderer()
        self.skeleton = None
        self.skeleton_renderer = None
        self.skeletal_mesh_renderer = None

        self.animation_offsets = []
        self.animation_buffers = []
        self.skeleton_instances = []
        self.robot_csv_animation_buffers = [None for _ in range(self.num_robots)]

    def gui(self, ui):
        self.ui_playback_controls(ui)
        self.ui_scene_options(ui)

    def load_csv_file(self, path):
        self.robot_csv_animation_buffers[0] = csv_utils.load_csv(path)
        self.compute_playback_total_time()

    def load_bvh_file(self, path):
        self.animation_buffers = []
        self.skeleton_instances = []
        if self.skeleton_renderer is not None:
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None:
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None:
            self.coordinate_renderer.clear(self.viewer)

        retarget_source = self.config.get('retarget_source', 'soma')
        self.skeleton, animation = bvh_utils.load_bvh(path, source_type=retarget_source)
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.skeleton_instances = [SkeletonInstance(self.skeleton, _DEFAULT_COLOR, self.converter.transform(wp.transform_identity()))]
        self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        self.animation_buffers = [animation]

        source_type = pipeline_utils.get_source_type_from_str(retarget_source)
        self.skeletal_mesh = pipeline_utils.get_source_model_mesh(source_type, self.skeleton)
        self.skeletal_mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)
        self.compute_playback_total_time()

    def compute_playback_total_time(self):
        bvh_max_time = 0.0
        for buffer in self.animation_buffers:
            if buffer is not None:
                bvh_max_time = max(bvh_max_time, buffer.num_frames * (1 / buffer.sample_rate))
        
        csv_max_time = 0.0
        for buffer in self.robot_csv_animation_buffers:
            if buffer is not None:
                csv_max_time = max(csv_max_time, buffer.num_frames * (1 / buffer.sample_rate))

        self.playback_total_time = max(bvh_max_time, csv_max_time)
        self.playback_time = wp.clamp(self.playback_time, 0.0, self.playback_total_time)

    def update_robot_states(self):
        for i in range(self.num_robots):
            robot_offset = self.robot_offsets[i]

            joint_q_offset = self.g1_joint_q_offsets[i]
            if self.robot_csv_animation_buffers[i] is not None:
                buffer = self.robot_csv_animation_buffers[i]
                # Apply visual offset
                prev_xform = wp.transform(buffer.xform)
                buffer.xform = robot_offset

                data = buffer.sample(self.playback_time)
                wp.copy(self.model.joint_q, wp.array(data, dtype=wp.float32), joint_q_offset, 0, self.g1_num_joint_q)
                buffer.xform = prev_xform
            else:
                root_tx = wp.mul(
                    robot_offset,
                    wp.transform(*self.g1_default_joint_q_values[joint_q_offset:(joint_q_offset + 7)]))

                wp.copy(
                    self.model.joint_q,
                    wp.array(self.g1_default_joint_q_values[joint_q_offset:(joint_q_offset + self.g1_num_joint_q)], dtype=wp.float32),
                    joint_q_offset,
                    0, self.g1_num_joint_q)
                wp.copy(self.model.joint_q, wp.array(root_tx[0:7], dtype=wp.float32), joint_q_offset, 0, 7)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def step(self):
        r_down = self.viewer.is_key_down("r")
        if r_down and not self._r_prev_down:
            self.toggle_recording()
        self._r_prev_down = r_down

        self.time += self.frame_dt
        if self.is_playing:
            self.playback_time += self.frame_dt * self.playback_speed
            if self.playback_loop and self.playback_total_time > 0.0:
                self.playback_time %= self.playback_total_time
            else:
                self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

        for i in range(len(self.animation_buffers)):
            self.skeleton_instances[i].set_local_transforms(self.animation_buffers[i].sample(self.playback_time))

        def clamp_gizmo_transform(tx: wp.transform):
            return wp.transform(
                wp.vec3(tx.p[0], tx.p[1], 0.0),
                math_utils.quat_twist(wp.vec3(0.0, 0.0, 1.0), tx.q))

        for i in range(len(self.robot_offsets)):
            self.robot_offsets[i] = clamp_gizmo_transform(self.robot_offsets[i])
        for i in range(len(self.animation_offsets)):
            self.animation_offsets[i] = clamp_gizmo_transform(self.animation_offsets[i])

        self.update_robot_states()

    def render(self):
        self.viewer.begin_frame(self.time)
        if len(self.animation_buffers) > 0:
            for i in range(len(self.skeleton_instances)):
                prev_xform = wp.transform(self.skeleton_instances[i].xform)
                self.skeleton_instances[i].xform = wp.mul(self.animation_offsets[i], self.skeleton_instances[i].xform)
                if self.show_skeleton:
                    self.skeleton_renderer.draw(self.viewer, self.skeleton_instances[i], i)
                if self.show_skeleton_joint_axes:
                    tx = self.skeleton_instances[i].compute_global_transforms()
                    self.coordinate_renderer.draw(self.viewer, tx, 0.1, i)
                if self.show_skeleton_mesh:
                    self.skeletal_mesh_renderer.draw(self.viewer, self.skeleton_instances[i], self.skeleton_instances[i].color, i)
                self.skeleton_instances[i].xform = prev_xform
        
        if self.show_gizmos:
            for i, offset in enumerate(self.robot_offsets):
                self.viewer.log_gizmo(f"robot_offset{i}", offset)
            for i, offset in enumerate(self.animation_offsets):
                self.viewer.log_gizmo(f"animation_offset{i}", offset)
        
        if self.show_robot:
            self.viewer.log_state(self.state)
        self.viewer.end_frame()
        self.record_frame()

    def run(self):
        while self.viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                self.step()
            with wp.ScopedTimer("render", active=False):
                self.render()

        self.stop_recording()
        self.viewer.close()

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.record_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.record_output_file = self.record_output_dir / f"viewer_record_{timestamp}.mp4"
        self.record_process = None
        self.record_frame_count = 0
        self.is_recording = True
        print(f"[INFO]: Recording started. Press R again to stop. Output: {self.record_output_file}")

    def stop_recording(self):
        if not self.is_recording and self.record_process is None:
            return

        self.is_recording = False
        if self.record_process is not None:
            if self.record_process.stdin is not None:
                self.record_process.stdin.close()
            exit_code = self.record_process.wait()
            if exit_code != 0:
                print(f"[WARNING]: Recording encoder exited with code {exit_code}")
            else:
                print(
                    f"[INFO]: Recording saved: {self.record_output_file} "
                    f"({self.record_frame_count} frames)"
                )
        self.record_process = None

    def _create_video_encoder(self, width: int, height: int):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.record_output_file),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def record_frame(self):
        if not self.is_recording:
            return

        frame_wp = self.viewer.get_frame(render_ui=self.record_capture_ui)
        frame = frame_wp.numpy()
        height, width, channels = frame.shape
        if channels != 3:
            print(f"[WARNING]: Unexpected frame channels={channels}, skip recording frame.")
            return

        if self.record_process is None:
            try:
                self.record_process = self._create_video_encoder(width, height)
            except FileNotFoundError:
                print("[ERROR]: ffmpeg not found. Stop recording.")
                self.stop_recording()
                return

        if self.record_process.stdin is not None:
            try:
                self.record_process.stdin.write(frame.tobytes())
                self.record_frame_count += 1
            except BrokenPipeError:
                print("[ERROR]: Recording pipe closed unexpectedly. Stop recording.")
                self.stop_recording()

    def retarget_motion(self):
        retarget_source = self.retarget_source_options[self.retarget_source_idx]
        retarget_target = self.retarget_target_options[self.retarget_target_idx]
        retarget_solver = self.retarget_solver_options[self.retarget_solver_idx]
        
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            pipeline = newton_pipeline.NewtonPipeline(self.skeleton, retarget_source, retarget_target)
        else:
            raise(ValueError(f"[ERROR]: Unknown retargeter solver [{retarget_solver}"))
        
        r_offsets = [wp.transform(wp.vec3(0,0,0), wp.quat(*s.xform[3:7])) for s in self.skeleton_instances]
        pipeline.add_input_motions(self.animation_buffers, r_offsets, True)
        buffers = pipeline.execute()
        
        if buffers is not None:
            t_offsets = [wp.transform(wp.vec3(*s.xform[:3]), wp.quat_identity()) for s in self.skeleton_instances]
            for i, buffer in enumerate(buffers):
                buffer.xform = t_offsets[i]

        self.robot_csv_animation_buffers[0] = buffers[0]

    def ui_scene_options(self, ui):
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog
        
        viewport = ui.get_main_viewport()

        panel_size = ui.ImVec2(320, 320)
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.size.x - _UI_NEWTON_PANEL_MARGIN - panel_size.x,
                viewport.size.y - _UI_NEWTON_PANEL_MARGIN - panel_size.y))
        
        ui.set_next_window_size(panel_size)
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Scene Options", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.separator()

        # Motion options
        if ui.collapsing_header("Motion", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()
            ui.align_text_to_frame_padding()
            ui.text("BVH Motion:")
            ui.same_line()
            
            ui.push_id(100)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                bvh_path = tk_filedialog.askopenfilename(
                    title='Load BVH File',
                    defaultextension=".bvh",
                    filetypes=[('BVH files', '*.bvh')])

                if bvh_path:
                    self.load_bvh_file(bvh_path)
            ui.pop_id()

            if (len(self.animation_buffers) == 0):
                ui.begin_disabled()

            ui.same_line()
            if ui.button("Retarget"):
                self.retarget_motion()
            
            if (len(self.animation_buffers) == 0):
                ui.end_disabled()

            ui.align_text_to_frame_padding()
            ui.text("CSV Motion:")
            ui.same_line()
            
            ui.push_id(200)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                csv_path = tk_filedialog.askopenfilename(
                    title='Load CSV File',
                    defaultextension=".csv",
                    filetypes=[('CSV files', '*.csv')])

                if csv_path:
                    self.load_csv_file(csv_path)

            if self.robot_csv_animation_buffers[0] is None:
                ui.begin_disabled()
            ui.pop_id()

            ui.same_line()
            if ui.button("Save"):
                root = tk.Tk()
                root.withdraw()

                save_path = tk_filedialog.asksaveasfilename(
                    title="Save CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if save_path:
                    csv_utils.save_csv(save_path, self.robot_csv_animation_buffers[0])

            if self.robot_csv_animation_buffers[0] is None:
                ui.end_disabled()

        # Visibility options
        ui.spacing()
        if ui.collapsing_header("Visibility", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()

            _, self.show_robot = ui.checkbox("Show Robot", self.show_robot)
            changed, self.show_skeleton_mesh = ui.checkbox("Show Mesh", self.show_skeleton_mesh)
            if changed and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.clear(self.viewer)
            changed, self.show_skeleton = ui.checkbox("Show Skeleton", self.show_skeleton)
            if changed and self.skeleton_renderer is not None:
                self.skeleton_renderer.clear(self.viewer)
            changed, self.show_skeleton_joint_axes = ui.checkbox("Show Joint Axes", self.show_skeleton_joint_axes)
            if changed and self.coordinate_renderer is not None:
                self.coordinate_renderer.clear(self.viewer)
            _, self.show_gizmos = ui.checkbox("Show Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset"):
                self.robot_offsets = [
                    wp.transform(
                        wp.vec3(
                            self.robot_base_offset[0],
                            self.robot_base_offset[1] + i - (self.num_robots - 1) / 2.0,
                            self.robot_base_offset[2]),
                        wp.quat_identity())
                    for i in range(self.num_robots)
                ]
                self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        ui.end()

    def ui_playback_controls(self, ui):
        viewport = ui.get_main_viewport()
        
        panel_height = 105
        panel_width = viewport.size.x - 2 * (2 * _UI_NEWTON_PANEL_MARGIN + _UI_NEWTON_PANEL_WIDTH)
        
        ui.set_next_window_pos(ui.ImVec2(_UI_NEWTON_PANEL_WIDTH + _UI_NEWTON_PANEL_MARGIN, viewport.size.y - _UI_NEWTON_PANEL_MARGIN - panel_height))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Playback Controls", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        # Time slider
        ui.align_text_to_frame_padding()
        ui.text("Time (s):")
        ui.same_line()
        ui.set_next_item_width(panel_width - 150)
        changed, new_time = ui.slider_float(
            "##TimeSlider",
            self.playback_time,
            0.0,
            self.playback_total_time,
            "%.2f")
        if changed:
            self.playback_time = wp.clamp(new_time, 0.0, self.playback_total_time)
        ui.same_line()
        ui.text_colored(ui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{self.playback_total_time:.2f}s")
        
        self.is_playing = not ui.button("Pause") if self.is_playing else ui.button("Play ")
        ui.same_line()

        # Speed slider
        ui.align_text_to_frame_padding()
        ui.text("Speed")
        ui.same_line()
        ui.set_next_item_width(100)
        changed, new_speed = ui.slider_float(
            "##SpeedSlider",
            self.playback_speed,
            -2.0, 2.0,
            "%.2f"
        )
        if changed:
            self.playback_speed = new_speed
        ui.same_line()
        _, self.playback_loop = ui.checkbox("Loop", self.playback_loop)
        ui.end()

    def batched_retargeting(self):
        if not os.path.isdir(self.config['import_folder']):
            print(f"[ERROR]: Import folder does not exist {self.config['import_folder']}.")
            exit(-1)

        import_path = pathlib.Path(self.config['import_folder'])
        try:
            export_path, export_pkl_only = _resolve_export_settings(self.config)
        except ValueError as exc:
            print(str(exc))
            exit(-1)
        if not export_path.is_dir():
            print(f"[WARNING]: Export folder does not exist! Creating new folder at {str(export_path)}!")
            export_path.mkdir(parents=True, exist_ok=True)
        if export_pkl_only:
            print(f"[INFO]: PKL-only export mode enabled for folder '{export_path}'.")

        execution_mode = str(self.config.get("execution_mode", "gpu_parallel")).strip().lower()
        configured_batch_size = int(self.config['batch_size'])
        batch_size = 1 if execution_mode == "cpu_serial" else configured_batch_size
        if execution_mode == "cpu_serial":
            print(f"[INFO]: CPU serial mode enabled. Using batch_size=1 (configured={configured_batch_size}).")

        source_ext = "*.npz" if self.config['retarget_source'] == "npz" else "*.bvh"
        motion_files = list(import_path.rglob(source_ext))
        if len(motion_files) == 0:
            print(f"[ERROR]: Import folder {str(import_path)}, does not contain any {source_ext} files.")
            exit(-1)

        # Sort files based on size (largest first)
        motion_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        motion_files = _apply_shard_filter(motion_files, self.config)
        if len(motion_files) == 0:
            print("[INFO]: Shard is empty. Nothing to process. Exiting.")
            return
        batches = [motion_files[i:i + batch_size] for i in range(0, len(motion_files), batch_size)]
        
        retarget_source = self.config['retarget_source']
        # All skeletons should be the same, load one as our reference.
        bvh_skeleton, _ = bvh_utils.load_bvh(batches[0][0], source_type=retarget_source)

        bvh_tx_converter = self.converter.transform(wp.transform_identity())
        expected_num_joints = bvh_skeleton.num_joints

        retarget_solver = self.config['retargeter']
        retarget_target = self.config["retarget_target"]
        retarget_pipeline = None
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
            retarget_pipeline = newton_pipeline.NewtonPipeline(bvh_skeleton, retarget_source, retarget_target)
        if retarget_pipeline is None:
            print(f"[ERROR]: Invalid retarget solver selected [{retarget_solver}]. Use 'Newton'.")
            exit(-1)

        # Select robot-specific CSV config
        _CSV_CONFIG_MAP = {
            'unitree_g1': csv_utils.UnitreeG129DOF_CSVConfig(),
            'adam_pro':   csv_utils.AdamPro29DOF_CSVConfig(),
            'kai':        csv_utils.Kai53DOF_CSVConfig(),
        }
        csv_config = _CSV_CONFIG_MAP.get(retarget_target)
        if csv_config is None:
            print(f"[ERROR]: No CSV config found for retarget target [{retarget_target}].")
            exit(-1)

        nb_retargeted_motions = 0
        start_time = time.time()

        for i, batch in enumerate(batches):
            print(f"[INFO]: Processing batch {i+1} of {len(batches)}")
            
            print(f"[INFO]: Loading {len(batch)} animations...")
            animations = []
            for file_path in batch:
                _, animation = bvh_utils.load_bvh(file_path, bvh_skeleton, source_type=retarget_source)
                # All animations should be on the same skeleton
                assert expected_num_joints == animation.skeleton.num_joints, (
                    f"[ERROR]: Unexpected number of joints in input motion. Expected {expected_num_joints}, "
                    f"got {animation.skeleton.num_joints}")
                
                animations.append(animation)
            assert(len(animations) == len(batch))

            if (len(animations) > 0):
                print("[INFO]: Retargeting...")
                retarget_pipeline.clear()
                retarget_pipeline.add_input_motions(animations, [bvh_tx_converter] * len(animations), True)
                csv_buffers = retarget_pipeline.execute()
                ik_targets, ik_joint_names = retarget_pipeline.get_scaled_ik_targets(trim_initialization_frames=True)

                assert(len(csv_buffers) == len(animations))
                assert(len(ik_targets) == len(animations))

                # DOF names from the robot-specific CSV config (skip Frame + 6 root fields)
                dof_names = csv_config.csv_header[7:]

                for i in trange(len(csv_buffers), desc="[INFO]: Exporting Files"):
                    csv_buffer = csv_buffers[i]
                    rel_path = pathlib.Path(batch[i]).relative_to(import_path)

                    # Save CSV unless this export folder is explicitly PKL-only.
                    if not export_pkl_only:
                        dst_csv = export_path / rel_path.with_suffix(".csv")
                        dst_csv.parent.mkdir(parents=True, exist_ok=True)
                        csv_utils.save_csv(dst_csv, csv_buffer, csv_config)

                    # Save PKL to the configured export root.
                    dst_pkl = export_path / rel_path.with_suffix(".pkl")
                    dst_pkl.parent.mkdir(parents=True, exist_ok=True)
                    pkl_utils.save_pkl(
                        dst_pkl,
                        csv_buffer,
                        retarget_pipeline.robot_builder,
                        dof_names,
                        ik_targets=ik_targets[i],
                        ik_joint_names=ik_joint_names,
                    )

            nb_retargeted_motions += len(batch)

        elapsed_time = time.time() - start_time
        elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
        print(
            f"[INFO]: Retargeted {nb_retargeted_motions} animations successfully "
            f"in {elapsed_str} "
            f"[{(elapsed_time/nb_retargeted_motions):.2f}s per motion]!")

def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer=("null"))
    parser.add_argument(
        "--config",
        type=lambda x: None if x == "None" else str(x),
        default="./assets/default_bvh_to_csv_converter_config.json",
        help="Input json config file.")

    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Main config json file not found: {args.config}")
        exit(1)

    config = io_utils.load_json(args.config)
    runtime_device, execution_mode = _resolve_runtime_settings(config, args.device)
    print(f"[INFO]: Runtime device = {runtime_device}, execution mode = {execution_mode}")

    with wp.ScopedDevice(runtime_device):
        app = Viewer(viewer, config)
        if not isinstance(viewer, newton.viewer.ViewerNull):
            app.run()
        else:
            app.batched_retargeting()

if __name__ == "__main__":
    main()
