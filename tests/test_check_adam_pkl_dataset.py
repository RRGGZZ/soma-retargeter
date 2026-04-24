from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from app.check_adam_pkl_dataset import (
    QualityThresholds,
    analyze_motion_file,
    run_quality_check,
)


def _write_motion(
    file_path: Path,
    *,
    fps: float,
    dof_pos: np.ndarray,
    root_pos: np.ndarray | None = None,
    root_rot: np.ndarray | None = None,
    dof_names: list[str] | None = None,
) -> None:
    num_frames = int(dof_pos.shape[0])
    if root_pos is None:
        root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    if root_rot is None:
        root_rot = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (num_frames, 1))
    if dof_names is None:
        dof_names = [f"dof_{idx}" for idx in range(dof_pos.shape[1])]

    motion = {
        "fps": float(fps),
        "root_pos": np.asarray(root_pos, dtype=np.float32),
        "root_rot": np.asarray(root_rot, dtype=np.float32),
        "dof_pos": np.asarray(dof_pos, dtype=np.float32),
        "dof_names": list(dof_names),
        "link_body_list": [],
        "local_body_pos": None,
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion, file_path)


def test_analyze_motion_flags_dof_velocity_threshold(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    motion_path = dataset_root / "subset" / "bad.pkl"
    _write_motion(
        motion_path,
        fps=10.0,
        dof_pos=np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dof_names=["hip", "knee"],
    )

    row = analyze_motion_file(
        motion_path,
        dataset_root=dataset_root,
        thresholds=QualityThresholds(max_dof_vel_threshold=30.0),
    )

    assert row["bad_flag"] is True
    assert row["dof_vel_bad_flag"] is True
    assert "dof_vel" in row["bad_reasons"]
    assert row["dof_vel_abs_max"] == 40.0
    assert row["dof_vel_peak_dof_index"] == 0
    assert row["dof_vel_peak_dof_name"] == "hip"


def test_run_quality_check_writes_summary_and_moves_bad_samples(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    good_path = dataset_root / "set_a" / "good.pkl"
    bad_path = dataset_root / "set_b" / "bad.pkl"

    _write_motion(
        good_path,
        fps=10.0,
        dof_pos=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_motion(
        bad_path,
        fps=10.0,
        dof_pos=np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    output_dir = dataset_root / "_quality_check" / "run1"
    rows, summary, errors = run_quality_check(
        dataset_root=dataset_root,
        output_dir=output_dir,
        thresholds=QualityThresholds(max_dof_vel_threshold=30.0),
        move_bad=True,
    )

    assert errors == []
    assert len(rows) == 2
    assert summary["num_files"] == 2
    assert summary["num_bad"] == 1
    assert summary["num_dof_vel_bad"] == 1
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "summary.json").is_file()

    moved_bad = output_dir / "bad_samples" / "set_b" / "bad.pkl"
    assert moved_bad.is_file()
    assert not bad_path.exists()
    assert good_path.is_file()

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["bad_relative_paths"] == ["set_b/bad.pkl"]


def test_run_quality_check_ignores_existing_quality_output_tree(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    source_path = dataset_root / "keep" / "sample.pkl"
    ignored_path = dataset_root / "_quality_check" / "older" / "bad_samples" / "ignored.pkl"

    _write_motion(
        source_path,
        fps=20.0,
        dof_pos=np.zeros((3, 2), dtype=np.float32),
    )
    _write_motion(
        ignored_path,
        fps=20.0,
        dof_pos=np.full((3, 2), 99.0, dtype=np.float32),
    )

    output_dir = dataset_root / "_quality_check" / "run2"
    rows, summary, errors = run_quality_check(
        dataset_root=dataset_root,
        output_dir=output_dir,
        thresholds=QualityThresholds(max_dof_vel_threshold=30.0),
        move_bad=False,
    )

    assert errors == []
    assert len(rows) == 1
    assert summary["num_files"] == 1
    assert rows[0]["relative_path"] == "keep/sample.pkl"
