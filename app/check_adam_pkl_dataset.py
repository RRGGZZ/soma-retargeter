#!/usr/bin/env python3
"""
Offline quality scan for retargeted adam_pro PKL trajectories.

This script scans a directory of ``.pkl`` motions, computes per-motion quality
metrics from the saved ``fps`` / ``dof_pos`` data, flags bad samples, writes a
summary report, and can optionally move bad samples into a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class QualityThresholds:
    max_dof_vel_threshold: float = 30.0


def find_motion_files(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    return sorted(
        path
        for path in root.rglob("*.pkl")
        if "_quality_check" not in path.parts
    )


def _to_float_array(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{key} must be an array, got scalar")
    return array


def _array_has_non_finite(array: np.ndarray) -> bool:
    return not np.isfinite(array).all()


def _safe_max_abs_diff(array: np.ndarray, fps: float) -> tuple[float, int | None, int | None]:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={array.shape}")
    if array.shape[0] < 2:
        return 0.0, None, None

    diff = np.diff(array, axis=0) * fps
    abs_diff = np.abs(diff)
    finite_mask = np.isfinite(abs_diff)
    if not finite_mask.any():
        return float("nan"), None, None

    masked = np.where(finite_mask, abs_diff, -np.inf)
    flat_index = int(np.argmax(masked))
    peak_value = float(masked.reshape(-1)[flat_index])
    peak_frame, peak_dof = np.unravel_index(flat_index, masked.shape)
    return peak_value, int(peak_frame + 1), int(peak_dof)


def _safe_root_linear_speed_max(root_pos: np.ndarray, fps: float) -> float:
    if root_pos.ndim != 2 or root_pos.shape[0] < 2 or root_pos.shape[1] != 3:
        return float("nan")
    diff = np.diff(root_pos, axis=0) * fps
    if diff.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(diff, axis=1)))


def analyze_motion_file(
    motion_path: str | Path,
    *,
    dataset_root: str | Path,
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or QualityThresholds()
    path = Path(motion_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    motion = joblib.load(path)

    required_keys = ("fps", "root_pos", "root_rot", "dof_pos")
    missing_keys = [key for key in required_keys if key not in motion]
    if missing_keys:
        raise KeyError(f"Missing required motion keys: {missing_keys}")

    fps = float(motion["fps"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid fps: {fps}")

    root_pos = _to_float_array(motion["root_pos"], "root_pos")
    root_rot = _to_float_array(motion["root_rot"], "root_rot")
    dof_pos = _to_float_array(motion["dof_pos"], "dof_pos")
    dof_names = list(motion.get("dof_names", []))

    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be 2D, got shape={dof_pos.shape}")

    num_frames = int(dof_pos.shape[0])
    dof_dim = int(dof_pos.shape[1])
    duration_seconds = max(num_frames - 1, 0) / fps

    shape_issues: list[str] = []
    metadata_warnings: list[str] = []
    if root_pos.shape != (num_frames, 3):
        shape_issues.append("root_pos_shape")
    if root_rot.shape != (num_frames, 4):
        shape_issues.append("root_rot_shape")
    if dof_names and len(dof_names) != dof_dim:
        metadata_warnings.append("dof_names_length")

    non_finite_fields: list[str] = []
    for key, array in (
        ("root_pos", root_pos),
        ("root_rot", root_rot),
        ("dof_pos", dof_pos),
    ):
        if _array_has_non_finite(array):
            non_finite_fields.append(key)

    dof_vel_abs_max, dof_vel_peak_frame, dof_vel_peak_dof_index = _safe_max_abs_diff(
        dof_pos, fps
    )
    dof_vel_peak_dof_name = ""
    if (
        dof_vel_peak_dof_index is not None
        and dof_names
        and 0 <= dof_vel_peak_dof_index < len(dof_names)
    ):
        dof_vel_peak_dof_name = str(dof_names[dof_vel_peak_dof_index])

    bad_reasons: list[str] = []
    if shape_issues:
        bad_reasons.append("shape_mismatch")
    if non_finite_fields:
        bad_reasons.append("non_finite")
    if math.isfinite(dof_vel_abs_max) and dof_vel_abs_max > thresholds.max_dof_vel_threshold:
        bad_reasons.append("dof_vel")

    row: dict[str, Any] = {
        "motion_name": path.stem,
        "file_path": str(path),
        "relative_path": str(path.relative_to(root)),
        "num_frames": num_frames,
        "dof_dim": dof_dim,
        "fps": fps,
        "duration_seconds": float(duration_seconds),
        "root_linear_speed_max": _safe_root_linear_speed_max(root_pos, fps),
        "dof_vel_abs_max": float(dof_vel_abs_max),
        "dof_vel_peak_frame": dof_vel_peak_frame,
        "dof_vel_peak_dof_index": dof_vel_peak_dof_index,
        "dof_vel_peak_dof_name": dof_vel_peak_dof_name,
        "shape_issue_flag": bool(shape_issues),
        "shape_issue_reasons": ",".join(shape_issues),
        "metadata_warning_flag": bool(metadata_warnings),
        "metadata_warning_reasons": ",".join(metadata_warnings),
        "non_finite_flag": bool(non_finite_fields),
        "non_finite_fields": ",".join(non_finite_fields),
        "dof_vel_bad_flag": bool(
            math.isfinite(dof_vel_abs_max)
            and dof_vel_abs_max > thresholds.max_dof_vel_threshold
        ),
        "bad_flag": bool(bad_reasons),
        "bad_reasons": ",".join(bad_reasons),
        "moved_to": "",
    }
    return row


def write_summary_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path = output_path / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bad_rows = [row for row in rows if bool(row.get("bad_flag"))]
    dof_vel_bad_rows = [row for row in rows if bool(row.get("dof_vel_bad_flag"))]
    non_finite_rows = [row for row in rows if bool(row.get("non_finite_flag"))]
    shape_issue_rows = [row for row in rows if bool(row.get("shape_issue_flag"))]
    metadata_warning_rows = [row for row in rows if bool(row.get("metadata_warning_flag"))]
    moved_rows = [row for row in rows if row.get("moved_to")]

    summary = {
        "output_dir": str(output_path),
        "csv_path": str(csv_path),
        "num_files": len(rows),
        "num_bad": len(bad_rows),
        "num_dof_vel_bad": len(dof_vel_bad_rows),
        "num_non_finite": len(non_finite_rows),
        "num_shape_issue": len(shape_issue_rows),
        "num_metadata_warning": len(metadata_warning_rows),
        "num_moved": len(moved_rows),
        "bad_relative_paths": [str(row["relative_path"]) for row in bad_rows],
        "dof_vel_bad_relative_paths": [
            str(row["relative_path"]) for row in dof_vel_bad_rows
        ],
    }

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    error_rows = errors or []
    if error_rows:
        (output_path / "errors.json").write_text(
            json.dumps(error_rows, indent=2), encoding="utf-8"
        )

    return summary


def move_bad_samples(
    *,
    rows: list[dict[str, Any]],
    dataset_root: str | Path,
    bad_output_dir: str | Path,
) -> list[dict[str, str]]:
    root = Path(dataset_root).expanduser().resolve()
    bad_root = Path(bad_output_dir).expanduser().resolve()
    errors: list[dict[str, str]] = []

    for row in rows:
        if not row.get("bad_flag"):
            continue

        src_path = Path(str(row["file_path"])).expanduser().resolve()
        dst_path = bad_root / Path(str(row["relative_path"]))
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists():
                raise FileExistsError(f"Destination already exists: {dst_path}")
            shutil.move(str(src_path), str(dst_path))
            row["moved_to"] = str(dst_path)
        except Exception as exc:  # pragma: no cover - exercised through errors path
            errors.append(
                {
                    "relative_path": str(row["relative_path"]),
                    "stage": "move",
                    "error": repr(exc),
                }
            )

    return errors


def run_quality_check(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    thresholds: QualityThresholds | None = None,
    move_bad: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    thresholds = thresholds or QualityThresholds()
    root = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for motion_path in find_motion_files(root):
        try:
            rows.append(
                analyze_motion_file(
                    motion_path,
                    dataset_root=root,
                    thresholds=thresholds,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "relative_path": str(motion_path.relative_to(root)),
                    "stage": "analyze",
                    "error": repr(exc),
                }
            )

    if move_bad:
        errors.extend(
            move_bad_samples(
                rows=rows,
                dataset_root=root,
                bad_output_dir=output_path / "bad_samples",
            )
        )

    summary = write_summary_outputs(rows=rows, output_dir=output_path, errors=errors)
    return rows, summary, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check adam_pro PKL dataset quality and optionally move bad samples."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("assets/motions/adam_pkl"),
        help="Input root directory containing PKL motions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to <input>/_quality_check/<timestamp>.",
    )
    parser.add_argument(
        "--max-dof-vel-threshold",
        type=float,
        default=30.0,
        help="Flag motions whose max absolute DOF velocity exceeds this value (rad/s).",
    )
    parser.add_argument(
        "--move-bad",
        action="store_true",
        help="Move flagged bad samples into <output-dir>/bad_samples while preserving relative paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = input_dir / "_quality_check" / time.strftime("%Y%m%d_%H%M%S")

    thresholds = QualityThresholds(
        max_dof_vel_threshold=float(args.max_dof_vel_threshold),
    )
    rows, summary, errors = run_quality_check(
        dataset_root=input_dir,
        output_dir=output_dir,
        thresholds=thresholds,
        move_bad=bool(args.move_bad),
    )

    print(f"Scanned files: {summary['num_files']}")
    print(f"Bad samples: {summary['num_bad']}")
    print(f"DOF-velocity bad: {summary['num_dof_vel_bad']}")
    print(f"Moved samples: {summary['num_moved']}")
    print(f"CSV: {output_dir / 'summary.csv'}")
    print(f"JSON: {output_dir / 'summary.json'}")
    if errors:
        print(f"Errors: {len(errors)} -> {output_dir / 'errors.json'}")
    if rows:
        print(
            f"Threshold: max |dof_vel| > {thresholds.max_dof_vel_threshold:.3f} rad/s"
        )


if __name__ == "__main__":
    main()
