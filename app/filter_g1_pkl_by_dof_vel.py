#!/usr/bin/env python3
"""
Filter PKL motions by max absolute DOF velocity.

Default behavior:
    - Scan assets/motions/g1_pkl recursively for *.pkl
    - Compute max |dof_vel| = max(|diff(dof_pos) * fps|)
    - Save paths with max |dof_vel| > threshold to a text file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def compute_max_abs_dof_vel(motion: dict) -> float:
    """Return max absolute DOF velocity (rad/s) for one motion."""
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = float(motion["fps"])
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be 2D, got shape={dof_pos.shape}")
    if dof_pos.shape[0] < 2:
        return 0.0
    dof_vel = np.diff(dof_pos, axis=0) * fps
    return float(np.max(np.abs(dof_vel)))


def filter_motions(input_dir: Path, threshold: float) -> list[tuple[Path, float]]:
    """Collect (file_path, max_abs_dof_vel) above threshold."""
    results: list[tuple[Path, float]] = []
    for pkl_path in sorted(input_dir.rglob("*.pkl")):
        try:
            motion = joblib.load(pkl_path)
            max_vel = compute_max_abs_dof_vel(motion)
        except Exception as exc:
            print(f"[WARN] Skip {pkl_path}: {exc}")
            continue

        if max_vel > threshold:
            results.append((pkl_path, max_vel))
    return results


def save_results(output_file: Path, results: list[tuple[Path, float]], root: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write("# path\tmax_abs_dof_vel(rad/s)\n")
        for p, max_vel in results:
            rel_path = p.relative_to(root)
            f.write(f"{rel_path}\t{max_vel:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter g1 PKL motions by max DOF velocity.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("assets/motions/g1_pkl"),
        help="Directory containing PKL motions (default: assets/motions/g1_pkl).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Velocity threshold in rad/s (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/motions/g1_pkl_max_dof_vel_over_50.txt"),
        help="Output txt path for filtered results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    threshold: float = args.threshold
    output_file: Path = args.output

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    results = filter_motions(input_dir, threshold)
    save_results(output_file, results, input_dir)

    total_files = len(list(input_dir.rglob("*.pkl")))
    print(f"Scanned: {total_files} PKL files")
    print(f"Threshold: max |dof_vel| > {threshold:.3f} rad/s")
    print(f"Matched: {len(results)}")
    print(f"Saved: {output_file.resolve()}")


if __name__ == "__main__":
    main()
