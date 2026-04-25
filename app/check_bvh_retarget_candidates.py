#!/usr/bin/env python3
"""
Folder-level coverage check between input BVH folders and output PKL folders.

This script compares the first-level folders under an input BVH root against the
first-level folders represented in an output PKL root, including PKL files that
were later moved under `_quality_check/*/bad_samples`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import soma_retargeter.assets.bvh as bvh_utils


def find_motion_files(input_dir: str | Path, source_type: str = "soma") -> list[Path]:
    root = Path(input_dir).expanduser().resolve()
    source_ext = "*.npz" if source_type == "npz" else "*.bvh"
    motion_files = list(root.rglob(source_ext))
    motion_files.sort(key=lambda path: path.stat().st_size, reverse=True)
    return motion_files


def _count_input_bvh_by_folder(input_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in input_dir.rglob("*.bvh"):
        rel = path.relative_to(input_dir)
        if len(rel.parts) >= 2:
            counts[rel.parts[0]] += 1
    return counts


def _count_current_pkl_by_folder(output_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in output_dir.rglob("*.pkl"):
        if "_quality_check" in path.parts:
            continue
        rel = path.relative_to(output_dir)
        if len(rel.parts) >= 2:
            counts[rel.parts[0]] += 1
    return counts


def _count_moved_bad_pkl_by_folder(output_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    quality_root = output_dir / "_quality_check"
    if not quality_root.exists():
        return counts

    for path in quality_root.rglob("*.pkl"):
        rel = path.relative_to(quality_root)
        if "bad_samples" not in rel.parts:
            continue
        bad_idx = rel.parts.index("bad_samples")
        if len(rel.parts) >= bad_idx + 2:
            counts[rel.parts[bad_idx + 1]] += 1
    return counts


def _write_outputs(rows: list[dict[str, Any]], report_dir: Path, summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with (report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _analyze_input_candidates(input_root: Path, source_type: str) -> dict[str, Any]:
    motion_files = find_motion_files(input_root, source_type=source_type)
    if not motion_files:
        raise FileNotFoundError(f"No candidate files found under {input_root}")

    reference_path = motion_files[0]
    reference_skeleton = None
    reference_num_joints: int | None = None

    folder_parse_ok: Counter[str] = Counter()
    folder_parse_failed: Counter[str] = Counter()
    folder_joint_match: Counter[str] = Counter()
    folder_joint_mismatch: Counter[str] = Counter()
    folder_conform_ok: Counter[str] = Counter()
    folder_conform_failed: Counter[str] = Counter()

    parse_failed_total = 0
    joint_mismatch_total = 0
    conform_ok_total = 0
    conform_failed_total = 0

    try:
        reference_skeleton, _ = bvh_utils.load_bvh(str(reference_path), source_type=source_type)
        reference_num_joints = int(reference_skeleton.num_joints)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load reference BVH [{reference_path}]: {exc}"
        ) from exc

    for motion_path in motion_files:
        rel = motion_path.relative_to(input_root)
        if len(rel.parts) < 2:
            continue
        folder_name = rel.parts[0]
        try:
            skeleton, _ = bvh_utils.load_bvh(str(motion_path), source_type=source_type)
            folder_parse_ok[folder_name] += 1
        except Exception:
            folder_parse_failed[folder_name] += 1
            parse_failed_total += 1
            continue

        if int(skeleton.num_joints) != reference_num_joints:
            folder_joint_mismatch[folder_name] += 1
            joint_mismatch_total += 1
            continue

        folder_joint_match[folder_name] += 1
        try:
            bvh_utils.load_bvh(
                str(motion_path),
                input_skeleton=reference_skeleton,
                source_type=source_type,
            )
            folder_conform_ok[folder_name] += 1
            conform_ok_total += 1
        except Exception:
            folder_conform_failed[folder_name] += 1
            conform_failed_total += 1

    return {
        "reference_path": reference_path,
        "reference_num_joints": reference_num_joints,
        "folder_parse_ok": folder_parse_ok,
        "folder_parse_failed": folder_parse_failed,
        "folder_joint_match": folder_joint_match,
        "folder_joint_mismatch": folder_joint_mismatch,
        "folder_conform_ok": folder_conform_ok,
        "folder_conform_failed": folder_conform_failed,
        "num_parse_failed_bvh": parse_failed_total,
        "num_num_joints_mismatch_bvh": joint_mismatch_total,
        "num_reference_conform_ok_bvh": conform_ok_total,
        "num_reference_conform_failed_bvh": conform_failed_total,
    }


def analyze_folder_coverage(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    source_type: str = "soma",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    report_root = Path(report_dir).expanduser().resolve()

    input_counts = _count_input_bvh_by_folder(input_root)
    if not input_counts:
        raise FileNotFoundError(f"No .bvh files found under {input_root}")

    current_counts = _count_current_pkl_by_folder(output_root)
    moved_bad_counts = _count_moved_bad_pkl_by_folder(output_root)
    candidate_stats = _analyze_input_candidates(input_root, source_type=source_type)

    rows: list[dict[str, Any]] = []
    for folder_name in sorted(input_counts):
        input_bvh_count = int(input_counts[folder_name])
        current_pkl_count = int(current_counts.get(folder_name, 0))
        moved_bad_pkl_count = int(moved_bad_counts.get(folder_name, 0))
        present_or_moved_pkl_count = current_pkl_count + moved_bad_pkl_count
        parse_ok_bvh_count = int(candidate_stats["folder_parse_ok"].get(folder_name, 0))
        parse_failed_bvh_count = int(candidate_stats["folder_parse_failed"].get(folder_name, 0))
        num_joints_match_bvh_count = int(
            candidate_stats["folder_joint_match"].get(folder_name, 0)
        )
        num_joints_mismatch_bvh_count = int(
            candidate_stats["folder_joint_mismatch"].get(folder_name, 0)
        )
        reference_conform_ok_bvh_count = int(
            candidate_stats["folder_conform_ok"].get(folder_name, 0)
        )
        reference_conform_failed_bvh_count = int(
            candidate_stats["folder_conform_failed"].get(folder_name, 0)
        )
        coverage_ratio = (
            float(present_or_moved_pkl_count / input_bvh_count)
            if input_bvh_count > 0
            else 0.0
        )
        rows.append(
            {
                "folder_name": folder_name,
                "input_bvh_count": input_bvh_count,
                "parse_ok_bvh_count": parse_ok_bvh_count,
                "parse_failed_bvh_count": parse_failed_bvh_count,
                "num_joints_match_bvh_count": num_joints_match_bvh_count,
                "num_joints_mismatch_bvh_count": num_joints_mismatch_bvh_count,
                "reference_conform_ok_bvh_count": reference_conform_ok_bvh_count,
                "reference_conform_failed_bvh_count": reference_conform_failed_bvh_count,
                "current_pkl_count": current_pkl_count,
                "moved_bad_pkl_count": moved_bad_pkl_count,
                "present_or_moved_pkl_count": present_or_moved_pkl_count,
                "coverage_ratio": coverage_ratio,
                "is_missing": present_or_moved_pkl_count == 0,
            }
        )

    missing_folders = [row["folder_name"] for row in rows if row["is_missing"]]
    summary = {
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "report_dir": str(report_root),
        "source_type": source_type,
        "reference_relative_path": str(
            candidate_stats["reference_path"].relative_to(input_root)
        ),
        "reference_num_joints": candidate_stats["reference_num_joints"],
        "num_input_folders": len(rows),
        "num_folders_with_any_output": len(rows) - len(missing_folders),
        "num_missing_folders": len(missing_folders),
        "num_parse_failed_bvh": candidate_stats["num_parse_failed_bvh"],
        "num_num_joints_mismatch_bvh": candidate_stats["num_num_joints_mismatch_bvh"],
        "num_reference_conform_ok_bvh": candidate_stats["num_reference_conform_ok_bvh"],
        "num_reference_conform_failed_bvh": candidate_stats["num_reference_conform_failed_bvh"],
        "missing_folders": missing_folders,
    }

    _write_outputs(rows, report_root, summary)
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare first-level input BVH folders against output PKL folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input root containing first-level BVH folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root containing PKL folders and optional _quality_check/bad_samples.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to <output-dir>/_folder_coverage_check/<timestamp>.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="soma",
        help="Source type passed to load_bvh when checking reference compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if args.report_dir is not None:
        report_dir = args.report_dir.expanduser().resolve()
    else:
        report_dir = output_dir / "_folder_coverage_check" / time.strftime("%Y%m%d_%H%M%S")

    rows, summary = analyze_folder_coverage(
        input_dir=input_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        source_type=str(args.source_type),
    )

    print(f"Input folders: {summary['num_input_folders']}")
    print(f"Folders with any output: {summary['num_folders_with_any_output']}")
    print(f"Missing folders: {summary['num_missing_folders']}")
    print(f"Reference: {summary['reference_relative_path']}")
    print(f"Reference num_joints: {summary['reference_num_joints']}")
    print(f"Parse failed BVH: {summary['num_parse_failed_bvh']}")
    print(f"num_joints mismatch BVH: {summary['num_num_joints_mismatch_bvh']}")
    print(f"CSV: {report_dir / 'summary.csv'}")
    print(f"JSON: {report_dir / 'summary.json'}")
    if rows:
        worst = min(rows, key=lambda row: row["coverage_ratio"])
        print(
            f"Worst coverage: {worst['folder_name']} "
            f"({worst['present_or_moved_pkl_count']}/{worst['input_bvh_count']}, "
            f"{worst['coverage_ratio']:.3%})"
        )


if __name__ == "__main__":
    main()
