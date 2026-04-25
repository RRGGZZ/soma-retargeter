from __future__ import annotations

import json
from pathlib import Path

from app.check_bvh_retarget_candidates import analyze_folder_coverage


def _write_text(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_analyze_folder_coverage_counts_current_and_moved_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_text(input_dir / "210531" / "a.bvh")
    _write_text(input_dir / "210531" / "b.bvh")
    _write_text(input_dir / "210707" / "c.bvh")
    _write_text(input_dir / "220705" / "d.bvh")

    _write_text(output_dir / "210531" / "a.pkl")
    _write_text(output_dir / "210707" / "c.pkl")
    _write_text(output_dir / "_quality_check" / "run1" / "bad_samples" / "210531" / "b.pkl")

    class FakeSkeleton:
        def __init__(self, num_joints: int):
            self.num_joints = num_joints

    class FakeAnimation:
        def __init__(self, skeleton: FakeSkeleton):
            self.skeleton = skeleton
            self.num_frames = 10
            self.sample_rate = 120.0

    def fake_load_bvh(path: str, input_skeleton=None, source_type: str = "soma"):
        skeleton = input_skeleton or FakeSkeleton(31)
        return skeleton, FakeAnimation(skeleton)

    import app.check_bvh_retarget_candidates as mod

    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(mod.bvh_utils, "load_bvh", fake_load_bvh)
    try:
        report_dir = tmp_path / "report"
        rows, summary = analyze_folder_coverage(
            input_dir=input_dir,
            output_dir=output_dir,
            report_dir=report_dir,
        )
    finally:
        monkeypatch.undo()

    assert summary["num_input_folders"] == 3
    assert summary["num_folders_with_any_output"] == 2
    assert summary["num_missing_folders"] == 1
    assert summary["missing_folders"] == ["220705"]

    row_by_folder = {row["folder_name"]: row for row in rows}
    assert row_by_folder["210531"]["input_bvh_count"] == 2
    assert row_by_folder["210531"]["current_pkl_count"] == 1
    assert row_by_folder["210531"]["moved_bad_pkl_count"] == 1
    assert row_by_folder["210531"]["present_or_moved_pkl_count"] == 2
    assert row_by_folder["210531"]["coverage_ratio"] == 1.0

    assert row_by_folder["210707"]["input_bvh_count"] == 1
    assert row_by_folder["210707"]["present_or_moved_pkl_count"] == 1
    assert row_by_folder["210707"]["is_missing"] is False

    assert row_by_folder["220705"]["present_or_moved_pkl_count"] == 0
    assert row_by_folder["220705"]["is_missing"] is True

    assert (report_dir / "summary.csv").is_file()
    summary_json = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["missing_folders"] == ["220705"]


def test_analyze_folder_coverage_ignores_non_matching_root_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    _write_text(input_dir / "230101" / "clip1.bvh")
    _write_text(output_dir / "230101" / "clip1.pkl")
    _write_text(output_dir / "random_clip.pkl")
    _write_text(output_dir / "_quality_check" / "run2" / "bad_samples" / "not_a_folder.pkl")

    class FakeSkeleton:
        def __init__(self, num_joints: int):
            self.num_joints = num_joints

    class FakeAnimation:
        def __init__(self, skeleton: FakeSkeleton):
            self.skeleton = skeleton
            self.num_frames = 10
            self.sample_rate = 120.0

    def fake_load_bvh(path: str, input_skeleton=None, source_type: str = "soma"):
        skeleton = input_skeleton or FakeSkeleton(31)
        return skeleton, FakeAnimation(skeleton)

    import app.check_bvh_retarget_candidates as mod

    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(mod.bvh_utils, "load_bvh", fake_load_bvh)
    try:
        rows, summary = analyze_folder_coverage(
            input_dir=input_dir,
            output_dir=output_dir,
            report_dir=tmp_path / "report",
        )
    finally:
        monkeypatch.undo()

    assert len(rows) == 1
    assert rows[0]["folder_name"] == "230101"
    assert rows[0]["current_pkl_count"] == 1
    assert rows[0]["moved_bad_pkl_count"] == 0
    assert summary["num_missing_folders"] == 0


def test_analyze_folder_coverage_tracks_parse_and_num_joint_match_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    ref_file = input_dir / "210531" / "ref.bvh"
    good_file = input_dir / "210531" / "good.bvh"
    parse_bad = input_dir / "210707" / "parse_bad.bvh"
    mismatch = input_dir / "210707" / "mismatch.bvh"

    _write_text(ref_file, "123456789")
    _write_text(good_file, "1234")
    _write_text(parse_bad, "12")
    _write_text(mismatch, "1")

    class FakeSkeleton:
        def __init__(self, num_joints: int):
            self.num_joints = num_joints

    class FakeAnimation:
        def __init__(self, skeleton: FakeSkeleton):
            self.skeleton = skeleton
            self.num_frames = 10
            self.sample_rate = 120.0

    def fake_load_bvh(path: str, input_skeleton=None, source_type: str = "soma"):
        name = Path(path).name
        if name == "parse_bad.bvh":
            raise ValueError("cannot parse")
        if input_skeleton is None:
            skeleton = FakeSkeleton(31 if name != "mismatch.bvh" else 28)
            return skeleton, FakeAnimation(skeleton)
        if name == "good.bvh":
            return input_skeleton, FakeAnimation(input_skeleton)
        if name == "mismatch.bvh":
            raise ValueError("joint mismatch")
        return input_skeleton, FakeAnimation(input_skeleton)

    import app.check_bvh_retarget_candidates as mod

    monkeypatch.setattr(mod.bvh_utils, "load_bvh", fake_load_bvh)

    rows, summary = analyze_folder_coverage(
        input_dir=input_dir,
        output_dir=output_dir,
        report_dir=tmp_path / "report",
    )

    assert summary["reference_relative_path"] == "210531/ref.bvh"
    assert summary["reference_num_joints"] == 31
    assert summary["num_parse_failed_bvh"] == 1
    assert summary["num_num_joints_mismatch_bvh"] == 1
    assert summary["num_reference_conform_ok_bvh"] == 2

    row_by_folder = {row["folder_name"]: row for row in rows}
    assert row_by_folder["210531"]["parse_ok_bvh_count"] == 2
    assert row_by_folder["210531"]["num_joints_match_bvh_count"] == 2
    assert row_by_folder["210531"]["reference_conform_ok_bvh_count"] == 2
    assert row_by_folder["210707"]["parse_failed_bvh_count"] == 1
    assert row_by_folder["210707"]["num_joints_mismatch_bvh_count"] == 1
