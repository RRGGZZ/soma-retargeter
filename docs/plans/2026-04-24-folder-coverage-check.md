# Folder Coverage Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the BVH candidate checker with a folder-level coverage checker that compares first-level input folders against first-level output folders and reports coverage.

**Architecture:** Update `app/check_bvh_retarget_candidates.py` to aggregate counts by the first directory under the input root and compare them against `.pkl` outputs in the output root, including optionally moved bad samples under `_quality_check/*/bad_samples`. Add focused tests using tiny fixture directories to verify counting and summary generation.

**Tech Stack:** Python, `argparse`, `csv`, `json`, `pathlib`, `pytest`

---

### Task 1: Add failing tests for folder aggregation

**Files:**
- Modify: `tests/test_check_bvh_retarget_candidates.py`
- Test: `tests/test_check_bvh_retarget_candidates.py`

**Step 1: Write the failing test**

Cover:
- first-level folder aggregation for input `.bvh`
- output `.pkl` counting by first-level folder
- moved bad sample counting from `_quality_check/*/bad_samples`
- summary writing

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_check_bvh_retarget_candidates.py`
Expected: FAIL because the script still performs per-file BVH parsing.

**Step 3: Write minimal implementation**

Implement folder-level aggregation and summary writing only.

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_check_bvh_retarget_candidates.py`
Expected: PASS

### Task 2: Verify on the real dataset layout

**Files:**
- Modify: `app/check_bvh_retarget_candidates.py`

**Step 1: Dry-run on the real directories**

Run:

```bash
uv run python app/check_bvh_retarget_candidates.py \
  --input-dir /home/humanoid/rgz_work/soma_dataset/soma_uniform/bvh \
  --output-dir assets/motions/adam_pkl
```

Expected:
- one row per first-level input folder
- counts for input, current output, moved bad samples, and coverage

**Step 2: Report usage**

Document the final command and the meaning of the key summary fields.
