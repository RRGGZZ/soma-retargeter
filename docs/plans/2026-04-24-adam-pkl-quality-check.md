# Adam PKL Quality Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a batch quality-check script for `assets/motions/adam_pkl` that flags bad samples, reports metrics, and can move bad `.pkl` files into a separate directory.

**Architecture:** Add a standalone script under `app/` that recursively scans PKL files, computes per-motion metrics from `fps` and `dof_pos`, classifies bad samples, writes `summary.csv/json`, and optionally moves bad files while preserving relative paths. Add focused tests that construct tiny PKL fixtures with `joblib` and verify scanning, classification, summary writing, and moving behavior.

**Tech Stack:** Python, `argparse`, `joblib`, `numpy`, `csv`, `json`, `pytest`

---

### Task 1: Add failing tests for PKL quality classification

**Files:**
- Create: `tests/test_check_adam_pkl_dataset.py`
- Test: `tests/test_check_adam_pkl_dataset.py`

**Step 1: Write the failing test**

Add tests covering:
- `dof_vel_abs_max > 30` marks a motion as bad
- batch scan writes summary files
- `--move-bad` behavior moves only bad samples and preserves relative paths

**Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest -q tests/test_check_adam_pkl_dataset.py`
Expected: FAIL because the target module does not exist yet.

**Step 3: Write minimal implementation**

Create the script module with just enough public functions for the tests:
- motion loading
- DOF velocity metric computation
- classification
- batch scan
- summary writing
- moving bad samples

**Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest -q tests/test_check_adam_pkl_dataset.py`
Expected: PASS

### Task 2: Implement the batch CLI script

**Files:**
- Create: `app/check_adam_pkl_dataset.py`
- Modify: `tests/test_check_adam_pkl_dataset.py`

**Step 1: Add CLI coverage expectations**

Add tests for:
- recursive file discovery
- exclusion of `_quality_check` output tree
- output directory defaults and summary counts

**Step 2: Run test to verify it fails**

Run: `uv run --with pytest python -m pytest -q tests/test_check_adam_pkl_dataset.py`
Expected: FAIL with missing CLI/data-handling behavior.

**Step 3: Write minimal implementation**

Implement:
- `QualityThresholds`
- recursive PKL discovery
- per-file metric extraction
- bad-sample classification
- summary CSV/JSON/errors JSON writing
- optional moving into `bad_samples/`
- CLI arguments for dataset root, output dir, threshold, and `--move-bad`

**Step 4: Run test to verify it passes**

Run: `uv run --with pytest python -m pytest -q tests/test_check_adam_pkl_dataset.py`
Expected: PASS

### Task 3: Verify on a real dataset sample

**Files:**
- Run only: `app/check_adam_pkl_dataset.py`

**Step 1: Dry-run the checker on the real dataset**

Run:

```bash
uv run python app/check_adam_pkl_dataset.py --input-dir assets/motions/adam_pkl --output-dir /tmp/adam_pkl_quality_check
```

Expected:
- summary files created
- no source files moved
- bad samples flagged in the report

**Step 2: Verify move mode on a temporary fixture dataset**

Run the pytest suite again to confirm move behavior in isolation.

**Step 3: Report usage**

Document the final command for:
- report-only mode
- move-bad mode
