# BVH Retarget Candidate Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a batch preflight checker for the SOMA BVH dataset that reports which files are discovered by the converter and which ones fail parsing or reference-skeleton compatibility before retargeting.

**Architecture:** Add a standalone script under `app/` that mirrors the converter's file discovery order, selects the same reference BVH, then analyzes each candidate and writes a summary CSV/JSON. Add focused tests that monkeypatch `load_bvh` so we can verify ordering and failure classification without depending on large real BVH fixtures.

**Tech Stack:** Python, `argparse`, `csv`, `json`, `pathlib`, `pytest`

---

### Task 1: Add failing tests for discovery and classification

**Files:**
- Create: `tests/test_check_bvh_retarget_candidates.py`
- Test: `tests/test_check_bvh_retarget_candidates.py`

**Step 1: Write the failing test**

Cover:
- discovery uses the same extension and file-size-desc order as the converter
- rows classify parse failures and reference-compatibility failures
- summary files are written

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_check_bvh_retarget_candidates.py`
Expected: FAIL because the module does not exist yet.

**Step 3: Write minimal implementation**

Create a script module with:
- file discovery
- per-file analysis
- summary writing

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_check_bvh_retarget_candidates.py`
Expected: PASS

### Task 2: Verify on real dataset

**Files:**
- Create: `app/check_bvh_retarget_candidates.py`

**Step 1: Dry-run the checker on the real dataset**

Run:

```bash
uv run python app/check_bvh_retarget_candidates.py --input-dir /home/humanoid/rgz_work/soma_dataset/soma_uniform/bvh
```

Expected:
- summary files created
- reference BVH path reported
- counts for discovered / eligible / failed files visible

**Step 2: Report usage**

Document the final command for checking the full dataset.
