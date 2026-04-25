# SOMA Retargeter Dual-GPU Sharding Plan

## TL;DR

> **Quick Summary**: Add explicit two-shard batch execution to `soma-retargeter` so the dataset under `/mnt/data1/rgz/soma_uniform/bvh` is processed in parallel without overlap, using alternating assignment over the converter's native size-descending file order.
>
> **Deliverables**:
> - Config-driven shard selection in the batch converter path
> - Two launcher scripts in `/mnt/data1/rgz/soma-retargeter`
> - Two dedicated configs for `cuda:0` and `cuda:1`
> - Verification flow proving zero overlap and complete coverage
>
> **Estimated Effort**: Short
> **Parallel Execution**: YES - 2 main waves + final verification
> **Critical Path**: Device validation/config shape → shard filtering → launcher/config wiring → dual-run verification

---

## Context

### Original Request
Use `/mnt/data1/rgz/soma-retargeter` to retarget the dataset under `/mnt/data1/rgz/soma_uniform/bvh`, and create two scripts in `/mnt/data1/rgz/soma-retargeter` so one script runs on `cuda:0` and the other on `cuda:1` in parallel.

### Interview Summary
**Key Discussions**:
- Preserve the converter's native ordering logic rather than introducing a new sort order.
- Avoid any possibility that the two scripts process the same file.
- Output should be written under `/mnt/data1/rgz/soma-retargeter/adam_soma_pkl`.
- Preferred balancing strategy is alternating assignment over the native sorted list: indices `0,2,4...` to `cuda:0`, `1,3,5...` to `cuda:1`.

**Research Findings**:
- Headless entrypoint from README: `python ./app/bvh_to_csv_converter.py --config ./assets/default_bvh_to_csv_converter_config.json --viewer null`.
- `app/bvh_to_csv_converter.py:606-614` recursively discovers source files and sorts them by file size descending before batching.
- `app/bvh_to_csv_converter.py:675-688` mirrors input-relative paths under `export_folder`, so shared export root is safe only if shards are disjoint.
- `app/bvh_to_csv_converter.py:722-727` resolves `runtime_device` and runs inside `wp.ScopedDevice(runtime_device)`.

### Metis Review
**Identified Gaps** (addressed in this plan):
- Need explicit guardrail that shard logic must preserve current sort order.
- Need explicit acceptance criteria for disjointness, completeness, and device pinning.
- Need to handle odd file counts and empty-shard edge cases gracefully.
- Need to validate actual source extension behavior using the converter's existing `retarget_source` logic rather than hardcoding assumptions.

---

## Work Objectives

### Core Objective
Introduce a minimal, config-driven dual-GPU sharding workflow in `soma-retargeter` so two independent launcher scripts can process one shared dataset in parallel with zero overlap and shared output under `adam_soma_pkl`.

### Concrete Deliverables
- Batch converter support for selecting one shard out of two after native discovery and size-desc sort.
- Two config files that pin runtime device and shard identity.
- Two launcher scripts in `/mnt/data1/rgz/soma-retargeter` for direct parallel execution.
- Logging or dry-run-style evidence sufficient to prove shard disjointness and coverage.

### Definition of Done
- [ ] Both scripts launch independently and print distinct runtime devices.
- [ ] Shard membership is disjoint and together covers the full discovered file list.
- [ ] Running both scripts in parallel writes outputs under `/mnt/data1/rgz/soma-retargeter/adam_soma_pkl` without collisions.
- [ ] Edge cases like odd file counts and empty shard are handled gracefully.

### Must Have
- Preserve current discovery behavior: recursive scan, then size-descending sort.
- Apply shard selection before batching.
- Keep export path derivation relative to the original import root.
- Keep scripts independently runnable.

### Must NOT Have (Guardrails)
- No change to export format or directory structure.
- No cross-process coordinator, IPC, or shared progress system.
- No hardcoded file extension outside the existing converter logic.
- No overlap in file assignment between shard 0 and shard 1.
- No change to sort order, except filtering by shard after the sort.

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - all verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: Tests-after
- **Framework**: pytest
- **If TDD**: Not selected for this work

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved under `.sisyphus/evidence/`.

- **CLI / scripts**: Use `bash` and `interactive_bash` only if needed
- **Python/module validation**: Use `bash` with `python`/`uv run python`
- **Repo checks**: Use tests plus captured stdout/stderr and generated manifest files

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately - foundations):
├── Task 1: Validate runtime device syntax and extension behavior [quick]
├── Task 2: Design shard config contract and launcher layout [quick]
└── Task 3: Add automated tests for shard selection and edge cases [quick]

Wave 2 (After Wave 1 - implementation):
├── Task 4: Implement config-driven shard filtering in batch discovery path [unspecified-high]
├── Task 5: Add two GPU-specific config files [quick]
└── Task 6: Add two launcher scripts for cuda:0 and cuda:1 [quick]

Wave 3 (After Wave 2 - integration validation):
├── Task 7: Verify disjointness, completeness, and shared export safety [unspecified-high]
└── Task 8: Verify direct usability and document invocation expectations in-script [quick]

Wave FINAL (After ALL tasks):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA execution (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: Task 1 → Task 4 → Task 5/6 → Task 7 → F1-F4
Parallel Speedup: ~40% faster than purely sequential
Max Concurrent: 3

### Dependency Matrix

- **1**: - → 4, 5, 6, 7
- **2**: - → 4, 5, 6
- **3**: - → 4, 7
- **4**: 1, 2, 3 → 5, 6, 7, 8
- **5**: 1, 2, 4 → 7, 8
- **6**: 1, 2, 4 → 7, 8
- **7**: 3, 4, 5, 6 → F1-F4
- **8**: 4, 5, 6 → F1, F4

### Agent Dispatch Summary

- **Wave 1**: 3 tasks - T1 `quick`, T2 `quick`, T3 `quick`
- **Wave 2**: 3 tasks - T4 `unspecified-high`, T5 `quick`, T6 `quick`
- **Wave 3**: 2 tasks - T7 `unspecified-high`, T8 `quick`
- **FINAL**: 4 tasks - F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [x] 1. Validate runtime device syntax and source extension behavior

  **What to do**:
  - Verify whether Warp accepts `cuda:0` and `cuda:1` directly in the same way the converter's `wp.ScopedDevice(runtime_device)` expects.
  - Verify whether the actual runtime config/path used for this dataset resolves to `*.bvh` or `*.npz` discovery, using the existing `retarget_source` logic already in `app/bvh_to_csv_converter.py`.
  - Record the chosen device-pinning strategy for the implementation tasks: config `runtime_device`, CLI device override, or environment-level fallback if direct device strings fail.

  **Must NOT do**:
  - Do not change converter logic in this task.
  - Do not hardcode `.bvh` if the code path derives extension from config.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: focused validation of a small number of runtime assumptions.
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - `playwright`: no browser interaction involved.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: 4, 5, 6, 7
  - **Blocked By**: None

  **References**:
  - `app/bvh_to_csv_converter.py:33-47` - `_resolve_runtime_settings()` determines whether config or fallback args provide the runtime device.
  - `app/bvh_to_csv_converter.py:606-614` - source extension selection and discovery happen here; this is the authoritative behavior to preserve.
  - `app/bvh_to_csv_converter.py:722-727` - runtime device enters `wp.ScopedDevice(runtime_device)` here.
  - `assets/default_bvh_to_csv_converter_config.json:1-14` - current baseline config values and retarget source/target expectations.

  **Acceptance Criteria**:
  - [ ] Evidence captured showing whether `cuda:0` and `cuda:1` are valid runtime device strings for this repo's execution path.
  - [ ] Evidence captured showing whether source discovery for the intended run resolves to `*.bvh` or `*.npz`.
  - [ ] Chosen pinning strategy is explicit and referenced by later tasks.

  **QA Scenarios**:
  ```
  Scenario: Validate direct device string support
    Tool: Bash (python)
    Preconditions: Project environment installed and importable
    Steps:
      1. Run a minimal Python snippet in /mnt/data1/rgz/soma-retargeter that imports warp and enters wp.ScopedDevice("cuda:0").
      2. Repeat for wp.ScopedDevice("cuda:1").
      3. Capture stdout/stderr and exit status.
    Expected Result: Both snippets either succeed cleanly or fail with a clear error that determines fallback strategy.
    Failure Indicators: Runtime import/device exceptions without a documented fallback decision.
    Evidence: .sisyphus/evidence/task-1-device-validation.txt

  Scenario: Validate source extension path
    Tool: Bash (python)
    Preconditions: Access to current config JSON
    Steps:
      1. Load assets/default_bvh_to_csv_converter_config.json.
      2. Print retarget_source and the computed source_ext using the same rule as line 606.
      3. Count matching files under /mnt/data1/rgz/soma_uniform/bvh for that extension.
    Expected Result: The actual source extension for this workflow is unambiguous and matches non-zero files or triggers a needed config adjustment.
    Failure Indicators: Zero matches with no follow-up decision recorded.
    Evidence: .sisyphus/evidence/task-1-source-extension.txt
  ```

  **Evidence to Capture**:
  - [ ] Device validation output
  - [ ] Source extension resolution output

  **Commit**: NO

- [x] 2. Design shard config contract and launcher layout

  **What to do**:
  - Define the minimal config keys needed for deterministic sharding, such as `shard_index` and `shard_count`.
  - Define where the two new config files will live and how they will differ from the default config.
  - Define the two launcher script names and the exact command each script should run.
  - Ensure the design preserves the original import root so relative export path mirroring still works.

  **Must NOT do**:
  - Do not introduce a coordinator script.
  - Do not redesign the overall config system beyond the minimum keys needed.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: small design task around config/script structure.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `assets/default_bvh_to_csv_converter_config.json:1-14` - baseline schema to extend minimally.
  - `README.md:100-115` - expected headless batch invocation pattern and PKL-only semantics.
  - `app/bvh_to_csv_converter.py:583-614` - sharding must slot in after discovery/sort and before batching.

  **Acceptance Criteria**:
  - [ ] Config contract documented in code/comments/tests and includes shard identity plus device selection.
  - [ ] Launcher script naming and invocation are fixed and unambiguous.
  - [ ] Shared export root design remains compatible with mirrored relative paths.

  **QA Scenarios**:
  ```
  Scenario: Review config and launcher contract for minimality
    Tool: Bash (python/read)
    Preconditions: Proposed config keys and script names are decided
    Steps:
      1. Read the default config and proposed additions.
      2. Confirm only shard-related keys and GPU-specific runtime_device differ across new configs.
      3. Confirm each launcher is a single direct invocation path with no coordinator logic.
    Expected Result: Minimal config delta and two independently runnable scripts.
    Failure Indicators: Extra unrelated config changes or launcher dependency on another script/process.
    Evidence: .sisyphus/evidence/task-2-contract-review.txt

  Scenario: Validate export-root preservation in the design
    Tool: Bash (python)
    Preconditions: Proposed design documented
    Steps:
      1. Inspect the planned import root and export root handling.
      2. Confirm shard filtering does not rewrite import_folder to a temporary subset folder.
      3. Confirm relative-path mirroring remains based on the original import root.
    Expected Result: Export path derivation remains unchanged for each processed file.
    Evidence: .sisyphus/evidence/task-2-export-root.txt
  ```

  **Evidence to Capture**:
  - [ ] Config contract notes
  - [ ] Launcher naming/command notes

  **Commit**: NO

- [x] 3. Add automated tests for shard selection and edge cases

  **What to do**:
  - Add or extend tests to cover shard filtering over a deterministic sorted file list.
  - Cover alternating assignment semantics for `shard_count=2`.
  - Cover odd file counts and empty-shard behavior.
  - Cover preservation of sort order within each shard after filtering.

  **Must NOT do**:
  - Do not require real GPU access for unit-level shard tests.
  - Do not couple these tests to full retarget execution.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: targeted test additions in a small number of files.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: 4, 7
  - **Blocked By**: None

  **References**:
  - `tests/test_bvh_to_csv_converter_exports.py` - likely home for export/converter-adjacent tests.
  - `app/bvh_to_csv_converter.py:606-614` - the ordering and batching behavior tests must preserve.
  - `tests/test_check_bvh_retarget_candidates.py` - candidate reference for file-list-oriented tests.

  **Acceptance Criteria**:
  - [ ] Tests prove shard 0 and shard 1 are disjoint for the same sorted input list.
  - [ ] Tests prove union of shards equals the full sorted list.
  - [ ] Tests cover odd counts and zero-length shard case gracefully.

  **QA Scenarios**:
  ```
  Scenario: Run targeted shard tests
    Tool: Bash (pytest)
    Preconditions: New/updated tests committed locally
    Steps:
      1. Run the targeted pytest file(s) covering shard selection behavior.
      2. Capture pass/fail output.
      3. Verify the alternating assignment assertions appear in the tested cases.
    Expected Result: Targeted shard tests pass with 0 failures.
    Failure Indicators: Any failed assertion around overlap, order preservation, or odd-count handling.
    Evidence: .sisyphus/evidence/task-3-pytest.txt

  Scenario: Validate empty-shard edge case
    Tool: Bash (pytest)
    Preconditions: Test fixture representing a one-file dataset exists
    Steps:
      1. Run the test case for total file count = 1 with shard_count = 2.
      2. Confirm shard 1 does not crash the converter path and returns a handled no-op/result.
    Expected Result: Empty shard is handled gracefully and documented by the tests.
    Evidence: .sisyphus/evidence/task-3-empty-shard.txt
  ```

  **Evidence to Capture**:
  - [ ] Targeted pytest output
  - [ ] Empty-shard test output

  **Commit**: NO

- [x] 4. Implement config-driven shard filtering in batch discovery path

  **What to do**:
  - Add minimal shard-selection logic in the batch converter path immediately after the full source file list is discovered and sorted, but before batching is constructed.
  - Use config-driven keys such as `shard_index` and `shard_count` to select the process's subset.
  - Preserve the current discovery rule and size-desc sort order exactly, then filter by alternating index for the two-shard case.
  - Ensure the implementation logs enough detail to prove shard membership count and selected shard identity.
  - Handle empty-shard case gracefully with an informational exit rather than a hard failure.

  **Must NOT do**:
  - Do not rewrite or refactor the core export loop.
  - Do not change `rel_path` export behavior.
  - Do not move shard filtering after batches are already built.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: small but correctness-sensitive production code change in the converter path.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential in Wave 2
  - **Blocks**: 5, 6, 7, 8
  - **Blocked By**: 1, 2, 3

  **References**:
  - `app/bvh_to_csv_converter.py:583-614` - exact insertion zone for shard filtering.
  - `app/bvh_to_csv_converter.py:647-697` - export loop that must remain intact.
  - `app/bvh_to_csv_converter.py:606-613` - canonical discovery and sorting behavior to preserve.
  - `tests/test_bvh_to_csv_converter_exports.py` - use corresponding test expectations as regression guard.

  **Acceptance Criteria**:
  - [ ] Same discovered full list + same sort order is preserved before filtering.
  - [ ] For `shard_count=2`, shard 0 receives indices `0,2,4...` and shard 1 receives `1,3,5...` from the sorted full list.
  - [ ] Empty-shard case exits cleanly with a clear info message rather than an error.
  - [ ] Existing non-sharded behavior remains available when no shard config is provided.

  **QA Scenarios**:
  ```
  Scenario: Verify alternating shard filtering against a controlled file list
    Tool: Bash (pytest or python)
    Preconditions: Converter supports shard_index/shard_count in config
    Steps:
      1. Create or use a deterministic fixture list with known sizes and sorted order.
      2. Run shard selection for shard 0 and shard 1 separately.
      3. Assert shard 0 contains sorted indices 0,2,4... and shard 1 contains 1,3,5....
    Expected Result: Alternating assignment exactly matches the sorted full list parity.
    Failure Indicators: Any reordered items, overlap, or missing items.
    Evidence: .sisyphus/evidence/task-4-alternating-shard.txt

  Scenario: Verify non-sharded backward compatibility
    Tool: Bash (python)
    Preconditions: No shard keys in config or shard_count absent
    Steps:
      1. Run the relevant discovery/filtering path without shard config.
      2. Compare resulting file list length with the original full discovered list.
    Expected Result: Full list is preserved unchanged when sharding is not enabled.
    Evidence: .sisyphus/evidence/task-4-backward-compat.txt
  ```

  **Evidence to Capture**:
  - [ ] Alternating shard proof
  - [ ] Backward-compatibility proof

  **Commit**: NO

- [x] 5. Add two GPU-specific config files

  **What to do**:
  - Create two dedicated config files derived from the default batch config.
  - Set shared import/export roots for the target dataset and output directory.
  - Set shard identity so one config selects shard 0 of 2 and the other selects shard 1 of 2.
  - Set device selection so one config targets `cuda:0` and the other targets `cuda:1`, unless Task 1 proves a different safe pinning method is required.

  **Must NOT do**:
  - Do not change unrelated retarget parameters.
  - Do not create divergent batch sizes or execution modes unless device validation forces it.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: small config file additions based on a fixed template.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 6)
  - **Blocks**: 7, 8
  - **Blocked By**: 1, 2, 4

  **References**:
  - `assets/default_bvh_to_csv_converter_config.json:1-14` - template for the two new configs.
  - `app/bvh_to_csv_converter.py:33-47` - runtime device resolution behavior.
  - `app/bvh_to_csv_converter.py:583-614` - shard config keys must be consumed here.

  **Acceptance Criteria**:
  - [ ] Two configs exist and differ only where required: dataset path, export path, device pinning, shard identity.
  - [ ] Both configs point at `/mnt/data1/rgz/soma_uniform/bvh` and `/mnt/data1/rgz/soma-retargeter/adam_soma_pkl`.
  - [ ] Both configs are valid JSON and load successfully through existing config loading.

  **QA Scenarios**:
  ```
  Scenario: Validate config file correctness
    Tool: Bash (python)
    Preconditions: Two new config files exist
    Steps:
      1. Load each config via the repo's JSON loader or Python json module.
      2. Print runtime_device, shard_index, shard_count, import_folder, and export_folder.
      3. Assert shard indices are 0 and 1, shard_count is 2, and import/export roots match the requested paths.
    Expected Result: Both config files load and contain the expected minimal deltas.
    Failure Indicators: Missing keys, invalid JSON, or path/device mismatches.
    Evidence: .sisyphus/evidence/task-5-config-validation.txt

  Scenario: Verify config parity outside required differences
    Tool: Bash (python diff)
    Preconditions: Both configs created from same baseline
    Steps:
      1. Diff the two configs structurally.
      2. Confirm only shard/device-specific fields differ where expected.
    Expected Result: No unintended divergence in retarget settings.
    Evidence: .sisyphus/evidence/task-5-config-diff.txt
  ```

  **Evidence to Capture**:
  - [ ] Config validation output
  - [ ] Config diff output

  **Commit**: NO

- [x] 6. Add two launcher scripts for cuda:0 and cuda:1

  **What to do**:
  - Add two directly runnable scripts under `/mnt/data1/rgz/soma-retargeter`.
  - Each script should invoke the headless converter with its corresponding config.
  - If Task 1 determines direct `runtime_device` strings are insufficient, apply the validated fallback pinning approach in the script while keeping behavior explicit and isolated.
  - Ensure script output makes it obvious which shard and GPU are being launched.

  **Must NOT do**:
  - Do not make one script call the other.
  - Do not add orchestration logic beyond launching one configured process.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: focused launcher file creation with minimal shell logic.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 5)
  - **Blocks**: 7, 8
  - **Blocked By**: 1, 2, 4

  **References**:
  - `README.md:100-110` - canonical headless invocation pattern.
  - `app/bvh_to_csv_converter.py:722-727` - script behavior should surface this runtime device selection in logs.
  - New config files from Task 5 - launchers should be thin wrappers around them.

  **Acceptance Criteria**:
  - [ ] Two scripts exist and are independently runnable.
  - [ ] One script launches shard 0 / GPU 0 and the other launches shard 1 / GPU 1.
  - [ ] Script stdout clearly shows which config/device is being used.

  **QA Scenarios**:
  ```
  Scenario: Verify each launcher invokes the correct config
    Tool: Bash
    Preconditions: Both scripts created and executable
    Steps:
      1. Run each launcher in a non-destructive validation mode or with a small controlled dataset.
      2. Capture stdout/stderr.
      3. Assert output references the expected config and runtime device.
    Expected Result: Launcher A reports shard 0 / cuda:0, launcher B reports shard 1 / cuda:1.
    Failure Indicators: Missing script execute bit, wrong config path, or wrong device output.
    Evidence: .sisyphus/evidence/task-6-launcher-validation.txt

  Scenario: Verify scripts can be launched concurrently
    Tool: Bash
    Preconditions: Small controlled dataset available
    Steps:
      1. Start launcher A in background.
      2. Start launcher B in background.
      3. Wait for both to complete or reach stable processing state.
    Expected Result: Both processes start without immediate collision or startup failure.
    Evidence: .sisyphus/evidence/task-6-concurrent-start.txt
  ```

  **Evidence to Capture**:
  - [ ] Launcher validation output
  - [ ] Concurrent start output

  **Commit**: NO

- [x] 7. Verify disjointness, completeness, and shared export safety

  **What to do**:
  - Run both shard paths against a controlled subset or manifestable sample of the dataset.
  - Capture the exact file list selected by shard 0 and shard 1.
  - Prove the intersection is empty and the union equals the full sorted list for the same input set.
  - Verify the resulting `.pkl` outputs under `/mnt/data1/rgz/soma-retargeter/adam_soma_pkl` match the expected count for the tested input set.
  - Verify shared export root behavior does not introduce collisions when both shards write concurrently.

  **Must NOT do**:
  - Do not rely on eyeballing file names manually.
  - Do not mark success without an explicit count/intersection proof.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: integration validation with multiple assertions and artifact checks.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential in Wave 3
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: 3, 4, 5, 6

  **References**:
  - `app/bvh_to_csv_converter.py:607-614` - full discovered list and sort behavior used as the ground truth.
  - `app/bvh_to_csv_converter.py:675-688` - export-path derivation that makes shared output safe only for disjoint shards.
  - New shard tests from Task 3 - unit-level proof to complement integration-level proof.
  - New configs and launchers from Tasks 5 and 6 - integration entrypoints to validate.

  **Acceptance Criteria**:
  - [ ] Captured shard-0 and shard-1 file lists have zero intersection.
  - [ ] Union of shard-0 and shard-1 lists equals the full discovered sorted list for the tested dataset.
  - [ ] Output `.pkl` count matches expected processed-file count for the tested run.
  - [ ] Parallel execution into the shared export root completes without file-collision symptoms.

  **QA Scenarios**:
  ```
  Scenario: Prove zero-overlap shard membership
    Tool: Bash (python)
    Preconditions: Both shard configs and filter logic implemented
    Steps:
      1. Generate the full sorted file list for the chosen test input set using the same discovery and sort rule as the converter.
      2. Generate shard-0 and shard-1 selected file lists using the implemented shard logic.
      3. Compute set intersection and union.
      4. Assert intersection size is 0 and union size equals full-list size.
    Expected Result: Formal proof of zero overlap and full coverage.
    Failure Indicators: Any shared path between shards or missing path from combined shards.
    Evidence: .sisyphus/evidence/task-7-disjointness-proof.txt

  Scenario: Validate shared export root under concurrent execution
    Tool: Bash
    Preconditions: Small controlled dataset and clean export directory
    Steps:
      1. Remove prior test outputs from the controlled export subtree.
      2. Start both launchers concurrently.
      3. Wait for both to complete.
      4. Count generated `.pkl` files and inspect for duplicate-write or overwrite errors in logs.
    Expected Result: Expected `.pkl` count is reached with no collision-related errors.
    Failure Indicators: Missing outputs, duplicate-write errors, overwrite warnings, or count mismatch.
    Evidence: .sisyphus/evidence/task-7-shared-export.txt
  ```

  **Evidence to Capture**:
  - [ ] Disjointness proof output
  - [ ] Shared export validation output

  **Commit**: NO

- [x] 8. Verify direct usability and document invocation expectations in-script

  **What to do**:
  - Ensure each launcher script is self-explanatory enough to run directly from the repo root or via absolute paths as intended.
  - Add lightweight inline usage comments or echoed startup context so the operator can tell which shard/GPU is running without opening the config.
  - Ensure the expected working directory and environment assumptions are explicit.

  **Must NOT do**:
  - Do not create extra documentation files outside the requested scope.
  - Do not add verbose operational docs unrelated to launching the two scripts.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: focused usability polish on a very small number of files.
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: F1, F4
  - **Blocked By**: 4, 5, 6

  **References**:
  - `README.md:100-110` - headless invocation style to match.
  - Launcher scripts from Task 6 - target files for usability expectations.

  **Acceptance Criteria**:
  - [ ] A user can identify which script is for `cuda:0` and which is for `cuda:1` from file name and startup output.
  - [ ] Expected invocation context is explicit in the script itself.
  - [ ] No additional external docs are required to understand the basic launch path.

  **QA Scenarios**:
  ```
  Scenario: Verify launcher self-description
    Tool: Bash
    Preconditions: Launcher scripts exist
    Steps:
      1. Read the first 20 lines of each launcher script.
      2. Confirm file name, comments, or echoed output identify GPU/shard clearly.
      3. Run each script once and capture its startup banner/output.
    Expected Result: Script identity and expected invocation context are obvious.
    Failure Indicators: Ambiguous script naming or no visible indication of GPU/shard mapping.
    Evidence: .sisyphus/evidence/task-8-launcher-clarity.txt

  Scenario: Verify no hidden working-directory assumptions
    Tool: Bash
    Preconditions: Launcher scripts exist
    Steps:
      1. Run each script from outside the repo root using an absolute path.
      2. Confirm the script still resolves its config and app entrypoint correctly, or clearly documents the required workdir if not supported.
    Expected Result: Either path-robust launch works, or script behavior clearly states the required working directory.
    Evidence: .sisyphus/evidence/task-8-workdir.txt
  ```

  **Evidence to Capture**:
  - [ ] Launcher clarity output
  - [ ] Working-directory validation output

  **Commit**: YES
  - Message: `feat(retarget): add dual-gpu shard launchers`
  - Files: `app/bvh_to_csv_converter.py`, new config files, new launcher scripts, relevant tests
  - Pre-commit: `pytest <targeted test files>`

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
- [x] F2. **Code Quality Review** — `unspecified-high`
- [x] F3. **Real Manual QA** — `unspecified-high`
- [x] F4. **Scope Fidelity Check** — `deep`
  Confirm only sharding/config/launcher behavior was added, with no spillover refactors.

---

## Commit Strategy

- **1**: `feat(retarget): add dual-gpu shard launch flow` - converter/configs/scripts/tests - run targeted tests before commit

---

## Success Criteria

### Verification Commands
```bash
python ./app/bvh_to_csv_converter.py --config <gpu-config> --viewer null
pytest tests/test_bvh_to_csv_converter_exports.py tests/test_check_bvh_retarget_candidates.py
```

### Final Checklist
- [ ] All "Must Have" conditions met
- [ ] All "Must NOT Have" conditions absent
- [ ] Device pinning verified
- [ ] Zero-overlap shard proof captured
- [ ] Complete coverage proof captured
