## [2026-04-25] Task 1: Device + Extension Validation

### Device Pinning
- Warp 1.12.0 fully supports "cuda:0" and "cuda:1" strings directly in wp.ScopedDevice()
- The host has 8x RTX 4090 GPUs (cuda:0 through cuda:7)
- Use runtime_device: "cuda:0" / "cuda:1" directly in the JSON configs — NO CUDA_VISIBLE_DEVICES needed
- _resolve_runtime_settings() in bvh_to_csv_converter.py:33-47 reads from config["runtime_device"] first

### Source Extension
- retarget_source: "soma" maps to source_ext = "*.bvh" (the else branch at line 606)
- Dataset has 142220 .bvh files under /mnt/data1/rgz/soma_uniform/bvh
- Zero .npz files — all data is .bvh format

### Shard Split
- Alternating parity sharding: shard 0 gets ~71110 files, shard 1 gets ~71110 files
- Total: 142220 files, even split

### Environment
- uv run python works correctly in /mnt/data1/rgz/soma-retargeter
- Environment auto-bootstraps with all deps via uv

## [2026-04-25] Task 2: Config Contract + Launcher Layout

### Config File Names
- gpu0: assets/gpu0_bvh_to_csv_converter_config.json (runtime_device: cuda:0, shard_index: 0)
- gpu1: assets/gpu1_bvh_to_csv_converter_config.json (runtime_device: cuda:1, shard_index: 1)
- Both: shard_count: 2, import_folder: /mnt/data1/rgz/soma_uniform/bvh,
        export_folder: /mnt/data1/rgz/soma-retargeter/adam_soma_pkl

### Launcher Script Names
- run_gpu0.sh (cd into repo, uv run python ./app/bvh_to_csv_converter.py --config ./assets/gpu0_bvh_to_csv_converter_config.json --viewer null)
- run_gpu1.sh (same pattern, gpu1 config)

### Shard Filter Logic (to insert at line 614, before batching)
  shard_index = int(config.get("shard_index", -1))
  shard_count = int(config.get("shard_count", 1))
  if shard_count > 1 and shard_index >= 0:
      motion_files = [f for i, f in enumerate(motion_files) if i % shard_count == shard_index]
      print(f"[INFO]: Shard {shard_index}/{shard_count}: {len(motion_files)} files selected.")
      if len(motion_files) == 0:
          print("[INFO]: Shard is empty. Nothing to process.")
          return

### Export Safety Confirmed
- rel_path = Path(batch[i]).relative_to(import_path) — import_path never changes
- Shard filtering is purely in-memory list slicing, does not affect import_path
- Shared export_folder safe because disjoint input => disjoint output paths
