# Phase 5: Optimization & Memory Efficiency

This document tracks the optimizations introduced in Phase 5 and provides quick
reference guidance for exercising the new controls.

## Objectives
- Introduce swappable attention backends (eager, xFormers, FlashAttention).
- Support attention slicing for reduced VRAM usage.
- Enable lightweight model variants via optional int8 / int4 quantization.
- Add VAE tiling controls for high-resolution panoramas.
- Offer alternate schedulers (flow-match vs. DDIM) for speed/quality trade-offs.
- Capture baseline metrics to quantify future improvements.

## Key Controls
### DiT360Loader
- attention_backend: choose attention implementation at load time.
- attention_slice_size: set slicing chunk size (0 disables, -1 loader default).
- quantization_mode: none, int8 (torch dynamic), or int4 (bitsandbytes).
- vae_tile_size / vae_tile_overlap: default tiling geometry.
- vae_auto_tile_pixels: auto-tiling threshold (0 disables heuristics).

### DiT360Sampler
- scheduler_type: flow_match (default) or ddim.
- timestep_schedule: linear, quadratic, or cosine (flow-match only).
- scheduler_eta: DDIM sigma parameter.
- attention_backend / attention_slice_size: per-run overrides.
- log_memory_stats: prints allocated/peak VRAM post-generation.

### DiT360Decode
- tiling_mode: auto, always, never.
- tile_size_override, tile_overlap_override, max_tile_pixels_override.

## Usage Tips
1. **Low VRAM**: Load with quantization_mode=int8, sampler attention_slice_size=512,
   decode with tiling_mode=always.
2. **Speed-Focused**: Keep quantization off, use scheduler_type=ddim,
   attention_backend=flash (if available), disable tiling.
3. **Quality-Focused**: Use scheduler_type=flow_match, cosine schedule, enable yaw/cube losses,
   leave tiling on auto to avoid seams.

## Follow-Up Tasks
- Populate docs/benchmarks/phase4_baseline.md with measured metrics.
- Add FlashAttention validation matrix once dependencies are packaged.
- Document recommended combinations in README examples.
- Explore VAE multi-axis tiling (Phase 6 candidate).
