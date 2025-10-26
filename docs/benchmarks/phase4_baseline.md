# Phase 4 Baseline Metrics (Pre-Optimization)

These measurements capture the pre-Phase 5 performance of the DiT360 pipeline. They
serve as a reference point for evaluating optimization gains.

> **Note**: Populate the tables below using your local hardware after running the
> traditional Phase 4 workflow (flow-match scheduler, fp16, no tiling).

## Test Environment
- **GPU**: _e.g., RTX 4090 24GB_
- **Driver / CUDA**: _e.g., 551.76 / CUDA 12.3_
- **PyTorch**: _e.g., 2.2.0+cu121_
- **ComfyUI Build**: _commit hash / date_
- **DiT360 Commit**: _commit hash_

## Baseline Workflow Settings
- Resolution: 2048 × 1024 (2:1)
- Steps: 50
- CFG: 3.0
- Precision: fp16
- Scheduler: flow_match (linear)
- Attention: eager (default)
- VAE: no tiling

## Metrics
| Metric | Value | Notes |
| --- | --- | --- |
| Wall-clock Time | _mm:ss_ | Measured from sampler start to decode end |
| Peak VRAM | _GB_ | Use the log_memory_stats option in sampler |
| Final VRAM | _GB_ | Post-generation allocation |
| GPU Utilization (avg) | _%_ | Optional (nvidia-smi) |
| CPU Utilization (avg) | _%_ | Optional |

## Observations
- _List any qualitative notes (e.g., edge artefacts, prompt adherence)._ 

## Attachments
- _Add screenshots, command outputs, or profiler traces if available._
