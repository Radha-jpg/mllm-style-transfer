# MLLM-Grounded Multi-Region Artistic Style Transfer

MSc dissertation project, University of Bath, 2026.

## Structure

- `ip_adapter/` - SD1.5-Inpainting + IP-Adapter pipeline and outputs
- `boundary_ablation/` - boundary coherence experiments (Notebooks 9, 9v2)
- `multidiffusion/outputs/` - MultiDiffusion compositing results

## Key results

| Method | Sky LPIPS | Foreground LPIPS |
|---|---|---|
| NST baseline | 0.0758 | 0.1627 |
| IP-Adapter SD1.5 | 0.0477 | 0.1188 |

Best boundary strategy: attention routing + feathered blend (combined score 0.7279).
