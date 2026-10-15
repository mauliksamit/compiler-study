# Write-up

Status: **NOT RUN.** Both notebooks need a Linux NVIDIA GPU; see `RUNBOOK.md`.

## 1. What each compiler fused (from the dumps only)

NOT RUN. To be written from `results/nb1_fusion_summary.json`,
`results/nb1_inductor_output_code.py` and `results/nb1_xla_optimized_hlo.txt`
after notebook 1 runs: for each of LayerNorm, the QKV/output/MLP GEMMs and
their bias adds, scale + mask + softmax, GELU, and the residual adds — which
kernel it ended up in for Inductor and for XLA, with the kernel/fusion name
from the dump. Then relate that to the steady-state times in the README table.

## 2. Triton fused softmax

NOT RUN. To be written from `results/nb2_bench.json` and
`results/nb2_correctness.json`: correctness first, then where the Triton
kernel matches, beats or loses to `torch.softmax` and to the composed eager
version, and whether the looped path's second read shows up at the widest
rows.
