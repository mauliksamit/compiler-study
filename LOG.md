# LOG.md — engineering log (plain language)

## 2026-10-15 — Notebooks written; NOT RUN (needs a Linux GPU)
- Wrote two notebooks (generated from `scripts/build_notebooks.py`): a
  transformer block in PyTorch eager vs torch.compile vs JAX jit with compile
  time, steady-state time, float64 numerics and saved compiler dumps; and a
  from-scratch Triton row softmax with a correctness gate and a benchmark.
- Local checks only: notebook 1 executes end to end on Windows with the
  torch.compile and JAX cells recording NOT RUN (TritonMissing; no jax) —
  smoke test, not results. Notebook 2 compiles. The looped kernel's algorithm
  is verified by a NumPy emulation (25 cases, including a fully -inf block,
  which needed a NaN guard found while writing the emulation).
- Next: run both notebooks per RUNBOOK.md and ingest the zips.
