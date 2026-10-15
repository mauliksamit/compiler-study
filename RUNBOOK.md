# RUNBOOK — running the notebooks

Both notebooks need a Linux machine with an NVIDIA GPU (Triton and JAX's CUDA
build do not run on native Windows). Free options: Google Colab or Kaggle.
Use your own account; no credentials go into this repository.

## Google Colab

1. Open https://colab.research.google.com → File → Upload notebook →
   `notebooks/01_block_eager_compile_xla.ipynb`.
2. Runtime → Change runtime type → **GPU** (any; T4 is the free one) → Save.
3. Runtime → **Run all**.
4. Check the first code cell's printout: `jax_devices` must list a CUDA/GPU
   device. If it shows only CPU, uncomment the `pip install -q -U "jax[cuda12]"`
   line in the second cell, run it, then Runtime → Restart session → Run all.
5. The last cell zips `/content/compiler-study-results` and starts a browser
   download of `compiler-study-results.zip`. If the download does not start,
   open the Files panel (folder icon, left), right-click the zip → Download.
6. Repeat for `notebooks/02_triton_fused_softmax.ipynb` in a new session (same
   output folder; its zip includes both notebooks' files if run in the same
   runtime, otherwise download each zip).

## Kaggle

1. https://www.kaggle.com/code → New Notebook → File → Import Notebook → upload
   the `.ipynb`.
2. Settings (right panel) → Accelerator → **GPU T4 x2** or **P100**; Internet → On
   (only needed if you must install `jax[cuda12]`).
3. Run All. Outputs land in `/kaggle/working/compiler-study-results` and the
   zip in `/kaggle/working/`; download from the **Output** tab (or Save Version →
   output files).

## Bring the results back

```bash
cd compiler-study
python scripts/ingest_results.py ~/Downloads/compiler-study-results.zip   # each zip, one at a time
python -m pytest -q
python scripts/ingest_results.py --check
git add results README.md docs/FACTS.md && git commit
```

Then write the two "Findings"/"Explanation" sections in `docs/WRITEUP.md` from the
dumps and tables only (every statement should point at a line in
`results/nb1_inductor_output_code.py`, `results/nb1_xla_optimized_hlo.txt`, or a
row in `results/nb2_bench.json`). Until then every result is NOT RUN.

## What to check before trusting a run

- `nb1_env.json` / `nb2_env.json` name the GPU, driver and library versions.
- `nb2_correctness.json`: all cases pass (the notebook stops before
  benchmarking if any fail).
- `nb2_drift.json`: cells that moved more than 5% on the repeat are flagged;
  rerun the benchmark cell if many are.
- Colab/Kaggle GPUs are shared and clocks vary; numbers compare
  implementations within one session, not across sessions or machines.
