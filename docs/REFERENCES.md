# References

No code was copied; adapted ideas are named in comments.

1. PyTorch `torch.compile` and TorchInductor documentation; `torch._inductor.utils.run_and_get_code`
   (used to capture Inductor's generated code).
2. J. Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode
   Transformation and Graph Compilation", ASPLOS 2024.
3. JAX documentation: `jax.jit`, ahead-of-time lowering (`.lower()`, `.compile()`,
   `Compiled.as_text()`), `jax_default_matmul_precision`.
4. OpenXLA / XLA documentation: HLO, fusion kinds (kLoop, kInput, kOutput), custom calls.
5. P. Tillet, H. T. Kung, D. Cox, "Triton: An Intermediate Language and Compiler for Tiled
   Neural Network Computations", MAPL 2019.
6. Triton tutorials, "Fused Softmax" — the one-row-per-program idea is adapted from it;
   the looped online-softmax variant and all code here are written for this study.
7. M. Milakov, N. Gimelshein, "Online normalizer calculation for softmax", 2018,
   arXiv:1805.02867 — the running-max/rescaled-sum recurrence in the looped kernel.
8. `triton.testing.do_bench` documentation (CUDA-event-based timing).
