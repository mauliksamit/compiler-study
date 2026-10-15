#!/usr/bin/env python3
"""Build the two study notebooks from the cell sources below.

Keeping the cell sources in a .py file makes them reviewable and diffable;
the .ipynb files in notebooks/ are generated (python scripts/build_notebooks.py).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    return {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }


COMMON_ENV = r'''
# --- Environment record (runs first; written to results/env.json) ---
import json, os, platform, subprocess, sys, time, traceback
OUT = os.environ.get("STUDY_OUT", "/kaggle/working/compiler-study-results" if os.path.isdir("/kaggle") else "/content/compiler-study-results")
os.makedirs(OUT, exist_ok=True)

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"

def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print("wrote", os.path.join(OUT, name))

def version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"NOT AVAILABLE ({type(e).__name__}: {e})"

env = {
    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "platform": platform.platform(), "python": sys.version.split()[0],
    "nvidia_smi": sh("nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,power.limit --format=csv,noheader"),
    "nvcc": sh("nvcc --version | tail -1"),
    "torch": version("torch"), "triton": version("triton"),
    "jax": version("jax"), "jaxlib": version("jaxlib"), "numpy": version("numpy"),
}
try:
    import torch
    env["torch_cuda"] = torch.version.cuda
    env["torch_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
except Exception as e:
    env["torch_cuda"] = f"unavailable: {e}"
try:
    import jax
    env["jax_devices"] = [str(d) for d in jax.devices()]
except Exception as e:
    env["jax_devices"] = f"unavailable: {e}"
print(json.dumps(env, indent=2))
'''

NB1 = [
    ("markdown", r'''
# Notebook 1 — one transformer block: PyTorch eager vs torch.compile (Inductor) vs JAX jit (XLA)

**What this is:** a measured study using existing compilers, not compiler
development. It compiles the same fixed-shape pre-norm transformer block
three ways, checks numerics against a float64 reference, times compile and
steady state separately, and saves the code each compiler generated
(Inductor's Triton/Python output code, XLA's optimized HLO) so the fusion
analysis is read from the dumps — nothing about MLIR or LLVM is claimed
beyond what those dumps show.

**How to run:** Colab or Kaggle, GPU runtime, Run all. Outputs go to
`/content/compiler-study-results` (Colab) or `/kaggle/working/compiler-study-results`
(Kaggle); the last cell zips them for download. See `RUNBOOK.md`.
'''),
    ("code", COMMON_ENV + '\nsave("nb1_env.json", env)\n'),
    ("code", r'''
# Optional: make JAX use the GPU if the runtime lacks a CUDA-enabled jaxlib.
# Uncomment ONLY if the env print above shows jax_devices without "cuda".
# !pip install -q -U "jax[cuda12]"
'''),
    ("code", r'''
# --- Fixed problem definition (shared by all three implementations) ---
import numpy as np
B, T, D, H = 8, 256, 512, 8          # batch, sequence, model width, heads
DH, F = D // H, 4 * D                # head dim, MLP width
SEED = 0
rng = np.random.default_rng(SEED)
def init(*shape, scale):
    return (rng.standard_normal(shape) * scale).astype(np.float32)
W = {
    "ln1_g": np.ones(D, np.float32), "ln1_b": np.zeros(D, np.float32),
    "ln2_g": np.ones(D, np.float32), "ln2_b": np.zeros(D, np.float32),
    "wqkv": init(D, 3 * D, scale=D ** -0.5), "bqkv": np.zeros(3 * D, np.float32),
    "wo": init(D, D, scale=D ** -0.5), "bo": np.zeros(D, np.float32),
    "w1": init(D, F, scale=D ** -0.5), "b1": np.zeros(F, np.float32),
    "w2": init(F, D, scale=F ** -0.5), "b2": np.zeros(D, np.float32),
}
X = rng.standard_normal((B, T, D)).astype(np.float32)
EPS = 1e-5
print({k: v.shape for k, v in W.items()}, X.shape)

def gelu_tanh(x, tanh, sqrt):
    return 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

# float64 reference on the CPU (NumPy) — the numerics yardstick for all three.
def block_ref(x, w):
    x = x.astype(np.float64); w = {k: v.astype(np.float64) for k, v in w.items()}
    def ln(v, g, b):
        m = v.mean(-1, keepdims=True); s = ((v - m) ** 2).mean(-1, keepdims=True)
        return (v - m) / np.sqrt(s + EPS) * g + b
    h = ln(x, w["ln1_g"], w["ln1_b"]) @ w["wqkv"] + w["bqkv"]
    q, k, v = np.split(h, 3, axis=-1)
    q, k, v = [a.reshape(B, T, H, DH).transpose(0, 2, 1, 3) for a in (q, k, v)]
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(DH)
    s = s + np.triu(np.full((T, T), -np.inf), 1)
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    a = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    x = x + a @ w["wo"] + w["bo"]
    g = gelu_tanh(ln(x, w["ln2_g"], w["ln2_b"]) @ w["w1"] + w["b1"], np.tanh, np.sqrt)
    return x + g @ w["w2"] + w["b2"]

t0 = time.perf_counter(); REF = block_ref(X, W); print("float64 reference done in %.1f s" % (time.perf_counter() - t0))
'''),
    ("code", r'''
# --- Timing helper (identical method for all three: wall clock around a device-synchronized call) ---
import statistics
WARMUP, TIMED = 10, 50
def time_steady(fn, sync):
    for _ in range(WARMUP):
        fn(); sync()
    ts = []
    for _ in range(TIMED):
        t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    ts_sorted = sorted(ts)
    return {"median_ms": statistics.median(ts), "p95_ms": ts_sorted[int(0.95 * (len(ts) - 1))],
            "min_ms": ts_sorted[0], "warmup": WARMUP, "timed": TIMED, "iters_ms": ts}

def numerics(out):
    out = np.asarray(out, dtype=np.float64)
    d = np.abs(out - REF)
    # normwise relative error: max |out - ref| / max |ref| (elementwise relative
    # error is meaningless where ref is near zero)
    return {"max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
            "normwise_rel_err": float(d.max() / np.abs(REF).max())}
RESULTS = {"shape": {"B": B, "T": T, "D": D, "H": H}, "dtype": "float32 (TF32 disabled / highest matmul precision)",
           "timing_method": f"wall clock around a synchronized call, {WARMUP} warmup + {TIMED} timed"}
'''),
    ("code", r'''
# --- PyTorch eager ---
import torch, torch.nn.functional as Fnn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = "cuda"
Wt = {k: torch.tensor(v, device=dev) for k, v in W.items()}
Xt = torch.tensor(X, device=dev)
mask = torch.triu(torch.full((T, T), float("-inf"), device=dev), 1)

def block_torch(x):
    h = Fnn.layer_norm(x, (D,), Wt["ln1_g"], Wt["ln1_b"], EPS) @ Wt["wqkv"] + Wt["bqkv"]
    q, k, v = h.split(D, dim=-1)
    q, k, v = [a.view(B, T, H, DH).transpose(1, 2) for a in (q, k, v)]
    s = q @ k.transpose(-2, -1) / DH ** 0.5 + mask
    a = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, T, D)
    x = x + a @ Wt["wo"] + Wt["bo"]
    g = Fnn.gelu(Fnn.layer_norm(x, (D,), Wt["ln2_g"], Wt["ln2_b"], EPS) @ Wt["w1"] + Wt["b1"], approximate="tanh")
    return x + g @ Wt["w2"] + Wt["b2"]

with torch.no_grad():
    RESULTS["torch_eager"] = {"numerics": numerics(block_torch(Xt).cpu().numpy()),
                              "steady": time_steady(lambda: block_torch(Xt), torch.cuda.synchronize)}
print({k: v for k, v in RESULTS["torch_eager"]["steady"].items() if k != "iters_ms"}, RESULTS["torch_eager"]["numerics"])
'''),
    ("code", r'''
# --- torch.compile (Inductor): compile time separate from steady state; save generated code ---
try:
    import torch._dynamo
    torch._dynamo.reset()
    from torch._inductor.utils import run_and_get_code
    compiled = torch.compile(block_torch, mode="default", fullgraph=True)
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out, codes = run_and_get_code(compiled, Xt)          # first call = compile + run
        torch.cuda.synchronize(); first_call_s = time.perf_counter() - t0
        code_path = os.path.join(OUT, "nb1_inductor_output_code.py")
        with open(code_path, "w") as f:
            f.write("\n\n# ======== next generated module ========\n\n".join(codes))
        RESULTS["torch_compile"] = {
            "first_call_s_includes_compile": first_call_s,
            "numerics": numerics(out.cpu().numpy()),
            "steady": time_steady(lambda: compiled(Xt), torch.cuda.synchronize),
            "generated_code_file": os.path.basename(code_path), "generated_modules": len(codes),
        }
    print({k: v for k, v in RESULTS["torch_compile"].items() if k != "steady"})
except Exception as e:
    RESULTS["torch_compile"] = {"status": "NOT RUN", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    print("torch.compile NOT RUN:", RESULTS["torch_compile"]["error"])
'''),
    ("code", r'''
# --- JAX jit (XLA): lower+compile timed separately; save optimized HLO ---
try:
    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")   # no TF32
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError(f"JAX has no GPU device: {jax.devices()}")
    Wj = {k: jnp.asarray(v) for k, v in W.items()}
    Xj = jnp.asarray(X)
    maskj = jnp.triu(jnp.full((T, T), -jnp.inf, dtype=jnp.float32), 1)
    def ln_j(v, g, b):
        m = v.mean(-1, keepdims=True); s = ((v - m) ** 2).mean(-1, keepdims=True)
        return (v - m) * jax.lax.rsqrt(s + EPS) * g + b
    def block_jax(x, w):
        h = ln_j(x, w["ln1_g"], w["ln1_b"]) @ w["wqkv"] + w["bqkv"]
        q, k, v = jnp.split(h, 3, axis=-1)
        q, k, v = [a.reshape(B, T, H, DH).transpose(0, 2, 1, 3) for a in (q, k, v)]
        s = q @ k.transpose(0, 1, 3, 2) / DH ** 0.5 + maskj
        a = (jax.nn.softmax(s, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + a @ w["wo"] + w["bo"]
        g = jax.nn.gelu(ln_j(x, w["ln2_g"], w["ln2_b"]) @ w["w1"] + w["b1"], approximate=True)
        return x + g @ w["w2"] + w["b2"]
    t0 = time.perf_counter()
    lowered = jax.jit(block_jax).lower(Xj, Wj)
    t1 = time.perf_counter()
    exe = lowered.compile()
    t2 = time.perf_counter()
    hlo_path = os.path.join(OUT, "nb1_xla_optimized_hlo.txt")
    with open(hlo_path, "w") as f:
        f.write(exe.as_text())
    out = exe(Xj, Wj).block_until_ready()
    RESULTS["jax_jit"] = {
        "lower_s": t1 - t0, "compile_s": t2 - t1,
        "numerics": numerics(np.asarray(out)),
        "optimized_hlo_file": os.path.basename(hlo_path),
    }
    # steady state: block_until_ready is the device sync for JAX
    holder = {}
    def run(): holder["o"] = exe(Xj, Wj)
    def sync(): holder["o"].block_until_ready()
    RESULTS["jax_jit"]["steady"] = time_steady(run, sync)
    print({k: v for k, v in RESULTS["jax_jit"].items() if k != "steady"})
except Exception as e:
    RESULTS["jax_jit"] = {"status": "NOT RUN", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    print("JAX NOT RUN:", RESULTS["jax_jit"]["error"])
'''),
    ("code", r'''
# --- What did each compiler fuse? Read ONLY from the saved dumps. ---
import re
def inductor_summary(path):
    src = open(path).read()
    kernels = re.findall(r"^def (triton_[A-Za-z0-9_]+)\(", src, flags=re.M)
    kernels = sorted(set(kernels))
    externs = re.findall(r"extern_kernels\.(\w+)\(", src)
    return {"triton_kernels": kernels, "n_triton_kernels": len(kernels),
            "extern_calls": {e: externs.count(e) for e in sorted(set(externs))},
            "note": "Inductor kernel names encode the fused ops (e.g. triton_per_fused_..._softmax_...)."}
def hlo_summary(path):
    txt = open(path).read()
    fusions = re.findall(r"^\s*(\S+) = .*? fusion\(.*?kind=(k\w+)", txt, flags=re.M)
    customs = re.findall(r'custom_call_target="([^"]+)"', txt)
    from collections import Counter
    return {"n_fusions": len(fusions), "fusion_kinds": dict(Counter(k for _, k in fusions)),
            "fusion_names": [f for f, _ in fusions][:60],
            "custom_calls": {c: customs.count(c) for c in sorted(set(customs))},
            "note": "custom_call targets such as __cublas$gemm are library GEMMs; fusion(kind=kLoop/kInput/kOutput) are XLA-generated kernels."}
FUSION = {}
tc = RESULTS.get("torch_compile", {})
if "generated_code_file" in tc:
    FUSION["inductor"] = inductor_summary(os.path.join(OUT, tc["generated_code_file"]))
jj = RESULTS.get("jax_jit", {})
if "optimized_hlo_file" in jj:
    FUSION["xla"] = hlo_summary(os.path.join(OUT, jj["optimized_hlo_file"]))
print(json.dumps(FUSION, indent=2))
save("nb1_fusion_summary.json", FUSION)
'''),
    ("code", r'''
# --- Summary table + save everything ---
def row(name, r):
    if r.get("status") == "NOT RUN":
        return f"| {name} | NOT RUN | | | | {r['error'][:80]} |"
    s, n = r["steady"], r["numerics"]
    comp = r.get("first_call_s_includes_compile", r.get("compile_s", 0.0))
    return f"| {name} | {s['median_ms']:.3f} | {s['p95_ms']:.3f} | {comp:.2f} | {n['max_abs_err']:.2e} | |"
lines = ["| impl | median ms | p95 ms | compile s | max abs err vs float64 | note |", "|---|---|---|---|---|---|"]
lines += [row(k, RESULTS[k]) for k in ("torch_eager", "torch_compile", "jax_jit")]
print("\n".join(lines))
save("nb1_results.json", RESULTS)
'''),
    ("markdown", r'''
## Findings (fill in from the dumps after running — every statement must point at a line in a dump)

NOT RUN yet. After running, read `nb1_fusion_summary.json`, `nb1_inductor_output_code.py`
and `nb1_xla_optimized_hlo.txt`, and write here: which ops each compiler fused
into one kernel (LayerNorm? bias+GELU? residual adds? softmax with scale and
mask?), which stayed separate (GEMMs as library calls?), and how that lines up
with the measured steady-state times. Do not claim anything about MLIR/LLVM
internals that the dumps do not show.
'''),
    ("code", r'''
# --- Zip results for download ---
import shutil
zip_path = shutil.make_archive(OUT, "zip", OUT)
print("download:", zip_path)
try:
    from google.colab import files  # Colab only
    files.download(zip_path)
except Exception:
    print("Not on Colab: download the zip from the file browser (Kaggle: Output tab).")
'''),
]

NB2 = [
    ("markdown", r'''
# Notebook 2 — a fused row softmax written in Triton, from scratch

Goal: write the kernel, prove it correct on awkward shapes and dtypes against
a float64 reference, then benchmark it against (a) the same softmax composed
from PyTorch eager ops and (b) the library kernel `torch.softmax`, and
explain the result. The one-row-per-program structure is the standard idea
from the Triton tutorials (adapted, not copied; see docs/REFERENCES.md); the
looped variant for rows wider than one block is this notebook's own.

How to run: Colab or Kaggle, GPU runtime, Run all. See `RUNBOOK.md`.
'''),
    ("code", COMMON_ENV + '\nsave("nb2_env.json", env)\n'),
    ("code", r'''
import torch, triton, triton.language as tl, math, statistics
assert torch.cuda.is_available(), "needs a CUDA GPU"
torch.backends.cuda.matmul.allow_tf32 = False
MAX_BLOCK = 8192   # widest row handled in one block; wider rows use the looped kernel

@triton.jit
def softmax_row_kernel(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr):
    # One program per row; the whole row fits in BLOCK (a power of two >= n_cols).
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols, mask=m, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                 # stabilize: exp of values <= 0
    e = tl.exp(x)                             # masked lanes: exp(-inf) = 0
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * y_stride + cols, y.to(y_ptr.dtype.element_ty), mask=m)

@triton.jit
def softmax_row_looped_kernel(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr):
    # Rows wider than one block. Pass 1 keeps, per lane, a running max m and a
    # running sum s of exp(x - m) rescaled whenever m grows (online softmax);
    # the lanes are combined after the loop. Pass 2 writes exp(x - max) / sum.
    # Per-lane [BLOCK] accumulators keep loop-carried types consistent.
    row = tl.program_id(0)
    m_vec = tl.full([BLOCK], -float("inf"), tl.float32)
    s_vec = tl.zeros([BLOCK], tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + row * x_stride + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m_vec, x)
        # Guards: (-inf) - (-inf) is NaN, so lanes that are still -inf contribute 0.
        alpha = tl.where(m_vec == m_new, 1.0, tl.exp(m_vec - m_new))
        term = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_new))
        s_vec = s_vec * alpha + term
        m_vec = m_new
    row_max = tl.max(m_vec, axis=0)
    scale = tl.where(m_vec == -float("inf"), 0.0, tl.exp(m_vec - row_max))
    row_sum = tl.sum(s_vec * scale, axis=0)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < n_cols
        x = tl.load(x_ptr + row * x_stride + cols, mask=m, other=-float("inf")).to(tl.float32)
        tl.store(y_ptr + row * y_stride + cols, (tl.exp(x - row_max) / row_sum).to(y_ptr.dtype.element_ty), mask=m)

def triton_softmax(x):
    assert x.dim() == 2 and x.is_cuda and x.stride(1) == 1
    rows, n = x.shape
    y = torch.empty_like(x)
    if n <= MAX_BLOCK:
        BLOCK = triton.next_power_of_2(max(n, 1))
        softmax_row_kernel[(rows,)](x, y, n, x.stride(0), y.stride(0), BLOCK=BLOCK, num_warps=4 if BLOCK <= 2048 else 8)
    else:
        softmax_row_looped_kernel[(rows,)](x, y, n, x.stride(0), y.stride(0), BLOCK=4096, num_warps=8)
    return y

def composed_softmax(x):          # the same math as separate eager PyTorch ops
    xf = x.float()
    e = torch.exp(xf - xf.max(dim=-1, keepdim=True).values)
    return (e / e.sum(dim=-1, keepdim=True)).to(x.dtype)
'''),
    ("code", r'''
# --- Correctness gate: awkward shapes, three dtypes, extreme values, against float64 ---
# Tolerance: compare in float32 after casting; one output ulp of the dtype
# (2^-23 fp32 is too tight for exp/sum reordering, so fp32 uses 1e-6 abs + 1e-5 rel;
# fp16 2^-10, bf16 2^-7 relative), justified by a single rounding of the output.
TOL = {torch.float32: (1e-6, 1e-5), torch.float16: (1e-4, 2 ** -10), torch.bfloat16: (1e-3, 2 ** -7)}
cases = []
g = torch.Generator(device="cuda").manual_seed(0)
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for rows, n in [(1, 1), (1, 3), (7, 127), (5, 1000), (3, 1023), (4, 1024), (2, 4097), (3, 8192), (2, 8193), (2, 30001)]:
        for kind in ("randn", "large", "neg_inf_lanes"):
            x = torch.randn(rows, n, device="cuda", generator=g)
            if kind == "large":
                x = x * 80.0          # exp overflows without max subtraction
            if kind == "neg_inf_lanes" and n > 1:
                x[:, ::3] = -float("inf")
            x = x.to(dtype)
            ref = torch.softmax(x.double(), dim=-1)
            out = triton_softmax(x).double()
            atol, rtol = TOL[dtype]
            err = (out - ref).abs()
            ok = bool((err <= atol + rtol * ref.abs()).all()) and bool(torch.isfinite(out).all())
            cases.append({"dtype": str(dtype), "rows": rows, "n_cols": n, "kind": kind, "ok": ok,
                          "max_abs_err": float(err.max()), "path": "single" if n <= MAX_BLOCK else "looped"})
bad = [c for c in cases if not c["ok"]]
print(f"{len(cases) - len(bad)}/{len(cases)} cases pass")
for c in bad: print("FAIL", c)
save("nb2_correctness.json", {"cases": cases, "passed": len(cases) - len(bad), "total": len(cases)})
assert not bad, "fix the kernel before benchmarking"
'''),
    ("code", r'''
# --- Benchmark: composed eager vs torch.softmax (library) vs Triton ---
# triton.testing.do_bench uses CUDA events and returns milliseconds; we take
# quantiles (median, p20, p80) over its repetitions. Effective GB/s = one
# read + one write of the tensor / time (minimum traffic; cache effects at
# small sizes can push it above DRAM bandwidth — read it as effective).
from triton.testing import do_bench
bench = []
for dtype in (torch.float32, torch.float16):
    for rows, n in [(4096, 256), (4096, 1024), (4096, 4096), (1024, 8192), (256, 32768)]:
        x = torch.randn(rows, n, device="cuda", dtype=dtype)
        nbytes = 2 * x.numel() * x.element_size()
        rec = {"dtype": str(dtype), "rows": rows, "n_cols": n}
        for name, fn in (("composed_eager", lambda: composed_softmax(x)), ("torch_softmax", lambda: torch.softmax(x, -1)),
                         ("triton", lambda: triton_softmax(x))):
            med, p20, p80 = do_bench(fn, warmup=25, rep=200, quantiles=[0.5, 0.2, 0.8])
            rec[name] = {"median_ms": med, "p20_ms": p20, "p80_ms": p80, "effective_gbps": nbytes / (med * 1e-3) / 1e9}
        bench.append(rec)
        print(f"{str(dtype):14s} {rows:5d}x{n:<6d}  composed {rec['composed_eager']['median_ms']:.4f}  "
              f"torch {rec['torch_softmax']['median_ms']:.4f}  triton {rec['triton']['median_ms']:.4f} ms")
save("nb2_bench.json", {"method": "triton.testing.do_bench, warmup=25 ms, rep=200 ms, median/p20/p80", "rows": bench})
'''),
    ("code", r'''
# --- Repeat once and check drift (kern/llm-bringup discipline: flag > 5%) ---
drift = []
for rec in bench:
    dtype = torch.float32 if "float32" in rec["dtype"] else torch.float16
    x = torch.randn(rec["rows"], rec["n_cols"], device="cuda", dtype=dtype)
    for name, fn in (("torch_softmax", lambda: torch.softmax(x, -1)), ("triton", lambda: triton_softmax(x))):
        again = do_bench(fn, warmup=25, rep=200, quantiles=[0.5])
        again = again[0] if isinstance(again, (list, tuple)) else again
        d = 100 * abs(again - rec[name]["median_ms"]) / max(again, rec[name]["median_ms"])
        drift.append({"dtype": rec["dtype"], "rows": rec["rows"], "n_cols": rec["n_cols"], "impl": name,
                      "run1_ms": rec[name]["median_ms"], "run2_ms": again, "drift_pct": d, "flagged": d > 5})
print(sum(d["flagged"] for d in drift), "of", len(drift), "cells drift > 5%")
save("nb2_drift.json", drift)
'''),
    ("markdown", r'''
## Explanation (fill in after running, from nb2_bench.json only)

NOT RUN yet. Expected shape of the argument, to be confirmed or refuted by
the numbers: softmax is memory-bound; the composed eager version launches
several kernels and moves the tensor through memory several times (max,
subtract, exp, sum, divide, casts), the fused Triton kernel reads once and
writes once, and the library kernel is also fused. Report where Triton
matches, beats, or loses to `torch.softmax`, and at which widths the looped
path costs a second read.
'''),
    ("code", r'''
import shutil
zip_path = shutil.make_archive(OUT, "zip", OUT)
print("download:", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    print("Not on Colab: download the zip from the file browser (Kaggle: Output tab).")
'''),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in (("01_block_eager_compile_xla.ipynb", NB1), ("02_triton_fused_softmax.ipynb", NB2)):
        (out / name).write_text(json.dumps(nb(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
