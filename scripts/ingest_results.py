#!/usr/bin/env python3
"""Ingest a downloaded notebook results zip and regenerate every number in
README.md, results/facts.json and docs/FACTS.md from it.

Usage:
  python scripts/ingest_results.py path/to/compiler-study-results.zip   # copy into results/, then render
  python scripts/ingest_results.py                                       # render from what is in results/
  python scripts/ingest_results.py --check                               # exit 1 if README/facts are stale

Anything whose files are missing renders as NOT RUN with the command to run.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BEGIN, END = "<!-- RESULTS-BEGIN -->", "<!-- RESULTS-END -->"
EXPECTED = ["nb1_env.json", "nb1_results.json", "nb1_fusion_summary.json",
            "nb2_env.json", "nb2_correctness.json", "nb2_bench.json", "nb2_drift.json"]


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ingest(src: Path):
    RES.mkdir(exist_ok=True)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                target = RES / Path(n).name
                with z.open(n) as fsrc, open(target, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
    else:
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, RES / p.name)
    missing = [e for e in EXPECTED if not (RES / e).exists()]
    print("ingested;", "missing: " + ", ".join(missing) if missing else "all expected files present")


def render():
    nb1, fus, nb2c, nb2b, nb2d = (load("nb1_results.json"), load("nb1_fusion_summary.json"),
                                  load("nb2_correctness.json"), load("nb2_bench.json"), load("nb2_drift.json"))
    env1, env2 = load("nb1_env.json"), load("nb2_env.json")
    facts = {"generated_by": "scripts/ingest_results.py", "notebook1": "NOT RUN", "notebook2": "NOT RUN"}
    lines = []
    if nb1:
        facts["notebook1"] = {"environment": env1, "rows": {}}
        lines += [f"### Notebook 1 — transformer block ({nb1['shape']}), {nb1['dtype']}", "",
                  f"Environment: {env1.get('nvidia_smi') if env1 else 'n/a'}; torch {env1.get('torch') if env1 else '?'}, "
                  f"jax {env1.get('jax') if env1 else '?'}.", "",
                  "| implementation | steady median ms | p95 ms | compile s | max abs err vs float64 |", "|---|---|---|---|---|"]
        for k, label in (("torch_eager", "PyTorch eager"), ("torch_compile", "torch.compile (Inductor)"), ("jax_jit", "JAX jit (XLA)")):
            r = nb1.get(k, {})
            if r.get("status") == "NOT RUN" or "steady" not in r:
                lines.append(f"| {label} | NOT RUN: {r.get('error', 'missing')[:70]} | | | |")
                facts["notebook1"]["rows"][k] = "NOT RUN"
                continue
            comp = r.get("first_call_s_includes_compile", r.get("compile_s"))
            lines.append(f"| {label} | {r['steady']['median_ms']:.3f} | {r['steady']['p95_ms']:.3f} | "
                         f"{'' if comp is None else f'{comp:.2f}'} | {r['numerics']['max_abs_err']:.2e} |")
            facts["notebook1"]["rows"][k] = {"median_ms": r["steady"]["median_ms"], "p95_ms": r["steady"]["p95_ms"],
                                             "compile_s": comp, "max_abs_err": r["numerics"]["max_abs_err"]}
        if fus:
            facts["notebook1"]["fusion"] = fus
            if "inductor" in fus:
                lines += ["", f"Inductor generated {fus['inductor']['n_triton_kernels']} Triton kernels; extern calls: "
                              f"{fus['inductor']['extern_calls']} (`results/nb1_inductor_output_code.py`)."]
            if "xla" in fus:
                lines += [f"XLA optimized HLO has {fus['xla']['n_fusions']} fusions {fus['xla']['fusion_kinds']}; custom calls: "
                          f"{fus['xla']['custom_calls']} (`results/nb1_xla_optimized_hlo.txt`)."]
    else:
        lines += ["### Notebook 1 — NOT RUN", "", "Run `notebooks/01_block_eager_compile_xla.ipynb` on a Linux GPU (see RUNBOOK.md), "
                  "then `python scripts/ingest_results.py <zip>`."]
    lines.append("")
    if nb2c and nb2b:
        facts["notebook2"] = {"environment": env2, "correctness": {"passed": nb2c["passed"], "total": nb2c["total"]},
                              "bench": nb2b["rows"], "drift_flagged": sum(d["flagged"] for d in nb2d) if nb2d else None}
        lines += [f"### Notebook 2 — Triton fused softmax", "",
                  f"Correctness: {nb2c['passed']}/{nb2c['total']} cases pass against float64.", "",
                  "| dtype | rows x cols | composed eager ms | torch.softmax ms | Triton ms | Triton vs torch.softmax |", "|---|---|---|---|---|---|"]
        for r in nb2b["rows"]:
            t, l, c = r["triton"]["median_ms"], r["torch_softmax"]["median_ms"], r["composed_eager"]["median_ms"]
            lines.append(f"| {r['dtype'].replace('torch.', '')} | {r['rows']} x {r['n_cols']} | {c:.4f} | {l:.4f} | {t:.4f} | {l / t:.2f}x |")
        if nb2d:
            lines += ["", f"Repeat-run drift above 5%: {sum(d['flagged'] for d in nb2d)} of {len(nb2d)} cells (`results/nb2_drift.json`)."]
    else:
        lines += ["### Notebook 2 — NOT RUN", "", "Run `notebooks/02_triton_fused_softmax.ipynb` on a Linux GPU (see RUNBOOK.md), "
                  "then `python scripts/ingest_results.py <zip>`."]
    block = BEGIN + "\n" + "\n".join(lines) + "\n" + END
    rf = ["# Verified facts (generated — do not edit)", "", "Generated by `scripts/ingest_results.py` from `results/`.", ""]
    if facts["notebook1"] == "NOT RUN" and facts["notebook2"] == "NOT RUN":
        rf += ["No verified numbers yet: both notebooks are NOT RUN (see RUNBOOK.md)."]
    else:
        rf += ["See the README results block; every value there is rendered from the files in `results/` named beside it."]
    return block, json.dumps(facts, indent=2, default=str) + "\n", "\n".join(rf) + "\n"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    if args:
        ingest(Path(args[0]))
    block, facts, factsDoc = render()
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    import re
    new = re.sub(re.escape(BEGIN) + r"[\s\S]*?" + re.escape(END), lambda _: block, text)
    targets = {readme: new, RES / "facts.json": facts, ROOT / "docs" / "FACTS.md": factsDoc}
    if check:
        stale = [str(p.relative_to(ROOT)) for p, c in targets.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != c]
        print("stale: " + ", ".join(stale) if stale else "up to date")
        return 1 if stale else 0
    RES.mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    for p, c in targets.items():
        p.write_text(c, encoding="utf-8")
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
