"""The committed notebooks must equal what scripts/build_notebooks.py generates."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_match_builder(tmp_path):
    before = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "notebooks").glob("*.ipynb")}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebooks.py")], check=True, capture_output=True)
    after = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "notebooks").glob("*.ipynb")}
    assert before == after, "notebooks are stale: run python scripts/build_notebooks.py"
    for text in after.values():
        nb = json.loads(text)
        assert nb["nbformat"] == 4 and any(c["cell_type"] == "code" for c in nb["cells"])
