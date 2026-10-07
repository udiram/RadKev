"""Run paper/external/analyse.py on the node for files whose compact rows exceed the bridge's 1 MiB artifact limit.

    python jobs/submit.py jobs/external_analyse_node.py --gpus 0 --cpus 4 --mem 16 --timeout 30 --outputs 'analysis_node.json' -- radgraph_xl
"""
import json, os, subprocess, sys
from pathlib import Path

INCLUDE = ["paper/external/analyse.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
CODE = RADKEV / "external/code_an"; CODE.mkdir(parents=True, exist_ok=True)
for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
INNER = r'''
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
import analyse as A
runs, out = Path(sys.argv[2]), Path(sys.argv[3]); res = {}
for name in sys.argv[4].split(","):
    rows = {}
    for mdir in sorted(p for p in (runs / name).iterdir() if (p / "rows.json").exists()):
        rows[mdir.name] = [[r["id"], r["question"], r["task"], int(r["label"]), int(np.argmax(r["p"])), float(r["p"][int(r["label"])]), float(max(r["p"]))]
                           for r in scored_rows(json.loads((mdir / "rows.json").read_text()))]
    res[name] = A.analyse(rows, set())
out.write_text(json.dumps(res, indent=1)); print("ok")
'''
p = subprocess.run([str(KEV_PY), "-c", INNER, str(CODE), str(RADKEV / "external/runs"), str(OUT / "analysis_node.json"), sys.argv[1] if len(sys.argv) > 1 else "radgraph_xl"],
                   env={**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}, cwd=KEV_DIR, capture_output=True, text=True)
if p.returncode: (OUT / "analysis_node.json").write_text(json.dumps({"error": p.stderr[-3000:]}))
