"""RadGraph-XL restricted to definite findings (post hoc sensitivity analysis; user request 2026-10-07). No GPU, no rescoring.

    python jobs/submit.py jobs/radgraph_xl_definite.py --gpus 0 --cpus 4 --mem 16 --timeout 30 --outputs 'analysis_definite.json'

The external-test analysis of jobs/external_analyse_node.py (paper/external/analyse.py: accuracy, ECE, paired differences with a
record-level bootstrap stratified by modality), run on the questions whose radiologist-annotated status is present or absent, i.e.
excluding the questions whose ground truth is uncertain (hedged). The models' outputs are unchanged: every question still offered
all four status options. Publishes aggregates only.
"""
import json, os, subprocess
from pathlib import Path

INCLUDE = ["paper/external/analyse.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
CODE = RADKEV / "external/code_def"; CODE.mkdir(parents=True, exist_ok=True)
for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
INNER = r'''
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
import analyse as A
W, out = Path(sys.argv[2]), Path(sys.argv[3])
recs = {json.loads(l)["_meta"]["id"]: json.loads(l) for l in open(W / "radgraph_xl.jsonl") if l.strip()}
gold = lambda rid, qid: recs[rid]["questions"][qid]["label"]
runs = W / "runs/radgraph_xl"; rows = {}; rn = {}; kept = Counter(); dropped = Counter()
for mdir in sorted(p for p in runs.iterdir() if (p / "rows.json").exists()):
    rr = []
    for r in scored_rows(json.loads((mdir / "rows.json").read_text())):
        g = gold(r["id"], r["question"])
        if g == "uncertain": dropped[mdir.name] += 1; continue
        kept[mdir.name] += 1
        rr.append([r["id"], r["question"], r["task"], int(r["label"]), int(np.argmax(r["p"])), float(r["p"][int(r["label"])]), float(max(r["p"]))])
        # variant: the "uncertain" option removed from the answer (probabilities renormalized over the other options)
        ks = list(recs[r["id"]]["questions"][r["question"]]["criteria"]); p_ = np.asarray(r["p"], float).copy(); p_[ks.index("uncertain")] = 0; p_ = p_ / p_.sum()
        rn.setdefault(mdir.name, []).append([r["id"], r["question"], r["task"], int(r["label"]), int(np.argmax(p_)), float(p_[int(r["label"])]), float(p_.max())])
    rows[mdir.name] = rr
res = {"radgraph_xl_definite": A.analyse(rows, set()), "radgraph_xl_definite_no_uncertain_option": A.analyse(rn, set()),
       "kept": dict(kept), "dropped_uncertain": dict(dropped)}
out.write_text(json.dumps(res, indent=1)); print("ok")
'''
p = subprocess.run([str(KEV_PY), "-c", INNER, str(CODE), str(RADKEV / "external"), str(OUT / "analysis_definite.json")],
                   env={**os.environ, "PYTHONPATH": str(KEV_DIR), "HF_HUB_OFFLINE": "1"}, cwd=KEV_DIR, capture_output=True, text=True)
if p.returncode: (OUT / "analysis_definite.json").write_text(json.dumps({"error": p.stderr[-3000:]}))
