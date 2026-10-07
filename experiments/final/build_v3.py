"""Build the v3 additions on the node (2026-10-05) and the radiology-relevance index of the knowledge test questions.

    python jobs/submit.py jobs/build_v3.py --gpus 0 --cpus 4 --mem 32 --timeout 60 --outputs 'build_v3.json'

1. data/radcases-v3/{train,dev,test}.jsonl: RadCases (open subsets) split 60/10/30 by case; cases whose text occurs in any
   existing RadKev split are dropped (build_external.radcases_splits).
2. data/rexerr/{train,dev,test}.jsonl: ReXErr-v1 report level (6,000 train and 500 dev sampled, all 2,708 test).
3. runs/test-final/radiology_index.json: for every knowledge-test question (MedMCQA, MedQA, MMLU, MedXpertQA, PubMedQA),
   whether build_external.is_radiology() keeps it; MedMCQA radiology is always kept. Counts per task are published.
"""
import json, os, subprocess
from pathlib import Path

INCLUDE = ["build_data.py", "build_external.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_PY = RADKEV / "kev/.venv/bin/python"
CODE = RADKEV / "code_v3"; CODE.mkdir(parents=True, exist_ok=True)
for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
INNER = r'''
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import build_data as bd, build_external as be
radkev = Path(sys.argv[2]); res = {}
chunks = []
for p in sorted((radkev / "data").glob("*/*.jsonl")):
    if p.parent.name in ("radcases-v3", "rexerr"): continue
    for line in open(p):
        if line.strip():
            s = json.loads(line)["state"]; chunks.append(bd.norm(s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)))
big = " | ".join(chunks)
class Contains:
    def __contains__(self, x): return len(x) > 20 and x in big
out = radkev / "data/radcases-v3"; out.mkdir(parents=True, exist_ok=True)
res["radcases"] = be.radcases_splits(out, check_text=Contains())
out = radkev / "data/rexerr"; out.mkdir(parents=True, exist_ok=True)
res["rexerr"] = be.rexerr("$RADKEV_HOME/raw", out)
test = radkev / "runs/test-final/test.jsonl"; idx = {}; cnt = Counter(); kept = Counter()
KNOW = ("medmcqa", "medqa", "mmlu", "medxpertqa", "pubmedqa")
for n, line in enumerate(open(test)):
    if not line.strip(): continue
    r = json.loads(line)
    for qid, q in r["questions"].items():
        src = str(q.get("src", ""))
        if not src.startswith(KNOW): continue
        keep = src == "medmcqa_rad" or be.is_radiology(r["state"], q)
        idx[f"rad/{n}|{qid}"] = keep; cnt[src] += 1; kept[src] += keep
(radkev / "runs/test-final/radiology_index.json").write_text(json.dumps(idx))
res["radiology_filter"] = {t: {"n": cnt[t], "kept": kept[t]} for t in sorted(cnt)}
print(json.dumps(res))
'''
p = subprocess.run([str(KEV_PY), "-c", INNER, str(CODE), str(RADKEV)], env={**os.environ, "PYTHONPATH": str(CODE)}, capture_output=True, text=True)
(OUT / "build_v3.json").write_text(json.dumps(json.loads(p.stdout.strip().splitlines()[-1]), indent=1) if p.returncode == 0 else json.dumps({"error": p.stderr[-3000:]}))
