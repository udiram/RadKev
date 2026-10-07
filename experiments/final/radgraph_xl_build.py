"""Build the RadGraph-XL external-test records on the node (analysis-plan addendum 2026-10-05). Publishes counts only.

    python jobs/submit.py jobs/radgraph_xl_build.py --gpus 0 --cpus 8 --mem 48 --timeout 60 --outputs 'radgraph_xl_build.json'

Reads the Stanford RadGraph-XL table (2,000 radiologist-annotated reports, DyGIE format) from $RADKEV_HOME/raw,
drops every report whose word 8-grams overlap by 50% or more with (a) any CheXpert Plus report (findings or impression; source
of the teacher-labeled training states) or (b) any record state of the RadKev splits ($RADKEV/data/*/*.jsonl), and writes
$RADKEV/external/radgraph_xl.jsonl with build_external.radgraph_records.
"""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

INCLUDE = ["build_data.py", "build_external.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")); RADKEV = CACHE / "radkev"
KEV_PY = RADKEV / "kev/.venv/bin/python"
WORK = RADKEV / "external"; CODE = WORK / "code_rgx"
RAW = Path("$RADKEV_HOME/raw")

INNER = r'''
import csv, json, sys
from collections import Counter
from pathlib import Path
import pandas as pd
sys.path.insert(0, sys.argv[1]); csv.field_size_limit(sys.maxsize)
import build_data as bd, build_external as be
raw, radkev, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
MOD = {"stanford-chest-x-ray": "cxr", "stanford-chest-ct": "chestct", "stanford-abd-pelvis-ct": "abdct", "stanford-brain-mr": "brainmr"}
tab = next((raw / "radgraph_xl").rglob("stanford_radgraph_xl_jsonl.csv"))
rows = []
for r in csv.DictReader(open(tab, newline="")):
    rows.append({"dataset": r["dataset"], "doc_key": r["doc_key"], "sentences": json.loads(r["sentences"]), "ner": json.loads(r["ner"])})
docs = list(be.dygie_docs(rows, lambda d: MOD.get(d, d.replace("stanford-", "").replace("-", ""))))
def grams(t, k=8):
    w = bd.norm(t).split(); return {hash(" ".join(w[i:i + k])) for i in range(max(0, len(w) - k + 1))}
target = {d["doc_key"]: grams(d["text"]) for d in docs}
need = set().union(*target.values())
hit_cp, hit_rk = set(), set()
cp = next((raw / "chexpert_plus").rglob("df_chexpert_plus_240401.csv"))
cols = [c for c in pd.read_csv(cp, nrows=1).columns if c.startswith("section_") or c in ("report",)]
for chunk in pd.read_csv(cp, usecols=cols, chunksize=20000, dtype=str):
    for row in chunk.fillna("").itertuples(index=False):
        g = grams(" ".join(row)); hit_cp |= (g & need)
for p in sorted((radkev / "data").glob("*/*.jsonl")):
    for line in open(p):
        if line.strip():
            s = json.loads(line)["state"]; g = grams(s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)); hit_rk |= (g & need)
over = {}
for k, g in target.items():
    over[k] = (len(g & hit_cp) / max(1, len(g)), len(g & hit_rk) / max(1, len(g)))
excl = {k for k, (a, b) in over.items() if a >= 0.5 or b >= 0.5}
by_text = {d["text"]: d["doc_key"] for d in docs}
recs = list(be.radgraph_records(docs, exclude=lambda t: by_text[t] in excl))
with open(out, "w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
mods = Counter(d["modality"] for d in docs)
rep = {"reports": len(docs), "by_modality": dict(mods), "cp_columns": cols,
       "overlap_chexpert_plus": dict(Counter(next(d["modality"] for d in docs if d["doc_key"] == k) for k, (a, b) in over.items() if a >= 0.5)),
       "overlap_radkev": dict(Counter(next(d["modality"] for d in docs if d["doc_key"] == k) for k, (a, b) in over.items() if b >= 0.5)),
       "excluded": len(excl), "stats": be.radgraph_records.stats,
       "entity_labels": dict(Counter(e["label"] for d in docs for e in d["entities"]).most_common(20))}
print(json.dumps(rep))
'''


def main():
    for p in (WORK, CODE): p.mkdir(parents=True, exist_ok=True)
    for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
    import subprocess
    p = subprocess.run([str(KEV_PY), "-c", INNER, str(CODE), str(RAW), str(RADKEV), str(WORK / "radgraph_xl.jsonl")],
                       env={**os.environ, "PYTHONPATH": str(CODE)}, capture_output=True, text=True)
    res = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-3000:]}
    (OUT / "radgraph_xl_build.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
