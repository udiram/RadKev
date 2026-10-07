"""RadGraph-XL error analysis for RadKev v3 (read-only; aggregates and generic finding terms only, no report text).

    python jobs/submit.py jobs/radgraph_xl_errors_v3.py --gpus 0 --cpus 4 --mem 16 --timeout 30 --outputs 'rgx_errors_v3.json'

For each system on $RADKEV/external/runs/radgraph_xl: confusion of gold status (present / absent / uncertain) against the
predicted option (incl. not_mentioned), by modality; accuracy by instruction wording; how often RadKev-27B and Kev-27B
give the same answer; agreement of the 27B pair on disagreements; mean confidence; the most frequent finding spans
(lower-cased, at most 4 words) among questions both 27B models miss, with counts; and, for the teacher-format comparison,
the same confusion on the test-final teacher finding-status task.
"""
import json, os, subprocess
from pathlib import Path

BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
INNER = r'''
import json, sys, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from kev.metrics import scored_rows
W = Path(sys.argv[1]); out = Path(sys.argv[2])
recs = {json.loads(l)["_meta"]["id"]: json.loads(l) for l in open(W / "radgraph_xl.jsonl") if l.strip()}
runs = W / "runs/radgraph_xl"
models = sorted(p.name for p in runs.iterdir() if (p / "rows.json").exists())
R = {m: {(r["id"], r["question"]): r for r in scored_rows(json.loads((runs / m / "rows.json").read_text()))} for m in models}
keys = sorted(set.intersection(*[set(v) for v in R.values()]))
def q(k): return recs[k[0]]["questions"][k[1]]
def okeys(k): return list(q(k)["criteria"])
res = {"n": len(keys), "models": {}}
for m in models:
    conf = defaultdict(Counter); bymod = defaultdict(lambda: defaultdict(Counter)); words = defaultdict(lambda: [0, 0]); pmax = []
    for k in keys:
        r = R[m][k]; ks = okeys(k); g = q(k)["label"]; p = ks[int(np.argmax(r["p"]))]
        conf[g][p] += 1; bymod[recs[k[0]]["_meta"]["modality"]][g][p] += 1
        pmax.append(float(max(r["p"])))
    res["models"][m] = {"confusion": {g: dict(c) for g, c in conf.items()}, "by_modality": {mo: {g: dict(c) for g, c in d.items()} for mo, d in bymod.items()},
                        "acc_by_gold": {g: c[g] / sum(c.values()) for g, c in conf.items()}, "mean_conf": float(np.mean(pmax)),
                        "templates": {}}
    for k in keys:
        ins = q(k)["instructions"]
        for t in ("What does the report say about", "How does this report characterise", "Which statement best describes", "Classify the report's mention of",):
            if ins.startswith(t):
                d = res["models"][m]["templates"].setdefault(t, [0, 0]); d[0] += int(okeys(k)[int(np.argmax(R[m][k]["p"]))] == q(k)["label"]); d[1] += 1
pred = lambda m, k: okeys(k)[int(np.argmax(R[m][k]["p"]))]
span = lambda k: re.sub(r"^(what does the report say about|how does this report characterise|which statement best describes|classify the report's mention of)\s*", "", q(k)["instructions"].lower()).replace(" in this report", "").rstrip("?.")
res["gold_dist"] = dict(Counter(q(k)["label"] for k in keys))
for m in models:
    res["models"][m]["pred_dist"] = dict(Counter(pred(m, k) for k in keys))
    mp = defaultdict(list)
    for k in keys:
        for o, pv in zip(okeys(k), R[m][k]["p"]): mp[o].append(float(pv))
    res["models"][m]["mean_p"] = {o: float(np.mean(v)) for o, v in mp.items()}
res["pairs"] = {}
for a, b in (("v3_27", "stock27"), ("v3_27", "qwen38"), ("v3_9", "stock9")):
    if a in R and b in R:
        ga = [k for k in keys if pred(a, k) == q(k)["label"] != pred(b, k)]; gb = [k for k in keys if pred(b, k) == q(k)["label"] != pred(a, k)]
        res["pairs"][f"{a}-{b}"] = {"same_prediction": sum(pred(a, k) == pred(b, k) for k in keys),
            "only_a_right": len(ga), "only_b_right": len(gb),
            "only_a_right_by_gold": dict(Counter(q(k)["label"] for k in ga)), "only_b_right_by_gold": dict(Counter(q(k)["label"] for k in gb)),
            "b_right_a_wrong_by_gold_pred": dict(Counter(f"{q(k)['label']}->{pred(a, k)}" for k in gb)),
            "a_right_b_wrong_by_gold_pred": dict(Counter(f"{q(k)['label']}->{pred(b, k)}" for k in ga)),
            "b_right_a_wrong_terms_top": Counter(span(k) for k in gb if len(span(k).split()) <= 4).most_common(30)}
# status-format questions in the v3 training and dev data: labels by source and by finding phrasing
ST = {"present", "absent", "uncertain", "not_mentioned"}; TR = {}
for split in ("train", "dev"):
    by_src = defaultdict(Counter); n_rec = 0
    for line in open(W.parent / "data/mix-v3r" / f"{split}.jsonl"):
        if not line.strip(): continue
        r = json.loads(line)
        for qq in r["questions"].values():
            crit = qq.get("criteria") or {}
            ks = set(crit) if isinstance(crit, dict) else set()
            if ks and ks <= ST and len(ks) >= 3:
                lab = qq.get("label"); lab = lab if isinstance(lab, str) else (list(crit)[int(lab)] if lab is not None else None)
                by_src[str(qq.get("src", "?"))][str(lab)] += 1
    TR[split] = {s: dict(c) for s, c in by_src.items()}
res["train_status_labels"] = TR
out.write_text(json.dumps(res, indent=1)); print("ok")
'''
p = subprocess.run([str(KEV_PY), "-c", INNER, str(RADKEV / "external"), str(OUT / "rgx_errors_v3.json")], env={**os.environ, "PYTHONPATH": str(KEV_DIR), "HF_HUB_OFFLINE": "1"},
                   cwd=KEV_DIR, capture_output=True, text=True)
if p.returncode: (OUT / "rgx_errors_v3.json").write_text(json.dumps({"error": p.stderr[-3000:]}))
