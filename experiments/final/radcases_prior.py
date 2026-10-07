"""Post hoc prior correction on the RadCases panel question (user request 2026-10-06; no retraining, no GPU).

    python jobs/submit.py jobs/radcases_prior.py --gpus 0 --cpus 2 --mem 8 --timeout 20 --outputs 'radcases_prior*.json'

The fine-tuned models learned the training-split frequency of each panel answer (the catch-all "None" option was the most
frequent). Prior correction divides each option's probability by its frequency in the RadCases training split (add-one
smoothed over the 12 options) and renormalizes, i.e. it assumes a uniform answer distribution at test time. It uses only
training labels; test labels are used only to score. Applied to RadKev-27B and RadKev-9B (trained on RadCases); Kev-27B and
Kev-9B (never trained on RadCases) are reported as scored. Paired bootstrap over the 132 test questions (2,000 resamples).
Publishes aggregates only (radcases_prior.json).
"""
import json, os, subprocess
from pathlib import Path
INCLUDE = []
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_PY = RADKEV / "kev/.venv/bin/python"; KEV_DIR = RADKEV / "kev"
INNER = r'''
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from kev.metrics import scored_rows
R = Path(sys.argv[1]); W = R / "runs/test-v3"
NONE = "None: no ACR Appropriateness Criteria topic applies"
# training prior over the panel options (by option text)
cnt = Counter(); opts = set()
for line in open(R / "data/radcases-v3/train.jsonl"):
    if not line.strip(): continue
    q = json.loads(line)["questions"].get("panel")
    if not q: continue
    cnt[q["criteria"][q["label"]]] += 1; opts |= set(q["criteria"].values())
assert len(opts) == 12 and NONE in opts, (len(opts), NONE in opts)
ntrain = sum(cnt.values()); prior = {o: (cnt[o] + 1) / (ntrain + len(opts)) for o in opts}
# test questions: option texts in criteria order, keyed by record id
crit = {}
for line in open(W / "new.jsonl"):
    if not line.strip(): continue
    r = json.loads(line); q = r["questions"].get("panel")
    if q: crit[r["_meta"]["id"]] = (list(q["criteria"].values()), list(q["criteria"]).index(q["label"]))
def rows(name):
    out = {}
    for r in scored_rows(json.loads((W / "new" / name / "rows.json").read_text())):
        if not str(r["task"]).startswith("radcases_panel"): continue
        texts, lab = crit[r["id"]]
        assert int(r["label"]) == lab and len(r["p"]) == len(texts), ("option order mismatch", r["id"], r["label"], lab, len(r["p"]), len(texts))
        p = np.clip(np.asarray(r["p"], float), 1e-12, 1); out[r["id"]] = (p / p.sum(), lab, texts)
    return out
S = {s: rows(s) for s in ("v3_27", "v3_9", "stock27", "stock9")}
ids = sorted(set.intersection(*[set(v) for v in S.values()])); assert len(ids) == 132, len(ids)
def correct(p, texts):
    q = p / np.array([prior[t] for t in texts]); return q / q.sum()
def stats(name, corr):
    ok, none_pick_on_panel, none_ok, n_panel, n_none = [], 0, 0, 0, 0
    for i in ids:
        p, lab, texts = S[name][i]
        if corr: p = correct(p, texts)
        pick = int(np.argmax(p)); ok.append(pick == lab); is_none_key = texts[lab] == NONE
        if is_none_key: n_none += 1; none_ok += pick == lab
        else: n_panel += 1; none_pick_on_panel += texts[pick] == NONE
    return np.array(ok, float), {"acc": float(np.mean(ok)), "none_picked_on_panel_keys": none_pick_on_panel / n_panel,
                                  "acc_on_none_keys": none_ok / n_none, "n_panel_keys": n_panel, "n_none_keys": n_none}
res = {"n": len(ids), "train_panel_questions": ntrain, "train_none_share": cnt[NONE] / ntrain,
       "train_prior": {o: round(prior[o], 4) for o in sorted(prior)}, "systems": {}}
V = {}
for s in ("v3_27", "v3_9"):
    V[s] = stats(s, False)[0]; V[s + "_corr"], res["systems"][s + "_corrected"] = stats(s, True); res["systems"][s] = stats(s, False)[1]
for s in ("stock27", "stock9"):
    V[s], res["systems"][s] = stats(s, False)
rng = np.random.default_rng(20261005); B = rng.integers(0, len(ids), (2000, len(ids)))
def pair(a, b):
    d = V[a] - V[b]; bs = d[B].mean(1); return {"d": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
res["pairs"] = {"v3_27_corrected-stock27": pair("v3_27_corr", "stock27"), "v3_27_corrected-v3_27": pair("v3_27_corr", "v3_27"),
                "v3_27-stock27": pair("v3_27", "stock27"), "v3_9_corrected-stock9": pair("v3_9_corr", "stock9"),
                "v3_9_corrected-v3_9": pair("v3_9_corr", "v3_9"), "v3_9-stock9": pair("v3_9", "stock9")}
print(json.dumps(res))
'''
p = subprocess.run([str(KEV_PY), "-c", INNER, str(RADKEV)], env={**os.environ, "PYTHONPATH": str(KEV_DIR)}, capture_output=True, text=True, cwd=str(KEV_DIR))
(OUT / "radcases_prior.json").write_text(json.dumps(json.loads(p.stdout.strip().splitlines()[-1]), indent=1) if p.returncode == 0 else json.dumps({"error": p.stderr[-4000:], "stdout": p.stdout[-1000:]}))
