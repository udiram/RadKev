"""RadCases panel diagnosis (post hoc, read-only, CPU): why the fine-tuned models lose on radcases_panel, and whether the scoring is
fair to every system. Aggregates only (panel names are RadCases' public label set).

    python jobs/submit.py jobs/radcases_panel_diag.py --gpus 0 --cpus 4 --mem 16 --timeout 20 --outputs 'radcases_panel_diag.json'

For every system scored by jobs/eval_v3.py on runs/test-v3/new.jsonl: accuracy by subset and by whether the key is the "None"
option, how often each system picks None, the mean probability it puts on None (key None vs not), the most frequent confusions,
and the rank of the key. Fairness checks: every system's rows cover the same questions with the same option keys and labels as the
records; option order per record is identical for all systems (one file). Training side: label distribution of RadCases panel
questions in data/radcases-v3/{train,dev,test} and in the v3 training mix, and how often a v3 dev-set "None" was predicted.
"""
import json, os, subprocess
from pathlib import Path

OUT = Path(os.environ["ZCB_OUTPUT_DIR"]); RK = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"
INNER = r'''
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from kev.metrics import scored_rows
RK = Path(sys.argv[1]); W = RK / "runs/test-v3"
NONE = "None: no ACR Appropriateness Criteria topic applies"
recs = {}
for l in open(W / "new.jsonl"):
    if not l.strip(): continue
    r = json.loads(l); q = r["questions"].get("panel")
    if q and str(q.get("src", "")).startswith("radcases_panel"): recs[r["_meta"]["id"]] = (r["_meta"].get("subset"), q)
res = {"n_questions": len(recs), "systems": {}, "fairness": {}}
truth = {rid: q["criteria"][q["label"]] for rid, (s, q) in recs.items()}
res["label_dist_test"] = dict(Counter(f"{recs[r][0]}/{'None' if t == NONE else 'panel'}" for r, t in truth.items()).most_common())
for s in ("stock9", "v3_9", "stock27", "v3_27", "qwen38", "medgemma_fix", "kev4", "kev08", "laya_typed", "gliner_decide"):
    f = W / "new" / s / "rows.json"
    if not f.exists(): continue
    rows = [r for r in scored_rows(json.loads(f.read_text())) if r["id"] in recs and r["question"] == "panel"]
    # fairness: same questions, same option keys in the same order, same label as the record
    bad = 0
    for r in rows:
        q = recs[r["id"]][1]; keys = list(q["criteria"])
        if r["keys"] != keys or keys[int(r["label"])] != q["label"]: bad += 1
    res["fairness"][s] = {"rows": len(rows), "of": len(recs), "key_or_label_mismatch": bad}
    acc = defaultdict(lambda: [0, 0]); pick_none = Counter(); pnone = defaultdict(list); conf = Counter(); ranks = []
    for r in rows:
        sub, q = recs[r["id"]]; keys = r["keys"]; p = np.asarray(r["p"], float)
        tr = q["criteria"][keys[int(r["label"])]]; pr = q["criteria"][keys[int(np.argmax(p))]]
        tn = "None" if tr == NONE else "panel"; ok = pr == tr
        for k in (("all",), (sub,), (tn,), (sub, tn)): a = acc["/".join(k)]; a[0] += 1; a[1] += ok
        pick_none[tn] += pr == NONE
        ni = [i for i, k in enumerate(keys) if q["criteria"][k] == NONE][0]; pnone[tn].append(float(p[ni]))
        if not ok: conf[f"{'None' if tr == NONE else tr} -> {'None' if pr == NONE else pr}"] += 1
        ranks.append(int((p > p[int(r["label"])]).sum()) + 1)
    res["systems"][s] = {"acc": {k: {"n": n, "acc": c / n} for k, (n, c) in sorted(acc.items())},
                         "pick_none_rate": {k: pick_none[k] / acc[k][0] for k in ("None", "panel") if acc[k][0]},
                         "mean_p_none": {k: float(np.mean(v)) for k, v in pnone.items()},
                         "top_confusions": conf.most_common(8), "mean_key_rank": float(np.mean(ranks)),
                         "top2_acc": float(np.mean([x <= 2 for x in ranks]))}
# training side
def dist(path):
    c = Counter()
    if not path.exists(): return None
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l)
        for qid, q in r["questions"].items():
            if str(q.get("src", "")).startswith("radcases_panel"):
                c[(str(q["src"]).replace("radcases_panel_", ""), "None" if q["criteria"][q["label"]] == NONE else "panel")] += 1
    return {f"{a}/{b}": n for (a, b), n in sorted(c.items())}
res["radcases_v3_splits"] = {sp: dist(RK / "data/radcases-v3" / f"{sp}.jsonl") for sp in ("train", "dev", "test")}
mixes = sorted(p.name for p in (RK / "data").glob("mix-v3*"))
res["mixes"] = {m: dist(RK / "data" / m / "train.jsonl") for m in mixes}
# how the v3 training mix treats imaging-necessity elsewhere (teacher order 'appropriate' labels)
c = Counter()
for m in ("mix-v3f",):
    p = RK / "data" / m / "train.jsonl"
    if p.exists():
        for l in open(p):
            r = json.loads(l)
            for q in r["questions"].values():
                if q.get("src") == "teacher_order_appropriate": c[(q.get("type"), str((q.get("criteria") or {}).get(q["label"], q["label"]) if isinstance(q.get("criteria"), dict) else q["label"])[:40])] += 1
res["mix_teacher_order_appropriate_labels"] = [[list(k), n] for k, n in c.most_common(6)]
print(json.dumps(res))
'''
p = subprocess.run([str(RK / "kev/.venv/bin/python"), "-c", INNER, str(RK)], env={**os.environ, "PYTHONPATH": str(RK / "kev")}, cwd=RK / "kev",
                   capture_output=True, text=True)
(OUT / "radcases_panel_diag.json").write_text(p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else json.dumps({"error": p.stderr[-2500:]}))
