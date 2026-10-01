"""Sensitivity analysis: how much of the orders/protocols result comes from questions whose clinical question already names
the imaging that was done? Post-hoc, on the held-out test predictions already computed; no model is run or changed.

    python experiments/leak_sensitivity.py          # CPU only; reads $RADKEV_HOME/runs/test-final

A teacher-labelled order question is "leaky" when its clinical question / indication text matches radkev.data.IMAGING_RE
(the gold-set v2 filter: MRI, CT, ultrasound, radiograph, scan, ...). The "ordered exam" field of appropriateness questions
names the exam by design and is not checked. For every model: accuracy on leaky vs leak-free order questions, and the
paired difference vs stock Kev-27B on the leak-free subset (record-clustered bootstrap, 2,000 resamples). Aggregates only.
"""
import json
import random
import re

from radkev import data as bd
from radkev.paths import RUNS

WORK = RUNS / "test-final"
MODELS = ["stock27", "v2_27", "v1med27", "v0open27", "stock9", "ft9", "qwen38", "medgemma", "laya"]


def main():
    recs = [json.loads(l) for l in (WORK / "test.jsonl").read_text().splitlines() if l.strip()]

    def clinical_text(state):
        if isinstance(state, dict):
            return " ".join(str(v) for k, v in state.items() if k.lower() not in ("ordered exam",))
        return re.sub(r"(?im)^ordered exam:.*$", "", str(state))

    leak = {i: bool(bd.IMAGING_RE.search(clinical_text(r["state"]))) for i, r in enumerate(recs)}
    origin = {i: str(r["_meta"].get("group_id", "")).split("/")[0] for i, r in enumerate(recs)}
    rep = {"definition": "order question (task teacher_order_*) whose clinical question/indication names an imaging test (radkev.data.IMAGING_RE)", "models": {}}
    correct = {}
    for m in MODELS:
        p = WORK / m / "rows.json"
        if not p.exists(): continue
        rows = json.loads(p.read_text())
        if isinstance(rows, dict): rows = rows.get("rows", list(rows.values()))
        for r in rows:
            if not str(r.get("task", "")).startswith("teacher_order_"): continue
            tail = str(r.get("parent") or r.get("id")).split("/")[-1].split(":")[0]
            if not tail.isdigit(): continue
            i = int(tail); pr = r["p"]
            correct[(m, i, r["question"])] = (max(range(len(pr)), key=lambda j: pr[j]) == r["label"], leak[i], r["task"], origin[i])
    keys = sorted({(i, q) for (_, i, q) in correct})
    rep["n_questions"] = len(keys); rep["n_leaky"] = sum(leak[i] for i, _ in keys)
    rep["leaky_by_origin"] = {}
    for i, q in keys:
        o = origin[i]; d = rep["leaky_by_origin"].setdefault(o, {"n": 0, "leaky": 0}); d["n"] += 1; d["leaky"] += leak[i]
    for m in MODELS:
        c = {k: v for k, v in correct.items() if k[0] == m}
        if not c: continue
        acc = lambda sel: (sum(v[0] for v in sel) / len(sel), len(sel)) if sel else (None, 0)
        a_all, n_all = acc(list(c.values())); a_l, n_l = acc([v for v in c.values() if v[1]]); a_c, n_c = acc([v for v in c.values() if not v[1]])
        rep["models"][m] = {"all": {"acc": a_all, "n": n_all}, "leaky": {"acc": a_l, "n": n_l}, "leak_free": {"acc": a_c, "n": n_c},
                            "leak_free_by_task": {t: acc([v for v in c.values() if not v[1] and v[2] == t])[0] for t in sorted({v[2] for v in c.values()})}}
    # paired delta vs stock27 on the leak-free subset, clustered by record
    rng = random.Random(0)
    clean = [(i, q) for i, q in keys if not leak[i]]
    by_rec = {}
    for i, q in clean: by_rec.setdefault(i, []).append(q)
    recs_ids = sorted(by_rec)
    for m in MODELS:
        if m == "stock27" or m not in rep["models"]: continue
        def diff(sample):
            tot = n = 0
            for i in sample:
                for q in by_rec[i]:
                    a, b = correct.get((m, i, q)), correct.get(("stock27", i, q))
                    if a and b: tot += int(a[0]) - int(b[0]); n += 1
            return tot / n if n else 0.0
        point = diff(recs_ids)
        boots = sorted(diff([recs_ids[rng.randrange(len(recs_ids))] for _ in recs_ids]) for _ in range(2000))
        rep["models"][m]["leak_free_delta_vs_stock27"] = {"delta": point, "ci95": [boots[50], boots[1949]], "records": len(recs_ids)}
    (WORK / "leak_sensitivity.json").write_text(json.dumps(rep, indent=1))
    print(WORK / "leak_sensitivity.json")


if __name__ == "__main__":
    main()
