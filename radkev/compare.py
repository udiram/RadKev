"""Paired comparisons on scored test rows (inside the Kev environment): every model vs a reference, overall, per decision
family and per task, with Kev's source-stratified cluster bootstrap (sibling questions of one record resample together).

    python -m radkev.compare RUNS_DIR --models a,b,c --reference a --out comparisons.json [--data test.jsonl]

RUNS_DIR/<model>/rows.json are the scored rows radkev.evaluate wrote for each model on the same records.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# decision families for reporting; first matching prefix wins
FAMILIES = [("report_cxr_human", ("iu_",)), ("report_cxr", ("chexpertplus_", "mimic_", "teacher_cxr_findings_")), ("report_ct", ("ctrate_",)),
            ("case_diagnosis", ("eurorad_dx",)), ("routing", ("eurorad_route", "teacher_report_route")),
            ("radiology_knowledge", ("medmcqa_rad",)), ("medical_knowledge", ("medmcqa_med", "medqa", "mmlu_", "medxpertqa", "pubmedqa")),
            ("orders_protocols", ("teacher_order_",)), ("triage_followup", ("teacher_report_",))]
KEEP = ("n", "acc", "brier", "ece", "nll", "confident_error_rate", "coverage_at_5pct_error")
# answer keys written by people (MeSH coders, case authors, exam boards) vs machine-derived (dataset NLP labellers, LLM teachers)
HUMAN = {"report_cxr_human", "case_diagnosis", "routing", "radiology_knowledge", "medical_knowledge"}
RADIOLOGY_HUMAN = {"report_cxr_human", "case_diagnosis", "routing", "radiology_knowledge"}
SUBSETS = {"human_keys": HUMAN, "radiology_human_keys": RADIOLOGY_HUMAN}


def family(task):
    return next((f for f, prefixes in FAMILIES if task.startswith(prefixes)), "other")


def summary(rows):
    from kev.metrics import metrics
    from sklearn.metrics import roc_auc_score
    m = metrics(rows); out = {k: m.get(k) for k in KEEP}
    if rows and all(r["type"] == "noul" for r in rows) and len({r["label"] for r in rows}) == 2:
        out["auroc"] = float(roc_auc_score([r["label"] for r in rows], [r["p"][1] for r in rows]))
    return out


def reliability(rows, bins=10):
    """Top-label reliability: per confidence bin, the number of questions, mean confidence and accuracy."""
    conf = np.array([max(r["p"]) for r in rows]); ok = np.array([int(np.argmax(r["p"]) == r["label"]) for r in rows])
    edges = np.linspace(0, 1, bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any(): out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": float(conf[m].mean()), "acc": float(ok[m].mean())})
    return out


def heldout_matchers():
    """Regexes for the instruction wordings that never appear in training (the last template of every list in
    radkev.data / radkev.teacher is reserved for dev/test). Question-level robustness to unseen phrasing."""
    import re

    from radkev import data as bd
    from radkev import teacher as te
    lists = [bd.PRESENT_T, bd.STATUS_T, bd.NORMAL_T, bd.WHICH_T, bd.DX_T, bd.SECTION_T, bd.MCQ_T, bd.PUBMEDQA_T, te.STATUS_T]
    lists += [q[2] for q in te.REPORT_Q + te.ORDER_Q]
    return [re.compile(re.escape(t[-1]).replace(re.escape("{f}"), ".+") + r"$") for t in lists if len(t) > 1]


def wording_of(data_path):
    """(record id, question id) -> 'held_out' | 'seen', for kev.data.load_records ids ("rad/<line>")."""
    if not data_path: return {}
    pats, out = heldout_matchers(), {}
    for n, line in enumerate(open(data_path)):
        if not line.strip(): continue
        for qid, q in json.loads(line)["questions"].items():
            ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
            out[(f"rad/{n}", qid)] = "held_out" if any(pt.match(ins) for pt in pats) else "seen"
    return out


def paired(a_rows, b_rows, metric, aggregation="micro", samples=1000):
    from kev.metrics import paired_bootstrap
    a = {(r["id"], r["question"]): r for r in a_rows}; b = {(r["id"], r["question"]): r for r in b_rows}
    keys = sorted(a.keys() & b.keys())
    if len(keys) < 20: return None
    res = paired_bootstrap([a[k] for k in keys], [b[k] for k in keys], samples=samples, metric=metric, aggregation=aggregation)
    return {"n": len(keys), "delta": res[f"{aggregation}_{metric}_delta"], "ci95": res["ci95"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--models", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--data", default="", help="the scored records file, for the unseen-wording robustness split")
    ap.add_argument("--also", default="", help="further references (e.g. the fine-tuned candidates) for overall + family deltas")
    a = ap.parse_args()
    from kev.metrics import scored_rows
    wording = wording_of(a.data)
    runs = Path(a.runs)
    rows = {n: scored_rows(json.loads((runs / n / "rows.json").read_text())) for n in a.models.split(",") if (runs / n / "rows.json").exists()}
    out = {"reference": a.reference, "models": {}, "vs_reference": {}, "families": dict(FAMILIES)}
    for name, rs in rows.items():
        fams, tasks = {}, {}
        for r in rs: fams.setdefault(family(r["task"]), []).append(r); tasks.setdefault(r["task"], []).append(r)
        out["models"][name] = {"overall": summary(rs), "families": {f: summary(v) for f, v in fams.items()}, "tasks": {t: summary(v) for t, v in tasks.items()},
                               "subsets": {k: summary([r for r in rs if family(r["task"]) in fs]) for k, fs in SUBSETS.items() if any(family(r["task"]) in fs for r in rs)},
                               "reliability": reliability(rs), "reliability_families": {f: reliability(v) for f, v in fams.items()}}
        if wording:
            split = {}
            for r in rs: split.setdefault(wording.get((r["id"], r["question"]), "unknown"), []).append(r)
            out["models"][name]["wording"] = {k: summary(v) for k, v in split.items()}
    def against(ref, with_tasks):
        res = {}
        for name, rs in rows.items():
            if rs is ref: continue
            comp = {"overall_macro_acc": paired(rs, ref, "acc", "macro", a.samples), "overall_micro_acc": paired(rs, ref, "acc", "micro", a.samples),
                    "overall_brier": paired(rs, ref, "brier", "micro", a.samples), "subsets": {}, "families": {}, "tasks": {}}
            for k, fs in SUBSETS.items():
                sa = [r for r in rs if family(r["task"]) in fs]; sb = [r for r in ref if family(r["task"]) in fs]
                comp["subsets"][k] = {"macro_acc": paired(sa, sb, "acc", "macro", a.samples), "micro_acc": paired(sa, sb, "acc", "micro", a.samples)}
            for f in {family(r["task"]) for r in rs}:
                fa = [r for r in rs if family(r["task"]) == f]; fb = [r for r in ref if family(r["task"]) == f]
                comp["families"][f] = {"acc": paired(fa, fb, "acc", "micro", a.samples), "brier": paired(fa, fb, "brier", "micro", a.samples)}
            if with_tasks:
                for t in {r["task"] for r in rs}:
                    comp["tasks"][t] = {"acc": paired([r for r in rs if r["task"] == t], [r for r in ref if r["task"] == t], "acc", "micro", a.samples)}
            res[name] = comp
        return res
    if a.reference in rows: out["vs_reference"] = against(rows[a.reference], True)
    # deltas are model minus reference: out["vs"][r][m] is m - r (so vs["v2_27"]["medgemma"] < 0 means RadKev is ahead)
    out["vs"] = {r: against(rows[r], False) for r in a.also.split(",") if r in rows}
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps({n: v["overall"] for n, v in out["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
