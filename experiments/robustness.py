"""Shared-bootstrap statistics on the scored held-out test rows (post hoc; CPU only; no model is run).

    python experiments/robustness.py                         # needs $RADKEV_HOME/runs/test-final/ (experiments/final_test.py --tag final)
    python experiments/robustness.py --demo demo_states.json # demonstration sample given explicitly (see demo_hashes)

Reads runs/test-final/<model>/rows.json for every scored model (Kev's scored rows: record id, question id, task, label, p)
and computes, with ONE set of bootstrap weights shared by all models (B resamples of records, clusters = records, stratified
by source, seed 20261002, so sibling questions resample together and every paired difference uses the same resamples):
  per model    point estimate + 95% percentile interval for micro and macro (mean over tasks) accuracy, overall, human-key,
               radiology human-key, per family and per task; ECE (10 equal-width top-label bins) and multi-class Brier
  pairs        paired differences (a minus b) with 95% interval and a two-sided bootstrap p value, Holm-adjusted within the
               listed comparison set; per family and per task for the main pairs
  demo420      the same main pairs with the 420 demonstration records (matched by state hash) removed
  wording      main pairs split by seen vs held-out instruction wording (radkev.compare.wording_of)
  selective    risk-coverage curves (accuracy at 10..100% coverage, most confident first) and coverage at <=1/2/5/10% error,
               with intervals (300 resamples)
Every point estimate is also checked against Kev's own summary (acc, ece) and the difference is reported.
These are the intervals, p values and Holm adjustments the paper reports (B = 2,000); radkev.compare (experiments/final_test.py)
uses Kev's paired bootstrap with 1,000 resamples and is not the source of the reported intervals.

The demonstration sample is the 60 records per open-licence source drawn from data/rad-open/test.jsonl with random.Random(0)
before the final evaluation; demo_hashes() redraws it. --demo takes a JSON list of state hashes, or a JSON file whose
"items" carry the demonstration records' states, instead.
Writes aggregates only: $RADKEV_HOME/runs/robustness/robustness.json (or --out).
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from radkev import compare as C
from radkev.paths import DATA, RUNS

TESTDIR = RUNS / "test-final"
OPEN = {"iu", "eurorad", "medmcqa", "medqa", "mmlu_med", "pubmedqa", "medxpertqa"}   # open-licence sources of the demonstration sample
SEED = 20261002


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def demo_hashes(path="", per_source=60):
    """State hashes of the demonstration sample: from --demo if given, else redrawn from data/rad-open/test.jsonl."""
    if path:
        d = json.loads(Path(path).read_text())
        return set(d) if isinstance(d, list) else {state_hash(it["state"]) for it in d["items"]}
    by_src = {}
    for line in (DATA / "rad-open" / "test.jsonl").read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["_meta"]["source"] in OPEN: by_src.setdefault(r["_meta"]["source"], []).append(r)
    rng, recs = random.Random(0), []
    for src, rs in sorted(by_src.items()): recs += rng.sample(rs, min(per_source, len(rs)))
    return {state_hash(r["state"]) for r in recs}


def stratified_weights(recs, src_of, B):
    """B x records bootstrap counts, records resampled with replacement within their source (shared by every model)."""
    ridx = {r: i for i, r in enumerate(recs)}
    strata = {}
    for r in recs: strata.setdefault(src_of[r], []).append(ridx[r])
    rng = np.random.default_rng(SEED)
    W = np.zeros((B, len(recs)), np.float32)
    for s, members in strata.items():
        members = np.array(members)
        draws = rng.integers(0, len(members), size=(B, len(members)))
        for b in range(B): np.add.at(W[b], members[draws[b]], 1)
    return W


def analyse(testdir, out_path, B, play):
    from kev.metrics import scored_rows
    t0 = time.time()
    models = sorted(p.name for p in testdir.iterdir() if (p / "rows.json").exists())
    rows = {m: scored_rows(json.loads((testdir / m / "rows.json").read_text())) for m in models}
    res = {"models_found": models, "row_keys": sorted(rows[models[0]][0].keys()) if models else []}
    keysets = {m: {(r["id"], r["question"]) for r in rs} for m, rs in rows.items()}
    res["n_per_model"] = {m: len(k) for m, k in keysets.items()}
    maxn = max(len(k) for k in keysets.values())
    res["excluded_models"] = [m for m in models if len(keysets[m]) < maxn]   # scored on fewer questions: left out, not shrunk to
    models = [m for m in models if len(keysets[m]) == maxn]
    common = set.intersection(*[keysets[m] for m in models])
    res["n_common"] = len(common)
    keys = sorted(common)
    ref = {(r["id"], r["question"]): r for r in rows[models[0]]}
    task = np.array([ref[k]["task"] for k in keys]); fam = np.array([C.family(t) for t in task])
    rec = np.array([k[0] for k in keys])
    # record -> source stratum (the task prefix of its first question), record -> demo flag (state hash)
    src_of = {}
    for k in keys: src_of.setdefault(k[0], ref[k]["task"].split("_")[0])
    demo_rec = set()
    for n, line in enumerate(open(testdir / "test.jsonl")):
        if not line.strip(): continue
        if state_hash(json.loads(line)["state"]) in play: demo_rec.add(f"rad/{n}")
    res["demo420"] = {"records_matched": len(demo_rec & set(rec.tolist())), "questions_matched": int(np.isin(rec, list(demo_rec)).sum())}
    wording = C.wording_of(str(testdir / "test.jsonl"))
    word = np.array([wording.get(k, "unknown") for k in keys])
    # correctness / confidence / brier per model, aligned to keys
    CORR, CONF, BRIER = {}, {}, {}
    for m in models:
        d = {(r["id"], r["question"]): r for r in rows[m]}
        c = np.zeros(len(keys), bool); f = np.zeros(len(keys)); b = np.zeros(len(keys))
        for i, k in enumerate(keys):
            r = d[k]; p = np.asarray(r["p"], float); c[i] = int(np.argmax(p)) == int(r["label"]); f[i] = p.max()
            y = np.zeros_like(p); y[int(r["label"])] = 1; b[i] = float(((p - y) ** 2).sum())
        CORR[m], CONF[m], BRIER[m] = c, f, b
    # bootstrap weights over records, stratified by source
    recs = sorted(set(rec.tolist())); ridx = {r: i for i, r in enumerate(recs)}; q2r = np.array([ridx[r] for r in rec])
    QW = stratified_weights(recs, src_of, B)[:, q2r]   # B x Q question weights (a record's questions share its weight)

    def micro(x, mask):
        w = QW[:, mask]; return float(x[mask].mean()), (w @ x[mask]) / w.sum(1)

    def macro(x, mask):
        ts = sorted(set(task[mask].tolist())); pts, bs = [], []
        for t in ts:
            mm = mask & (task == t); w = QW[:, mm]; pts.append(x[mm].mean()); bs.append((w @ x[mm]) / np.maximum(w.sum(1), 1e-9))
        return float(np.mean(pts)), np.mean(bs, 0)

    def ci(bs): return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    def pval(d): return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    ALL = np.ones(len(keys), bool); HK = np.isin(fam, list(C.HUMAN)); RHK = np.isin(fam, list(C.RADIOLOGY_HUMAN))
    SUBS = {"overall": ALL, "human_keys": HK, "radiology_human_keys": RHK}

    def ece(conf, corr, w=None, bins=10):
        e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1)
        if w is None: w = np.ones_like(conf)
        tot = w.sum(); s = 0.0
        for k in range(bins):
            mk = idx == k
            if mk.any():
                ww = w[mk]; s += ww.sum() / tot * abs((ww * corr[mk]).sum() / ww.sum() - (ww * conf[mk]).sum() / ww.sum())
        return float(s)
    per = {}
    for m in models:
        c = CORR[m].astype(float); o = {}
        for sname, mask in SUBS.items():
            p1, b1 = micro(c, mask); p2, b2 = macro(c, mask)
            o[sname] = {"n": int(mask.sum()), "micro": p1, "micro_ci": ci(b1), "macro": p2, "macro_ci": ci(b2), "brier": float(BRIER[m][mask].mean()),
                        "ece": ece(CONF[m][mask], c[mask])}
        o["families"] = {}
        for f_ in sorted(set(fam.tolist())):
            mask = fam == f_; p1, b1 = micro(c, mask); o["families"][f_] = {"n": int(mask.sum()), "acc": p1, "ci": ci(b1)}
        o["tasks"] = {}
        for t in sorted(set(task.tolist())):
            mask = task == t; p1, b1 = micro(c, mask); o["tasks"][t] = {"n": int(mask.sum()), "acc": p1, "ci": ci(b1)}
        # ECE intervals (first 400 resamples; weighted bins)
        eb = [ece(CONF[m][HK], c[HK], QW[b, HK]) for b in range(min(B, 400))]; o["human_keys"]["ece_ci"] = ci(np.array(eb))
        eb = [ece(CONF[m], c, QW[b]) for b in range(min(B, 400))]; o["overall"]["ece_ci"] = ci(np.array(eb))
        try:
            s = json.loads((testdir / m / "summary.json").read_text())["overall"]
            o["check_vs_kev"] = {"acc_diff": o["overall"]["micro"] - s["acc"], "ece_kev": s.get("ece"), "brier_kev": s.get("brier")}
        except Exception as e: o["check_vs_kev"] = {"error": str(e)[:200]}
        per[m] = o
    res["per_model"] = per
    PAIRS = [("v2_27", "stock27"), ("v2_27", "qwen38"), ("v2_27", "medgemma_brief"), ("v2_27", "r9"), ("v2_27", "v1med27"), ("v2_27", "v0open27"),
             ("r9", "stock9"), ("r9", "qwen38"), ("r9", "b9"), ("r9", "stock27"), ("r9f10", "b9f10"), ("r9", "r9f10"),
             ("stock27", "stock9"), ("qwen38", "stock27"), ("qwen38", "medgemma_brief"), ("stock27", "medgemma_brief"), ("ft9", "stock9"),
             ("v2_27", "medgemma_fix"), ("r9", "medgemma_fix"), ("qwen38", "medgemma_fix"), ("stock27", "medgemma_fix")]
    MAIN = [("v2_27", "stock27"), ("v2_27", "qwen38"), ("v2_27", "medgemma_brief"), ("v2_27", "medgemma_fix"), ("r9", "medgemma_fix"), ("r9", "stock9"),
            ("r9", "qwen38"), ("r9", "b9"), ("stock27", "stock9")]

    def pair(a, b, extra_mask=None):
        ca, cb = CORR[a].astype(float), CORR[b].astype(float); out = {}
        for sname, mask in SUBS.items():
            mk = mask if extra_mask is None else mask & extra_mask
            pa, ba = micro(ca, mk); pb, bb = micro(cb, mk); d = ba - bb
            qa, qa_b = macro(ca, mk); qb, qb_b = macro(cb, mk); dm = qa_b - qb_b
            out[sname] = {"n": int(mk.sum()), "micro": pa - pb, "micro_ci": ci(d), "micro_p": pval(d), "macro": qa - qb, "macro_ci": ci(dm), "macro_p": pval(dm),
                          "brier": float(BRIER[a][mk].mean() - BRIER[b][mk].mean())}
        return out
    PAIRS = [(a, b) for a, b in PAIRS if a in CORR and b in CORR]
    MAIN = [(a, b) for a, b in MAIN if a in CORR and b in CORR]
    pairs = {f"{a}-{b}": pair(a, b) for a, b in PAIRS}
    for sname in SUBS:   # Holm adjustment within the comparison set, per subset and aggregation
        for agg in ("micro", "macro"):
            lst = sorted([(v[sname][f"{agg}_p"], k) for k, v in pairs.items()]); m_ = len(lst); run = 0.0
            for i, (p, k) in enumerate(lst):
                run = max(run, min(1.0, (m_ - i) * p)); pairs[k][sname][f"{agg}_p_holm"] = run
    res["pairs"] = pairs
    detail = {}
    for a, b in MAIN:
        ca, cb = CORR[a].astype(float), CORR[b].astype(float); o = {"families": {}, "tasks": {}}
        for f_ in sorted(set(fam.tolist())):
            mk = fam == f_; pa, ba = micro(ca, mk); pb, bb = micro(cb, mk); o["families"][f_] = {"n": int(mk.sum()), "d": pa - pb, "ci": ci(ba - bb), "p": pval(ba - bb)}
        for t in sorted(set(task.tolist())):
            mk = task == t; pa, ba = micro(ca, mk); pb, bb = micro(cb, mk); o["tasks"][t] = {"n": int(mk.sum()), "d": pa - pb, "ci": ci(ba - bb), "p": pval(ba - bb)}
        detail[f"{a}-{b}"] = o
    res["pair_detail"] = detail
    nodemo = ~np.isin(rec, list(demo_rec))
    res["demo420"]["pairs_excluded"] = {f"{a}-{b}": pair(a, b, nodemo) for a, b in MAIN}
    res["wording"] = {}
    for wv in ("seen", "held_out"):
        mk = word == wv
        if mk.sum() > 50: res["wording"][wv] = {f"{a}-{b}": pair(a, b, mk) for a, b in MAIN}
    res["wording_counts"] = {w_: int((word == w_).sum()) for w_ in set(word.tolist())}
    SEL = ["v2_27", "stock27", "r9", "stock9", "qwen38", "medgemma_brief", "medgemma_fix", "b9"]
    cov_grid = np.round(np.arange(0.10, 1.0001, 0.05), 2)

    def sel_curve(m, mask, w):
        conf, err = CONF[m][mask], (~CORR[m][mask]).astype(float); order = np.argsort(-conf, kind="stable")
        ww = w[mask][order]; e = err[order]; cw = np.cumsum(ww); ce = np.cumsum(ww * e); tot = cw[-1]
        cov = cw / tot; risk = ce / cw
        acc_at = [1 - float(risk[min(len(risk) - 1, np.searchsorted(cov, g))]) for g in cov_grid]
        cov_at = {}
        for tau in (0.01, 0.02, 0.05, 0.10):
            ok = np.where(risk <= tau)[0]; cov_at[str(tau)] = float(cov[ok[-1]]) if len(ok) else 0.0
        return acc_at, cov_at
    sel = {}
    for m in SEL:
        if m not in CORR: continue
        sel[m] = {}
        for sname, mask in (("overall", ALL), ("human_keys", HK)):
            a0, c0 = sel_curve(m, mask, np.ones(len(keys)))
            bs = [sel_curve(m, mask, QW[b]) for b in range(min(B, 300))]
            acc_ci = [ci(np.array([x[0][i] for x in bs])) for i in range(len(cov_grid))]
            cov_ci = {t: ci(np.array([x[1][t] for x in bs])) for t in c0}
            sel[m][sname] = {"coverage": cov_grid.tolist(), "acc": a0, "acc_ci": acc_ci, "coverage_at_error": c0, "coverage_at_error_ci": cov_ci}
    res["selective"] = sel
    res["B"] = B; res["seconds"] = round(time.time() - t0, 1)
    Path(out_path).write_text(json.dumps(res, separators=(",", ":")))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--testdir", default=str(TESTDIR), help="scored test rows, one directory per model, plus test.jsonl")
    ap.add_argument("--samples", type=int, default=2000, help="bootstrap resamples shared by every system (the paper: 2,000)")
    ap.add_argument("--demo", default="", help="JSON list of demonstration-sample state hashes, or a JSON file with items[].state")
    ap.add_argument("--out", default="", help="output file (default: $RADKEV_HOME/runs/robustness/robustness.json)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else RUNS / "robustness" / "robustness.json"; out.parent.mkdir(parents=True, exist_ok=True)
    analyse(Path(a.testdir), out, a.samples, demo_hashes(a.demo))
    print(out)


if __name__ == "__main__":
    main()
