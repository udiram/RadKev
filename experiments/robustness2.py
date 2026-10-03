"""Shared-bootstrap statistics, second set (post hoc; CPU only; no model is run). Same rows and the same resamples as
experiments/robustness.py (identical seed and source stratification), for the prespecified analysis and the post hoc checks.

    python experiments/robustness2.py                        # needs $RADKEV_HOME/runs/test-final/ (experiments/final_test.py --tag final)

  nodemo       the prespecified analysis: every system and every paired comparison with the 420 demonstration records
               (matched by state hash) removed; Holm adjustment over the comparison set and over the primary pair's families
  calibration  paired differences in ECE (10 bins), confident-error rate (max p >= 0.9 and wrong), coverage at <= 5% error and
               multi-class Brier, with 95% intervals (the first 800 resamples)
  did          differences of differences on identical resamples: specialization vs scale, initialization x data size, data
               size within each initialization, wording (held-out minus seen) for the main pairs
  recal        every system recalibrated by temperature scaling cross-fitted on the test rows (two folds by record; T on the
               grid 2^(k/30) in [0.25, 4] by minimum NLL on one fold, applied to the other): ECE, confident errors, coverage
               at <= 5% error before and after; accuracy is unchanged by construction
  family_macro the primary comparison aggregated as the mean over families (all, human-key) instead of over tasks
  equivalence  90% intervals for the pairs whose 95% interval includes 0
  majority     per task, the accuracy of always answering the most frequent label index (a prevalence baseline)
  eurorad_leak Eurorad diagnosis questions split by whether a content word of the correct option (>= 6 letters, absent from every
               distractor) occurs in the case text; accuracy per stratum and the RadKev gain per stratum
Writes aggregates only: $RADKEV_HOME/runs/robustness/robustness2.json (or --out).
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from robustness import TESTDIR, demo_hashes, state_hash, stratified_weights

from radkev import compare as C
from radkev.paths import RUNS


def analyse(testdir, out_path, B, play):
    from kev.metrics import scored_rows
    t0 = time.time()
    models = sorted(p.name for p in testdir.iterdir() if (p / "rows.json").exists())
    rows = {m: scored_rows(json.loads((testdir / m / "rows.json").read_text())) for m in models}
    keysets = {m: {(r["id"], r["question"]) for r in rs} for m, rs in rows.items()}
    maxn = max(len(k) for k in keysets.values()); models = [m for m in models if len(keysets[m]) == maxn]
    keys = sorted(set.intersection(*[keysets[m] for m in models]))
    ref = {(r["id"], r["question"]): r for r in rows[models[0]]}
    task = np.array([ref[k]["task"] for k in keys]); fam = np.array([C.family(t) for t in task]); rec = np.array([k[0] for k in keys])
    src_of = {}
    for k in keys: src_of.setdefault(k[0], ref[k]["task"].split("_")[0])
    demo_rec, state_of, q_of = set(), {}, {}
    for n, line in enumerate(open(testdir / "test.jsonl")):
        if not line.strip(): continue
        r = json.loads(line); st = r["state"]
        if state_hash(st) in play: demo_rec.add(f"rad/{n}")
        state_of[f"rad/{n}"] = st
        for qid, q in r["questions"].items(): q_of[(f"rad/{n}", qid)] = q
    wording = C.wording_of(str(testdir / "test.jsonl")); word = np.array([wording.get(k, "unknown") for k in keys])
    CORR, CONF, BRIER, LOGP = {}, {}, {}, {}
    for m in models:
        d = {(r["id"], r["question"]): r for r in rows[m]}
        c = np.zeros(len(keys), bool); f = np.zeros(len(keys)); b = np.zeros(len(keys)); lp = []
        for i, k in enumerate(keys):
            r = d[k]; p = np.clip(np.asarray(r["p"], float), 1e-12, 1); p = p / p.sum(); lab = int(r["label"])
            c[i] = int(np.argmax(p)) == lab; f[i] = p.max(); y = np.zeros_like(p); y[lab] = 1; b[i] = float(((p - y) ** 2).sum()); lp.append((np.log(p), lab))
        CORR[m], CONF[m], BRIER[m], LOGP[m] = c, f, b, lp
    recs = sorted(set(rec.tolist())); ridx = {r: i for i, r in enumerate(recs)}; q2r = np.array([ridx[r] for r in rec])
    QW = stratified_weights(recs, src_of, B)[:, q2r]

    def micro(x, mask): w = QW[:, mask]; return float(x[mask].mean()), (w @ x[mask]) / np.maximum(w.sum(1), 1e-9)

    def macro(x, mask):
        pts, bs = [], []
        for t in sorted(set(task[mask].tolist())):
            mm = mask & (task == t); w = QW[:, mm]; pts.append(x[mm].mean()); bs.append((w @ x[mm]) / np.maximum(w.sum(1), 1e-9))
        return float(np.mean(pts)), np.mean(bs, 0)

    def fmacro(x, mask):
        pts, bs = [], []
        for f_ in sorted(set(fam[mask].tolist())):
            mm = mask & (fam == f_); w = QW[:, mm]; pts.append(x[mm].mean()); bs.append((w @ x[mm]) / np.maximum(w.sum(1), 1e-9))
        return float(np.mean(pts)), np.mean(bs, 0)
    ci = lambda v, a=2.5: [float(np.percentile(v, a)), float(np.percentile(v, 100 - a))]
    pval = lambda d: float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    ALL = np.ones(len(keys), bool); HK = np.isin(fam, list(C.HUMAN)); RHK = np.isin(fam, list(C.RADIOLOGY_HUMAN))
    SUBS = {"overall": ALL, "human_keys": HK, "radiology_human_keys": RHK}
    NODEMO = ~np.isin(rec, list(demo_rec))
    res = {"n": len(keys), "demo_questions": int((~NODEMO).sum()), "models": models}
    PAIRS = [("v2_27", "stock27"), ("v2_27", "qwen38"), ("v2_27", "medgemma_brief"), ("v2_27", "r9"), ("v2_27", "v1med27"), ("v2_27", "v0open27"),
             ("r9", "stock9"), ("r9", "stock27"), ("r9", "qwen38"), ("r9", "b9"), ("r9f10", "b9f10"), ("r9", "r9f10"), ("b9", "b9f10"),
             ("stock27", "stock9"), ("qwen38", "stock27"), ("qwen38", "medgemma_brief"), ("stock27", "medgemma_brief"), ("ft9", "stock9"),
             ("v2_27", "medgemma_fix"), ("r9", "medgemma_fix"), ("qwen38", "medgemma_fix"), ("stock27", "medgemma_fix")]
    PAIRS = [(a, b) for a, b in PAIRS if a in CORR and b in CORR]
    # ---- nodemo: the prespecified analysis
    nd = {"per_model": {}, "pairs": {}, "primary_families": {}}
    for m in models:
        x = CORR[m].astype(float); nd["per_model"][m] = {}
        for s, mask in SUBS.items():
            mk = mask & NODEMO; a, ab = micro(x, mk); g, gb = macro(x, mk)
            nd["per_model"][m][s] = {"n": int(mk.sum()), "micro": a, "micro_ci": ci(ab), "macro": g, "macro_ci": ci(gb)}
    for a_, b_ in PAIRS:
        xa, xb = CORR[a_].astype(float), CORR[b_].astype(float); o = {}
        for s, mask in SUBS.items():
            mk = mask & NODEMO; pa, ba = micro(xa, mk); pb, bb = micro(xb, mk); qa, qab = macro(xa, mk); qb, qbb = macro(xb, mk)
            o[s] = {"n": int(mk.sum()), "micro": pa - pb, "micro_ci": ci(ba - bb), "micro_p": pval(ba - bb), "macro": qa - qb, "macro_ci": ci(qab - qbb),
                    "macro_p": pval(qab - qbb)}
        nd["pairs"][f"{a_}-{b_}"] = o
    for s in SUBS:
        for agg in ("micro", "macro"):
            lst = sorted([(v[s][f"{agg}_p"], k) for k, v in nd["pairs"].items()]); m_ = len(lst); run = 0.0
            for i, (p, k) in enumerate(lst): run = max(run, min(1.0, (m_ - i) * p)); nd["pairs"][k][s][f"{agg}_p_holm"] = run
    xa, xb = CORR["v2_27"].astype(float), CORR["stock27"].astype(float)
    for f_ in sorted(set(fam.tolist())):
        mk = (fam == f_) & NODEMO; pa, ba = micro(xa, mk); pb, bb = micro(xb, mk)
        nd["primary_families"][f_] = {"n": int(mk.sum()), "d": pa - pb, "ci": ci(ba - bb), "p": pval(ba - bb)}
    fl = sorted([(v["p"], k) for k, v in nd["primary_families"].items()]); run = 0.0
    for i, (p, k) in enumerate(fl): run = max(run, min(1.0, (len(fl) - i) * p)); nd["primary_families"][k]["p_holm"] = run
    res["nodemo"] = nd

    # ---- calibration metrics and paired calibration differences
    def ece(conf, corr, w, bins=10):
        idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1); tot = w.sum(); s = 0.0
        for k in range(bins):
            mk = idx == k
            if mk.any():
                ww = w[mk]; sw = ww.sum()
                if sw > 0: s += sw / tot * abs((ww * corr[mk]).sum() / sw - (ww * conf[mk]).sum() / sw)
        return s

    def conf_err(conf, corr, w): return float((w * ((conf >= 0.9) & (corr == 0))).sum() / w.sum())

    def cov5(conf, corr, w, tau=0.05):
        o = np.argsort(-conf, kind="stable"); ww = w[o]; e = 1 - corr[o]; cw = np.cumsum(ww); risk = np.cumsum(ww * e) / np.maximum(cw, 1e-9)
        ok = np.where(risk <= tau)[0]; return float(cw[ok[-1]] / cw[-1]) if len(ok) else 0.0

    def calib(conf, corr, brier, mask, w): return {"ece": ece(conf[mask], corr[mask], w[mask]), "conf_err": conf_err(conf[mask], corr[mask], w[mask]),
                                                  "cov5": cov5(conf[mask], corr[mask], w[mask]), "brier": float((w[mask] * brier[mask]).sum() / w[mask].sum())}
    NB = min(B, 800); ones = np.ones(len(keys))
    cal = {"per_model": {}, "pairs": {}}
    CALP = [("v2_27", "stock27"), ("v2_27", "qwen38"), ("v2_27", "medgemma_brief"), ("v2_27", "medgemma_fix"), ("r9", "stock9"), ("r9", "qwen38"), ("r9", "b9"),
            ("stock27", "qwen38")]
    boot = {}
    for m in {x for p in CALP for x in p} | {"stock9", "b9", "v1med27", "medgemma_fix"}:
        if m not in CORR: continue
        c = CORR[m].astype(float); cal["per_model"][m] = {}
        for s, mask in (("overall", ALL), ("human_keys", HK)):
            pt = calib(CONF[m], c, BRIER[m], mask, ones); bs = [calib(CONF[m], c, BRIER[m], mask, QW[b]) for b in range(NB)]
            boot[(m, s)] = bs
            cal["per_model"][m][s] = {k: {"est": pt[k], "ci": ci(np.array([x[k] for x in bs]))} for k in pt}
    for a_, b_ in CALP:
        if (a_, "overall") not in boot or (b_, "overall") not in boot: continue
        cal["pairs"][f"{a_}-{b_}"] = {}
        for s in ("overall", "human_keys"):
            cal["pairs"][f"{a_}-{b_}"][s] = {k: {"d": cal["per_model"][a_][s][k]["est"] - cal["per_model"][b_][s][k]["est"],
                                                "ci": ci(np.array([x[k] - y[k] for x, y in zip(boot[(a_, s)], boot[(b_, s)])]))}
                                            for k in ("ece", "conf_err", "cov5", "brier")}
    res["calibration"] = cal

    # ---- differences of differences
    def dvec(a_, b_, mask, agg=micro): pa, ba = agg(CORR[a_].astype(float), mask); pb, bb = agg(CORR[b_].astype(float), mask); return pa - pb, ba - bb
    did = {}

    def put(name, x, y, s):
        (px, bx), (py, by) = x, y
        did.setdefault(name, {})[s] = {"first": px, "first_ci": ci(bx), "second": py, "second_ci": ci(by), "did": px - py, "did_ci": ci(bx - by), "p": pval(bx - by)}
    for s, mask in SUBS.items():
        for agg, tag in ((micro, ""), (macro, "_macro")):
            put("specialisation27_minus_scale" + tag, dvec("v2_27", "stock27", mask, agg), dvec("stock27", "stock9", mask, agg), s)
            put("specialisation9_minus_scale" + tag, dvec("r9", "stock9", mask, agg), dvec("stock27", "stock9", mask, agg), s)
            put("specialisation27_minus_specialisation9" + tag, dvec("v2_27", "stock27", mask, agg), dvec("r9", "stock9", mask, agg), s)
            put("init_full_minus_init_10pct" + tag, dvec("r9", "b9", mask, agg), dvec("r9f10", "b9f10", mask, agg), s)
            put("data_kevinit_minus_data_baseinit" + tag, dvec("r9", "r9f10", mask, agg), dvec("b9", "b9f10", mask, agg), s)
    for a_, b_ in (("v2_27", "stock27"), ("r9", "stock9"), ("v2_27", "qwen38"), ("r9", "qwen38")):
        for s, mask in SUBS.items():
            put(f"wording_{a_}-{b_}_heldout_minus_seen", dvec(a_, b_, mask & (word == "held_out")), dvec(a_, b_, mask & (word == "seen")), s)
    res["did"] = did

    # ---- recalibration (cross-fitted temperature scaling)
    grid = 2.0 ** (np.arange(-60, 61) / 30.0)
    fold = np.array([int(hashlib.sha256(r.encode()).hexdigest(), 16) % 2 for r in rec])

    def mat(lp):
        K = max(len(l) for l, _ in lp); L = np.full((len(lp), K), -np.inf); lab = np.array([b for _, b in lp])
        for i, (l, _) in enumerate(lp): L[i, :len(l)] = l
        return L, lab

    def softT(L, T):
        z = L / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    def nllT(L, lab, T):
        z = L / T; mx = z.max(1, keepdims=True); lse = np.log(np.exp(z - mx).sum(1)) + mx[:, 0]
        return float(-(z[np.arange(len(lab)), lab] - lse).sum())
    recal = {}
    for m in models:
        L, lab = mat(LOGP[m]); P2 = np.zeros_like(L); Ts = {}
        for fo in (0, 1):
            fit = fold == fo; app = ~fit
            T = float(grid[int(np.argmin([nllT(L[fit], lab[fit], t) for t in grid]))]); Ts[str(fo)] = T
            P2[app] = softT(L[app], T)
        P2 = np.nan_to_num(P2)
        conf2 = P2.max(1); c = CORR[m].astype(float)
        Y = np.zeros_like(P2); Y[np.arange(len(lab)), lab] = 1; brier2 = ((P2 - Y) ** 2).sum(1)
        recal[m] = {"T": Ts}
        for s, mask in (("overall", ALL), ("human_keys", HK)):
            recal[m][s] = {"before": calib(CONF[m], c, BRIER[m], mask, ones), "after": calib(conf2, c, brier2, mask, ones)}
    res["recal"] = recal

    # ---- family macro, equivalence, majority
    fmc = {}
    for s, mask in (("overall", ALL), ("human_keys", HK)):
        for a_, b_ in (("v2_27", "stock27"), ("v2_27", "qwen38"), ("r9", "stock9"), ("r9", "qwen38")):
            pa, ba = fmacro(CORR[a_].astype(float), mask); pb, bb = fmacro(CORR[b_].astype(float), mask)
            fmc.setdefault(f"{a_}-{b_}", {})[s] = {"d": pa - pb, "ci": ci(ba - bb)}
    res["family_macro"] = fmc
    eq = {}
    for a_, b_ in PAIRS:
        for s, mask in SUBS.items():
            pa, ba = micro(CORR[a_].astype(float), mask); pb, bb = micro(CORR[b_].astype(float), mask)
            c95 = ci(ba - bb)
            if c95[0] <= 0 <= c95[1]: eq.setdefault(f"{a_}-{b_}", {})[s] = {"d": pa - pb, "ci90": ci(ba - bb, 5.0), "ci95": c95}
    res["equivalence"] = eq
    maj = {}
    for t in sorted(set(task.tolist())):
        mk = task == t; labs = np.array([int(ref[keys[i]]["label"]) for i in np.where(mk)[0]])
        vals, cnt = np.unique(labs, return_counts=True)
        maj[t] = {"n": int(mk.sum()), "majority_index": int(vals[np.argmax(cnt)]), "majority_acc": float(cnt.max() / cnt.sum())}
    res["majority"] = maj

    # ---- Eurorad answer-word overlap
    WORD = re.compile(r"[a-z]{6,}")
    STOP = {"disease", "syndrome", "normal", "variant", "lesion", "tumour", "tumor", "primary", "chronic", "benign", "malignant", "acute"}

    def norm(s): return str(s).lower()
    flag = {}
    for i, k in enumerate(keys):
        if task[i] != "eurorad_dx": continue
        q = q_of.get(k); st = state_of.get(k[0])
        if not q or st is None: continue
        crit = q.get("criteria") or {}
        opts = {kk: norm(vv if vv else kk) for kk, vv in (crit.items() if isinstance(crit, dict) else enumerate(crit))}
        key_text = opts.get(q.get("label"), norm(q.get("label")))
        others = " ".join(v for kk, v in opts.items() if kk != q.get("label"))
        words = {w for w in WORD.findall(key_text) if w not in STOP and w not in others}
        text = norm(json.dumps(st, ensure_ascii=False))
        flag[i] = any(w in text for w in words)
    idx = np.array(sorted(flag)); fl = np.array([flag[i] for i in idx])
    el = {"n": int(len(idx)), "flagged": int(fl.sum())}
    for name, sel in (("answer_word_in_case", fl), ("no_answer_word", ~fl)):
        mk = np.zeros(len(keys), bool); mk[idx[sel]] = True; el[name] = {"n": int(mk.sum())}
        for m in ("v2_27", "stock27", "r9", "stock9", "qwen38", "medgemma_brief", "medgemma_fix"):
            if m in CORR: a, ab = micro(CORR[m].astype(float), mk); el[name][m] = {"acc": a, "ci": ci(ab)}
        for a_, b_ in (("v2_27", "stock27"), ("r9", "stock9"), ("v2_27", "qwen38")):
            d, db = dvec(a_, b_, mk); el[name][f"{a_}-{b_}"] = {"d": d, "ci": ci(db)}
    mk1 = np.zeros(len(keys), bool); mk1[idx[fl]] = True; mk0 = np.zeros(len(keys), bool); mk0[idx[~fl]] = True
    put("eurorad_gain_flagged_minus_unflagged", dvec("v2_27", "stock27", mk1), dvec("v2_27", "stock27", mk0), "eurorad_dx")
    el["did_v2_27-stock27"] = did["eurorad_gain_flagged_minus_unflagged"]["eurorad_dx"]
    res["eurorad_leak"] = el
    res["seconds"] = round(time.time() - t0, 1)
    Path(out_path).write_text(json.dumps(res, separators=(",", ":")))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--testdir", default=str(TESTDIR), help="scored test rows, one directory per model, plus test.jsonl")
    ap.add_argument("--samples", type=int, default=2000, help="bootstrap resamples shared by every system (the paper: 2,000)")
    ap.add_argument("--demo", default="", help="JSON list of demonstration-sample state hashes, or a JSON file with items[].state")
    ap.add_argument("--out", default="", help="output file (default: $RADKEV_HOME/runs/robustness/robustness2.json)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else RUNS / "robustness" / "robustness2.json"; out.parent.mkdir(parents=True, exist_ok=True)
    analyse(Path(a.testdir), out, a.samples, demo_hashes(a.demo))
    print(out)


if __name__ == "__main__":
    main()
