#!/usr/bin/env python3
"""Paired analysis of the external tests (analysis-plan addendum 2026-10-05) from the compact rows of jobs/external_tests.py.

    python paper/external/analyse.py --rows DIR [--rows DIR ...] --out paper/artifacts/external_tests/analysis.json

Accuracy, expected calibration error (10 equal-width bins) and paired differences with 95% intervals from a record-level
bootstrap (2,000 resamples, stratified by subset or modality, seed 20261005). RadCases subsets come from the task name
(radcases_panel_synthetic, ...); the "None" panel label is read from the rebuilt records (build_external.radcases).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B, SEED = 2000, 20261005
PAIRS = [("v2_27", "stock27"), ("r9", "stock9"), ("v2_27", "qwen38"), ("v2_27", "medgemma_fix"), ("r9", "qwen38"), ("qwen38", "stock27"),
         ("v3_27", "stock27"), ("v3_9", "stock9"), ("v3_27", "v2_27"), ("v3_9", "r9"), ("v3_27", "qwen38"), ("v3_27", "medgemma_fix"), ("v3_9", "qwen38")]


def ece(conf, corr, bins=10):
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((idx == k).mean() * abs(corr[idx == k].mean() - conf[idx == k].mean()) for k in range(bins) if (idx == k).any()))


def none_ids():
    sys.path.insert(0, str(ROOT)); import build_external as be
    with tempfile.TemporaryDirectory() as d:
        be.radcases(d)
        return {json.loads(l)["_meta"]["id"] for l in open(Path(d) / "radcases_panel.jsonl") if json.loads(l)["_meta"]["none"]}


def analyse(rows, nones):
    models = sorted(rows)
    keys = sorted(set.intersection(*[{(r[0], r[1]) for r in rows[m]} for m in models]))
    R = {m: {(r[0], r[1]): r for r in rows[m]} for m in models}
    task = np.array([R[models[0]][k][2] for k in keys]); rec = np.array([k[0] for k in keys])
    stratum = np.array([t.rsplit("_", 1)[-1] for t in task])
    corr = {m: np.array([R[m][k][3] == R[m][k][4] for k in keys], float) for m in models}
    conf = {m: np.array([R[m][k][6] for k in keys]) for m in models}
    recs = sorted(set(rec.tolist())); ridx = {r: i for i, r in enumerate(recs)}; q2r = np.array([ridx[r] for r in rec])
    rng = np.random.default_rng(SEED); W = np.zeros((B, len(recs)), np.float32)
    rs = np.array([stratum[np.where(rec == r)[0][0]] for r in recs])
    for s in sorted(set(rs.tolist())):
        mem = np.where(rs == s)[0]; draws = rng.integers(0, len(mem), size=(B, len(mem)))
        for b in range(B): np.add.at(W[b], mem[draws[b]], 1)
    QW = W[:, q2r]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    def acc(x, mk): w = QW[:, mk]; return float(x[mk].mean()), (w @ x[mk]) / np.maximum(w.sum(1), 1e-9)
    subsets = {"all": np.ones(len(keys), bool)}
    for s in sorted(set(stratum.tolist())): subsets[s] = stratum == s
    if nones: subsets["excl_none"] = ~np.isin(rec, list(nones)); subsets["none_only"] = np.isin(rec, list(nones))
    out = {"n": {s: int(mk.sum()) for s, mk in subsets.items()}, "models": {}, "pairs": {}}
    for m in models:
        out["models"][m] = {s: {"acc": acc(corr[m], mk)[0], "acc_ci": ci(acc(corr[m], mk)[1]), "ece": ece(conf[m][mk], corr[m][mk])} for s, mk in subsets.items()}
    for a, b in PAIRS:
        if a in corr and b in corr:
            o = {}
            for s, mk in subsets.items():
                pa, ba = acc(corr[a], mk); pb, bb = acc(corr[b], mk); d = ba - bb
                o[s] = {"d": pa - pb, "ci": ci(d), "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
            out["pairs"][f"{a}-{b}"] = o
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = {}
    for d in a.rows:
        for f in sorted(Path(d).rglob("rows_*.json")): files[f.stem[5:]] = f
    res = {}
    nones = none_ids() if "radcases_panel" in files else set()
    for name, f in sorted(files.items()):
        res[name] = analyse(json.loads(f.read_text()), nones if name == "radcases_panel" else set())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(name, r["n"]); [print("  ", m, {s: round(100 * v["acc"], 1) for s, v in r["models"][m].items()}) for m in r["models"]]
        [print("  ", p, {s: f"{100 * v['d']:+.1f} [{100 * v['ci'][0]:.1f}, {100 * v['ci'][1]:.1f}]" for s, v in r["pairs"][p].items()}) for p in r["pairs"]]


if __name__ == "__main__":
    main()
