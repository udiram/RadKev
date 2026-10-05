"""External tests (post hoc; analysis-plan addendum of 2026-10-05, docs/ANALYSIS_PLAN.md): RadCases (ACR Appropriateness
Criteria panel and topic) and RadGraph-XL (radiologist-annotated entity status in CT and MRI reports). No model is trained.

    python experiments/external_tests.py --sets radcases                         # open data; four GPUs (two pairs) or two
    python experiments/external_tests.py --sets radgraph_xl --radgraph-xl FILE   # FILE: stanford_radgraph_xl_jsonl.csv from
                                                                                 # Stanford AIMI, plus --chexpert-plus CSV for the overlap rule

RadCases: radkev.external.radcases() downloads the pinned open sources (GPT-3.5 synthetic one-liners; first sentence of each
Medbullets question) and keeps the cases whose SHA-512 matches a label row; one-liners that occur verbatim in any record of the
RadKev splits ($RADKEV_HOME/data) are dropped. RadGraph-XL: reports with 50% or more word 8-gram overlap with CheXpert Plus or the
RadKev splits are dropped; up to three observation entities per report become the four-option status question of the training
format. RadKev-27B, Kev-27B, RadKev-9B and Kev-9B are scored with radkev.evaluate; Qwen3.8-27B and MedGemma-27B-text with
radkev.teacher predict on every file whose questions have at most 16 options. Writes compact per-question rows (no text) to
$RADKEV_HOME/runs/external/rows_<file>.json; paper/inputs holds the published rows and paper/build.py the paired bootstrap.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from radkev import data as bd
from radkev import external as be
from radkev.paths import DATA, KEV_9B_PINNED, KEV_27B_PINNED, KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS, resolve_run

WORK = RUNS / "external"; SCORED = WORK / "runs"
KEV_MODELS = {"v2_27": ("v2mg-kev-27b", 2), "stock27": (KEV_27B_PINNED, 2), "r9": ("v2x9-kev-9b", 1), "stock9": (KEV_9B_PINNED, 1)}
LLMS = {"qwen38": (QWEN38_27B, []), "medgemma_fix": (MEDGEMMA_27B, ["--think_off", "--single_bos"])}
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16", "TOKENIZERS_PARALLELISM": "false"}
MOD = {"stanford-chest-x-ray": "cxr", "stanford-chest-ct": "chestct", "stanford-abd-pelvis-ct": "abdct", "stanford-brain-mr": "brainmr"}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}
LOCK = threading.Lock()


def save():
    with LOCK: (WORK / "external_tests.json").write_text(json.dumps(status, indent=1))


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as e: status["phases"][name] = {"state": "failed", "error": str(e)[-2000:], "minutes": round((time.time() - t0) / 60, 1)}
    save(); return status["phases"][name]["state"] == "ok"


def run(cmd, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))[:300]}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"exit {rc}: " + "".join(Path(log).read_text().splitlines(True)[-20:])[-1800:])


def split_states():
    for p in sorted(DATA.glob("*/*.jsonl")):
        for line in open(p):
            if line.strip():
                s = json.loads(line)["state"]; yield s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def build_radcases():
    big = " | ".join(bd.norm(s) for s in split_states())

    class Contains:
        def __contains__(self, x): return len(x) > 20 and x in big
    status["radcases_build"] = be.radcases(WORK, check_text=Contains()); save()


def build_radgraph_xl(table, chexpert_plus):
    import pandas as pd
    csv.field_size_limit(sys.maxsize)
    rows = [{"dataset": r["dataset"], "doc_key": r["doc_key"], "sentences": json.loads(r["sentences"]), "ner": json.loads(r["ner"])}
            for r in csv.DictReader(open(table, newline=""))]
    docs = list(be.dygie_docs(rows, lambda d: MOD.get(d, d)))

    def grams(t, k=8):
        w = bd.norm(t).split(); return {hash(" ".join(w[i:i + k])) for i in range(max(0, len(w) - k + 1))}
    target = {d["doc_key"]: grams(d["text"]) for d in docs}; need = set().union(*target.values()); seen = set()
    if chexpert_plus:
        cols = [c for c in pd.read_csv(chexpert_plus, nrows=1).columns if c.startswith("section_") or c == "report"]
        for chunk in pd.read_csv(chexpert_plus, usecols=cols, chunksize=20000, dtype=str):
            for row in chunk.fillna("").itertuples(index=False): seen |= grams(" ".join(row)) & need
    for s in split_states(): seen |= grams(s) & need
    excl = {k for k, g in target.items() if len(g & seen) >= 0.5 * max(1, len(g))}
    by_text = {d["text"]: d["doc_key"] for d in docs}
    recs = list(be.radgraph_records(docs, exclude=lambda t: by_text[t] in excl))
    with open(WORK / "radgraph_xl.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    status["radgraph_xl_build"] = {"reports": len(docs), "excluded": len(excl),
                                   "excluded_by_modality": dict(Counter(d["modality"] for d in docs if d["doc_key"] in excl)),
                                   "stats": be.radgraph_records.stats}; save()


def max_options(path):
    return max(len(q.get("criteria") or {}) or 2 for l in open(path) if l.strip() for q in json.loads(l)["questions"].values())


def score_kev(m, devs, files):
    target, cards = KEV_MODELS[m]
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[:cards])}
    if cards == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    for f in files:
        if not (SCORED / f.stem / m / "summary.json").exists():
            run([KEV_PY, "-m", "radkev.evaluate", "--run", resolve_run(target), "--data", f, "--out", SCORED / f.stem / m], env, WORK / f"{m}.log")


def score_llm(name, devs, files):
    model, flags = LLMS[name]
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[:2])}; log = WORK / f"{name}.log"
    for f in files:
        if (SCORED / f.stem / name / "summary.json").exists() or max_options(f) > 16: continue
        raw, fixed = WORK / f"{f.stem}.{name}.preds.jsonl", WORK / f"{f.stem}.{name}.preds_ids.jsonl"
        run([KEV_PY, "-m", "radkev.teacher", "predict", "--model", model, "--data", f, "--out", raw, "--batch", "16", *flags], env, log)
        ids = [json.loads(l)["_meta"]["id"] for l in open(f) if l.strip()]
        with open(fixed, "w") as g:
            for l in open(raw):
                x = json.loads(l); x["id"] = ids[int(x["id"].split("/")[1])]; g.write(json.dumps(x) + "\n")
        run([KEV_PY, "-m", "radkev.evaluate", "--preds", fixed, "--data", f, "--out", SCORED / f.stem / name], env, log)


def publish():
    from kev.metrics import scored_rows
    for fdir in sorted(p for p in SCORED.iterdir() if p.is_dir()):
        res = {}
        for mdir in sorted(p for p in fdir.iterdir() if (p / "rows.json").exists()):
            res[mdir.name] = [[r["id"], r["question"], r["task"], int(r["label"]), int(np.argmax(r["p"])), round(float(r["p"][int(r["label"])]), 5),
                               round(float(max(r["p"])), 5)] for r in scored_rows(json.loads((mdir / "rows.json").read_text()))]
        (WORK / f"rows_{fdir.name}.json").write_text(json.dumps(res, separators=(",", ":")))


def lanes():
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    if len(devs) < 2: raise SystemExit("needs at least two GPUs (one pair)")
    return [devs[i:i + 2] for i in range(0, len(devs) - 1, 2)][:2]


def parallel(jobs, pairs):
    for i in range(0, len(jobs), len(pairs)):
        ts = [threading.Thread(target=phase, args=(n, (lambda f=f, d=d: f(d)))) for (n, f), d in zip(jobs[i:i + len(pairs)], pairs)]
        for t in ts: t.start()
        for t in ts: t.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", default="radcases"); ap.add_argument("--radgraph-xl", dest="rgx"); ap.add_argument("--chexpert-plus", dest="cxp")
    a = ap.parse_args(); sets = a.sets.split(",")
    for p in (WORK, SCORED): p.mkdir(parents=True, exist_ok=True)
    pairs = lanes(); status["gpu_pairs"] = pairs; save()
    files = []
    if "radcases" in sets:
        if not (WORK / "radcases_panel.jsonl").exists() and not phase("build_radcases", build_radcases): return
        files += [WORK / "radcases_panel.jsonl", WORK / "radcases_topic.jsonl"]
    if "radgraph_xl" in sets:
        if not a.rgx: raise SystemExit("--radgraph-xl FILE is required (Stanford AIMI release)")
        if not (WORK / "radgraph_xl.jsonl").exists() and not phase("build_radgraph_xl", lambda: build_radgraph_xl(a.rgx, a.cxp)): return
        files.append(WORK / "radgraph_xl.jsonl")
    parallel([(f"score_{m}", lambda d, m=m: score_kev(m, d, files)) for m in ("v2_27", "stock27")], pairs)
    parallel([(f"score_{m}", lambda d, m=m: score_kev(m, d, files)) for m in ("r9", "stock9")], pairs)
    parallel([(f"score_{m}", lambda d, m=m: score_llm(m, d, files)) for m in LLMS], pairs)
    phase("publish", publish)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()
    print(WORK)


if __name__ == "__main__":
    main()
