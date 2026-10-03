"""Options-only control (post hoc; no training): how much of each system's accuracy on Eurorad diagnosis and MedQA survives
when the case is withheld and only the question and its options are shown.

    python experiments/options_only.py                       # decision models and both LLMs; four GPUs (two pairs) or two
    python experiments/options_only.py --llms ""             # decision models only

Every eurorad_dx and medqa question of runs/test-final/test.jsonl is posed again with its record's state replaced by one neutral
line ("No case information is available."); question wording, options and answer keys are unchanged, and records keep their
test ids and cluster ids. RadKev-27B and Kev-27B (two cards each), then RadKev-9B and Kev-9B (one card each), are scored with
radkev.evaluate exactly as in experiments/final_test.py (bf16, each checkpoint's own temperature; stock Kev at the revisions
scored in the paper). The LLMs are scored as in the main evaluation: Qwen3.8-27B by option-letter probabilities with reasoning
off, and MedGemma-27B-text with the one-line thought pre-fill and a single <bos> (the medgemma_fix protocol,
experiments/medgemma_rescore.py). The full-case rows (runs/test-final/<model>/rows.json) give the paired reference.
Analysis: per system and task, accuracy with and without the case and the drop; for the listed pairs, the gain with and without
the case and its difference, with 95% intervals from 2,000 record resamples stratified by task (seed 20261002).
Writes aggregates only: $RADKEV_HOME/runs/options_only/options_only.json (scored rows stay in runs/options_only/runs/).
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from radkev.paths import KEV_9B_PINNED, KEV_27B_PINNED, KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS, resolve_run

TESTDIR = RUNS / "test-final"; TEST = TESTDIR / "test.jsonl"
WORK = RUNS / "options_only"; SCORED = WORK / "runs"
KEV_MODELS = {"v2_27": ("v2mg-kev-27b", 2), "stock27": (KEV_27B_PINNED, 2), "r9": ("v2x9-kev-9b", 1), "stock9": (KEV_9B_PINNED, 1)}
LLMS = {"qwen38": (QWEN38_27B, []), "medgemma_fix": (MEDGEMMA_27B, ["--think_off", "--single_bos"])}
PAIRS = [("v2_27", "stock27"), ("r9", "stock9"), ("v2_27", "qwen38"), ("v2_27", "medgemma_fix"), ("r9", "qwen38")]
TASKS = ("eurorad_dx", "medqa")
NEUTRAL = {"note": "No case information is available."}
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16", "TOKENIZERS_PARALLELISM": "false"}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}
LOCK = threading.Lock()


def save():
    with LOCK: (WORK / "options_only.json").write_text(json.dumps(status, indent=1))


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


def build():
    n_q = {t: 0 for t in TASKS}; n_r = 0
    with open(WORK / "blind.jsonl", "w") as f:
        for n, line in enumerate(open(TEST)):
            if not line.strip(): continue
            r = json.loads(line)
            qs = {k: v for k, v in r["questions"].items() if str(v.get("src", "")) in TASKS}
            if not qs: continue
            meta = dict(r.get("_meta", {})); meta.setdefault("id", f"rad/{n}"); meta.setdefault("group_id", f"rad/{n}")
            for v in qs.values(): n_q[str(v["src"])] += 1
            f.write(json.dumps({"state": NEUTRAL, "questions": qs, "_meta": meta}, ensure_ascii=False) + "\n"); n_r += 1
    status["built"] = {"records": n_r, "questions": n_q}; save()


def score_kev(m, devs):
    target, cards = KEV_MODELS[m]
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[:cards])}
    if cards == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    run([KEV_PY, "-m", "radkev.evaluate", "--run", resolve_run(target), "--data", WORK / "blind.jsonl", "--out", SCORED / m], env, WORK / f"{m}.log")


def score_llm(name, devs):
    """radkev.teacher predict keys its output by line number (rad/<n> of blind.jsonl); the ids are mapped back to the blind
    records' own test ids before radkev.evaluate, so every row pairs with the full-case rows."""
    model, flags = LLMS[name]
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[:2])}; log = WORK / f"{name}.log"
    raw, fixed = WORK / f"{name}.preds.jsonl", WORK / f"{name}.preds_ids.jsonl"
    run([KEV_PY, "-m", "radkev.teacher", "predict", "--model", model, "--data", WORK / "blind.jsonl", "--out", raw, "--batch", "16", *flags], env, log)
    ids = [json.loads(l)["_meta"]["id"] for l in open(WORK / "blind.jsonl") if l.strip()]
    with open(fixed, "w") as f:
        for l in open(raw):
            x = json.loads(l); x["id"] = ids[int(x["id"].split("/")[1])]; f.write(json.dumps(x) + "\n")
    run([KEV_PY, "-m", "radkev.evaluate", "--preds", fixed, "--data", WORK / "blind.jsonl", "--out", SCORED / name], env, log)


def analyse():
    from kev.metrics import scored_rows
    models = [p.name for p in SCORED.iterdir() if (p / "rows.json").exists()]
    blind = {m: {(r["id"], r["question"]): r for r in scored_rows(json.loads((SCORED / m / "rows.json").read_text()))} for m in models}
    keys = sorted(set.intersection(*[set(v) for v in blind.values()]))
    full = {}
    for m in models:
        d = {(r["id"], r["question"]): r for r in scored_rows(json.loads((TESTDIR / m / "rows.json").read_text()))}
        full[m] = d; keys = [k for k in keys if k in d]
    ok = lambda r: int(np.argmax(r["p"]) == int(r["label"]))
    task = np.array([blind[models[0]][k]["task"] for k in keys]); rec = np.array([k[0] for k in keys])
    recs = sorted(set(rec.tolist())); ridx = {r: i for i, r in enumerate(recs)}; q2r = np.array([ridx[r] for r in rec])
    rng = np.random.default_rng(20261002); Bn = 2000
    W = np.zeros((Bn, len(recs)), np.float32)
    for t in sorted(set(task.tolist())):
        mem = np.array(sorted({ridx[r] for r, tt in zip(rec, task) if tt == t}))
        dr = rng.integers(0, len(mem), size=(Bn, len(mem)))
        for b in range(Bn): np.add.at(W[b], mem[dr[b]], 1)
    QW = W[:, q2r]
    Cx = {(m, s): np.array([ok((blind if s == "blind" else full)[m][k]) for k in keys], float) for m in models for s in ("blind", "full")}

    def est(x, mk): w = QW[:, mk]; return float(x[mk].mean()), (w @ x[mk]) / np.maximum(w.sum(1), 1e-9)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {"n": {t: int((task == t).sum()) for t in set(task.tolist())}, "models": {}, "gain": {}}
    for t in sorted(set(task.tolist())):
        mk = task == t
        for m in models:
            pf, bf = est(Cx[(m, "full")], mk); pb, bb = est(Cx[(m, "blind")], mk)
            res["models"].setdefault(m, {})[t] = {"full": pf, "full_ci": ci(bf), "blind": pb, "blind_ci": ci(bb), "drop": pf - pb, "drop_ci": ci(bf - bb)}
        for a, b in PAIRS:
            if a in models and b in models:
                _, af = est(Cx[(a, "full")], mk); _, bf_ = est(Cx[(b, "full")], mk); _, ab = est(Cx[(a, "blind")], mk); _, bb_ = est(Cx[(b, "blind")], mk)
                gf = Cx[(a, "full")][mk].mean() - Cx[(b, "full")][mk].mean(); gb = Cx[(a, "blind")][mk].mean() - Cx[(b, "blind")][mk].mean()
                res["gain"].setdefault(f"{a}-{b}", {})[t] = {"full": float(gf), "full_ci": ci(af - bf_), "blind": float(gb), "blind_ci": ci(ab - bb_),
                                                           "did": float(gf - gb), "did_ci": ci((af - bf_) - (ab - bb_))}
    status["result"] = res


def lanes():
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    if len(devs) < 2: raise SystemExit("needs at least two GPUs (one pair)")
    return [devs[i:i + 2] for i in range(0, len(devs) - 1, 2)][:2]


def parallel(jobs, pairs):
    """Run jobs [(name, fn(devs))] on the GPU pairs, as many at a time as there are pairs."""
    for i in range(0, len(jobs), len(pairs)):
        ts = [threading.Thread(target=phase, args=(n, (lambda f=f, d=d: f(d)))) for (n, f), d in zip(jobs[i:i + len(pairs)], pairs)]
        for t in ts: t.start()
        for t in ts: t.join()


def main():
    global LLMS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llms", default=None, help="comma-separated subset of qwen38,medgemma_fix ('' for none; default both)")
    a = ap.parse_args()
    if a.llms is not None: LLMS = {k: v for k, v in LLMS.items() if k in a.llms.split(",")}
    for p in (WORK, SCORED): p.mkdir(parents=True, exist_ok=True)
    pairs = lanes(); status["gpu_pairs"] = pairs; save()
    if not phase("build", build): return
    todo = lambda m: not (SCORED / m / "summary.json").exists()
    parallel([(f"score_{m}", lambda d, m=m: score_kev(m, d)) for m in ("v2_27", "stock27") if todo(m)], pairs)
    parallel([(f"score_{m}", lambda d, m=m: score_kev(m, d)) for m in ("r9", "stock9") if todo(m)], pairs)
    parallel([(f"score_{m}", lambda d, m=m: score_llm(m, d)) for m in LLMS if todo(m)], pairs)
    phase("analyse", analyse)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()
    print(WORK / "options_only.json")


if __name__ == "__main__":
    main()
