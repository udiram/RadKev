"""Option-only control for v3 (post hoc; no training; addendum 2026-10-05b, amendment 8): how much of each system's accuracy on the
radiology case questions survives when the case is withheld, and whether the gains depend on the answer's words appearing in the case.

    python jobs/submit.py jobs/blind_v3.py --gpus 4 --cpus 16 --mem 96 --timeout 360 --outputs 'blind_v3*.json'

Questions: every eurorad_dx question of runs/test-final/test.jsonl (ids rad/<line>, as in jobs/eval_v3.py) and every RSNA-RadioQA
question (data/rsna-radioqa/test.jsonl, its own ids). Each is posed twice on the same record id: with its state (full.jsonl) and with
the state replaced by one neutral line, "No case information is available." (blind.jsonl); wording, options and keys are unchanged.
Systems: RadKev-27B v3, Kev-27B, RadKev-9B v3, Kev-9B (pinned; jobs/kev_eval.py, bf16, each checkpoint's own temperature), and
Qwen3.8-27B and MedGemma-27B-text (teacher.py option-letter probabilities, reasoning off, MedGemma with --think_off).
Distinctive-word split (as jobs/robustness2.py): a question is flagged when a word of the correct option (>= 6 letters, not a stop
word, absent from every distractor) occurs in the case text.
Analysis: per task, accuracy with the case, without it, and the drop; the paired gains (full, blind, difference in differences); the
same per stratum of the distinctive-word split. Record-level bootstrap, 2,000 resamples, stratified by task, seed 20261005.
Publishes aggregates only.
"""
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

INCLUDE = ["teacher.py", "build_data.py", "jobs/kev_eval.py", "jobs/kev_multigpu.patch"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
TEST = RADKEV / "runs/test-final/test.jsonl"; RADIOQA = RADKEV / "data/rsna-radioqa/test.jsonl"
WORK = RADKEV / "blind_v3"; CODE = WORK / "code"; RUNS = WORK / "runs"
PIN27, PIN9 = "jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"
LLMS = {"qwen38": ("Qwen/Qwen3.8-27B", []), "medgemma_fix": ("google/medgemma-27b-text-it", ["--think_off"])}
NEUTRAL = {"note": "No case information is available."}
WORD = re.compile(r"[a-z]{6,}")
STOP = {"disease", "syndrome", "normal", "variant", "lesion", "tumour", "tumor", "primary", "chronic", "benign", "malignant", "acute"}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}, "scored": {}}
LOCK = threading.Lock()


def save():
    with LOCK: (OUT / "blind_v3_status.json").write_text(json.dumps(status, indent=1, default=str))


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as e: status["phases"][name] = {"state": "failed", "error": str(e)[-2000:], "minutes": round((time.time() - t0) / 60, 1)}
    save(); return status["phases"][name]["state"] == "ok"


def run(cmd, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))[:300]}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=KEV_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"exit {rc}: " + "".join(l for l in Path(log).read_text().splitlines(True)[-20:] if "hf_" not in l)[-1800:])


def ckpt(*names):
    for n in names:
        p = RADKEV / "runs" / n / "checkpoint"
        if (p / "head.pt").exists(): return str(p)
    return None


def env_for(devs):
    e = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16",
         "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ",".join(devs)}
    if len(devs) == 2: e.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    return e


def flagged(state, q):
    crit = q.get("criteria") or {}
    opts = {k: str(v if v else k).lower() for k, v in crit.items()}
    key_text = opts.get(q.get("label"), str(q.get("label")).lower())
    others = " ".join(v for k, v in opts.items() if k != q.get("label"))
    words = {w for w in WORD.findall(key_text) if w not in STOP and w not in others}
    text = json.dumps(state, ensure_ascii=False).lower()
    return any(w in text for w in words)


def build():
    recs = []
    for n, line in enumerate(open(TEST)):
        if not line.strip(): continue
        r = json.loads(line)
        qs = {k: v for k, v in r["questions"].items() if str(v.get("src", "")) == "eurorad_dx"}
        if qs: recs.append({"state": r["state"], "questions": qs, "_meta": {**r.get("_meta", {}), "id": f"rad/{n}", "group_id": f"rad/{n}"}})
    for line in open(RADIOQA):
        if line.strip(): recs.append(json.loads(line))
    flags, n_q = {}, {}
    with open(WORK / "full.jsonl", "w") as f, open(WORK / "blind.jsonl", "w") as g:
        for r in recs:
            for qid, q in r["questions"].items():
                flags[f"{r['_meta']['id']}|{qid}"] = flagged(r["state"], q); t = str(q.get("src", "")); n_q[t] = n_q.get(t, 0) + 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g.write(json.dumps({**r, "state": NEUTRAL}, ensure_ascii=False) + "\n")
    (WORK / "flags.json").write_text(json.dumps(flags))
    status["built"] = {"records": len(recs), "questions": n_q, "flagged": sum(flags.values())}; save()


def kev_task(name, run_, cards):
    def fn(devs):
        for v in ("full", "blind"):
            out = RUNS / v / name
            if not (out / "summary.json").exists():
                shutil.rmtree(out, ignore_errors=True)
                run([KEV_PY, CODE / "kev_eval.py", "--run", run_, "--data", WORK / f"{v}.jsonl", "--out", out], env_for(devs), WORK / f"{name}.log")
    return (name, cards, fn)


def llm_task(name):
    model, extra = LLMS[name]
    def fn(devs):
        for v in ("full", "blind"):
            out = RUNS / v / name
            if (out / "summary.json").exists(): continue
            data = WORK / f"{v}_llm.jsonl"   # questions with at most 16 options (single-letter scoring)
            if not data.exists():
                keep = []
                for l in open(WORK / f"{v}.jsonl"):
                    r = json.loads(l); r["questions"] = {k: q for k, q in r["questions"].items() if len(q.get("criteria") or {}) <= 16}
                    if r["questions"]: keep.append(json.dumps(r, ensure_ascii=False))
                data.write_text("\n".join(keep) + "\n")
            preds = WORK / f"{v}.{name}.preds.jsonl"
            run([KEV_PY, CODE / "teacher.py", "predict", "--model", model, "--data", data, "--out", preds, "--batch", "16", *extra], env_for(devs), WORK / f"{name}.log")
            ids = [json.loads(l)["_meta"]["id"] for l in open(data) if l.strip()]   # predict writes rad/<line>; records carry their own ids
            fixed = preds.with_suffix(".ids.jsonl")
            with open(fixed, "w") as g:
                for l in open(preds):
                    x = json.loads(l); x["id"] = ids[int(x["id"].split("/")[1])]; g.write(json.dumps(x) + "\n")
            shutil.rmtree(out, ignore_errors=True)   # Kev refuses an existing output directory
            run([KEV_PY, CODE / "kev_eval.py", "--preds", fixed, "--data", data, "--out", out], env_for(devs), WORK / f"{name}.log")
    return (name, 2, fn)


ANALYSE = r'''
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
runs, flags, out = Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4])
models = sorted(p.name for p in (runs / "full").iterdir() if (p / "rows.json").exists() and (runs / "blind" / p.name / "rows.json").exists())
R = {(m, v): {(r["id"], r["question"]): r for r in scored_rows(json.loads((runs / v / m / "rows.json").read_text()))} for m in models for v in ("full", "blind")}
res = {"models_scored": models, "tasks": {}}
B, SEED = 2000, 20261005
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
PAIRS = [("v3_27", "stock27"), ("v3_9", "stock9"), ("v3_27", "qwen38"), ("v3_27", "medgemma_fix"), ("v3_9", "qwen38"), ("qwen38", "stock27")]
for t in ("eurorad_dx", "rsna_radioqa"):
    # the common question set of every system in both conditions (the LLMs score only questions with at most 16 options)
    ks = None
    for (m, v), d in R.items():
        s = {k for k, r in d.items() if r["task"] == t}; ks = s if ks is None else ks & s
    ks = sorted(ks or [])
    if not ks: continue
    rec = sorted({k[0] for k in ks}); ridx = {r: i for i, r in enumerate(rec)}; q2r = np.array([ridx[k[0]] for k in ks])
    rng = np.random.default_rng(SEED); dr = rng.integers(0, len(rec), size=(B, len(rec))); W = np.zeros((B, len(rec)), np.float32)
    for b in range(B): np.add.at(W[b], dr[b], 1)
    QW = W[:, q2r]
    C = {(m, v): np.array([float(np.argmax(R[(m, v)][k]["p"]) == int(R[(m, v)][k]["label"])) for k in ks]) for m in models for v in ("full", "blind")}
    fl = np.array([bool(flags.get(f"{k[0]}|{k[1]}")) for k in ks])
    def est(x, mk): w = QW[:, mk]; return float(x[mk].mean()), (w @ x[mk]) / np.maximum(w.sum(1), 1e-9)
    o = {"n": len(ks), "records": len(rec), "flagged": int(fl.sum()), "models": {}, "pairs": {}, "split": {}}
    mk = np.ones(len(ks), bool)
    for m in models:
        pf, bf = est(C[(m, "full")], mk); pb, bb = est(C[(m, "blind")], mk)
        o["models"][m] = {"full": pf, "full_ci": ci(bf), "blind": pb, "blind_ci": ci(bb), "drop": pf - pb, "drop_ci": ci(bf - bb)}
    for a, b in PAIRS:
        if a not in models or b not in models: continue
        pa, af = est(C[(a, "full")], mk); pb_, bf_ = est(C[(b, "full")], mk); qa, ab = est(C[(a, "blind")], mk); qb, bb_ = est(C[(b, "blind")], mk)
        o["pairs"][f"{a}-{b}"] = {"full": pa - pb_, "full_ci": ci(af - bf_), "blind": qa - qb, "blind_ci": ci(ab - bb_),
                                 "did": (pa - pb_) - (qa - qb), "did_ci": ci((af - bf_) - (ab - bb_))}
    for name, sel in (("answer_word_in_case", fl), ("no_answer_word", ~fl)):
        if not sel.any(): continue
        s = {"n": int(sel.sum()), "models": {}, "pairs": {}}
        for m in models:
            p, b = est(C[(m, "full")], sel); q, bb = est(C[(m, "blind")], sel)
            s["models"][m] = {"full": p, "full_ci": ci(b), "blind": q, "blind_ci": ci(bb)}
        for a, b_ in PAIRS:
            if a in models and b_ in models:
                pa, af = est(C[(a, "full")], sel); pb_, bf_ = est(C[(b_, "full")], sel); s["pairs"][f"{a}-{b_}"] = {"full": pa - pb_, "full_ci": ci(af - bf_)}
        o["split"][name] = s
    if fl.any() and (~fl).any():
        for a, b_ in PAIRS:
            if a in models and b_ in models:
                _, a1 = est(C[(a, "full")], fl); _, b1 = est(C[(b_, "full")], fl); _, a0 = est(C[(a, "full")], ~fl); _, b0 = est(C[(b_, "full")], ~fl)
                d = (C[(a, "full")][fl].mean() - C[(b_, "full")][fl].mean()) - (C[(a, "full")][~fl].mean() - C[(b_, "full")][~fl].mean())
                o["split"].setdefault("gain_flagged_minus_unflagged", {})[f"{a}-{b_}"] = {"d": float(d), "ci": ci((a1 - b1) - (a0 - b0))}
    res["tasks"][t] = o
out.write_text(json.dumps(res, indent=1)); print("ok")
'''


def analyse():
    run([KEV_PY, "-c", ANALYSE, CODE, RUNS, WORK / "flags.json", OUT / "blind_v3.json"], {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"},
        WORK / "analyse.log")


def run_lanes(items, lanes):
    work = queue.Queue()
    for t in items: work.put(t)
    tries = {}
    def worker(use):
        bad = 0
        while bad < 3:
            try: name, g, fn = work.get_nowait()
            except queue.Empty: return
            t0 = time.time()
            try: fn(use); status["scored"][name] = {"ok": True, "minutes": round((time.time() - t0) / 60, 1)}; bad = 0
            except Exception as e:
                msg = str(e)   # intermittent CUDA initialisation failure on this node (seen as a CPU fallback in fla/causal-conv1d too)
                if any(x in msg for x in ("CUDA driver initialization failed", "cuda_init", "Expected x.is_cuda()")):
                    tries[name] = tries.get(name, 0) + 1; bad += 1
                    status.setdefault("cuda_init_retries", []).append({"task": name, "lane": use, "at": time.strftime("%H:%M:%S")})
                    if tries[name] < 4:
                        for v in ("full", "blind"):   # a partial predictions file is rewritten from scratch
                            for f in WORK.glob(f"{v}.{name}.preds*.jsonl"): f.unlink()
                        work.put((name, g, fn)); save(); time.sleep(45); continue
                status["scored"][name] = {"ok": False, "error": msg[-1200:]}
            save()
    ths = [threading.Thread(target=worker, args=(l,)) for l in lanes]
    for t in ths: t.start()
    for t in ths: t.join()


def main():
    for p in (WORK, CODE, RUNS): p.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    patch = CODE / "kev_multigpu.patch"
    if subprocess.run(["git", "-C", KEV_DIR, "apply", "--check", "--reverse", patch], capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", KEV_DIR, "apply", patch], check=True)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    status["gpus"] = devs; save()
    if len(devs) < 4: raise SystemExit("needs 4 GPUs")
    if not phase("build", build): return
    v3_27, v3_9 = ckpt("v3f-kev-27b-dp"), ckpt("v3f-kev-9b-dp", "v3g-kev-9b")
    status["checkpoints"] = {"v3_27": v3_27, "v3_9": v3_9}; save()
    two = [t for t in (kev_task("v3_27", v3_27, 2) if v3_27 else None, kev_task("stock27", PIN27, 2), llm_task("qwen38"), llm_task("medgemma_fix")) if t]
    one = [t for t in (kev_task("v3_9", v3_9, 1) if v3_9 else None, kev_task("stock9", PIN9, 1)) if t]
    run_lanes(two, [devs[0:2], devs[2:4]])
    run_lanes(one, [[d] for d in devs])
    phase("analyse", analyse)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    main()
