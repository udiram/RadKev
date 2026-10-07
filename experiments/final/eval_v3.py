"""v3 evaluation on the radiology benchmark (analysis-plan addendum 2026-10-05b, amendments 1-6). No training.

    python jobs/submit.py jobs/eval_v3.py --gpus 4 --cpus 24 --mem 110 --timeout 1440 --outputs 'eval_v3*.json'

Phases (each resumable; a scored system is skipped when its summary.json exists):
1 build    runs/test-v3/new.jsonl = RadCases test + ReXErr test + RSNA-RadioQA (explicit record ids); the existing held-out test file
           runs/test-final/test.jsonl is split into two shards with explicit ids rad/<line> (ids identical to the pilot's rows).
2 score    every system on what it has not been scored on yet, on a pool of GPUs (2 per 27B decision model or LLM, 1 otherwise):
           RadKev-27B v3 and RadKev-9B v3 (and the 9B initialisation-ablation arms once trained) on both test shards and on new.jsonl; Kev-27B/9B (pinned), RadKev v2 27B/9B, Kev-4B/0.8B,
           Laya, Laya-typed-decisions, GLiNER2.5-Decide, Julia-1 on new.jsonl (their held-out rows exist from the pilot); Qwen3.8-27B
           and MedGemma-27B-text on new.jsonl (option-letter probabilities, reasoning off; questions with more than 16 options are
           left out). Unpinned Hub models load offline from the node cache (the pilot's snapshots); snapshot hashes are recorded.
3 analyse  benchmark tasks (amendment 6), per system and task accuracy with 95% intervals (record-level bootstrap, 2,000 resamples
           stratified by source), task means (15 tasks; 14 human-assigned), pooled accuracy, paired differences for the planned pairs
           (Holm over the 15 tasks for the primary pair), calibration (ECE, confident errors, coverage at 5% error, Brier) before and
           after two-fold cross-fitted temperature recalibration, held-out vs seen wordings, and CT-RATE agreement.
Publishes aggregates only (eval_v3.json, eval_v3_status.json).
"""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

INCLUDE = ["teacher.py", "build_data.py", "build_external.py", "jobs/kev_eval.py", "jobs/laya_predict.py", "jobs/compare.py",
           "jobs/decision_baselines.py", "jobs/kev_multigpu.patch", "jobs/kev_locked.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ["XDG_CACHE_HOME"]); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY, LAYA_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python", RADKEV / "laya-venv/bin/python"
WORK = RADKEV / "runs/test-v3"; CODE = WORK / "code"; FINAL = RADKEV / "runs/test-final"
PIN27, PIN9 = "jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}, "scored": {}}
LOCK = threading.Lock()


def save():
    with LOCK: (OUT / "eval_v3_status.json").write_text(json.dumps(status, indent=1, default=str))


def ckpt(*names):
    for n in names:
        p = RADKEV / "runs" / n / "checkpoint"
        if (p / "head.pt").exists(): return str(p)
    return None


def env(devs, extra=None):
    e = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16",
         "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ",".join(devs)}
    if len(devs) == 2: e.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    return {**e, **(extra or {})}


def sh(cmd, e, log, cwd=KEV_DIR):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))[:400]}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=e, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"exit {rc}: " + "".join(Path(log).read_text().splitlines(True)[-15:])[-1500:])


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as ex: status["phases"][name] = {"state": "failed", "error": str(ex)[-2500:], "minutes": round((time.time() - t0) / 60, 1)}
    save(); return status["phases"][name]["state"] == "ok"


# ------------------------------------------------------------------------------------------------------------ 1 build
def build():
    WORK.mkdir(parents=True, exist_ok=True)
    lines = [l for l in (FINAL / "test.jsonl").read_text().splitlines()]
    shards = [[], []]
    for n, l in enumerate(lines):
        if not l.strip(): continue
        r = json.loads(l); r.setdefault("_meta", {})["id"] = f"rad/{n}"; r["_meta"].setdefault("group_id", f"rad/{n}")
        shards[n % 2].append(json.dumps(r, ensure_ascii=False))
    for i, s in enumerate(shards): (WORK / f"final_s{i}.jsonl").write_text("\n".join(s) + "\n")
    new = []
    for d in ("radcases-v3", "rexerr", "rsna-radioqa"):
        new += [l for l in (RADKEV / "data" / d / "test.jsonl").read_text().splitlines() if l.strip()]
    (WORK / "new.jsonl").write_text("\n".join(new) + "\n")
    status["build"] = {"final_records": sum(len(s) for s in shards), "new_records": len(new)}


# ------------------------------------------------------------------------------------------------------------ 2 score
def kev_task(name, run, data, out, gpus):
    def fn(devs):
        if (out / "summary.json").exists(): return
        sh([KEV_PY, CODE / "kev_locked.py", WORK / "load.lock", CODE / "kev_eval.py", "--run", run, "--data", data, "--out", out], env(devs), WORK / f"{name}.log")
    return (name, gpus, fn)


def preds_task(name, gpus, make, data, out):
    """make(devs, preds_path) writes {"id": "rad/<line>", "probabilities"}; ids are mapped to the records' own ids, then kev_eval."""
    def fn(devs):
        if (out / "summary.json").exists(): return
        raw = WORK / f"{out.parent.name}.{name}.preds.jsonl"; fixed = raw.with_suffix(".ids.jsonl")
        data_used = make(devs, raw) or data
        ids = [json.loads(l)["_meta"]["id"] for l in open(data_used) if l.strip()]
        with open(fixed, "w") as g:
            for l in open(raw):
                x = json.loads(l); x["id"] = ids[int(x["id"].split("/")[1])]; g.write(json.dumps(x) + "\n")
        sh([KEV_PY, CODE / "kev_eval.py", "--preds", fixed, "--data", data_used, "--out", out], env(devs), WORK / f"{name}.log")
    return (name, gpus, fn)


def llm_make(model, flags):
    def make(devs, raw):
        sub = WORK / f"llm_{Path(raw).stem}.jsonl"   # questions with at most 16 options (single-letter scoring)
        data = WORK / "new.jsonl"; keep = []
        for l in open(data):
            if not l.strip(): continue
            r = json.loads(l); r["questions"] = {k: q for k, q in r["questions"].items() if len(q.get("criteria") or {}) <= 16}
            if r["questions"]: keep.append(json.dumps(r, ensure_ascii=False))
        sub.write_text("\n".join(keep) + "\n")
        sh([KEV_PY, CODE / "teacher.py", "predict", "--model", model, "--data", sub, "--out", raw, "--batch", "16", *flags], env(devs), WORK / "llm.log")
        return sub
    return make


def gen_make(kind):
    """The pilot's non-Kev decision baselines, run with decision_baselines.py's own code on new.jsonl."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("db", CODE / "decision_baselines.py"); db = importlib.util.module_from_spec(spec)
    os.environ.setdefault("ZCB_OUTPUT_DIR", str(OUT)); spec.loader.exec_module(db); db.ENV["HF_HUB_OFFLINE"] = "1"
    data = WORK / "new.jsonl"
    def make(devs, raw):
        e = {**db.ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs), "HF_HUB_OFFLINE": "1"}
        if kind == "laya":
            sh([LAYA_PY, CODE / "laya_predict.py", "--data", data, "--out", raw, "--device", "cuda"], e, WORK / "laya.log")
        elif kind == "laya_typed":
            sh([LAYA_PY, CODE / "laya_predict.py", "--data", data, "--out", raw, "--model", "convaiinnovations/laya-typed-decisions", "--device", "cuda"], e, WORK / "laya.log")
        elif kind == "julia1":
            h = db.hub("SupersonicLabs/Julia-1")
            sh([db.venv("julia", []), "-c", db.JULIA, data, raw, h], {**e, "PYTHONPATH": h}, WORK / "julia.log")
        elif kind == "gliner_decide":
            sh([db.venv("gliner2", []), "-c", db.GLINER, data, raw, "fastino/GLiNER2.5-Decide"], e, WORK / "gliner.log")
    return make


def snapshots():
    hub = CACHE / "huggingface/hub"; res = {}
    for repo in ("jaredpalmer/kev-27b", "jaredpalmer/kev-9b", "jaredpalmer/kev-4b", "jaredpalmer/kev-0.8b", "Qwen/Qwen3.8-27B", "google/medgemma-27b-text-it",
                 "convaiinnovations/laya-typed-decisions", "SupersonicLabs/Julia-1", "fastino/GLiNER2.5-Decide"):
        d = hub / ("models--" + repo.replace("/", "--"))
        ref = d / "refs/main"
        res[repo] = {"main": ref.read_text().strip() if ref.exists() else None, "snapshots": sorted(p.name for p in (d / "snapshots").glob("*")) if d.exists() else []}
    status["snapshots"] = res


def score():
    v3_27, v3_9 = ckpt("v3f-kev-27b-dp"), ckpt("v3f-kev-9b-dp", "v3g-kev-9b")
    v2_27, v2_9 = ckpt("v2mg-kev-27b"), ckpt("v2x9-kev-9b")
    # 9B initialisation ablation (amendment 5): scored when its checkpoints exist (a second run after those trainings adds them)
    base9, k9_10, b9_10 = ckpt("v3f-base-9b-dp", "v3g-base-9b"), ckpt("v3f10-kev-9b-dp", "v3g10-kev-9b"), ckpt("v3f10-base-9b-dp", "v3g10-base-9b")
    status["checkpoints"] = {"v3_27": v3_27, "v3_9": v3_9, "v2_27": v2_27, "v2_9": v2_9, "base9": base9, "k9_10": k9_10, "b9_10": b9_10}; snapshots(); save()
    new = WORK / "new.jsonl"; tasks = []
    for nm, run, g in (("v3_27", v3_27, 2), ("v3_9", v3_9, 1), ("base9", base9, 1), ("k9_10", k9_10, 1), ("b9_10", b9_10, 1)):
        if run:
            for i in (0, 1): tasks.append(kev_task(f"{nm}_s{i}", run, WORK / f"final_s{i}.jsonl", WORK / "final" / f"{nm}_s{i}", g))
            tasks.append(kev_task(nm, run, new, WORK / "new" / nm, g))
    for nm, run, g in (("stock27", PIN27, 2), ("stock9", PIN9, 1), ("v2_27", v2_27, 2), ("r9", v2_9, 1),
                       ("kev4", "jaredpalmer/kev-4b", 1), ("kev08", "jaredpalmer/kev-0.8b", 1)):
        if run: tasks.append(kev_task(nm, run, new, WORK / "new" / nm, g))
    tasks.append(preds_task("qwen38", 2, llm_make("Qwen/Qwen3.8-27B", []), new, WORK / "new" / "qwen38"))
    tasks.append(preds_task("medgemma_fix", 2, llm_make("google/medgemma-27b-text-it", ["--think_off"]), new, WORK / "new" / "medgemma_fix"))
    # not scorable in this environment (2026-10-06): Laya's head cannot encode the 225-option RadCases topic question
    # (head_max_len 192) and its venv falls back to CPU on this driver; Julia-1's venv has no CUDA build for driver 575
    status["not_scored"] = {"laya": "options exceed head_max_len=192 (radcases_topic); laya-venv CUDA unavailable",
                            "julia1": "CUDA unavailable in its venv (driver 575); CPU fallback disabled by Julia-1"}
    for k in ("laya_typed", "gliner_decide"):
        tasks.append(preds_task(k, 1, gen_make(k), new, WORK / "new" / k))
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]

    def run_lanes(items, lanes):
        """A task whose process fails CUDA initialisation (intermittent on this node) goes back to the queue (at most 4 tries)
        and its lane pauses; a lane that fails initialisation 3 times in a row stops taking work."""
        work = queue.Queue(); tries = {}
        for t in items: work.put(t)
        def worker(use):
            bad = 0
            while bad < 3:
                try: name, g, fn = work.get_nowait()
                except queue.Empty: return
                t0 = time.time()
                try: fn(use); status["scored"][name] = {"ok": True, "minutes": round((time.time() - t0) / 60, 1), "gpus": len(use)}; bad = 0
                except Exception as ex:
                    msg = str(ex)
                    if "CUDA driver initialization failed" in msg or "cuda_init" in msg:
                        tries[name] = tries.get(name, 0) + 1; bad += 1
                        status.setdefault("cuda_init_retries", []).append({"task": name, "lane": use, "at": time.strftime("%H:%M:%S")})
                        if tries[name] < 4: work.put((name, g, fn)); save(); time.sleep(45); continue
                    status["scored"][name] = {"ok": False, "error": msg[-1200:]}
                save()
        ths = [threading.Thread(target=worker, args=(l,)) for l in lanes]
        for t in ths: t.start()
        for t in ths: t.join()
    run_lanes([t for t in tasks if t[1] == 2], [devs[0:2], devs[2:4]])       # 27B decision models and the LLMs, one NVLink pair each
    run_lanes([t for t in tasks if t[1] == 1], [[d] for d in devs])           # 9B and smaller, one GPU each


# ------------------------------------------------------------------------------------------------------------ 3 analyse
ANALYSE = r'''
import json, sys, re
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
import compare as C
W, FINAL, OUTF = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
B, SEED = 2000, 20261005
radidx = json.loads((FINAL / "radiology_index.json").read_text())
HUMAN = ["iu_finding", "iu_normal", "iu_which", "eurorad_dx", "eurorad_route", "medmcqa_rad", "medmcqa_other_rad", "medqa_rad", "medxpertqa_rad",
         "mmlu_rad", "pubmedqa_rad", "radcases_panel", "radcases_topic", "rsna_radioqa"]
CONSTR = ["rexerr_error"]; BENCH = HUMAN + CONSTR
def bench_task(task, rid, qid):
    if task in ("iu_finding", "iu_normal", "iu_which", "eurorad_dx", "eurorad_route", "medmcqa_rad", "rsna_radioqa", "rexerr_error"): return task
    if task.startswith("radcases_panel"): return "radcases_panel"
    if task.startswith("radcases_topic"): return "radcases_topic"
    pooled = {"medmcqa_med": "medmcqa_other_rad", "medqa": "medqa_rad", "medxpertqa": "medxpertqa_rad", "pubmedqa": "pubmedqa_rad"}
    t = pooled.get(task) or ("mmlu_rad" if task.startswith("mmlu_") else None)
    if t and radidx.get(f"{rid}|{qid}"): return t
    if task.startswith("ctrate_"): return "ctrate:" + task
    return None
def load(name):
    """rows of one system: pilot rows on the held-out file (runs/test-final/<name>) or the v3 shards, plus new.jsonl rows."""
    rows = []
    if (W / "final" / f"{name}_s0" / "rows.json").exists():
        for i in (0, 1): rows += scored_rows(json.loads((W / "final" / f"{name}_s{i}" / "rows.json").read_text()))
    elif (FINAL / name / "rows.json").exists(): rows += scored_rows(json.loads((FINAL / name / "rows.json").read_text()))
    if (W / "new" / name / "rows.json").exists(): rows += scored_rows(json.loads((W / "new" / name / "rows.json").read_text()))
    out = {}
    for r in rows:
        bt = bench_task(r["task"], r["id"], r["question"])
        if bt is None: continue
        p = np.clip(np.asarray(r["p"], float), 1e-12, 1); p = p / p.sum()
        out[(r["id"], r["question"])] = (bt, int(r["label"]), p)
    return out
SYSTEMS = ["v3_27", "v3_9", "base9", "k9_10", "b9_10", "stock27", "stock9", "v2_27", "r9", "kev4", "kev08", "laya", "laya_typed", "gliner_decide", "julia1", "qwen38", "medgemma_fix"]
D = {s: load(s) for s in SYSTEMS}; D = {s: v for s, v in D.items() if v}
keys = sorted(set().union(*[set(v) for v in D.values()]))
task = {k: next(v[k][0] for v in D.values() if k in v) for k in keys}
rec = sorted({k[0] for k in keys}); ridx = {r: i for i, r in enumerate(rec)}
rsrc = {}
for k in keys: rsrc.setdefault(k[0], task[k].replace("ctrate:", "").split("_")[0])
rng = np.random.default_rng(SEED); W8 = np.zeros((B, len(rec)), np.float32)
strata = {}
for r, i in ridx.items(): strata.setdefault(rsrc[r], []).append(i)
for s, mem in sorted(strata.items()):
    mem = np.array(mem); dr = rng.integers(0, len(mem), size=(B, len(mem)))
    for b in range(B): np.add.at(W8[b], mem[dr[b]], 1)
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
pv = lambda d: float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
def vec(s, ks):
    c = np.array([float(np.argmax(D[s][k][2]) == D[s][k][1]) for k in ks]); w = W8[:, [ridx[k[0]] for k in ks]]
    return c, w
def acc(c, w): return float(c.mean()), (w @ c) / np.maximum(w.sum(1), 1e-9)
def tmean(s, ks, tasks):
    pts, bs = [], []
    for t in tasks:
        kk = [k for k in ks if task[k] == t]
        if not kk: continue
        c, w = vec(s, kk); a, b = acc(c, w); pts.append(a); bs.append(b)
    return (float(np.mean(pts)), np.mean(bs, 0), len(pts)) if pts else (None, None, 0)
res = {"n": {t: sum(1 for k in keys if task[k] == t) for t in sorted(set(task.values()))}, "systems": {}, "pairs": {}, "calibration": {}}
for s in D:
    ks = [k for k in keys if k in D[s]]; r = {"tasks": {}}
    for t in sorted(set(task[k] for k in ks)):
        c, w = vec(s, [k for k in ks if task[k] == t]); a, b = acc(c, w); r["tasks"][t] = {"n": int(len(c)), "acc": a, "ci": ci(b)}
    for nm, tl in (("bench", BENCH), ("human", HUMAN)):
        m, mb, nt = tmean(s, [k for k in ks if task[k] in tl], tl)
        kk = [k for k in ks if task[k] in tl]; c, w = vec(s, kk); a, b = acc(c, w)
        r[nm] = {"task_mean": m, "task_mean_ci": ci(mb) if mb is not None else None, "tasks": nt, "pooled": a, "pooled_ci": ci(b), "n": len(kk)}
    res["systems"][s] = r
PAIRS = [("v3_27", "stock27"), ("v3_9", "stock9"), ("v3_27", "v2_27"), ("v3_9", "r9"), ("v3_27", "qwen38"), ("v3_27", "medgemma_fix"), ("v3_9", "qwen38"),
         ("v3_27", "v3_9"), ("stock27", "stock9"), ("v3_9", "stock27"), ("qwen38", "stock27"),
         ("v3_9", "base9"), ("base9", "stock9"), ("k9_10", "b9_10"), ("k9_10", "stock9"), ("b9_10", "stock9"), ("v3_9", "k9_10"), ("base9", "b9_10")]
for a_, b_ in PAIRS:
    if a_ not in D or b_ not in D: continue
    ks = [k for k in keys if k in D[a_] and k in D[b_]]; o = {"tasks": {}}
    for nm, tl in (("bench", BENCH), ("human", HUMAN)):
        kk = [k for k in ks if task[k] in tl]
        ma, mab, nt = tmean(a_, kk, tl); mb_, mbb, _ = tmean(b_, kk, tl)
        if ma is None or mb_ is None or not kk: continue
        ca, wa = vec(a_, kk); cb, _ = vec(b_, kk); pa, ba = acc(ca, wa); pb, bb = acc(cb, wa)
        o[nm] = {"task_mean_d": ma - mb_, "task_mean_ci": ci(mab - mbb), "task_mean_p": pv(mab - mbb), "tasks": nt,
                 "pooled_d": pa - pb, "pooled_ci": ci(ba - bb), "pooled_p": pv(ba - bb), "n": len(kk)}
    for t in sorted(set(task[k] for k in ks)):
        kk = [k for k in ks if task[k] == t]; ca, wa = vec(a_, kk); cb, _ = vec(b_, kk); pa, ba = acc(ca, wa); pb, bb = acc(cb, wa)
        o["tasks"][t] = {"n": len(kk), "d": pa - pb, "ci": ci(ba - bb), "p": pv(ba - bb)}
    bt = sorted([(v["p"], t) for t, v in o["tasks"].items() if t in BENCH]); run = 0.0
    for i, (p, t) in enumerate(bt): run = max(run, min(1.0, (len(bt) - i) * p)); o["tasks"][t]["p_holm"] = run
    res["pairs"][f"{a_}-{b_}"] = o
# calibration on the benchmark (and human-assigned), as scored and after two-fold cross-fitted temperature scaling
def calib(ps, ys, ws=None):
    conf = np.array([p.max() for p in ps]); corr = np.array([float(np.argmax(p) == y) for p, y in zip(ps, ys)])
    bins = np.clip((conf * 10).astype(int), 0, 9); e = sum(abs(corr[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    o = np.argsort(-conf, kind="stable"); risk = np.cumsum(1 - corr[o]) / np.arange(1, len(o) + 1); okk = np.where(risk <= 0.05)[0]
    brier = float(np.mean([((p - np.eye(len(p))[y]) ** 2).sum() for p, y in zip(ps, ys)]))
    return {"ece": float(e), "conf_err": float(((conf >= 0.9) & (corr == 0)).mean()), "cov5": float((okk[-1] + 1) / len(o)) if len(okk) else 0.0, "brier": brier}
def temp(ps, T):
    out = []
    for p in ps: z = np.log(p) / T; z = np.exp(z - z.max()); out.append(z / z.sum())
    return out
GRID = np.array([2 ** (k / 30) for k in range(-60, 61)])
import zlib
def fit_T(ps, ys):
    K = max(len(p) for p in ps); L = np.full((len(ps), K), -np.inf)
    for i, p in enumerate(ps): L[i, :len(p)] = np.log(p)
    y = np.array(ys); best, bT = np.inf, 1.0
    for T in GRID:
        Z = L / T; m = Z.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(Z - m).sum(1))
        nll = float(np.mean(lse - Z[np.arange(len(y)), y]))
        if nll < best: best, bT = nll, float(T)
    return bT
for s in D:
    o = {}
    for nm, tl in (("bench", BENCH), ("human", HUMAN)):
        ks = [k for k in keys if k in D[s] and task[k] in tl]; ps = [D[s][k][2] for k in ks]; ys = [D[s][k][1] for k in ks]
        fold = np.array([zlib.crc32(k[0].encode()) % 2 for k in ks]); rec_ps = list(ps); Ts = []
        for f in (0, 1):
            tr = [i for i in range(len(ks)) if fold[i] != f]; te = [i for i in range(len(ks)) if fold[i] == f]
            Tb = fit_T([ps[i] for i in tr], [ys[i] for i in tr]); Ts.append(Tb)
            for i, q in zip(te, temp([ps[i] for i in te], Tb)): rec_ps[i] = q
        o[nm] = {"as_scored": calib(ps, ys), "recalibrated": calib(rec_ps, ys), "n": len(ks), "fold_T": Ts}
    res["calibration"][s] = o
# differences of differences on the shared resamples (specialisation vs scale), on the questions all three systems answered
def did(a_, b_, c_, d_, tl, kind):
    ks = [k for k in keys if all(k in D[s] for s in (a_, b_, c_, d_)) and task[k] in tl]
    if not ks: return None
    if kind == "tm": v = {s: tmean(s, ks, tl)[1] for s in {a_, b_, c_, d_}}
    else: v = {s: acc(*vec(s, ks))[1] for s in {a_, b_, c_, d_}}
    return ci((v[a_] - v[b_]) - (v[c_] - v[d_]))
res["did"] = {}
for key, args in (("r3_did27_tm_ci", ("v3_27", "stock27", "stock27", "stock9", BENCH, "tm")), ("r3_did27_htm_ci", ("v3_27", "stock27", "stock27", "stock9", HUMAN, "tm")),
                  ("r3_did9_tm_ci", ("v3_9", "stock9", "stock27", "stock9", BENCH, "tm")), ("r3_did27_hpool_ci", ("v3_27", "stock27", "stock27", "stock9", HUMAN, "pool"))):
    if all(s in D for s in args[:4]): res["did"][key] = did(*args)
# paired calibration differences on human-assigned questions (as scored), with 95% intervals on the shared resamples
def cov5_boot(s, ks):
    conf = np.array([D[s][k][2].max() for k in ks]); c, w = vec(s, ks); o = np.argsort(-conf, kind="stable"); w = w[:, o]; c = c[o]
    cw = np.cumsum(w, 1); risk = np.cumsum(w * (1 - c), 1) / np.maximum(cw, 1e-9); okm = risk <= 0.05
    last = okm.shape[1] - 1 - np.argmax(okm[:, ::-1], 1); has = okm.any(1)
    return np.where(has, cw[np.arange(len(cw)), last] / np.maximum(cw[:, -1], 1e-9), 0.0)
res["calibration_pairs"] = {}
for a_, b_ in (("v3_27", "stock27"), ("v3_9", "stock9")):
    if a_ in D and b_ in D:
        ks = [k for k in keys if k in D[a_] and k in D[b_] and task[k] in HUMAN]
        if not ks: continue
        pa = calib([D[a_][k][2] for k in ks], [D[a_][k][1] for k in ks]); pb = calib([D[b_][k][2] for k in ks], [D[b_][k][1] for k in ks])
        res["calibration_pairs"][f"{a_}-{b_}"] = {"n": len(ks), "cov5_d": pa["cov5"] - pb["cov5"], "cov5_ci": ci(cov5_boot(a_, ks) - cov5_boot(b_, ks)),
                                                 "ece_d": pa["ece"] - pb["ece"], "conf_err_d": pa["conf_err"] - pb["conf_err"], "scope": "human-assigned, as scored"}
# wording: held-out vs seen, pilot test file questions only (compare.wording_of), main pairs
try:
    wd = C.wording_of(str(FINAL / "test.jsonl"))
    for a_, b_ in (("v3_27", "stock27"), ("v3_9", "stock9")):
        if a_ in D and b_ in D:
            o = {}; boot = {}
            for wv in ("held_out", "seen"):   # compare.wording_of's labels; published as "heldout" / "seen"
                ks = [k for k in keys if k in D[a_] and k in D[b_] and task[k] in BENCH and wd.get(k) == wv]
                if ks:
                    ca, wa = vec(a_, ks); cb, _ = vec(b_, ks); pa, ba = acc(ca, wa); pb, bb = acc(cb, wa); boot[wv] = ba - bb
                    o["heldout" if wv == "held_out" else wv] = {"n": len(ks), "d": pa - pb, "ci": ci(ba - bb), "acc_a": pa, "acc_b": pb}
            if len(boot) == 2: o["did"] = {"d": o["heldout"]["d"] - o["seen"]["d"], "ci": ci(boot["held_out"] - boot["seen"])}
            res.setdefault("wording", {})[f"{a_}-{b_}"] = o
except Exception as ex:
    res["wording_error"] = str(ex)[-500:]
OUTF.write_text(json.dumps(res, indent=1)); print("ok")
'''


def analyse():
    e = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}
    sh([KEV_PY, "-c", ANALYSE, CODE, WORK, FINAL, OUT / "eval_v3.json"], e, WORK / "analyse.log")


def main():
    for p in (WORK, CODE): p.mkdir(parents=True, exist_ok=True)
    for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
    patch = CODE / "kev_multigpu.patch"
    if subprocess.run(["git", "-C", KEV_DIR, "apply", "--check", "--reverse", patch], capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", KEV_DIR, "apply", patch], check=True)
    if "--analyse_only" in os.sys.argv:   # recompute eval_v3.json from the rows already scored (CPU)
        phase("analyse", analyse); status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save(); return
    if len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]) < 4: raise SystemExit("needs 4 GPUs")
    if not (WORK / "new.jsonl").exists() and not phase("build", build): return
    phase("score", score)
    phase("analyse", analyse)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    main()
