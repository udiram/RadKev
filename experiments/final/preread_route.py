"""Pre-read routing (post hoc; no training; Addendum 2, amendment 2): Eurorad subspecialty questions with and without the findings.

    python jobs/submit.py jobs/preread_route.py --gpus 4 --cpus 8 --mem 96 --timeout 240 --expire 900 --outputs 'preread_route.json'

Every eurorad_route question of runs/test-final/test.jsonl is posed twice: with its original state (route_full.jsonl) and with
the imaging findings removed, keeping age, sex and clinical history (route_preread.jsonl). Wording, options and keys are
unchanged. Decision models are scored with jobs/kev_eval.py as in jobs/final_test.py; the LLMs with teacher.py predict (letter
scoring, MedGemma with --think_off) as in jobs/medgemma_fix.py. Paired bootstrap over records (2,000 resamples). Publishes
aggregates only.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

INCLUDE = ["teacher.py", "build_data.py", "jobs/kev_eval.py", "jobs/compare.py", "jobs/kev_multigpu.patch"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
TEST = RADKEV / "runs/test-final/test.jsonl"
WORK = RADKEV / "preread_route"; CODE = WORK / "code"; RUNS = WORK / "runs"
ARGS = dict(zip(sys.argv[1::2], sys.argv[2::2]))   # --v3_27 runs/<tag>-<model> --v3_9 ... (the v3 training run that succeeded)
def _first(v):   # "runs/a|runs/b": the first run whose checkpoint exists
    return next((x for x in v.split("|") if (RADKEV / x / "checkpoint/head.pt").exists()), v.split("|")[0])


KEV_MODELS = {
    "v3_27": (str(RADKEV / _first(ARGS.get("--v3_27", "runs/v3f-kev-27b-dp")) / "checkpoint"), 2),
    "v3_9": (str(RADKEV / _first(ARGS.get("--v3_9", "runs/v3f-kev-9b-dp|runs/v3g-kev-9b")) / "checkpoint"), 1),
    "v2_27": (str(RADKEV / "runs/v2mg-kev-27b/checkpoint"), 2),
    "stock27": ("jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", 2),
    "r9": (str(RADKEV / "runs/v2x9-kev-9b/checkpoint"), 1),
    "stock9": ("jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", 1),
}
LLMS = {"qwen38": ("Qwen/Qwen3.8-27B", []), "medgemma_fix": ("google/medgemma-27b-text-it", ["--think_off"])}
CONDS = ("full", "preread")
FINDINGS = "imaging findings"
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}, "checkpoints": {m: KEV_MODELS[m][0] for m in ("v3_27", "v3_9")}}
LOCK = threading.Lock()


def save():
    with LOCK: (OUT / "preread_route.json").write_text(json.dumps(status, indent=1))


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


def preread(state):
    """The state without its imaging findings: a dict loses the key, a text state loses the 'Imaging findings:' line."""
    if isinstance(state, dict):
        assert FINDINGS in state, sorted(state)
        return {k: v for k, v in state.items() if k != FINDINGS}
    lines = state.split("\n")
    kept = [l for l in lines if not l.lower().startswith(FINDINGS + ":")]
    assert len(kept) == len(lines) - 1, lines[:1]
    return "\n".join(kept)


def build():
    n = {"records": 0, "dict_states": 0, "text_states": 0}
    with open(WORK / "route_full.jsonl", "w") as ff, open(WORK / "route_preread.jsonl", "w") as fp:
        for i, line in enumerate(open(TEST)):
            if not line.strip(): continue
            r = json.loads(line)
            qs = {k: v for k, v in r["questions"].items() if str(v.get("src", "")) == "eurorad_route"}
            if not qs: continue
            meta = dict(r.get("_meta", {})); meta.setdefault("id", f"rad/{i}"); meta.setdefault("group_id", f"rad/{i}")
            n["records"] += 1; n["dict_states" if isinstance(r["state"], dict) else "text_states"] += 1
            ff.write(json.dumps({"state": r["state"], "questions": qs, "_meta": meta}, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"state": preread(r["state"]), "questions": qs, "_meta": meta}, ensure_ascii=False) + "\n")
    status["built"] = n; save()


def env_for(devs, cards):
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ",".join(devs[:cards])}
    if cards == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    return env


def score_kev(m, devs):
    path, cards = KEV_MODELS[m]
    if path.startswith("/") and not (Path(path) / "head.pt").exists(): raise RuntimeError(f"no checkpoint at {path}")
    for c in CONDS:
        run([KEV_PY, CODE / "kev_eval.py", "--run", path, "--data", WORK / f"route_{c}.jsonl", "--out", RUNS / c / m], env_for(devs, cards), WORK / f"{m}.log")


def score_llm(m, devs):
    mid, extra = LLMS[m]; env = env_for(devs, 2)
    for c in CONDS:
        data = WORK / f"route_{c}.jsonl"; raw = WORK / f"{m}_{c}.preds.jsonl"; fixed = WORK / f"{m}_{c}.preds_ids.jsonl"
        run([KEV_PY, CODE / "teacher.py", "predict", "--model", mid, "--data", data, "--out", raw, "--batch", "16", *extra], env, WORK / f"{m}.log")
        ids = [json.loads(l)["_meta"]["id"] for l in open(data) if l.strip()]
        with open(fixed, "w") as f:
            for l in open(raw):
                x = json.loads(l); x["id"] = ids[int(x["id"].split("/")[1])]; f.write(json.dumps(x) + "\n")
        run([KEV_PY, CODE / "kev_eval.py", "--preds", fixed, "--data", data, "--out", RUNS / c / m], env, WORK / f"{m}.log")


ANALYSE = r'''
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
runs, out = Path(sys.argv[2]), Path(sys.argv[3])
conds = ("full", "preread")
models = sorted(p.name for p in (runs / "full").iterdir() if (p / "rows.json").exists() and (runs / "preread" / p.name / "rows.json").exists())
R = {(m, c): {(r["id"], r["question"]): r for r in scored_rows(json.loads((runs / c / m / "rows.json").read_text()))} for m in models for c in conds}
keys = sorted(set.intersection(*[set(v) for v in R.values()]))
ok = lambda r: int(np.argmax(r["p"]) == int(r["label"]))
C = {mc: np.array([ok(R[mc][k]) for k in keys], float) for mc in R}
recs = sorted({k[0] for k in keys}); ridx = {r: i for i, r in enumerate(recs)}; q2r = np.array([ridx[k[0]] for k in keys])
rng = np.random.default_rng(20261005); Bn = 2000
W = np.zeros((Bn, len(recs)), np.float32)
for b in range(Bn): np.add.at(W[b], rng.integers(0, len(recs), len(recs)), 1)
QW = W[:, q2r]; den = np.maximum(QW.sum(1), 1e-9)
boot = lambda x: (QW @ x) / den
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
res = {"n_questions": len(keys), "n_records": len(recs), "models": {}, "pairs": {}}
for m in models:
    f, p = C[(m, "full")], C[(m, "preread")]
    res["models"][m] = {"full": float(f.mean()), "full_ci": ci(boot(f)), "preread": float(p.mean()), "preread_ci": ci(boot(p)),
                        "change": float(p.mean() - f.mean()), "change_ci": ci(boot(p) - boot(f))}
for a, b in (("v3_27", "stock27"), ("v3_27", "v2_27"), ("v3_27", "qwen38"), ("v3_27", "medgemma_fix"), ("v2_27", "stock27"), ("r9", "stock9"), ("v3_9", "stock9"), ("v3_9", "r9")):
    if a in models and b in models:
        res["pairs"][f"{a}-{b}"] = {c: {"diff": float(C[(a, c)].mean() - C[(b, c)].mean()), "ci": ci(boot(C[(a, c)]) - boot(C[(b, c)]))} for c in conds}
# errors under the pre-read state as key -> predicted section (labels only, no text); skipped unless row label indices follow the
# criteria order of the data file
qopts = {}
for l in open(Path(sys.argv[4])):
    if l.strip():
        r = json.loads(l)
        for qid, q in r["questions"].items(): qopts[(r["_meta"]["id"], qid)] = (list(q["criteria"]), q["label"])
aligned = all(qopts[k][0][int(R[(m, c)][k]["label"])] == qopts[k][1] for m in models for c in conds for k in keys if k in qopts)
res["label_order_aligned"] = aligned
if aligned:
    res["errors_preread"] = {}
    for m in models:
        cnt = {}
        for k in keys:
            r = R[(m, "preread")][k]
            if not ok(r): t = f"{qopts[k][1]} -> {qopts[k][0][int(np.argmax(r['p']))]}"; cnt[t] = cnt.get(t, 0) + 1
        res["errors_preread"][m] = dict(sorted(cnt.items(), key=lambda x: -x[1])[:15])
out.write_text(json.dumps(res, indent=1)); print("ok")
'''


def analyse():
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}
    run([KEV_PY, "-c", ANALYSE, CODE, RUNS, WORK / "analysis.json", WORK / "route_full.jsonl"], env, WORK / "analyse.log")
    status["result"] = json.loads((WORK / "analysis.json").read_text())


def main():
    for p in (WORK, CODE, RUNS / "full", RUNS / "preread"): p.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    patch = CODE / "kev_multigpu.patch"
    if subprocess.run(["git", "-C", KEV_DIR, "apply", "--check", "--reverse", patch], capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", KEV_DIR, "apply", patch], check=True)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    status["gpus"] = devs; save()
    if len(devs) < 4: raise SystemExit("needs 4 GPUs")
    if not phase("build", build): return
    rounds = ((("v3_27", score_kev, devs[0:2]), ("v2_27", score_kev, devs[2:4])),
              (("stock27", score_kev, devs[0:2]), ("qwen38", score_llm, devs[2:4])),
              (("medgemma_fix", score_llm, devs[0:2]), ("r9", score_kev, devs[2:3]), ("stock9", score_kev, devs[3:4])),
              (("v3_9", score_kev, devs[0:1]),))
    for rnd in rounds:
        ts = [threading.Thread(target=phase, args=(f"score_{m}", (lambda m=m, fn=fn, d=d: fn(m, d)))) for m, fn, d in rnd]
        for t in ts: t.start()
        for t in ts: t.join()
    phase("analyse", analyse)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    main()
