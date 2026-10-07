"""External tests (analysis-plan addendum 2026-10-05; post hoc, no training): RadCases (ACR AC panel and topic) and, when its
records exist, RadGraph-XL (radiologist entity status).

    python jobs/submit.py jobs/external_tests.py --gpus 4 --cpus 8 --mem 64 --timeout 240 --expire 600 --outputs 'external_tests.json,rows_*.json' [-- --sets radcases,radgraph_xl]
    v3 rescore of RadGraph-XL (rows stay on the node; jobs/external_analyse_node.py analyses them): -- --sets radgraph_xl --models v3_27,v3_9 --no_publish

1. build: build_external.radcases() downloads the pinned open sources and writes $RADKEV/external/radcases_{panel,topic}.jsonl;
   one-liners that occur verbatim (normalized) inside any record of $RADKEV/data/*/{train,dev,test}.jsonl are dropped and counted.
   RadGraph-XL records ($RADKEV/external/radgraph_xl.jsonl) are written by jobs/radgraph_xl_build.py.
2. score: RadKev-27B, Kev-27B (two cards each, in parallel), then RadKev-9B, Kev-9B, with jobs/kev_eval.py exactly as in
   jobs/final_test.py; Qwen3.8-27B and MedGemma-27B-text (single-BOS fix, --think_off) with teacher.py predict on every file
   whose questions have at most 16 options (letters A to P).
3. publish: summaries, and compact per-question rows without text (id, question, task, label, argmax, p_label, p_max) for the
   paired bootstrap, which runs on the Mac (paper/external/analyse.py).
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

INCLUDE = ["build_data.py", "build_external.py", "teacher.py", "jobs/kev_eval.py", "jobs/kev_multigpu.patch"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
WORK = RADKEV / "external"; CODE = WORK / "code"; RUNS = WORK / "runs"
KEV_MODELS = {
    "v2_27": (str(RADKEV / "runs/v2mg-kev-27b/checkpoint"), 2),
    "stock27": ("jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", 2),
    "r9": (str(RADKEV / "runs/v2x9-kev-9b/checkpoint"), 1),
    "stock9": ("jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", 1),
}
for _n, _runs, _c in (("v3_27", ("v3f-kev-27b-dp",), 2), ("v3_9", ("v3f-kev-9b-dp", "v3g-kev-9b"), 1)):   # v3 (amendment 8)
    _p = next((RADKEV / "runs" / r / "checkpoint" for r in _runs if (RADKEV / "runs" / r / "checkpoint/head.pt").exists()), None)
    if _p: KEV_MODELS[_n] = (str(_p), _c)
LLMS = {"qwen38": ("Qwen/Qwen3.8-27B", []), "medgemma_fix": ("google/medgemma-27b-text-it", ["--think_off"])}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}
LOCK = threading.Lock()
SSL = {"SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt", "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt"}


def save():
    with LOCK: (OUT / "external_tests.json").write_text(json.dumps(status, indent=1))


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


BUILD = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import build_data as bd, build_external as be
radkev, out = Path(sys.argv[2]), Path(sys.argv[3])
chunks = []
for p in sorted((radkev / "data").glob("*/*.jsonl")):
    for line in open(p):
        if line.strip():
            r = json.loads(line); s = r["state"]
            chunks.append(bd.norm(s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)))
big = " | ".join(chunks)
class Contains:
    def __contains__(self, x): return len(x) > 20 and x in big
rep = be.radcases(out, check_text=Contains())
rep["radkev_records_checked"] = len(chunks)
print(json.dumps(rep))
'''


def build():
    env = {**os.environ, **SSL, "PYTHONPATH": f"{KEV_DIR}:{CODE}"}
    log = WORK / "build.log"
    p = subprocess.run([str(KEV_PY), "-c", BUILD, str(CODE), str(RADKEV), str(WORK)], env=env, cwd=KEV_DIR, capture_output=True, text=True)
    Path(log).write_text(p.stdout + p.stderr)
    if p.returncode: raise RuntimeError(p.stderr[-1800:])
    status["radcases_build"] = json.loads(p.stdout.strip().splitlines()[-1]); save()


def max_options(path):
    m = 0
    for l in open(path):
        if l.strip():
            for q in json.loads(l)["questions"].values(): m = max(m, len(q.get("criteria") or {}) or 2)
    return m


def score_kev(m, devs, files):
    run_, cards = KEV_MODELS[m]
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ",".join(devs[:cards])}
    if cards == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    for f in files:
        out = RUNS / f.stem / m
        if (out / "summary.json").exists(): continue
        run([KEV_PY, CODE / "kev_eval.py", "--run", run_, "--data", f, "--out", out], env, WORK / f"{m}.log")


def score_llm(name, devs, files):
    model, extra = LLMS[name]
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": ",".join(devs)}
    for f in files:
        out = RUNS / f.stem / name
        if (out / "summary.json").exists() or max_options(f) > 16: continue
        preds = WORK / f"{f.stem}.{name}.preds.jsonl"
        run([KEV_PY, CODE / "teacher.py", "predict", "--model", model, "--data", f, "--out", preds, "--batch", "16", *extra], env, WORK / f"{name}.log")
        ids = [json.loads(l)["_meta"]["id"] for l in open(f) if l.strip()]   # predict writes rad/<line>; records carry their own ids
        fixed = preds.with_suffix(".ids.jsonl")
        with open(fixed, "w") as g:
            for l in open(preds):
                x = json.loads(l); x["id"] = ids[int(x["id"].split("/")[1])]; g.write(json.dumps(x) + "\n")
        run([KEV_PY, CODE / "kev_eval.py", "--preds", fixed, "--data", f, "--out", out], env, WORK / f"{name}.log")


ROWS = r'''
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
runs, out = Path(sys.argv[2]), Path(sys.argv[3])
res = {}
for fdir in sorted(p for p in runs.iterdir() if p.is_dir()):
    for mdir in sorted(p for p in fdir.iterdir() if (p / "rows.json").exists()):
        rows = scored_rows(json.loads((mdir / "rows.json").read_text()))
        res.setdefault(fdir.name, {})[mdir.name] = [[r["id"], r["question"], r["task"], int(r["label"]), int(np.argmax(r["p"])),
                                                    round(float(r["p"][int(r["label"])]), 5), round(float(max(r["p"])), 5)] for r in rows]
        s = json.loads((mdir / "summary.json").read_text())
        res.setdefault("_summary", {}).setdefault(fdir.name, {})[mdir.name] = {"overall": s["overall"], "coverage": s.get("coverage"),
            "tasks": {t: {"n": v["n"], "acc": v["acc"], "ece": v["ece"]} for t, v in s["tasks"].items()}, "temperature": s.get("temperature")}
for name, v in res.items():
    (out / f"rows_{name}.json").write_text(json.dumps(v, separators=(",", ":")))
print("ok")
'''


def publish():
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}
    run([KEV_PY, "-c", ROWS, CODE, RUNS, OUT], env, WORK / "rows.log")
    status["summary"] = json.loads((OUT / "rows__summary.json").read_text()); (OUT / "rows__summary.json").unlink()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="radcases")
    ap.add_argument("--models", default="v2_27,stock27,r9,stock9,qwen38,medgemma_fix"); ap.add_argument("--no_publish", action="store_true"); a = ap.parse_args()
    want = set(a.models.split(","))
    sets = a.sets.split(",")
    for p in (WORK, CODE, RUNS): p.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    patch = CODE / "kev_multigpu.patch"
    if subprocess.run(["git", "-C", KEV_DIR, "apply", "--check", "--reverse", patch], capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", KEV_DIR, "apply", patch], check=True)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    status["gpus"] = devs; status["sets"] = sets; save()
    if len(devs) < 4: raise SystemExit("needs 4 GPUs")
    files = []
    if "radcases" in sets:
        if not (WORK / "radcases_panel.jsonl").exists() and not phase("build_radcases", build): return
        files += [WORK / "radcases_panel.jsonl", WORK / "radcases_topic.jsonl"]
    if "radgraph_xl" in sets:
        f = WORK / "radgraph_xl.jsonl"
        if f.exists(): files.append(f)
        else: status["radgraph_xl"] = "records missing: run jobs/radgraph_xl_build.py first"; save()
    status["files"] = {f.stem: {"records": sum(1 for l in open(f) if l.strip()), "max_options": max_options(f)} for f in files}; save()
    big = [m for m in KEV_MODELS if m in want and KEV_MODELS[m][1] == 2]; small = [m for m in KEV_MODELS if m in want and KEV_MODELS[m][1] == 1]
    status["models"] = big + small + [n for n in LLMS if n in want]; save()
    for i in range(0, max(len(big), len(small)), 2):
        pair = [(m, d) for m, d in zip(big[i:i + 2], (devs[0:2], devs[2:4]))]
        ts = [threading.Thread(target=phase, args=(f"score_{m}", (lambda m=m, d=d: score_kev(m, d, files)))) for m, d in pair]
        for t in ts: t.start()
        for t in ts: t.join()
        pair = [(m, d) for m, d in zip(small[i:i + 2], (devs[0:1], devs[2:3]))]
        ts = [threading.Thread(target=phase, args=(f"score_{m}", (lambda m=m, d=d: score_kev(m, d, files)))) for m, d in pair]
        for t in ts: t.start()
        for t in ts: t.join()
    ts = [threading.Thread(target=phase, args=(f"score_{n}", (lambda n=n, d=d: score_llm(n, d, files)))) for n, d in (("qwen38", devs[0:2]), ("medgemma_fix", devs[2:4])) if n in want]
    for t in ts: t.start()
    for t in ts: t.join()
    if not a.no_publish: phase("publish", publish)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    main()
