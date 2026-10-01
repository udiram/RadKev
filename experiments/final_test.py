"""Final held-out evaluation: every model scored ONCE on the test split with identical items and metrics, then paired
bootstrap comparisons against the reference (stock Kev-27B). Model selection never looks at these numbers.

    python experiments/final_test.py --tag final --data rad-open,rad-gated,teacher \
        --runs stock27=jaredpalmer/kev-27b,v2_27=v2mg-kev-27b,stock9=jaredpalmer/kev-9b,r9=v2x9-kev-9b \
        --llms qwen38=Qwen/Qwen3.8-27B,medgemma=google/medgemma-27b-text-it --laya 1 \
        --also v2_27,r9

Runs named like "v2mg-kev-27b" resolve to $RADKEV_HOME/runs/<name>/checkpoint; Hub ids and paths pass through. GPU lanes
(two cards each) work through the queue in parallel; 27B models and the LLM baselines take a lane, 9B models one card of
it, Laya runs on CPU in its own environment. Every Kev model is scored in bf16, so every row uses the same numerics.
Scored rows land in $RADKEV_HOME/runs/test-<tag>/<name>/; comparisons.json and per-model summaries in --out.
A model that is already scored there is not scored again, so other scripts (experiments/decision_baselines.py,
experiments/llm_scoring.py) can add rows that a later comparison run picks up with --runs name=cached.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from radkev.paths import DATA, KEV_PY, LAYA_PY, RUNS, resolve_run

HERE = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16"}
LOCK = threading.Lock()


def sh(cmd, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0: raise RuntimeError(f"exit {rc}: {Path(log).read_text()[-1500:]}")


def run_task(task, lane, test, work):
    kind, name, target = task
    out, log = work / name, work / f"{name}.log"
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(lane)}
    if (out / "summary.json").exists(): return {"cached": True}
    t0 = time.time()
    if kind == "kev":
        if "27b" in target.lower(): env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
        else: env["CUDA_VISIBLE_DEVICES"] = lane[0]
        sh([KEV_PY, "-m", "radkev.evaluate", "--run", resolve_run(target), "--data", test, "--out", out], env, log)
    elif kind == "llm":
        preds = work / f"{name}.preds.jsonl"
        sh([KEV_PY, "-m", "radkev.teacher", "predict", "--model", target, "--data", test, "--out", preds], env, log)
        sh([KEV_PY, "-m", "radkev.evaluate", "--preds", preds, "--data", test, "--out", out], env, log)
    elif kind == "laya":
        preds = work / f"{name}.preds.jsonl"
        sh([LAYA_PY, HERE / "laya_predict.py", "--data", test, "--out", preds, "--device", "cpu"], env, log)
        sh([KEV_PY, "-m", "radkev.evaluate", "--preds", preds, "--data", test, "--out", out], env, log)
    return {"hours": round((time.time() - t0) / 3600, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="final"); ap.add_argument("--data", default="rad-open"); ap.add_argument("--split", default="test")
    ap.add_argument("--runs", default="stock27=jaredpalmer/kev-27b", help="name=run pairs; run is a Hub id, a path, a run name, or 'cached'")
    ap.add_argument("--llms", default="", help="name=hf_id pairs, scored by option-letter probabilities (radkev.teacher predict)")
    ap.add_argument("--laya", type=int, default=0)
    ap.add_argument("--reference", default="stock27"); ap.add_argument("--also", default="", help="extra comparison references, passed to radkev.compare")
    ap.add_argument("--no-compare", dest="no_compare", action="store_true", help="score only; a later run with every model does the comparisons")
    ap.add_argument("--out", default="", help="where summaries and comparisons.json go (default: the work dir)")
    a = ap.parse_args()
    work = RUNS / f"test-{a.tag}"; work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.out) if a.out else work; out_dir.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y%m%d-%H%M%S"), "tasks": {}, "split": a.split, "data": a.data}
    save = lambda: (out_dir / "final_test.json").write_text(json.dumps(report, indent=2))
    test = work / f"{a.split}.jsonl"
    if not test.exists():
        lines, counts = [], {}
        for d in a.data.split(","):
            p = DATA / d / f"{a.split}.jsonl"
            if p.exists(): ls = [l for l in p.read_text().splitlines() if l.strip()]; lines += ls; counts[d] = len(ls)
        test.write_text("\n".join(lines) + "\n"); report["test_counts"] = counts
    tasks = [("kev", *kv.split("=", 1)) for kv in a.runs.split(",") if kv] + [("llm", *kv.split("=", 1)) for kv in a.llms.split(",") if kv]
    if a.laya: tasks.append(("laya", "laya", ""))
    cached = [t for t in tasks if t[2] == "cached"]
    for t in cached: report["tasks"][t[1]] = {"kind": "cached", "ok": (work / t[1] / "summary.json").exists()}
    tasks = [t for t in tasks if t[2] != "cached"]
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    lanes = [devs[i:i + 2] for i in range(0, len(devs) - 1, 2)] or [devs]
    q = queue.Queue()
    for t in sorted(tasks, key=lambda t: t[0] == "laya"): q.put(t)

    def worker(lane):
        while True:
            try: t = q.get_nowait()
            except queue.Empty: return
            try: res = {"ok": True, **run_task(t, lane, test, work)}
            except Exception as e: res = {"ok": False, "error": str(e)[-1500:]}
            with LOCK: report["tasks"][t[1]] = {"kind": t[0], "target": t[2], "lane": lane, **res}; save()
    threads = [threading.Thread(target=worker, args=(lane,)) for lane in lanes]
    for th in threads: th.start()
    for th in threads: th.join()
    done = [n for n, r in report["tasks"].items() if r.get("ok")]
    if not a.no_compare:
        for n in done:
            s = work / n / "summary.json"
            if s.exists(): (out_dir / f"test_{n}.json").write_text(s.read_text())
        if a.reference in done:
            p = subprocess.run([str(KEV_PY), "-m", "radkev.compare", str(work), "--models", ",".join(done), "--reference", a.reference,
                                "--also", a.also, "--out", str(out_dir / "comparisons.json"), "--data", str(test)],
                               env=ENV, capture_output=True, text=True)
            report["compare"] = "ok" if p.returncode == 0 else p.stderr[-1500:]
    report["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()
    sys.exit(0 if all(r.get("ok") for r in report["tasks"].values()) else 1)


if __name__ == "__main__":
    main()
