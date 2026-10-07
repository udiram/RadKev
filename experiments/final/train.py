"""Delta fine-tune Kev on the radiology suite, fit its temperature on dev, and score dev + Kev's own transfer suite.

    python jobs/submit.py jobs/train.py --gpus 2 --timeout 4320 -- --models kev-27b --data rad-open,rad-gated,teacher --tag v1

Per model, in parallel, each on its share of the GPUs the scheduler allocated (Kev-27B: 2, split; Kev-9B: 1):
  kev.train --init_from jaredpalmer/<model> --replay N (Kev's decision-v7 training records, so general skills are kept)
  -> kev_eval on radiology dev (raw) -> scripts/calibrate_checkpoint.py on those dev rows -> kev_eval on dev (calibrated)
  -> transfer-v4 development (Kev's out-of-domain suite) for both the fine-tune and the released checkpoint: regression check.
Test splits are not read here; jobs/final_test.py reads them once for the selected run.
"""
import argparse
import json
import random
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

INCLUDE = ["build_data.py", "jobs/kev_eval.py", "jobs/kev_multigpu.patch", "jobs/kev_lora_dp.py", "jobs/kev_locked.py"]
BUNDLE = {}

# Kev revisions are pinned: the Hub repos received new weights on 2026-09-30; the paper's runs start from these
MODELS = {
    "kev-27b": {"init": "jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", "base": "Qwen/Qwen3.8-27B", "rev": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "gpus": 2,
                "train_args": ["--weights_dtype", "bf16", "--lr", "3e-5"],
                "env": {"KEV_DEVICE_MAP": "auto", "KEV_MAX_MEMORY": "0:26GiB,1:40GiB", "KEV_DTYPE": "bf16", "RADKEV_LOG_ACC": "1"}},
    "kev-9b": {"init": "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", "base": "Qwen/Qwen3.5-9B-Base", "rev": "68c46c4b3498877f3ef123c856ecfde50c39f404", "gpus": 1,
               "train_args": ["--lr", "2e-5"], "env": {"RADKEV_LOG_ACC": "1"}},
    # Kev-27B on all four A6000s (v3, 2026-10-05): two data-parallel ranks, each holding the split backbone on one NVLink pair
    # (jobs/kev_lora_dp.py); --accum 4 per rank keeps the 8 records per optimizer step of the two-GPU recipe
    "kev-27b-dp": {"init": "jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", "base": "Qwen/Qwen3.8-27B", "rev": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "gpus": 4, "ranks": 2, "accum": "4",
                   "train_args": ["--weights_dtype", "bf16", "--lr", "3e-5"],
                   "env": {"KEV_DEVICE_MAP": "auto", "KEV_MAX_MEMORY": "0:26GiB,1:40GiB", "KEV_DTYPE": "bf16"}},
    # 9B on all four A6000s (v3, 2026-10-05): four data-parallel ranks, one GPU each, 2 records per micro-batch, no accumulation
    # (4 x 2 = 8 records per optimizer step, as in the one-GPU recipe)
    "kev-9b-dp": {"init": "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", "base": "Qwen/Qwen3.5-9B-Base", "rev": "68c46c4b3498877f3ef123c856ecfde50c39f404",
                  "gpus": 4, "ranks": 4, "accum": "1", "train_args": ["--lr", "2e-5"], "env": {"RADKEV_LOG_ACC": "1"}},
    "base-9b-dp": {"init": None, "ref": "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", "base": "Qwen/Qwen3.5-9B-Base", "rev": "68c46c4b3498877f3ef123c856ecfde50c39f404",
                   "gpus": 4, "ranks": 4, "accum": "1", "train_args": ["--lr", "2e-4"], "env": {"RADKEV_LOG_ACC": "1"}},
    # initialisation ablation: the same data and recipe from the plain base (fresh LoRA + pointer head, Kev's own from-scratch
    # lr 2e-4), i.e. no generalist decision pretraining; stock Kev-9B is its paired reference on dev and on the transfer suite
    "base-9b": {"init": None, "ref": "jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", "base": "Qwen/Qwen3.5-9B-Base", "rev": "68c46c4b3498877f3ef123c856ecfde50c39f404", "gpus": 1,
                "train_args": ["--lr", "2e-4"], "env": {"RADKEV_LOG_ACC": "1"}},
    # size-matched control for the fine-tuned Laya (421M) comparison: Kev-0.8B with RadKev-9B's recipe (Kev's own delta lr
    # 2e-5, the base revision from kev-0.8b's training_config.json)
    "kev-0.8b": {"init": "jaredpalmer/kev-0.8b", "base": "Qwen/Qwen3.5-0.8B-Base", "rev": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68", "gpus": 1,
                 "train_args": ["--lr", "2e-5"], "env": {}},
}
SERVICES = []   # inference services sharing the GPUs on the compute node, stopped before training (names omitted)
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"
KEV_DIR, CODE = RADKEV / "kev", RADKEV / "code"
KEV_PY = KEV_DIR / ".venv/bin/python"
uid = os.getuid()
ENV = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONPATH": str(KEV_DIR), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
ENV.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}"); ENV.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
report = {"started": time.strftime("%Y%m%d-%H%M%S"), "steps": {}}


def save(): (OUT / "train.json").write_text(json.dumps(report, indent=2))


def sh(cmd, env, log, cwd=KEV_DIR, timeout=None):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    if rc != 0: raise RuntimeError(f"{' '.join(map(str, cmd[1:3]))} exited {rc}: {Path(log).read_text()[-2500:]}")


def pids_now():
    """Tasks in this job's cgroup and its limit (the bridge caps a job's tasks)."""
    try:
        cg = open("/proc/self/cgroup").read().strip().splitlines()[-1].split(":", 2)[-1].lstrip("/")
        b = Path("/sys/fs/cgroup") / cg
        return f"{(b / 'pids.current').read_text().strip()}/{(b / 'pids.max').read_text().strip()}"
    except Exception as e: return repr(e)


def gpu_env(devices, extra):
    return {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devices), **extra}


def allocate(models):
    """Split the scheduler's allocation (CUDA_VISIBLE_DEVICES, in the job's order) between the models."""
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    out, i = {}, 0
    for name in models:
        n = MODELS[name]["gpus"]
        if i + n > len(devs): raise RuntimeError(f"not enough GPUs allocated for {models}: have {len(devs)}")
        out[name] = devs[i:i + n]; i += n
    return out


CAUSAL_CONV1D = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
                 "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl")   # Kev's own pin (modal_app.py)


def ensure_causal_conv1d(kev_py, uv):
    """The DeltaNet short convolution falls back to slow reference code without this kernel (forward and backward)."""
    if subprocess.run([str(kev_py), "-c", "import causal_conv1d"], capture_output=True).returncode != 0:
        subprocess.run([str(uv), "pip", "install", "--python", str(kev_py), "--no-deps", CAUSAL_CONV1D], capture_output=True)
    return subprocess.run([str(kev_py), "-c", "import causal_conv1d; print(causal_conv1d.__version__)"], capture_output=True, text=True).stdout.strip() or "missing"

def prepare(a):
    CODE.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    active = {s: subprocess.run(["systemctl", "--user", "is-active", s], env=ENV, capture_output=True, text=True).stdout.strip() for s in SERVICES}
    if "active" in active.values(): subprocess.run(["systemctl", "--user", "stop", *SERVICES], env=ENV); time.sleep(15)
    patch = CODE / "kev_multigpu.patch"
    if subprocess.run(["git", "-C", KEV_DIR, "apply", "--check", "--reverse", patch], capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", KEV_DIR, "apply", patch], check=True)
    if True:   # radkev logging/data-parallel edits of kev/train.py, all gated by environment variables (jobs/kev_lora_dp.py)
        subprocess.run([str(KEV_PY), str(CODE / "kev_lora_dp.py"), "apply", str(KEV_DIR)], check=True)
    cc1d = ensure_causal_conv1d(KEV_PY, RADKEV / "tools/bin/uv")
    mix = RADKEV / "data" / f"mix-{a.tag}"; mix.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "dev"):
        lines = []
        for d in a.data.split(","):
            p = RADKEV / "data" / d / f"{split}.jsonl"
            if p.exists(): lines += [l for l in p.read_text().splitlines() if l.strip()]
            counts[f"{d}/{split}"] = sum(1 for _ in p.open()) if p.exists() else 0
        if split == "train" and a.train_fraction < 1:   # data-efficiency runs: a seeded, fixed subset of the same training mix
            random.Random(0).shuffle(lines); lines = lines[:max(1, int(len(lines) * a.train_fraction))]
            counts["train_fraction"] = a.train_fraction; counts["train_kept"] = len(lines)
        (mix / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    return {"services": active, "mix": str(mix), "counts": counts, "causal_conv1d": cc1d}


def one_model(name, a, mix, devices):
    m = MODELS[name]; env = gpu_env(devices, m["env"])
    run_dir = RADKEV / "runs" / f"{a.tag}-{name}"; log = RADKEV / "runs" / f"{a.tag}-{name}.log"
    ckpt = run_dir / "checkpoint"
    res = {}
    if not (ckpt / "head.pt").exists():
        t0 = time.time()
        cmd = [KEV_PY, "-m", "kev.train", "--data", mix / "train.jsonl", "--suite", KEV_DIR / "evals/v7/decision-v7", "--replay", a.replay,
               *(["--init_from", m["init"]] if m["init"] else []), "--base", m["base"], "--base_revision", m["rev"], "--dtype", "bf16", "--checkpointing", "1",
               "--batch", a.batch, "--accum", a.accum or m.get("accum", "8"), "--epochs", a.epochs, "--max_state", a.max_state, "--p_none_pair", "0.25",
               "--device", "cuda", "--seed", a.seed, "--out", ckpt, *m["train_args"],
               *(["--max_steps", a.max_steps.get(name, "0")] if a.max_steps.get(name, "0") != "0" else [])]
        ranks = m.get("ranks", 1)
        if ranks == 1: sh(cmd, env, log)
        else:   # one process per GPU pair; NCCL across the pairs; rank 0's log is the run log
            per = len(devices) // ranks
            for attempt in range(2):   # a rank that fails at start-up (seen once: CUDA driver initialisation) gets one retry
                port = str(29500 + (int(time.time()) + attempt * 7) % 400); procs = []; started = time.time()
                renvs = [{**env, "CUDA_VISIBLE_DEVICES": ",".join(devices[r * per:(r + 1) * per]), "RADKEV_LORA_DP": "1", "RADKEV_LOG_ACC": "1",
                          "RANK": str(r), "LOCAL_RANK": "0", "WORLD_SIZE": str(ranks), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": port, "NCCL_DEBUG": "WARN",
                          "OMP_NUM_THREADS": str(max(4, 20 // ranks)), "MKL_NUM_THREADS": str(max(4, 20 // ranks)), "TOKENIZERS_PARALLELISM": "false", "TORCHINDUCTOR_COMPILE_THREADS": "1"} for r in range(ranks)]
                pre = [subprocess.Popen([str(KEV_PY), "-c", "import ctypes,torch; c=ctypes.CDLL('libcuda.so.1'); print('cuInit', c.cuInit(0), 'devices', torch.cuda.device_count(), flush=True)"],
                                        env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for e in renvs]
                with open(log, "a") as f: f.write(f"launcher: preflight {[q.communicate()[0].strip()[-200:] for q in pre]} pids {pids_now()}\n")
                for r in range(ranks):
                    renv = renvs[r]
                    f = open(log if r == 0 else log.with_suffix(f".rank{r}.log"), "a"); f.write(f"\n$ rank {r} (attempt {attempt}): {' '.join(map(str, cmd))}\n"); f.flush()
                    procs.append((subprocess.Popen([str(c) for c in cmd], env=renv, cwd=KEV_DIR, stdout=f, stderr=subprocess.STDOUT), f))
                    time.sleep(10)
                ticks = 0
                while all(pr.poll() is None for pr, _ in procs):
                    time.sleep(10); ticks += 1
                    if ticks <= 90 and ticks % 3 == 0:
                        with open(log, "a") as f: f.write(f"launcher: t={int(time.time() - started)}s pids {pids_now()}\n")
                if any(pr.poll() not in (None, 0) for pr, _ in procs):   # one rank failed: stop the other instead of waiting on NCCL
                    for pr, _ in procs:
                        if pr.poll() is None: pr.terminate()
                    time.sleep(30)
                    for pr, _ in procs:
                        if pr.poll() is None: pr.kill()
                rcs = [pr.wait() for pr, _ in procs]
                for _, f in procs: f.close()
                if not any(rcs): break
                if attempt == 1 or time.time() - started > 1800 or (ckpt / "head.pt").exists():
                    raise RuntimeError(f"kev.train ranks exited {rcs}: {Path(log).read_text()[-2500:]}")
                shutil.rmtree(ckpt, ignore_errors=True)
        res["train_hours"] = round((time.time() - t0) / 3600, 2)
    for f in ("training_metrics.json", "training_config.json"):
        if (ckpt / f).exists(): (OUT / f"{name}_{f}").write_text((ckpt / f).read_text())
    if a.train_only:   # smoke tests: step timing and the rank check, no evaluation
        tail = [l for l in Path(log).read_text().splitlines() if "step " in l or "radkev_dp_param_check" in l or "trainable params" in l][-12:]
        return {**res, "log_tail": tail}
    # Evaluation on every allocated GPU (2026-10-06): lanes of one GPU (two for a 27B), dev scored in record shards and merged by
    # replaying the stored predictions through Kev's evaluate_records (jobs/kev_eval.py --shard/--merge: identical rows and
    # summaries). Phase 1: dev with the checkpoint's own temperature (dev_raw). Then a temperature is fitted on dev_raw, and
    # phase 2 scores dev again (dev_cal), the released checkpoint on the same dev items (dev_stock, a paired stock baseline for
    # every task) and Kev's out-of-domain transfer suite for both. Stock results depend only on the reference checkpoint and
    # the data, so they are cached by (reference, sha256 of the file) and reused by later runs.
    import hashlib, threading, queue as _queue
    g = 2 if "27b" in name else 1
    lanes = [devices[i:i + g] for i in range(0, len(devices) - g + 1, g)] or [devices]
    dev = mix / "dev.jsonl"; ref = m.get("ref") or m["init"]; transfer = KEV_DIR / "evals/v4/transfer-v4"
    cache = RADKEV / "runs/_stock_cache"; cache.mkdir(parents=True, exist_ok=True)
    key = lambda *parts: hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]
    dev_sha = hashlib.sha256(dev.read_bytes()).hexdigest()
    cached = {"dev_stock": cache / key(ref, dev_sha), "transfer_base": cache / key(ref, "transfer-v4")}
    for tag, c in cached.items():
        done = run_dir / tag / ("report.json" if tag.startswith("transfer") else "summary.json")
        if not done.exists() and (c / done.name).exists():
            shutil.rmtree(run_dir / tag, ignore_errors=True); shutil.copytree(c, run_dir / tag)
            with open(log, "a") as f: f.write(f"eval: {tag} reused from {c}\n")
    res["eval_lanes"] = lanes
    LOCK = run_dir / "load.lock"   # checkpoint loads one at a time (host RAM); scoring in parallel (jobs/kev_locked.py)

    def lane_env(use): return gpu_env(use, {k: v for k, v in m["env"].items() if k != "RADKEV_LOG_ACC"})

    def shards(run_id, out):
        """the shard tasks of one dev scoring, plus its merge (run after the shards)"""
        if (out / "summary.json").exists(): return [], None
        n = len(lanes); dirs = [out.parent / f"{out.name}.s{i}of{n}" for i in range(n)]
        def one(use, i):
            if (dirs[i] / "summary.json").exists(): return
            shutil.rmtree(dirs[i], ignore_errors=True)
            sh([KEV_PY, CODE / "kev_locked.py", LOCK, CODE / "kev_eval.py", "--run", run_id, "--data", dev, "--out", dirs[i], "--shard", f"{i}/{n}"], lane_env(use), log)
        def merge():
            shutil.rmtree(out, ignore_errors=True)
            sh([KEV_PY, CODE / "kev_eval.py", "--run", run_id, "--data", dev, "--out", out, "--merge", ",".join(map(str, dirs))], {**ENV, "CUDA_VISIBLE_DEVICES": ""}, log)
        return [(lambda use, i=i: one(use, i)) for i in range(n)], merge

    def transfer_task(tag, run_id):
        out = run_dir / tag
        if (out / "report.json").exists(): return []
        def fn(use):
            shutil.rmtree(out, ignore_errors=True)
            sh([KEV_PY, CODE / "kev_locked.py", LOCK, "kev.benchmark", "--run", run_id, "--suite", transfer, "--out", out, "--device", "cuda"], lane_env(use), log)
        return [fn]

    def run_pool(tasks):
        work = _queue.Queue(); errors = []
        for t in tasks: work.put(t)
        def worker(use):
            while True:
                try: t = work.get_nowait()
                except _queue.Empty: return
                try: t(use)
                except Exception as e: errors.append(e)
        ths = [threading.Thread(target=worker, args=(l,)) for l in lanes]
        for t in ths: t.start()
        for t in ths: t.join()
        if errors: raise errors[0]

    t_eval = time.time()
    raw_tasks, raw_merge = shards(ckpt, run_dir / "dev_raw")
    stock_tasks, stock_merge = shards(ref, run_dir / "dev_stock")
    # phase 1: dev_raw first in the queue; lanes that finish early start on the stock scoring
    run_pool(raw_tasks + stock_tasks + transfer_task("transfer_base", ref))
    if raw_merge: raw_merge()
    if stock_merge: stock_merge()
    # temperature fitted on dev_raw, written into the checkpoint; then dev and the transfer suite with it
    if not (run_dir / "dev_cal/summary.json").exists():
        sh([KEV_PY, KEV_DIR / "scripts/calibrate_checkpoint.py", "--run", ckpt, "--rows", run_dir / "dev_raw/rows.json"], ENV, log)
    cal_tasks, cal_merge = shards(ckpt, run_dir / "dev_cal")
    run_pool(cal_tasks + transfer_task("transfer_ft", ckpt))
    if cal_merge: cal_merge()
    res["eval_minutes"] = round((time.time() - t_eval) / 60, 1)
    for tag, c in cached.items():
        src = run_dir / tag
        if not (c / ("report.json" if tag.startswith("transfer") else "summary.json")).exists():
            shutil.rmtree(c, ignore_errors=True); shutil.copytree(src, c)
    for stage in ("dev_raw", "dev_cal", "dev_stock"):
        (OUT / f"{name}_{stage}.json").write_text((run_dir / stage / "summary.json").read_text())
    for tag in ("transfer_ft", "transfer_base"):
        rep = json.loads((run_dir / tag / "report.json").read_text())
        (OUT / f"{name}_{tag}.json").write_text(json.dumps({"clean": rep["clean"], "calibrated_clean": rep.get("calibrated_clean"), "coverage": rep["coverage"]}, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kev-27b"); ap.add_argument("--data", default="rad-open")
    ap.add_argument("--tag", default="v1"); ap.add_argument("--replay", default="1000"); ap.add_argument("--epochs", default="1")
    ap.add_argument("--max_state", default="1536"); ap.add_argument("--seed", default="0")
    ap.add_argument("--max_steps", default="", help="per-model optimizer-step caps for pilots, e.g. kev-27b=400")
    ap.add_argument("--batch", default="1", help="records per forward pass"); ap.add_argument("--accum", default="", help="micro-batches per optimizer step (default: per model)")
    ap.add_argument("--train_only", action="store_true", help="train, copy the metrics and stop (smoke tests)")
    ap.add_argument("--train_fraction", type=float, default=1.0, help="train on a seeded fraction of the training mix (data-efficiency runs)")
    a = ap.parse_args()
    a.max_steps = dict(kv.split("=") for kv in a.max_steps.split(",") if kv)
    t0 = time.time()
    try: prep = prepare(a); report["steps"]["prepare"] = {"ok": True, **prep}
    except Exception as e: report["steps"]["prepare"] = {"ok": False, "error": f"{e}"[-2000:]}; save(); sys.exit(1)
    save()
    mix = Path(prep["mix"])
    from concurrent.futures import ThreadPoolExecutor   # the work is in subprocesses; threads just wait on them
    with ThreadPoolExecutor(max_workers=4) as pool:
        alloc = allocate(a.models.split(","))
        futures = {name: pool.submit(one_model, name, a, mix, alloc[name]) for name in a.models.split(",")}
        for name, fut in futures.items():
            try: report["steps"][name] = {"ok": True, **fut.result()}
            except Exception as e: report["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[-2500:], "trace": traceback.format_exc()[-1000:]}
            save()
    report["hours"] = round((time.time() - t0) / 3600, 2); save()
    sys.exit(0 if all(s["ok"] for s in report["steps"].values()) else 1)


if __name__ == "__main__":
    main()
