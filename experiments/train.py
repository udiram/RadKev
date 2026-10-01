"""Delta fine-tune Kev on the radiology suite, fit its temperature on dev, and score dev + Kev's own transfer suite.

    python experiments/train.py --models kev-27b --data rad-open,rad-gated,teacher --tag v2mg --replay 1000   # RadKev-27B
    python experiments/train.py --models kev-9b,base-9b --data rad-open,rad-gated,teacher --tag v2x9          # 9B init ablation

Per model, in parallel, each on its share of the visible GPUs (Kev-27B: 2, split over an NVLink pair; 9B: 1):
  kev.train --init_from jaredpalmer/<model> --replay N (Kev's decision-v7 training records, so general skills are kept)
  -> radkev.evaluate on radiology dev (raw) -> Kev's scripts/calibrate_checkpoint.py on those rows -> dev again (calibrated)
  -> the released checkpoint on the same dev items (paired baseline for every task)
  -> transfer-v4 development (Kev's out-of-domain suite) for the fine-tune and the released checkpoint: regression check.
Test splits are not read here; experiments/final_test.py reads them once for the selected run.
Writes $RADKEV_HOME/runs/<tag>-<model>/ (checkpoint, scored rows) and a summary under --out.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from radkev.paths import DATA, KEV_9B, KEV_27B, KEV_DIR, KEV_PY, QWEN35_9B, QWEN35_9B_REV, QWEN38_27B, QWEN38_27B_REV, RUNS

MODELS = {
    "kev-27b": {"init": KEV_27B, "base": QWEN38_27B, "rev": QWEN38_27B_REV, "gpus": 2,
                "train_args": ["--weights_dtype", "bf16", "--lr", "3e-5"],
                "env": {"KEV_DEVICE_MAP": "auto", "KEV_MAX_MEMORY": "0:26GiB,1:40GiB", "KEV_DTYPE": "bf16"}},
    "kev-9b": {"init": KEV_9B, "base": QWEN35_9B, "rev": QWEN35_9B_REV, "gpus": 1, "train_args": ["--lr", "2e-5"], "env": {}},
    # initialisation ablation: the same data and recipe from the plain base (fresh LoRA + pointer head, Kev's own from-scratch
    # lr 2e-4), i.e. no generalist decision pretraining; stock Kev-9B is its paired reference on dev and on the transfer suite
    "base-9b": {"init": None, "ref": KEV_9B, "base": QWEN35_9B, "rev": QWEN35_9B_REV, "gpus": 1, "train_args": ["--lr", "2e-4"], "env": {}},
}
ENV = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
CAUSAL_CONV1D = ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
                 "causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl")   # Kev's own pin (modal_app.py)


def sh(cmd, env, log, cwd=KEV_DIR, timeout=None):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    if rc != 0: raise RuntimeError(f"{' '.join(map(str, cmd[1:3]))} exited {rc}: {Path(log).read_text()[-2500:]}")


def allocate(models):
    """Split the visible GPUs (CUDA_VISIBLE_DEVICES, in order; default all) between the models."""
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    out, i = {}, 0
    for name in models:
        n = MODELS[name]["gpus"]
        if i + n > len(devs): raise RuntimeError(f"not enough GPUs for {models}: have {len(devs)}")
        out[name] = devs[i:i + n]; i += n
    return out


def causal_conv1d():
    """The DeltaNet short convolution falls back to slow reference code without this kernel (forward and backward)."""
    if subprocess.run([str(KEV_PY), "-c", "import causal_conv1d"], capture_output=True).returncode != 0:
        subprocess.run(["uv", "pip", "install", "--python", str(KEV_PY), "--no-deps", CAUSAL_CONV1D], capture_output=True)
    return subprocess.run([str(KEV_PY), "-c", "import causal_conv1d; print(causal_conv1d.__version__)"], capture_output=True, text=True).stdout.strip() or "missing"


def prepare(a):
    """Concatenate the requested suites into data/mix-<tag>/{train,dev}.jsonl (optionally a seeded fraction of train)."""
    mix = DATA / f"mix-{a.tag}"; mix.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "dev"):
        lines = []
        for d in a.data.split(","):
            p = DATA / d / f"{split}.jsonl"
            if p.exists(): lines += [l for l in p.read_text().splitlines() if l.strip()]
            counts[f"{d}/{split}"] = sum(1 for _ in p.open()) if p.exists() else 0
        if split == "train" and a.train_fraction < 1:   # data-efficiency runs: a seeded, fixed subset of the same training mix
            random.Random(0).shuffle(lines); lines = lines[:max(1, int(len(lines) * a.train_fraction))]
            counts["train_fraction"] = a.train_fraction; counts["train_kept"] = len(lines)
        (mix / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    return {"mix": str(mix), "counts": counts, "causal_conv1d": causal_conv1d()}


def one_model(name, a, mix, devices, out_dir):
    m = MODELS[name]; env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devices), **m["env"]}
    run_dir = RUNS / f"{a.tag}-{name}"; run_dir.mkdir(parents=True, exist_ok=True)
    log, ckpt = RUNS / f"{a.tag}-{name}.log", run_dir / "checkpoint"
    res = {}
    if not (ckpt / "head.pt").exists():
        t0 = time.time()
        sh([KEV_PY, "-m", "kev.train", "--data", mix / "train.jsonl", "--suite", KEV_DIR / "evals/v7/decision-v7", "--replay", a.replay,
            *(["--init_from", m["init"]] if m["init"] else []), "--base", m["base"], "--base_revision", m["rev"], "--dtype", "bf16", "--checkpointing", "1",
            "--batch", "1", "--accum", "8", "--epochs", a.epochs, "--max_state", a.max_state, "--p_none_pair", "0.25",
            "--device", "cuda", "--seed", a.seed, "--out", ckpt, *m["train_args"],
            *(["--max_steps", a.max_steps.get(name, "0")] if a.max_steps.get(name, "0") != "0" else [])], env, log)
        res["train_hours"] = round((time.time() - t0) / 3600, 2)
    for f in ("training_metrics.json", "training_config.json"):
        if (ckpt / f).exists(): (out_dir / f"{name}_{f}").write_text((ckpt / f).read_text())
    # dev with the temperature inherited from the released checkpoint, fit a new temperature on those rows, then dev again
    for stage in ("dev_raw", "dev_cal"):
        out = run_dir / stage
        if not (out / "summary.json").exists():
            if stage == "dev_cal":
                sh([KEV_PY, KEV_DIR / "scripts/calibrate_checkpoint.py", "--run", ckpt, "--rows", run_dir / "dev_raw/rows.json"], env, log)
            sh([KEV_PY, "-m", "radkev.evaluate", "--run", ckpt, "--data", mix / "dev.jsonl", "--out", out], env, log)
        (out_dir / f"{name}_{stage}.json").write_text((out / "summary.json").read_text())
    # the released checkpoint on the same dev items, so every task (including new ones) has a paired stock baseline
    out = run_dir / "dev_stock"
    if not (out / "summary.json").exists():
        sh([KEV_PY, "-m", "radkev.evaluate", "--run", m.get("ref") or m["init"], "--data", mix / "dev.jsonl", "--out", out], env, log)
    (out_dir / f"{name}_dev_stock.json").write_text((out / "summary.json").read_text())
    # Kev's own out-of-domain development suite, fine-tune vs released checkpoint (paired items)
    for tag, run_id in (("transfer_ft", ckpt), ("transfer_base", m.get("ref") or m["init"])):
        out = run_dir / tag
        if not (out / "report.json").exists():
            sh([KEV_PY, "-m", "kev.benchmark", "--run", run_id, "--suite", KEV_DIR / "evals/v4/transfer-v4", "--out", out, "--device", "cuda"], env, log)
        rep = json.loads((out / "report.json").read_text())
        (out_dir / f"{name}_{tag}.json").write_text(json.dumps({"clean": rep["clean"], "calibrated_clean": rep.get("calibrated_clean"), "coverage": rep["coverage"]}, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="kev-27b", help=f"comma-separated: {', '.join(MODELS)}")
    ap.add_argument("--data", default="rad-open", help="suites under $RADKEV_HOME/data, e.g. rad-open,rad-gated,teacher")
    ap.add_argument("--tag", default="v1"); ap.add_argument("--replay", default="1000"); ap.add_argument("--epochs", default="1")
    ap.add_argument("--max_state", default="1536"); ap.add_argument("--seed", default="0")
    ap.add_argument("--max_steps", default="", help="per-model optimizer-step caps for pilots, e.g. kev-27b=400")
    ap.add_argument("--train_fraction", type=float, default=1.0, help="train on a seeded fraction of the training mix (data-efficiency runs)")
    ap.add_argument("--out", default="", help="summary directory (default: $RADKEV_HOME/runs/<tag>-summary)")
    a = ap.parse_args()
    a.max_steps = dict(kv.split("=") for kv in a.max_steps.split(",") if kv)
    out_dir = Path(a.out) if a.out else RUNS / f"{a.tag}-summary"; out_dir.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y%m%d-%H%M%S"), "args": vars(a), "steps": {}}
    save = lambda: (out_dir / "train.json").write_text(json.dumps(report, indent=2))
    t0 = time.time()
    try: prep = prepare(a); report["steps"]["prepare"] = {"ok": True, **prep}
    except Exception as e: report["steps"]["prepare"] = {"ok": False, "error": f"{e}"[-2000:]}; save(); sys.exit(1)
    save()
    names = a.models.split(",")
    with ThreadPoolExecutor(max_workers=4) as pool:   # the work is in subprocesses; threads just wait on them
        alloc = allocate(names)
        futures = {name: pool.submit(one_model, name, a, Path(prep["mix"]), alloc[name], out_dir) for name in names}
        for name, fut in futures.items():
            try: report["steps"][name] = {"ok": True, **fut.result()}
            except Exception as e: report["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[-2500:], "trace": traceback.format_exc()[-1000:]}
            save()
    report["hours"] = round((time.time() - t0) / 3600, 2); save()
    print(json.dumps({k: v["ok"] for k, v in report["steps"].items()}))
    sys.exit(0 if all(s["ok"] for s in report["steps"].values()) else 1)


if __name__ == "__main__":
    main()
