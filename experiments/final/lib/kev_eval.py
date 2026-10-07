"""Score a Kev checkpoint, or predictions made elsewhere (Laya), on labelled records with Kev's own metrics.

Runs inside the Kev venv:
    python kev_eval.py --run jaredpalmer/kev-9b --data dev.jsonl --out runs/x          # a Kev checkpoint (local or Hub)
    python kev_eval.py --preds laya.jsonl --data dev.jsonl --out runs/laya               # precomputed {"id", "probabilities"}
Multi-GPU, identical results: score shards in parallel (one process per GPU or GPU pair), then replay their stored predictions
through Kev's evaluate_records on the full file, in record order (rows, report, summary and temperature as one process would give):
    python kev_eval.py --run CKPT --data dev.jsonl --out runs/x.s0of4 --shard 0/4      # records 0, 4, 8, ...
    python kev_eval.py --run CKPT --data dev.jsonl --out runs/x --merge runs/x.s0of4,runs/x.s1of4,...
Kev-27B on 48 GB cards: KEV_DEVICE_MAP=auto KEV_DTYPE=bf16 with two GPUs visible (needs jobs/kev_multigpu.patch).

Writes Kev's report.json and rows.json into --out, plus summary.json: per task accuracy / Brier / ECE / NLL and, for
yes/no tasks, AUROC. States up to the serving limit (8,192 tokens) are scored, not only the 384-token training context.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.data import load_records
from kev.metrics import grouped_metrics
from kev.benchmark import ContextOverflow   # the class evaluate_records catches for skip_overlong
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT


class Precomputed:
    """Predictor over probabilities computed by another runtime, keyed by kev.data.load_records record id."""
    temperature = 1.0

    def __init__(self, path):
        self.p = {}
        for line in Path(path).read_text().splitlines():
            r = json.loads(line); self.p[r["id"]] = r["probabilities"]

    def __call__(self, record):
        return {"probabilities": self.p[record["_meta"]["id"]], "latency_ms": 0.0, "input_tokens": 0}


class Replay:
    """Predictions stored by --shard runs (predictions.jsonl); records a shard rejected as over-long raise ContextOverflow again."""

    def __init__(self, dirs):
        self.p, self.rej, temps = {}, {}, set()
        for d in map(Path, dirs):
            if not (d / "summary.json").exists(): raise FileNotFoundError(f"shard {d} has no summary.json")
            for line in (d / "predictions.jsonl").read_text().splitlines():
                x = json.loads(line); self.p[x["id"]] = x["prediction"]
            if (d / "rejected.json").exists():
                for x in json.loads((d / "rejected.json").read_text()): self.rej[x["id"]] = x["error"]
            temps.add(json.loads((d / "summary.json").read_text()).get("temperature"))
        if len(temps) != 1: raise ValueError(f"shards disagree on the temperature: {temps}")
        self.temperature = temps.pop()

    def __call__(self, record):
        i = record["_meta"]["id"]
        if i in self.rej: raise ContextOverflow(self.rej[i])
        return self.p[i]


def summary(rows, report):
    out = {"coverage": report["coverage"], "overall": {k: report["clean"][k] for k in ("n", "acc", "brier", "ece", "nll")}, "tasks": {}}
    for task, m in grouped_metrics(rows, "task").items():
        t = {k: m[k] for k in ("n", "acc", "brier", "ece", "nll")}
        rs = [r for r in rows if r["task"] == task]
        if rs[0]["type"] == "noul" and len({r["label"] for r in rs}) == 2:
            t["auroc"] = float(roc_auc_score([r["label"] for r in rs], [r["p"][1] for r in rs]))
            t["prevalence"] = float(np.mean([r["label"] for r in rs]))
        out["tasks"][task] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--preds")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", help="i/n: score records i, i+n, i+2n, ... of --data only")
    ap.add_argument("--merge", help="comma-separated --shard output directories to replay on the full --data (--run labels the result)")
    a = ap.parse_args()
    if bool(a.run) == bool(a.preds): ap.error("give exactly one of --run or --preds")
    records = load_records(a.data, source="rad")
    if a.shard:
        i, n = map(int, a.shard.split("/")); records = records[i::n]
    if a.merge: predictor = Replay(a.merge.split(","))
    else: predictor = Precomputed(a.preds) if a.preds else LocalPredictor(a.run, a.device, LoadOptions.from_env(), context=SERVING_CONTEXT)
    report, rows = evaluate_records(records, predictor, a.out, skip_overlong=True)
    s = summary(rows, report)
    s.update(run=a.run or a.preds, data=a.data, **({"shard": a.shard} if a.shard else {}), **({"merged_shards": len(a.merge.split(","))} if a.merge else {}), temperature=getattr(predictor, "temperature", None), latency_ms=report.get("latency_ms"))
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s["overall"]))


if __name__ == "__main__":
    main()
