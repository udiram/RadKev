"""Score a Kev checkpoint, or predictions made elsewhere (Laya), on labelled records with Kev's own metrics.

Runs inside the Kev environment (scripts/setup.sh installs radkev into it):
    python -m radkev.evaluate --run jaredpalmer/kev-9b --data dev.jsonl --out runs/x       # a Kev checkpoint (local or Hub)
    python -m radkev.evaluate --preds laya.jsonl --data dev.jsonl --out runs/laya          # precomputed {"id", "probabilities"}
Kev-27B on 48 GB cards: KEV_DEVICE_MAP=auto KEV_DTYPE=bf16 with two GPUs visible (needs patches/kev_multigpu.patch).

Writes Kev's report.json and rows.json into --out, plus summary.json: per task accuracy / Brier / ECE / NLL and, for
yes/no tasks, AUROC. States up to the serving limit (8,192 tokens) are scored, not only the 384-token training context.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.data import load_records
from kev.metrics import grouped_metrics
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT
from sklearn.metrics import roc_auc_score


class Precomputed:
    """Predictor over probabilities computed by another runtime, keyed by kev.data.load_records record id."""
    temperature = 1.0

    def __init__(self, path):
        self.p = {}
        for line in Path(path).read_text().splitlines():
            r = json.loads(line); self.p[r["id"]] = r["probabilities"]

    def __call__(self, record):
        return {"probabilities": self.p[record["_meta"]["id"]], "latency_ms": 0.0, "input_tokens": 0}


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
    a = ap.parse_args()
    if bool(a.run) == bool(a.preds): ap.error("give exactly one of --run or --preds")
    records = load_records(a.data, source="rad")
    predictor = Precomputed(a.preds) if a.preds else LocalPredictor(a.run, a.device, LoadOptions.from_env(), context=SERVING_CONTEXT)
    report, rows = evaluate_records(records, predictor, a.out, skip_overlong=True)
    s = summary(rows, report)
    s.update(run=a.run or a.preds, data=a.data, temperature=getattr(predictor, "temperature", None), latency_ms=report.get("latency_ms"))
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s["overall"]))


if __name__ == "__main__":
    main()
