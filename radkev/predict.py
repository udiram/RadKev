"""Ask a RadKev (or any Kev) checkpoint for decisions in-process, without a server.

    python -m radkev.predict --run jaredpalmer/kev-27b --request examples/cxr_report.json
    python -m radkev.predict --run $RADKEV_HOME/runs/v2mg-kev-27b/checkpoint --request requests.jsonl --out answers.jsonl

A request is the System One body Kev serves (POST /v1/systemone): a `state` (text, or a dict of named sections) and typed
`questions` (choice / noul / score). Each answer carries the full probability distribution over the allowed options. For an
HTTP endpoint, use Kev's server on the same checkpoint: python -m kev.serve --run <checkpoint> --port 8009.
Kev-27B-sized checkpoints on two 48 GB GPUs: KEV_DEVICE_MAP=auto KEV_DTYPE=bf16 (needs patches/kev_multigpu.patch).
"""
import argparse
import json
import sys
from pathlib import Path


def placeholder_label(q):
    """Kev's record loader needs a label on every question; any valid one will do because only probabilities are read."""
    if q["type"] == "choice": return next(iter(q["criteria"]))
    return False if q["type"] == "noul" else 0


def to_record(request, rid="q0"):
    qs = {qid: {**q, "label": q.get("label", placeholder_label(q)), "src": q.get("src", "predict")} for qid, q in request["questions"].items()}
    return {"state": request["state"], "questions": qs, "_meta": {"id": rid, "source": "predict", "variant": "clean", "group_id": rid}}


def answer(q, probs):
    """Kev probabilities for one question -> a System One style answer."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"], "probabilities": probs}
    top = max(probs, key=probs.get)
    if q["type"] == "choice":
        return {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs}
    return {"type": "score", "score": sum(int(k) * p for k, p in probs.items()), "level": int(top), "confidence": probs[top],
            "legend": dict(enumerate(q["criteria"])), "probabilities": probs}


class Predictor:
    """Load once, ask many times: Predictor(run)(request) -> {"answers": {qid: answer}, "latency_ms": float}."""

    def __init__(self, run, device=None):
        import torch
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor
        from kev.suite import SERVING_CONTEXT
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.kev = LocalPredictor(run, device, LoadOptions.from_env(), context=SERVING_CONTEXT)
        self.temperature = self.kev.temperature

    def __call__(self, request, rid="q0"):
        out = self.kev(to_record(request, rid))
        return {"answers": {qid: answer(q, out["probabilities"][qid]) for qid, q in request["questions"].items()},
                "latency_ms": round(out["latency_ms"], 1), "input_tokens": out["input_tokens"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="a Kev checkpoint: Hub id or local directory")
    ap.add_argument("--request", required=True, help="a .json request, or .jsonl with one request per line")
    ap.add_argument("--out", default="", help="write answers as JSONL here (default: print)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    path = Path(a.request)
    requests = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.suffix == ".jsonl" else [json.loads(path.read_text())]
    predictor = Predictor(a.run, a.device)
    out = open(a.out, "w") if a.out else sys.stdout
    for i, r in enumerate(requests):
        out.write(json.dumps(predictor(r, f"q{i}"), indent=None if a.out else 2) + "\n")


if __name__ == "__main__":
    main()
