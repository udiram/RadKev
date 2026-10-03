"""MedGemma-27B-text rescored with a single <bos> (post hoc correction): the MedGemma row the paper reports (medgemma_fix).

    CUDA_VISIBLE_DEVICES=0,1 python experiments/medgemma_rescore.py   # needs $RADKEV_HOME/runs/test-final/test.jsonl

radkev.teacher.LetterScorer passes chat-template text that already starts with <bos> to a tokenizer that adds another by default,
so Gemma models saw two <bos> tokens on every prompt (Qwen's tokenizer adds none, so its rows are unaffected). This script scores
the full held-out test split with the medgemma_brief protocol of experiments/llm_scoring.py (first-token option-letter
probabilities with the one-line thought pre-fill, --think_off) and the prompt tokenized without special tokens (--single_bos),
then radkev.evaluate --preds, into runs/test-final/medgemma_fix/ so that experiments/robustness.py, robustness2.py and
llm_reasoning.py pair it with every other row. The earlier rows (medgemma, medgemma_gen, medgemma_brief) are kept.
The options-only control for this row is part of experiments/options_only.py.
Writes the summary (accuracy, calibration per task) to $RADKEV_HOME/runs/test-final/medgemma_rescore.json; scored rows stay in
runs/test-final/medgemma_fix/.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from radkev.paths import KEV_PY, MEDGEMMA_27B, RUNS

WORK = RUNS / "test-final"; TEST = WORK / "test.jsonl"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"}


def run(cmd, log):
    with open(log, "a") as f:
        rc = subprocess.run([str(c) for c in cmd], env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError("".join(Path(log).read_text().splitlines(True)[-15:])[-1500:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="medgemma_fix", help="row name under runs/test-final/")
    ap.add_argument("--batch", default="16")
    a = ap.parse_args()
    out, preds, log = WORK / a.name, WORK / f"{a.name}.preds.jsonl", WORK / f"{a.name}.log"
    rep, t0 = {"protocol": "first-token option letters, --think_off, --single_bos", "model": MEDGEMMA_27B}, time.time()
    if not (out / "summary.json").exists():
        run([KEV_PY, "-m", "radkev.teacher", "predict", "--model", MEDGEMMA_27B, "--data", TEST, "--out", preds, "--think_off", "--single_bos",
             "--batch", a.batch], log)
        run([KEV_PY, "-m", "radkev.evaluate", "--preds", preds, "--data", TEST, "--out", out], log)
    s = json.loads((out / "summary.json").read_text())
    rep.update(overall=s["overall"], tasks={t: {"n": v["n"], "acc": v["acc"]} for t, v in s["tasks"].items()}, minutes=round((time.time() - t0) / 60, 1))
    (WORK / "medgemma_rescore.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["overall"]))


if __name__ == "__main__":
    main()
