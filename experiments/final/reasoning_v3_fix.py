"""Reasoning-sample analysis, corrected (post hoc; no GPU, no rescoring; 2026-10-07).

    python jobs/submit.py jobs/reasoning_v3_fix.py --gpus 0 --timeout 60 --outputs 'eval_v3_prior_reasoning*.json'

Reruns step (2) of jobs/eval_v3_prior.py, the paired analysis of the reasoning sample (ANALYSE_V3 of jobs/llm_reasoning.py) with
the RadCases panel prior correction of RadKev, from the rows already scored on the node, with two changes in ANALYSE_V3:
(a) bootstrap records are resampled within their source, each record in one stratum (the strata were tasks, so IU and Eurorad
records with questions in several tasks were drawn once per task and their counts summed), and (b) an additional task mean in
which the six examination tasks, which the sample leaves with 1 to 69 questions each, are pooled into one task (task_mean_k).
Publishes aggregates only: eval_v3_prior_reasoning.json, eval_v3_prior_reasoning_status.json.
"""
import json
import os
import time
from pathlib import Path

INCLUDE = ["jobs/eval_v3_prior.py", "jobs/llm_reasoning.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
status = {"started": time.strftime("%Y%m%d-%H%M%S")}


def save(): (OUT / "eval_v3_prior_reasoning_status.json").write_text(json.dumps(status, indent=1, default=str))


def main():
    ns = {"__name__": "eval_v3_prior"}; exec(BUNDLE["jobs/eval_v3_prior.py"], ns)   # prior_and_options, patch, const, run, paths
    prior, crit = ns["prior_and_options"](); P = ns["patch"](prior, crit)
    R = ns["const"](BUNDLE["jobs/llm_reasoning.py"], "ANALYSE_V3")
    anchor = "D = {m: v for m, v in D.items() if v}\n"
    assert R.count(anchor) == 1; R = R.replace(anchor, anchor + P, 1)
    assert "SRC_OF" in R and "task_mean_k" in R, "ANALYSE_V3 without the fix"
    RADKEV, RS = ns["RADKEV"], ns["RS"]
    ns["run"](R, [RS / "code", RADKEV / "reasoning/runs", RS / "runs", ns["EV"], ns["FINAL"], OUT / "eval_v3_prior_reasoning.json"], OUT / "analyse.log")
    r = json.loads((OUT / "eval_v3_prior_reasoning.json").read_text())
    status["task_mean_k"] = {m: v["bench"]["task_mean_k"] for m, v in r["systems"].items()}
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    try: main()
    except Exception as e: status["error"] = str(e)[-3000:]; save(); raise
