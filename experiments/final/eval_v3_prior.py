"""RadCases panel prior correction as the default for RadKev (user decision 2026-10-06). No GPU, no rescoring.

    python jobs/submit.py jobs/eval_v3_prior.py --gpus 0 --cpus 4 --mem 16 --timeout 60 --outputs 'eval_v3_prior*.json'

Reruns, from the rows already scored on the node, (1) eval_v3's analysis with the OpenAI Decisions system added exactly as in
openai_decisions.analyse (systems, pairs, calibration pairs; CT-RATE dropped for openai_dec), and (2) the reasoning-sample paired
analysis of llm_reasoning (v3), with one change: on the RadCases panel question, the answer distributions of RadKev-27B (v3_27)
and RadKev-9B (v3_9) are prior-corrected as in jobs/radcases_prior.py: each option's probability is divided by that option's
frequency among the RadCases training-split panel answers (add-one smoothed over the 12 options) and renormalized. Only training
labels enter the correction; every other system and task is unchanged. Same seeds and resamples as the original analyses.
Publishes aggregates only: eval_v3_prior.json, eval_v3_prior_reasoning.json, eval_v3_prior_status.json.
"""
import ast
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

INCLUDE = ["jobs/eval_v3.py", "jobs/llm_reasoning.py", "jobs/openai_decisions.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
EV = RADKEV / "runs/test-v3"; FINAL = RADKEV / "runs/test-final"; RS = RADKEV / "reasoning_v3"
NONE = "None: no ACR Appropriateness Criteria topic applies"
status = {"started": time.strftime("%Y%m%d-%H%M%S")}


def save(): (OUT / "eval_v3_prior_status.json").write_text(json.dumps(status, indent=1, default=str))


def prior_and_options():
    cnt, opts = Counter(), set()
    for line in open(RADKEV / "data/radcases-v3/train.jsonl"):
        if not line.strip(): continue
        q = json.loads(line)["questions"].get("panel")
        if q: cnt[q["criteria"][q["label"]]] += 1; opts |= set(q["criteria"].values())
    assert len(opts) == 12 and NONE in opts
    n = sum(cnt.values()); prior = {o: (cnt[o] + 1) / (n + len(opts)) for o in opts}
    crit = {}
    for line in open(EV / "new.jsonl"):
        if not line.strip(): continue
        r = json.loads(line); q = r["questions"].get("panel")
        if q: crit[r["_meta"]["id"]] = [list(q["criteria"].values()), list(q["criteria"]).index(q["label"])]
    assert len(crit) == 132, len(crit)
    status["prior"] = {"train_panel_questions": n, "none_share": cnt[NONE] / n, "prior": prior}
    return prior, crit


def patch(prior, crit):
    """Inserted right after the systems' rows are loaded into D; corrects v3_27/v3_9 on radcases_panel in place."""
    return ("\n_PRIOR = json.loads(%r); _CRIT = json.loads(%r)\n" % (json.dumps(prior), json.dumps(crit)) + '''
for _m in ("v3_27", "v3_9"):
    if _m not in D: continue
    _n = 0
    for _k, (_bt, _lab, _p) in list(D[_m].items()):
        if _bt != "radcases_panel": continue
        _t, _li = _CRIT[_k[0]]
        assert _lab == _li and len(_t) == len(_p), ("option order", _m, _k, _lab, _li, len(_t), len(_p))
        _q = np.clip(np.asarray(_p, float), 1e-12, 1); _q = _q / _q.sum() / np.array([_PRIOR[x] for x in _t])
        D[_m][_k] = (_bt, _lab, _q / _q.sum()); _n += 1
    assert _n == 132, (_m, _n)
''')


def const(src, name):
    return next(n.value.value for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)


def run(code, args, log):
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{args[0]}", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    with open(log, "a") as f:
        rc = subprocess.run([str(KEV_PY), "-c", code, *map(str, args)], env=env, cwd=KEV_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"exit {rc}: " + Path(log).read_text()[-2000:])


def main():
    prior, crit = prior_and_options(); P = patch(prior, crit); save()
    # (1) eval_v3 analysis + OpenAI (as openai_decisions.analyse) + correction
    A = const(BUNDLE["jobs/eval_v3.py"], "ANALYSE")
    for old, new in (('"qwen38", "medgemma_fix"]\nD = ', '"qwen38", "medgemma_fix", "openai_dec"]\nD = '),
                     ("PAIRS = [", 'PAIRS = [("v3_27", "openai_dec"), ("v3_9", "openai_dec"), ("openai_dec", "stock27"), ("openai_dec", "qwen38"), ("openai_dec", "medgemma_fix"), '),
                     ('res["calibration_pairs"] = {}\nfor a_, b_ in (("v3_27", "stock27"), ("v3_9", "stock9")):',
                      'res["calibration_pairs"] = {}\nfor a_, b_ in (("v3_27", "stock27"), ("v3_9", "stock9"), ("v3_27", "openai_dec"), ("v3_9", "openai_dec")):')):
        assert A.count(old) == 1, old[:60]; A = A.replace(old, new, 1)
    anchor = "D = {s: load(s) for s in SYSTEMS}; D = {s: v for s, v in D.items() if v}\n"
    assert A.count(anchor) == 1; A = A.replace(anchor, anchor + P, 1)
    run(A, [EV / "code", EV, FINAL, OUT / "eval_v3_prior.json"], OUT / "eval_v3_prior_analyse.log")
    r = json.loads((OUT / "eval_v3_prior.json").read_text())   # openai_decisions.strip_unsent: CT-RATE never sent to the API
    for t in list(r["systems"].get("openai_dec", {}).get("tasks", {})):
        if t.startswith("ctrate:"): del r["systems"]["openai_dec"]["tasks"][t]
    for k, o in r["pairs"].items():
        if "openai_dec" in k.split("-"):
            for t in list(o.get("tasks", {})):
                if t.startswith("ctrate:"): del o["tasks"][t]
    r["prior_correction"] = status["prior"]; (OUT / "eval_v3_prior.json").write_text(json.dumps(r, indent=1))
    status["eval"] = {s: r["systems"][s]["bench"]["task_mean"] for s in ("v3_27", "v3_9", "stock27", "stock9", "qwen38", "openai_dec") if s in r["systems"]}; save()
    # (2) reasoning-sample paired analysis + correction
    R = const(BUNDLE["jobs/llm_reasoning.py"], "ANALYSE_V3")
    anchor = "D = {m: v for m, v in D.items() if v}\n"
    assert R.count(anchor) == 1; R = R.replace(anchor, anchor + P, 1)
    run(R, [RS / "code", RADKEV / "reasoning/runs", RS / "runs", EV, FINAL, OUT / "eval_v3_prior_reasoning.json"], OUT / "eval_v3_prior_analyse.log")
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    try: main()
    except Exception as e: status["error"] = str(e)[-3000:]; save(); raise
