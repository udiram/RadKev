"""Latency vs number of questions per case: Kev answers every question of a state in one pass; an LLM needs one call per
question (or a padded batch). Same hardware for every row. Latency only (the questions have no labels).

    python experiments/latency_scaling.py           # one GPU pair (CUDA_VISIBLE_DEVICES=0,1)

30 held-out IU chest X-ray reports (seed 1). For each report, K = 1, 2, 4, 8, 14 finding questions ("Does this report
describe {f}?", the 14 IU findings in a fixed shuffled order). One A6000 NVLink pair, bf16, Hugging Face transformers,
batch 1 unless stated, CUDA synchronised, the first 3 reports per mode excluded as warm-up. Modes:
  radkev27        RadKev-27B (v2), one forward pass per report for all K questions
  qwen38 letter   Qwen3.8-27B option-letter scoring, one forward pass per question, sequential
  qwen38 batched  the same K prompts in one padded batch (the fastest way to ask an LLM K questions at once)
  qwen38 direct   generate the answer letter per question, thinking off
  medgemma letter MedGemma-27B option-letter scoring, sequential
Writes timings only, to $RADKEV_HOME/runs/latency_scaling/latency_scaling.json.
"""
import json
import os
import random
import subprocess

from radkev import data as bd
from radkev.paths import KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS, resolve_run

WORK = RUNS / "latency_scaling"
KS, N_REPORTS, WARMUP = [1, 2, 4, 8, 14], 30, 3

INNER = r'''
import json, sys, time, torch
mode, model, inp, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cases = [json.loads(l) for l in open(inp)]
def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t)
res = []
if mode == "kev":
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT
    pred = LocalPredictor(model, "cuda", LoadOptions.from_env(), context=SERVING_CONTEXT)
    for c in cases: res.append({"k": c["_meta"]["k"], "ms": timed(lambda: pred(c))})
else:
    from radkev import teacher as te
    S = te.LetterScorer(model); tok, lm = S.tok, S.model
    for c in cases:
        qs = list(c["questions"].values())
        if mode == "batched":
            items = [(i, c["state"], q) for i, q in enumerate(qs)]
            res.append({"k": c["_meta"]["k"], "ms": timed(lambda: list(S.score(items, batch=len(items))))})
            continue
        total = 0.0
        for q in qs:
            keys, prompt = S.prompt(c["state"], q)
            enc = tok(prompt, return_tensors="pt").to(lm.device)
            if mode == "letter":
                def f():
                    with torch.no_grad(): lm(**enc, logits_to_keep=1)
            else:
                def f():
                    with torch.no_grad(): lm.generate(**enc, max_new_tokens=8, do_sample=False)
            total += timed(f)
        res.append({"k": c["_meta"]["k"], "ms": total})
json.dump(res, open(out, "w"))
'''


def stats(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(v), "median": q(.5), "p95": q(.95), "mean": sum(v) / len(v)} if v else None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in (RUNS / "test-final/test.jsonl").read_text().splitlines() if l.strip()]
    iu = [r for r in recs if r["_meta"].get("source") == "iu"]
    rng = random.Random(1); reports = rng.sample(iu, N_REPORTS); findings = list(bd.IU_VOCAB)
    cases = []
    for i, r in enumerate(reports):
        order = findings[:]; random.Random(i).shuffle(order)
        for k in KS:
            qs = {f"f{j}": {"type": "noul", "instructions": bd.PRESENT_T[0].format(f=f), "label": False, "src": "latency"} for j, f in enumerate(order[:k])}
            cases.append({"state": r["state"], "questions": qs, "_meta": {"id": f"s{i}k{k}", "source": "latency", "variant": "clean", "group_id": f"s{i}", "k": k}})
    inp = WORK / "cases.jsonl"; inp.write_text("".join(json.dumps(c) + "\n" for c in cases))
    env = {**os.environ, "KEV_DEVICE_MAP": "auto", "KEV_DTYPE": "bf16", "KEV_MAX_MEMORY": "0:26GiB,1:40GiB",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    runs = [("radkev27", "kev", resolve_run("v2mg-kev-27b")), ("qwen38 letter", "letter", QWEN38_27B),
            ("qwen38 batched", "batched", QWEN38_27B), ("qwen38 direct", "direct", QWEN38_27B), ("medgemma letter", "letter", MEDGEMMA_27B)]
    rep = {"hardware": "2x RTX A6000 48 GB (one NVLink pair), bf16, Hugging Face transformers", "reports": N_REPORTS, "ks": KS,
           "warmup_reports_excluded": WARMUP, "modes": {}}
    for name, mode, model in runs:
        out = WORK / f"{name.replace(' ', '_')}.json"
        p = subprocess.run([str(KEV_PY), "-c", INNER, mode, model, str(inp), str(out)], env=env, capture_output=True, text=True)
        if p.returncode: rep["modes"][name] = {"error": "".join(l for l in p.stderr.splitlines(True)[-8:] if "hf_" not in l)[-500:]}; continue
        rows = json.loads(out.read_text())
        by_k = {k: [r["ms"] for r in rows if r["k"] == k][WARMUP:] for k in KS}
        rep["modes"][name] = {str(k): stats(v) for k, v in by_k.items()}
    (WORK / "latency_scaling.json").write_text(json.dumps(rep, indent=1))
    print(WORK / "latency_scaling.json")


if __name__ == "__main__":
    main()
