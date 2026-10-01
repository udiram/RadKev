"""Latency study: Kev one-pass decisions vs the ways an LLM is used for the same decision. Same hardware for every row.

    python experiments/latency.py                    # 27B tier on one GPU pair (CUDA_VISIBLE_DEVICES=0,1)
    python experiments/latency.py --kev stock9=jaredpalmer/kev-9b,radkev9=v2x9-kev-9b --llms "" --out latency_9b.json

One A6000 NVLink pair, bf16, Hugging Face transformers for every model (Kev's own stack), batch 1, requests one at a time,
CUDA synchronised around each timing, the first 5 requests per mode excluded as warm-up. Sample: 150 held-out test records
(seed 0, all sources), every question on them. Modes:
  kev        stock Kev-27B and RadKev-27B (v2): one forward pass answers every question of a record (shared state prefix)
  letter     Qwen3.8-27B / MedGemma-27B scored as in the accuracy benchmark: one forward pass per question, option-letter logits
  direct     the same LLM prompt, greedy generation of the answer letter (max 8 new tokens), thinking off
  reasoning  Qwen3.8-27B with thinking on (max 2,048 new tokens), then the answer letter; 40-question subsample
Writes timings, token counts and accuracy on the sample only. The sample is drawn from $RADKEV_HOME/runs/test-final/test.jsonl
(written by experiments/final_test.py --tag final).
"""
import argparse
import json
import os
import random
import subprocess

from radkev.paths import KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS, resolve_run

WORK = RUNS / "latency"
N_RECORDS, N_REASONING, WARMUP = 150, 40, 5
KEV = {"stock27": "jaredpalmer/kev-27b", "radkev27": "v2mg-kev-27b"}
LLMS = {"qwen38": QWEN38_27B, "medgemma": MEDGEMMA_27B}

KEV_INNER = r'''
import json, sys, time, torch
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT
run, inp, out = sys.argv[1:4]
recs = [json.loads(l) for l in open(inp)]
t0 = time.time(); pred = LocalPredictor(run, "cuda", LoadOptions.from_env(), context=SERVING_CONTEXT); load = time.time() - t0
rows = []
for r in recs:
    torch.cuda.synchronize(); t = time.perf_counter()
    try: p = pred(r); ok = True
    except Exception: p, ok = None, False
    torch.cuda.synchronize(); wall = 1000 * (time.perf_counter() - t)
    correct = []
    if ok:
        for qid, q in r["questions"].items():
            d = p["probabilities"].get(qid) or {}
            if d: top = max(d, key=d.get); correct.append(str(top).lower() == str(q["label"]).lower())
    rows.append({"ok": ok, "ms": wall, "questions": len(r["questions"]), "correct": correct})
json.dump({"load_seconds": load, "rows": rows}, open(out, "w"))
'''

LLM_INNER = r'''
import json, re, sys, time, torch
from radkev import teacher as te
model_id, inp, out, reasoning_n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
recs = [json.loads(l) for l in open(inp)]
t0 = time.time(); S = te.LetterScorer(model_id); load = time.time() - t0
tok, model = S.tok, S.model
def label_key(q, keys):
    return str(q["label"]).lower()
def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); v = fn(); torch.cuda.synchronize(); return v, 1000 * (time.perf_counter() - t)
def letter_of(text, n):
    m = re.findall(r"\b([A-Z])\b", text.split("</think>")[-1])
    for L in m:
        i = ord(L) - 65
        if 0 <= i < n: return i
    return None
res = {"load_seconds": load, "letter": [], "direct": [], "reasoning": []}
qs = [(r, qid, q) for r in recs for qid, q in r["questions"].items()]
for r, qid, q in qs:
    keys, prompt = S.prompt(r["state"], q)
    want = str(q["label"]).lower(); key_s = [str(k).lower() for k in keys]
    enc = tok(prompt, return_tensors="pt").to(model.device)
    # letter scoring: the accuracy benchmark's method (one forward pass, option-letter logits)
    def score():
        with torch.no_grad():
            z = model(**enc, logits_to_keep=1).logits[0, -1].float()
        s = torch.stack([torch.logsumexp(z[ids], 0) for ids in S.letter_ids[:len(keys)]]); return int(s.argmax())
    i, ms = timed(score)
    res["letter"].append({"ms": ms, "prompt_tokens": int(enc["input_ids"].shape[1]), "correct": key_s[i] == want})
    # direct generation: answer letter, thinking off
    def gen(n):
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=n, do_sample=False)
        return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True), int(o.shape[1] - enc["input_ids"].shape[1])
    (text, n_new), ms = timed(lambda: gen(8))
    j = letter_of(text, len(keys))
    res["direct"].append({"ms": ms, "new_tokens": n_new, "parsed": j is not None, "correct": j is not None and key_s[j] == want})
# reasoning: thinking on (Qwen only), subsample
if reasoning_n:
    for r, qid, q in qs[:reasoning_n]:
        keys, _ = S.prompt(r["state"], q); key_s = [str(k).lower() for k in keys]; want = str(q["label"]).lower()
        ks, lines = te.render_options(q)
        user = f"{te.state_text(r['state'])}\n\nQuestion: {q['instructions'] if isinstance(q['instructions'], str) else json.dumps(q['instructions'])}\nOptions:\n" + "\n".join(lines) + "\n\nThink it through, then answer with one letter."
        prompt = tok.apply_chat_template([{"role": "system", "content": te.SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        def g():
            with torch.no_grad(): o = model.generate(**enc, max_new_tokens=2048, do_sample=False)
            return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True), int(o.shape[1] - enc["input_ids"].shape[1])
        (text, n_new), ms = timed(g)
        j = letter_of(text, len(keys))
        res["reasoning"].append({"ms": ms, "new_tokens": n_new, "parsed": j is not None, "correct": j is not None and key_s[j] == want, "hit_cap": n_new >= 2048})
json.dump(res, open(out, "w"))
'''


def stats(v):
    v = sorted(v)
    if not v: return None
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return {"n": len(v), "median": q(.5), "p90": q(.9), "p95": q(.95), "mean": sum(v) / len(v)}


def main():
    global KEV, LLMS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kev", default="", help="name=run pairs (default: stock Kev-27B and RadKev-27B)")
    ap.add_argument("--llms", default=None, help="name=hf_id pairs (default: Qwen3.8-27B and MedGemma-27B-text; '' for none)")
    ap.add_argument("--out", default="latency_bench.json", help="file name under $RADKEV_HOME/runs/latency/")
    a = ap.parse_args()
    if a.kev: KEV = dict(x.split("=", 1) for x in a.kev.split(","))
    KEV = {k: resolve_run(v) for k, v in KEV.items()}
    if a.llms is not None: LLMS = dict(x.split("=", 1) for x in a.llms.split(",") if x)
    WORK.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in (RUNS / "test-final/test.jsonl").read_text().splitlines() if l.strip()]
    sample = random.Random(0).sample(recs, N_RECORDS)
    for i, r in enumerate(sample): r["_meta"]["id"] = f"lat{i}"
    inp = WORK / "sample.jsonl"; inp.write_text("".join(json.dumps(r) + "\n" for r in sample))
    env = {**os.environ, "KEV_DEVICE_MAP": "auto", "KEV_DTYPE": "bf16", "KEV_MAX_MEMORY": "0:26GiB,1:40GiB",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    rep = {"hardware": "2x RTX A6000 48 GB (one NVLink pair), bf16, Hugging Face transformers, batch 1, sequential",
           "sample": {"records": len(sample), "questions": sum(len(r["questions"]) for r in sample), "seed": 0}, "warmup_excluded": WARMUP, "models": {}}
    for name, run in KEV.items():
        out = WORK / f"{name}.json"
        kenv = env if "27b" in run.lower() else {**env, "KEV_DEVICE_MAP": "", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]}
        p = subprocess.run([str(KEV_PY), "-c", KEV_INNER, run, str(inp), str(out)], env=kenv, capture_output=True, text=True)
        if p.returncode: rep["models"][name] = {"error": p.stderr[-600:]}; continue
        d = json.loads(out.read_text()); rows = [r for r in d["rows"] if r["ok"]][WARMUP:]
        cor = [c for r in rows for c in r["correct"]]
        rep["models"][name] = {"mode": "kev one pass per record", "load_seconds": round(d["load_seconds"], 1),
                               "per_record_ms": stats([r["ms"] for r in rows]), "per_question_ms": stats([r["ms"] / r["questions"] for r in rows]),
                               "accuracy": sum(cor) / len(cor) if cor else None, "failures": sum(not r["ok"] for r in d["rows"])}
    for name, model_id in LLMS.items():
        out = WORK / f"{name}.json"
        p = subprocess.run([str(KEV_PY), "-c", LLM_INNER, model_id, str(inp), str(out), str(N_REASONING if name == "qwen38" else 0)],
                           env={**env, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, capture_output=True, text=True)
        if p.returncode: rep["models"][name] = {"error": "".join(l for l in p.stderr.splitlines(True)[-10:] if "hf_" not in l)[-600:]}; continue
        d = json.loads(out.read_text()); m = {"load_seconds": round(d["load_seconds"], 1)}
        for mode in ("letter", "direct", "reasoning"):
            rows = d[mode][WARMUP:] if mode != "reasoning" else d[mode][1:]
            if not rows: continue
            m[mode] = {"per_question_ms": stats([r["ms"] for r in rows]), "accuracy": sum(r["correct"] for r in rows) / len(rows),
                       **({"new_tokens": stats([r["new_tokens"] for r in rows]), "parsed_share": sum(r["parsed"] for r in rows) / len(rows)} if mode != "letter" else
                          {"prompt_tokens": stats([r["prompt_tokens"] for r in rows])}),
                       **({"hit_token_cap": sum(r["hit_cap"] for r in rows)} if mode == "reasoning" else {})}
        rep["models"][name] = m
    (WORK / a.out).write_text(json.dumps(rep, indent=1))
    print(WORK / a.out)


if __name__ == "__main__":
    main()
