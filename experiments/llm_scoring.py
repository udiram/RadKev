"""How the LLM baselines are read: a check that each LLM's option-letter scores are its answer, and the corrected rows.

    python experiments/llm_scoring.py               # one GPU pair; needs $RADKEV_HOME/runs/test-final/test.jsonl

The pre-registered LLM rows take option-letter logits at the first generated token (radkev.teacher predict --mode first).
MedGemma-27B-text scored 42.7% on MedQA that way against a published ~87%: it opens every answer with a thought channel
(<unused94>thought ... <unused95>) that enable_thinking=False does not suppress, so its first-token letters score the start
of a thought, not an answer. This script reproduces the diagnosis and the fix (all post hoc, none of it changes RadKev):

1. MedQA-300 sanity check (the first 300 MedQA test records): MedGemma first-token vs generated-letter accuracy, with and
   without --think_off (LetterScorer.THOUGHT_OFF pre-fills a one-line thought and closes the channel; an EMPTY thought does
   not work, the model keeps reasoning). The one-line thought was chosen on 5 items by letter-probability mass, not accuracy.
2. MedGemma with thinking on, 100 MedQA items (up to 2,048 new tokens): accuracy and seconds per question.
3. Full-test rows in runs/test-final/: qwen38_gen and medgemma_gen (letter read where generated), and medgemma_brief
   (first-token scoring with the thought channel closed). The original rows are kept. The MedGemma row the paper reports
   (medgemma_fix) is this protocol with a single <bos>: experiments/medgemma_rescore.py.
Writes accuracies and summaries only, to $RADKEV_HOME/runs/test-final/llm_scoring.json.
"""
import json
import os
import subprocess
import time

from radkev.paths import KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS

WORK = RUNS / "test-final"; TEST = WORK / "test.jsonl"; SUB = RUNS / "llm_scoring"
MODELS = {"medgemma": MEDGEMMA_27B, "qwen38": QWEN38_27B}
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
rep = {}


def save():
    (WORK / "llm_scoring.json").write_text(json.dumps(rep, indent=1))


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True)
    if p.returncode: raise RuntimeError((p.stderr or p.stdout)[-1500:])
    for l in p.stderr.splitlines()[::-1]:
        if l.startswith("{") and "letter_found_share" in l: return json.loads(l)
    return {}


def summary(path):
    s = json.loads((path / "summary.json").read_text())
    return {"overall": {k: s["overall"].get(k) for k in ("n", "acc", "ece", "brier")}, "tasks": {t: {"n": v["n"], "acc": v["acc"]} for t, v in s["tasks"].items()}}


def score(model, data, out, *flags):
    """radkev.teacher predict -> radkev.evaluate --preds; returns the predictor's letter_found_share line, if any."""
    preds = out.parent / f"{out.name}.preds.jsonl"
    info = run([KEV_PY, "-m", "radkev.teacher", "predict", "--model", model, "--data", data, "--out", preds, *flags])
    run([KEV_PY, "-m", "radkev.evaluate", "--preds", preds, "--data", data, "--out", out])
    return info


THINK_INNER = r'''
import json, re, sys, time, torch
from radkev import teacher as te
recs = [json.loads(l) for l in open(sys.argv[1])][:int(sys.argv[3])]
S = te.LetterScorer(sys.argv[2]); tok, m = S.tok, S.model
items = [(r, q) for r in recs for q in r["questions"].values()]
rows = []
for i in range(0, len(items), 8):
    chunk = items[i:i + 8]; ps = [S.prompt(r["state"], q) for r, q in chunk]
    enc = tok([p for _, p in ps], return_tensors="pt", padding=True).to(m.device)
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad(): o = m.generate(**enc, max_new_tokens=2048, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t) / len(chunk)
    for (r, q), (keys, _), g in zip(chunk, ps, o[:, enc["input_ids"].shape[1]:]):
        text = tok.decode(g, skip_special_tokens=False); closed = "<unused95>" in text
        ans = text.split("<unused95>")[-1] if closed else ""
        L = next((x for x in re.findall(r"\b([A-Z])\b", ans) if ord(x) - 65 < len(keys)), None)
        n_new = int((g != tok.pad_token_id).sum())
        rows.append({"ms_batched": ms, "new_tokens": n_new, "closed": closed, "parsed": L is not None,
                     "correct": L is not None and str(keys[ord(L) - 65]).lower() == str(q["label"]).lower()})
n = len(rows); tok_s = sorted(r["new_tokens"] for r in rows)
print(json.dumps({"n": n, "acc": sum(r["correct"] for r in rows) / n, "parsed_share": sum(r["parsed"] for r in rows) / n,
                  "thought_closed_share": sum(r["closed"] for r in rows) / n, "median_new_tokens": tok_s[n // 2],
                  "seconds_per_question_batch8": sum(r["ms_batched"] for r in rows) / n / 1000}))
'''


def main():
    SUB.mkdir(parents=True, exist_ok=True)
    # 1. sanity check on the first 300 MedQA test records
    medqa = SUB / "medqa300.jsonl"
    lines = [l for l in TEST.read_text().splitlines() if l.strip() and json.loads(l)["_meta"].get("source") == "medqa"][:300]
    medqa.write_text("\n".join(lines) + "\n")
    rep["sanity_medqa300"] = {}
    for think_off in (False, True):
        for mode in ("first", "gen"):
            key = f"{mode}{'_think_off' if think_off else ''}"
            out = SUB / f"medgemma_{key}"
            info = score(MEDGEMMA_27B, medqa, out, "--mode", mode, *(["--think_off"] if think_off else []))
            rep["sanity_medqa300"][key] = {"acc": summary(out)["overall"]["acc"], **info}; save()
    # 2. thinking on, 100 MedQA items
    p = subprocess.run([str(KEV_PY), "-c", THINK_INNER, str(medqa), MEDGEMMA_27B, "100"], env=ENV, capture_output=True, text=True)
    rep["medgemma_thinking_medqa100"] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-800:]}; save()
    # 3. full-test rows
    for name, model, flags in (("qwen38_gen", QWEN38_27B, ["--mode", "gen"]), ("medgemma_gen", MEDGEMMA_27B, ["--mode", "gen"]),
                               ("medgemma_brief", MEDGEMMA_27B, ["--think_off"])):
        out, info, t0 = WORK / name, {}, time.time()
        if not (out / "summary.json").exists(): info = score(model, TEST, out, *flags)
        rep[name] = {**summary(out), **info, "hours": round((time.time() - t0) / 3600, 2)}; save()
    print(WORK / "llm_scoring.json")


if __name__ == "__main__":
    main()
