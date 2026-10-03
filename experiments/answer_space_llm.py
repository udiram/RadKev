"""Answer-space study for the two LLMs (post hoc; no training): Qwen3.8-27B and MedGemma-27B-text scored on the answer spaces of
experiments/answer_space.py that have at most 16 options, plus their latency as the number of options grows.

    python experiments/answer_space_llm.py          # after experiments/answer_space.py; two GPU pairs in parallel, or one in turn

Inputs (written by experiments/answer_space.py, read only): runs/answer_space/records.jsonl (one record per question, one
question per answer space) and latency_records.jsonl (one question per request, key + random pool distractors).
Scoring is the main evaluation's LLM protocol (radkev.teacher LetterScorer: lettered options, softmax over the next-token letter
logits, reasoning off; MedGemma with the one-line thought pre-fill and a single <bos>), so only answer spaces with at most 16
options (A-P) are scored. Latency: batch 1, CUDA-synchronized, the first 10 requests excluded, 2-16 options.
Writes $RADKEV_HOME/runs/answer_space_llm/answer_space_llm.json: per-question correctness and confidence per model and
condition, keyed by question id (no text), and latencies.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from radkev.paths import KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS

AS = RUNS / "answer_space"; WORK = RUNS / "answer_space_llm"
# the job that produced the paper's rows tokenized both prompts without special tokens; Qwen's tokenizer adds none anyway
MODELS = {"qwen38": (QWEN38_27B, ["--single_bos"]), "medgemma_fix": (MEDGEMMA_27B, ["--think_off", "--single_bos"])}
MAXK = 16
res = {"phases": {}, "maxk": MAXK}
LOCK = threading.Lock()
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"}

LAT = r'''
import json, sys, time, torch
from radkev import teacher
model, think_off, inp, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4]
S = teacher.LetterScorer(model, think_off, single_bos=True)
rows = [json.loads(l) for l in open(inp) if l.strip()]
res = []
for i, r in enumerate(rows):
    (qid, q), = r["questions"].items()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    list(S.score([(qid, r["state"], q)], batch=1))
    torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t0)
    if i >= 10: res.append([r["_meta"]["K"], ms])
json.dump(res, open(out, "w"))
'''


def save():
    with LOCK: (WORK / "answer_space_llm.json").write_text(json.dumps(res))


def run(cmd, env, log):
    with open(log, "a") as f:
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError("".join(Path(log).read_text().splitlines(True)[-15:])[-1500:])


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    # answer spaces with at most MAXK options; keep the record ids and the labels
    recs = [json.loads(l) for l in open(AS / "records.jsonl") if l.strip()]
    keep, labels = [], {}
    for r in recs:
        qs = {c: q for c, q in r["questions"].items() if len(q.get("criteria", {})) <= MAXK}
        rid = r["_meta"]["rid"]
        for c, q in qs.items(): labels[(rid, c)] = q["label"]
        keep.append({"state": r["state"], "questions": qs, "_meta": r["_meta"]})
    with open(WORK / "records_llm.jsonl", "w") as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat = [json.loads(l) for l in open(AS / "latency_records.jsonl") if l.strip()]
    lat = [r for r in lat if r["_meta"]["K"] <= MAXK]
    with open(WORK / "latency_llm.jsonl", "w") as f:
        for r in lat: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res["n_records"] = len(keep); res["n_questions"] = sum(len(r["questions"]) for r in keep); res["n_latency"] = len(lat); save()

    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    pairs = [",".join(devs[i:i + 2]) for i in range(0, len(devs) - 1, 2)][:2] or [",".join(devs)]
    lane = {"qwen38": pairs[0], "medgemma_fix": pairs[-1]}

    def one(name):
        mid, flags = MODELS[name]; env = {**ENV, "CUDA_VISIBLE_DEVICES": lane[name]}; log = WORK / f"{name}.log"; t0 = time.time()
        try:
            preds = WORK / f"{name}.preds.jsonl"
            run([KEV_PY, "-m", "radkev.teacher", "predict", "--model", mid, "--data", WORK / "records_llm.jsonl", "--out", preds, "--batch", "16", *flags], env, log)
            correct, conf = {}, {}
            for l in open(preds):   # radkev.teacher predict ids are rad/<line of records_llm.jsonl>
                p = json.loads(l); line = int(p["id"].split("/")[1]); rid = keep[line]["_meta"]["rid"]
                for c, dist in p["probabilities"].items():
                    top = max(dist, key=dist.get)
                    correct.setdefault(c, {})[rid] = int(top == labels[(rid, c)]); conf.setdefault(c, {})[rid] = round(dist[top], 4)
            lat_out = WORK / f"lat_{name}.json"
            run([KEV_PY, "-c", LAT, mid, "1" if "--think_off" in flags else "0", WORK / "latency_llm.jsonl", lat_out], env, log)
            with LOCK:
                res[name] = {"correct": correct, "conf": conf, "latency": json.loads(lat_out.read_text())}
                res["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
        except Exception as e:
            with LOCK: res["phases"][name] = {"state": "failed", "error": str(e)[-1500:]}
        save()

    if len(pairs) > 1:   # two GPU pairs: both models at once
        ts = [threading.Thread(target=one, args=(m,)) for m in MODELS]
        for t in ts: t.start()
        for t in ts: t.join()
    else:
        for m in MODELS: one(m)
    save()
    print(WORK / "answer_space_llm.json")


if __name__ == "__main__":
    main()
