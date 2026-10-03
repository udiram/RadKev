"""LLM baselines with reasoning enabled (post hoc; no training): Qwen3.8-27B and MedGemma-27B-text-it scored on a fixed,
stratified sample of the held-out human-labeled test questions and paired with every existing row on the same questions.

    CUDA_VISIBLE_DEVICES=0,1 python experiments/llm_reasoning.py      # needs runs/test-final/ and a vLLM environment ($VLLM_PYTHON)

Sample (fixed before any reasoning output exists): every human-labeled task of runs/test-final/test.jsonl (chest radiograph
reports, case diagnosis, routing, radiology and medical knowledge; never teacher-labeled questions), at most CFG["cap"]
questions per task (CFG["caps"] for the radiology tasks: every Eurorad and MedMCQA radiology question, 200 per IU task), chosen
in sha256("reason-sub|<record>|<qid>") order. Records keep their test ids and cluster ids, so every row pairs with the full-test
rows of the other models (1,800 questions in 1,569 records in the paper).
Prompt: radkev.teacher's system and user text ("... Answer with one letter."); only the chat template changes:
  qwen38_think    enable_thinking=True
  medgemma_think  the thought channel is opened in the prompt ("<unused94>thought\\n"), so the model always reasons
Decoding: vLLM (0.19 in the paper) in its own environment, bf16, tensor parallel over one GPU pair, each model's own
generation_config sampling (temperature/top_p/top_k; greedy if it declares none), seed 0, at most CFG["think_cap"] new tokens;
the two models run one after the other on the same pair.
Answer read-out: the reasoning is closed where the model closed it (or force-closed at the cap), "\\n\\nAnswer:" is appended,
and the option-letter distribution is the softmax over the next-token log-probabilities of the option letters (top-20
log-probabilities; "A" and " A" pooled). The answer the model wrote after its reasoning is parsed as well; when it names a
different option than the read-out, it decides: the distribution is set to 0.98 on the written option and the rest spread
evenly (this override is counted in the per-question metadata).
Scored with radkev.evaluate --preds (Kev's metrics), then paired with radkev.compare's cluster bootstrap (2,000 resamples)
against the reasoning-off rows of the same LLMs, RadKev-27B/9B and Kev-27B/9B on the same questions.
Writes aggregates only: $RADKEV_HOME/runs/llm_reasoning/llm_reasoning.json (sample counts, generation statistics, paired
comparisons); per-question probabilities and token counts stay under runs/llm_reasoning/.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from radkev.paths import HOME, KEV_PY, MEDGEMMA_27B, QWEN38_27B, RUNS, VLLM_PY

ROOT = Path(__file__).resolve().parents[1]
TESTDIR = RUNS / "test-final"; TEST = TESTDIR / "test.jsonl"
WORK = RUNS / "llm_reasoning"; SCORED = WORK / "runs"
CFG = {"cap": 60, "caps": {"eurorad_dx": 1000, "eurorad_route": 1000, "medmcqa_rad": 1000, "iu_finding": 200, "iu_normal": 200, "iu_which": 200},
       "think_cap": 4096, "max_model_len": 16384, "max_prompt": 11000, "chunk": 800, "samples": 2000}
LLMS = {"qwen38_think": QWEN38_27B, "medgemma_think": MEDGEMMA_27B}
BASE = ["v2_27", "stock27", "r9", "stock9", "qwen38", "qwen38_gen", "medgemma_brief", "medgemma_fix"]
HUMAN_PREFIX = ("iu_", "eurorad_dx", "eurorad_route", "medmcqa_rad", "medmcqa_med", "medqa", "mmlu_", "medxpertqa", "pubmedqa")
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "cfg": CFG, "phases": {}}
result = {"what": "reasoning-on Qwen3.8-27B and MedGemma-27B on a stratified human-label test sample; aggregates only", "cfg": CFG}
LOCK = threading.Lock()


def save():
    with LOCK:
        (WORK / "llm_reasoning_status.json").write_text(json.dumps(status, indent=1))
        (WORK / "llm_reasoning.json").write_text(json.dumps(result, indent=1))


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as e: status["phases"][name] = {"state": "failed", "error": str(e)[-2500:], "minutes": round((time.time() - t0) / 60, 1)}
    save(); return status["phases"][name]["state"] == "ok"


def run(cmd, env, log, cwd=None):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))[:300]}\n"); f.flush()
        pr = subprocess.Popen([str(c) for c in cmd], env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        rc = pr.wait()
        try: os.killpg(pr.pid, signal.SIGKILL)   # vLLM tensor-parallel workers can outlive a parent that exits with os._exit
        except ProcessLookupError: pass
        time.sleep(5)
    if rc: raise RuntimeError(f"exit {rc}: " + "".join(Path(log).read_text().splitlines(True)[-25:])[-2400:])


# ---------------------------------------------------------------- 1. the sample (CPU, stdlib)
def sample():
    per = {}
    lines = [l for l in open(TEST)]
    for n, line in enumerate(lines):
        if not line.strip(): continue
        r = json.loads(line)
        for qid, q in r["questions"].items():
            t = str(q.get("src", ""))
            if t.startswith(HUMAN_PREFIX) and not t.startswith("teacher_"):
                per.setdefault(t, []).append((hashlib.sha256(f"reason-sub|rad/{n}|{qid}".encode()).hexdigest(), n, qid))
    keep, counts = {}, {}
    for t, xs in sorted(per.items()):
        cap = CFG["caps"].get(t, CFG["cap"])
        xs.sort(); counts[t] = {"population": len(xs), "sampled": min(len(xs), cap)}
        for _, n, qid in xs[:cap]: keep.setdefault(n, set()).add(qid)
    with open(WORK / "sub.jsonl", "w") as f:
        for n in sorted(keep):
            r = json.loads(lines[n]); meta = dict(r.get("_meta", {}))
            meta.setdefault("id", f"rad/{n}"); meta.setdefault("group_id", f"rad/{n}")
            r["questions"] = {k: v for k, v in r["questions"].items() if k in keep[n]}; r["_meta"] = meta
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    result["sample"] = {"tasks": counts, "questions": sum(c["sampled"] for c in counts.values()), "records": len(keep)}


# ---------------------------------------------------------------- 2. reasoning with vLLM (one process per model, one GPU pair)
GEN = r'''
import json, math, os, re, sys, time
sys.path.insert(0, sys.argv[1]); from radkev import teacher as te
name, model_id, sub, out_dir, cfg = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], json.loads(sys.argv[6])
deadline_min = float(sys.argv[7])
out_dir = os.path.abspath(out_dir); os.makedirs(out_dir, exist_ok=True)
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
tok = AutoTokenizer.from_pretrained(model_id)
qwen = "qwen" in model_id.lower()
close_tok = "</think>" if qwen else "<unused95>"
close_id = tok.convert_tokens_to_ids(close_tok)
suffix_ids = tok.encode("\n\nAnswer:", add_special_tokens=False)
letter_ids = []
for L in te.LETTERS:
    ids = {tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L) if tok.encode(v, add_special_tokens=False)}
    letter_ids.append(ids)
try: gc = GenerationConfig.from_pretrained(model_id)
except Exception: gc = None
samp = {}
if gc is not None and getattr(gc, "do_sample", False):   # the model's own sampling defaults; greedy when it declares none
    samp = {k: getattr(gc, k) for k in ("temperature", "top_p", "top_k") if getattr(gc, k, None) not in (None, 0)}
info = {"model": model_id, "sampling": samp or "greedy", "close_token": close_tok, "close_id": close_id, "suffix_ids": suffix_ids}
items = []
for line in open(sub):
    r = json.loads(line)
    for qid, q in r["questions"].items():
        keys, opts = te.render_options(q)
        ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
        user = f"{te.state_text(r['state'])}\n\nQuestion: {ins}\nOptions:\n" + "\n".join(opts) + "\n\nAnswer with one letter."
        msgs = [{"role": "system", "content": te.SYSTEM}, {"role": "user", "content": user}]
        kw = {"enable_thinking": True} if qwen else {}
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        if not qwen: text += "<unused94>thought\n"   # open MedGemma's thought channel so it always reasons
        ids = tok.encode(text, add_special_tokens=False)
        items.append({"rid": r["_meta"]["id"], "qid": qid, "keys": keys, "ids": ids, "task": q.get("src")})
import hashlib
items.sort(key=lambda x: hashlib.sha256(f"order|{x['rid']}|{x['qid']}".encode()).hexdigest())   # random order: a partial run is a random subsample
done = {}
meta_path, pred_path = os.path.join(out_dir, "meta.jsonl"), os.path.join(out_dir, "probs.jsonl")
if os.path.exists(meta_path):
    for l in open(meta_path):
        m = json.loads(l); done[(m["rid"], m["qid"])] = m
too_long = [it for it in items if len(it["ids"]) > cfg["max_prompt"]]
todo = [it for it in items if (it["rid"], it["qid"]) not in done and len(it["ids"]) <= cfg["max_prompt"]]
info.update(n_items=len(items), n_too_long=len(too_long), n_done_before=len(done))
json.dump(info, open(os.path.join(out_dir, "info.json"), "w"))
kw = dict(model=model_id, tensor_parallel_size=2, dtype="bfloat16", max_model_len=cfg["max_model_len"], gpu_memory_utilization=0.90,
          max_num_seqs=128, enable_prefix_caching=True, seed=0, max_logprobs=20)
if qwen: kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
t0 = time.time()
try: llm = LLM(**kw)
except Exception as e:
    print("compiled engine failed, retrying eager:", str(e)[-500:], flush=True); llm = LLM(**kw, enforce_eager=True); info["eager"] = True
info["load_s"] = round(time.time() - t0, 1); json.dump(info, open(os.path.join(out_dir, "info.json"), "w"))
gen_sp = SamplingParams(max_tokens=cfg["think_cap"], seed=0, skip_special_tokens=False, **(samp if samp else {"temperature": 0.0}))
read_sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
deadline = time.time() + 60 * deadline_min if deadline_min > 0 else float("inf")
first = True
while todo and time.time() < deadline:
    n = 16 if first else cfg["chunk"]
    chunk, todo = todo[:n], todo[n:]
    t1 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=it["ids"]) for it in chunk], gen_sp, use_tqdm=False)
    t_gen = time.time() - t1
    reads = []
    for it, o in zip(chunk, outs):
        g = list(o.outputs[0].token_ids)
        closed = close_id in g
        thought = g[:g.index(close_id) + 1] if closed else g + [close_id]
        post = tok.decode(g[g.index(close_id) + 1:], skip_special_tokens=True) if closed else ""
        own = next((x for x in re.findall(r"\b([A-P])\b", post) if ord(x) - 65 < len(it["keys"])), None)
        it.update(n_new=len(g), closed=closed, hit_cap=len(g) >= cfg["think_cap"], own=own)
        reads.append(TokensPrompt(prompt_token_ids=it["ids"] + thought + suffix_ids))
    routs = llm.generate(reads, read_sp, use_tqdm=False)
    with open(meta_path, "a") as fm, open(pred_path, "a") as fp:
        for it, ro in zip(chunk, routs):
            lp = ro.outputs[0].logprobs[0] if ro.outputs[0].logprobs else {}
            s = []
            for ids in letter_ids[:len(it["keys"])]:
                v = [lp[t].logprob for t in ids if t in lp]
                s.append(max(v) + math.log(sum(math.exp(x - max(v)) for x in v)) if v else -1e9)
            found = any(x > -1e8 for x in s)
            if not found: s = [0.0] * len(s)
            mx = max(s); e = [math.exp(x - mx) for x in s]; z = sum(e); p = [x / z for x in e]
            mass = sum(math.exp(x) for x in s) if found else 0.0
            arg = it["keys"][max(range(len(p)), key=p.__getitem__)]
            own_key = it["keys"][ord(it["own"]) - 65] if it["own"] else None
            override = own_key is not None and own_key != arg   # the model's own written answer decides
            if override: p = [0.98 if k == own_key else 0.02 / (len(p) - 1) for k in it["keys"]]
            fp.write(json.dumps({"rid": it["rid"], "qid": it["qid"], "p": dict(zip(it["keys"], p))}) + "\n")
            fm.write(json.dumps({"rid": it["rid"], "qid": it["qid"], "task": it["task"], "n_new": it["n_new"], "closed": it["closed"],
                                 "hit_cap": it["hit_cap"], "letter_found": found, "letter_mass": mass, "own_parsed": own_key is not None,
                                 "own_agrees": own_key == arg if own_key is not None else None, "override": override, "chunk_gen_s": t_gen / len(chunk)}) + "\n")
    print(f"{name}: +{len(chunk)} in {t_gen:.0f}s, {len(todo)} left", flush=True)
    first = False
info["gen_s"] = round(time.time() - t0, 1); info["left_at_deadline"] = len(todo)
json.dump(info, open(os.path.join(out_dir, "info.json"), "w")); sys.stdout.flush()
os._exit(0)   # vLLM engines can hang at interpreter shutdown
'''


def gpu_indices(devs):
    """vLLM parses CUDA_VISIBLE_DEVICES as integers; GPU UUIDs are mapped to their indices."""
    if all(d.isdigit() for d in devs): return devs
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    idx = {u.strip(): i.strip() for i, u in (l.split(",") for l in out.strip().splitlines())}
    return [idx.get(d, d) for d in devs]


def lane(name, devs, deadline_min):
    d = WORK / name; d.mkdir(parents=True, exist_ok=True)
    short = Path(tempfile.mkdtemp(prefix=f"rk{name[:2]}"))   # vLLM's IPC socket paths must stay under 108 characters
    env = {**os.environ, "TMPDIR": str(short), "VLLM_RPC_BASE_PATH": str(short), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUDA_VISIBLE_DEVICES": ",".join(gpu_indices(devs)), "TOKENIZERS_PARALLELISM": "false", "VLLM_CACHE_ROOT": str(HOME / "vllm-cache"),
           "VLLM_PORT": "29610" if name.startswith("qwen") else "29710", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    env.pop("PYTHONPATH", None)
    status["interpreter"] = "vllm environment"
    neutral = WORK / "cwd"; neutral.mkdir(exist_ok=True)
    run([VLLM_PY, "-c", GEN, ROOT, name, LLMS[name], WORK / "sub.jsonl", d, json.dumps(CFG), deadline_min], env, WORK / f"{name}.log", cwd=neutral)


# ---------------------------------------------------------------- 3. Kev metrics on the sample
def preds_file(name):
    """probs.jsonl (one line per question) -> radkev.evaluate --preds records; records missing a question are dropped (partial runs)."""
    want = {}
    for line in open(WORK / "sub.jsonl"):
        r = json.loads(line); want[r["_meta"]["id"]] = set(r["questions"])
    got = {}
    for line in open(WORK / name / "probs.jsonl"):
        x = json.loads(line); got.setdefault(x["rid"], {})[x["qid"]] = x["p"]
    keep = {rid for rid, qs in got.items() if set(qs) == want.get(rid)}
    with open(WORK / f"{name}.sub.jsonl", "w") as fs, open(WORK / f"{name}.preds.jsonl", "w") as fp:
        for line in open(WORK / "sub.jsonl"):
            r = json.loads(line)
            if r["_meta"]["id"] in keep: fs.write(line); fp.write(json.dumps({"id": r["_meta"]["id"], "probabilities": got[r["_meta"]["id"]]}) + "\n")
    return len(keep)


def score(name):
    n = preds_file(name); status.setdefault("scored_records", {})[name] = n; save()
    run([KEV_PY, "-m", "radkev.evaluate", "--preds", WORK / f"{name}.preds.jsonl", "--data", WORK / f"{name}.sub.jsonl", "--out", SCORED / name],
        dict(os.environ), WORK / "score.log")


def gen_stats(name):
    ms = [json.loads(l) for l in open(WORK / name / "meta.jsonl")]
    info = json.loads((WORK / name / "info.json").read_text())

    def agg(xs):
        if not xs: return {}
        nn = sorted(x["n_new"] for x in xs)
        own = [x for x in xs if x["own_parsed"]]
        return {"n": len(xs), "median_new_tokens": nn[len(nn) // 2], "p90_new_tokens": nn[int(0.9 * (len(nn) - 1))], "mean_new_tokens": sum(nn) / len(nn),
                "closed_share": sum(x["closed"] for x in xs) / len(xs), "hit_cap_share": sum(x["hit_cap"] for x in xs) / len(xs),
                "letter_found_share": sum(x["letter_found"] for x in xs) / len(xs), "mean_letter_mass": sum(x["letter_mass"] for x in xs) / len(xs),
                "own_parsed_share": len(own) / len(xs), "own_agrees_share": (sum(bool(x["own_agrees"]) for x in own) / len(own)) if own else None,
                "gen_seconds_per_question_batched": sum(x["chunk_gen_s"] for x in xs) / len(xs)}
    by = {}
    for x in ms: by.setdefault(x["task"], []).append(x)
    return {"engine": {k: v for k, v in info.items() if k not in ("suffix_ids",)}, "overall": agg(ms), "tasks": {t: agg(v) for t, v in by.items()}}


# ---------------------------------------------------------------- 4. paired comparisons on the same questions (Kev environment)
def analyse():
    from kev.metrics import scored_rows

    from radkev import compare as C
    samples = CFG["samples"]
    think = [p.name for p in SCORED.iterdir() if (p / "rows.json").exists()]
    rows = {m: scored_rows(json.loads((SCORED / m / "rows.json").read_text())) for m in think}
    keys = set.intersection(*[{(r["id"], r["question"]) for r in rs} for rs in rows.values()]) if rows else set()
    for m in BASE:
        f = TESTDIR / m / "rows.json"
        if f.exists(): rows[m] = [r for r in scored_rows(json.loads(f.read_text())) if (r["id"], r["question"]) in keys]
    for m in think: rows[m] = [r for r in rows[m] if (r["id"], r["question"]) in keys]

    def summ(rs):
        s = C.summary(rs); tasks = {}
        for r in rs: tasks.setdefault(r["task"], []).append(r)
        s["macro_acc"] = sum(C.summary(v)["acc"] for v in tasks.values()) / len(tasks)
        return s
    res = {"n_questions": len(keys), "models": {}, "pairs": {}}
    for m, rs in rows.items():
        fams, tasks = {}, {}
        for r in rs: fams.setdefault(C.family(r["task"]), []).append(r); tasks.setdefault(r["task"], []).append(r)
        res["models"][m] = {"overall": summ(rs), "radiology_human": summ([r for r in rs if C.family(r["task"]) in C.RADIOLOGY_HUMAN]),
                            "families": {f: C.summary(v) for f, v in fams.items()}, "tasks": {t: C.summary(v) for t, v in tasks.items()},
                            "reliability": C.reliability(rs)}
    PAIRS = [("qwen38_think", "qwen38"), ("medgemma_think", "medgemma_brief"), ("medgemma_think", "medgemma_fix"), ("v2_27", "medgemma_fix"),
             ("qwen38_think", "medgemma_think"), ("v2_27", "qwen38_think"), ("v2_27", "medgemma_think"), ("v2_27", "qwen38"), ("v2_27", "medgemma_brief"),
             ("v2_27", "stock27"), ("r9", "qwen38_think"), ("r9", "medgemma_think"), ("r9", "qwen38"), ("r9", "stock9"), ("stock27", "qwen38_think"),
             ("stock27", "medgemma_think")]
    rh = lambda rs: [r for r in rs if C.family(r["task"]) in C.RADIOLOGY_HUMAN]
    for a, b in PAIRS:
        if a not in rows or b not in rows: continue
        A, B = rows[a], rows[b]
        p = {"macro_acc": C.paired(A, B, "acc", "macro", samples), "micro_acc": C.paired(A, B, "acc", "micro", samples),
             "brier": C.paired(A, B, "brier", "micro", samples),
             "radiology_human_macro_acc": C.paired(rh(A), rh(B), "acc", "macro", samples), "families": {}, "tasks": {}}
        for f in {C.family(r["task"]) for r in A}:
            p["families"][f] = C.paired([r for r in A if C.family(r["task"]) == f], [r for r in B if C.family(r["task"]) == f], "acc", "micro", samples)
        for t in {r["task"] for r in A}:
            p["tasks"][t] = C.paired([r for r in A if r["task"] == t], [r for r in B if r["task"] == t], "acc", "micro", samples)
        res["pairs"][f"{a}-{b}"] = p
    (WORK / "paired.json").write_text(json.dumps(res, separators=(",", ":")))
    result["paired"] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(LLMS), help="which reasoning rows to generate (each resumes where it stopped)")
    ap.add_argument("--deadline-min", dest="deadline_min", type=float, default=0, help="stop generating a model after N minutes (0: run to the end)")
    a = ap.parse_args()
    for p in (WORK, SCORED): p.mkdir(parents=True, exist_ok=True)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d] or ["0", "1"]
    status["gpus"] = devs; save()
    if len(devs) < 2: raise SystemExit("needs a GPU pair")
    if not phase("sample", sample): return
    for n in a.models.split(","): phase(f"gen_{n}", lambda n=n: lane(n, devs[0:2], a.deadline_min))   # one GPU pair, in turn
    result["generation"] = {}
    for n in LLMS:
        if (WORK / n / "meta.jsonl").exists() and (WORK / n / "probs.jsonl").exists():
            try: result["generation"][n] = gen_stats(n)
            except Exception as e: result["generation"][n] = {"error": str(e)[-500:]}
            phase(f"score_{n}", lambda n=n: score(n))
    save()
    phase("analyse", analyse)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()
    print(WORK / "llm_reasoning.json")


if __name__ == "__main__":
    main()
