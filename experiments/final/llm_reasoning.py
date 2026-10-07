"""Reasoning-on LLM baselines (post hoc; no training): Qwen3.8-27B and MedGemma-27B-text-it with their thinking turned on,
scored on a prespecified stratified sample of the held-out human-key test questions and paired with every existing row.

    python jobs/submit.py jobs/llm_reasoning.py --gpus 4 --cpus 16 --mem 120 --timeout 330 --expire 900 \
        --outputs 'llm_reasoning.json,llm_reasoning_status.json'
    v3 (amendment 8): --outputs 'llm_reasoning_v3*.json' -- v3

Sample (fixed before any reasoning output exists): every human-key task of runs/test-final/test.jsonl (report_cxr_human,
case_diagnosis, routing, radiology_knowledge, medical_knowledge; never teacher-labelled questions), at most CAP questions per
task, chosen by sha256("reason-sub|<record>|<qid>") order. Records keep their test ids and cluster ids, so every row pairs
with the full-test rows of the other models.
Prompt: exactly teacher.py's system and user text ("... Answer with one letter."); only the chat template changes:
  qwen38_think    enable_thinking=True
  medgemma_think  no pre-fill (MedGemma opens its own thought channel <unused94>thought ... <unused95>)
Decoding: vLLM 0.19 ($RADKEV/vllm-venv), bf16, tensor parallel over one NVLink pair per model, each model's own
generation_config sampling (temperature/top_p/top_k; greedy if it has none), seed 0, at most THINK_CAP new tokens.
Answer read-out (the same rule for both models, the reasoning analogue of the letter scoring used for every other row): the
reasoning is closed where the model closed it (or force-closed at the cap), "\\n\\nAnswer:" is appended, and the option-letter
distribution is the softmax over the next-token log-probabilities of the option letters (top-20 log-probs; " A" and "A"
variants pooled). The model's own written answer is parsed too and reported as agreement, not used for scoring.
Scored with jobs/kev_eval.py --preds (Kev's metrics), then paired with compare.py's cluster bootstrap against the no-thinking
rows of the same LLMs, RadKev-27B/9B and Kev-27B/9B on the same questions. Publishes aggregates only (no text).
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

INCLUDE = ["teacher.py", "build_data.py", "jobs/kev_eval.py", "jobs/compare.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
VLLM_PY = RADKEV / "vllm-venv/bin/python"
CONDA_PY = Path("/opt/conda/envs/llm/bin/python")   # fallback interpreter only if the private venv is missing
TESTDIR = RADKEV / "runs/test-final"; TEST = TESTDIR / "test.jsonl"
WORK = RADKEV / "reasoning"; CODE = WORK / "code"; RUNS = WORK / "runs"
CFG = {"cap": 60, "caps": {"eurorad_dx": 1000, "eurorad_route": 1000, "medmcqa_rad": 1000, "iu_finding": 200, "iu_normal": 200, "iu_which": 200},
       "think_cap": 4096, "max_model_len": 16384, "max_prompt": 11000, "chunk": 800, "deadline_min": {"qwen38_think": 32, "medgemma_think": 42}, "samples": 2000}
LLMS = {"qwen38_think": "Qwen/Qwen3.8-27B", "medgemma_think": "google/medgemma-27b-text-it"}
BASE = ["v2_27", "stock27", "r9", "stock9", "qwen38", "qwen38_gen", "medgemma_brief", "medgemma_fix"]
HUMAN_PREFIX = ("iu_", "eurorad_dx", "eurorad_route", "medmcqa_rad", "medmcqa_med", "medqa", "mmlu_", "medxpertqa", "pubmedqa")
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "cfg": CFG, "phases": {}}
result = {"what": "reasoning-on Qwen3.8-27B and MedGemma-27B on a stratified human-key test sample; aggregates only", "cfg": CFG}
LOCK = threading.Lock()


PREFIX = "llm_reasoning"


def save():
    with LOCK:
        (OUT / f"{PREFIX}_status.json").write_text(json.dumps(status, indent=1))
        (OUT / f"{PREFIX}.json").write_text(json.dumps(result, indent=1))


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as e: status["phases"][name] = {"state": "failed", "error": str(e)[-2500:], "minutes": round((time.time() - t0) / 60, 1)}
    save(); return status["phases"][name]["state"] == "ok"


def run(cmd, env, log, cwd=None):
    import signal
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))[:300]}\n"); f.flush()
        pr = subprocess.Popen([str(c) for c in cmd], env=env, cwd=cwd or KEV_DIR, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        rc = pr.wait()
        try: os.killpg(pr.pid, signal.SIGKILL)   # vLLM tensor-parallel workers outlive a parent that exits with os._exit (round 5)
        except ProcessLookupError: pass
        time.sleep(5)
    if rc: raise RuntimeError(f"exit {rc}: " + "".join(l for l in Path(log).read_text().splitlines(True)[-25:] if "hf_" not in l)[-2400:])


# ---------------------------------------------------------------- 1. the sample (CPU, stdlib)
def sample():
    import hashlib
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
        cap = CFG["caps"].get(t, CFG["cap"])   # round 5: every radiology question (IU 200 per task); same hash order, so a superset
        xs.sort(); counts[t] = {"population": len(xs), "sampled": min(len(xs), cap)}
        for _, n, qid in xs[:cap]: keep.setdefault(n, set()).add(qid)
    with open(WORK / "sub.jsonl", "w") as f:
        for n in sorted(keep):
            r = json.loads(lines[n]); meta = dict(r.get("_meta", {}))
            meta.setdefault("id", f"rad/{n}"); meta.setdefault("group_id", f"rad/{n}")
            r["questions"] = {k: v for k, v in r["questions"].items() if k in keep[n]}; r["_meta"] = meta
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    result["sample"] = {"tasks": counts, "questions": sum(c["sampled"] for c in counts.values()), "records": len(keep)}


# ---------------------------------------------------------------- 2. reasoning with vLLM (one process per model / GPU pair)
GEN = r'''
import json, math, os, re, sys, time
sys.path.insert(0, sys.argv[1]); import teacher as te
name, model_id, sub, out_dir, cfg = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], json.loads(sys.argv[6])
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
        if not qwen: text += "<unused94>thought\n"   # round 5: open MedGemma's thought channel so it always reasons
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
deadline = time.time() + 60 * cfg["deadline_min"][name]
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
            override = own_key is not None and own_key != arg   # round 5: the model's own written answer decides
            if override: p = [0.98 if k == own_key else 0.02 / (len(p) - 1) for k in it["keys"]]
            fp.write(json.dumps({"rid": it["rid"], "qid": it["qid"], "p": dict(zip(it["keys"], p))}) + "\n")
            fm.write(json.dumps({"rid": it["rid"], "qid": it["qid"], "task": it["task"], "n_new": it["n_new"], "closed": it["closed"],
                                 "hit_cap": it["hit_cap"], "letter_found": found, "letter_mass": mass, "own_parsed": own_key is not None,
                                 "own_agrees": own_key == arg if own_key is not None else None, "override": override, "chunk_gen_s": t_gen / len(chunk)}) + "\n")
    print(f"{name}: +{len(chunk)} in {t_gen:.0f}s, {len(todo)} left", flush=True)
    first = False
info["gen_s"] = round(time.time() - t0, 1); info["left_at_deadline"] = len(todo)
json.dump(info, open(os.path.join(out_dir, "info.json"), "w")); sys.stdout.flush()
os._exit(0)   # vLLM engines hung at interpreter shutdown in round 4
'''


def gpu_indices(devs):
    """The scheduler hands out GPU UUIDs; vLLM parses CUDA_VISIBLE_DEVICES as integers (round 2 failed on this)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    idx = {u.strip(): i.strip() for i, u in (l.split(",") for l in out.strip().splitlines())}
    return [idx.get(d, d) for d in devs]


def lane(name, devs):
    d = WORK / name; d.mkdir(parents=True, exist_ok=True)
    short = Path(f"/dev/shm/rk{name[:2]}"); short.mkdir(exist_ok=True)   # zmq IPC paths must be < 108 characters (round 3)
    env = {**os.environ, "TMPDIR": str(short), "VLLM_RPC_BASE_PATH": str(short),
           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(gpu_indices(devs)), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "VLLM_CACHE_ROOT": str(RADKEV / "vllm-cache"), "VLLM_PORT": "29610" if name.startswith("qwen") else "29710",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    env.pop("PYTHONPATH", None)
    py = VLLM_PY if VLLM_PY.exists() else CONDA_PY
    status["interpreter"] = str(py)
    neutral = WORK / "cwd"; neutral.mkdir(exist_ok=True)
    run([py, "-c", GEN, CODE, name, LLMS[name], WORK / "sub.jsonl", d, json.dumps(CFG)], env, WORK / f"{name}.log", cwd=neutral)


# ---------------------------------------------------------------- 3. Kev metrics on the sample
def preds_file(name):
    """probs.jsonl (one line per question) -> kev_eval --preds records; records missing a question are dropped (partial runs)."""
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
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}
    run([KEV_PY, CODE / "kev_eval.py", "--preds", WORK / f"{name}.preds.jsonl", "--data", WORK / f"{name}.sub.jsonl", "--out", RUNS / name], env, WORK / "score.log")


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


# ---------------------------------------------------------------- 4. paired comparisons on the same questions (Kev venv)
ANALYSE = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import compare as C
from kev.metrics import scored_rows
testdir, runs, out, samples = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
think = [p.name for p in runs.iterdir() if (p / "rows.json").exists()]
rows = {m: scored_rows(json.loads((runs / m / "rows.json").read_text())) for m in think}
keys = set.intersection(*[{(r["id"], r["question"]) for r in rs} for rs in rows.values()]) if rows else set()
for m in sys.argv[6].split(","):
    f = testdir / m / "rows.json"
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
PAIRS = [("qwen38_think", "qwen38"), ("medgemma_think", "medgemma_brief"), ("medgemma_think", "medgemma_fix"), ("v2_27", "medgemma_fix"), ("qwen38_think", "medgemma_think"),
         ("v2_27", "qwen38_think"), ("v2_27", "medgemma_think"), ("v2_27", "qwen38"), ("v2_27", "medgemma_brief"), ("v2_27", "stock27"),
         ("r9", "qwen38_think"), ("r9", "medgemma_think"), ("r9", "qwen38"), ("r9", "stock9"), ("stock27", "qwen38_think"), ("stock27", "medgemma_think")]
for a, b in PAIRS:
    if a not in rows or b not in rows: continue
    A, B = rows[a], rows[b]
    rh = lambda rs: [r for r in rs if C.family(r["task"]) in C.RADIOLOGY_HUMAN]
    p = {"macro_acc": C.paired(A, B, "acc", "macro", samples), "micro_acc": C.paired(A, B, "acc", "micro", samples),
         "brier": C.paired(A, B, "brier", "micro", samples),
         "radiology_human_macro_acc": C.paired(rh(A), rh(B), "acc", "macro", samples), "families": {}, "tasks": {}}
    for f in {C.family(r["task"]) for r in A}:
        p["families"][f] = C.paired([r for r in A if C.family(r["task"]) == f], [r for r in B if C.family(r["task"]) == f], "acc", "micro", samples)
    for t in {r["task"] for r in A}:
        p["tasks"][t] = C.paired([r for r in A if r["task"] == t], [r for r in B if r["task"] == t], "acc", "micro", samples)
    res["pairs"][f"{a}-{b}"] = p
out.write_text(json.dumps(res, separators=(",", ":")))
print(json.dumps({m: v["overall"] for m, v in res["models"].items()}))
'''


def analyse():
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}
    run([KEV_PY, "-c", ANALYSE, CODE, TESTDIR, RUNS, WORK / "paired.json", CFG["samples"], ",".join(BASE)], env, WORK / "analyse.log")
    result["paired"] = json.loads((WORK / "paired.json").read_text())


# ---------------------------------------------------------------- v3 (addendum 2026-10-05b, amendment 8)
# The LLMs are unchanged in v3, so their reasoning rows on the held-out test file are reused ($RADKEV/reasoning/runs); reasoning is
# run once more on a sample of the new test sets (RadCases, ReXErr, RSNA-RadioQA; $RADKEV/runs/test-v3/new.jsonl): every
# RSNA-RadioQA question, at most 150 per RadCases benchmark task (panel, topic) and 200 ReXErr questions, by sha256 order; questions
# with more than 16 options are left out (letter read-out). The paired analysis is restricted to the radiology benchmark questions
# (jobs/eval_v3.py's task mapping) and pairs the reasoning rows with the v3 systems' rows on identical questions.
V3_CAPS = {"rsna_radioqa": 10 ** 6, "radcases_panel": 150, "radcases_topic": 150, "rexerr_error": 200}


def v3_task(t):
    return "radcases_panel" if t.startswith("radcases_panel") else "radcases_topic" if t.startswith("radcases_topic") else t


def sample_v3():
    import hashlib
    per, recs = {}, {}
    for line in open(RADKEV / "runs/test-v3/new.jsonl"):
        if not line.strip(): continue
        r = json.loads(line); rid = r["_meta"]["id"]; recs[rid] = r
        for qid, q in r["questions"].items():
            t = v3_task(str(q.get("src", "")))
            if t in V3_CAPS and len(q.get("criteria") or {}) <= 16:
                per.setdefault(t, []).append((hashlib.sha256(f"reason-v3|{rid}|{qid}".encode()).hexdigest(), rid, qid))
    keep, counts = {}, {}
    for t, xs in sorted(per.items()):
        xs.sort(); counts[t] = {"population": len(xs), "sampled": min(len(xs), V3_CAPS[t])}
        for _, rid, qid in xs[:V3_CAPS[t]]: keep.setdefault(rid, set()).add(qid)
    with open(WORK / "sub.jsonl", "w") as f:
        for rid in sorted(keep):
            r = dict(recs[rid]); r["questions"] = {k: v for k, v in r["questions"].items() if k in keep[rid]}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    result["sample"] = {"tasks": counts, "questions": sum(c["sampled"] for c in counts.values()), "records": len(keep)}


ANALYSE_V3 = r'''
import json, sys, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
old_runs, new_runs, W, FINAL, out = map(Path, sys.argv[2:7])
radidx = json.loads((FINAL / "radiology_index.json").read_text())
HUMAN = ["iu_finding", "iu_normal", "iu_which", "eurorad_dx", "eurorad_route", "medmcqa_rad", "medmcqa_other_rad", "medqa_rad", "medxpertqa_rad",
         "mmlu_rad", "pubmedqa_rad", "radcases_panel", "radcases_topic", "rsna_radioqa"]
BENCH = HUMAN + ["rexerr_error"]
def bench_task(task, rid, qid):   # identical to jobs/eval_v3.py
    if task in ("iu_finding", "iu_normal", "iu_which", "eurorad_dx", "eurorad_route", "medmcqa_rad", "rsna_radioqa", "rexerr_error"): return task
    if task.startswith("radcases_panel"): return "radcases_panel"
    if task.startswith("radcases_topic"): return "radcases_topic"
    pooled = {"medmcqa_med": "medmcqa_other_rad", "medqa": "medqa_rad", "medxpertqa": "medxpertqa_rad", "pubmedqa": "pubmedqa_rad"}
    t = pooled.get(task) or ("mmlu_rad" if task.startswith("mmlu_") else None)
    if t and radidx.get(f"{rid}|{qid}"): return t
    return None
def rows_of(paths):
    o = {}
    for f in paths:
        if f.exists():
            for r in scored_rows(json.loads(f.read_text())):
                bt = bench_task(r["task"], r["id"], r["question"])
                if bt: o[(r["id"], r["question"])] = (bt, int(r["label"]), np.asarray(r["p"], float))
    return o
D = {}
for m in ("qwen38_think", "medgemma_think"):
    D[m] = rows_of([old_runs / m / "rows.json", new_runs / m / "rows.json"])
for m in ("v3_27", "v3_9", "stock27", "stock9", "qwen38", "medgemma_fix"):
    D[m] = rows_of([W / "final" / f"{m}_s0" / "rows.json", W / "final" / f"{m}_s1" / "rows.json", FINAL / m / "rows.json", W / "new" / m / "rows.json"])
D = {m: v for m, v in D.items() if v}
think = [m for m in ("qwen38_think", "medgemma_think") if m in D]
keys = sorted(set.intersection(*[set(D[m]) for m in think])) if think else []
task = {k: D[think[0]][k][0] for k in keys}
res = {"n": {t: sum(1 for k in keys if task[k] == t) for t in BENCH if any(task[k] == t for k in keys)}, "systems": {}, "pairs": {}, "coverage": {}}
rec = sorted({k[0] for k in keys}); ridx = {r: i for i, r in enumerate(rec)}
B = 2000; rng = np.random.default_rng(20261005); W8 = np.zeros((B, len(rec)), np.float32); strata = {}
for k in keys: strata.setdefault(task[k], set()).add(ridx[k[0]])
for t in sorted(strata):
    mem = np.array(sorted(strata[t])); dr = rng.integers(0, len(mem), size=(B, len(mem)))
    for b in range(B): np.add.at(W8[b], mem[dr[b]], 1)
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
pv = lambda d: float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
def vec(m, ks):
    return np.array([float(np.argmax(D[m][k][2]) == D[m][k][1]) for k in ks]), W8[:, [ridx[k[0]] for k in ks]]
def acc(c, w): return float(c.mean()), (w @ c) / np.maximum(w.sum(1), 1e-9)
def tmean(m, ks):
    pts, bs = [], []
    for t in BENCH:
        kk = [k for k in ks if task[k] == t]
        if kk: c, w = vec(m, kk); a, b = acc(c, w); pts.append(a); bs.append(b)
    return float(np.mean(pts)), np.mean(bs, 0), len(pts)
for m in D:
    ks = [k for k in keys if k in D[m]]; res["coverage"][m] = len(ks)
    if len(ks) < len(keys): continue   # every system is reported on the identical question set only
    r = {"tasks": {}}
    for t in res["n"]:
        c, w = vec(m, [k for k in ks if task[k] == t]); a, b = acc(c, w); r["tasks"][t] = {"n": int(len(c)), "acc": a, "ci": ci(b)}
    tm, tb, nt = tmean(m, ks); c, w = vec(m, ks); a, b = acc(c, w)
    r["bench"] = {"task_mean": tm, "task_mean_ci": ci(tb), "tasks": nt, "pooled": a, "pooled_ci": ci(b), "n": len(ks)}
    res["systems"][m] = r
PAIRS = [("qwen38_think", "qwen38"), ("medgemma_think", "medgemma_fix"), ("v3_27", "qwen38_think"), ("v3_27", "medgemma_think"), ("v3_9", "qwen38_think"),
         ("v3_9", "medgemma_think"), ("stock27", "qwen38_think"), ("qwen38_think", "medgemma_think"), ("v3_27", "stock27"), ("v3_27", "qwen38")]
for a_, b_ in PAIRS:
    if a_ not in res["systems"] or b_ not in res["systems"]: continue
    ma, mab, nt = tmean(a_, keys); mb, mbb, _ = tmean(b_, keys)
    ca, wa = vec(a_, keys); cb, _ = vec(b_, keys); pa, ba = acc(ca, wa); pb, bb = acc(cb, wa)
    o = {"bench": {"task_mean_d": ma - mb, "task_mean_ci": ci(mab - mbb), "task_mean_p": pv(mab - mbb), "tasks": nt,
                   "pooled_d": pa - pb, "pooled_ci": ci(ba - bb), "pooled_p": pv(ba - bb), "n": len(keys)}, "tasks": {}}
    for t in res["n"]:
        kk = [k for k in keys if task[k] == t]; ca, wa = vec(a_, kk); cb, _ = vec(b_, kk); pa, ba = acc(ca, wa); pb, bb = acc(cb, wa)
        o["tasks"][t] = {"n": len(kk), "d": pa - pb, "ci": ci(ba - bb), "p": pv(ba - bb)}
    res["pairs"][f"{a_}-{b_}"] = o
out.write_text(json.dumps(res, indent=1)); print(json.dumps({m: v["bench"]["task_mean"] for m, v in res["systems"].items()}))
'''


def analyse_v3():
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1"}
    run([KEV_PY, "-c", ANALYSE_V3, CODE, RADKEV / "reasoning/runs", RUNS, RADKEV / "runs/test-v3", TESTDIR, WORK / "paired_v3.json"], env, WORK / "analyse.log")
    result["paired"] = json.loads((WORK / "paired_v3.json").read_text())


def main_v3():
    global WORK, CODE, RUNS, PREFIX
    WORK = RADKEV / "reasoning_v3"; CODE = WORK / "code"; RUNS = WORK / "runs"; PREFIX = "llm_reasoning_v3"
    CFG["deadline_min"] = {"qwen38_think": 60, "medgemma_think": 75}; result["what"] = "v3: " + result["what"]
    for p in (WORK, CODE, RUNS): p.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    status["gpus"] = devs; save()
    if len(devs) < 2: raise SystemExit("needs 2 GPUs")
    if not (WORK / "sub.jsonl").exists():
        if not phase("sample", sample_v3): return
        (WORK / "sample.json").write_text(json.dumps(result["sample"]))
    result["sample"] = json.loads((WORK / "sample.json").read_text())
    lanes = [("qwen38_think", devs[0:2]), ("medgemma_think", devs[2:4] if len(devs) >= 4 else devs[0:2])]
    if len(devs) >= 4:
        ts = [threading.Thread(target=phase, args=(f"gen_{n}", (lambda n=n, d=d: lane(n, d)))) for n, d in lanes]
        for t in ts: t.start()
        for t in ts: t.join()
    else:
        for n, d in lanes: phase(f"gen_{n}", lambda n=n, d=d: lane(n, d))
    result["generation"] = {}
    for n in LLMS:
        if (WORK / n / "meta.jsonl").exists():
            try: result["generation"][n] = gen_stats(n)
            except Exception as e: result["generation"][n] = {"error": str(e)[-500:]}
            phase(f"score_{n}", lambda n=n: score(n))
    save()
    phase("analyse", analyse_v3)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "v3": return main_v3()
    for p in (WORK, CODE, RUNS): p.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    status["gpus"] = devs; save()
    if len(devs) < 2: raise SystemExit("needs 2 GPUs")
    if True:
        if not phase("sample", sample): return
    else:
        result["sample"] = json.loads((WORK / "sample.json").read_text()) if (WORK / "sample.json").exists() else None
    (WORK / "sample.json").write_text(json.dumps(result["sample"]))
    old_mg = WORK / "medgemma_think"
    if old_mg.exists() and not (old_mg / "round5").exists(): old_mg.rename(WORK / f"medgemma_think_round4_{int(time.time())}")
    (WORK / "medgemma_think").mkdir(exist_ok=True); (WORK / "medgemma_think" / "round5").write_text("1")
    for n in ("qwen38_think", "medgemma_think"): phase(f"gen_{n}", lambda n=n: lane(n, devs[0:2]))   # one NVLink pair, in turn
    result["generation"] = {}
    for n in LLMS:
        if (WORK / n / "meta.jsonl").exists():
            try: result["generation"][n] = gen_stats(n)
            except Exception as e: result["generation"][n] = {"error": str(e)[-500:]}
            phase(f"score_{n}", lambda n=n: score(n))
    save()
    phase("analyse", analyse)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    main()
