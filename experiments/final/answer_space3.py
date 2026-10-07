"""Answer-space study for v3 (post hoc; no training): how the size and composition of the offered answers change the accuracy,
confidence and latency of the decision models and the LLMs, with no artificial ceiling on the number of options.

    python jobs/submit.py jobs/answer_space3.py --gpus 4 --cpus 24 --mem 110 --timeout 1440 --outputs 'answer_space3*.json'

Questions (radiology only; the human-assigned key never changes, only the options offered with it): every Eurorad diagnosis question
of the held-out test split and every RSNA-RadioQA question. Distractor pools: every distinct Eurorad diagnosis option (train, dev,
test); RSNA-RadioQA additionally draws on its own options. Conditions (each its own question on the same state):
  orig            the options as scored in the benchmark
  sim_K           key + the K-1 pool distractors most similar to the key (PubMedBERT cosine)        K in sizes (2 to 255)
  rand_K, randb_K key + K-1 random pool distractors, two independent seeded draws                   K in sizes
  llm_K, orig_llm Qwen3.8-27B-written plausible alternatives (as in the pilot)                     K <= 16
A candidate is dropped if it contains, or is contained in, the key or an earlier pick, or if its cosine to either is >= 0.95.
Systems: RadKev-27B v3, Kev-27B, RadKev-9B v3, Kev-9B (each option scored directly; Kev accepts up to 255 options);
Qwen3.8-27B and MedGemma-27B-text, (a) numbered options with greedy generation of the answer number (every K; an answer that is not
a number in range counts as wrong and is reported), (b) the pilot's option-letter probabilities where K <= 16 (method check).
Latency: 60 Eurorad cases x every K, one request at a time, CUDA synchronised, the first 10 requests excluded; decision models on
their usual GPUs, the LLMs on one NVLink pair, letter scoring (K <= 16) and numbered generation (every K).
Publishes per-question correctness and confidence (no text), latencies and filter counts.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

INCLUDE = ["teacher.py", "build_data.py", "jobs/kev_multigpu.patch"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
WORK = RADKEV / "answer_space3s"; CODE = WORK / "code"   # reduced run (2026-10-06): fresh work directory
TEST = RADKEV / "runs/test-final/test.jsonl"; DATA = RADKEV / "data/rad-open"; RADIOQA = RADKEV / "data/rsna-radioqa/test.jsonl"
EMBED = ("NeuML/pubmedbert-base-embeddings", "b79526d6ef3645e0df4530322e266f24c829f5ef")
GENLLM = "Qwen/Qwen3.8-27B"
LLMS = {"qwen38": ("Qwen/Qwen3.8-27B", "0"), "medgemma_fix": ("google/medgemma-27b-text-it", "1")}
# reduced post hoc for runtime at the user's request (2026-10-06, before any answer-space result): a fixed sha256-ordered sample of
# 30 Eurorad diagnosis + 30 RSNA-RadioQA questions, five sizes, one random draw, latency on 15 cases at four sizes
CFG = {"sizes": [2, 4, 16, 64, 255], "llm_sizes": [2, 4, 8, 16], "add": 8, "n_alt": 15, "second_draw": False,
       "n_questions": {"eurorad_dx": 30, "rsna_radioqa": 30},
       "lat_sizes": [2, 16, 64, 255], "lat_cases": 15, "lat_warmup": 5, "sim_dup": 0.95}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}
LOCK = threading.Lock()


def save_status():
    with LOCK: (OUT / "answer_space3_status.json").write_text(json.dumps(status, indent=1))


ENV = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def ckpt(*names):
    for n in names:
        p = RADKEV / "runs" / n / "checkpoint"
        if (p / "head.pt").exists(): return str(p)
    return None


def run_inner(name, code, args, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {name}\n"); f.flush()
        rc = subprocess.run([str(KEV_PY), "-c", code, *map(str, args)], env=env, cwd=KEV_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"{name} exited {rc}: " + "".join(Path(log).read_text().splitlines(True)[-15:])[-1800:])


SELECT = r'''
import hashlib, json, random, re, sys
from pathlib import Path
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def seed(*a): return int(hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()[:12], 16)
def text(crit, k): v = crit[k]; return str(k if v is None else v).strip()
def contains(a, b): return len(a) >= 4 and len(b) >= 4 and (a in b or b in a)

test_path, radioqa, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
qs = []
for n, line in enumerate(open(test_path)):
    if not line.strip(): continue
    r = json.loads(line)
    for qid, q in r["questions"].items():
        if q.get("src") == "eurorad_dx" and q["type"] == "choice":
            qs.append({"rid": f"eurorad_dx/{n}/{qid}", "src": "eurorad_dx", "state": r["state"], "q": q,
                       "options": [text(q["criteria"], k) for k in q["criteria"]], "key": text(q["criteria"], q["label"])})
for line in open(radioqa):
    if not line.strip(): continue
    r = json.loads(line)
    for qid, q in r["questions"].items():
        qs.append({"rid": f"rsna_radioqa/{r['_meta']['id'].split('/')[-1]}/{qid}", "src": "rsna_radioqa", "state": r["state"], "q": q,
                   "options": [text(q["criteria"], k) for k in q["criteria"]], "key": text(q["criteria"], q["label"])})
n_per = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
if n_per:   # fixed subsample: the first n questions per source in sha256(rid) order
    keep = []
    for s_ in ("eurorad_dx", "rsna_radioqa"):
        xs = sorted((t for t in qs if t["src"] == s_), key=lambda t: hashlib.sha256(("as3-sub|" + t["rid"]).encode()).hexdigest())
        keep += xs[:n_per.get(s_, len(xs))]
    qs = keep
with open(work / "questions.jsonl", "w") as f:
    for t in qs: f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(json.dumps({"eurorad_dx": sum(t["src"] == "eurorad_dx" for t in qs), "rsna_radioqa": sum(t["src"] == "rsna_radioqa" for t in qs)}))
'''
GEN = r'''
import json, sys, torch
sys.path.insert(0, sys.argv[5])
import teacher as te
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id, inp, out, n_alt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
items = [json.loads(l) for l in open(inp) if l.strip()]
done = set()
try:
    for l in open(out): done.add(json.loads(l)["rid"])
except FileNotFoundError: pass
tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"; tok.truncation_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
INSTR = ("List {n} different answers to this question that are plausible but incorrect, such as alternative diagnoses or "
         "options that a knowledgeable clinician might consider. Each must differ clearly from the correct answer and from the "
         "others. Write one answer per line, with no numbering and no explanations.")
def prompt(it):
    user = (f"{te.state_text(it['state'])}\n\nQuestion: {it['q']['instructions']}\nCorrect answer: {it['key']}\n\n" + INSTR.format(n=n_alt))
    msgs = [{"role": "system", "content": te.SYSTEM.split(' Read the case')[0]}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
todo = sorted([it for it in items if it["rid"] not in done], key=lambda it: len(prompt(it)))
with open(out, "a") as f, torch.no_grad():
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        enc = tok([prompt(it) for it in chunk], return_tensors="pt", padding=True, truncation=True, max_length=3072).to(model.device)
        gen = model.generate(**enc, max_new_tokens=260, do_sample=False, pad_token_id=tok.pad_token_id)
        for it, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            f.write(json.dumps({"rid": it["rid"], "raw": tok.decode(g, skip_special_tokens=True)}, ensure_ascii=False) + "\n")
        f.flush(); print(f"{i + len(chunk)}/{len(todo)}", flush=True)
'''
BUILD = r'''
import hashlib, json, random, re, sys
from pathlib import Path
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def seed(*a): return int(hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()[:12], 16)
def text(crit, k): v = crit[k]; return str(k if v is None else v).strip()
def contains(a, b): return len(a) >= 4 and len(b) >= 4 and (a in b or b in a)

import os
import numpy as np
work, data_dir, embed_repo, embed_rev, cfg, radioqa = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], json.loads(sys.argv[5]), sys.argv[6]
DUP = cfg["sim_dup"]
qs = [json.loads(l) for l in open(work / "questions.jsonl") if l.strip()]
raw = {json.loads(l)["rid"]: json.loads(l)["raw"] for l in open(work / "gen.jsonl") if l.strip()}
pools = {s: {} for s in ("eurorad_dx", "rsna_radioqa")}
for split in ("train", "dev", "test"):
    for line in open(data_dir / f"{split}.jsonl"):
        if not line.strip(): continue
        for q in json.loads(line)["questions"].values():
            if q.get("src") == "eurorad_dx" and q["type"] == "choice":
                for k in q["criteria"]:
                    t = text(q["criteria"], k); n_ = norm(t)
                    for s_ in pools:
                        if n_ and n_ not in pools[s_]: pools[s_][n_] = t
for line in open(radioqa):
    if line.strip():
        for q in json.loads(line)["questions"].values():
            for k in q["criteria"]:
                t = text(q["criteria"], k); n_ = norm(t)
                if n_ and n_ not in pools["rsna_radioqa"]: pools["rsna_radioqa"][n_] = t
if os.environ.get("AS_FAKE_EMBED"):
    def embed(texts):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f"  {norm(t)}  "
            for j in range(len(s) - 2): out[i, seed(s[j:j + 3]) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
else:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(embed_repo, revision=embed_rev, allow_patterns=["*.json", "*.txt", "model.safetensors"])
    etok = AutoTokenizer.from_pretrained(path); enc = AutoModel.from_pretrained(path).cuda().eval()
    def embed(texts, bs=512):
        out = []
        for i in range(0, len(texts), bs):
            b = etok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
            with torch.no_grad(): h = enc(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
P = {s: list(v.values()) for s, v in pools.items()}; E = {s: embed(P[s]) for s in P}
def parse(t):
    out = []
    for line in str(t).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s*", "", line).strip().strip('"').strip()
        if 2 <= len(line) <= 160 and not line.startswith("<") and not line.lower().startswith(("correct answer", "here are", "note")): out.append(line)
    return out
def choose(cands, cvecs, kv, kn, taken, need):
    """first `need` candidates that are not near-duplicates of the key or of an earlier pick (or of `taken`)"""
    chosen, cv, cn, drop = [], [v for _, v in taken], [n for n, _ in taken], {"dup_key": 0, "dup_other": 0}
    for t, e in zip(cands, cvecs):
        tn = norm(t)
        if not tn or tn == kn or contains(tn, kn) or float(e @ kv) >= DUP: drop["dup_key"] += 1; continue
        if any(tn == c or contains(tn, c) for c in cn) or (cv and float((np.stack(cv) @ e).max()) >= DUP): drop["dup_other"] += 1; continue
        chosen.append(t); cv.append(e); cn.append(tn)
        if len(chosen) == need: break
    return chosen, drop
def make_q(t, texts, cond, keep=False):
    if keep: q = dict(t["q"]); q["src"] = f"as3_{t['src']}_{cond}"; return q
    rng = random.Random(seed(t["rid"], cond)); xs = list(texts); rng.shuffle(xs); kn = norm(t["key"])
    crit = {f"opt_{j + 1}": x for j, x in enumerate(xs)}
    label = next(k for k, x in crit.items() if norm(x) == kn)
    return {"type": "choice", "instructions": t["q"]["instructions"], "criteria": crit, "label": label, "src": f"as3_{t['src']}_{cond}"}
recs, meta, gstats, examples = [], {}, {"parsed": [], "kept": [], "dup_key": 0, "dup_other": 0}, []
for t in qs:
    s, kn = t["src"], norm(t["key"]); kv = embed([t["key"]])[0]
    dis = [o for o in t["options"] if norm(o) != kn]
    sim_order = np.argsort(-(E[s] @ kv), kind="stable").tolist()
    rnd_order = list(range(len(P[s]))); random.Random(seed(t["rid"], "rand")).shuffle(rnd_order)
    need = max(cfg["sizes"]) - 1
    sim_d, _ = choose([P[s][i] for i in sim_order[:4000]], [E[s][i] for i in sim_order[:4000]], kv, kn, [], need)
    rnd_d, _ = choose([P[s][i] for i in rnd_order[:4000]], [E[s][i] for i in rnd_order[:4000]], kv, kn, [], need)
    rnd_order2 = list(range(len(P[s]))); random.Random(seed(t["rid"], "rand2")).shuffle(rnd_order2)
    rnd2_d, _ = choose([P[s][i] for i in rnd_order2[:4000]], [E[s][i] for i in rnd_order2[:4000]], kv, kn, [], need)
    cands = parse(raw.get(t["rid"], "")); cvec = embed(cands) if cands else []
    llm_d, drop = choose(cands, cvec, kv, kn, [], max(cfg["llm_sizes"]) - 1)
    taken = [(norm(o), embed([o])[0]) for o in dis]
    add_d, _ = choose(cands, cvec, kv, kn, taken, cfg["add"])
    gstats["parsed"].append(len(cands)); gstats["kept"].append(len(llm_d)); gstats["dup_key"] += drop["dup_key"]; gstats["dup_other"] += drop["dup_other"]
    qd, m = {}, {}
    def add(cond, texts, keep=False):
        q = make_q(t, texts, cond, keep); qd[cond] = q
        others = [x for k, x in q["criteria"].items() if k != q["label"]]
        m[cond] = {"K": len(q["criteria"]), "dcos": round(float(np.mean(embed([text(q["criteria"], k) for k in q["criteria"] if k != q["label"]]) @ kv)), 4) if others else None}
    add("orig", t["options"], keep=True)
    for K in cfg["sizes"]:
        if len(rnd_d) >= K - 1: add(f"rand_{K}", [t["key"]] + rnd_d[:K - 1])
        if cfg.get("second_draw", True) and len(rnd2_d) >= K - 1: add(f"randb_{K}", [t["key"]] + rnd2_d[:K - 1])
        if len(sim_d) >= K - 1: add(f"sim_{K}", [t["key"]] + sim_d[:K - 1])
    for K in cfg["llm_sizes"]:
        if len(llm_d) >= K - 1: add(f"llm_{K}", [t["key"]] + llm_d[:K - 1])
    if add_d: add("orig_llm", t["options"] + add_d)
    recs.append({"state": t["state"], "questions": qd, "_meta": {"rid": t["rid"]}}); meta[t["rid"]] = {"src": s, "conds": m}
    if s == "eurorad_dx" and len(examples) < 2 and len(llm_d) >= 7 and isinstance(t["state"], dict):
        examples.append({"rid": t["rid"], "findings_head": str(t["state"].get("imaging findings", ""))[:90], "key": t["key"],
                         "original": t["options"], "llm": llm_d[:7]})
with open(work / "records.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
rng = random.Random(0); lat_qs = rng.sample([t for t in qs if t["src"] == "eurorad_dx"], cfg["lat_cases"]); lat = []
for t in lat_qs:
    kn = norm(t["key"]); kv = embed([t["key"]])[0]
    order = list(range(len(P[t["src"]]))); random.Random(seed("lat", t["rid"])).shuffle(order)
    d, _ = choose([P[t["src"]][i] for i in order[:4000]], [E[t["src"]][i] for i in order[:4000]], kv, kn, [], max(cfg["lat_sizes"]) - 1)
    for K in cfg["lat_sizes"]:
        lat.append({"state": t["state"], "questions": {"q": make_q(t, [t["key"]] + d[:K - 1], f"lat_{K}")}, "_meta": {"rid": f"lat/{t['rid']}/{K}", "K": K}})
rng.shuffle(lat)
with open(work / "latency_records.jsonl", "w") as f:
    for r in lat: f.write(json.dumps(r, ensure_ascii=False) + "\n")
(work / "meta.json").write_text(json.dumps({"meta": meta, "gen": gstats, "examples": examples, "pool": {s: len(P[s]) for s in P}}))
print(json.dumps({"records": len(recs), "questions": sum(len(r["questions"]) for r in recs), "latency": len(lat)}))
'''
KEV = r'''
import json, sys, time, torch
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT
run, inp, out = sys.argv[1:4]
recs = [json.loads(l) for l in open(inp) if l.strip()]
done = set()
try:
    for l in open(out):
        try: done.add(json.loads(l)["rid"])
        except Exception: pass
except FileNotFoundError: pass
pred = LocalPredictor(run, "cuda", LoadOptions.from_env(), context=SERVING_CONTEXT)
with open(out, "a") as f:
    for i, r in enumerate(recs):
        rid = r["_meta"]["rid"]
        if rid in done: continue
        torch.cuda.synchronize(); t = time.perf_counter()
        p = pred(r)
        torch.cuda.synchronize(); wall = 1000 * (time.perf_counter() - t)
        f.write(json.dumps({"rid": rid, "ms": wall, "p": p["probabilities"]}) + "\n"); f.flush()
        if i % 100 == 0: print(f"{i}/{len(recs)}", flush=True)
'''
LLMNUM = r'''
import json, re, sys, time, torch
sys.path.insert(0, sys.argv[5])
import teacher as te
model_id, inp, out, mode, think_off = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6] == "1"
recs = [json.loads(l) for l in open(inp) if l.strip()]
done = set()
try:
    for l in open(out): done.add(json.loads(l)["key"])
except FileNotFoundError: pass
S = te.LetterScorer(model_id, think_off); tok, model = S.tok, S.model
SYSTEM_NUM = te.SYSTEM.replace("single letter of the best option", "number of the best option")
PREFILL = S.prefill.replace("the letter only", "the number only")
def prompt(state, q):
    keys = list(q["criteria"]); lines = [f"{i + 1}. {q['criteria'][k] or k}" for i, k in enumerate(keys)]
    user = f"{te.state_text(state)}\n\nQuestion: {q['instructions']}\nOptions:\n" + "\n".join(lines) + "\n\nAnswer with the number of the best option only."
    msgs = [{"role": "system", "content": SYSTEM_NUM}, {"role": "user", "content": user}]
    try: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return keys, text + PREFILL
items = []
for r in recs:
    for c, q in r["questions"].items():
        k = f"{r['_meta']['rid']}|{c}"
        if k in done: continue
        if mode == "letter" and len(q["criteria"]) > 16: continue
        items.append((k, r["state"], q))
with open(out, "a") as f:
    if mode == "letter":
        for k, p in S.score(items, batch=8, max_len=8192, log="letter "):
            f.write(json.dumps({"key": k, "p": p}) + "\n"); f.flush()
    else:
        P = sorted(((k, *prompt(st, q)) for k, st, q in items), key=lambda x: len(x[2]))
        with torch.no_grad():
            for i in range(0, len(P), 8):
                ch = P[i:i + 8]
                enc = tok([c[2] for c in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                g = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t0) / len(ch)
                for (k, keys, _), seq in zip(ch, g[:, enc["input_ids"].shape[1]:]):
                    txt = tok.decode(seq, skip_special_tokens=True); m = re.search(r"\d+", txt)
                    n = int(m.group(0)) if m else None; ok = n is not None and 1 <= n <= len(keys)
                    f.write(json.dumps({"key": k, "pick": keys[n - 1] if ok else None, "raw": txt[:20], "ms_batched": round(ms, 1)}) + "\n")
                f.flush()
                if (i // 8) % 50 == 0: print(f"num {i + len(ch)}/{len(P)}", flush=True)
'''
LLMLAT = r'''
import json, re, sys, time, torch
sys.path.insert(0, sys.argv[4])
import teacher as te
model_id, inp, out, think_off = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5] == "1"
S = te.LetterScorer(model_id, think_off); tok, model = S.tok, S.model
SYSTEM_NUM = te.SYSTEM.replace("single letter of the best option", "number of the best option")
PREFILL = S.prefill.replace("the letter only", "the number only")
def num_prompt(state, q):
    keys = list(q["criteria"]); lines = [f"{i + 1}. {q['criteria'][k] or k}" for i, k in enumerate(keys)]
    user = f"{te.state_text(state)}\n\nQuestion: {q['instructions']}\nOptions:\n" + "\n".join(lines) + "\n\nAnswer with the number of the best option only."
    msgs = [{"role": "system", "content": SYSTEM_NUM}, {"role": "user", "content": user}]
    try: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + PREFILL
recs = [json.loads(l) for l in open(inp) if l.strip()]
with open(out, "w") as f, torch.no_grad():
    for r in recs:
        q = r["questions"]["q"]; K = len(q["criteria"]); row = {"rid": r["_meta"]["rid"], "K": K}
        if K <= 16:
            keys, p = S.prompt(r["state"], q)
            enc = tok([p], return_tensors="pt", add_special_tokens=False).to(model.device)
            torch.cuda.synchronize(); t0 = time.perf_counter(); model(**enc, logits_to_keep=1); torch.cuda.synchronize()
            row["letter_ms"] = 1000 * (time.perf_counter() - t0)
        enc = tok([num_prompt(r["state"], q)], return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id); torch.cuda.synchronize()
        row["num_ms"] = 1000 * (time.perf_counter() - t0); row["prompt_tokens"] = int(enc["input_ids"].shape[1])
        f.write(json.dumps(row) + "\n"); f.flush()
'''
AGG = r'''
import json, sys
from pathlib import Path
work, out, dms, llms, warm = Path(sys.argv[1]), Path(sys.argv[2]), [x for x in sys.argv[3].split(",") if x], [x for x in sys.argv[4].split(",") if x], int(sys.argv[5])
M = json.loads((work / "meta.json").read_text())
recs = [json.loads(l) for l in open(work / "records.jsonl") if l.strip()]
rids = [r["_meta"]["rid"] for r in recs]; conds = sorted({c for r in recs for c in r["questions"]})
res = {"rids": rids, "src": [M["meta"][x]["src"] for x in rids], "conditions": conds,
       "K": {c: [M["meta"][x]["conds"].get(c, {}).get("K") for x in rids] for c in conds},
       "dcos": {c: [M["meta"][x]["conds"].get(c, {}).get("dcos") for x in rids] for c in conds},
       "gen": M["gen"], "pool": M["pool"], "models": {}, "latency": {}}
def bits(get, fn):
    o = {}
    for c in conds:
        s = []
        for r in recs:
            q = r["questions"].get(c); v = get(r["_meta"]["rid"], c) if q is not None else None
            s.append("-" if v is None else fn(v, q))
        o[c] = "".join(s)
    return o
def confs(get, top):
    return {c: [int(round(100 * top(get(r["_meta"]["rid"], c)))) if (c in r["questions"] and get(r["_meta"]["rid"], c) is not None) else -1 for r in recs] for c in conds}
for m in dms:
    f = work / f"{m}.jsonl"
    if not f.exists(): continue
    rows = {}
    for l in open(f):
        try: d = json.loads(l); rows[d["rid"]] = d
        except Exception: pass
    get = lambda rid, c, rows=rows: rows.get(rid, {}).get("p", {}).get(c)
    res["models"][m] = {"kind": "decision", "correct": bits(get, lambda p, q: "1" if max(p, key=p.get) == q["label"] else "0"),
                        "conf": confs(get, lambda p: max(p.values()))}
    lf = work / f"lat_{m}.jsonl"
    if lf.exists():
        lat = [json.loads(l) for l in open(lf) if l.strip()]
        res["latency"][m] = [[int(d["rid"].rsplit("/", 1)[1]), round(d["ms"], 2)] for d in lat[warm:]]
for m in llms:
    for mode in ("num", "letter"):
        f = work / f"{m}_{mode}.jsonl"
        if not f.exists(): continue
        rows = {}
        for l in open(f):
            try: d = json.loads(l); rows[d["key"]] = d
            except Exception: pass
        get = lambda rid, c, rows=rows: rows.get(f"{rid}|{c}")
        if mode == "num":
            res["models"][f"{m}_num"] = {"kind": "llm_numbered", "correct": bits(get, lambda d, q: "1" if d["pick"] == q["label"] else ("x" if d["pick"] is None else "0"))}
        else:
            res["models"][f"{m}_letter"] = {"kind": "llm_letter", "correct": bits(get, lambda d, q: "1" if max(d["p"], key=d["p"].get) == q["label"] else "0"),
                                             "conf": confs(get, lambda d: max(d["p"].values()))}
    lf = work / f"lat_{m}.jsonl"
    if lf.exists():
        lat = [json.loads(l) for l in open(lf) if l.strip()][warm:]
        res["latency"][f"{m}_letter"] = [[d["K"], round(d["letter_ms"], 2)] for d in lat if "letter_ms" in d]
        res["latency"][f"{m}_num"] = [[d["K"], round(d["num_ms"], 2), d["prompt_tokens"]] for d in lat]
(out / "answer_space3.json").write_text(json.dumps(res, separators=(",", ":")))
print("bytes", (out / "answer_space3.json").stat().st_size)
'''


def lane_env(devs):
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs)}
    if len(devs) == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    return env


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save_status()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as e: status["phases"][name] = {"state": "failed", "error": str(e)[-1800:], "minutes": round((time.time() - t0) / 60, 1)}
    save_status()
    return status["phases"][name]["state"] == "ok"


def parallel(*jobs):
    ts = [threading.Thread(target=phase, args=j) for j in jobs]
    for t in ts: t.start()
    for t in ts: t.join()


def main():
    WORK.mkdir(parents=True, exist_ok=True); CODE.mkdir(parents=True, exist_ok=True)
    for name, text in BUNDLE.items(): (CODE / Path(name).name).write_text(text)
    patch = CODE / "kev_multigpu.patch"
    if subprocess.run(["git", "-C", KEV_DIR, "apply", "--check", "--reverse", patch], capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", KEV_DIR, "apply", patch], check=True)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    status["gpus"] = devs; save_status()
    if len(devs) < 4: raise SystemExit("needs 4 GPUs")
    dms = {"v3_27": (ckpt("v3f-kev-27b-dp"), 2), "stock27": ("jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195", 2),
           "v3_9": (ckpt("v3f-kev-9b-dp", "v3g-kev-9b"), 1), "stock9": ("jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c", 1)}
    dms = {k: v for k, v in dms.items() if v[0]}; status["decision_models"] = {k: v[0] for k, v in dms.items()}
    cfg = json.dumps(CFG)
    if not (WORK / "records.jsonl").exists():
        if not phase("select", lambda: run_inner("select", SELECT, [TEST, RADIOQA, WORK, json.dumps(CFG["n_questions"])], ENV, WORK / "select.log")): return
        nq = sum(1 for _ in open(WORK / "questions.jsonl"))
        if not (WORK / "gen.jsonl").exists() or sum(1 for _ in open(WORK / "gen.jsonl")) < nq:
            if not phase("generate", lambda: run_inner("generate", GEN, [GENLLM, WORK / "questions.jsonl", WORK / "gen.jsonl", CFG["n_alt"], CODE],
                                                       {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[0:2])}, WORK / "gen.log")): return
        if not phase("build", lambda: run_inner("build", BUILD, [WORK, DATA, *EMBED, cfg, RADIOQA], {**ENV, "CUDA_VISIBLE_DEVICES": devs[0]}, WORK / "build.log")): return
    try: status["build"] = json.loads((WORK / "build.log").read_text().strip().splitlines()[-1])
    except Exception: pass
    save_status()
    rec, latrec = WORK / "records.jsonl", WORK / "latency_records.jsonl"
    kev = lambda m, d, inp, out: (lambda: run_inner(f"{m}:{Path(out).name}", KEV, [dms[m][0], inp, out], lane_env(d), WORK / f"{Path(out).stem}.log"))
    llm = lambda m, d, mode: (lambda: run_inner(f"{m}:{mode}", LLMNUM, [LLMS[m][0], rec, WORK / f"{m}_{mode}.jsonl", mode, CODE, LLMS[m][1]],
                                                lane_env(d), WORK / f"{m}_{mode}.log"))
    big = [k for k in ("v3_27", "stock27") if k in dms]; small = [k for k in ("v3_9", "stock9") if k in dms]
    parallel(*[(f"acc_{m}", kev(m, devs[0:2] if i == 0 else devs[2:4], rec, WORK / f"{m}.jsonl")) for i, m in enumerate(big)])
    parallel(*[(f"acc_{m}", kev(m, devs[i:i + 1], rec, WORK / f"{m}.jsonl")) for i, m in enumerate(small)])
    parallel(("num_qwen38", llm("qwen38", devs[0:2], "num")), ("num_medgemma_fix", llm("medgemma_fix", devs[2:4], "num")))
    parallel(("letter_qwen38", llm("qwen38", devs[0:2], "letter")), ("letter_medgemma_fix", llm("medgemma_fix", devs[2:4], "letter")))
    for m in big: phase(f"lat_{m}", kev(m, devs[0:2], latrec, WORK / f"lat_{m}.jsonl"))   # latency: one system at a time
    for m in small: phase(f"lat_{m}", kev(m, devs[0:1], latrec, WORK / f"lat_{m}.jsonl"))
    for m in LLMS:
        phase(f"lat_{m}", lambda m=m: run_inner(f"lat_{m}", LLMLAT, [LLMS[m][0], latrec, WORK / f"lat_{m}.jsonl", CODE, LLMS[m][1]],
                                                lane_env(devs[0:2]), WORK / f"lat_{m}.log"))
    phase("aggregate", lambda: run_inner("aggregate", AGG, [WORK, OUT, ",".join(dms), ",".join(LLMS), CFG["lat_warmup"]], ENV, WORK / "aggregate.log"))
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save_status()


if __name__ == "__main__":
    main()
