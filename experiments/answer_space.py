"""Answer-space study (post hoc; no training): how the set of answers offered with a question changes the accuracy, confidence
and latency of RadKev and Kev, with LLM-generated distractors added to the human answer key.

    python experiments/answer_space.py              # four GPUs (two pairs) or two; needs $RADKEV_HOME/runs/test-final/test.jsonl

Questions (held-out test split; the key is human-assigned and never changes, only the options offered with it):
  eurorad_dx  all Eurorad diagnosis questions (the case's own differential; case authors' diagnosis)
  medqa       a seeded sample of CFG["medqa_n"] MedQA questions (four options; examination key)
Distractor sources:
  pool   every distinct option text of the same source in data/rad-open train/dev/test
  llm    Qwen3.8-27B, reasoning disabled, given the case, the question and the correct answer, lists CFG["n_alt"] plausible but
         incorrect answers (greedy decoding). The correct answer is never taken from the LLM.
A candidate distractor is dropped if its normalized text contains, or is contained in, the key's or an already chosen
distractor's, or if its PubMedBERT cosine (NeuML/pubmedbert-base-embeddings, pinned revision) to either is >= CFG["sim_dup"].
Conditions (each is its own question on the same state; Kev's branch mask keeps the answers independent):
  orig        the options exactly as scored in the main evaluation
  rand_K      key + K-1 random pool distractors            K in CFG["sizes"]
  sim_K       key + the K-1 pool distractors most similar to the key
  llm_K       key + the first K-1 LLM distractors           K in CFG["llm_sizes"]
  orig_llm    the original options + up to CFG["add"] LLM distractors
Models: RadKev-27B, Kev-27B, RadKev-9B and Kev-9B (stock Kev at the revisions scored in the paper), bf16.
Latency: RadKev-27B and RadKev-9B, one question per request, key + K-1 random pool distractors, K in CFG["lat_sizes"],
CFG["lat_cases"] Eurorad cases, shuffled order, batch 1, CUDA synchronized, the first CFG["lat_warmup"] requests excluded.
experiments/answer_space_llm.py scores the two LLMs on the same answer spaces (at most 16 options).
Writes $RADKEV_HOME/runs/answer_space/answer_space2.json: per-question correctness and confidence per model and condition (no
text), distractor-to-key similarities, latencies, filter counts and two example Eurorad answer spaces (public cases).
Intermediate files (questions, generated distractors, built records) stay under runs/answer_space/.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from radkev.paths import DATA, KEV_9B_PINNED, KEV_27B_PINNED, KEV_PY, QWEN38_27B, RUNS, resolve_run

WORK = RUNS / "answer_space"
TEST = RUNS / "test-final" / "test.jsonl"
OPEN = DATA / "rad-open"
EMBED = ("NeuML/pubmedbert-base-embeddings", "b79526d6ef3645e0df4530322e266f24c829f5ef")
LLM = QWEN38_27B
KEV_MODELS = {"radkev27": ("v2mg-kev-27b", 2), "kev27": (KEV_27B_PINNED, 2), "radkev9": ("v2x9-kev-9b", 1), "kev9": (KEV_9B_PINNED, 1)}
CFG = {"sizes": [2, 4, 8, 16, 32, 64], "llm_sizes": [2, 4, 8, 16], "add": 8, "n_alt": 15, "medqa_n": 200,
       "lat_sizes": [2, 4, 8, 16, 32, 64, 128], "lat_cases": 60, "lat_warmup": 10, "sim_dup": 0.95}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "cfg": CFG, "phases": {}}
LOCK = threading.Lock()
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16", "TOKENIZERS_PARALLELISM": "false"}


def save_status():
    with LOCK: (WORK / "answer_space_status.json").write_text(json.dumps(status, indent=1))


def run_inner(name, code, args, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {name}\n"); f.flush()
        rc = subprocess.run([str(KEV_PY), "-c", code, *map(str, args)], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"{name} exited {rc}: " + "".join(Path(log).read_text().splitlines(True)[-15:])[-1800:])


COMMON = r'''
import hashlib, json, random, re, sys
from pathlib import Path
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def seed(*a): return int(hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()[:12], 16)
def text(crit, k): v = crit[k]; return str(k if v is None else v).strip()
def contains(a, b): return len(a) >= 4 and len(b) >= 4 and (a in b or b in a)
'''

# ---------------------------------------------------------------- 1. select questions (CPU)
SELECT = COMMON + r'''
test_path, work, cfg = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3])
qs = []
for n, line in enumerate(open(test_path)):
    if not line.strip(): continue
    r = json.loads(line)
    for qid, q in r["questions"].items():
        if q.get("src") in ("eurorad_dx", "medqa") and q["type"] == "choice":
            qs.append({"rid": f"{q['src']}/{n}/{qid}", "src": q["src"], "state": r["state"], "q": q,
                       "options": [text(q["criteria"], k) for k in q["criteria"]], "key": text(q["criteria"], q["label"])})
med = sorted([t for t in qs if t["src"] == "medqa"], key=lambda t: seed("medqa-sample", t["rid"]))[:cfg["medqa_n"]]
sel = [t for t in qs if t["src"] == "eurorad_dx"] + med
with open(work / "questions.jsonl", "w") as f:
    for t in sel: f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(json.dumps({"eurorad_dx": sum(t["src"] == "eurorad_dx" for t in sel), "medqa": len(med)}))
'''

# ---------------------------------------------------------------- 2. LLM distractors (Qwen3.8-27B, two cards)
GEN = r'''
import json, sys, torch
from radkev import teacher as te
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

# ---------------------------------------------------------------- 3. build answer spaces (one card for embeddings)
BUILD = COMMON + r'''
import os
import numpy as np
work, data_dir, embed_repo, embed_rev, cfg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], json.loads(sys.argv[5])
DUP = cfg["sim_dup"]
qs = [json.loads(l) for l in open(work / "questions.jsonl") if l.strip()]
raw = {json.loads(l)["rid"]: json.loads(l)["raw"] for l in open(work / "gen.jsonl") if l.strip()}
pools = {s: {} for s in ("eurorad_dx", "medqa")}
for split in ("train", "dev", "test"):
    for line in open(data_dir / f"{split}.jsonl"):
        if not line.strip(): continue
        for q in json.loads(line)["questions"].values():
            if q.get("src") in pools and q["type"] == "choice":
                for k in q["criteria"]:
                    t = text(q["criteria"], k); n_ = norm(t)
                    if n_ and n_ not in pools[q["src"]]: pools[q["src"]][n_] = t
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
    if keep: q = dict(t["q"]); q["src"] = f"as2_{t['src']}_{cond}"; return q
    rng = random.Random(seed(t["rid"], cond)); xs = list(texts); rng.shuffle(xs); kn = norm(t["key"])
    crit = {f"opt_{j + 1}": x for j, x in enumerate(xs)}
    label = next(k for k, x in crit.items() if norm(x) == kn)
    return {"type": "choice", "instructions": t["q"]["instructions"], "criteria": crit, "label": label, "src": f"as2_{t['src']}_{cond}"}
recs, meta, gstats, examples = [], {}, {"parsed": [], "kept": [], "dup_key": 0, "dup_other": 0}, []
for t in qs:
    s, kn = t["src"], norm(t["key"]); kv = embed([t["key"]])[0]
    dis = [o for o in t["options"] if norm(o) != kn]
    sim_order = np.argsort(-(E[s] @ kv), kind="stable").tolist()
    rnd_order = list(range(len(P[s]))); random.Random(seed(t["rid"], "rand")).shuffle(rnd_order)
    need = max(cfg["sizes"]) - 1
    sim_d, _ = choose([P[s][i] for i in sim_order[:4000]], [E[s][i] for i in sim_order[:4000]], kv, kn, [], need)
    rnd_d, _ = choose([P[s][i] for i in rnd_order[:4000]], [E[s][i] for i in rnd_order[:4000]], kv, kn, [], need)
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

# ---------------------------------------------------------------- 4. Kev scoring (resumable)
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

# ---------------------------------------------------------------- 5. compact outputs (no text except the two examples)
AGG = r'''
import json, sys
from pathlib import Path
work, out, models, warm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
M = json.loads((work / "meta.json").read_text())
recs = [json.loads(l) for l in open(work / "records.jsonl") if l.strip()]
rids = [r["_meta"]["rid"] for r in recs]; conds = sorted({c for r in recs for c in r["questions"]})
res = {"rids": rids, "src": [M["meta"][x]["src"] for x in rids], "conditions": conds,
       "K": {c: [M["meta"][x]["conds"].get(c, {}).get("K") for x in rids] for c in conds},
       "dcos": {c: [M["meta"][x]["conds"].get(c, {}).get("dcos") for x in rids] for c in conds},
       "gen": M["gen"], "examples": M["examples"], "pool": M["pool"], "models": {}, "latency": {}}
for m in models:
    f = work / f"{m}.jsonl"
    if not f.exists(): continue
    rows = {}
    for l in open(f):
        try: d = json.loads(l); rows[d["rid"]] = d
        except Exception: pass
    out_m = {"correct": {}, "conf": {}, "ms_per_record": [round(rows[x]["ms"], 1) if x in rows else None for x in rids]}
    for c in conds:
        cs, ps = [], []
        for r in recs:
            q = r["questions"].get(c); row = rows.get(r["_meta"]["rid"])
            if q is None or row is None or c not in row["p"]: cs.append("-"); ps.append(-1); continue
            p = row["p"][c]; top = max(p, key=p.get)
            cs.append("1" if top == q["label"] else "0"); ps.append(int(round(100 * p[top])))
        out_m["correct"][c] = "".join(cs); out_m["conf"][c] = ps
    res["models"][m] = out_m
    lf = work / f"lat_{m}.jsonl"
    if lf.exists():
        lat = [json.loads(l) for l in open(lf) if l.strip()]
        res["latency"][m] = [[int(d["rid"].rsplit("/", 1)[1]), round(d["ms"], 2)] for d in lat[warm:]]
(out / "answer_space2.json").write_text(json.dumps(res, separators=(",", ":")))
print("bytes", (out / "answer_space2.json").stat().st_size)
'''


def lane_env(devs, cards):
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[:cards])}
    if cards == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    return env


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save_status()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as e: status["phases"][name] = {"state": "failed", "error": str(e)[-1800:], "minutes": round((time.time() - t0) / 60, 1)}
    save_status()
    return status["phases"][name]["state"] == "ok"


def kev(m, d, inp, out):
    target, cards = KEV_MODELS[m]
    return lambda: run_inner(f"{m}:{out.name}", KEV, [resolve_run(target), inp, out], lane_env(d, cards), WORK / f"{out.stem}.log")


def parallel(*jobs):
    ts = [threading.Thread(target=phase, args=j) for j in jobs]
    for t in ts: t.start()
    for t in ts: t.join()


def gpu_pairs():
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    if len(devs) < 2: raise SystemExit("needs at least two GPUs (one pair)")
    return [devs[i:i + 2] for i in range(0, len(devs) - 1, 2)][:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="", help="directory for answer_space2.json (default: $RADKEV_HOME/runs/answer_space/)")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else WORK; out.mkdir(parents=True, exist_ok=True)
    pairs = gpu_pairs(); p0 = pairs[0]; p1 = pairs[-1]
    status["gpu_pairs"] = pairs; save_status()
    cfg = json.dumps(CFG)
    if not phase("select", lambda: run_inner("select", SELECT, [TEST, WORK, cfg], ENV, WORK / "select.log")): return
    gen_env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(p0)}
    if not phase("generate", lambda: run_inner("generate", GEN, [LLM, WORK / "questions.jsonl", WORK / "gen.jsonl", CFG["n_alt"]], gen_env, WORK / "gen.log")): return
    if not phase("build", lambda: run_inner("build", BUILD, [WORK, OPEN, *EMBED, cfg], {**ENV, "CUDA_VISIBLE_DEVICES": p0[0]}, WORK / "build.log")): return
    try: status["build"] = json.loads((WORK / "build.log").read_text().strip().splitlines()[-1])
    except Exception: pass
    save_status()
    rec, latrec = WORK / "records.jsonl", WORK / "latency_records.jsonl"
    together = len(pairs) > 1   # two GPU pairs: the two models of a size run side by side; one pair: one after the other
    for a_, b_ in (("radkev27", "kev27"), ("radkev9", "kev9")):
        ja = (f"acc_{a_}", kev(a_, p0, rec, WORK / f"{a_}.jsonl")); jb = (f"acc_{b_}", kev(b_, p1, rec, WORK / f"{b_}.jsonl"))
        if together: parallel(ja, jb)
        else: phase(*ja); phase(*jb)
    phase("lat_radkev27", kev("radkev27", p0, latrec, WORK / "lat_radkev27.jsonl"))   # alone on the machine
    phase("lat_radkev9", kev("radkev9", p0, latrec, WORK / "lat_radkev9.jsonl"))
    phase("aggregate", lambda: run_inner("aggregate", AGG, [WORK, out, ",".join(KEV_MODELS), CFG["lat_warmup"]], ENV, WORK / "aggregate.log"))
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save_status()
    print(out / "answer_space2.json")


if __name__ == "__main__":
    main()
