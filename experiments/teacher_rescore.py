"""Two-teacher labels regenerated with MedGemma's reasoning segment closed (post hoc; no training of any kind), and the decision
models' agreement with the regenerated teacher-labeled test questions.

    python experiments/teacher_rescore.py                 # the regenerated labels the paper reports; two GPU pairs, or one in turn
    python experiments/teacher_rescore.py --single-bos    # the same with MedGemma's prompt tokenized with a single <bos>

The first-run MedGemma votes ($RADKEV_HOME/teacher/medgemma.jsonl, experiments/teacher_labels.py) were read at the first
generated token, which for MedGemma-27B-text-it is the start of a thought its chat template does not suppress. Here every task in
$RADKEV_HOME/teacher/tasks.jsonl is rescored with radkev.teacher.LetterScorer(think_off=True) (the one-line THOUGHT_OFF pre-fill,
as in the MedGemma rows of the main evaluation); Qwen3.8-27B's votes (teacher/qwen38.jsonl) are reused unchanged.
1. medgemma: the tasks sharded by task id parity over the GPU pairs (device_map auto), batch 16 halving on out-of-memory, each
   shard appended to teacher/<votes>.part{0,1}.jsonl (finished task ids are skipped on rerun), then concatenated into
   teacher/<votes>.jsonl once every task has a vote (<votes> = medgemma_brief, or medgemma_bos1 with --single-bos).
2. merge: radkev.teacher merge (unchanged rule: keep a question only if both teachers' argmax agree; label = argmax, target =
   mean distribution) of the new MedGemma votes and the Qwen votes into a NEW directory, data/teacher_v2 (data/teacher_v3).
3. changes: per split and question kind, against the first-run data/teacher (questions keyed by split, group, state and kind):
   same label, different label, dropped (kept before, now disagreeing), added (newly agreed); and how often MedGemma's argmax
   changed. Counts only.
4. kev: the new teacher test split scored with radkev.evaluate as in experiments/final_test.py (bf16, each checkpoint's own
   temperature): RadKev-27B and Kev-27B, then RadKev-9B and Kev-9B (stock Kev at the revisions scored in the paper).
5. paired: RadKev minus Kev on the regenerated test questions, overall and per task, with 95% intervals from 2,000 record
   resamples (seed 20261002).
Never overwrites teacher/medgemma.jsonl, teacher/qwen38.jsonl or data/teacher. Writes aggregates only (no text):
$RADKEV_HOME/runs/teacher_rescore[_bos1]/teacher_rescore.json and teacher_rescore_status.json.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from radkev.paths import DATA, HOME, KEV_9B_PINNED, KEV_27B_PINNED, KEV_PY, MEDGEMMA_27B, RUNS, resolve_run

TEACH = HOME / "teacher"
TASKS, QWEN, MG_OLD = TEACH / "tasks.jsonl", TEACH / "qwen38.jsonl", TEACH / "medgemma.jsonl"
OLD_DATA = DATA / "teacher"
BATCH = 16
KEV_MODELS = {"radkev27": ("v2mg-kev-27b", 2), "kev27": (KEV_27B_PINNED, 2), "radkev9": ("v2x9-kev-9b", 1), "kev9": (KEV_9B_PINNED, 1)}
SPLITS = ("train", "dev", "test")
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "KEV_DTYPE": "bf16", "TOKENIZERS_PARALLELISM": "false"}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}
result = {"what": "two-teacher labels with MedGemma think_off (THOUGHT_OFF pre-fill) + reused Qwen3.8 votes; counts and accuracies only"}
LOCK = threading.Lock()
P = {}   # paths of this variant, set in main()


def save_status():
    with LOCK: (P["work"] / "teacher_rescore_status.json").write_text(json.dumps(status, indent=1))


def save_result():
    with LOCK: (P["work"] / "teacher_rescore.json").write_text(json.dumps(result, indent=1))


def run_proc(name, cmd, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {name}\n"); f.flush()
        rc = subprocess.run([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc: raise RuntimeError(f"{name} exited {rc}: " + "".join(Path(log).read_text().splitlines(True)[-15:])[-1800:])


def last_json(log):
    for l in Path(log).read_text().splitlines()[::-1]:
        if l.startswith("{"):
            try: return json.loads(l)
            except Exception: pass
    return {}


def phase(name, fn):
    t0 = time.time()
    with LOCK: status["phases"][name] = {"state": "running"}
    save_status()
    try: res = fn(); st = {"state": "ok", **(res if isinstance(res, dict) else {})}
    except Exception as e: st = {"state": "failed", "error": str(e)[-1800:]}
    st["minutes"] = round((time.time() - t0) / 60, 1)
    with LOCK: status["phases"][name] = st
    save_status()
    return st["state"] == "ok"


def parallel(*jobs):
    ts = [threading.Thread(target=phase, args=j) for j in jobs]
    for t in ts: t.start()
    for t in ts: t.join()
    return all(status["phases"][j[0]]["state"] == "ok" for j in jobs)


def read_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- 1. MedGemma votes with the thought closed (one process per GPU pair)
SCORE = r'''
import gc, json, sys
from pathlib import Path
from radkev import teacher as te
model_id, tasks_path, out, shard, n_shard, batch, single_bos = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7] == "1"
mine = [t for t in (json.loads(l) for l in open(tasks_path) if l.strip()) if t["tid"] % n_shard == shard]
done = set()
if out.exists():
    data = out.read_bytes()
    if data and not data.endswith(b"\n"):          # a killed run can leave half a line; drop it before appending
        with open(out, "r+b") as f: f.truncate(data.rfind(b"\n") + 1)
    for l in open(out):
        if l.strip(): done.add(json.loads(l)["tid"])
todo = [t for t in mine if t["tid"] not in done]
info = {"shard": shard, "tasks": len(mine), "done_before": len(mine) - len(todo), "oom": 0, "batch_final": batch}
if todo:
    import torch
    S = te.LetterScorer(model_id, think_off=True, single_bos=single_bos)
    if S.prefill != te.LetterScorer.THOUGHT_OFF or "letter only" not in S.prefill: raise SystemExit("THOUGHT_OFF pre-fill is not active")
    items = sorted(((t["tid"], t["state"], t["question"]) for t in todo), key=lambda x: len(S.prompt(x[1], x[2])[1]))
    i, b = 0, batch
    with open(out, "a") as f:
        while i < len(items):
            chunk, oom = items[i:i + b], False
            try: res = list(S.score(chunk, batch=len(chunk), log=f"shard{shard} {i}+"))
            except torch.cuda.OutOfMemoryError: oom = True
            if oom:
                res = None; gc.collect(); torch.cuda.empty_cache(); info["oom"] += 1
                if b == 1: raise SystemExit("out of memory at batch 1")
                b = max(1, b // 2); continue
            for tid, p in res: f.write(json.dumps({"tid": tid, "p": p}) + "\n")
            f.flush(); i += len(chunk)
    info["batch_final"] = b
info["done_after"] = len(done) + len(todo)
print(json.dumps(info), flush=True)
'''


def score_shard(shard, n_shard, devs, single_bos):
    def fn():
        env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs)}
        log = P["work"] / f"medgemma_part{shard}.log"
        run_proc(f"medgemma shard {shard}", [KEV_PY, "-c", SCORE, MEDGEMMA_27B, TASKS, TEACH / f"{P['votes']}.part{shard}.jsonl", shard, n_shard, BATCH,
                                             int(single_bos)], env, log)
        return last_json(log)
    return fn


def concat(n_shard):
    tids = {t["tid"] for t in read_jsonl(TASKS)}
    rows = {}
    for s in range(n_shard):
        for r in read_jsonl(TEACH / f"{P['votes']}.part{s}.jsonl"): rows[r["tid"]] = r
    missing = tids - set(rows)
    if missing: raise RuntimeError(f"{len(missing)} tasks have no corrected MedGemma vote")
    tmp = P["mg_new"].with_suffix(".tmp")
    with open(tmp, "w") as f:
        for tid in sorted(tids): f.write(json.dumps(rows[tid]) + "\n")
    tmp.replace(P["mg_new"])
    return {"votes": len(tids)}


# ---------------------------------------------------------------- 2-3. merge and label changes (stdlib)
def merge():
    for p in (P["mg_new"], P["new_data"]): assert p not in (MG_OLD, QWEN, OLD_DATA)
    run_proc("merge", [KEV_PY, "-m", "radkev.teacher", "merge", "--tasks", TASKS, "--labels", P["mg_new"], QWEN, "--out", P["new_data"]], ENV,
             P["work"] / "merge.log")
    result["manifest_new"] = json.loads((P["new_data"] / "manifest.json").read_text())
    if (OLD_DATA / "manifest.json").exists(): result["manifest_old"] = json.loads((OLD_DATA / "manifest.json").read_text())
    save_result()
    return {"records": result["manifest_new"]["records"]}


def question_labels(d):
    """{(split, group, state, kind): label} for every question in a merged teacher dir (the keys merge itself groups by)."""
    out = {}
    for split in SPLITS:
        for r in read_jsonl(d / f"{split}.jsonl"):
            st = json.dumps(r["state"], sort_keys=True)
            for qid, q in r["questions"].items(): out[(split, r["_meta"]["group_id"], st, qid)] = q["label"]
    return out


def changes(old_dir, new_dir):
    old, new = question_labels(old_dir), question_labels(new_dir)
    c = defaultdict(Counter)
    for k in old.keys() | new.keys():
        split, kind = k[0], k[3]
        if k in old and k in new: cat = "same_label" if old[k] == new[k] else "different_label"
        elif k in old: cat = "dropped"
        else: cat = "added"
        for key in (f"{split}/{kind}", f"{split}/ALL", f"ALL/{kind}", "ALL/ALL"):
            c[key][cat] += 1
            c[key]["kept_old"] += k in old; c[key]["kept_new"] += k in new
    cats = ("kept_old", "kept_new", "same_label", "different_label", "dropped", "added")
    res = {}
    for key in sorted(c):
        split, kind = key.split("/")
        res.setdefault(split, {})[kind] = {x: c[key][x] for x in cats}
    return res


def vote_shift(tasks_path, old_path, new_path):
    """How often the corrected MedGemma argmax equals its original argmax, per kind (counts)."""
    kind = {t["tid"]: t["qid"] for t in read_jsonl(tasks_path)}
    old = {r["tid"]: r["p"] for r in read_jsonl(old_path)}
    c = defaultdict(Counter)
    for r in read_jsonl(new_path):
        p0 = old.get(r["tid"])
        if p0 is None: continue
        same = max(p0, key=p0.get) == max(r["p"], key=r["p"].get)
        for k in (kind[r["tid"]], "ALL"): c[k]["same" if same else "different"] += 1
    return {k: dict(v) for k, v in sorted(c.items())}


def changes_phase():
    result["label_changes"] = changes(OLD_DATA, P["new_data"])
    if MG_OLD.exists(): result["medgemma_argmax_old_vs_corrected"] = vote_shift(TASKS, MG_OLD, P["mg_new"])
    save_result()
    a = result["label_changes"]["ALL"]["ALL"]
    return {"kept_old": a["kept_old"], "kept_new": a["kept_new"]}


# ---------------------------------------------------------------- 4. Kev models on the new teacher test split
def lane_env(devs, cards):
    env = {**ENV, "CUDA_VISIBLE_DEVICES": ",".join(devs[:cards])}
    if cards == 2: env.update(KEV_DEVICE_MAP="auto", KEV_MAX_MEMORY="0:26GiB,1:40GiB")
    return env


def kev(m, devs):
    def fn():
        target, cards = KEV_MODELS[m]; out = P["work"] / "eval" / m
        if not (out / "summary.json").exists():
            run_proc(f"evaluate {m}", [KEV_PY, "-m", "radkev.evaluate", "--run", resolve_run(target), "--data", P["new_data"] / "test.jsonl", "--out", out],
                     lane_env(devs, cards), P["work"] / f"eval_{m}.log")
        return {"cached_or_done": True}
    return fn


def aggregate():
    result["test_split"] = {k: v.get("kept_new") for k, v in result.get("label_changes", {}).get("test", {}).items()}
    result["models"] = {}
    for m, (target, _) in KEV_MODELS.items():
        f = P["work"] / "eval" / m / "summary.json"
        if not f.exists(): continue
        s = json.loads(f.read_text())
        result["models"][m] = {"run": target if "/" in target else "radkev:" + target, "temperature": s.get("temperature"),
                               "coverage": s.get("coverage"), "overall": {k: s["overall"].get(k) for k in ("n", "acc", "ece", "brier")},
                               "tasks": {t: {"n": v["n"], "acc": v["acc"]} for t, v in sorted(s["tasks"].items())}}
    save_result()
    return {"models": sorted(result["models"])}


# ---------------------------------------------------------------- 5. paired differences on the regenerated test questions
def paired():
    from kev.metrics import scored_rows
    ev = P["work"] / "eval"; out = {}
    rows = {m: {(r["id"], r["question"]): r for r in scored_rows(json.loads((ev / m / "rows.json").read_text()))}
            for m in ("radkev27", "kev27", "radkev9", "kev9") if (ev / m / "rows.json").exists()}
    keys = sorted(set.intersection(*[set(v) for v in rows.values()]))
    task = np.array([rows["radkev27"][k]["task"] for k in keys]); rec = np.array([k[0] for k in keys])
    recs = sorted(set(rec.tolist())); ridx = {r: i for i, r in enumerate(recs)}; q2r = np.array([ridx[r] for r in rec])
    rng = np.random.default_rng(20261002); B = 2000; W = np.zeros((B, len(recs)), np.float32)
    for b in range(B): np.add.at(W[b], rng.integers(0, len(recs), len(recs)), 1)
    QW = W[:, q2r]
    Cx = {m: np.array([int(np.argmax(rows[m][k]["p"]) == int(rows[m][k]["label"])) for k in keys], float) for m in rows}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def est(x, mk): w = QW[:, mk]; return float(x[mk].mean()), (w @ x[mk]) / np.maximum(w.sum(1), 1e-9)
    for a, b in (("radkev27", "kev27"), ("radkev9", "kev9")):
        o = {}
        for t in ["ALL"] + sorted(set(task.tolist())):
            mk = np.ones(len(keys), bool) if t == "ALL" else task == t
            pa, ba = est(Cx[a], mk); pb, bb = est(Cx[b], mk)
            o[t] = {"n": int(mk.sum()), "a": pa, "b": pb, "d": pa - pb, "ci": ci(ba - bb)}
        out[f"{a}-{b}"] = o
    result["paired"] = out; save_result()
    return {"pairs": list(out)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--single-bos", dest="single_bos", action="store_true", help="tokenize MedGemma's prompt without a second <bos> (see radkev.teacher)")
    a = ap.parse_args()
    tag = "bos1" if a.single_bos else "brief"
    P.update(work=RUNS / ("teacher_rescore_bos1" if a.single_bos else "teacher_rescore"), votes=f"medgemma_{tag}",
             mg_new=TEACH / f"medgemma_{tag}.jsonl", new_data=DATA / ("teacher_v3" if a.single_bos else "teacher_v2"))
    result["single_bos"] = a.single_bos
    P["work"].mkdir(parents=True, exist_ok=True)
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if not devs:
        import torch
        devs = [str(i) for i in range(torch.cuda.device_count())]
    pairs = [devs[i:i + 2] for i in range(0, len(devs) - 1, 2)][:2]
    if not pairs: raise SystemExit("needs at least two GPUs (one pair)")
    status["gpu_pairs"] = pairs; save_status()
    if not P["mg_new"].exists():
        n_shard = 2   # task-id parity, as in the run the paper reports; with one pair the two shards run one after the other
        jobs = [(f"medgemma_shard{s}", score_shard(s, n_shard, pairs[s % len(pairs)], a.single_bos)) for s in range(n_shard)]
        ok = parallel(*jobs) if len(pairs) > 1 else all(phase(*j) for j in jobs)
        if not ok or not phase("concat", lambda: concat(n_shard)): return
    if not phase("merge", merge): return
    phase("changes", changes_phase)
    for big, small in (("radkev27", "kev27"), ("radkev9", "kev9")):
        jobs = [(f"kev_{big}", kev(big, pairs[0])), (f"kev_{small}", kev(small, pairs[-1]))]
        if len(pairs) > 1: parallel(*jobs)
        else:
            for j in jobs: phase(*j)
    phase("aggregate", aggregate)
    phase("paired", paired)
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save_status()
    print(P["work"] / "teacher_rescore.json")
    sys.exit(0 if all(p["state"] == "ok" for p in status["phases"].values()) else 1)


if __name__ == "__main__":
    main()
