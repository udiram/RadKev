"""TypeSafe Jev (System One API, model jev-latest) on the v3 radiology benchmark (analysis-plan amendment 11). No GPU.

    python jobs/submit.py jobs/jev_decisions.py --gpus 0 --cpus 2 --mem 8 --timeout 300 --outputs 'jev_*.json' -- --phase probe
    python jobs/submit.py jobs/jev_decisions.py --gpus 0 --cpus 2 --mem 8 --timeout 300 --outputs 'jev_*.json' -- --phase all

The same protocol as jobs/openai_decisions.py (amendment 10), on the same records and questions. Every benchmark record of the
eval_v3 files (runs/test-v3/final_s0, final_s1, new.jsonl) is sent ONCE, all of its benchmark questions in one request to
POST https://api.typesafe.ai/v1/systemone. Jev takes the benchmark's own format, so the state and each question's type,
instructions and criteria are sent unchanged (no label or metadata field); CT-RATE questions and knowledge questions outside
the radiology filter are not sent. The full HTTP response (answers, usage, request-id header), the request and the end-to-end
wall time are cached per record in runs/jev/responses.jsonl on the node; a record already in the cache is never sent again, so
every phase can be rerun from the cache. Spend is counted from the API's reported input tokens ($0.042 / 1M; output tokens are
not billed) and the job stops before a request that would take the total past CAP_USD. A request rejected as invalid (HTTP 400
or 422) is retried question by question; a question that is still rejected gets a uniform distribution and is counted as
unsupported.

Phases: probe (one record per task with that task's largest option count plus one random record per task, sequential, so its
latencies are unaffected by concurrency), run (the rest, WORKERS concurrent requests), preds (kev_eval --preds into
runs/test-v3/{new,final}/jev*), latency, inspect. The paired analysis is jobs/eval_v3_prior.py, with jev added to the systems.
Publishes aggregates only: jev_status.json, jev_latency.json.
"""
import json
import os
import random
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INCLUDE = ["jobs/kev_eval.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ["XDG_CACHE_HOME"]); RADKEV = CACHE / "radkev"
KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
WORK = RADKEV / "runs/test-v3"; FINAL = RADKEV / "runs/test-final"
DIR = RADKEV / "runs/jev"; CODE = DIR / "code"; RESP = DIR / "responses.jsonl"
KEY = CACHE / "raddecision/jev_key"
URL, MODEL = "https://api.typesafe.ai/v1/systemone", "jev-latest"
USD_PER_TOKEN, CAP_USD, WORKERS, SEED = 0.042 / 1e6, 4.50, 4, 20261006
FILES = {"s0": WORK / "final_s0.jsonl", "s1": WORK / "final_s1.jsonl", "new": WORK / "new.jsonl"}
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "model": MODEL, "cap_usd": CAP_USD, "phases": {}}
LOCK = threading.Lock()


def save():
    with LOCK: (OUT / "jev_status.json").write_text(json.dumps(status, indent=1, default=str))


def phase(name, fn):
    t0 = time.time(); status["phases"][name] = {"state": "running"}; save()
    try: fn(); status["phases"][name] = {"state": "ok", "minutes": round((time.time() - t0) / 60, 1)}
    except Exception as ex: status["phases"][name] = {"state": "failed", "error": str(ex)[-2500:], "minutes": round((time.time() - t0) / 60, 1)}
    save(); return status["phases"][name]["state"] == "ok"


# ------------------------------------------------------------------------------------------------------------ plan
RADIDX = {}


def bench_task(src, rid, qid):
    """eval_v3's benchmark-task mapping (CT-RATE and non-radiology knowledge questions are None here: not sent)."""
    if src is None: return "unknown"
    if src in ("iu_finding", "iu_normal", "iu_which", "eurorad_dx", "eurorad_route", "medmcqa_rad", "rsna_radioqa", "rexerr_error"): return src
    if src.startswith("radcases_panel"): return "radcases_panel"
    if src.startswith("radcases_topic"): return "radcases_topic"
    pooled = {"medmcqa_med": "medmcqa_other_rad", "medqa": "medqa_rad", "medxpertqa": "medxpertqa_rad", "pubmedqa": "pubmedqa_rad"}
    t = pooled.get(src) or ("mmlu_rad" if src.startswith("mmlu_") else None)
    return t if t and RADIDX.get(f"{rid}|{qid}") else None


def keys(q):
    if q["type"] == "choice": return list(q["criteria"])
    if q["type"] == "noul": return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def text(x): return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def api_question(q):
    """The benchmark question as Jev takes it: type, instructions and criteria only (score levels as a list of 2 to 10)."""
    o = {"type": q["type"], "instructions": q.get("instructions") or ""}
    crit = q.get("criteria")
    if q["type"] == "score" and isinstance(crit, dict): crit = [f"{k}: {text(v)}" for k, v in crit.items()]
    if crit: o["criteria"] = crit
    return o


def plan():
    RADIDX.update(json.loads((FINAL / "radiology_index.json").read_text()))
    items, skipped = [], {}
    for f, p in FILES.items():
        for l in open(p):
            if not l.strip(): continue
            r = json.loads(l); rid = r["_meta"]["id"]; qs = {}
            for qid, q in r["questions"].items():
                t = bench_task(q.get("src"), rid, qid)
                if t is None: skipped[q.get("src") or "?"] = skipped.get(q.get("src") or "?", 0) + 1; continue
                qs[qid] = (t, q)
            if not qs: continue
            names = {qid: f"q{i}" for i, qid in enumerate(qs)}
            body = {"model": MODEL, "state": r["state"], "questions": {names[qid]: api_question(q) for qid, (t, q) in qs.items()}}
            items.append({"id": rid, "file": f, "tasks": {qid: t for qid, (t, q) in qs.items()}, "names": names,
                          "nopt": max(len(keys(q)) for t, q in qs.values()), "body": body})
    est = sum(200 + len(json.dumps(it["body"], ensure_ascii=False)) / 3 for it in items)
    by = {}
    for it in items:
        for t in it["tasks"].values(): by[t] = by.get(t, 0) + 1
    status["plan"] = {"records": len(items), "questions": sum(by.values()), "by_task": by, "not_sent": skipped,
                      "est_tokens": int(est), "est_usd": round(est * USD_PER_TOKEN, 3), "max_options": max(it["nopt"] for it in items)}
    save()
    if est * USD_PER_TOKEN > CAP_USD: raise SystemExit(f"estimated ${est * USD_PER_TOKEN:.2f} exceeds the cap")
    return items


# ------------------------------------------------------------------------------------------------------------ send
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt") if Path("/etc/ssl/certs/ca-certificates.crt").exists() else None
HDRS = ("x-typesafe-request-id", "retry-after", "cf-ray")
CACHED, SPENT, PLANNED, REFUSED = {}, {"tokens": 0}, {}, []
STOP = threading.Event()


def load_cache():
    if not RESP.exists(): return
    for l in open(RESP):   # every line counts toward spend, including attempts that were not final
        if not l.strip(): continue
        e = json.loads(l); SPENT["tokens"] += e.get("billed_tokens", 0)
        if e.get("final"): CACHED[e["id"]] = e


def post(body):
    """One HTTP attempt: (status, response json or error text, wall ms, headers)."""
    req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Authorization": "Bearer " + KEY.read_text().strip(), "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
            raw = r.read(); ms = (time.perf_counter() - t0) * 1000
            return r.status, json.loads(raw), ms, {h: r.headers.get(h) for h in HDRS}
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - t0) * 1000
        return e.code, e.read().decode(errors="replace")[:2000], ms, {h: e.headers.get(h) for h in HDRS}
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:2000], (time.perf_counter() - t0) * 1000, {}


def call(body, attempts):
    """POST with retries on 429 / 5xx / network errors; returns the last attempt."""
    for k in range(6):
        est = 200 + len(json.dumps(body, ensure_ascii=False)) / 3
        with LOCK:
            if (SPENT["tokens"] + est) * USD_PER_TOKEN > CAP_USD: STOP.set()
        if STOP.is_set(): return None
        code, resp, ms, h = post(body)
        tok = resp.get("usage", {}).get("input_tokens", 0) if isinstance(resp, dict) else 0
        with LOCK: SPENT["tokens"] += tok
        attempts.append({"status": code, "ms": round(ms, 1), "headers": h, "tokens": tok, **({} if code == 200 else {"error": resp})})
        if code == 200 or (code is not None and 400 <= code < 500 and code != 429): return code, resp, ms, h
        time.sleep(2 ** (k + 1))
    return code, resp, ms, h


def send(it):
    if it["id"] in CACHED or STOP.is_set(): return
    attempts, entry = [], {"id": it["id"], "file": it["file"], "names": it["names"], "tasks": it["tasks"], "request": it["body"],
                           "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "workers": it.get("_workers", WORKERS)}
    r = call(it["body"], attempts)
    if r is None: return
    code, resp, ms, h = r
    entry["attempts"] = attempts
    if code == 200:
        entry.update(final=True, ok=True, response=resp, latency_ms=ms, retried=len(attempts) > 1)
    elif code in (400, 422):   # split into one request per question; a question still rejected is unsupported
        parts = []
        for nm, qa in it["body"]["questions"].items():
            pa = []; rr = call({**it["body"], "questions": {nm: qa}}, pa)
            if rr is None: return
            parts.append({"name": nm, "status": rr[0], "response": rr[1] if rr[0] == 200 else None, "attempts": pa})
        entry.update(final=all(p["status"] in (200, 400, 422) for p in parts), ok=False, split=True, parts=parts)
    else:
        entry.update(final=False, ok=False)   # transient failure exhausted: tried again on the next run
    entry["billed_tokens"] = sum(a["tokens"] for a in attempts) + sum(a["tokens"] for p in entry.get("parts", []) for a in p["attempts"])
    with LOCK:
        with open(RESP, "a") as f: f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if entry["final"]: CACHED[it["id"]] = entry
        n = len(CACHED)
        status["progress"] = {"records_done": n, "spent_tokens": SPENT["tokens"], "spent_usd": round(SPENT["tokens"] * USD_PER_TOKEN, 4), "at": time.strftime("%H:%M:%S")}
    if n % 200 == 0: save()


def probe_items(items):
    rng = random.Random(SEED); per = {}
    for it in items:
        for t in set(it["tasks"].values()): per.setdefault(t, []).append(it)
    pick = {}
    for t, its in sorted(per.items()):
        big = max(its, key=lambda x: x["nopt"]); pick[big["id"]] = big
        r = rng.choice(its); pick[r["id"]] = r
    return list(pick.values())


def run_probe(items):
    ps = probe_items(items)
    for it in ps: it["_workers"] = 1; send(it)
    summ = []
    for it in ps:
        e = CACHED.get(it["id"])
        summ.append({"tasks": sorted(set(it["tasks"].values())), "options": it["nopt"], "questions": len(it["tasks"]),
                     "ok": bool(e and e.get("ok")), "split": bool(e and e.get("split")), "latency_ms": e and e.get("latency_ms"),
                     "tokens": e and e.get("billed_tokens"), "model": e and (e.get("response") or {}).get("model"),
                     "errors": [a.get("error") for a in (e or {}).get("attempts", []) if a.get("error")][:2]})
    status["probe"] = summ; save()


def run_all(items):
    rest = [it for it in items if it["id"] not in CACHED]; random.Random(SEED).shuffle(rest)
    with ThreadPoolExecutor(WORKERS) as ex: list(ex.map(send, rest))
    status["stopped_by_cap"] = STOP.is_set(); save()


# ------------------------------------------------------------------------------------------------------------ preds + analyse
def probs_of(entry, qid, q):
    ks = keys(q); name = entry["names"].get(qid)
    ans = {}
    if entry.get("ok"): ans.update(entry["response"].get("answers", {}))
    for p in entry.get("parts", []):
        if p["status"] == 200: ans.update(p["response"].get("answers", {}))
    a = ans.get(name)
    if a is None: return None
    if a.get("type") not in ("noul", "choice", "score"): REFUSED.append((entry["id"], qid)); return None   # no distribution: uniform, reported
    if a["type"] == "noul": p = float(a["noul"]); d = {"false": 1 - p, "true": p}
    else: d = {str(k): float(v) for k, v in a["probabilities"].items()}
    d = {k: max(0.0, d.get(k, 0.0)) for k in ks}; s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else None


def preds():
    unsupported, missing = {}, 0
    env = {**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    for f, p in FILES.items():
        pf = DIR / f"preds_{f}.jsonl"
        with open(pf, "w") as g:
            for l in open(p):
                if not l.strip(): continue
                r = json.loads(l); rid = r["_meta"]["id"]; e = CACHED.get(rid); out = {}
                for qid, q in r["questions"].items():
                    d = probs_of(e, qid, q) if e else None
                    if d is None:
                        d = {k: 1 / len(keys(q)) for k in keys(q)}
                        if e and qid in e["tasks"]: unsupported[e["tasks"][qid]] = unsupported.get(e["tasks"][qid], 0) + 1
                        elif not e and qid in PLANNED.get(rid, {}): missing += 1
                    out[qid] = d
                g.write(json.dumps({"id": rid, "probabilities": out}) + "\n")
        dest = WORK / "new" / "jev" if f == "new" else WORK / "final" / f"jev_{f}"
        with open(DIR / "kev_eval.log", "a") as log:
            rc = subprocess.run([str(KEV_PY), str(CODE / "kev_eval.py"), "--preds", str(pf), "--data", str(p), "--out", str(dest)],
                                env=env, cwd=KEV_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc: raise RuntimeError(f"kev_eval {f} exit {rc}: " + (DIR / "kev_eval.log").read_text()[-1500:])
    status["unsupported_questions"] = unsupported; status["records_not_in_cache"] = missing
    status["refusals"] = {"questions": len(REFUSED), "records": len({r for r, q in REFUSED}),
                          "by_task": {t: sum(1 for r, q in REFUSED if PLANNED[r][q] == t) for t in {PLANNED[r][q] for r, q in REFUSED}}}; save()


def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else None


def latency():
    def summ(v): return {"n": len(v), "median": pct(v, 50), "p90": pct(v, 90), "p95": pct(v, 95), "mean": sum(v) / len(v) if v else None}
    es = [e for e in CACHED.values() if e.get("ok") and not e.get("retried")]
    res = {"where": "the compute node (UW Health network, Madison WI) to api.typesafe.ai, HTTPS, urllib, one request per record",
           "model": MODEL, "first_attempt_only": True}
    for nm, sel in (("sequential", [e for e in es if e.get("workers") == 1]), ("concurrent", [e for e in es if e.get("workers", WORKERS) > 1]), ("all", es)):
        res[nm] = {"per_record_ms": summ([e["latency_ms"] for e in sel]),
                   "per_question_ms": summ([e["latency_ms"] / len(e["tasks"]) for e in sel])}
    res["concurrency"] = WORKERS
    bt = {}
    for e in es:
        for t in set(e["tasks"].values()): bt.setdefault(t, []).append(e["latency_ms"])
    res["per_record_ms_by_task"] = {t: summ(v) for t, v in sorted(bt.items())}
    (OUT / "jev_latency.json").write_text(json.dumps(res, indent=1))


def inspect():
    """Answer shapes in the cache (no case text: answer names, option keys and probabilities only)."""
    shapes, ex = {}, {}
    for e in CACHED.values():
        for a in (e.get("response") or {}).get("answers", {}).values():
            k = a.get("type", "?") + ":" + ",".join(sorted(a)); shapes[k] = shapes.get(k, 0) + 1
            if k not in ex or len(ex[k]) < 3: ex.setdefault(k, []).append({"tasks": e["tasks"], "answer": a, "n_options": [len(q.get("criteria") or []) for q in e["request"]["questions"].values()]})
    status["answer_shapes"] = shapes; status["answer_examples"] = ex; status["models"] = {}
    for e in CACHED.values():
        for m in [(e.get("response") or {}).get("model")] + [(p.get("response") or {}).get("model") for p in e.get("parts", [])]:
            if m: status["models"][m] = status["models"].get(m, 0) + 1


def main():
    import sys
    ph = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "all"
    for p in (DIR, CODE): p.mkdir(parents=True, exist_ok=True)
    for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
    if not KEY.exists(): status["error"] = "no key at $XDG_CACHE_HOME/raddecision/jev_key (a key file placed there by the operator)"; save(); return
    load_cache(); items = plan(); PLANNED.update({it["id"]: it["tasks"] for it in items})
    if ph == "inspect": phase("inspect", inspect); return
    phase("probe", lambda: run_probe(items))
    if ph != "probe":
        phase("run", lambda: run_all(items))
        phase("preds", preds)
    phase("latency", latency); phase("inspect", inspect)
    status["cache"] = {"records": len(CACHED), "of": len(items), "failed_final": sum(1 for e in CACHED.values() if not e.get("ok") and not e.get("split")),
                       "split": sum(1 for e in CACHED.values() if e.get("split"))}
    status["spent"] = {"tokens": SPENT["tokens"], "usd": round(SPENT["tokens"] * USD_PER_TOKEN, 4)}
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    main()
