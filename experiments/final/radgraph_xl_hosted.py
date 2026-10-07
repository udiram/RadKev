"""RadGraph-XL external test for the hosted decision models (OpenAI Decisions, Jev) and the combined external analysis
(user request 2026-10-07). No GPU.

    python jobs/submit.py jobs/radgraph_xl_hosted.py --gpus 0 --cpus 4 --mem 16 --timeout 120 --outputs 'radgraph_xl_hosted*.json'

1. Every RadGraph-XL record ($RADKEV/external/radgraph_xl.jsonl, 1,505 reports, 4,037 status questions) is sent ONCE to each API
   with all of its questions in one request, built by the benchmark jobs' own api_question (no label is sent); cached per record
   in runs/radgraph_xl_hosted/<system>.jsonl, so a rerun never resends. A question an API declines receives a uniform distribution.
2. Analysis with paper/external/analyse.py (record-level bootstrap stratified by modality, the seed and resamples of the reported
   external analysis) for RadKev-27B/9B, Kev-27B/9B, Qwen3.8-27B and both APIs:
     all          every question, as scored (four options)
     definite     questions whose annotated status is present or absent, with the "uncertain" option removed from every system's
                  answer (probabilities renormalized over present, absent and not mentioned)
     hedged       questions whose annotated status is uncertain, as scored
   plus, per system, accuracy by annotated status and the confusion of annotated against predicted status.
Publishes aggregates only: radgraph_xl_hosted.json (and _status.json).
"""
import json
import os
import subprocess
import time
from pathlib import Path

INCLUDE = ["jobs/openai_decisions.py", "jobs/jev_decisions.py", "paper/external/analyse.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ["XDG_CACHE_HOME"]); RADKEV = CACHE / "radkev"; KEV_DIR, KEV_PY = RADKEV / "kev", RADKEV / "kev/.venv/bin/python"
EXT = RADKEV / "external"; DIR = RADKEV / "runs/radgraph_xl_hosted"; DIR.mkdir(parents=True, exist_ok=True)
CODE = EXT / "code_hosted"; CODE.mkdir(parents=True, exist_ok=True)
CAP_USD = 2.00
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}


def save(): (OUT / "radgraph_xl_hosted_status.json").write_text(json.dumps(status, indent=1, default=str))


def load(name):
    ns = {"__name__": name, "__file__": name}; exec(BUNDLE[f"jobs/{name}.py"], ns); return ns


class API:
    def __init__(self, sysname, ns):
        self.sys, self.ns, self.tokens, self.price = sysname, ns, 0, ns["USD_PER_TOKEN"]
        self.cache = DIR / f"{sysname}.jsonl"; self.done = {}
        if self.cache.exists():
            for l in open(self.cache):
                if l.strip(): e = json.loads(l); self.done[e["id"]] = e; self.tokens += e.get("tokens", 0)

    def body(self, state, qs):
        if self.sys == "openai_dec":
            return {"model": self.ns["MODEL"], "input": self.ns["text"](state), "questions": [self.ns["api_question"](n, q) for n, q in qs.items()]}
        return {"model": self.ns["MODEL"], "state": state, "questions": {n: self.ns["api_question"](q) for n, q in qs.items()}}

    def answers(self, resp):
        out = {}
        if self.sys == "openai_dec":
            for a in resp.get("answers", []):
                out[a["name"]] = None if a.get("type") == "refusal" else {str(x["value"]): float(x["probability"]) for x in a.get("probabilities", [])}
        else:
            for n, a in resp.get("answers", {}).items():
                out[n] = {str(k): float(v) for k, v in a["probabilities"].items()} if a.get("type") == "choice" else None
        return out

    def call(self, body):
        for k in range(6):
            if self.tokens * self.price > CAP_USD: raise SystemExit(f"{self.sys}: spend cap reached")
            code, resp, ms, h = self.ns["post"](body)
            if isinstance(resp, dict): self.tokens += resp.get("usage", {}).get("input_tokens", 0)
            if code == 200 or (code is not None and 400 <= code < 500 and code != 429): return code, resp
            time.sleep(2 ** (k + 1))
        return code, resp

    def send(self, rid, state, qs):
        if rid in self.done: return
        t0 = self.tokens; code, resp = self.call(self.body(state, qs)); e = {"id": rid, "status": code}
        if code == 200: e["answers"] = self.answers(resp)
        elif code in (400, 422):
            e["answers"] = {}
            for n, q in qs.items():
                c2, r2 = self.call(self.body(state, {n: q})); e["answers"][n] = self.answers(r2).get(n) if c2 == 200 else None
        else: e["error"] = str(resp)[:500]
        e["tokens"] = self.tokens - t0
        with open(self.cache, "a") as f: f.write(json.dumps(e) + "\n")
        self.done[rid] = e


ANALYSE = r'''
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from kev.metrics import scored_rows
import analyse as A
EXT, DIR, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
A.PAIRS = [("v3_27", "stock27"), ("v3_9", "stock9"), ("v3_27", "qwen38"), ("v3_9", "qwen38"), ("qwen38", "stock27"),
           ("v3_27", "jev"), ("v3_27", "openai_dec"), ("v3_9", "jev"), ("v3_9", "openai_dec"), ("jev", "stock27"), ("openai_dec", "stock27"),
           ("jev", "qwen38"), ("openai_dec", "qwen38"), ("openai_dec", "jev")]
recs = {json.loads(l)["_meta"]["id"]: json.loads(l) for l in open(EXT / "radgraph_xl.jsonl") if l.strip()}
runs = EXT / "runs/radgraph_xl"; P = {}; TASK = {}
for m in ("v3_27", "v3_9", "stock27", "stock9", "qwen38"):
    for r in scored_rows(json.loads((runs / m / "rows.json").read_text())):
        P.setdefault(m, {})[(r["id"], r["question"])] = np.asarray(r["p"], float); TASK[(r["id"], r["question"])] = r["task"]
refused = Counter()
for m in ("openai_dec", "jev"):
    for l in open(DIR / f"{m}.jsonl"):
        if not l.strip(): continue
        e = json.loads(l); ans = e.get("answers") or {}
        for qid, q in recs[e["id"]]["questions"].items():
            if (e["id"], qid) not in TASK: continue
            ks = list(q["criteria"]); a = ans.get(qid)
            if not a: refused[m] += 1; p = np.full(len(ks), 1 / len(ks))
            else: p = np.array([max(0.0, a.get(k, 0.0)) for k in ks]); p = p / p.sum() if p.sum() > 0 else np.full(len(ks), 1 / len(ks))
            P.setdefault(m, {})[(e["id"], qid)] = p
keys = sorted(set.intersection(*[set(v) for v in P.values()]))
gold = {k: recs[k[0]]["questions"][k[1]]["label"] for k in keys}; okeys = {k: list(recs[k[0]]["questions"][k[1]]["criteria"]) for k in keys}
def rows(sel, drop_uncertain):
    o = {}
    for m, pm in P.items():
        rr = []
        for k in keys:
            if not sel(k): continue
            p = pm[k].copy(); ks = okeys[k]; li = ks.index(gold[k])
            if drop_uncertain: p[ks.index("uncertain")] = 0; p = p / p.sum() if p.sum() > 0 else np.full(len(ks), 1 / len(ks))
            rr.append([k[0], k[1], TASK[k], li, int(np.argmax(p)), float(p[li]), float(p.max())])
        o[m] = rr
    return o
res = {"n_keys": len(keys), "refused": dict(refused), "gold_dist": dict(Counter(gold.values())),
       "all": A.analyse(rows(lambda k: True, False), set()),
       "definite": A.analyse(rows(lambda k: gold[k] != "uncertain", True), set()),
       "hedged": A.analyse(rows(lambda k: gold[k] == "uncertain", False), set()),
       "by_status": {}, "confusion": {}}
for m, pm in P.items():
    conf = defaultdict(Counter)
    for k in keys: conf[gold[k]][okeys[k][int(np.argmax(pm[k]))]] += 1
    res["confusion"][m] = {g: dict(c) for g, c in conf.items()}
    res["by_status"][m] = {g: c[g] / sum(c.values()) for g, c in conf.items()}
out.write_text(json.dumps(res, indent=1)); print("ok", len(keys))
'''


def main():
    recs = [json.loads(l) for l in open(EXT / "radgraph_xl.jsonl") if l.strip()]
    status["records"] = len(recs); save()
    for sysname, mod in (("openai_dec", "openai_decisions"), ("jev", "jev_decisions")):
        api = API(sysname, load(mod)); status["phases"][sysname] = "sending"; save()
        for r in recs:
            qs = {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in r["questions"].items()}   # no label leaves the node
            api.send(r["_meta"]["id"], r["state"], qs)
        status["phases"][sysname] = "done"; status.setdefault("spend_usd", {})[sysname] = round(api.tokens * api.price, 4); save()
    (CODE / "analyse.py").write_text(BUNDLE["paper/external/analyse.py"])
    p = subprocess.run([str(KEV_PY), "-c", ANALYSE, str(CODE), str(EXT), str(DIR), str(OUT / "radgraph_xl_hosted.json")],
                       env={**os.environ, "PYTHONPATH": str(KEV_DIR), "HF_HUB_OFFLINE": "1"}, cwd=KEV_DIR, capture_output=True, text=True)
    status["analyse"] = {"rc": p.returncode, "out": p.stdout[-500:], "err": p.stderr[-3000:]}
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); save()


if __name__ == "__main__":
    try: main()
    except BaseException as ex: status["error"] = f"{type(ex).__name__}: {ex}"[-3000:]; save(); raise
