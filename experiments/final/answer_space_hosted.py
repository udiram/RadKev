"""Answer-space study for the hosted decision models (post hoc; no GPU; user request 2026-10-07): OpenAI Decisions (gpt-6-luna)
and Jev (jev-latest) on exactly the questions and latency requests of jobs/answer_space3.py.

    python jobs/submit.py jobs/answer_space_hosted.py --gpus 0 --cpus 2 --mem 8 --timeout 120 --outputs 'answer_space_hosted*.json'

Inputs (built by answer_space3 on the node): answer_space3s/records.jsonl (60 records: 30 Eurorad diagnosis and 30 RSNA-RadioQA
questions, each with every answer-space condition as its own question on the same state) and answer_space3s/latency_records.jsonl
(15 Eurorad cases x 2, 16, 64 and 255 options, one question per request, shuffled). Each record is sent ONCE to each API with all of
its conditions in one request, as the decision models received it; the request bodies are built by the benchmark jobs'
own api_question (jobs/openai_decisions.py, jobs/jev_decisions.py), and nothing else is set. A request rejected as invalid (HTTP 400
or 422) is resent one question at a time; a question that is still rejected, or that the API declines, counts as incorrect ("x")
and is reported. Latency: every latency request sent one at a time, end to end from the node; the first 5 are excluded in the
analysis, as for the other systems. Spend is counted from the reported input tokens and capped at CAP_USD per API.
Publishes per-question correctness and confidence (no text), latencies, refusals and spend: answer_space_hosted.json.
"""
import json
import os
import time
from pathlib import Path

INCLUDE = ["jobs/openai_decisions.py", "jobs/jev_decisions.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ["XDG_CACHE_HOME"]); RADKEV = CACHE / "radkev"
AS3 = RADKEV / "answer_space3s"; DIR = RADKEV / "runs/answer_space_hosted"; DIR.mkdir(parents=True, exist_ok=True)
CAP_USD = 1.00
status = {"started": time.strftime("%Y%m%d-%H%M%S"), "phases": {}}


def save(): (OUT / "answer_space_hosted_status.json").write_text(json.dumps(status, indent=1, default=str))


def load(name):
    ns = {"__name__": name, "__file__": name}; exec(BUNDLE[f"jobs/{name}.py"], ns); return ns


class API:
    def __init__(self, sysname, ns):
        self.sys, self.ns = sysname, ns; self.tokens = 0; self.cache = DIR / f"{sysname}.jsonl"
        self.price = ns["USD_PER_TOKEN"]; self.done = {}
        if self.cache.exists():
            for l in open(self.cache):
                if l.strip(): e = json.loads(l); self.done[e["id"]] = e; self.tokens += e.get("tokens", 0)

    def body(self, state, qs):   # qs: {name: question}
        if self.sys == "openai_dec":
            return {"model": self.ns["MODEL"], "input": self.ns["text"](state), "questions": [self.ns["api_question"](n, q) for n, q in qs.items()]}
        return {"model": self.ns["MODEL"], "state": state, "questions": {n: self.ns["api_question"](q) for n, q in qs.items()}}

    def answers(self, resp):   # {name: {option key: probability}} or {name: None} for a declined question
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
            if code == 200 or (code is not None and 400 <= code < 500 and code != 429): return code, resp, ms, h
            time.sleep(2 ** (k + 1))
        return code, resp, ms, h

    def send(self, rid, state, qs):
        if rid in self.done: return self.done[rid]
        t0 = self.tokens; code, resp, ms, h = self.call(self.body(state, qs))
        e = {"id": rid, "status": code, "ms": round(ms, 2), "server_ms": h.get("openai-processing-ms") if h else None}
        if code == 200: e["answers"] = self.answers(resp)
        elif code in (400, 422):   # one question per request; a question still rejected stays None
            e["answers"], e["split"] = {}, True
            for n, q in qs.items():
                c2, r2, _, _ = self.call(self.body(state, {n: q}))
                e["answers"][n] = self.answers(r2).get(n) if c2 == 200 else None
        else: e["error"] = str(resp)[:500]
        e["tokens"] = self.tokens - t0
        with open(self.cache, "a") as f: f.write(json.dumps(e) + "\n")
        self.done[rid] = e; return e


def main():
    apis = [API("openai_dec", load("openai_decisions")), API("jev", load("jev_decisions"))]
    recs = [json.loads(l) for l in open(AS3 / "records.jsonl") if l.strip()]
    lat = [json.loads(l) for l in open(AS3 / "latency_records.jsonl") if l.strip()]
    rids = [r["_meta"]["rid"] for r in recs]; conds = sorted({c for r in recs for c in r["questions"]})
    res = {"rids": rids, "conditions": conds, "models": {}, "latency": {}, "refused": {}, "spend_usd": {}}
    for api in apis:
        status["phases"][api.sys] = "accuracy"; save()
        for r in recs:
            api.send(r["_meta"]["rid"], r["state"], r["questions"])
        bits, conf, refused = {}, {}, 0
        for c in conds:
            s, cf = [], []
            for r in recs:
                q = r["questions"].get(c)
                if q is None: s.append("-"); cf.append(-1); continue
                p = (api.done[r["_meta"]["rid"]].get("answers") or {}).get(c)
                if not p: s.append("x"); cf.append(-1); refused += 1; continue
                top = max(p, key=p.get); s.append("1" if top == q["label"] else "0"); cf.append(int(round(100 * p[top])))
            bits[c], conf[c] = "".join(s), cf
        res["models"][api.sys] = {"kind": "hosted_decision", "correct": bits, "conf": conf}; res["refused"][api.sys] = refused
        status["phases"][api.sys] = "latency"; save()
        rows = []
        for r in lat:   # one at a time, end to end; cached requests are not re-timed
            e = api.send("lat|" + r["_meta"]["rid"], r["state"], r["questions"])
            rows.append([r["_meta"]["K"], e["ms"]] + ([float(e["server_ms"])] if e.get("server_ms") else []))
        res["latency"][api.sys] = rows; res["spend_usd"][api.sys] = round(api.tokens * api.price, 4)
        status["phases"][api.sys] = "done"; save()
    (OUT / "answer_space_hosted.json").write_text(json.dumps(res, separators=(",", ":")))
    status["finished"] = time.strftime("%Y%m%d-%H%M%S"); status["refused"] = res["refused"]; status["spend_usd"] = res["spend_usd"]; save()


if __name__ == "__main__":
    try: main()
    except BaseException as ex: status["error"] = f"{type(ex).__name__}: {ex}"[-3000:]; save(); raise
