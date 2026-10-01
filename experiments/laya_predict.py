"""Run Laya on labelled records and write its probabilities in Kev's key order (scored afterwards by kev_eval.py --preds).

Runs inside the Laya venv:  python laya_predict.py --data dev.jsonl --out laya_dev.jsonl [--model convaiinnovations/laya]
Record ids follow kev.data.load_records(path, source="rad"): "rad/<line number>".
"""
import argparse
import json
import time

import laya


def keys(q):
    if q["type"] == "choice": return list(q["criteria"])
    if q["type"] == "noul": return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def probabilities(q, ans):
    if q["type"] == "noul":
        p = float(ans["noul"]); return {"false": 1 - p, "true": p}
    raw = ans["probabilities"]
    p = {k: float(raw.get(k, 0.0)) for k in keys(q)}
    s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="convaiinnovations/laya"); ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    agent = laya.load(a.model, subfolder=a.subfolder, device=a.device)
    n_done, t0 = 0, time.time()
    with open(a.data) as f, open(a.out, "w") as out:
        for n, line in enumerate(f):
            if not line.strip(): continue
            r = json.loads(line)
            questions = {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")} for qid, q in r["questions"].items()}
            res = agent.predict(r["state"], questions)
            out.write(json.dumps({"id": f"rad/{n}", "probabilities": {qid: probabilities(q, res["answers"][qid]) for qid, q in questions.items()}}) + "\n")
            n_done += 1
    print(json.dumps({"records": n_done, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
