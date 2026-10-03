"""Data for two Methods figures (post hoc; CPU only; reads files and writes only its own output).

    python experiments/figure_data.py                  # training loss curves and the teacher example pools
    python experiments/figure_data.py --teacher-only   # the teacher example pools only

1. loss    per-step training loss parsed from the kev.train logs written by experiments/train.py ($RADKEV_HOME/runs/<tag>-<model>.log)
           of RadKev-27B (v2mg-kev-27b), RadKev-9B (v2x9-kev-9b), the base-initialized 9B run (v2x9-base-9b) and the 10% runs
           (v2x9f10-*). The log format is not assumed: every line containing "loss" is parsed for (step, loss) pairs; the first
           lines of each log and a few raw loss lines are kept so that the parse can be checked.
2. teacher agreed and disagreed two-teacher examples from Eurorad order questions only (public cases, CC BY-NC-SA 4.0), with
           both teachers' answer distributions (MedGemma with the pre-filled segment, teacher/medgemma_brief.jsonl from
           experiments/teacher_rescore.py if present, else teacher/medgemma.jsonl; Qwen3.8-27B teacher/qwen38.jsonl). Chosen
           deterministically: indications of 90-300 characters, order questions with 3-6 options, candidates in
           sha256("fig|<task id>") order, agreed examples with both top probabilities >= 0.5, disagreed examples with both
           >= 0.45; up to 25 of each.
Writes $RADKEV_HOME/runs/figure_data/figure_data.json: losses (numbers) and the Eurorad example pools (public case text); nothing
from gated sources.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from radkev.paths import HOME, RUNS

STEP = re.compile(r"\bstep\D{0,3}(\d+)", re.I)
LOSS = re.compile(r"\bloss\b\D{0,4}([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+(?:\.\d+)?)", re.I)
RUN_LOGS = ("v2mg-kev-27b", "v2x9-kev-9b", "v2x9-base-9b", "v2x9f10")


def losses():
    out = {}
    for log in sorted(RUNS.glob("*.log")):
        if not any(k in log.name for k in RUN_LOGS): continue
        head, raw, pts = [], [], []
        with open(log, errors="replace") as f:
            for i, line in enumerate(f):
                if i < 25: head.append(line.rstrip()[:300])
                if "loss" not in line.lower(): continue
                m = LOSS.search(line)
                if not m: continue
                s = STEP.search(line)
                pts.append([int(s.group(1)) if s else None, float(m.group(1)), i])
                if len(raw) < 8: raw.append(line.rstrip()[:300])
        out[log.name] = {"bytes": log.stat().st_size, "head": head, "raw_loss_lines": raw, "n": len(pts), "points": pts}
    return out


def teacher_examples():
    T = HOME / "teacher"
    def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
    res = {}
    tasks = {t["tid"]: t for t in jl(T / "tasks.jsonl") if t["source"] == "eurorad" and t["family"] == "order"}
    mg_path = T / "medgemma_brief.jsonl" if (T / "medgemma_brief.jsonl").exists() else T / "medgemma.jsonl"
    mg = {r["tid"]: r["p"] for r in jl(mg_path) if r["tid"] in tasks}
    qw = {r["tid"]: r["p"] for r in jl(T / "qwen38.jsonl") if r["tid"] in tasks}
    res["medgemma_file"] = mg_path.name
    def text(st): return st if isinstance(st, str) else " ".join(str(v) for v in st.values())
    cands = []
    for tid, t in tasks.items():
        if tid not in mg or tid not in qw: continue
        crit = t["question"].get("criteria")
        if not isinstance(crit, dict) or not (3 <= len(crit) <= 6): continue
        a, b = mg[tid], qw[tid]
        ta, tb = max(a, key=a.get), max(b, key=b.get)
        cands.append((len(text(t["state"])), tid, ta == tb, a[ta], b[tb]))
    cands.sort(key=lambda c: hashlib.sha256(f"fig|{c[1]}".encode()).hexdigest())
    pool = {"agreed": [], "disagreed": []}
    for c in cands:
        if not (90 <= c[0] <= 300): continue
        k = "agreed" if c[2] else "disagreed"
        if (c[2] and min(c[3], c[4]) >= 0.5) or (not c[2] and min(c[3], c[4]) >= 0.45):
            if len(pool[k]) < 25: pool[k].append(c)
    for k, cs in pool.items():
        res[k + "_pool"] = [{"tid": c[1], "group": tasks[c[1]]["group"], "qid": tasks[c[1]]["qid"], "state": tasks[c[1]]["state"],
                             "question": tasks[c[1]]["question"], "medgemma": mg[c[1]], "qwen38": qw[c[1]]} for c in cs]
    res["n_candidates"] = len(cands)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher-only", dest="teacher_only", action="store_true", help="skip the training-loss logs")
    ap.add_argument("--out", default="", help="output file (default: $RADKEV_HOME/runs/figure_data/figure_data.json)")
    a = ap.parse_args()
    res = {"logs": {} if a.teacher_only else losses(), "teacher": {}}
    try: res["teacher"] = teacher_examples()
    except Exception as e: res["teacher"]["error"] = repr(e)
    out = Path(a.out) if a.out else RUNS / "figure_data" / "figure_data.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: {n: v["n"] for n, v in res["logs"].items()} if k == "logs" else list(v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
