"""Generalist decision ("System One") models, zero-shot, on the same held-out test items and metrics as the main comparison.
Added after the pre-registered test read (secondary baselines); nothing about RadKev is selected or changed with them.

    CUDA_VISIBLE_DEVICES=0 python experiments/decision_baselines.py

Models (all public, all released in September 2026, none trained on medicine):
  kev08, kev4      jaredpalmer/kev-0.8b, kev-4b                  Kev family, smaller sizes (Kev venv, kev_eval --run)
  laya_typed       convaiinnovations/laya-typed-decisions        421M ModernBERT, typed-decisions fine-tune (Laya venv)
  julia1           SupersonicLabs/Julia-1                        144M mmBERT-small decision model (own venv; 2-20 options)
  gliner_decide    fastino/GLiNER2.5-Decide                      340M GLiNER2 classifier (own venv; label scores normalised)
Each writes {"id": "rad/<line>", "probabilities": {qid: {key: p}}} and is scored by radkev.evaluate --preds into
runs/test-final/<name>/, so a later experiments/final_test.py comparison (--runs <name>=cached) includes it. A question a model cannot take (e.g. more
than 20 options for Julia) gets a uniform distribution and is counted in "unsupported_questions".
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from radkev.paths import HOME, KEV_PY, LAYA_PY, RUNS

HERE = Path(__file__).resolve().parent
UV = "uv"
WORK = RUNS / "test-final"; TEST = WORK / "test.jsonl"
VENVS = HOME / "venvs"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ENV = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
rep = {"models": {}}

COMMON = r'''
import json, sys
def keys(q):
    if q["type"] == "choice": return list(q["criteria"])
    if q["type"] == "noul": return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]
def uniform(q):
    k = keys(q); return {x: 1 / len(k) for x in k}
def norm(d, q):
    k = keys(q); p = {x: max(0.0, float(d.get(x, 0.0))) for x in k}; s = sum(p.values())
    return {x: v / s for x, v in p.items()} if s > 0 else uniform(q)
'''

JULIA = COMMON + r'''
from julia import load_model
data, out = sys.argv[1], sys.argv[2]
eng = load_model(sys.argv[3], device="cuda", strict_encoding=False, max_length=8192, head_length=512)
unsupported = 0
with open(data) as f, open(out, "w") as o:
    for n, line in enumerate(f):
        if not line.strip(): continue
        r = json.loads(line); probs = {}; ask = {}
        for qid, q in r["questions"].items():
            nk = len(keys(q))
            if not 2 <= nk <= 20: probs[qid] = uniform(q); unsupported += 1; continue
            qq = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
            if q["type"] == "choice": qq["criteria"] = {k: (v or k) for k, v in q["criteria"].items()}
            ask[qid] = qq
        if ask:
            try:
                res = eng.predict(state=r["state"] if isinstance(r["state"], str) else json.dumps(r["state"]), questions=ask)
                for qid, a in res["answers"].items(): probs[qid] = norm({str(k): v for k, v in a["probabilities"].items()}, r["questions"][qid])
            except Exception:
                for qid in ask: probs[qid] = uniform(r["questions"][qid]); unsupported += 1
        o.write(json.dumps({"id": f"rad/{n}", "probabilities": probs}) + "\n")
print(json.dumps({"unsupported_questions": unsupported}))
'''

GLINER = COMMON + r'''
from gliner2 import AutoExtractor
data, out = sys.argv[1], sys.argv[2]
model = AutoExtractor.from_pretrained(sys.argv[3])
try: model.to("cuda")
except Exception: pass
unsupported, shapes = 0, {}
def parse(res):
    """label -> score from any shape classify_text returns (dict, {label, confidence}, list of those, list of labels)."""
    shapes[type(res).__name__] = shapes.get(type(res).__name__, 0) + 1
    if isinstance(res, str): return {res: 1.0}
    if isinstance(res, dict) and "label" in res: return {res["label"]: float(res.get("confidence", 1.0))}
    if isinstance(res, dict): return {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}
    out = {}
    for x in res if isinstance(res, list) else []:
        if isinstance(x, dict) and "label" in x: out[x["label"]] = float(x.get("confidence", 1.0))
        elif isinstance(x, (list, tuple)) and len(x) == 2: out[str(x[0])] = float(x[1])
        elif isinstance(x, str): out[x] = 1.0
    return out
def labels(q):
    if q["type"] == "choice": return {f"{k}: {v}" if v else str(k): k for k, v in q["criteria"].items()}
    if q["type"] == "noul": return {"no": "false", "yes": "true"}
    return {f"{i}: {c}": str(i) for i, c in enumerate(q["criteria"])}
with open(data) as f, open(out, "w") as o:
    for n, line in enumerate(f):
        if not line.strip(): continue
        r = json.loads(line); probs = {}
        state = r["state"] if isinstance(r["state"], str) else "\n".join(f"{k}: {v}" for k, v in r["state"].items())
        for qid, q in r["questions"].items():
            lab = labels(q)
            try:
                res = model.classify_text(f"{state}\n\nQuestion: {q['instructions']}",
                                          {"a": {"labels": list(lab), "multi_label": True, "cls_threshold": 0.0}}, include_confidence=True)["a"]
                scores = {lab[k]: v for k, v in parse(res).items() if k in lab}
                probs[qid] = norm(scores, q) if scores else uniform(q)
                if not scores: unsupported += 1
            except Exception:
                probs[qid] = uniform(q); unsupported += 1
        o.write(json.dumps({"id": f"rad/{n}", "probabilities": probs}) + "\n")
print(json.dumps({"unsupported_questions": unsupported, "output_shapes": shapes}))
'''


def sh(cmd, env=ENV, cwd=None, timeout=None):
    p = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if p.returncode: raise RuntimeError(f"{' '.join(map(str, cmd[:3]))} exited {p.returncode}: {(p.stderr or p.stdout)[-1500:]}")
    return p.stdout


def venv(name, pkgs, python="3.12"):
    v = VENVS / name
    if not (v / "bin/python").exists():
        sh([UV, "venv", "--python", python, str(v)])
        sh([UV, "pip", "install", "--python", str(v / "bin/python"), "torch", "--index-url", TORCH_INDEX], timeout=3600)
        sh([UV, "pip", "install", "--python", str(v / "bin/python"), *pkgs], timeout=3600)
    return v / "bin/python"


def hub(repo):
    return sh([KEV_PY, "-c", "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))", repo], timeout=3600).strip().splitlines()[-1]


def score(name, preds):
    out = WORK / name
    sh([KEV_PY, "-m", "radkev.evaluate", "--preds", preds, "--data", TEST, "--out", out])
    s = json.loads((out / "summary.json").read_text())
    return {"overall": s["overall"], "tasks": {t: {"n": v["n"], "acc": v["acc"]} for t, v in s["tasks"].items()}}


def main():
    VENVS.mkdir(parents=True, exist_ok=True)
    kenv = {**ENV, "KEV_DTYPE": "bf16"}
    jobs = [
        ("kev08", lambda: sh([KEV_PY, "-m", "radkev.evaluate", "--run", "jaredpalmer/kev-0.8b", "--data", TEST, "--out", WORK / "kev08"], env=kenv)),
        ("kev4", lambda: sh([KEV_PY, "-m", "radkev.evaluate", "--run", "jaredpalmer/kev-4b", "--data", TEST, "--out", WORK / "kev4"], env=kenv)),
        ("laya_typed", lambda: sh([LAYA_PY, HERE / "laya_predict.py", "--data", TEST, "--out", WORK / "laya_typed.preds.jsonl",
                                   "--model", "convaiinnovations/laya-typed-decisions", "--device", "cuda"])),
        ("julia1", lambda: sh([venv("julia", ["transformers>=5.0,<5.1", "safetensors>=0.5", "numpy>=1.26", "huggingface_hub"]), "-c", JULIA,
                               TEST, WORK / "julia1.preds.jsonl", hub("SupersonicLabs/Julia-1")], env={**ENV, "PYTHONPATH": hub("SupersonicLabs/Julia-1")})),
        ("gliner_decide", lambda: sh([venv("gliner2", ["gliner2[local]", "transformers<5"]), "-c", GLINER, TEST, WORK / "gliner_decide.preds.jsonl",
                                      "fastino/GLiNER2.5-Decide"])),
    ]
    for name, fn in jobs:
        t0 = time.time()
        try:
            if (WORK / name / "summary.json").exists(): out = ""
            else: out = fn()
            extra = {}
            for l in (out or "").splitlines()[::-1]:
                if l.startswith("{"):
                    try: extra = json.loads(l); break
                    except Exception: pass
            if not (WORK / name / "summary.json").exists(): rep["models"][name] = {**score(name, WORK / f"{name}.preds.jsonl"), **extra}
            else: s = json.loads((WORK / name / "summary.json").read_text()); rep["models"][name] = {"overall": s["overall"], "tasks": {t: {"n": v["n"], "acc": v["acc"]} for t, v in s["tasks"].items()}, **extra}
            rep["models"][name]["hours"] = round((time.time() - t0) / 3600, 2)
        except Exception as e:
            rep["models"][name] = {"error": "".join(l for l in str(e).splitlines(True) if "hf_" not in l)[-800:]}
        (WORK / "decision_baselines.json").write_text(json.dumps(rep, indent=1))
    sys.exit(0 if all("error" not in v for v in rep["models"].values()) else 1)


if __name__ == "__main__":
    main()
