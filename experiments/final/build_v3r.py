"""Build data/mix-v3r (2026-10-05): the v3 training mix restricted to radiology (user decision). Knowledge records (MedMCQA,
MedQA) are kept only if their question is radiology-relevant (build_external.is_radiology, or the MedMCQA radiology subject);
Eurorad, CT-RATE, teacher-labeled, RadCases and ReXErr records are kept. Train and dev are filtered alike. Counts only.

    python jobs/submit.py jobs/build_v3r.py --gpus 0 --cpus 4 --mem 16 --timeout 30 --outputs 'build_v3r.json'
"""
import json, os, subprocess
from pathlib import Path

INCLUDE = ["build_data.py", "build_external.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"; KEV_PY = RADKEV / "kev/.venv/bin/python"
CODE = RADKEV / "code_v3r"; CODE.mkdir(parents=True, exist_ok=True)
for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
INNER = r'''
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import build_external as be
R = Path(sys.argv[2]); out = R / "data/mix-v3r"; out.mkdir(parents=True, exist_ok=True); res = {}
KNOW = ("medmcqa", "medqa")
for split in ("train", "dev"):
    kept, dropped = Counter(), Counter(); lines = []
    for d in ("mix-v2mg", "radcases-v3", "rexerr"):
        p = R / "data" / d / f"{split}.jsonl"
        for line in open(p):
            if not line.strip(): continue
            r = json.loads(line); srcs = {str(q.get("src", "")) for q in r["questions"].values()}; src = sorted(srcs)[0].split("_")[0] if srcs else "other"
            if any(s.startswith(KNOW) for s in srcs):
                qs = {k: q for k, q in r["questions"].items() if str(q.get("src", "")) == "medmcqa_rad" or be.is_radiology(r["state"], q)}
                if not qs: dropped[src] += 1; continue
                r["questions"] = qs; line = json.dumps(r, ensure_ascii=False)
            kept[src] += 1; lines.append(line)
    (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    res[split] = {"kept": dict(kept), "dropped": dict(dropped), "total": len(lines)}
print(json.dumps(res))
'''
p = subprocess.run([str(KEV_PY), "-c", INNER, str(CODE), str(RADKEV)], env={**os.environ, "PYTHONPATH": str(CODE)}, capture_output=True, text=True)
(OUT / "build_v3r.json").write_text(json.dumps(json.loads(p.stdout.strip().splitlines()[-1]), indent=1) if p.returncode == 0 else json.dumps({"error": p.stderr[-3000:]}))
