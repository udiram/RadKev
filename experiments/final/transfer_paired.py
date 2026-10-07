"""Paired bootstrap on Kev's out-of-domain transfer suite (general decision skill): does starting from Kev keep it?

    python jobs/submit.py jobs/transfer_paired.py --gpus 0 --timeout 30 --outputs 'transfer_paired.json'
    python jobs/submit.py jobs/transfer_paired.py --gpus 0 --timeout 30 --outputs 'transfer_paired_v3.json' -- v3

Reads rows.json that kev.benchmark already wrote for each training run's transfer_ft (the fine-tune) and transfer_base (the
released Kev checkpoint it is compared with), and compares them on identical items with Kev's paired bootstrap (clean,
knowable rows; sibling questions resample together). No model is run. Publishes accuracies, ECE and deltas only.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

INCLUDE = ["jobs/compare.py"]
BUNDLE = {}
OUT = Path(os.environ.get("ZCB_OUTPUT_DIR", "out")); OUT.mkdir(parents=True, exist_ok=True)
RADKEV = Path(os.environ["XDG_CACHE_HOME"]) / "radkev"
KEV_DIR, KEV_PY, CODE = RADKEV / "kev", RADKEV / "kev/.venv/bin/python", RADKEV / "code_transfer"
ROWS = {"stock9": "v2x9-kev-9b/transfer_base", "r9": "v2x9-kev-9b/transfer_ft", "b9": "v2x9-base-9b/transfer_ft",
        "r9f10": "v2x9f10-kev-9b/transfer_ft", "b9f10": "v2x9f10-base-9b/transfer_ft",
        "stock27": "v2mg-kev-27b/transfer_base", "v2_27": "v2mg-kev-27b/transfer_ft"}
# v3 runs (amendment 8); each entry lists the 4-GPU run first and the one-GPU fallback run second
ROWS_V3 = {"stock9": ["v3f-kev-9b-dp/transfer_base", "v3g-kev-9b/transfer_base"], "v3_9": ["v3f-kev-9b-dp/transfer_ft", "v3g-kev-9b/transfer_ft"],
           "base9": ["v3f-base-9b-dp/transfer_ft", "v3g-base-9b/transfer_ft"], "k9_10": ["v3f10-kev-9b-dp/transfer_ft", "v3g10-kev-9b/transfer_ft"],
           "b9_10": ["v3f10-base-9b-dp/transfer_ft", "v3g10-base-9b/transfer_ft"],
           "stock27": ["v3f-kev-27b-dp/transfer_base"], "v3_27": ["v3f-kev-27b-dp/transfer_ft"]}
PAIRS_V3 = [("v3_9", "stock9"), ("base9", "stock9"), ("v3_9", "base9"), ("k9_10", "stock9"), ("b9_10", "stock9"), ("k9_10", "b9_10"), ("v3_27", "stock27")]
PAIRS = [("r9", "stock9"), ("b9", "stock9"), ("r9", "b9"), ("r9f10", "stock9"), ("b9f10", "stock9"), ("r9f10", "b9f10"), ("v2_27", "stock27")]

INNER = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from compare import paired
from kev.metrics import metrics, scored_rows
rows = {k: scored_rows(json.loads(open(p).read())) for k, p in json.loads(sys.argv[2]).items()}
out = {"models": {k: {m: v for m, v in metrics(r).items() if m in ("n", "acc", "ece", "brier", "confident_error_rate", "coverage_at_5pct_error")} for k, r in rows.items()}, "pairs": {}}
for a, b in json.loads(sys.argv[3]):
    if a in rows and b in rows:
        out["pairs"][f"{a}-{b}"] = {"acc_micro": paired(rows[a], rows[b], "acc", "micro"), "acc_macro": paired(rows[a], rows[b], "acc", "macro"),
                                    "ece": paired(rows[a], rows[b], "ece", "micro")}
print(json.dumps(out))
'''


def main():
    CODE.mkdir(parents=True, exist_ok=True)
    for n, t in BUNDLE.items(): (CODE / Path(n).name).write_text(t)
    v3 = len(sys.argv) > 1 and sys.argv[1] == "v3"
    rows, pairs, name = (ROWS_V3, PAIRS_V3, "transfer_paired_v3.json") if v3 else ({k: [v] for k, v in ROWS.items()}, PAIRS, "transfer_paired.json")
    have = {}
    for k, vs in rows.items():
        f = next((RADKEV / "runs" / v / "rows.json" for v in vs if (RADKEV / "runs" / v / "rows.json").exists()), None)
        if f: have[k] = str(f)
    p = subprocess.run([str(KEV_PY), "-c", INNER, str(CODE), json.dumps(have), json.dumps(pairs)], env={**os.environ, "PYTHONPATH": f"{KEV_DIR}:{CODE}"},
                       cwd=KEV_DIR, capture_output=True, text=True)
    res = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-1500:]}
    res["missing"] = sorted(set(rows) - set(have)); res["rows"] = have
    (OUT / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
