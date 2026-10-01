"""Paired bootstrap on Kev's out-of-domain transfer suite (general decision skill): does starting from Kev keep it?

    python experiments/transfer_paired.py           # CPU only; reads $RADKEV_HOME/runs/<run>/transfer_*/rows.json

Reads rows.json that kev.benchmark already wrote for each training run's transfer_ft (the fine-tune) and transfer_base (the
released Kev checkpoint it is compared with), and compares them on identical items with Kev's paired bootstrap (clean,
knowable rows; sibling questions resample together). No model is run. Writes accuracies, ECE and deltas only.
"""
import json

from radkev.compare import paired
from radkev.paths import RUNS

ROWS = {"stock9": "v2x9-kev-9b/transfer_base", "r9": "v2x9-kev-9b/transfer_ft", "b9": "v2x9-base-9b/transfer_ft",
        "r9f10": "v2x9f10-kev-9b/transfer_ft", "b9f10": "v2x9f10-base-9b/transfer_ft",
        "stock27": "v2mg-kev-27b/transfer_base", "v2_27": "v2mg-kev-27b/transfer_ft"}
PAIRS = [("r9", "stock9"), ("b9", "stock9"), ("r9", "b9"), ("r9f10", "stock9"), ("b9f10", "stock9"), ("r9f10", "b9f10"), ("v2_27", "stock27")]


def main():
    from kev.metrics import metrics, scored_rows
    have = {k: RUNS / v / "rows.json" for k, v in ROWS.items() if (RUNS / v / "rows.json").exists()}
    rows = {k: scored_rows(json.loads(p.read_text())) for k, p in have.items()}
    keep = ("n", "acc", "ece", "brier", "confident_error_rate", "coverage_at_5pct_error")
    res = {"models": {k: {m: v for m, v in metrics(r).items() if m in keep} for k, r in rows.items()}, "pairs": {}}
    for a, b in PAIRS:
        if a in rows and b in rows:
            res["pairs"][f"{a}-{b}"] = {"acc_micro": paired(rows[a], rows[b], "acc", "micro"), "acc_macro": paired(rows[a], rows[b], "acc", "macro"),
                                        "ece": paired(rows[a], rows[b], "ece", "micro")}
    res["missing"] = sorted(set(ROWS) - set(have))
    out = RUNS / "transfer_paired.json"; out.write_text(json.dumps(res, indent=1)); print(out)


if __name__ == "__main__":
    main()
