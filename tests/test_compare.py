import json
import subprocess
import sys

import pytest

from radkev import compare
from radkev import data as bd
from radkev import teacher as te


def test_families():
    assert compare.family("iu_finding") == "report_cxr_human"
    assert compare.family("ctrate_normal") == "report_ct"
    assert compare.family("eurorad_dx") == "case_diagnosis"
    assert compare.family("teacher_report_route") == "routing"
    assert compare.family("teacher_report_urgency") == "triage_followup"
    assert compare.family("teacher_order_exam") == "orders_protocols"
    assert compare.family("medmcqa_rad") == "radiology_knowledge"
    assert compare.family("mmlu_anatomy") == "medical_knowledge"
    assert compare.family("something_else") == "other"
    assert compare.HUMAN >= compare.RADIOLOGY_HUMAN


def test_held_out_wording_matchers():
    pats = compare.heldout_matchers()
    assert any(p.match(bd.PRESENT_T[-1].format(f="a lung nodule")) for p in pats)
    assert any(p.match(te.ORDER_Q[0][2][-1]) for p in pats)
    assert not any(p.match(bd.PRESENT_T[0].format(f="a lung nodule")) for p in pats)
    assert not any(p.match(bd.MCQ_T[0]) for p in pats)


def records_and_preds(tmp_path, n=60):
    """Synthetic labelled records and two predictors' probabilities: 'good' is right more often than 'weak'."""
    recs, good, weak = [], [], []
    for i in range(n):
        label = "opt_1" if i % 2 else "opt_2"
        recs.append({"state": f"Synthetic report {i}.", "questions": {
            "answer": {"type": "choice", "instructions": bd.MCQ_T[0], "criteria": {"opt_1": "first", "opt_2": "second"}, "label": label, "src": "medqa"},
            "finding": {"type": "noul", "instructions": bd.PRESENT_T[-1].format(f="an effusion"), "label": bool(i % 3), "src": "iu_finding"}},
            "_meta": {"source": "synthetic", "group_id": f"g{i}"}})
        other = "opt_2" if label == "opt_1" else "opt_1"
        g = {label: 0.9, other: 0.1} if i % 10 else {label: 0.4, other: 0.6}
        w = {label: 0.6, other: 0.4} if i % 3 else {label: 0.3, other: 0.7}
        y = bool(i % 3)
        good.append({"id": f"rad/{i}", "probabilities": {"answer": g, "finding": {"true": 0.85 if y else 0.1, "false": 0.15 if y else 0.9}}})
        weak.append({"id": f"rad/{i}", "probabilities": {"answer": w, "finding": {"true": 0.55, "false": 0.45}}})
    data = tmp_path / "test.jsonl"; data.write_text("".join(json.dumps(r) + "\n" for r in recs))
    for name, rows in (("good", good), ("weak", weak)):
        (tmp_path / f"{name}.preds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return data


def test_evaluate_and_compare_end_to_end(tmp_path):
    """radkev.evaluate --preds and radkev.compare with Kev's metrics and bootstrap (skipped without the Kev environment)."""
    pytest.importorskip("kev.metrics")
    pytest.importorskip("sklearn")
    data = records_and_preds(tmp_path)
    runs = tmp_path / "runs"
    for name in ("good", "weak"):
        subprocess.run([sys.executable, "-m", "radkev.evaluate", "--preds", str(tmp_path / f"{name}.preds.jsonl"), "--data", str(data),
                        "--out", str(runs / name)], check=True, capture_output=True)
        s = json.loads((runs / name / "summary.json").read_text())
        assert s["overall"]["n"] == 120 and "auroc" in s["tasks"]["iu_finding"]
    out = tmp_path / "comparisons.json"
    subprocess.run([sys.executable, "-m", "radkev.compare", str(runs), "--models", "good,weak", "--reference", "weak", "--out", str(out),
                    "--data", str(data), "--samples", "200"], check=True, capture_output=True)
    c = json.loads(out.read_text())
    assert c["models"]["good"]["overall"]["acc"] > c["models"]["weak"]["overall"]["acc"]
    d = c["vs_reference"]["good"]["overall_micro_acc"]
    assert d["delta"] > 0 and d["ci95"][0] <= d["delta"] <= d["ci95"][1]
    assert set(c["models"]["good"]["wording"]) == {"seen", "held_out"}
    assert set(c["models"]["good"]["families"]) == {"medical_knowledge", "report_cxr_human"}
