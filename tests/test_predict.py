import json
from pathlib import Path

from radkev import predict

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_examples_are_valid_requests():
    for p in EXAMPLES.glob("*.json"):
        req = json.loads(p.read_text())
        assert req["state"] and req["questions"], p
        rec = predict.to_record(req)
        for qid, q in rec["questions"].items():
            assert "label" in q and q["type"] in ("choice", "noul", "score"), (p, qid)


def test_answer_shapes():
    assert predict.answer({"type": "noul"}, {"false": 0.2, "true": 0.8})["noul"] == 0.8
    a = predict.answer({"type": "choice", "criteria": {"ct": None, "mri": None}}, {"ct": 0.3, "mri": 0.7})
    assert a["choice"] == "mri" and a["confidence"] == 0.7
    s = predict.answer({"type": "score", "criteria": ["low", "mid", "high"]}, {"0": 0.1, "1": 0.2, "2": 0.7})
    assert s["level"] == 2 and abs(s["score"] - 1.6) < 1e-9 and s["legend"][2] == "high"
