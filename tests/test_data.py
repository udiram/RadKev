import json
import subprocess
import sys

from conftest import read_jsonl

from radkev import data as bd


def build(raw, out, only="iu,eurorad,medqa"):
    subprocess.run([sys.executable, "-m", "radkev.data", "--raw", str(raw), "--out", str(out), "--only", only], check=True, capture_output=True)
    return {s: read_jsonl(out / f"{s}.jsonl") for s in ("train", "dev", "test")}


def test_records_are_valid_kev_requests(raw, tmp_path):
    splits = build(raw, tmp_path / "out")
    assert sum(map(len, splits.values())) > 0
    for recs in splits.values():
        for r in recs:
            assert r["state"] and r["questions"] and r["_meta"]["group_id"]
            for q in r["questions"].values():
                assert q["type"] in ("choice", "noul", "score") and q["instructions"] and q["src"]
                if q["type"] == "choice": assert q["label"] in q["criteria"]
                if q["type"] == "noul": assert isinstance(q["label"], bool)


def test_iu_is_never_trained_on(raw, tmp_path):
    splits = build(raw, tmp_path / "out")
    assert not [r for r in splits["train"] if r["_meta"]["source"] == "iu"]
    assert [r for r in splits["test"] if r["_meta"]["source"] == "iu"]


def test_groups_do_not_cross_splits(raw, tmp_path):
    splits = build(raw, tmp_path / "out")
    owner = {}
    for split, recs in splits.items():
        for r in recs: assert owner.setdefault(r["_meta"]["group_id"], split) == split


def test_held_out_wording_never_in_train(raw, tmp_path):
    splits = build(raw, tmp_path / "out")
    held_out = {bd.DX_T[-1], bd.MCQ_T[-1], bd.NORMAL_T[-1], bd.WHICH_T[-1], bd.SECTION_T[-1]}
    held_out_present = [t.split("{f}")[0] for t in (bd.PRESENT_T[-1],)]
    for r in splits["train"]:
        for q in r["questions"].values():
            assert q["instructions"] not in held_out
            assert not any(q["instructions"].startswith(p) for p in held_out_present)
    seen = {q["instructions"] for s in ("dev", "test") for r in splits[s] for q in r["questions"].values()}
    assert seen & held_out, "dev/test should use the reserved wordings"


def test_build_is_deterministic(raw, tmp_path):
    a = build(raw, tmp_path / "a"); b = build(raw, tmp_path / "b")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_eurorad_states_drop_outcome_sentences(raw, tmp_path):
    splits = build(raw, tmp_path / "out", only="eurorad")
    for recs in splits.values():
        for r in recs:
            text = json.dumps(r["state"]).lower()
            assert "biopsy" not in text and "surgery" not in text


def test_neutral_mcq_hides_answer_position():
    import random
    crit, key = bd.neutral_mcq(random.Random(0), ["right", "w1", "w2", "w3"], 0)
    assert crit[key] == "right" and set(crit) == {"opt_1", "opt_2", "opt_3", "opt_4"}


def test_imaging_only_and_presentation_only():
    text = "A mass is seen on CT. Histology confirmed a schwannoma. The patient was treated surgically."
    assert bd.imaging_only(text) == "A mass is seen on CT."
    assert bd.presentation_only("Fever for three days. An MRI of the spine was requested.") == "Fever for three days."
