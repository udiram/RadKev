import json
import subprocess
import sys

from conftest import read_jsonl

from radkev import teacher as te


def test_render_options():
    assert te.render_options({"type": "noul"}) == (["true", "false"], ["A. Yes", "B. No"])
    keys, lines = te.render_options({"type": "score", "criteria": ["Routine", "Urgent"]})
    assert keys == ["0", "1"] and lines == ["A. Routine", "B. Urgent"]
    keys, lines = te.render_options({"type": "choice", "criteria": {"ct": "CT head", "none": None}})
    assert keys == ["ct", "none"] and lines == ["A. CT head", "B. none"]


def test_merge_keeps_agreements_and_averages(tmp_path):
    state = {"clinical question": "Synthetic: sudden severe headache."}
    tasks = [
        {"tid": 0, "source": "eurorad", "group": "eurorad/1", "split": "train", "family": "order", "qid": "exam", "state": state,
         "question": {"type": "choice", "instructions": "Which exam first?", "criteria": {"ct_head_nc": "CT head", "mri_brain": "MRI brain"}}},
        {"tid": 1, "source": "eurorad", "group": "eurorad/1", "split": "train", "family": "order", "qid": "priority", "state": state,
         "question": {"type": "score", "instructions": "Priority?", "criteria": ["Routine", "Urgent", "STAT"]}},
        {"tid": 2, "source": "ctrate", "group": "ctrate/9", "split": "test", "family": "report", "qid": "critical", "state": "FINDINGS: x",
         "question": {"type": "noul", "instructions": "Critical?"}},
    ]
    a = [{"tid": 0, "p": {"ct_head_nc": 0.9, "mri_brain": 0.1}}, {"tid": 1, "p": {"0": 0.1, "1": 0.2, "2": 0.7}}, {"tid": 2, "p": {"true": 0.8, "false": 0.2}}]
    b = [{"tid": 0, "p": {"ct_head_nc": 0.7, "mri_brain": 0.3}}, {"tid": 1, "p": {"0": 0.1, "1": 0.6, "2": 0.3}}, {"tid": 2, "p": {"true": 0.6, "false": 0.4}}]
    for name, rows in (("tasks", tasks), ("a", a), ("b", b)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "merged"
    subprocess.run([sys.executable, "-m", "radkev.teacher", "merge", "--tasks", str(tmp_path / "tasks.jsonl"),
                    "--labels", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--out", str(out)], check=True, capture_output=True)
    train, test = read_jsonl(out / "train.jsonl"), read_jsonl(out / "test.jsonl")
    assert len(train) == 1 and set(train[0]["questions"]) == {"exam"}            # priority: the teachers disagree -> dropped
    q = train[0]["questions"]["exam"]
    assert q["label"] == "ct_head_nc" and abs(q["target"]["ct_head_nc"] - 0.8) < 1e-9 and q["src"] == "teacher_order_exam"
    assert test[0]["questions"]["critical"]["label"] is True
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["agreement"] == {"exam": 1.0, "priority": 0.0, "critical": 1.0}


def test_make_tasks_order_questions_never_see_findings(raw, tmp_path):
    out = tmp_path / "tasks.jsonl"
    subprocess.run([sys.executable, "-m", "radkev.teacher", "tasks", "--raw", str(raw), "--out", str(out), "--per-source", "20"],
                   check=True, capture_output=True)
    tasks = read_jsonl(out)
    orders = [t for t in tasks if t["family"] == "order"]
    assert orders
    for t in orders:
        assert "lesion" not in json.dumps(t["state"]).lower()                    # the synthetic findings text
