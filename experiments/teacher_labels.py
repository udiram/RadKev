"""Two-teacher labels for the decisions open data has no labels for (orders/protocols, triage, follow-up, finding status).

    CUDA_VISIBLE_DEVICES=0,1 python experiments/teacher_labels.py --per-source 1500

Builds teacher tasks from the report and case pools under $RADKEV_HOME/raw (CT-RATE, ReXGradient-160K, CheXpert Plus,
Eurorad; MIMIC-CXR if present), labels them with MedGemma-27B-text-it and then Qwen3.8-27B on the visible GPUs (each 27B
teacher spreads over the pair with device_map="auto"), and keeps only the questions both teachers answer the same way
(radkev.teacher merge) -> $RADKEV_HOME/data/teacher/{train,dev,test}.jsonl. Resumable: each teacher appends to its label file
and skips finished tasks. MedGemma is gated on Hugging Face: accept its terms and log in (`hf auth login`) first.
"""
import argparse
import json
import subprocess
import sys
import time

from radkev.paths import DATA, HOME, KEV_PY, MEDGEMMA_27B, QWEN38_27B, RAW

TEACHERS = {"medgemma": MEDGEMMA_27B, "qwen38": QWEN38_27B}
WORK = HOME / "teacher"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-source", default="1500"); ap.add_argument("--batch", default="8")
    ap.add_argument("--teachers", default="", help="name=hf_id,name=hf_id (default: MedGemma-27B-text-it + Qwen3.8-27B)")
    a = ap.parse_args()
    teachers = dict(kv.split("=", 1) for kv in a.teachers.split(",") if kv) or TEACHERS
    WORK.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y%m%d-%H%M%S"), "teachers": teachers}
    save = lambda: (WORK / "teacher.json").write_text(json.dumps(report, indent=2))
    tasks = WORK / "tasks.jsonl"
    if not tasks.exists():
        p = subprocess.run([str(KEV_PY), "-m", "radkev.teacher", "tasks", "--raw", str(RAW), "--out", str(tasks), "--per-source", a.per_source],
                           capture_output=True, text=True)
        report["tasks"] = p.stdout[-3000:] if p.returncode == 0 else f"failed: {p.stderr[-2000:]}"; save()
        if p.returncode != 0: sys.exit(1)
    for name, model in teachers.items():   # sequential: each 27B teacher needs the whole GPU pair
        with open(WORK / f"{name}.log", "a") as log:
            rc = subprocess.run([str(KEV_PY), "-m", "radkev.teacher", "label", "--model", model, "--tasks", str(tasks),
                                 "--out", str(WORK / f"{name}.jsonl"), "--batch", a.batch], stdout=log, stderr=subprocess.STDOUT).returncode
        report[f"label_{name}"] = "ok" if rc == 0 else f"exit {rc}: {(WORK / f'{name}.log').read_text()[-2000:]}"; save()
    if all(report[f"label_{n}"] == "ok" for n in teachers):
        p = subprocess.run([str(KEV_PY), "-m", "radkev.teacher", "merge", "--tasks", str(tasks), "--labels", *[str(WORK / f"{n}.jsonl") for n in teachers],
                            "--out", str(DATA / "teacher")], capture_output=True, text=True)
        report["merge"] = "ok" if p.returncode == 0 else f"failed: {p.stderr[-2000:]}"
    save()
    print(json.dumps({k: v for k, v in report.items() if k.startswith(("label_", "merge"))}))
    sys.exit(0 if report.get("merge") == "ok" else 1)


if __name__ == "__main__":
    main()
