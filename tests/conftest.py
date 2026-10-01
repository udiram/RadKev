"""Synthetic stand-ins for the raw sources, in the exact on-disk shapes radkev.data reads.

Every text here is invented for the tests; no dataset content is included in this repository.
"""
import json

import pytest

IU_XML = """<?xml version="1.0" encoding="utf-8"?>
<eCitation><MedlineCitation><Article><Abstract>
<AbstractText Label="COMPARISON">None.</AbstractText>
<AbstractText Label="INDICATION">Synthetic case {i}: cough and fever.</AbstractText>
<AbstractText Label="FINDINGS">{findings}</AbstractText>
<AbstractText Label="IMPRESSION">{impression}</AbstractText>
</Abstract></Article></MedlineCitation>
<MeSH>{mesh}</MeSH></eCitation>
"""
IU_CASES = [
    ("The heart is enlarged. Small left pleural effusion.", "Cardiomegaly with left effusion.", ["Cardiomegaly/mild", "Pleural Effusion/left"]),
    ("Lungs are clear. No effusion or pneumothorax.", "No acute cardiopulmonary disease.", ["normal"]),
    ("Right lower lobe airspace opacity.", "Right lower lobe pneumonia.", ["Opacity/lung/base/right", "Pneumonia/right"]),
    ("Calcified granuloma in the left upper lobe. No other abnormality.", "Old granulomatous disease.", ["Calcified Granuloma/lung/upper lobe/left"]),
    ("Dual-chamber pacemaker in place. Heart size normal.", "Pacemaker, no acute findings.", ["Implanted Medical Device"]),
    ("Hyperinflated lungs with flattened diaphragms.", "Emphysema.", ["Pulmonary Emphysema"]),
]


def write_iu(raw, copies=8):
    d = raw / "iu" / "ecgen-radiology"; d.mkdir(parents=True)
    n = 0
    for c in range(copies):
        for findings, impression, mesh in IU_CASES:
            tags = "".join(f"<major>{m}</major>" for m in mesh)
            (d / f"{n}.xml").write_text(IU_XML.format(i=n, findings=findings, impression=impression, mesh=tags)); n += 1
    return n


def write_eurorad(raw, n=60):
    import pandas as pd
    rows = []
    for i in range(n):
        dx = ["Synthetic adenoma", "Synthetic cyst", "Synthetic lymphoma", "Synthetic abscess"][i % 4]
        rows.append({"case_id": i, "age": 40 + i % 30, "gender": "female" if i % 2 else "male",
                     "history": f"Patient {i} presented with abdominal pain. Biopsy confirmed the diagnosis.",
                     "image_finding": f"CT shows a {10 + i % 7} mm hypodense lesion in the left lobe with peripheral enhancement. "
                                      "Surgery was performed. There is no free fluid in the pelvis.",
                     "differential_diagnosis": ["A: Synthetic adenoma", "B: Synthetic cyst", "C: Synthetic lymphoma", "D: Synthetic abscess"],
                     "diagnosis": dx, "section": "Abdominal imaging"})
    (raw / "eurorad").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(raw / "eurorad" / "eurorad.parquet")


def write_medqa(raw, n_train=50, n_test=20):
    import pandas as pd
    (raw / "medqa").mkdir(parents=True)
    for name, n in (("medqa_train.parquet", n_train), ("medqa_test.parquet", n_test)):
        rows = [{"question": f"Synthetic vignette {name} {i}: which drug?", "answer_idx": "ABCD"[i % 4],
                 "options": {"A": f"drug a{i}", "B": f"drug b{i}", "C": f"drug c{i}", "D": f"drug d{i}"}} for i in range(n)]
        pd.DataFrame(rows).to_parquet(raw / "medqa" / name)


@pytest.fixture
def raw(tmp_path):
    r = tmp_path / "raw"
    write_iu(r); write_eurorad(r); write_medqa(r)
    return r


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
