#!/usr/bin/env python3
"""Turn open radiology datasets into Kev training/eval records (one labelled /v1/systemone request per line).

    python -m radkev.data --raw DIR --out data/rad-open [--only iu,eurorad,...]

Record shape (kev.data.load_records): {"state": ..., "questions": {qid: {type, instructions, criteria, label, src}}, "_meta": {...}}
Labels: choice -> option key, noul -> bool, score -> level index.

Sources (each is skipped with a note if its files are missing under --raw):
  iu        Open-i Indiana CXR reports + human MeSH codes      CC BY-NC-ND -> never trained on: 25% dev (model selection), 75% test
  eurorad   wanglab/eurorad-reasoning case vignettes            CC BY-NC-SA -> dx from differential + subspecialty routing
  medmcqa   openlifescienceai/medmcqa, all subjects             Apache-2.0  -> knowledge MCQ (radiology uncapped, other subjects sampled)
  medqa     MedQA USMLE 4-option (GBaker)                       CC-BY-4.0   -> clinical vignette MCQ; official test kept
  mmlu_med / pubmedqa / medxpertqa                              MIT         -> eval only (never trained on)
  ctrate    CT-RATE chest CT reports + 18 abnormality labels    CC BY-NC-SA, gated HF
  mimic     MIMIC-CXR reports + CheXpert labels                 PhysioNet credentialed (private models only)
  chexpert_plus  CheXpert Plus reports + CheXbert labels        Stanford AIMI research use agreement

Splits are by patient / case (hashed), official test splits are kept, and one instruction wording per question kind is
reserved for dev/test so the evaluation also measures robustness to phrasing the model never trained on.
"""
import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Further directories searched for the gated sources (CT-RATE, ReXGradient, CheXpert Plus, MIMIC) when they are not under
# --raw, e.g. a shared data disk: RADKEV_EXTRA_RAW=/data/a:/data/b
EXTRA_ROOTS = [Path(d) for d in os.environ.get("RADKEV_EXTRA_RAW", "").split(os.pathsep) if d]


def locate(raw, dirs, *names):
    """First file called one of `names` under raw/<dir> or an extra root's <dir>, searched recursively. None if absent."""
    for root in [raw, *EXTRA_ROOTS]:
        for d in dirs:
            base = Path(root) / d
            if not base.exists(): continue
            for n in names:
                hit = next(iter(sorted(base.rglob(n))), None)
                if hit: return hit
    return None


def h(s):
    return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:8], "big")


def split_of(group, dev=0.1, test=0.1):
    u = (h(group) % 10_000) / 10_000
    return "test" if u < test else "dev" if u < test + dev else "train"


def pick(rng, templates, split):
    """Last template is held out: dev/test use it half the time, train never does."""
    if split == "train" or len(templates) == 1:
        return rng.choice(templates[:-1] if len(templates) > 1 else templates)
    return templates[-1] if rng.random() < 0.5 else rng.choice(templates[:-1])


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def report_state(rng, sections, kind="report"):
    """Sections dict -> the shapes real callers send: plain text, a dict of sections, or a wrapped document."""
    sections = {k: v.strip() for k, v in sections.items() if v and str(v).strip()}
    text = "\n".join(f"{k.upper()}: {v}" for k, v in sections.items())
    r = rng.random()
    if r < 0.45: return text
    if r < 0.85: return sections
    return {kind: text}


def noul(instr, label, src, crit=None):
    q = {"type": "noul", "instructions": instr, "label": bool(label), "src": src}
    if crit: q["criteria"] = crit
    return q


def choice(instr, crit, label, src):
    assert label in crit, (label, list(crit))
    return {"type": "choice", "instructions": instr, "criteria": crit, "label": label, "src": src}


def neutral_mcq(rng, options, answer_idx):
    """Shuffle options under neutral keys so position and key names carry no information."""
    order = list(range(len(options))); rng.shuffle(order)
    keys = [f"opt_{i + 1}" for i in range(len(options))]
    return {keys[i]: options[j] for i, j in enumerate(order)}, keys[order.index(answer_idx)]


# ---------------------------------------------------------------- report-reading questions (shared by CXR/CT sources)

PRESENT_T = ["Does this report describe {f}?", "Is {f} reported as present?", "According to the report, is there {f}?",
             "Does the radiologist report {f} in this study?", "Based on this report, does the patient have {f}?"]
STATUS_T = ["What does the report say about {f}?", "How does this report characterise {f}?",
            "Which statement best describes {f} in this report?", "Classify the report's mention of {f}."]
STATUS = {"present": "Reported as present", "absent": "Explicitly reported as absent", "uncertain": "Possible, equivocal or cannot be excluded",
          "not_mentioned": "Not mentioned in the report"}
NORMAL_T = ["Is this a normal study with no abnormal findings?", "Does this report describe a normal examination?",
            "Is the study unremarkable?", "Would you classify this report as entirely normal?"]
WHICH_T = ["Which of these findings is described in the report?", "Which finding does this report mention as present?",
           "The report describes exactly one of these findings. Which one?", "Pick the finding that is present according to the report."]


def finding_questions(rng, split, src, labels, vocab, n_noul=2, with_status=False, normal=None):
    """labels: {finding: 'present'|'absent'|'uncertain'|'not_mentioned'} over vocab. Balanced present/absent nouls,
    optional 4-way status choice, optional normal/abnormal, and one 'which finding' choice when something is present."""
    qs = {}
    present = [f for f in vocab if labels.get(f) == "present"]
    negative = [f for f in vocab if labels.get(f) in ("absent", "not_mentioned")]
    picks = []
    if present: picks.append((rng.choice(present), True))
    if negative: picks.append((rng.choice(negative), False))
    rest = [f for f in vocab if labels.get(f) in ("present", "absent", "not_mentioned") and f not in [p for p, _ in picks]]
    rng.shuffle(rest)
    for f in rest[: max(0, n_noul - len(picks))]: picks.append((f, labels[f] == "present"))
    for i, (f, y) in enumerate(picks[:n_noul]):
        qs[f"finding_{i}"] = noul(pick(rng, PRESENT_T, split).format(f=f), y, f"{src}_finding")
    if with_status:
        cand = [f for f in vocab if f in labels]
        unc = [f for f in cand if labels[f] == "uncertain"]
        f = rng.choice(unc) if unc and rng.random() < 0.4 else rng.choice(cand)
        qs["status"] = choice(pick(rng, STATUS_T, split).format(f=f), dict(STATUS), labels[f], f"{src}_status")
    if normal is not None:
        qs["normal"] = noul(pick(rng, NORMAL_T, split), normal, f"{src}_normal")
    neg_pool = [f for f in vocab if labels.get(f) in ("absent", "not_mentioned")]
    if present and len(neg_pool) >= 3 and rng.random() < 0.5:
        y = rng.choice(present)
        opts = [y] + rng.sample(neg_pool, 3)
        crit, key = neutral_mcq(rng, opts, 0)
        qs["which"] = choice(pick(rng, WHICH_T, split), crit, key, f"{src}_which")
    return qs


# ---------------------------------------------------------------- IU / Open-i (evaluation only)

IU_VOCAB = {"cardiomegaly": ["cardiomegaly"], "pleural effusion": ["pleural effusion"], "atelectasis": ["pulmonary atelectasis"],
            "a lung opacity": ["opacity", "airspace disease", "infiltrate", "consolidation"], "pneumonia": ["pneumonia"],
            "pulmonary edema": ["pulmonary edema"], "pneumothorax": ["pneumothorax"], "a lung nodule": ["nodule", "mass"],
            "a calcified granuloma": ["calcified granuloma", "granuloma"], "a rib or spine fracture": ["fractures, bone"],
            "emphysema or COPD": ["emphysema", "pulmonary emphysema", "pulmonary disease, chronic obstructive"],
            "a hiatal hernia": ["hernia, hiatal"], "scoliosis": ["scoliosis"],
            "a line, tube or implanted device": ["catheters, indwelling", "implanted medical device", "surgical instruments", "medical device"]}


def iu(raw, rng):
    files = sorted(glob.glob(str(raw / "iu" / "ecgen-radiology" / "*.xml")))
    for f in files:
        root = ET.parse(f).getroot()
        sec = {a.get("Label"): (a.text or "") for a in root.iter("AbstractText")}
        if len((sec.get("FINDINGS") or "") + (sec.get("IMPRESSION") or "")) < 20: continue
        heads = {m.text.split("/")[0].strip().lower() for m in root.iter("major") if m.text}
        if "no indexing" in heads or "technical quality of image unsatisfactory" in heads: continue
        labels = {f: ("present" if heads & set(terms) else "not_mentioned") for f, terms in IU_VOCAB.items()}
        state = report_state(rng, {"exam": "Chest radiograph", "indication": sec.get("INDICATION"), "comparison": sec.get("COMPARISON"),
                                   "findings": sec.get("FINDINGS"), "impression": sec.get("IMPRESSION")})
        split = "dev" if h(f"iu/{Path(f).stem}") % 4 == 0 else "test"
        qs = finding_questions(rng, split, "iu", labels, list(IU_VOCAB), n_noul=2, normal=("normal" in heads))
        yield split, {"state": state, "questions": qs, "_meta": {"source": "iu", "group_id": f"iu/{Path(f).stem}", "license": "CC-BY-NC-ND-4.0"}}


# ---------------------------------------------------------------- Eurorad case vignettes

DX_T = ["What is the most likely diagnosis?", "Which diagnosis best explains the clinical history and imaging findings?",
        "Given this case, which of the following is the diagnosis?", "Select the most likely final diagnosis for this case.",
        "Based on the history and imaging, what is the leading diagnosis?"]
SECTION_T = ["Which subspecialty reading group should this case be routed to?", "Route this case to the right radiology section.",
             "Which radiology subspecialty does this case belong to?", "Which reading worklist should this study go to?"]
SECTIONS = {"Abdominal imaging": "Liver, bowel, pancreas, abdominal viscera", "Neuroradiology": "Brain, spine and nervous system",
            "Musculoskeletal system": "Bones, joints, soft tissues", "Chest imaging": "Lungs, pleura, mediastinum",
            "Uroradiology & genital male imaging": "Kidneys, urinary tract, male genital organs", "Paediatric radiology": "Children",
            "Head & neck imaging": "Head and neck, ENT, orbits, thyroid", "Cardiovascular": "Heart and vessels",
            "Genital (female) imaging": "Female pelvis and reproductive organs", "Breast imaging": "Breast",
            "Interventional radiology": "Image-guided procedures"}
SECTION_KEYS = {k: re.sub(r"[^a-z]+", "_", k.lower()).strip("_") for k in SECTIONS}


# Sentences that report the answer (pathology, surgery, treatment, outcome) rather than the imaging.
OUTCOME_RE = re.compile(r"biops|patholog|histolog|cytolog|diagnos|confirm|resect|surg|operat|treat|chemo|follow-?up|excis|specimen|immunohisto", re.I)


def imaging_only(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return " ".join(s for s in sentences if s.strip() and not OUTCOME_RE.search(s))


# Sentences naming an imaging test: in an order question they give the answer away ("an MRI of the spine was requested").
IMAGING_RE = re.compile(r"\b(?-i:MRI|MR|CTA?|US|PET|DSA|MRA|MRCP)\b|\b(?:magnetic resonanc|computed tomogra|tomogra|ultraso|sonogra|echogra|"
                        r"echocardiogra|doppler|radiogra|x-?ray|plain film|scan|scintigra|angiogra|mammogra|imaging|fluorosco|urogra|"
                        r"cholangiogra|endoscop|colonoscop)|\bsubsequent", re.I)


def presentation_only(text):
    """History for order questions: imaging_only() minus sentences that name an imaging test, and figure references."""
    text = re.sub(r"\s*\(fig\.?[^)]*\)", "", imaging_only(text), flags=re.I)
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip() and not IMAGING_RE.search(s))


def eurorad(raw, rng):
    import pandas as pd
    p = raw / "eurorad" / "eurorad.parquet"
    df = pd.read_parquet(p)
    for r in df.itertuples():
        opts = [re.sub(r"^[A-Z]\s*:\s*", "", str(o)).strip() for o in r.differential_diagnosis]
        opts = [o for o in dict.fromkeys(opts) if o]
        d = norm(r.diagnosis)
        idx = next((i for i, o in enumerate(opts) if norm(o) == d), None)
        if idx is None: idx = next((i for i, o in enumerate(opts) if norm(o) and (norm(o) in d or d in norm(o))), None)
        findings, history = imaging_only(r.image_finding), imaging_only(r.history)
        if idx is None or len(opts) < 2 or len(findings) < 40: continue
        split = split_of(f"eurorad/{r.case_id}")
        leak = d in norm(findings) or d in norm(history)
        if leak and split != "train": continue   # dev/test measure reasoning, not string matching
        case = {"age": r.age, "sex": r.gender, "clinical history": history, "imaging findings": findings}
        state = case if rng.random() < 0.7 else "\n".join(f"{k.capitalize()}: {v}" for k, v in case.items())
        crit, key = neutral_mcq(rng, opts, idx)
        qs = {"diagnosis": choice(pick(rng, DX_T, split), crit, key, "eurorad_dx")}
        sec = str(r.section).split(",")[0].strip()
        if sec in SECTIONS and rng.random() < 0.5:
            crit = {SECTION_KEYS[k]: (v if rng.random() < 0.6 else None) for k, v in SECTIONS.items()}
            qs["route"] = choice(pick(rng, SECTION_T, split), crit, SECTION_KEYS[sec], "eurorad_route")
        yield split, {"state": state, "questions": qs, "_meta": {"source": "eurorad", "group_id": f"eurorad/{r.case_id}", "license": "CC-BY-NC-SA-4.0", "leak": leak}}


# ---------------------------------------------------------------- medical knowledge MCQ (MedMCQA, MedQA) + eval-only medicine sets

MCQ_T = ["Which option correctly answers the question?", "Choose the correct answer.", "Which of the options is right?",
         "Select the best answer to this medical question."]
MEDMCQA_CAP = 2500   # training records per MedMCQA subject (Radiology uncapped); keeps Dental etc. from dominating


def medmcqa(raw, rng):
    import pandas as pd
    for fname, official in (("medmcqa_train.parquet", "train"), ("medmcqa_val.parquet", "test")):
        p = raw / "medmcqa" / fname
        if not p.exists(): print(f"  medmcqa: {fname} missing, skipped", file=sys.stderr); continue
        df = pd.read_parquet(p)
        per_subject = Counter()
        for r in df.itertuples():
            opts = [r.opa, r.opb, r.opc, r.opd]
            if any(not str(o).strip() for o in opts) or len(set(map(norm, opts))) < 4: continue
            rad = r.subject_name == "Radiology"
            split = official if official == "test" else ("dev" if h(f"medmcqa/{r.id}") % 20 == 0 else "train")
            if split == "train" and not rad:
                if per_subject[r.subject_name] >= MEDMCQA_CAP or h(f"medmcqa-keep/{r.id}") % 5: continue   # ~20% sample, capped per subject
                per_subject[r.subject_name] += 1
            if split == "dev" and not rad and h(f"medmcqa-dev/{r.id}") % 4: continue   # dev: all radiology, a quarter of the rest
            crit, key = neutral_mcq(rng, [str(o).strip() for o in opts], int(r.cop))
            yield split, {"state": {"subject": r.subject_name, "question": r.question.strip()} if rng.random() < 0.5 else {"question": r.question.strip()},
                          "questions": {"answer": choice(pick(rng, MCQ_T, split), crit, key, "medmcqa_rad" if rad else "medmcqa_med")},
                          "_meta": {"source": "medmcqa", "group_id": f"medmcqa/{r.id}", "license": "Apache-2.0", "subject": r.subject_name}}


def medqa(raw, rng):
    """MedQA (USMLE, 4 options; GBaker/MedQA-USMLE-4-options, CC-BY-4.0): official train -> train/dev, official test -> test."""
    import ast

    import pandas as pd
    for fname, official in (("medqa_train.parquet", "train"), ("medqa_test.parquet", "test")):
        p = raw / "medqa" / fname
        if not p.exists(): print(f"  medqa: {fname} missing, skipped", file=sys.stderr); continue
        for i, r in enumerate(pd.read_parquet(p).itertuples()):
            opts = r.options if isinstance(r.options, dict) else ast.literal_eval(str(r.options))
            keys = sorted(opts)
            if r.answer_idx not in keys: continue
            gid = f"medqa/{official}/{i}"
            split = "test" if official == "test" else ("dev" if h(gid) % 10 == 0 else "train")
            crit, key = neutral_mcq(rng, [str(opts[k]) for k in keys], keys.index(r.answer_idx))
            yield split, {"state": {"question": r.question.strip()}, "questions": {"answer": choice(pick(rng, MCQ_T, split), crit, key, "medqa")},
                          "_meta": {"source": "medqa", "group_id": gid, "license": "CC-BY-4.0"}}


MMLU_MED = ["anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine", "college_biology"]


def mmlu_med(raw, rng):
    """MMLU medical subjects, never trained on (Kev keeps MMLU eval-only too): validation -> dev, test -> test."""
    import pandas as pd
    for sub in MMLU_MED:
        for sp, split in (("validation", "dev"), ("test", "test")):
            p = raw / "mmlu" / f"{sub}_{sp}.parquet"
            if not p.exists(): continue
            for i, r in enumerate(pd.read_parquet(p).itertuples()):
                crit, key = neutral_mcq(rng, [str(c) for c in r.choices], int(r.answer))
                yield split, {"state": {"subject": sub.replace("_", " "), "question": r.question.strip()},
                              "questions": {"answer": choice(pick(rng, MCQ_T, split), crit, key, f"mmlu_{sub}")},
                              "_meta": {"source": "mmlu_med", "group_id": f"mmlu/{sub}/{sp}/{i}", "license": "MIT"}}


PUBMEDQA_T = ["Based on the abstract, what is the answer to the research question?", "Does the evidence in this abstract answer yes, no or maybe?",
              "What do the study's results say about the question?"]


def pubmedqa(raw, rng):
    """PubMedQA expert-labelled 1k (MIT), eval only: half dev, half test by PubMed id."""
    import pandas as pd
    p = raw / "pubmedqa" / "pqa_labeled.parquet"
    if not p.exists(): return
    for r in pd.read_parquet(p).itertuples():
        ctx = r.context["contexts"] if isinstance(r.context, dict) else []
        split = "dev" if h(f"pubmedqa/{r.pubid}") % 2 == 0 else "test"
        crit = {"yes": "Yes, the evidence supports it", "no": "No, the evidence does not support it", "maybe": "Maybe: the evidence is mixed or inconclusive"}
        yield split, {"state": {"research question": r.question, "abstract": " ".join(map(str, ctx))},
                      "questions": {"answer": choice(pick(rng, PUBMEDQA_T, split), crit, str(r.final_decision), "pubmedqa")},
                      "_meta": {"source": "pubmedqa", "group_id": f"pubmedqa/{r.pubid}", "license": "MIT"}}


def medxpertqa(raw, rng):
    """MedXpertQA text subset (MIT): expert-level, up to 10 options, eval only (20% dev, 80% test)."""
    import ast

    import pandas as pd
    p = raw / "medxpertqa" / "text_test.parquet"
    if not p.exists(): return
    for r in pd.read_parquet(p).itertuples():
        opts = r.options if isinstance(r.options, dict) else ast.literal_eval(str(r.options))
        keys = sorted(k for k, v in opts.items() if v is not None and str(v).strip())
        if r.label not in keys: continue
        q = re.split(r"\nAnswer Choices:", r.question)[0].strip()
        split = "dev" if h(f"medxpertqa/{r.id}") % 5 == 0 else "test"
        crit, key = neutral_mcq(rng, [str(opts[k]) for k in keys], keys.index(r.label))
        yield split, {"state": {"question": q}, "questions": {"answer": choice(pick(rng, MCQ_T, split), crit, key, "medxpertqa")},
                      "_meta": {"source": "medxpertqa", "group_id": f"medxpertqa/{r.id}", "license": "MIT", "body_system": r.body_system}}


# ---------------------------------------------------------------- CT-RATE (gated on Hugging Face)

def ctrate(raw, rng):
    for rep_name, lab_name, official in (("train_reports.csv", "train_predicted_labels.csv", "train"),
                                          ("validation_reports.csv", "valid_predicted_labels.csv", "test")):
        rep_p, lab_p = locate(raw, ("ctrate", "ct_rate"), rep_name), locate(raw, ("ctrate", "ct_rate"), lab_name)
        if not (rep_p and lab_p): print(f"  ctrate: {rep_name}/{lab_name} missing, skipped", file=sys.stderr); continue
        with lab_p.open(newline="", encoding="utf-8") as f: labels = {row["VolumeName"]: row for row in csv.DictReader(f)}
        seen = set()
        with rep_p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vol = row["VolumeName"]; m = re.match(r"(\w+?_\d+)_([a-z])_", vol)
                study = f"{m.group(1)}_{m.group(2)}" if m else vol
                if study in seen or vol not in labels: continue
                seen.add(study)
                lab = labels[vol]
                vocab = [k for k in lab if k != "VolumeName"]
                ls = {k.lower(): ("present" if str(lab[k]).strip() in ("1", "1.0") else "not_mentioned") for k in vocab}
                patient = m.group(1) if m else vol
                split = official if official == "test" else ("dev" if split_of(f"ctrate/{patient}", dev=0.08, test=0.0) == "dev" else "train")
                state = report_state(rng, {"exam": "CT chest", "clinical information": row.get("ClinicalInformation_EN"), "technique": row.get("Technique_EN"),
                                           "findings": row.get("Findings_EN"), "impression": row.get("Impressions_EN")})
                qs = finding_questions(rng, split, "ctrate", ls, [k.lower() for k in vocab], n_noul=3,
                                       normal=not any(v == "present" for v in ls.values()))
                yield split, {"state": state, "questions": qs, "_meta": {"source": "ctrate", "group_id": f"ctrate/{patient}", "license": "CC-BY-NC-SA-4.0"}}


# ---------------------------------------------------------------- MIMIC-CXR (PhysioNet credentialed; keep its text on approved machines)

CHEXPERT = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
CHEX_NAMES = {"Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation", "Edema": "pulmonary edema",
              "Enlarged Cardiomediastinum": "an enlarged cardiomediastinal silhouette", "Fracture": "a fracture", "Lung Lesion": "a lung lesion (nodule or mass)",
              "Lung Opacity": "a lung opacity", "Pleural Effusion": "pleural effusion", "Pleural Other": "pleural thickening or other pleural abnormality",
              "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax", "Support Devices": "a support device (line, tube, pacer)"}
SECTION_RE = re.compile(r"^\s*([A-Z][A-Z /&]+):", re.M)


def mimic_sections(text):
    parts, last, key = {}, 0, "preamble"
    for m in SECTION_RE.finditer(text):
        parts[key] = text[last:m.start()]; key, last = m.group(1).strip().lower(), m.end()
    parts[key] = text[last:]
    return {k: " ".join(v.split()) for k, v in parts.items() if v.strip()}


def mimic_reports(raw, dirs, peek=False):
    """(subject, study, text) for every MIMIC-CXR report, from mimic-cxr-reports.zip or its extracted files/ tree."""
    zp = locate(raw, dirs, "mimic-cxr-reports.zip")
    tree = None if zp else next((p.parent.parent.parent for p in [locate(raw, dirs, "s5*.txt")] if p), None)
    if peek: return bool(zp or tree)
    pat = re.compile(r"p(\d+)/s(\d+)\.txt$")
    if zp:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                m = pat.search(name)
                if m: yield m.group(1), m.group(2), z.read(name).decode("utf-8", "replace")
    elif tree:
        for f in tree.rglob("s*.txt"):
            m = pat.search(f.as_posix())
            if m: yield m.group(1), m.group(2), f.read_text(errors="replace")


def mimic(raw, rng):
    dirs = ("mimic", "mimic_cxr_reports", "mimic_cxr_jpg")
    lp, sp = locate(raw, dirs, "mimic-cxr-2.0.0-chexpert.csv.gz", "mimic-cxr-2.0.0-chexpert.csv"), locate(raw, dirs, "mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.0.0-split.csv")
    if not (lp and sp and mimic_reports(raw, dirs, peek=True)): print("  mimic: files missing, skipped", file=sys.stderr); return
    opener = lambda p: gzip.open(p, "rt") if p.suffix == ".gz" else p.open(newline="")
    with opener(lp) as f: labels = {row["study_id"]: row for row in csv.DictReader(f)}
    with opener(sp) as f:
        official = {}
        for row in csv.DictReader(f): official[row["study_id"]] = {"validate": "dev"}.get(row["split"], row["split"])
    val = {"1.0": "present", "0.0": "absent", "-1.0": "uncertain", "": "not_mentioned"}
    for subject, study, text in mimic_reports(raw, dirs):
            if study not in labels: continue
            sec = mimic_sections(text)
            if not (sec.get("findings") or sec.get("impression")): continue
            lab = labels[study]
            ls = {CHEX_NAMES[k]: val.get(str(lab.get(k, "")).strip(), "not_mentioned") for k in CHEXPERT}
            split = official.get(study, "train")
            state = report_state(rng, {"exam": sec.get("examination") or "Chest radiograph", "indication": sec.get("indication") or sec.get("history"),
                                       "comparison": sec.get("comparison"), "findings": sec.get("findings"), "impression": sec.get("impression")})
            qs = finding_questions(rng, split, "mimic", ls, list(CHEX_NAMES.values()), n_noul=2, with_status=True,
                                   normal=str(lab.get("No Finding", "")).strip() == "1.0")
            yield split, {"state": state, "questions": qs, "_meta": {"source": "mimic", "group_id": f"mimic/{subject}", "license": "PhysioNet-credentialed"}}


# ---------------------------------------------------------------- CheXpert Plus (Stanford AIMI research use agreement)

CHEXBERT_VAL = {"1.0": "present", "1": "present", "0.0": "absent", "0": "absent", "-1.0": "uncertain", "-1": "uncertain", "": "not_mentioned", "None": "not_mentioned", "nan": "not_mentioned"}


def chexpert_plus(raw, rng):
    """df_chexpert_plus_*.csv (report sections, patient id, split) + CheXbert labels (JSON lines keyed by path_to_image,
    one file for the impression). One record per report; images of the same report share it."""
    dirs = ("chexpert_plus", "chexpertplus", "chexpert-plus")
    table = locate(raw, dirs, "df_chexpert_plus*.csv")
    # CheXbert labels only (JSON lines keyed by path_to_image); never the RadGraph annotation files that sit next to them
    labels_p = locate(raw, dirs, "impression_fixed.json", "*chexbert*impression*.json", "report_fixed.json")
    if not (table and labels_p): print("  chexpert_plus: CheXbert label file missing, skipped (reports still feed teacher.py)", file=sys.stderr); return
    labels = {}
    with labels_p.open() as f:
        for line in f:
            try: r = json.loads(line)
            except ValueError: continue
            if isinstance(r, dict) and r.get("path_to_image"): labels[r["path_to_image"]] = r
    if not labels: print(f"  chexpert_plus: {labels_p.name} has no per-image label lines, skipped", file=sys.stderr); return
    csv.field_size_limit(sys.maxsize)
    seen = set()
    with table.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            patient = row.get("deid_patient_id") or row.get("patient_id") or row["path_to_image"].split("/")[1]
            key = (patient, row.get("patient_report_date_order"), row.get("section_impression", "")[:80])
            lab = labels.get(row.get("path_to_image"))
            if key in seen or lab is None: continue
            seen.add(key)
            if not (row.get("section_findings") or row.get("section_impression")): continue
            ls = {CHEX_NAMES[k]: CHEXBERT_VAL.get(str(lab.get(k)).strip(), "not_mentioned") for k in CHEXPERT if k in lab}
            if len(ls) < 10: continue
            official = str(row.get("split", "train")).lower()
            split = "test" if official in ("valid", "validation", "test") else ("dev" if split_of(f"chexpert_plus/{patient}", dev=0.05, test=0.0) == "dev" else "train")
            state = report_state(rng, {"exam": "Chest radiograph", "indication": row.get("section_clinical_history") or row.get("section_history"),
                                       "comparison": row.get("section_comparison"), "findings": row.get("section_findings"), "impression": row.get("section_impression")})
            qs = finding_questions(rng, split, "chexpertplus", ls, list(ls), n_noul=2, with_status=True,
                                   normal=str(lab.get("No Finding")).strip() in ("1", "1.0"))
            yield split, {"state": state, "questions": qs, "_meta": {"source": "chexpert_plus", "group_id": f"chexpert_plus/{patient}", "license": "Stanford-AIMI-research"}}


SOURCES = {"iu": iu, "eurorad": eurorad, "medmcqa": medmcqa, "medqa": medqa, "mmlu_med": mmlu_med, "pubmedqa": pubmedqa, "medxpertqa": medxpertqa,
           "ctrate": ctrate, "mimic": mimic, "chexpert_plus": chexpert_plus}
OPEN_SOURCES = ("iu", "eurorad", "medmcqa", "medqa", "mmlu_med", "pubmedqa", "medxpertqa")   # downloadable without a login (radkev.fetch)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, required=True, help="directory with one subfolder per source")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated sources")
    ap.add_argument("--cap", default="", help="per-source train caps, e.g. mimic=8000,ctrate=5000")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    caps = {k: int(v) for k, v in (kv.split("=") for kv in a.cap.split(",") if kv)}
    only = [s for s in a.only.split(",") if s] or list(SOURCES)
    splits = defaultdict(list)
    for name in only:
        rng = random.Random(h(f"{a.seed}:{name}"))
        n0 = sum(len(v) for v in splits.values())
        try:
            for split, rec in SOURCES[name](a.raw, rng):
                splits[split].append(rec)
        except FileNotFoundError as e:
            print(f"  {name}: {e}, skipped", file=sys.stderr)
        print(f"{name}: {sum(len(v) for v in splits.values()) - n0} records", file=sys.stderr)
    # group ids must not cross splits
    owner = {}
    for split, recs in splits.items():
        for r in recs:
            g = r["_meta"]["group_id"]
            assert owner.setdefault(g, split) == split, f"group {g} in {owner[g]} and {split}"
    rng = random.Random(a.seed)
    for name, cap in caps.items():
        tr = [r for r in splits["train"] if r["_meta"]["source"] == name]
        if len(tr) > cap:
            keep = set(map(id, rng.sample(tr, cap)))
            splits["train"] = [r for r in splits["train"] if r["_meta"]["source"] != name or id(r) in keep]
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": a.seed, "sources": only, "splits": {}}
    for split in ("train", "dev", "test"):
        recs = splits.get(split, []); rng.shuffle(recs)
        with (a.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        qsrc = Counter(q["src"] for r in recs for q in r["questions"].values())
        manifest["splits"][split] = {"records": len(recs), "questions": sum(qsrc.values()),
                                     "by_source": dict(Counter(r["_meta"]["source"] for r in recs)), "by_task": dict(sorted(qsrc.items()))}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
