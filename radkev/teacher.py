#!/usr/bin/env python3
"""Two-teacher labels for the decisions open data has no labels for: orders/protocols, triage, and extra report questions.

    python -m radkev.teacher tasks --raw DIR --out tasks.jsonl [--per-source 3000]
    python -m radkev.teacher label --model google/medgemma-27b-text-it --tasks tasks.jsonl --out a.jsonl   # GPUs via device_map
    python -m radkev.teacher merge --tasks tasks.jsonl --labels a.jsonl b.jsonl --out DIR             # -> train/dev/test.jsonl
    python -m radkev.teacher predict --model Qwen/Qwen3.8-27B --data test.jsonl --out preds.jsonl     # zero-shot LLM baseline

Each teacher reads the state and a lettered option list and is scored by one forward pass: the next-token probability of
each option letter after the chat prompt (thinking off), so every teacher answer is a full distribution, not a sample.
`merge` keeps a question only when both teachers pick the same option, and trains on the mean of their distributions
(Kev's soft `target`), which also carries their uncertainty into calibration. Splits follow radkev.data's patient/case
hashing, so teacher-labelled dev/test never share a patient with training.
"""
import argparse
import csv
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from radkev import data as bd

LETTERS = "ABCDEFGHIJKLMNOP"

# ---------------------------------------------------------------- question bank (what the teachers label)

URGENCY = ["Routine: no time-sensitive finding", "Needs attention within days", "Urgent: needs same-day action", "Emergent: needs immediate action"]
FOLLOW = {"none": "No further imaging recommended", "short_interval": "Follow-up imaging within 3 months", "long_interval": "Follow-up imaging in more than 3 months",
          "other_test": "Further evaluation with a different imaging test or procedure", "clinical": "Clinical correlation or non-imaging workup only"}
CHANGE = {"improved": "Improved compared with the prior study", "worse": "Worse compared with the prior study", "stable": "Unchanged / stable",
          "mixed": "Some findings better, some worse", "new": "New abnormality not present before"}
ROUTE = {"chest": "Chest / thoracic", "neuro": "Neuroradiology (brain, spine)", "abdominal": "Abdominal / GI", "msk": "Musculoskeletal",
         "cardiovascular": "Cardiovascular", "gu": "Genitourinary", "womens": "Women's imaging / breast", "head_neck": "Head and neck",
         "pediatric": "Paediatric", "interventional": "Interventional radiology"}
EXAMS = {"xr_chest": "Chest radiograph", "ct_head_nc": "CT head without contrast", "cta_chest_pe": "CT pulmonary angiography (PE protocol)",
         "ct_chest": "CT chest", "ct_abd_pelvis_c": "CT abdomen and pelvis with IV contrast", "us_abdomen": "Ultrasound abdomen (incl. RUQ)",
         "us_pelvis": "Ultrasound pelvis / transvaginal", "mri_brain": "MRI brain", "mri_spine": "MRI spine", "xr_msk": "Radiograph of the painful bone or joint",
         "mri_msk": "MRI of a joint or limb", "us_doppler": "Doppler ultrasound of vessels", "mammo_us_breast": "Mammography and/or breast ultrasound",
         "no_imaging": "No imaging indicated"}
CONTRAST = {"none": "No contrast", "iv": "IV contrast", "with_without": "Without and with IV contrast", "na": "Not applicable (radiograph or ultrasound)"}
APPROPRIATE = ["Usually not appropriate", "May be appropriate", "Usually appropriate"]
PRIORITY = ["Routine", "Urgent (same day)", "STAT (immediate)"]

REPORT_Q = [  # (qid, type, instructions templates, criteria, when)
    ("critical", "noul", ["Does this report contain a critical or actionable finding that should be communicated to the ordering clinician right away?",
                          "Is there a finding here that requires urgent notification of the referring team?",
                          "Should the radiologist call the ordering clinician about this report?"], None, None),
    ("urgency", "score", ["How urgent is clinical action based on this report?", "Rate the urgency of the findings in this report.",
                          "How time-sensitive are the findings described?"], URGENCY, None),
    ("follow_up", "choice", ["What follow-up does this report call for?", "Which follow-up recommendation best fits this report?",
                             "What is the appropriate next step after this study?"], FOLLOW, None),
    ("change", "choice", ["How have the findings changed compared with the prior study?", "What is the interval change since the comparison exam?",
                          "Compared with prior imaging, how does this study look?"], CHANGE, "comparison"),
    ("incidental", "noul", ["Does the report describe an incidental finding that needs its own follow-up?",
                            "Is there an incidental finding here that should be tracked?", "Does this report mention an incidental lesion needing follow-up?"], None, None),
    ("route", "choice", ["Which subspecialty reader should this study go to?", "Route this study to the right subspecialty worklist.",
                         "Which radiology section should read this study?"], ROUTE, None),
]
ORDER_Q = [
    ("exam", "choice", ["Which imaging study is most appropriate first for this clinical question?", "What is the best initial imaging exam for this presentation?",
                        "Which exam should be ordered first?"], EXAMS, None),
    ("contrast", "choice", ["Should the most appropriate initial study use contrast?", "What contrast does the best initial study need?",
                            "Choose the contrast for the recommended first exam."], CONTRAST, None),
    ("priority", "score", ["How should this imaging order be prioritised?", "What priority should this order get on the worklist?",
                           "How quickly should this study be performed?"], PRIORITY, None),
    ("appropriate", "score", ["How appropriate is the ordered exam for this clinical question?", "Rate the appropriateness of the requested study.",
                              "Is the ordered imaging study appropriate for this indication?"], APPROPRIATE, "ordered_exam"),
]


CXR_KEYWORDS = {"pleural effusion": r"effusion", "pneumothorax": r"pneumothora", "cardiomegaly": r"cardiomegal|heart size|cardiac silhouette",
                "consolidation": r"consolidat", "pulmonary edema": r"edema|oedema", "atelectasis": r"atelecta", "a lung opacity": r"opacit",
                "pneumonia": r"pneumonia", "a fracture": r"fractur", "a lung lesion (nodule or mass)": r"nodul|mass",
                "a support device (line, tube, pacer)": r"tube|line|catheter|pacer|pacemaker|port", "an enlarged cardiomediastinal silhouette": r"mediastin",
                "pleural thickening or other pleural abnormality": r"pleural thicken|pleural plaque"}
STATUS = {"present": "Reported as present", "absent": "Explicitly reported as absent", "uncertain": "Possible, equivocal or cannot be excluded",
          "not_mentioned": "Not mentioned in the report"}
STATUS_T = ["What does the report say about {f}?", "How does this report characterise {f}?", "Which statement best describes {f} in this report?"]
CXR_POOLS = ("chexpert_plus", "rexgradient", "mimic")


def render_options(q):
    """-> (keys in order, text lines) for the teacher prompt. noul is a two-option yes/no."""
    if q["type"] == "noul": return ["true", "false"], ["A. Yes", "B. No"]
    if q["type"] == "score": return [str(i) for i in range(len(q["criteria"]))], [f"{LETTERS[i]}. {c}" for i, c in enumerate(q["criteria"])]
    keys = list(q["criteria"]); return keys, [f"{LETTERS[i]}. {q['criteria'][k] or k}" for i, k in enumerate(keys)]


def state_text(state):
    if isinstance(state, str): return state
    return "\n".join(f"{k.capitalize()}: {v}" for k, v in state.items() if v)


# ---------------------------------------------------------------- state pools from the sources on disk

def _mimic_reports(raw):
    dirs = ("mimic", "mimic_cxr_reports", "mimic_cxr_jpg")
    sp = bd.locate(raw, dirs, "mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.0.0-split.csv")
    if not (sp and bd.mimic_reports(raw, dirs, peek=True)): return
    with (gzip.open(sp, "rt") if sp.suffix == ".gz" else sp.open(newline="")) as f:
        split = {r["study_id"]: {"validate": "dev"}.get(r["split"], r["split"]) for r in csv.DictReader(f)}
    for subject, study, text in bd.mimic_reports(raw, dirs):
        sec = bd.mimic_sections(text)
        yield {"source": "mimic", "group": f"mimic/{subject}", "split": split.get(study, "train"),
               "exam": sec.get("examination") or "Chest radiograph", "indication": sec.get("indication") or sec.get("history"),
               "comparison": sec.get("comparison"), "findings": sec.get("findings"), "impression": sec.get("impression")}


def _ctrate_reports(raw):
    for name, official in (("train_reports.csv", "train"), ("validation_reports.csv", "test")):
        p = bd.locate(raw, ("ctrate", "ct_rate"), name)
        if not p: continue
        seen = set()
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = re.match(r"(\w+?_\d+)_([a-z])_", r["VolumeName"]); study = m.group(0) if m else r["VolumeName"]
                if study in seen: continue
                seen.add(study); patient = m.group(1) if m else study
                split = official if official == "test" else ("dev" if bd.split_of(f"ctrate/{patient}", dev=0.08, test=0.0) == "dev" else "train")
                yield {"source": "ctrate", "group": f"ctrate/{patient}", "split": split, "exam": "CT chest", "indication": r.get("ClinicalInformation_EN"),
                       "comparison": None, "findings": r.get("Findings_EN"), "impression": r.get("Impressions_EN")}


def _rexgradient_reports(raw):
    """ReXGradient-160K metadata CSVs; column names are matched loosely and logged."""
    for name, split in (("train_metadata.csv", "train"), ("valid_metadata.csv", "dev"), ("test_metadata.csv", "test")):
        p = bd.locate(raw, ("rexgradient", "rexgradient_160k"), name)
        if not p: continue
        csv.field_size_limit(sys.maxsize)
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f); cols = reader.fieldnames or []
            col = lambda *keys: next((c for c in cols if any(k in c.lower() for k in keys)), None)
            cf, ci, cind, ccmp, cpat, cstudy = col("finding"), col("impression"), col("indication", "history", "clinical"), col("comparison"), col("patient"), col("study")
            print(f"  rexgradient {name}: findings={cf} impression={ci} indication={cind} comparison={ccmp} patient={cpat} study={cstudy}", file=sys.stderr)
            seen = set()
            for r in reader:
                sid = r.get(cstudy) if cstudy else None
                if sid in seen: continue
                seen.add(sid)
                yield {"source": "rexgradient", "group": f"rexgradient/{r.get(cpat) or sid}", "split": split, "exam": "Chest radiograph",
                       "indication": r.get(cind) if cind else None, "comparison": r.get(ccmp) if ccmp else None,
                       "findings": r.get(cf) if cf else None, "impression": r.get(ci) if ci else None}


def _chexpert_plus_reports(raw):
    p = bd.locate(raw, ("chexpert_plus",), "df_chexpert_plus*.csv")
    if not p: return
    csv.field_size_limit(sys.maxsize)
    seen = set()
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            patient = r.get("deid_patient_id") or r["path_to_image"].split("/")[1]
            key = (patient, r.get("patient_report_date_order"))
            if key in seen: continue
            seen.add(key)
            official = str(r.get("split", "train")).lower()
            split = "test" if official in ("valid", "validation", "test") else ("dev" if bd.split_of(f"chexpert_plus/{patient}", dev=0.05, test=0.0) == "dev" else "train")
            yield {"source": "chexpert_plus", "group": f"chexpert_plus/{patient}", "split": split, "exam": "Chest radiograph",
                   "indication": r.get("section_clinical_history") or r.get("section_history"), "comparison": r.get("section_comparison"),
                   "findings": r.get("section_findings"), "impression": r.get("section_impression")}


def _eurorad_cases(raw):
    import pandas as pd
    p = raw / "eurorad" / "eurorad.parquet"
    if not p.exists(): return
    for r in pd.read_parquet(p).itertuples():
        yield {"source": "eurorad", "group": f"eurorad/{r.case_id}", "split": bd.split_of(f"eurorad/{r.case_id}"),
               "exam": None, "indication": f"{r.age}-year-old {r.gender}. {bd.imaging_only(r.history)}", "comparison": None,
               "findings": bd.imaging_only(r.image_finding), "impression": None, "section": str(r.section).split(",")[0]}


POOLS = {"mimic": _mimic_reports, "ctrate": _ctrate_reports, "rexgradient": _rexgradient_reports, "chexpert_plus": _chexpert_plus_reports, "eurorad": _eurorad_cases}
EXAM_OF = {"mimic": "xr_chest", "rexgradient": "xr_chest", "chexpert_plus": "xr_chest", "ctrate": "ct_chest"}


def make_tasks(a):
    out, stats = [], Counter()
    for name, pool in POOLS.items():
        rng = random.Random(bd.h(f"teacher:{a.seed}:{name}"))
        items = [x for x in pool(a.raw) if (x["findings"] or x["impression"] or x["indication"])]
        rng.shuffle(items)
        n_rep = n_ord = 0
        for x in items:
            if n_rep >= a.per_source and n_ord >= a.per_source: break
            # report questions: full report (Eurorad cases read as a case summary)
            if n_rep < a.per_source and (x["findings"] or x["impression"]) and x["source"] != "eurorad":
                state = bd.report_state(rng, {"exam": x["exam"], "indication": x["indication"], "comparison": x["comparison"],
                                              "findings": x["findings"], "impression": x["impression"]})
                pool_q = [q for q in REPORT_Q if q[0] != "route" and (q[4] != "comparison" or (x["comparison"] and "none" not in str(x["comparison"]).lower()))]
                for qid, t, instr, crit, _ in rng.sample(pool_q, min(3, len(pool_q))):
                    out.append(task(x, state, qid, t, bd.pick(rng, instr, x["split"]), crit, "report")); stats[f"{name}:{qid}"] += 1
                if name in CXR_POOLS:   # finding status, biased toward findings the report text mentions so every status appears
                    text = f"{x['findings'] or ''} {x['impression'] or ''}".lower()
                    mentioned = [f for f, kw in CXR_KEYWORDS.items() if re.search(kw, text)]
                    f = rng.choice(mentioned) if mentioned and rng.random() < 0.7 else rng.choice(list(CXR_KEYWORDS))
                    out.append(task(x, state, "finding_status", "choice", bd.pick(rng, STATUS_T, x["split"]).format(f=f), STATUS, "cxr_findings"))
                    stats[f"{name}:finding_status"] += 1
                n_rep += 1
            # order questions: the clinical question only, never the findings
            if n_ord < a.per_source and x["indication"] and len(str(x["indication"])) > 15:
                state = {"clinical question": x["indication"]} if rng.random() < 0.6 else f"Indication: {x['indication']}"
                qs = [q for q in ORDER_Q if q[4] != "ordered_exam"]
                for qid, t, instr, crit, _ in rng.sample(qs, 2):
                    out.append(task(x, state, qid, t, bd.pick(rng, instr, x["split"]), crit, "order")); stats[f"{name}:{qid}"] += 1
                # appropriateness of the exam actually ordered, or (half the time) of a different exam, so both ends appear
                ordered = EXAM_OF.get(name)
                if ordered and rng.random() < 0.5: ordered = rng.choice([k for k in EXAMS if k not in (ordered, "no_imaging")])
                if ordered:
                    st = {"clinical question": x["indication"], "ordered exam": EXAMS[ordered]}
                    qid, t, instr, crit, _ = ORDER_Q[3]
                    out.append(task(x, st, qid, t, bd.pick(rng, instr, x["split"]), crit, "order")); stats[f"{name}:{qid}"] += 1
                n_ord += 1
    with open(a.out, "w") as f:
        for i, t in enumerate(out): f.write(json.dumps({"tid": i, **t}, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(out), "by_source_question": dict(sorted(stats.items()))}, indent=2))


def task(x, state, qid, t, instr, crit, family):
    q = {"type": t, "instructions": instr}
    if crit is not None: q["criteria"] = crit if isinstance(crit, list) else dict(crit)
    return {"source": x["source"], "group": x["group"], "split": x["split"], "family": family, "qid": qid, "state": state, "question": q}


# ---------------------------------------------------------------- labelling with one teacher

SYSTEM = ("You are an experienced attending radiologist. Read the case and answer the multiple-choice question with the single "
          "letter of the best option. Answer only from the information given and standard radiology practice.")


class LetterScorer:
    """An instruction-tuned causal LM answering a lettered multiple-choice prompt: P(option) = softmax over the next-token
    logits of the option letters (thinking off). One forward pass per question, batched and length-sorted."""

    # MedGemma-27B opens every answer with a thought channel (<unused94>thought ... <unused95>) that enable_thinking=False does not
    # suppress, so its first-token letter logits score the start of a thought, not an answer. think_off pre-fills a one-line thought
    # and closes the channel, the equivalent of Qwen's enable_thinking=False (post-registration fairness fix; the pre-registered rows
    # are reproducible without it). An EMPTY thought does not work: the model carries on reasoning after <unused95> (letter mass ~0 on
    # 5 MedQA items); this one-line thought gives letter mass 0.90-0.99. Chosen on format compliance only, never on accuracy.
    THOUGHT_OFF = "<unused94>thought\nI will answer with the letter only.<unused95>"

    def __init__(self, model_id, think_off=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
        self.prefill = self.THOUGHT_OFF if think_off and "<unused94>" in self.tok.get_vocab() else ""
        self.letter_ids = []
        for L in LETTERS:
            ids = {self.tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L) if self.tok.encode(v, add_special_tokens=False)}
            self.letter_ids.append(sorted(ids))

    def prompt(self, state, q):
        keys, lines = render_options(q)
        user = f"{state_text(state)}\n\nQuestion: {q['instructions'] if isinstance(q['instructions'], str) else json.dumps(q['instructions'])}\nOptions:\n" + "\n".join(lines) + "\n\nAnswer with one letter."
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try: text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception: text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return keys, text + self.prefill

    def score(self, items, batch=8, max_len=3072, log=""):
        """items: [(key, state, question)] -> {key: {option_key: p}}, streamed in length order."""
        prompts = sorted(((k, *self.prompt(st, q)) for k, st, q in items), key=lambda x: len(x[2]))
        with self.torch.no_grad():
            for i in range(0, len(prompts), batch):
                chunk = prompts[i:i + batch]
                enc = self.tok([c[2] for c in chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.model.device)
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()   # never materialise [B, L, vocab]
                for (k, keys, _), z in zip(chunk, logits):
                    s = self.torch.stack([self.torch.logsumexp(z[ids], 0) for ids in self.letter_ids[:len(keys)]])
                    yield k, dict(zip(keys, self.torch.softmax(s, 0).tolist()))
                if (i // batch) % 50 == 0: print(f"{log}{i + len(chunk)}/{len(prompts)}", file=sys.stderr, flush=True)


    def score_gen(self, items, batch=8, max_len=3072, max_new=16):
        """Like score(), but read the letter where the model actually writes it: greedy-generate up to max_new tokens and take
        the softmax over option letters at the first generated letter (post-registration fairness fix: some instruction
        models, e.g. MedGemma, open with formatting or a phrase, so their first-token letter distribution is not the answer).
        Falls back to the first-step letter distribution when no letter is generated. Yields (key, {option: p}, found)."""
        torch = self.torch
        letter_set = {t: i for i, ids in enumerate(self.letter_ids) for t in ids}
        prompts = sorted(((k, *self.prompt(st, q)) for k, st, q in items), key=lambda x: len(x[2]))
        with torch.no_grad():
            for i in range(0, len(prompts), batch):
                chunk = prompts[i:i + batch]
                enc = self.tok([c[2] for c in chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.model.device)
                out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True, return_dict_in_generate=True,
                                          pad_token_id=self.tok.pad_token_id)
                gen = out.sequences[:, enc["input_ids"].shape[1]:]
                for r, (k, keys, _) in enumerate(chunk):
                    step = next((t for t in range(gen.shape[1]) if int(gen[r, t]) in letter_set and letter_set[int(gen[r, t])] < len(keys)), None)
                    z = out.scores[step if step is not None else 0][r].float()
                    s_ = torch.stack([torch.logsumexp(z[ids], 0) for ids in self.letter_ids[:len(keys)]])
                    yield k, dict(zip(keys, torch.softmax(s_, 0).tolist())), step is not None


def label(a):
    tasks = [json.loads(l) for l in open(a.tasks)]
    done = {json.loads(l)["tid"] for l in open(a.out)} if Path(a.out).exists() else set()
    scorer = LetterScorer(a.model)
    with open(a.out, "a") as f:
        for tid, p in scorer.score([(t["tid"], t["state"], t["question"]) for t in tasks if t["tid"] not in done], a.batch, a.max_len, f"{a.model}: "):
            f.write(json.dumps({"tid": tid, "p": p}) + "\n"); f.flush()


def predict(a):
    """Zero-shot LLM baseline on labelled Kev records: writes {"id": "rad/<line>", "probabilities": {qid: {key: p}}} for
    radkev.evaluate --preds, so the LLM is scored with exactly the same metrics as the decision models."""
    items, n_q = [], {}
    for n, line in enumerate(open(a.data)):
        if getattr(a, "limit", 0) and n >= a.limit: break
        if not line.strip(): continue
        r = json.loads(line); n_q[f"rad/{n}"] = len(r["questions"])
        for qid, q in r["questions"].items(): items.append(((f"rad/{n}", qid), r["state"], q))
    out, found, total = {}, 0, 0
    S = LetterScorer(a.model, getattr(a, "think_off", False))
    stream = (((k, p, True) for k, p in S.score(items, a.batch, a.max_len, f"{a.model}: ")) if a.mode == "first"
              else S.score_gen(items, a.batch, a.max_len))
    for (rid, qid), p, ok in stream:
        out.setdefault(rid, {})[qid] = p; found += ok; total += 1
    if a.mode != "first": print(json.dumps({"letter_found_share": found / max(1, total)}), file=sys.stderr)
    with open(a.out, "w") as f:
        for rid, probs in out.items():
            if len(probs) == n_q[rid]: f.write(json.dumps({"id": rid, "probabilities": probs}) + "\n")


# ---------------------------------------------------------------- agreement merge -> Kev records

def merge(a):
    tasks = {json.loads(l)["tid"]: json.loads(l) for l in open(a.tasks)}
    teachers = []
    for path in a.labels: teachers.append({json.loads(l)["tid"]: json.loads(l)["p"] for l in open(path)})
    by_state, stats = defaultdict(list), defaultdict(Counter)
    for tid, t in tasks.items():
        ps = [T.get(tid) for T in teachers]
        if any(p is None for p in ps): continue
        keys = list(ps[0])
        tops = [max(p, key=p.get) for p in ps]
        agree = len(set(tops)) == 1
        stats[t["qid"]]["agree" if agree else "disagree"] += 1
        if not agree: continue
        mean = {k: sum(p[k] for p in ps) / len(ps) for k in keys}
        q = dict(t["question"]); q["src"] = f"teacher_{t['family']}_{t['qid']}"
        q["label"] = (tops[0] == "true") if q["type"] == "noul" else (int(tops[0]) if q["type"] == "score" else tops[0])
        q["target"] = mean
        stats[t["qid"]][f"label={q['label']}"] += 1
        by_state[(t["group"], t["split"], json.dumps(t["state"], sort_keys=True))].append((t["qid"], q, t["source"]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = {s: (out / f"{s}.jsonl").open("w") for s in ("train", "dev", "test")}
    counts = Counter()
    for (group, split, state), qs in by_state.items():
        rec = {"state": json.loads(state), "questions": {qid: q for qid, q, _ in qs},
               "_meta": {"source": f"teacher_{qs[0][2]}", "group_id": group, "license": "teacher-labelled", "teachers": [Path(p).stem for p in a.labels]}}
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n"); counts[split] += 1
    for f in files.values(): f.close()
    report = {"records": dict(counts), "questions": {k: dict(v) for k, v in stats.items()},
              "agreement": {k: round(v["agree"] / max(1, v["agree"] + v["disagree"]), 3) for k, v in stats.items()}}
    (out / "manifest.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks"); t.add_argument("--raw", type=Path, required=True); t.add_argument("--out", required=True)
    t.add_argument("--per-source", dest="per_source", type=int, default=3000); t.add_argument("--seed", type=int, default=0)
    l = sub.add_parser("label"); l.add_argument("--model", required=True); l.add_argument("--tasks", required=True); l.add_argument("--out", required=True)
    l.add_argument("--batch", type=int, default=8); l.add_argument("--max-len", dest="max_len", type=int, default=3072)
    m = sub.add_parser("merge"); m.add_argument("--tasks", required=True); m.add_argument("--labels", nargs="+", required=True); m.add_argument("--out", required=True)
    pr = sub.add_parser("predict"); pr.add_argument("--model", required=True); pr.add_argument("--data", required=True); pr.add_argument("--out", required=True)
    pr.add_argument("--batch", type=int, default=8); pr.add_argument("--max-len", dest="max_len", type=int, default=3072)
    pr.add_argument("--mode", choices=["first", "gen"], default="first", help="first: letter logits at the first token (pre-registered); gen: at the generated letter")
    pr.add_argument("--limit", type=int, default=0, help="score only the first N records (sanity checks)")
    pr.add_argument("--think_off", action="store_true", help="close the thought channel with a one-line thought (MedGemma; see LetterScorer.THOUGHT_OFF)")
    a = ap.parse_args()
    {"tasks": make_tasks, "label": label, "merge": merge, "predict": predict}[a.cmd](a)


if __name__ == "__main__":
    main()
