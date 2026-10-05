#!/usr/bin/env python3
"""External test sets (post hoc; analysis-plan addendum 2026-10-05). Never used for training.

    python -m radkev.external radcases --out DIR     # downloads the pinned open sources, writes DIR/radcases_{panel,topic}.jsonl
    (RadGraph-XL records: experiments/external_tests.py --sets radgraph_xl, from the Stanford AIMI release you download)

RadCases (Yao et al., 2025): patient one-liners labeled with the ACR Appropriateness Criteria panel and topic by medical
students under an attending radiologist. The release holds only SHA-512 hashes of the one-liners, so the case text is
rebuilt from the two open sources (GPT-3.5 synthetic one-liners; first sentence of each Medbullets USMLE question, split with
radGPT's own sentence splitter) and kept only where the hash matches a label row.

RadGraph-XL (Delbrouck et al., 2024): reports with radiologist-annotated entities. Each record is one report; up to three
observation entities become the four-option status question of the training format, keyed by the radiologists' label.
An entity that is not annotated is never treated as absent, so "not mentioned" is never a key.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from radkev import data as bd

SEED = 20261005
RADCASES = "https://huggingface.co/datasets/michaelsyao/RadCases/resolve/93c9955961ff3f522f353d3d0a9944690de8d7aa"
RADGPT = "https://raw.githubusercontent.com/michael-s-yao/radGPT/74700a8b762232729caec61455a3169c5499b14c"
MEDBULLETS = "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/dc1bc9f6923ea0ecafe2a033346b8f898e8a622c/medbullets"
PANEL_Q = "Which ACR Appropriateness Criteria panel covers the imaging work-up of this patient?"
TOPIC_Q = "Which ACR Appropriateness Criteria topic best matches this patient's presentation?"
NONE_OPT = "None: no ACR Appropriateness Criteria topic applies"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "radkev-build"}), timeout=120) as r:
        return r.read().decode("utf-8")


# radGPT/radgpt/data/utils.py @74700a8 (MIT, University of Pennsylvania), copied verbatim: the hashes depend on it.
def split_into_sentences(text):
    """
    Splits an input text paragraph into its respective sentences. This
    implementation is adapted from Vladimir Fokow's original solution at
    https://stackoverflow.com/questions/4576077
    Input:
        text: the paragraph to split into sentences.
    Returns:
        A list of the sentences in the original input paragraph.
    """
    alphabets = r"([A-Za-z])"
    prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
    starters = (
        r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|"
        r"Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    )
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov|edu|me)"
    digits = r"([0-9])"
    multiple_dots = r"\.{2,}"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(
        multiple_dots,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text
    )
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text
    )
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    text = text.replace(".”", "”.") if "”" in text else text
    text = text.replace(".\"", "\".") if "\"" in text else text
    text = text.replace("!\"", "\"!") if "!" in text else text
    text = text.replace("?\"", "\"?") if "?" in text else text
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def sha512(s):
    return hashlib.sha512(s.encode()).hexdigest()


def radcases(out, check_text=None):
    """check_text: optional set of normalized record states from the RadKev splits; verbatim one-liner hits are dropped."""
    ac = json.loads(get(f"{RADGPT}/radgpt/ac.json"))
    panels, topics = list(ac["panels"]), list(ac["topics"])
    labels = {}
    for subset, fn in (("synthetic", "synthetic.jsonl"), ("medbullets", "usmle.jsonl")):
        for line in get(f"{RADCASES}/{fn}").splitlines():
            if line.strip():
                r = json.loads(line); labels[r["case"]] = (subset, r["panel"], r["topic"])
    texts = [("synthetic", t) for t in (r["case_readable"] for r in csv.DictReader(io.StringIO(get(f"{RADGPT}/radgpt/data/synthetic.csv"))))]
    for fn in ("medbullets_op4.csv", "medbullets_op5.csv"):
        for r in csv.DictReader(io.StringIO(get(f"{MEDBULLETS}/{fn}"))):
            texts.append(("medbullets", next(iter(split_into_sentences(r["question"])))))
    stats = Counter(label_rows=len(labels), source_texts=len(texts))
    cases = {}
    for subset, t in texts:
        k = sha512(t)
        if k not in labels or labels[k][0] != subset: stats["unmatched_text"] += 1; continue
        cases[k] = (subset, t, *labels[k][1:])
    stats["matched_cases"] = len(cases); stats["unmatched_labels"] = len(set(labels) - set(cases))
    rng = random.Random(SEED)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fp, ft = (out / "radcases_panel.jsonl").open("w"), (out / "radcases_topic.jsonl").open("w")
    by_subset = defaultdict(Counter)
    for k in sorted(cases):
        subset, text, panel, topic = cases[k]
        if check_text is not None and bd.norm(text) in check_text: stats["overlap_with_radkev_splits"] += 1; continue
        meta = {"source": f"radcases_{subset}", "id": f"radcases/{k[:16]}", "group_id": f"radcases/{k[:16]}", "subset": subset,
                "panel": panel, "topic": topic, "license": "RadCases labels MIT; text from its open sources"}
        if "," not in panel:
            opts = panels + [NONE_OPT]; idx = len(panels) if panel == "None" else panels.index(panel)
            crit, key = bd.neutral_mcq(rng, opts, idx)
            fp.write(json.dumps({"state": text, "questions": {"panel": bd.choice(PANEL_Q, crit, key, f"radcases_panel_{subset}")},
                                 "_meta": {**meta, "none": panel == "None"}}, ensure_ascii=False) + "\n")
            by_subset[subset]["panel"] += 1; by_subset[subset]["panel_none"] += panel == "None"
        else: stats["multi_panel_excluded"] += 1
        if topic != "None" and "," not in topic and topic in topics:
            crit, key = bd.neutral_mcq(rng, topics, topics.index(topic))
            ft.write(json.dumps({"state": text, "questions": {"topic": bd.choice(TOPIC_Q, crit, key, f"radcases_topic_{subset}")},
                                 "_meta": meta}, ensure_ascii=False) + "\n")
            by_subset[subset]["topic"] += 1
        elif topic != "None": stats["topic_excluded_multi_or_unknown"] += 1
    fp.close(); ft.close()
    rep = {"stats": dict(stats), "by_subset": {k: dict(v) for k, v in by_subset.items()}, "n_panels": len(panels) + 1, "n_topics": len(topics)}
    (out / "radcases_manifest.json").write_text(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------- RadGraph-XL

STATUS_OF = {"definitely present": "present", "definitely absent": "absent", "uncertain": "uncertain"}


def entity_status(label):
    """'Observation::definitely present' / 'OBS-DP' / 'Observation::uncertain' ... -> present|absent|uncertain|None"""
    s = str(label).lower()
    if s.startswith(("anatomy", "anat")): return None
    for k, v in STATUS_OF.items():
        if k in s: return v
    return {"obs-dp": "present", "obs-da": "absent", "obs-u": "uncertain"}.get(s)


# observation spans that are generic qualifiers rather than findings ("What does the report say about normal?")
GENERIC = {"normal", "unremarkable", "stable", "clear", "intact", "unchanged", "negative", "appropriate", "patent", "midline",
           "well", "good", "improved", "grossly", "within normal limits", "limits", "satisfactory", "expected", "similar", "change",
           "changes", "new", "increased", "decreased", "large", "small", "mild", "moderate", "severe", "minimal", "acute", "chronic"}


def dygie_docs(rows, modality_of):
    """RadGraph-XL rows in DyGIE format (dataset, doc_key, sentences, ner): token lists per sentence, ner [start, end, label]
    with document-level inclusive token indices -> docs for radgraph_records."""
    for r in rows:
        sents = r["sentences"]; toks = [t for s in sents for t in s]; ents = []
        for sent in r["ner"]:
            for a, b, lab in sent: ents.append({"tokens": " ".join(toks[a:b + 1]), "label": lab})
        yield {"doc_key": f"{r['dataset']}/{r['doc_key']}", "text": " ".join(toks), "entities": ents, "modality": modality_of(r["dataset"])}


def radgraph_records(docs, exclude=lambda text: False, max_q=3):
    """docs: iterable of {"doc_key", "text", "entities": [{"tokens": str, "label": str}], "modality"?, "split"?}.
    exclude(text) -> True drops the report (overlap with text the models were trained on)."""
    rng = random.Random(SEED)
    stats = Counter()
    for d in sorted(docs, key=lambda d: str(d["doc_key"])):
        stats["reports"] += 1
        if exclude(d["text"]): stats["excluded_overlap"] += 1; continue
        by = defaultdict(list)
        for e in d["entities"]:
            st = entity_status(e["label"]); span = str(e["tokens"]).strip()
            if not st or len(span) < 3 or span.lower() in GENERIC: continue
            if span.lower() not in [x.lower() for x in by[st]]: by[st].append(span)
        # an entity whose span text also carries another status in the same report is ambiguous: drop it everywhere
        seen = Counter(x.lower() for v in by.values() for x in v)
        by = {k: [x for x in v if seen[x.lower()] == 1] for k, v in by.items()}
        picks = []
        for st in ("present", "absent", "uncertain"):
            if by.get(st): picks.append((rng.choice(by[st]), st))
        rng.shuffle(picks)
        if not picks: stats["no_usable_entity"] += 1; continue
        qs = {}
        for i, (span, st) in enumerate(picks[:max_q]):
            crit = {k: v for k, v in bd.STATUS.items()}
            qs[f"status_{i}"] = bd.choice(bd.pick(rng, bd.STATUS_T, "test").format(f=span), crit, st, f"radgraphxl_status_{d.get('modality') or 'unknown'}")
            stats[f"q_{d.get('modality')}"] += 1
            stats[f"key_{st}"] += 1
        stats["records"] += 1
        yield {"state": d["text"], "questions": qs, "_meta": {"source": "radgraph_xl", "id": f"radgraphxl/{d['doc_key']}",
                                                             "group_id": f"radgraphxl/{d['doc_key']}", "modality": d.get("modality"),
                                                             "split": d.get("split")}}
    radgraph_records.stats = dict(stats)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("radcases"); r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "radcases": print(json.dumps(radcases(a.out), indent=1))


if __name__ == "__main__":
    main()
