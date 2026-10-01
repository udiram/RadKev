# Data

RadKev is trained and evaluated only on public datasets. This repository ships **no dataset text**: `radkev.data` and
`radkev.teacher` rebuild every record from the original files, deterministically, and [`results/manifests/`](../results/manifests/)
lists the exact counts per source, split and task so you can check your build against ours.

## Sources

| Source | Decision family | Use | Labels from | Licence / access | How to get it |
|---|---|---|---|---|---|
| IU / Open-i chest X-ray reports | report reading | **dev/test only, never trained on** | human MeSH coding | CC BY-NC-ND 4.0 | `python -m radkev.fetch` |
| Eurorad case vignettes (`wanglab/eurorad-reasoning`) | case → diagnosis, subspecialty routing | train/dev/test | case authors' diagnosis and section | CC BY-NC-SA 4.0 | `python -m radkev.fetch` |
| MedMCQA (all subjects; radiology uncapped) | radiology and medical knowledge | train/dev; official validation = test | exam keys | Apache-2.0 | `python -m radkev.fetch` |
| MedQA (USMLE, 4 options) | medical knowledge | train/dev; official test = test | exam keys | CC BY 4.0 | `python -m radkev.fetch` |
| MMLU medical subjects, PubMedQA (expert set), MedXpertQA (text) | medical knowledge | **dev/test only** | exam / expert keys | MIT | `python -m radkev.fetch` |
| CT-RATE chest CT reports + 18 abnormality labels | report reading | train/dev; official validation = test | the dataset's report classifier (NLP) | CC BY-NC-SA 4.0, gated on Hugging Face | accept terms at [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
| CheXpert Plus, ReXGradient-160K, CT-RATE, Eurorad presentations | orders/protocols, triage/follow-up, CXR finding status | teacher-labelled questions | two LLM teachers, agreement only | Stanford AIMI research use; ReXGradient research terms (gated) | Stanford AIMI / Redivis; [rajpurkarlab/ReXGradient-160K](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K) |
| MIMIC-CXR reports + CheXpert labels | report reading | supported, **not used** for the released models | CheXpert labeller | PhysioNet credentialed | physionet.org |

`python -m radkev.fetch --list` prints where each file is expected under `$RADKEV_HOME/raw`. Gated sources can also live on a
shared disk: `RADKEV_EXTRA_RAW=/data/a:/data/b` adds search roots.

## What a record is

One line per record, in Kev's training shape: the System One request plus a `label` on every question.

```jsonc
{"state": {"exam": "CT chest", "findings": "...", "impression": "..."},     // text, a dict of sections, or {"report": text}
 "questions": {
   "finding_0": {"type": "noul",   "instructions": "Does this report describe pleural effusion?", "label": true,  "src": "ctrate_finding"},
   "which":     {"type": "choice", "instructions": "Which finding does this report mention as present?",
                 "criteria": {"opt_1": "emphysema", "opt_2": "...", "opt_3": "...", "opt_4": "..."}, "label": "opt_3", "src": "ctrate_which"}},
 "_meta": {"source": "ctrate", "group_id": "ctrate/train_1234", "license": "CC-BY-NC-SA-4.0"}}
```

`src` names the task; [`radkev/compare.py`](../radkev/compare.py) maps tasks to decision families and to answer-key type
(human vs machine).

## Splits and controls

- **By patient or case.** Splits hash the patient or case id, so all records of one patient land in one split; the builder
  asserts that no group crosses splits. Official test splits (MedMCQA validation, MedQA test, CT-RATE validation) are kept.
- **Never-trained sources.** IU (25% dev for model selection, 75% test), MMLU, PubMedQA and MedXpertQA appear only in dev/test.
- **Held-out wordings.** Every question kind has several instruction templates; the last one is never used in training and is
  used for half of dev/test questions. The test reports accuracy on seen vs unseen wordings
  ([`results/tables/wording_test.md`](../results/tables/wording_test.md)).
- **Neutral options.** Multiple-choice options are shuffled under neutral keys (`opt_1`...), so neither position nor key name
  carries the answer.
- **No answer in the question.** Eurorad states keep the history and imaging findings and drop sentences about pathology,
  surgery, treatment or outcome; dev/test cases whose findings or history still name the diagnosis are removed. Order questions
  see only the clinical question, never the findings, and sentences that name an imaging test are dropped.
  [`experiments/leak_sensitivity.py`](../experiments/leak_sensitivity.py) checks what is left: 55 of 520 test order questions
  still mention an imaging test, and RadKev's gain holds on the leak-free rest (+22.6 pp [+18.6, +26.7] vs Kev-27B).
- **Shape variety.** States arrive as plain text, a dict of sections or a wrapped document, as real callers send them.

## Teacher labels

Orders, protocols, triage, follow-up and critical-result questions have no open labels. `radkev.teacher` writes them as
lettered multiple-choice prompts over report and case pools, scores each with two open LLMs by one forward pass (the softmax
over option-letter logits, thinking off), and keeps a question only when both teachers' top answers agree. Training uses the
mean of the two distributions as Kev's soft `target`. Agreement rates per question kind are in
[`results/tables/teacher_agreement.md`](../results/tables/teacher_agreement.md) (from 0.28 for contrast to 0.85 for finding status).

Teachers: MedGemma-27B-text-it and Qwen3.8-27B. Two caveats follow from this and are reported with every result:

1. Teacher-labelled families measure **agreement with the teachers**, not correctness, and are always reported apart from
   human-key families.
2. Both LLM baselines produced those labels, so **no RadKev-vs-LLM claim is made on teacher-labelled families**; LLM
   comparisons use human-key questions only. MedGemma's votes also came from the pre-registered first-token scoring, which later
   turned out to read its hidden thought channel ([EVALUATION.md](EVALUATION.md#scoring-the-llms)); re-labelling is future work.

## Counts (released models)

| Suite | Train records | Dev records | Test records | Test questions |
|---|---:|---:|---:|---:|
| `rad-open` (IU, Eurorad, MedMCQA, MedQA, MMLU, PubMedQA, MedXpertQA) | 47,721 | 5,765 | 12,189 | 18,744 |
| `rad-gated` (CT-RATE) | 8,000 | 1,911 | 1,564 | 6,950 |
| `teacher` | 11,443 | 877 | 626 | 1,025 |
| **Total** | **67,164** | **8,553** | **14,379** | **26,719** |

Per source and task: [`results/tables/data_counts.md`](../results/tables/data_counts.md) and [`results/manifests/`](../results/manifests/).
CheXpert Plus reports contributed through teacher questions only (its CheXbert label file was not part of the download), and
MIMIC-CXR was not used.

## Licences of what you build

The records you build inherit their sources' terms. In particular IU is no-derivatives and Eurorad and CT-RATE are
non-commercial share-alike: keep built records for research, and do not redistribute them. Models trained on them are
released for non-commercial research use only.
