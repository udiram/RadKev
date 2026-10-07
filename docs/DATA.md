# Data

RadKev is trained and evaluated only on public datasets (manuscript, Section 2.1, Table 1 and Supplementary Notes S1 and S2).
`radkev.data` and `radkev.teacher` rebuild every record from the original files, deterministically, and
[`results/manifests/`](../results/manifests/) lists the exact counts per source, split and task so that a build can be checked
against ours. Released records and labels are described under [Released data](#released-data).

## Sources

| Source | Decision family | Use | Labels from | License / access | How to get it |
|---|---|---|---|---|---|
| IU/Open-i chest radiograph reports | report reading | **development and test only** | MeSH indexing by human indexers | CC BY-NC-ND 4.0 | `python -m radkev.fetch` |
| Eurorad case vignettes (`wanglab/eurorad-reasoning`) | case diagnosis, subspecialty routing | train/dev/test | case authors' final diagnosis and section | CC BY-NC-SA 4.0 | `python -m radkev.fetch` |
| MedMCQA (all subjects; radiology uncapped) | radiology and medical knowledge | train/dev; official validation = test | examination key | Apache-2.0 | `python -m radkev.fetch` |
| MedQA (USMLE, 4 options) | medical knowledge | train/dev; official test = test | examination key | CC BY 4.0 | `python -m radkev.fetch` |
| MMLU medical subjects, PubMedQA (expert set), MedXpertQA (text) | medical knowledge | **dev/test only** | examination key / expert annotation | MIT | `python -m radkev.fetch` |
| ReXErr (error-injected MIMIC-CXR reports) | report reading: does the report contain an error? | train/dev; official test = test | known by construction | PhysioNet credentialed (MIMIC-CXR) | physionet.org |
| RadCases (openly available subsets: GPT-3.5 one-liners, Medbullets Step 2/3) | ACR Appropriateness Criteria panel and topic | train/dev/test (by case) | two senior medical students under an attending radiologist, adjudicated | MIT (labels); case texts openly available | the RadCases repository |
| RSNA-RadioQA (RSNA Case Collection) | case diagnosis | **test only** (79 questions) | published reference answer; options of the RaR study | as published with RadioRAG | the RadioRAG supplement |
| RadGraph-XL | report reading: status of an annotated observation | **external test only** (4,037 questions) | radiologist annotations | PhysioNet credentialed | physionet.org |
| CT-RATE chest CT reports + 18 abnormality labels | report reading | train/dev; official validation = test | the dataset's classifier labels for 18 abnormalities | CC BY-NC-SA 4.0, gated on Hugging Face | accept terms at [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
| CheXpert Plus, ReXGradient-160K, CT-RATE, Eurorad presentations | imaging orders, triage and follow-up, chest radiograph finding status | teacher-labeled questions | agreement of two LLM teachers | Stanford AIMI research use agreement; ReXGradient research terms (gated) | Stanford AIMI / Redivis; [rajpurkarlab/ReXGradient-160K](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K) |
| MIMIC-CXR reports + CheXpert labels | report reading | supported, not used directly (only through ReXErr) | CheXpert labeler | PhysioNet credentialed | physionet.org |

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
(human-labeled or model-labeled).

## Splits and controls

- **By patient or case.** Splits hash the patient or case id, so all records of one patient land in one split; the builder
  asserts that no group crosses splits. Official test splits (MedMCQA validation, MedQA test, CT-RATE validation) are kept.
- **Sources used only for evaluation.** IU (25% development, 75% test), MMLU, PubMedQA and MedXpertQA appear only in the
  development and test splits.
- **Withheld wordings.** Every question kind has several instruction templates; one is withheld from training and used for half
  of the development and test questions, so the test also measures robustness to unseen phrasing
  (manuscript, Section 3.5).
- **Neutral options.** Multiple-choice options are shuffled under neutral keys (`opt_1`...), so neither position nor key name
  carries the answer.
- **No answer in the question.** Eurorad states keep the history and imaging findings and drop sentences about pathology,
  surgery, treatment or outcome; dev/test cases whose findings or history still name the diagnosis are removed. Order questions
  see only the clinical question, never the findings, and sentences that name an imaging test are dropped.
  [`experiments/leak_sensitivity.py`](../experiments/leak_sensitivity.py) checks what is left: 55 of 520 test order questions
  still mention an imaging test, and on the remaining questions RadKev-27B's agreement with the teachers exceeds that of Kev-27B
  by 22.6 points (95% CI 18.6 to 26.7). Indications taken from Eurorad sometimes name the examination performed, which can reveal
  the answer to order questions; results on these tasks are therefore reported only as agreement (manuscript, Limitations).
- **Shape variety.** States arrive as plain text, a dict of sections or a wrapped document, as real callers send them.

## Teacher labels

Imaging orders, triage and follow-up have no public answer keys. `radkev.teacher` generates candidate questions of ten kinds from
reports in CheXpert Plus, ReXGradient-160K and CT-RATE and from clinical indications in these sources and in Eurorad, poses each
as a lettered multiple-choice prompt to two LLM teachers, MedGemma-27B-text and Qwen3.8-27B (reasoning disabled; the answer
distribution is the softmax of the next-token logits over the option letters), and retains a question only if both teachers rank the
same answer first. That answer is the label, and the mean of the two distributions is the soft training target. Of 33,000
candidates, 22,557 (68.4%) were retained for training; the rates per question kind range from 28.1% (contrast) to 84.7% (finding
status) (manuscript, Supplementary Table S1; [`results/manuscript/TableS1_teacher_labels.csv`](../results/manuscript/TableS1_teacher_labels.csv)).

The teacher prompt of MedGemma-27B-text was later revised (see [EVALUATION.md](EVALUATION.md#departures-from-the-analysis-plan)),
and the labels were regenerated with the revised prompt (22,607 retained). RadKev was trained on the first-run labels; the
regenerated labels of the test split are used only to report agreement.

Two rules follow from the use of LLM teachers and apply to every result:

1. Teacher-labeled tasks measure **agreement with the teachers**, not correctness, and are reported apart from human-labeled tasks.
2. Both LLM comparators produced these labels, so **no comparison between RadKev and the LLMs is made on model-labeled questions**.

## Counts (final models)

Records per source and split, and questions of the radiology benchmark (manuscript, Table 1;
[`results/manuscript/Table1_data.csv`](../results/manuscript/Table1_data.csv)). Knowledge sources contribute only the questions
selected by the radiology filter.

| Source | Task | Key | Train | Dev | Test | Benchmark questions |
|---|---|---|---:|---:|---:|---:|
| IU/Open-i | finding detection (report) | human | – | 886 | 2,848 | 9,218 |
| Eurorad | diagnosis; subspecialty | human | 2,912 | 329 | 346 | 531 |
| MedMCQA | radiology knowledge | human | 5,070 | 303 | 4,151 | 220 |
| MedQA | radiology knowledge | human | 2,420 | 250 | 1,273 | 331 |
| MMLU | radiology knowledge | human | – | 123 | 1,089 | 79 |
| PubMedQA | radiology knowledge | human | – | 469 | 531 | 108 |
| MedXpertQA | radiology knowledge | human | – | 499 | 1,951 | 641 |
| RSNA-RadioQA | diagnosis | human | – | – | 79 | 79 |
| RadCases | imaging appropriateness (ACR) | human | 264 | 29 | 138 | 227 |
| ReXErr | error detection (report) | construction | 6,000 | 500 | 2,708 | 2,708 |
| CT-RATE | finding detection (report) | classifier | 8,000 | 1,911 | 1,564 | – |
| Teacher-labeled | orders, triage, follow-up | LLM agreement | 11,443 | 877 | 626 | – |
| **Total** | | | **36,109** | **6,176** | **17,304** | **14,142** |

CheXpert Plus and ReXGradient-160K reports contributed through teacher-labeled questions only. The counts of earlier training runs
are in [`results/tables/data_counts.md`](../results/tables/data_counts.md) and [`results/manifests/`](../results/manifests/).

## Released data

As stated in the manuscript (Data availability), the following are released:

- the question records built from Eurorad, MedMCQA, MedQA, MMLU, PubMedQA and MedXpertQA, each under the license of its source;
- for IU/Open-i (whose license does not permit derivative works) and for the access-controlled sources (CT-RATE, CheXpert Plus,
  ReXGradient-160K), no record text: the report identifiers, the answer keys, the teacher labels and a script that rebuilds the
  records from the original datasets;
- per-question model outputs (the probability of every answer option, without text) for all systems and analyses.

Location: the Hugging Face dataset [`ramu9703/radkev-data`](https://huggingface.co/datasets/ramu9703/radkev-data) (access requires
acceptance of its terms). Records without text carry the SHA-256 of their state, so a rebuild with `radkev.data` and
`experiments/teacher_labels.py` can be checked record by record; `test_index.jsonl` maps the identifiers `rad/<line>` used by every
output file to source, group identifier and membership of the demonstration sample.

## Licenses

Built records inherit the terms of their sources. IU/Open-i is CC BY-NC-ND 4.0 (no derivatives); Eurorad and CT-RATE are CC BY-NC-SA
4.0 (non-commercial, share-alike); CheXpert Plus and ReXGradient-160K are available under access agreements that do not permit
redistribution. Models trained on these records are released for non-commercial research use only.
