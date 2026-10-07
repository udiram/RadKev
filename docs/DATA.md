# Data

RadKev is trained and evaluated only on public datasets (manuscript, Section 2.1, Table 1 and Supplementary Notes S1 and S2).
`radkev.data` and `radkev.teacher` rebuild every record from the original files, deterministically; the counts below and the
state hashes of the released data allow a rebuild to be checked record by record. Released records and labels are described under [Released data](#released-data).

## Sources

| Source | Decision family | Use | Ground truth from | License / access | How to get it |
|---|---|---|---|---|---|
| IU/Open-i chest radiograph reports | report reading | **development and test only** | MeSH indexing by human indexers | CC BY-NC-ND 4.0 | `python -m radkev.fetch` |
| Eurorad case vignettes (`wanglab/eurorad-reasoning`) | case diagnosis, subspecialty routing | train/dev/test | case authors' final diagnosis and section | CC BY-NC-SA 4.0 | `python -m radkev.fetch` |
| MedMCQA (all subjects; radiology uncapped) | radiology and medical knowledge | train/dev; official validation = test | examination answer | Apache-2.0 | `python -m radkev.fetch` |
| MedQA (USMLE, 4 options) | medical knowledge | train/dev; official test = test | examination answer | CC BY 4.0 | `python -m radkev.fetch` |
| MMLU medical subjects, PubMedQA (expert set), MedXpertQA (text) | medical knowledge | **dev/test only** | examination answer / expert annotation | MIT | `python -m radkev.fetch` |
| ReXErr (error-injected MIMIC-CXR reports) | report reading: does the report contain an error? | train/dev; official test = test | known by construction | PhysioNet credentialed (MIMIC-CXR) | physionet.org |
| RadCases (openly available subsets: GPT-3.5 single-sentence patient summaries, Medbullets Step 2/3) | ACR Appropriateness Criteria panel and topic | train/dev/test (by case) | two senior medical students under an attending radiologist, adjudicated | MIT (labels); case texts openly available | the RadCases repository |
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

`src` names the task; [`radkev/compare.py`](../radkev/compare.py) maps tasks to decision families and to the type of
ground truth (human-labeled or model-labeled).

## Splits and controls

- **By patient or case.** Splits hash the patient or case id, so all records of one patient land in one split; the builder
  asserts that no group crosses splits. Official test splits (MedMCQA validation, MedQA test, CT-RATE validation) are retained.
- **Sources used only for evaluation.** IU (25% development, 75% test), MMLU, PubMedQA and MedXpertQA appear only in the
  development and test splits.
- **Withheld wordings.** Every question kind has several instruction templates; one is withheld from training and used for half
  of the development and test questions, so the test also measures robustness to unseen phrasing
  (manuscript, Section 3.5).
- **Neutral options.** Multiple-choice options are shuffled under neutral identifiers (`opt_1`...), so neither position nor option
  identifier carries the answer.
- **No answer in the question.** Eurorad states keep the history and imaging findings and drop sentences about pathology,
  surgery, treatment or outcome; dev/test cases whose findings or history still name the diagnosis are removed. Order questions
  see only the clinical question, never the findings, and sentences that name an imaging test are dropped. These questions are
  used for training only and are not benchmarked.
- **Shape variety.** States arrive as plain text, a dict of sections or a wrapped document, as real callers send them.

## Teacher labels

Imaging orders, triage and follow-up have no public ground truth. `radkev.teacher` generates candidate questions of ten kinds from
reports in CheXpert Plus, ReXGradient-160K and CT-RATE and from clinical indications in these sources and in Eurorad, poses each
as a lettered multiple-choice prompt to two LLM teachers, MedGemma-27B-text and Qwen3.8-27B (reasoning disabled; the answer
distribution is the softmax of the next-token logits over the option letters), and retains a question only if both teachers rank the
same answer first. That answer is the label, and the mean of the two distributions is the soft training target. Of 33,000
candidates, 22,557 (68.4%) were retained for training; the rates per question kind range from 28.1% (contrast) to 84.7% (finding
status) (manuscript, Supplementary Table S1; [`results/manuscript/TableS1_teacher_labels.csv`](../results/manuscript/TableS1_teacher_labels.csv)).

In the first run, the letter probabilities of MedGemma-27B-text were read at the start of its reasoning segment rather than after
it. When the labels were regenerated with the segment closed (22,607 retained), 93.1% of the retained training questions were
retained again and none received a different label (manuscript, Supplementary Note S2). RadKev-27B and RadKev-9B were trained
with the labels of the first run.

Two rules follow from the use of LLM teachers and apply to every result:

1. Teacher-labeled tasks measure **agreement with the teachers**, not correctness, and are reported apart from human-labeled tasks.
2. Both LLM comparators produced these labels, so **no comparison between RadKev and the LLMs is made on model-labeled questions**.

## Counts

Records per source and split, and questions of the radiology benchmark (manuscript, Table 1;
[`results/manuscript/Table1_data.csv`](../results/manuscript/Table1_data.csv)). Knowledge sources contribute only the questions
selected by the radiology filter.

| Source | Task | Ground truth | Train | Dev | Test | Benchmark questions |
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

CheXpert Plus and ReXGradient-160K reports contributed through teacher-labeled questions only.

## Released data

The Hugging Face dataset [`ramu9703/radkev-data`](https://huggingface.co/datasets/ramu9703/radkev-data) (access requires
acceptance of its terms) holds the data of the manuscript's Data availability statement:

| Path | Contents |
|---|---|
| `records/<source>/<split>.jsonl` | question records built from Eurorad, MedMCQA, MedQA, MMLU, PubMedQA and MedXpertQA, each under the license of its source |
| `labels/<source>/<split>.jsonl` | records without text (IU/Open-i, CT-RATE, RadCases, ReXErr, RSNA-RadioQA, RadGraph-XL): identifier, group identifier, SHA-256 of the state and the ground-truth labels |
| `labels/teacher/`, `labels/teacher_regenerated/`, `labels/teacher_candidates.jsonl` | the LLM-labeled questions (training labels and the regenerated labels of Supplementary Note S2) and every candidate with both teachers' distributions |
| `training/{train,dev}.jsonl` | the training and development records of RadKev-27B and RadKev-9B, by state hash and question identifiers |
| `test_index.jsonl` | every held-out test record, with the task, benchmark task and type of ground truth of each question |
| `outputs/` | per-question probabilities, without text, of every system on the test records and in every additional analysis (reasoning sample, options-only control, subspecialty classification before the read, answer space, external test, Kev's transfer suite), and the RadCases panel prior |

Records without text carry the SHA-256 of their state, so a rebuild with `radkev.data` and `experiments/teacher_labels.py` can be
checked record by record. The output files contain the raw model probabilities; the reported results of RadKev-27B and RadKev-9B
apply the RadCases panel prior correction ([EVALUATION.md](EVALUATION.md#radcases-panel-prior-correction-post-hoc)).

## Licenses

Built records inherit the terms of their sources. IU/Open-i is CC BY-NC-ND 4.0 (no derivatives); Eurorad and CT-RATE are CC BY-NC-SA
4.0 (non-commercial, share-alike); CheXpert Plus and ReXGradient-160K are available under access agreements that do not permit
redistribution. Models trained on these records are released for non-commercial research use only.
