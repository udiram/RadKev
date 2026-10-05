# Evaluation

This document summarizes the evaluation of the manuscript (Methods, Sections 2.4 to 2.6, and Supplementary Note S4) and lists
the scripts that produce each result. Terminology follows the manuscript: *human-labeled* questions have answers assigned by
people (IU MeSH indexers, Eurorad case authors, examination boards, PubMedQA annotators); *model-labeled* questions have answers
from the CT-RATE classifier or from agreement of the two LLM teachers.

## Analysis plan

The analysis plan ([ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)) was committed to version control on 26 September 2026, before any test
result was read. It specified the primary model (RadKev-27B), the primary comparator (Kev-27B), the secondary comparators, the
outcomes, the model-selection rule and the statistical method. All other systems and analyses are post hoc.

**Model selection.** As prespecified, RadKev-27B trained on all sources was retained as the primary model, because its development
accuracy was not more than 1.0 percentage point below that of a model trained without the CT-RATE and teacher-labeled data
(86.71% vs 86.74%).

**Primary outcome.** Task-averaged accuracy, the unweighted mean of the accuracies of the 29 test tasks, compared between
RadKev-27B and Kev-27B after excluding the demonstration sample of 420 records (Supplementary Note S4: matched by state, 424 test
records and 594 questions); it was also computed over the 16 human-labeled tasks.

**Prespecified ablations.** The plan also listed the pilot models and a 27B model trained without the CT-RATE and teacher-labeled
data as secondary comparators. They are reported in Supplementary Table S13 (`results/manuscript/TableS13_ablations.csv`): RadKev-27B
exceeded the model trained without these data by 0.7 points on human-labeled questions (95% CI 0.4 to 1.1) and by 5.6 points in
task-averaged accuracy (4.7 to 6.6), so most of the gain on human-labeled questions was obtained without the model-labeled data.

**Secondary outcomes.** Accuracy over all questions, by decision family and on withheld instruction wordings; expected calibration
error (ECE, ten equal-width bins) and Brier score; the rate of confident errors (incorrect with confidence ≥ 0.9); coverage at a
5% error budget (the largest proportion of questions that can be accepted in descending order of confidence while the error among
the accepted questions does not exceed 5%); and general decision performance on Kev's out-of-domain transfer suite (656 questions).

## Systems

Every system was scored on the same 14,379 test records (26,719 questions). Decision models received each record once, with all
of its questions in a single request, in bfloat16 and with their own temperature.

| Role | Systems |
|---|---|
| Specialized decision models | RadKev-27B, RadKev-9B |
| Starting points and scale | Kev-27B, Kev-9B, Kev-4B, Kev-0.8B |
| Other general-purpose decision models | Laya, Laya-typed-decisions, GLiNER2.5-Decide, Julia-1 |
| LLMs, same size | Qwen3.8-27B (backbone of Kev-27B and RadKev-27B), MedGemma-27B-text |
| Initialization ablation | the 9B model fine-tuned from Qwen3.5-9B-Base, at 10% and 100% of the training data |

**LLM scoring.** The LLMs were scored zero-shot with the teacher prompt: the state, the question and its options as a lettered list,
followed by an instruction to answer with one letter. The answer distribution is the softmax of the next-token logits over the
option letters, with reasoning disabled (`radkev.teacher` `LetterScorer`). MedGemma-27B-text was scored with a one-sentence pre-filled
reasoning segment, so that the answer letter is the first generated token. Because the LLMs had produced the teacher labels, they are
compared with the decision models only on human-labeled questions.

**Reasoning sample.** Both LLMs were also run with reasoning enabled on a stratified sample of 1,800 human-labeled questions in 1,569
records (every Eurorad and MedMCQA radiology question, 200 questions from each IU task and 60 from each other human-labeled task,
selected in a fixed hash order). Qwen3.8-27B ran in its thinking mode, and for MedGemma-27B-text the prompt opened its reasoning
segment so that it always reasoned, using vLLM 0.19 in bfloat16, default sampling settings, seed 0 and at most 4,096 reasoning
tokens. "Answer:" was then appended and the answer read from the next-token probabilities of the option letters. When the model
had written an answer after its reasoning (the first option letter in the generated text) that differed from this read-out, the
written answer was taken with probability 0.98; this applied to no question for Qwen3.8-27B and to 7.7% for MedGemma-27B-text.

## Statistical analysis

All comparisons are paired: both systems are evaluated on identical questions, and differences are the first system minus the
second. Confidence intervals (95%) come from a cluster bootstrap with 2,000 resamples, in which test records are resampled within
each source together with all of their questions. One set of resamples is shared by all systems, so every difference, and every
difference between two differences, is computed on the same resamples. Intervals are the 2.5th and 97.5th percentiles; for
task-averaged accuracy the statistic is the mean of the per-task differences, and differences in calibration, confident errors and
coverage are estimated on the first 800 resamples. Two-sided p values from the bootstrap distribution are adjusted by the Holm
procedure within each set of comparisons reported together. Intervals for Kev's transfer suite, which consists of different
questions, use 1,000 resamples, and the answer-space analysis resamples questions (2,000 resamples).

The shared bootstrap and the Holm adjustment are computed by [`experiments/robustness.py`](../experiments/robustness.py) and
[`experiments/robustness2.py`](../experiments/robustness2.py); [`radkev/compare.py`](../radkev/compare.py) implements the
per-script paired bootstrap (1,000 resamples by default) behind `results/comparisons.json`. [`paper/build.py`](../paper/build.py)
reads these outputs and writes every number, table and figure of the manuscript.

## Post hoc analyses

These were added after the primary results were known. Apart from the initialization ablation, none involved training.

| Analysis | What it does |
|---|---|
| Specialization and scale | gain from specialization (RadKev minus Kev at the same size) minus the gain from scale (Kev-27B minus Kev-9B), on identical resamples |
| Initialization ablation | the 9B training repeated from Qwen3.5-9B-Base with a new adapter and head, at Kev's default learning rate (2e-4), on all and on 10% of the training data (6,716 records, 965 steps) |
| Withheld wordings | gain on instruction wordings withheld from training vs wordings seen in training |
| Recalibration | two-fold cross-fitted temperature scaling of every system on the test split |
| Family average and prevalence baseline | primary comparison with the mean over decision families; a baseline that always selects the answer position most often correct in each task |
| Options-only control | every Eurorad diagnosis and MedQA test question posed again with the state replaced by "No case information is available.", instructions and options unchanged |
| Distinctive words | Eurorad diagnosis questions split by whether a distinctive word of the correct option (≥ 6 letters, absent from the other options) occurs in the case text |
| Answer space | 346 Eurorad diagnosis and 200 MedQA questions with the case and key fixed and the other options replaced by random or most similar options from the dataset (2 to 64 options) or by alternatives written by Qwen3.8-27B (Supplementary Note S5) |
| LLMs on the answer spaces | Qwen3.8-27B and MedGemma-27B-text on the answer spaces with at most 16 options (single-letter labels A to P) |
| Latency | 150 test records (293 questions), one request at a time, one NVLink pair of RTX A6000 GPUs, Hugging Face Transformers, bfloat16, first 5 requests of each mode excluded (first request for the LLM with reasoning, 40 questions); the per-question latency of a decision model is its latency per record divided by its number of questions |
| Latency and answer-space size | one question per request with 2 to 128 options on 60 Eurorad cases, first 10 requests excluded |
| Agreement with regenerated teacher labels | agreement of the decision models with the teacher labels of the test split regenerated after the MedGemma-27B-text rescoring |
| External test sets | RadCases (ACR Appropriateness Criteria panel among 11 panels and "no topic applies", 419 questions; topic among 225 topics, 266 questions, decision models only) and RadGraph-XL (status of radiologist-annotated findings in Stanford CT and MRI reports, 4,037 questions), built by `radkev.external` and scored by `experiments/external_tests.py` under the addendum to the analysis plan (Supplementary Note S6) |

The scripts for each analysis are listed in [REPRODUCE.md](REPRODUCE.md#post-hoc-analyses).

## Departures from the analysis plan

- **Demonstration sample.** The plan excluded the demonstration sample from every reported result. It was excluded from the
  primary comparison only; all other analyses use the full test set. Excluding it changed no difference by more than 0.8
  percentage points and no conclusion.
- **Timing.** The first test-scoring job was submitted 26 minutes before the plan was committed; its results were read after the
  commit. The plan is held in a private repository, so its timestamp is not independently verifiable.
- **MedGemma-27B-text prompt.** The prompt was revised after its first test results had been read, so that the answer is the first
  generated token (MedGemma opens its response with a reasoning segment that the template's thinking-off switch does not
  suppress), and a duplicated beginning-of-sequence token introduced by the prompt template was later removed. The revision was
  selected on five MedQA test questions by the probability assigned to option letters, a criterion of format compliance rather than
  accuracy.
- **Bootstrap.** The plan specified 1,000 resamples; 2,000 were used, shared across systems. The Holm adjustment was not part of the
  plan.

## Reporting rules

- Model-labeled tasks are reported apart from human-labeled tasks, and teacher-labeled results are reported only as agreement with
  the teachers.
- No comparison between RadKev and the LLMs is made on model-labeled questions.
- Public examination questions and Eurorad cases may have been part of the pretraining data of every model; paired differences on
  identical questions, not absolute accuracies, are the basis of the conclusions.
