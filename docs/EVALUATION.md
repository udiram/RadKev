# Evaluation

This document summarizes the evaluation of the manuscript (Methods, Sections 2.4 to 2.6, and Supplementary Note S4) and lists the
scripts that produce each result. Every number is generated from the aggregate job outputs by the manuscript build
([`paper/`](../paper/)); the job scripts are in [`experiments/final/`](../experiments/final/).

## Analysis plan

The models, the radiology benchmark and the outcomes were specified in two addenda to the analysis plan, written before any test
result of the final models was read ([ANALYSIS_PLAN_ADDENDUM.md](ANALYSIS_PLAN_ADDENDUM.md)). The analysis plan was not publicly
registered. Analyses added later are identified as post hoc in the manuscript.

## Benchmark

The benchmark comprises 14,142 held-out questions in fifteen tasks: three IU/Open-i tasks (finding present, normal study, which
finding), Eurorad diagnosis and subspecialty classification, the RadCases ACR Appropriateness Criteria panel and topic, RSNA-RadioQA,
ReXErr error detection, and one task for each examination source (the MedMCQA radiology subject and the radiology questions of the
other MedMCQA subjects, MedQA, MedXpertQA, MMLU and PubMedQA). 11,434 questions have ground-truth labels generated or verified by human annotators; the ground truth of the 2,708 ReXErr
questions is known by construction. CT-RATE (classifier labels) is reported separately as agreement with the classifier, and the LLM-labeled
questions are not benchmarked.

## Systems

| System | Role | Scoring |
|---|---|---|
| RadKev-27B, RadKev-9B | specialized decision models | each record once, all of its questions in one request; temperature from the development split |
| Kev-27B, Kev-9B | starting points (specialization) and scale | as above |
| Qwen3.8-27B | the backbone of Kev-27B queried as an LLM | zero-shot, one question per prompt, softmax of the next-token logits over the option letters, reasoning disabled; questions with at most 16 options |
| OpenAI Decisions API (gpt-6-luna) | hosted general-purpose decision model | each record once, all of its questions in one request; probabilities as returned (two decimals), not recalibrated |
| Jev (jev-1.13.0, TypeSafe System One API) | hosted general-purpose decision model whose interface Kev reproduces | each record once, all of its questions in one request in the benchmark's own format; probabilities as returned (two decimals), not recalibrated |

## Outcomes and statistics

- **Primary:** task-averaged accuracy (unweighted mean of the fifteen task accuracies), RadKev-27B minus Kev-27B.
- **Secondary:** the mean over the fourteen human-labeled tasks; accuracy pooled over questions and on every task; RadKev-27B
  minus each LLM on the questions both answered; expected calibration error, Brier score, confident errors (incorrect with
  confidence at least 0.9) and coverage at a 5% error budget, as scored and after two-fold cross-fitted temperature recalibration.
- **Intervals:** paired, from one shared cluster bootstrap with 2,000 resamples in which test records are resampled within each
  source together with all of their questions. Two-sided p values from the bootstrap, Holm-adjusted across the fifteen tasks for
  the primary comparison.

## RadCases panel prior correction (post hoc)

The residual option of the RadCases panel question ("no ACR Appropriateness Criteria topic applies") was the most frequent panel
answer in the training split (88 of 258 questions, 34%), and the specialized models selected it preferentially (RadKev-27B: 70.7%
of the questions whose ground truth was a panel; accuracy 43.9%). Their answer distributions on this question were corrected for the training
answer frequencies (Saerens et al., *Neural Computation* 2002): each option's probability is divided by its add-one-smoothed
relative frequency among the training-split panel answers, and the distribution is renormalized. The correction uses only training
labels, involves no retraining, and was applied to RadKev-27B and RadKev-9B, which were trained on RadCases, not to the other
systems. All reported results for the RadKev models include it (`experiments/final/radcases_prior.py`, `eval_v3_prior.py`).

## Additional analyses (post hoc)

| Analysis | Script | Manuscript |
|---|---|---|
| Specialization vs scale (difference of differences on shared resamples) | `eval_v3.py` | Results 3.3 |
| Kev's out-of-domain transfer suite (656 questions) | `transfer_paired.py` | Results 3.3 |
| Withheld vs seen instruction wordings | `eval_v3.py` | Results 3.5 |
| Subspecialty classification before the read (imaging findings removed) | `preread_route.py` | Results 3.5 |
| Options-only control (case removed) and distinctive words of the correct option | `blind_v3.py` | Results 3.5 |
| LLMs with reasoning on a sample of 1,675 questions | `llm_reasoning.py` (`v3`), `eval_v3_prior.py` | Results 3.2 |
| OpenAI Decisions API | `openai_decisions.py` | Results 3.2 |
| Jev | `jev_decisions.py`, `eval_v3_prior.py` | Results 3.2 |
| Latency, one request at a time (60 records) | `latency_bench.py` | Results 3.2 |
| Size and composition of the answer space | `answer_space3.py` | Results 3.6 |
| RadGraph-XL external test (4,037 questions: 2,991 definite and 1,046 hedged findings) | `radgraph_xl_build.py`, `external_tests.py`, `external_analyse_node.py`, `radgraph_xl_errors_v3.py`, `radgraph_xl_definite.py` | Results 3.7 |
| External test for the hosted decision models | `radgraph_xl_hosted.py` | Results 3.7, Supplementary Note S6 |
| Answer space for the hosted decision models | `answer_space_hosted.py` | Results 3.6 |
| Reasoning-sample analysis, resampled within source | `reasoning_v3_fix.py` | Results 3.2, Supplementary Note S4 |

## External test

RadGraph-XL comprises radiologist-annotated reports, predominantly CT and MRI, from an institution not represented in training.
Each annotated observation was posed as a question on its status. The main external analysis uses the 2,991 definite findings
(present or absent), with the "uncertain" option removed from every answer and the distribution renormalized; the 1,046 hedged
findings are analyzed separately, and all 4,037 questions are reported in Supplementary Table S12
([`results/manuscript/TableS12_external.csv`](../results/manuscript/TableS12_external.csv)).

| Definite findings (2,991) | Accuracy (%) | RadKev minus system (95% CI) |
|---|---:|---|
| RadKev-27B | 90.2 | – |
| Kev-27B | 91.3 | −1.1 (−1.6 to −0.6) |
| Qwen3.8-27B | 90.3 | −0.1 (−0.8 to 0.7) |
| Jev | 89.3 | 0.9 (0.3 to 1.6) |
| OpenAI Decisions | 81.9 | 8.3 (7.3 to 9.3) |
| RadKev-9B | 87.5 | – |
| Kev-9B | 85.2 | 2.4 (1.5 to 3.3), RadKev-9B minus Kev-9B |

On the hedged findings, Qwen3.8-27B was the most accurate system (55.4%); every decision model, including the hosted ones,
predominantly answered "present".
