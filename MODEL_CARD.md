# Model card: RadKev-27B and RadKev-9B

Manuscript: *RadKev: An Open-Weight Decision Model for Radiology* (Udbhav Ram and Ran Zhang).

## Summary

| | RadKev-27B | RadKev-9B |
|---|---|---|
| Type | decision model: one forward pass, a probability for every admissible answer | same |
| Fine-tuned from | [Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) (revision `01b8199`) | [Kev-9B](https://huggingface.co/jaredpalmer/kev-9b) (revision `2629c06`) |
| Backbone (frozen) | Qwen3.8-27B (revision `1d4bf0f`) | Qwen3.5-9B-Base (revision `68c46c4`) |
| Trained parameters | rank-16 LoRA on every linear projection (attention, MLP, Gated DeltaNet) + pointer head | same |
| Training | 1 epoch, 4,620 optimizer steps, AdamW, one-cycle schedule, peak learning rate 3e-5, effective batch 8 records, bfloat16; 13.1 h on four RTX A6000 GPUs | same, peak learning rate 2e-5; 3.4 h on four RTX A6000 GPUs |
| Temperature (fitted on the development split) | 1.35 | 1.35 |
| Interface | TypeSafe System One (`POST /v1/systemone`) via `kev.serve`; `radkev.predict` in-process | same |
| Weights | [`ramu9703/radkev-27b`](https://huggingface.co/ramu9703/radkev-27b) | [`ramu9703/radkev-9b`](https://huggingface.co/ramu9703/radkev-9b) |
| License | weights: CC BY-NC-SA 4.0, non-commercial research use (several training sources carry that license); access requires acceptance of these terms; code: Apache-2.0 | same |

## Intended use

- Research on specialized decision models, calibration, selective prediction and categorical decisions made on radiology text.
- Comparison with the released results, and a starting point for further fine-tuning on locally labeled decisions.

## Out of scope

- **Clinical use.** RadKev is not a medical device, has not been prospectively validated, and its outputs must not inform the care
  of a patient.
- Images. RadKev reads text only (reports, case descriptions, clinical indications, examination questions).
- Languages other than English, and institutional conventions not represented in the training data.
- Commercial use (the license of the training data does not permit it).

## Inputs and outputs

Input: a *state* (a string or a dictionary of named sections) and one or more typed questions: `noul` (yes/no), `choice` (named
options) and `score` (ordered levels). Output: for every question, a probability distribution over exactly the admissible options,
with the top choice and its confidence. Questions about the same state are answered in one pass and cannot read each other. Kev
accepts up to 255 options per question; training used states of up to 1,536 tokens.

## Training data

36,109 records from public radiology datasets, plus 1,000 replayed records of Kev's own training data: IU/Open-i is used only for
development and testing; CT-RATE chest CT reports (classifier labels), ReXErr reports with and without injected errors, Eurorad
teaching cases (final diagnosis, subspecialty), RadCases one-liners (ACR Appropriateness Criteria panel and topic), the radiology
questions of MedMCQA and MedQA, and 11,443 records of LLM-labeled questions on imaging orders, triage and follow-up (labeled by
agreement of MedGemma-27B-text and Qwen3.8-27B). Details: [docs/DATA.md](docs/DATA.md) and the manuscript's Methods.

## Evaluation

Radiology benchmark: 14,142 held-out questions in fifteen tasks, of which 11,434 have keys assigned by people. Accuracy in percent,
unweighted mean over tasks; calibration and coverage on the human-assigned questions. 95% confidence intervals of paired differences
in the manuscript and in [`results/manuscript/`](results/manuscript/).

| | RadKev-27B | RadKev-9B | Kev-27B | Kev-9B | Qwen3.8-27B | OpenAI Decisions |
|---|---:|---:|---:|---:|---:|---:|
| Task mean, benchmark (15 tasks) | **80.8** | 77.0 | 75.9 | 70.1 | – | 76.0 |
| Task mean, 13 human-assigned tasks answered by every system | **81.1** | 77.0 | 77.3 | 72.0 | 78.1 | 79.4 |
| Expected calibration error | 0.015 | 0.021 | 0.018 | 0.023 | 0.022 | 0.019 |
| Confident errors (%) | 1.4 | 1.1 | 0.6 | 0.7 | 2.0 | 2.4 |
| Coverage at a 5% error budget (%) | 88.4 | **89.4** | 85.5 | 86.3 | 86.6 | 85.9 |
| Median latency per question (ms)¹ | 128 | 43 | 129 | – | 145 | 175² |

<sub>¹ 60 benchmark records, one request at a time, two RTX A6000 GPUs, bfloat16. A decision model answers all questions of a record
in one pass; Qwen3.8-27B was scored from its option-letter logits, one question per request. Qwen3.8-27B took 234 ms to generate an
answer letter and 16.9 s with reasoning. ² End to end from the institution's network, one request at a time.</sub>

Primary outcome (prespecified), RadKev-27B minus Kev-27B, benchmark task mean: **+4.9 percentage points (95% CI 3.8 to 6.2)**;
over the fourteen human-assigned tasks, +3.6 (2.4 to 5.0). RadKev-27B minus Qwen3.8-27B: +4.7 (3.4 to 6.0); minus the OpenAI
Decisions API: +4.8 (3.2 to 6.4).

**RadCases panel question.** The catch-all option "no ACR Appropriateness Criteria topic applies" was the most frequent training
answer (34%), and the raw outputs of these weights select it by default (RadKev-27B accuracy 43.9%). The manuscript's results
divide each option's probability by its add-one-smoothed frequency among the training-split panel answers and renormalize
(RadKev-27B accuracy 67.4%, Kev-27B 66.7%); apply the same correction, or curate the training frequency of catch-all options, when
using a catch-all answer.

## Limitations and risks

- **External data.** On 4,037 status questions from radiologist-annotated RadGraph-XL reports of another institution, specialization
  lowered the accuracy of RadKev-27B by 1.1 points (−1.6 to −0.6) relative to Kev-27B; hedged findings were frequently classified
  as present.
- **Answer options.** In the options-only control, RadKev-27B selected the correct Eurorad diagnosis in 76.9% of questions without
  the case (Kev-27B 47.7%, chance 24%); its gain in Eurorad diagnosis reflects recognition of the correct option rather than reading
  of the case.
- **Answer space.** The model can only choose among the answers it is offered. Accuracy depends strongly on how similar the
  alternatives are to the correct answer; answer spaces should be checked to contain the correct answer before use.
- **Calibration.** RadKev-27B makes more confident errors than Kev-27B. The temperature should be refitted, and thresholds for
  automation set, on locally labeled cases.
- **LLM-labeled training questions.** The labels of the order, triage and follow-up questions come from agreement of two LLMs
  without validation against human judgment.
- **Contamination.** MedQA, MedMCQA, MMLU, PubMedQA and Eurorad are public and may be part of any backbone's pretraining data.
  Paired differences on identical questions, not absolute accuracies, are the basis of the conclusions.
- **General decision performance.** On Kev's out-of-domain transfer suite, RadKev-27B did not differ detectably from Kev-27B (−0.6
  points, 95% CI −2.3 to 1.2).
- **Populations.** The sources are teaching cases, examination questions and reports from a limited number of institutions and
  countries; performance on other populations, report styles and languages is unknown.
- **Single seed.** Each model was trained once.

## Citation

See [CITATION.cff](CITATION.cff) and the [README](README.md#citation).
