# Model card: RadKev-27B and RadKev-9B

Manuscript: *Specializing a Decision Model for Radiology: Comparison with General-Purpose Decision Models and Language Models*
(Udbhav Ram and Ran Zhang).

## Summary

| | RadKev-27B | RadKev-9B |
|---|---|---|
| Type | decision model: one forward pass, a probability for every admissible answer | same |
| Fine-tuned from | [Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) (revision `01b8199`) | [Kev-9B](https://huggingface.co/jaredpalmer/kev-9b) (revision `2629c06`) |
| Backbone (frozen) | Qwen3.8-27B (revision `1d4bf0f`) | Qwen3.5-9B-Base (revision `68c46c4`) |
| Trained parameters | rank-16 LoRA on every linear projection (attention, MLP, Gated DeltaNet) + pointer head | same |
| Training | 1 epoch, 8,521 steps, AdamW, one-cycle schedule, peak learning rate 3e-5, effective batch 8 records, bfloat16; 25.7 h on two RTX A6000 GPUs | same, peak learning rate 2e-5; 10.9 h on one RTX A6000 GPU |
| Temperature (fitted on the development split) | 1.26 | 1.23 |
| Interface | TypeSafe System One (`POST /v1/systemone`) via `kev.serve`; `radkev.predict` in-process | same |
| Weights | [`ramu9703/radkev-27b-v2`](https://huggingface.co/ramu9703/radkev-27b-v2) | [`ramu9703/radkev-9b`](https://huggingface.co/ramu9703/radkev-9b) |
| License | weights: CC BY-NC-SA 4.0, non-commercial research use (several training sources carry that license); access requires acceptance of these terms; code: Apache-2.0 | same |

## Intended use

- Research on specialized decision models, calibration, selective prediction and decisions made on radiology text.
- Comparison with the released results, and a starting point for further fine-tuning on locally labeled decisions.

## Out of scope

- **Clinical use.** RadKev is not a medical device, has not been prospectively or externally validated, and its outputs must not
  inform the care of a patient.
- Images. RadKev reads text only (reports, case descriptions, clinical indications, examination questions).
- Languages other than English, and institutional conventions not represented in the training data.
- Commercial use (the license of the training data does not permit it).

## Inputs and outputs

Input: a *state* (a string or a dictionary of named sections) and one or more typed questions: `noul` (yes/no), `choice` (named
options) and `score` (ordered levels). Output: for every question, a probability distribution over exactly the admissible options,
with the top choice and its confidence. Questions about the same state are answered in one pass and cannot read each other. Kev
accepts up to 255 options per question; training used states of up to 1,536 tokens.

## Training data

67,164 records from public radiology and medical sources, plus 1,000 replayed records of Kev's own training data: Eurorad teaching
cases (final diagnosis, subspecialty routing), CT-RATE chest CT reports (classifier labels for 18 abnormalities), MedMCQA and MedQA
examination questions, and 11,443 records of teacher-labeled questions on imaging orders, triage and follow-up generated from
CheXpert Plus, ReXGradient-160K, CT-RATE and Eurorad text and labeled by agreement of MedGemma-27B-text and Qwen3.8-27B. IU/Open-i,
MMLU, PubMedQA and MedXpertQA were used only for development and testing. Details: [docs/DATA.md](docs/DATA.md).

## Evaluation

Held-out test split: 14,379 records, 26,719 questions, of which 18,744 are human-labeled. The analysis plan was committed before any
test result was read ([docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)); the protocol and its departures are in
[docs/EVALUATION.md](docs/EVALUATION.md). Accuracy in percent; ECE and coverage on human-labeled questions.

| | RadKev-27B | RadKev-9B | Kev-27B | Kev-9B | Qwen3.8-27B | MedGemma-27B-text |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy, human-labeled questions | **82.4** | 79.3 | 78.2 | 75.6 | 80.3 | 73.7 |
| Accuracy, radiology human-labeled questions | **95.4** | 95.0 | 93.0 | 94.0 | 94.9 | 91.3 |
| Accuracy, all tasks (task mean)¹ | **90.1** | 86.9 | 83.3 | 79.1 | 89.1 | 82.0 |
| Expected calibration error | 0.038 | 0.038 | 0.025 | 0.023 | 0.031 | 0.177 |
| Expected calibration error, after recalibration | 0.020 | 0.026 | 0.020 | 0.023 | 0.017 | 0.058 |
| Confident errors (%) | 2.0 | 1.6 | 0.5 | 0.6 | 2.2 | 12.0 |
| Coverage at a 5% error budget (%) | **72.4** | 68.1 | 66.9 | 62.6 | 65.7 | 51.1 |
| Median latency per question (ms)² | 130 | 44 | 132 | 44 | 163 | 153 |

<sub>¹ Includes model-labeled tasks; for the LLMs these measure in part agreement with their own labels. ² 150 test records, one
request at a time, RTX A6000 GPUs, bfloat16. A decision model answers all questions of a record in one pass, and its per-question
latency is the latency per record divided by the number of questions; the LLMs were scored from their option-letter logits, one
question per request. Qwen3.8-27B took 252 ms to generate an answer letter and 14.1 s with reasoning.</sub>

Primary outcome (prespecified), RadKev-27B minus Kev-27B, task-averaged accuracy with the demonstration sample excluded: **+6.7
percentage points (95% CI 5.7 to 7.7)**; over the 16 human-labeled tasks, +3.6 (2.5 to 4.7). Every task and every system:
[`results/tables/`](results/tables/).

## Limitations and risks

- **Model-labeled tasks.** CT-RATE labels come from the dataset's classifier, and the labels of imaging orders, triage and
  follow-up from agreement of two LLMs without validation against human judgment; results on these tasks measure agreement with the
  labeler. The LLM comparators produced the teacher labels, so they are compared with RadKev only on human-labeled questions.
- **Answer options.** In the options-only control, RadKev-27B selected the correct Eurorad diagnosis in 78.0% of questions without
  the case (Kev-27B 47.7%, chance 24%). Part of its gain in case diagnosis derives from regularities of the answer options, and it
  should not be interpreted as improved diagnostic reasoning.
- **Answer space.** The model can only choose among the answers it is offered. Accuracy depends strongly on how similar the
  alternatives are to the correct answer; answer spaces should be checked to contain the correct answer before use.
- **Calibration.** On human-labeled questions, RadKev-27B is less well calibrated than Kev-27B and makes more confident errors.
  The temperature should be refitted, and thresholds for automation set, on locally labeled cases.
- **Contamination.** MedQA, MedMCQA, MMLU, PubMedQA and Eurorad are public and may be part of any backbone's pretraining data.
  Paired differences on identical questions, not absolute accuracies, are the basis of the conclusions.
- **General decision performance.** On Kev's out-of-domain transfer suite, RadKev-27B did not differ detectably from Kev-27B (−0.8
  points, 95% CI −2.3 to 0.8).
- **Populations.** The sources are teaching cases, examination questions and reports from a limited number of institutions and
  countries; performance on other populations, report styles and languages is unknown.
- **Single seed.** Each model was trained once.

## Citation

See [CITATION.cff](CITATION.cff) and the [README](README.md#citation).
