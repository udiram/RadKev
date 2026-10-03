# Model card: RadKev-27B and RadKev-9B

## Summary

| | RadKev-27B | RadKev-9B |
|---|---|---|
| Type | decision model (one forward pass, a probability per allowed answer) | same |
| Fine-tuned from | [Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) (snapshot `01b8199`) | [Kev-9B](https://huggingface.co/jaredpalmer/kev-9b) |
| Backbone (frozen) | Qwen3.8-27B @ `1d4bf0f` | Qwen3.5-9B-Base @ `68c46c4` |
| Trained parameters | rank-16 LoRA (attention, MLP, Gated-DeltaNet projections) + pointer head | same |
| Training | 1 epoch, lr 3e-5, effective batch 8, bf16, 8,521 steps, 25.7 h on 2× RTX A6000 | 1 epoch, lr 2e-5, 8,521 steps, 10.9 h on 1× RTX A6000 |
| Temperature (fitted on dev) | 1.26 | 1.23 |
| Interface | TypeSafe System One (`POST /v1/systemone`) via `kev.serve`; `radkev.predict` in-process | same |
| Weights | [`ramu9703/radkev-27b-v2`](https://huggingface.co/ramu9703/radkev-27b-v2) (gated) | [`ramu9703/radkev-9b`](https://huggingface.co/ramu9703/radkev-9b) (gated) |
| Licence | weights: non-commercial research use (training data includes CC BY-NC-SA 4.0 sources); code: Apache-2.0 | same |

## Intended use

- **Research** on domain-specialised decision models, calibration, selective prediction and clinical text decisions.
- Benchmarking against the released results, and as a starting point for further fine-tuning on your own labelled decisions.

## Out of scope

- **Any clinical use.** RadKev is not a medical device, has not been prospectively or externally validated, and its outputs must
  not inform the care of a patient.
- Images. RadKev reads text only (reports, vignettes, orders, clinical questions).
- Languages other than English, and institutional conventions it was not trained on.
- Commercial use (the training data's terms do not allow it).

## Inputs and outputs

Input: a *state* (string, or a dict of named sections) and one or more typed questions: `noul` (yes/no), `choice` (named
options, optionally described), `score` (ordered levels). Output: for every question a probability distribution over exactly the
allowed options, with the top choice and its confidence. Questions about the same state are answered in one pass and cannot see
each other. States up to 8,192 tokens are accepted; training used states up to 1,536 tokens.

## Training data

67,164 records from open radiology and medical text (IU reports never trained on), plus 1,000 replayed Kev training records:
Eurorad case vignettes (diagnosis, subspecialty routing), CT-RATE chest CT reports (18 NLP-labelled findings), MedMCQA and MedQA,
and 11,443 records of questions on CheXpert Plus, ReXGradient-160K, CT-RATE and Eurorad text labelled by two LLM teachers
(MedGemma-27B-text, Qwen3.8-27B; agreement only). Details, licences and counts: [docs/DATA.md](docs/DATA.md).

## Evaluation

Pre-registered held-out test, read once: 14,379 records, 26,719 questions (18,744 with human answer keys). Paired source-stratified
cluster bootstrap, 95% CIs. Protocol: [docs/EVALUATION.md](docs/EVALUATION.md).

| | RadKev-27B | RadKev-9B | Kev-27B | Qwen3.8-27B | MedGemma-27B-text |
|---|---:|---:|---:|---:|---:|
| Accuracy, all questions | **87.2** | 84.8 | 82.9 | 85.0 | 79.6 |
| Accuracy, human answer keys | **82.4** | 79.3 | 78.2 | 80.3 | 74.1 |
| CXR report reading (IU, human labels) | **95.8** | 95.6 | 93.7 | 95.6 | 93.2 |
| CT report reading (CT-RATE) | 98.7 | 98.4 | 95.3 | 95.9 | 92.7 |
| Case → diagnosis (Eurorad) | **93.9** | 91.0 | 82.7 | 84.4 | 72.5 |
| Radiology knowledge (MedMCQA radiology) | **84.1** | 68.1 | 72.5 | 75.4 | 68.1 |
| Medical knowledge (5 exam sets) | **68.1** | 62.0 | 62.0 | 64.2 | 54.4 |
| ECE, all questions | 0.017 | 0.018 | 0.026 | 0.021 | 0.135 |
| Human-key decisions automatable at ≤5% error | **72.4** | 68.1 | 66.9 | 65.7 | 53.3 |
| Median latency per question (A6000, bf16) | 130 ms | 44 ms | 132 ms | 163 ms scored, 252 ms generated | 153 ms scored |

Primary endpoint, RadKev-27B vs Kev-27B, macro accuracy over tasks: **+6.8 pp [+5.8, +7.7]**. Every task and every system:
[`results/tables/`](results/tables/).

## Limitations and risks

- **Machine-derived labels.** CT-RATE labels come from the dataset's report classifier, and orders/triage labels from two LLMs;
  scores on those families measure agreement with the labeller. The LLM baselines produced the teacher labels, so they are compared
  with RadKev on human-key questions only. MedGemma's teacher votes came from scoring later found to read its hidden thought channel.
- **Contamination.** MedQA, MedMCQA, MMLU, PubMedQA and Eurorad are on the web and may be in any backbone's pretraining. Paired
  differences on identical items are the claim.
- **Calibration.** Calibrated overall, but on human-key questions RadKev-27B's ECE (0.038) is higher than Kev-27B's (0.025); confident
  errors rise from 0.7% to 1.5% overall. Check thresholds on your own data.
- **Narrow transfer.** Specialisation follows the training distribution: general decision skill is kept rather than improved (Kev
  transfer suite −0.8 pp [−2.3, +0.8]).
- **Bias.** The sources are teaching cases, exam questions and reports from a handful of institutions and countries; performance on
  other populations, report styles and languages is unknown.
- **Single seed.** Each run was trained once.

## Citation

See [CITATION.cff](CITATION.cff).
