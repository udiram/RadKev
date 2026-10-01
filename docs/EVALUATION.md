# Evaluation protocol

## What was fixed in advance

The final analysis was [pre-registered](PREREGISTRATION.md) on 2026-09-26, before any model was scored on the held-out test:
the primary model (RadKev-27B = run `v2mg`), the primary comparison (vs the released Kev-27B, per decision family and macro over
tasks), the secondary comparisons, the metrics and the reporting rules. The selection rule between candidate runs was decided on
dev alone. The test split was then read once. Everything added afterwards is labelled post hoc below and in every table.

## Metrics

All systems are scored by the same code ([`radkev/evaluate.py`](../radkev/evaluate.py), Kev's metric implementation) on the same
26,719 questions. Every system returns a full probability distribution over the allowed options, so every metric applies to all.

| Metric | Definition |
|---|---|
| Accuracy | top-probability option = label; micro over questions and macro over tasks |
| ECE | expected calibration error of the top-label confidence, 10 equal-width bins |
| Brier | squared error of the full distribution |
| Confident errors | share of questions answered wrongly with top probability ≥ 0.9 |
| Coverage at 5% error | the largest share of questions, taken in order of confidence, whose error rate is ≤ 5% |
| AUROC | yes/no tasks with both classes present |

**Uncertainty.** Differences use a paired, source-stratified cluster bootstrap (1,000 resamples, 95% percentile intervals) in which
all questions about one record are resampled together ([`radkev/compare.py`](../radkev/compare.py)). Single accuracies in the
figures carry Wilson 95% intervals.

**Decision families** map each task to one of: CXR report reading (human labels, IU), CXR report reading (machine labels), CT report
reading, case → diagnosis, subspecialty routing, radiology knowledge, medical knowledge, orders/protocols, triage/follow-up.
**Human answer keys** are the families labelled by people (IU MeSH coders, Eurorad authors, exam boards): 18,744 test questions.
CT-RATE and teacher labels are machine-derived.

## Scoring the LLMs

LLMs are scored the way a decision model is: the chat prompt lists the options as letters, and the probability of each option is
the softmax over the next-token logits of the option letters (thinking off; [`radkev/teacher.py`](../radkev/teacher.py)
`LetterScorer`). That gives every LLM a full distribution and one forward pass per question.

A post hoc check found that this underrates MedGemma-27B-text. MedGemma opens every response with a hidden thought channel
(`<unused94>thought ... <unused95>`) that its thinking-off template does not suppress, so the first-token letter logits score the
start of a thought, not an answer (MedQA 42.7% against a published ~87%).

| MedGemma scoring, first 300 MedQA test questions | Accuracy | Answer letter found |
|---|---:|---:|
| first-token letters (pre-registered) | 44% | — |
| generate 16 tokens, read the first letter | 45% | 1% |
| pre-fill an *empty* thought: the model keeps reasoning | 28% | 9% |
| pre-fill a one-line thought, "I will answer with the letter only." (`--think_off`) | 68% | 99% |
| thinking on, up to 2,048 tokens (first 100 questions; median 847 tokens, ~34 s each) | 72% | 99% |

On the full MedQA test (1,273 questions) the one-line thought takes MedGemma from 42.7% to 69.1%.

The one-line thought was chosen on five items by how much probability lands on option letters (0.90–0.99), not by accuracy. The
corrected row is the MedGemma row in every table; the pre-registered row (78.1% overall) is kept and marked. Re-reading Qwen3.8-27B's
answers where it actually generates them changed nothing (85.0% both ways). Both MedGemma numbers remain below its developers' report,
so MedGemma results describe the model without reasoning under this prompt. Reproduce with [`experiments/llm_scoring.py`](../experiments/llm_scoring.py);
results in [`results/studies/llm_scoring_round*.json`](../results/studies/).

## Latency

[`experiments/latency.py`](../experiments/latency.py): 150 random held-out test records, RTX A6000 GPUs (27B models split over one NVLink
pair), bf16, Hugging Face transformers for every model, one request at a time, CUDA-synchronised timing, warm-up excluded. A decision
model answers every question about a record in one pass. The LLM is timed three ways: option-letter scoring (one forward pass per
question; this is how its accuracy is measured), generating the answer letter (thinking off), and reasoning (thinking on, 40 questions).
[`experiments/latency_scaling.py`](../experiments/latency_scaling.py) asks 1 to 14 questions about each of 30 IU reports.

Read these numbers for what they are: with one question, option scoring costs about the same as a decision model. The large gaps are
against generating the answer and against reasoning, which is how LLMs are usually deployed.

## Post hoc analyses

Added after the pre-registered test read; none of them changed RadKev or its selection.

- **Landscape.** Generalist decision models Kev-0.8B and Kev-4B, Laya-typed-decisions, Julia-1 and GLiNER2.5-Decide, zero-shot
  ([`experiments/decision_baselines.py`](../experiments/decision_baselines.py)). Questions a model cannot take (e.g. more than 20
  options for Julia-1) get a uniform distribution and are counted.
- **Initialisation ablation.** Two 9B arms on identical data, steps, seed and replay: from released Kev-9B (lr 2e-5) or from plain
  Qwen3.5-9B-Base with a fresh adapter and head (Kev's from-scratch lr 2e-4), at 100% and 10% of the training mix. The learning rates
  differ because a fresh head needs the higher one; this is a confound. One seed per arm.
- **General skill.** Kev's out-of-domain transfer suite (656 questions), fine-tune vs its starting checkpoint, paired
  ([`experiments/transfer_paired.py`](../experiments/transfer_paired.py)).
- **Order-question leakage**, the corrected LLM scoring and the latency studies above.

## Deviations from the pre-registration

- **Demo items were not removed.** The pre-registration says the 420 test records sampled for an internal demo (60 from each
  open source) would be excluded from every result. They were not removed from the scored test file: the reported test
  includes all 14,379 records, those 420 among them (2.9% of records). The demo showed stock Kev-27B and the pilot fine-tune on
  them, before the final model existed; no training or selection used them.
- **MedGemma's scoring** was corrected post hoc (above); the pre-registered row is kept alongside.
- **Added analyses** (below) were not in the plan and are labelled post hoc.

## Reporting rules we kept

- Teacher-labelled families are reported apart from human-key families, and never as ground truth.
- No RadKev-vs-LLM claim on teacher-labelled families, because the LLMs produced those labels.
- Paired deltas on identical items are the claim; absolute scores on web-published benchmarks may be inflated by pretraining.
- No test-driven changes: negative results (routing, calibration on human keys) are reported as they are.
