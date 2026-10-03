# Prespecified analysis plan

This is the analysis plan of the study, reproduced verbatim below the rule. The manuscript calls it the prespecified analysis plan.
It was committed to the private development repository on 26 September 2026, before any test result was read; it was not
registered in a public registry, so its timestamp is not independently verifiable. The first test-scoring job was submitted
26 minutes before the commit, and its results were read after it.

Run names in the plan: v2 = `v2mg` (RadKev-27B); v1med = RadKev-27B trained without the CT-RATE and teacher-labeled data;
v0 = the pilot. `paper/handoff/RESULTS.md` was the internal results file, and the "playground" was an internal demonstration, whose
420 test records form the demonstration sample. The departures from the plan are listed in
[EVALUATION.md](EVALUATION.md#departures-from-the-analysis-plan).

---

# Pre-registration of the final analysis (written 2026-09-26, before any held-out test result was read)

Fixed at this point:
- the development results of v0 (pilot) and v1med are known;
- v2 is still training;
- no model has been scored on the held-out test split. The playground's 420-item sample is excluded from every reported result.

## Primary model
**RadKev-27B = v2** (Kev-27B delta fine-tune on every source).

v1med becomes primary only if v2's calibrated dev accuracy, macro-averaged over the tasks that both runs' dev sets share, is more than 1.0 percentage point below v1med's. The decision is made on dev results alone and recorded in `paper/handoff/RESULTS.md` before the test tables are generated.

## Primary comparison
Primary model vs stock Kev-27B on the held-out test split:
- accuracy per decision family and overall (macro over tasks), with a paired, source-stratified cluster bootstrap (1,000 resamples, 95% percentile CI);
- a family "improves" only if the CI excludes 0.

## Secondary comparisons (same test items, same metrics)
- **Baselines:** Kev-9B stock and pilot, Laya, and zero-shot Qwen3.8-27B and MedGemma-27B-text (option-letter probabilities, thinking off).
- **Ablations:** v1med (no report or teacher data) and v0 (pilot).
- **Calibration:** ECE (10 bins), Brier score, confident-error rate (p >= 0.9 and wrong), and coverage at <= 5% error.
- **Robustness:** accuracy on questions that use the held-out wording vs the seen wording.
- **Latency:** median and p95 per request.

## Reporting rules
- Teacher-labelled families (orders/protocols, triage/follow-up, CXR finding status) are always reported apart from families with human answer keys, and never as ground truth.
- Qwen3.8-27B and MedGemma-27B produced those teacher labels, so on teacher-labelled families their scores measure agreement with themselves. They are shown for completeness only, and no RadKev-vs-LLM claim is made on those families.
- Public benchmarks carry a possible-pretraining-contamination caveat; the paired deltas are the claim.
- No test-driven changes: if a result is bad, it is reported as it is.
