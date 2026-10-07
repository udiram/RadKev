# Prespecified analysis plan

This is the analysis plan of the study, reproduced verbatim below the rule. The manuscript calls it the prespecified analysis plan.
It was committed to the private development repository on 26 September 2026, before any test result was read; it was not
registered in a public registry, so its timestamp is not independently verifiable. The first test-scoring job was submitted
26 minutes before the commit, and its results were read after it.

Run names in the plan: v2 = `v2mg` (RadKev-27B); v1med = RadKev-27B trained without the CT-RATE and teacher-labeled data;
v0 = the pilot. `paper/handoff/RESULTS.md` was the internal results file, and the "playground" was an internal demonstration, whose
420 test records form the demonstration sample. The final models, benchmark and outcomes were specified in
the addenda in [ANALYSIS_PLAN_ADDENDUM.md](ANALYSIS_PLAN_ADDENDUM.md).

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

---

## Addendum to the analysis plan: external tests (written 2026-10-05, before any external-test result was read)

The main analysis plan (the plan above) is unchanged. This addendum fixes, before scoring, how
two external test sets are built, scored and reported. They were added after the primary results were known, in response
to an internal review that the human-labeled radiology evidence rests mostly on IU/Open-i. No model is trained on them,
and no training, checkpoint or temperature changes. Everything below is reported as it comes out.

### Systems
Decision models: RadKev-27B (v2), Kev-27B (@01b8199), RadKev-9B (v2x9), Kev-9B (@2629c06), each with its own fitted
temperature, scored with `radkev.evaluate` exactly as in the main evaluation. LLMs: Qwen3.8-27B and MedGemma-27B-text
(single-BOS fix), zero-shot, reasoning off, option-letter probabilities from `radkev.teacher predict` exactly as in the main
evaluation (MedGemma with `--think_off`). LLMs are scored only on questions with at most 16 options (letters A to P).

### External test 1: RadCases (Yao et al., 2025), ACR Appropriateness Criteria
- **Labels:** Hugging Face `michaelsyao/RadCases` at commit 93c99559, files `synthetic.jsonl` and `usmle.jsonl` (original
  labels against the ACR AC version accessed 2024-06-20, `radgpt/ac.json`). Labels were assigned by two senior medical
  students under an attending radiologist, with radiologist adjudication.
- **Case text:** rebuilt from the two open sources: the GPT-3.5 synthetic one-liners (`radGPT` @74700a8,
  `radgpt/data/synthetic.csv`) and the Medbullets USMLE Step 2/3 questions (`ChallengeClinicalQA` @dc1bc9f), whose first
  sentence is extracted with radGPT's own `split_into_sentences`. A case is used only if the SHA-512 of its one-liner matches
  a label row. JAMA, NEJM and BIDMC (MIMIC-IV) cases are not used (no open text).
- **Exclusions:** duplicate one-liners (kept once if their labels agree, dropped otherwise); for the panel question, cases
  labeled with more than one panel; for the topic question, cases labeled with more than one topic or with no topic;
  cases whose one-liner occurs verbatim in any record of the RadKev training, development or test splits (counted and
  reported).
- **Questions:** the state is the one-liner alone (plain text).
  - *Panel (main):* "Which ACR Appropriateness Criteria panel covers the imaging work-up of this patient?" Options:
    the 11 panels of `ac.json` plus "None: no ACR Appropriateness Criteria topic applies" (12 options), under neutral
    shuffled keys (seed 20261005). The dataset's "None" label maps to the last option.
  - *Topic (secondary, decision models only):* "Which ACR Appropriateness Criteria topic best matches this patient's
    presentation?" Options: all 225 topics of `ac.json`, shuffled under neutral keys. LLMs are not scored on it because
    single-letter scoring is limited to 26 options; this is stated as a property of the scoring method.
- **Outcomes:** accuracy (primary contrast RadKev-27B minus Kev-27B on the panel question); secondary: RadKev-9B minus
  Kev-9B, RadKev-27B minus Qwen3.8-27B and minus MedGemma-27B-text, expected calibration error (10 bins); every result also
  by subset (synthetic, Medbullets) and on the panel question excluding cases labeled "None".

### External test 2: RadGraph-XL (Delbrouck et al., 2024), radiologist-annotated reports (only if access is obtained)
- **Data:** the radiologist-annotated reports of the RadGraph-XL release that are reachable (Stanford AIMI / Redivis
  and, if credentialed access exists, PhysioNet), with the official split if one is provided (test split used; otherwise
  all annotated reports, since none were used in training). Reports whose normalized text matches a CheXpert Plus report
  (used for teacher-labeled training questions) are excluded and counted.
- **Questions:** one record per report; the state is the full annotated report text. From the observation entities
  (`Observation::definitely present`, `Observation::definitely absent`, `Observation::uncertain` or the release's
  equivalents), up to three per report are sampled with a fixed seed (20261005), at most one per status where available.
  Each becomes the four-option status question of the RadKev training format (`radkev.data.STATUS`: reported as present;
  explicitly reported as absent; possible, equivocal or cannot be excluded; not mentioned), with the entity's annotated
  text span as the finding and the radiologists' label as the key. "Not mentioned" is never a key: an entity that is not
  annotated is never treated as absent. Wordings are drawn from `radkev.data.STATUS_T` with the test-split rule.
- **Outcomes:** accuracy, overall and by modality / anatomy as given in the release (chest radiograph, chest CT,
  abdomen/pelvis CT, brain MRI); same contrasts as above.

### Statistics and reporting
- Paired comparisons on identical questions; 95% CIs from a record-level cluster bootstrap with 2,000 resamples,
  stratified by subset (RadCases) or modality (RadGraph-XL); no multiplicity adjustment; differences are first minus second.
- Both tests are reported in a short Results subsection and the supplement as external tests on tasks the specialized
  models were not trained on (RadCases: no ACR-panel or topic question exists in the training data; RadGraph-XL: the status
  question format exists in training only for teacher-labeled CXR states). They do not change the primary outcome.
- Results are reported whatever their direction, and the abstract and conclusions are revised if a conclusion is qualified.
- The synthetic RadCases one-liners were written by GPT-3.5; Medbullets questions are public and may be in pretraining data.

### Amendment (2026-10-05, before any RadGraph-XL record was built or scored)
The reachable release is the Stanford portion of RadGraph-XL on Stanford AIMI (Redivis): 2,000 reports, 500 each of chest
radiograph, chest CT, abdomen/pelvis CT and brain MRI, with no official split, so all are used. Two construction details are
fixed here: (1) the overlap rule is a word 8-gram overlap of 50% or more with any CheXpert Plus report (findings or impression)
or with any record state of the RadKev splits; (2) observation spans consisting only of a generic qualifier (normal,
unremarkable, stable, clear, intact, unchanged, negative and similar; list in `radkev.external.GENERIC`) are not used as the
finding of a question. Strata for the bootstrap are the four modalities.
