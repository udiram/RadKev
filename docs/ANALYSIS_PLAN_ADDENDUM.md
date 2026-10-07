# Analysis plan addenda (final models)

The two addenda to the [analysis plan](ANALYSIS_PLAN.md) that specified the final models, the radiology benchmark and the outcomes before any test result of the final models was read, reproduced verbatim (apart from the name of the compute node). In the addenda, "v2" denotes earlier training runs and "v3" the final RadKev-27B and RadKev-9B. Analyses added after the results were known, including the RadCases panel prior correction, are identified as post hoc in the manuscript.

---

# Addendum to the analysis plan: external tests (written 2026-10-05, before any external-test result was read)

The main analysis plan (`paper/PREREGISTRATION.md`, commit 090f354) is unchanged. This addendum fixes, before scoring, how
two external test sets are built, scored and reported. They were added after the primary results were known, in response
to an internal review that the human-labeled radiology evidence rests mostly on IU/Open-i. No model is trained on them,
and no training, checkpoint or temperature changes. Everything below is reported as it comes out.

## Systems
Decision models: RadKev-27B (v2), Kev-27B (@01b8199), RadKev-9B (v2x9), Kev-9B (@2629c06), each with its own fitted
temperature, scored with `jobs/kev_eval.py` exactly as in the main evaluation. LLMs: Qwen3.8-27B and MedGemma-27B-text
(single-BOS fix), zero-shot, reasoning off, option-letter probabilities from `teacher.py predict` exactly as in the main
evaluation (MedGemma with `--think_off`). LLMs are scored only on questions with at most 16 options (letters A to P).

## External test 1: RadCases (Yao et al., 2025), ACR Appropriateness Criteria
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

## External test 2: RadGraph-XL (Delbrouck et al., 2024), radiologist-annotated reports (only if access is obtained)
- **Data:** the radiologist-annotated reports of the RadGraph-XL release that are reachable (Stanford AIMI / Redivis
  and, if credentialed access exists, PhysioNet), with the official split if one is provided (test split used; otherwise
  all annotated reports, since none were used in training). Reports whose normalized text matches a CheXpert Plus report
  (used for teacher-labeled training questions) are excluded and counted.
- **Questions:** one record per report; the state is the full annotated report text. From the observation entities
  (`Observation::definitely present`, `Observation::definitely absent`, `Observation::uncertain` or the release's
  equivalents), up to three per report are sampled with a fixed seed (20261005), at most one per status where available.
  Each becomes the four-option status question of the RadKev training format (`build_data.STATUS`: reported as present;
  explicitly reported as absent; possible, equivocal or cannot be excluded; not mentioned), with the entity's annotated
  text span as the finding and the radiologists' label as the key. "Not mentioned" is never a key: an entity that is not
  annotated is never treated as absent. Wordings are drawn from `build_data.STATUS_T` with the test-split rule.
- **Outcomes:** accuracy, overall and by modality / anatomy as given in the release (chest radiograph, chest CT,
  abdomen/pelvis CT, brain MRI); same contrasts as above.

## Statistics and reporting
- Paired comparisons on identical questions; 95% CIs from a record-level cluster bootstrap with 2,000 resamples,
  stratified by subset (RadCases) or modality (RadGraph-XL); no multiplicity adjustment; differences are first minus second.
- Both tests are reported in a short Results subsection and the supplement as external tests on tasks the specialized
  models were not trained on (RadCases: no ACR-panel or topic question exists in the training data; RadGraph-XL: the status
  question format exists in training only for teacher-labeled CXR states). They do not change the primary outcome.
- Results are reported whatever their direction, and the abstract and conclusions are revised if a conclusion is qualified.
- The synthetic RadCases one-liners were written by GPT-3.5; Medbullets questions are public and may be in pretraining data.

## Amendment (2026-10-05, before any RadGraph-XL record was built or scored)
The reachable release is the Stanford portion of RadGraph-XL on Stanford AIMI (Redivis): 2,000 reports, 500 each of chest
radiograph, chest CT, abdomen/pelvis CT and brain MRI, with no official split, so all are used. Two construction details are
fixed here: (1) the overlap rule is a word 8-gram overlap of 50% or more with any CheXpert Plus report (findings or impression)
or with any record state of the RadKev splits; (2) observation spans consisting only of a generic qualifier (normal,
unremarkable, stable, clear, intact, unchanged, negative and similar; list in `build_external.GENERIC`) are not used as the
finding of a question. Strata for the bootstrap are the four modalities.

---

# Addendum 2 to the analysis plan: RadKev-27B v3 and a radiology-only benchmark (written 2026-10-05, before training finished and before any v3 test result was read)

## Model
RadKev-27B v3 is a delta fine-tune of Kev-27B (@01b8199) with the recipe of RadKev-27B (v2): rank-16 LoRA, pointer head,
one epoch, peak learning rate 3e-5, one-cycle schedule, 1,000 replayed Kev records, 8 records per optimizer step, seed 0. The
training data are the v2 mix (`data/mix-v2mg`) plus the training splits of RadCases (`data/radcases-v3`, 264 cases with an ACR
Appropriateness Criteria panel and, where defined, topic question) and ReXErr-v1 report level (`data/rexerr`, 6,000 reports, about
half error-injected, asking whether the report contains an error). It is trained on four GPUs as two data-parallel ranks, each with
the backbone split across one NVLink pair and four accumulation steps (`jobs/kev_lora_dp.py`); gradients are summed over the ranks,
so the optimizer sees the same 8-record steps as the two-GPU recipe. The temperature is refitted on the combined development split.
Only the 27B model is retrained; RadKev-9B (v2) is reported unchanged.

## Benchmark
The benchmark is restricted to radiology. Its human-labeled part (the basis of all conclusions) consists of:
- IU/Open-i report reading (finding present, normal study, which finding); Eurorad diagnosis and subspecialty routing;
- MedMCQA radiology, and the radiology-relevant questions of MedMCQA (other subjects), MedQA, MedXpertQA, MMLU and PubMedQA,
  selected by `build_external.is_radiology` (imaging-modality and radiology terms in the stem, state or options; acronyms such as
  CT, MRI and PET matched in capitals only; word list `RAD_TERMS` fixed in this commit); all other knowledge questions are not
  benchmarked;
- the RadCases test split (138 cases; panel and topic questions) and the ReXErr test split (2,708 reports; labels known by
  construction);
- RSNA-RadioQA (80 RSNA Case Collection questions with the four-option answer sets of the RaR study), evaluated only, never trained
  on, when its question text has been obtained.
The teacher-labeled tasks remain in training but are not benchmarked: no agreement with the teacher labels is reported. CT-RATE (classifier labels) is reported separately as agreement with the classifier and is not part of the benchmark.

## Outcomes and statistics
- Primary: accuracy averaged over the human-labeled radiology benchmark tasks, RadKev-27B v3 minus Kev-27B.
- Secondary: the same as a mean over questions; RadKev-27B v3 minus RadKev-27B v2; RadKev-27B v3 minus Qwen3.8-27B and
  MedGemma-27B-text (zero-shot, reasoning off, option-letter scoring, questions with at most 16 options); every task separately;
  calibration and coverage at a 5% error budget as in the main evaluation.
- Paired record-level bootstrap with 2,000 resamples stratified by source, as in the main evaluation; differences are first
  minus second.
- RadCases and ReXErr are now training sources, so their test results are in-distribution; the former external-test sentence of
  the manuscript is replaced by these results. Results are reported whatever their direction.

## Amendment (2026-10-05, before any v3 test result was read)
At the user's request the teacher-labeled tasks are removed from benchmarking altogether (previously: reported separately as agreement).

## Amendment 2 (2026-10-05, before any v3 test result was read): subspecialty classification and pre-read routing
The Eurorad question with the 11 sections as options is posed on a state that contains the imaging findings, which do not exist
when a study is routed to a reading worklist (routing uses the order, the indication and the examination type). This task is
therefore reported as subspecialty classification of the case, with the Eurorad section as the key, and it stays in the benchmark
unchanged. A post hoc pre-read condition is added (`jobs/preread_route.py`; no training):
- Every `eurorad_route` test question is re-posed with the imaging findings removed from its state; age, sex and the clinical
  history (as in the original state) are kept. Question wording, options and keys are unchanged. Eurorad does not record the
  order or the examination as separate fields; the history, which sometimes mentions earlier imaging, is the closest available
  equivalent of the indication.
- Systems: RadKev-27B v3, RadKev-27B v2, Kev-27B, RadKev-9B, Kev-9B (decision models, scored as in the main evaluation) and
  Qwen3.8-27B and MedGemma-27B-text (letter scoring as in the main evaluation, MedGemma with the one-sentence pre-filled
  reasoning segment). Each system is scored on the same questions with the full state and with the pre-read state.
- Reported: accuracy per system and condition, the change from full to pre-read state, and RadKev-27B v3 minus Kev-27B and minus
  each LLM under each condition, with paired bootstrap 95% CIs (2,000 resamples). This condition is secondary and is not part of
  the primary outcome. Results are reported whatever their direction.

## Amendment 2 (2026-10-05 ~12:10, during training, before any v3 result was read)
- RadKev-9B is also retrained (user request): RadKev-9B v3, the v2 recipe of RadKev-9B (peak learning rate 2e-5, one GPU) on the
  same v3 training data. Both runs start from the Kev revisions used throughout (Kev-27B @01b8199, Kev-9B @2629c06; the Hub repos
  later received new weights) and were trained on the original two-GPU (27B) and one-GPU (9B) recipe, in parallel (run tag v3p);
  the four-GPU data-parallel variant was not used.
- ReXErr keys are known by construction (errors injected by GPT-4o into MIMIC-CXR reports), not assigned by people. The benchmark
  therefore has two key types: human-assigned (IU/Open-i, Eurorad, MedMCQA radiology and the radiology-filtered knowledge questions,
  RadCases, RSNA-RadioQA) and by construction (ReXErr). The primary outcome is computed over both, and over the human-assigned
  tasks alone as a secondary outcome.

## Amendment 3 (2026-10-05 ~13:20, before any v3 result was read)
At the user's request, to shorten training and to match the radiology-only benchmark, the v3 training data are restricted to
radiology: MedMCQA and MedQA records are kept only if their question passes `build_external.is_radiology` (or is a MedMCQA
radiology question); Eurorad, CT-RATE, the teacher-labeled records, RadCases and ReXErr are kept unchanged; the development split
is filtered the same way (`data/mix-v3r`, built by `jobs/build_v3r.py`: 36,109 training records instead of 73,273). Each
micro-batch holds 2 records with 4 accumulation steps, so an optimizer step still sees 8 records. Both RadKev-27B v3 and
RadKev-9B v3 are trained this way (run tag v3r), from the pinned Kev revisions; the earlier v3p runs on the full mix were stopped.

## Amendment 4 (2026-10-05 ~13:25, before any v3 result was read): pre-read routing after amendment 3
The first section headed "Amendment 2" above (subspecialty classification and pre-read routing) and the second (RadKev-9B v3,
ReXErr keys) were written independently on the same date; both stand. Following amendment 3, `jobs/preread_route.py` scores the
v3r checkpoints (RadKev-27B v3 and RadKev-9B v3), and RadKev-9B v3 is added to its systems, with RadKev-9B v3 minus Kev-9B and
minus RadKev-9B v2 reported under each condition. Nothing else in the pre-read condition changes.

## Amendment 4 (2026-10-05 ~15:30, before any RSNA-RadioQA result was read)
RSNA-RadioQA records: question text and reference answers from Appendix S1 of the RadioRAG article (Radiology: Artificial
Intelligence, doi 10.1148/ryai.240476, supplement ryai240476suppa1.pdf); four answer options per question from the public RadSaFE-200
release (RadioRAG subset, created and expert-reviewed in the RaR study). Question 44 is absent from the published appendix, so 79
questions are used. The key is the option that matches the appendix's reference answer; RadSaFE's answer index points to a different
option for questions 62 and 77 (reference answers Takayasu arteritis and right vocal cord palsy, both among the options), and the
reference answer is used. The state is the case description, the question its final sentence ("What is the most likely
diagnosis?"), options under neutral shuffled keys (seed 20261005). Evaluation only; the text is not redistributed
(`data/rsna-radioqa/test.jsonl` on the node, sha256 e58dffed…).

## Amendment 5 (2026-10-05 ~17:10, before any v3 result was read)
The 9B runs follow RadKev-27B v3 on the same four GPUs, one after another, each as four data-parallel ranks with one GPU per rank,
2 records per micro-batch and no accumulation (8 records per optimizer step), on `data/mix-v3r`: RadKev-9B v3 (warm start from Kev-9B
@2629c06, peak lr 2e-5) and, for the initialization ablation, the base-model start (Qwen3.5-9B-Base @68c46c4 with a new adapter and
head, peak lr 2e-4, Kev's default), each on all of the training data and on the same seeded 10% subset (as in the pilot, both arms
at 10%). If a four-GPU 9B run fails, the remaining runs use the one-GPU recipe (2 records per micro-batch, 4 accumulation steps).

## Amendment 6 (2026-10-05 ~17:45, before any v3 result was read): benchmark tasks
After the radiology filter several knowledge tasks keep almost nothing (MMLU subjects 0 to 71 questions), and a task mean would weight
a one-question task like IU's 5,696. The filtered knowledge questions are therefore pooled into one task per source. The benchmark
has 15 tasks (14,142 questions):
- human-assigned keys (14 tasks, 11,434 questions): iu_finding, iu_normal, iu_which, eurorad_dx, eurorad_route, medmcqa_rad
  (MedMCQA radiology subject), medmcqa_other_rad, medqa_rad, medxpertqa_rad, mmlu_rad, pubmedqa_rad, radcases_panel,
  radcases_topic, rsna_radioqa;
- keys by construction (1 task, 2,708 questions): rexerr_error.
Primary: unweighted mean of the 15 task accuracies, RadKev-27B v3 minus Kev-27B. Secondary: the mean over the 14 human-assigned
tasks; the question-pooled accuracy; every task (Holm across the 15). CT-RATE (classifier labels) is reported as agreement; the
teacher-labeled tasks are not reported. The LLMs are scored with option-letter probabilities (reasoning off) wherever a question has
at most 16 options, i.e. on every benchmark task except radcases_topic (225 options); comparisons with them use the questions both
systems answered. Every decision model is scored on all 15 tasks. Hub models not pinned in the pilot (Kev-4B, Kev-0.8B, Laya,
Laya-typed-decisions, GLiNER2.5-Decide, Julia-1, Qwen3.8-27B, MedGemma-27B-text) are loaded offline from the node's cache, i.e. the
snapshots scored in the pilot, and their snapshot hashes are recorded.

## Amendment 7 (2026-10-05 ~18:30, before any v3 result was read): answer-space study
`jobs/answer_space3.py`. Questions: every Eurorad diagnosis question of the held-out test split and every RSNA-RadioQA question
(radiology only; the pilot's MedQA sample is not used). Distractors from the Eurorad diagnosis pool (RSNA-RadioQA also from its own
options), near-duplicates of the key or of each other removed (containment, or PubMedBERT cosine >= 0.95). Conditions: original
options; the K-1 most similar distractors (sim_K) and two independent random draws (rand_K, randb_K), K = 2, 4, 8, 16, 32, 64, 128,
255; Qwen3.8-27B-written alternatives (K <= 16, as in the pilot). Systems: RadKev-27B v3, Kev-27B, RadKev-9B v3, Kev-9B (each option
scored directly), Qwen3.8-27B and MedGemma-27B-text. The LLMs are scored at every K with numbered options and greedy generation of the
answer number (reasoning off; an answer that is not a number in range counts as wrong, and the rate is reported), and additionally
with the pilot's option-letter probabilities where K <= 16 as a method check. Outcomes: accuracy by K and condition with 95% intervals
(question bootstrap), the random-draw difference as a robustness check, confidence of the decision models; latency per request at
every K on 60 Eurorad cases, one request at a time, CUDA synchronised, first 10 requests excluded, one system at a time.

## Amendment 8 (2026-10-05 ~19:30, before any v3 result was read): the remaining post hoc analyses with v3
Every analysis below compares systems on identical questions of the radiology benchmark (amendment 6's task mapping), uses the
record-level bootstrap (2,000 resamples, seed 20261005) and publishes aggregates only.
- **9B initialisation ablation, scored.** `jobs/eval_v3.py` runs a second time after the ablation trainings and adds base-init 9B
  (100%), Kev-init 9B (10%) and base-init 9B (10%) on all 15 tasks. Planned pairs: Kev-init minus base-init at 100% and at 10%, each
  arm minus Kev-9B, RadKev-9B v3 minus Kev-init 10%, base-init 100% minus base-init 10%. `jobs/transfer_paired.py v3`: the same arms
  and both v3 models against their released checkpoints on Kev's transfer suite (paired bootstrap, as in the pilot).
- **Option-only control and distinctive words** (`jobs/blind_v3.py`). Every Eurorad diagnosis question of the held-out split and
  every RSNA-RadioQA question, with the case and with the case replaced by one neutral line; RadKev-27B/9B v3, Kev-27B/9B, Qwen3.8-27B
  and MedGemma-27B-text, both conditions scored in the same job. Outcomes: accuracy with and without the case, the drop, the paired
  gains in both conditions and their difference; the same within the distinctive-word strata (a word of the correct option of at least
  six letters, absent from every distractor, appears in the case; the pilot's rule and stop list).
- **Latency** (`jobs/latency_bench.py --sample radbench`). 150 records (seed 0) drawn from the radiology benchmark records (held-out
  records with their benchmark questions only, plus the RadCases, ReXErr and RSNA-RadioQA test records); RadKev-27B/9B v3, Kev-27B/9B,
  Qwen3.8-27B and MedGemma-27B-text (one <bos>, empty thought pre-filled) on one NVLink pair; the pilot's modes (Kev one pass per
  record; LLM letter scoring, direct generation, Qwen reasoning on 40 questions); LLM questions with more than 16 options are skipped.
  Latency against answer-space size is amendment 7.
- **Reasoning sample** (`jobs/llm_reasoning.py v3`). The LLMs are unchanged, so the pilot's reasoning rows on the held-out file are
  reused; reasoning (same vLLM settings and answer read-out) is added on every RSNA-RadioQA question, at most 150 RadCases panel and 150
  RadCases topic questions (at most 16 options) and 200 ReXErr questions, by sha256 order. Analysis on benchmark questions only, each
  system on the identical set: task mean and pooled accuracy, reasoning on minus off, and RadKev v3 minus reasoning on.
- **RadGraph-XL, rescored with v3** (`jobs/external_tests.py --models v3_27,v3_9`, analysed on the node with
  `paper/external/analyse.py`): RadKev-27B/9B v3 added to the existing rows; pairs v3 minus Kev, v3 minus v2, v3 minus the LLMs.
- RadCases is now part of training and the benchmark, so the overnight external RadCases test is superseded and not repeated.
- Calibration (as scored and after cross-fitted temperature scaling) and held-out against seen wordings are computed inside
  `jobs/eval_v3.py` (amendment 6).
- **Pre-read routing** runs as specified in the first "Amendment 2" (`jobs/preread_route.py`), with RadKev-27B v3 and RadKev-9B v3
  from the final runs (v3f; the one-GPU 9B run if the four-GPU run failed).

## Amendment 9 (2026-10-06 ~14:50, before any answer-space or latency result was read): reduced sample sizes for runtime
At the user's request, both studies are run on smaller samples (post hoc, for runtime only; systems, keys and methods unchanged).
Answer space (amendment 7): 30 Eurorad diagnosis and 30 RSNA-RadioQA questions (the first in sha256 order of their ids), K = 2, 4,
16, 64, 255, one random draw (the second, randb_K, is dropped); latency on 15 Eurorad cases at K = 2, 16, 64, 255, the first 5
requests excluded. Latency study (amendment 8): 30 radiology-benchmark records (seed 0) and 8 reasoning questions.

## Amendment 10 (2026-10-06 ~19:00, before any request was sent): OpenAI Decisions API as an additional comparison system
At the user's request, the OpenAI Decisions API (`POST /v1/decisions`, model `gpt-6-luna`, public beta released 2026-10-06), an
externally hosted general-purpose decision model, is added as a comparison system (`openai_dec`; `jobs/openai_decisions.py`).
- **Questions.** Every benchmark question of the eval_v3 files (held-out shards and new.jsonl; CT-RATE and knowledge questions outside
  the radiology filter are not sent). One request per record carrying all of its benchmark questions; yes/no questions are sent as
  predicates, multiple-choice questions as choices (option keys as values, option texts as descriptions), ordinal questions as scores;
  the state is sent as text (structured states as JSON). Instructions and options are those every other system received. No prompt
  engineering, no system prompt, no repeated sampling: each record is sent once and the response is cached.
- **Scoring.** The returned probability of every admissible answer is the system's distribution, scored exactly as the other systems
  (`kev_eval --preds`, then eval_v3's analysis unchanged, with `openai_dec` added). A request rejected by the API is resent once per
  question; a question still rejected receives a uniform distribution and is reported as unsupported.
- **Pairs.** RadKev-27B v3 and RadKev-9B v3 minus openai_dec; openai_dec minus Kev-27B, Qwen3.8-27B and MedGemma-27B-text (task mean,
  pooled, per task with Holm); calibration as scored and recalibrated, and the cov5/ECE pair differences for both RadKev models.
- **Latency.** End-to-end wall time per request (client to response) from the compute node, as a deployer would experience it, plus the
  server processing time the API reports. The probe records are sent sequentially; the remainder with four concurrent requests.
- **Cost cap.** $5 in total (the job stops at $4.50 of reported input tokens).
- The result is reported as obtained, whatever its direction.
- *Note (2026-10-06 ~19:15, after all requests were sent and before any accuracy was computed):* the API answered 5 of 14,142
  questions with `"type": "refusal"` (RadCases records). These are handled as unsupported questions (uniform distribution) and
  reported; no request was resent.

## Amendment 11 (2026-10-07 ~12:10, while requests were being sent and before any accuracy was computed): Jev as an additional comparison system
At the user's request, Jev (TypeSafe System One API, `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`, which resolved
to `jev-1.13.0` on every probe request), the externally hosted general-purpose decision model whose interface Kev reproduces, is added
as a comparison system (`jev`; `jobs/jev_decisions.py`), with the protocol of amendment 10 unchanged except where the API differs.
- **Questions.** The same records and questions as amendment 10 (7,498 records, 14,142 questions; CT-RATE and knowledge questions
  outside the radiology filter are not sent). Jev accepts the benchmark's own format, so each request carries the record's state and
  each question's `type`, `instructions` and `criteria` unchanged (no label or metadata field), keyed by an anonymous name. No prompt
  engineering, no system prompt, no repeated sampling: each record is sent once and the response is cached.
- **Scoring.** Jev returns the probability of `true` (noul) or a probability per option (choice, score); these are the system's
  distribution, scored exactly as the other systems (`kev_eval --preds`, then `jobs/eval_v3_prior.py` with `jev` added; the RadCases
  prior correction applies to RadKev only, as before). A request rejected as invalid (HTTP 400/422) is resent once per question; a
  question still rejected, or answered without a distribution, receives a uniform distribution and is reported as unsupported.
  Jev's probabilities are rounded to two decimals (as the OpenAI API's), so calibration is reported as scored only.
- **Pairs.** RadKev-27B and RadKev-9B minus jev; jev minus Kev-27B, Kev-9B and Qwen3.8-27B; OpenAI Decisions minus jev (task mean,
  pooled, per task with Holm); calibration as scored, and the cov5/ECE pair differences for both RadKev models and for Kev-27B.
- **Latency.** End-to-end wall time per request from the compute node, probe records sequential, the remainder with four concurrent
  requests. Jev reports no server processing time.
- **Cost cap.** $5 in total (the job stops at $4.50 of reported input tokens; $0.042 per million input tokens, output not billed).
- The result is reported as obtained, whatever its direction.
