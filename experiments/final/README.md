# Final runs

The job scripts behind the manuscript's results, as they were run on the study's compute node (four RTX A6000 GPUs). Each script is
self-contained: it reads `$XDG_CACHE_HOME/radkev` (the Kev checkout, its virtual environment, the built data and the trained
runs), writes aggregate outputs to `$ZCB_OUTPUT_DIR`, and publishes aggregates only. Their outputs, scrubbed of local paths, are
the inputs of the manuscript build in [`paper/inputs/artifacts/`](../../paper/inputs/artifacts/).

| Script | Produces | Manuscript |
|---|---|---|
| `build_v3r.py`, `build_v3.py` | the radiology-only training and development records; the RadCases and ReXErr splits and the radiology filter | Methods 2.1, Table 1 |
| `train.py` | RadKev-27B and RadKev-9B (runs `v3f-kev-27b-dp`, `v3f-kev-9b-dp`): training, temperature, development and transfer evaluations | Methods 2.3, Figure 3 |
| `eval_v3.py` | every system on the 14,142 benchmark questions; per-task accuracy, task means, paired differences, calibration, wording | Results 3.1 to 3.4 |
| `radcases_prior.py`, `eval_v3_prior.py` | the RadCases panel prior correction and the rerun of the benchmark and reasoning-sample analyses with it (the reported results) | Methods 2.4, Results |
| `reasoning_v3_fix.py` | reruns the reasoning-sample analysis with records resampled within their source and the examination tasks pooled into one task (the reported reasoning results) | Results 3.2, Supplementary Note S4 |
| `radcases_panel_diag.py` | selection of the catch-all option on the RadCases panel question by key type | Results 3.1 |
| `llm_reasoning.py` (`v3`) | Qwen3.8-27B and MedGemma-27B-text with reasoning on a sample of 1,675 questions | Results 3.2 |
| `openai_decisions.py` | the OpenAI Decisions API on the benchmark, its latency and cost | Methods 2.6, Results 3.2 |
| `jev_decisions.py` | Jev on the benchmark, its latency and cost (scored within `eval_v3_prior.py`) | Methods 2.6, Results 3.2 |
| `latency_bench.py` | median latency per question, one request at a time | Results 3.2, Figure 5c |
| `transfer_paired.py` | Kev's out-of-domain transfer suite, specialized vs released models | Results 3.3 |
| `preread_route.py`, `blind_v3.py` | subspecialty classification before the read; the options-only control | Results 3.5 |
| `answer_space3.py` | accuracy and latency as the answer space grows | Results 3.6 |
| `answer_space_hosted.py` | the answer-space study for the hosted decision models (OpenAI Decisions, Jev) on the same questions and latency requests | Results 3.6, Discussion 4.1 |
| `radgraph_xl_hosted.py`, `radgraph_xl_definite.py` | the external test for the hosted decision models and the analysis on all questions, definite findings and hedged findings | Results 3.7, Supplementary Note S6 |
| `radgraph_xl_build.py`, `external_tests.py`, `external_analyse_node.py`, `radgraph_xl_errors_v3.py` | the RadGraph-XL external test | Results 3.7 |

`lib/` holds the helper modules the scripts bundle (`kev_eval.py`, `compare.py`, `kev_lora_dp.py`, `kev_locked.py`,
`build_external.py`, `external_analyse.py`). The scripts also import `build_data` and `teacher`, which are the modules published
as [`radkev/data.py`](../../radkev/data.py) and [`radkev/teacher.py`](../../radkev/teacher.py) (the manuscript build asserts that
their constants are identical).
