# Results

Every aggregate number behind the README, the model card and the figures. Nothing here contains dataset text or per-item
predictions. Produced by the runs in [docs/REPRODUCE.md](../docs/REPRODUCE.md); local paths are written as `$RADKEV_HOME/...`.

| Path | What it is |
|---|---|
| `comparisons.json` | the final comparison over all 21 systems on the held-out test (`radkev.compare` output): per model overall, per family, per task, per answer-key subset and per wording, reliability bins, and paired bootstrap deltas vs Kev-27B (`vs_reference`) and vs RadKev-27B / RadKev-9B (`vs`) |
| `tables/accuracy_test.*` | accuracy by decision family |
| `tables/delta_test.*` | paired Δ vs Kev-27B by family, 95% CI |
| `tables/per_task_test.*` | every task: n, accuracy, Brier, ECE, AUROC |
| `tables/landscape_test.*` | all 21 systems: overall, human keys, radiology human keys, medical knowledge, ECE, confident errors, coverage, latency |
| `tables/calibration_test.*`, `coverage5_test.*` | calibration and selective-prediction metrics |
| `tables/init_ablation_test.*`, `transfer_retention.*` | the 9B initialisation ablation and Kev transfer-suite retention |
| `tables/wording_test.*` | seen vs held-out instruction wordings |
| `tables/latency_*` | latency per question and per record; scaling with questions per report |
| `tables/llm_rescoring_test.*`, `medgemma_scoring_medqa300.*` | how the LLM baselines were read (see docs/EVALUATION.md) |
| `tables/orders_leak_sensitivity.*` | order questions whose clinical question names an imaging test |
| `tables/teacher_agreement.*`, `data_counts.*` | two-teacher agreement rates; records per source and split |
| `tables/training_runs.*`, `dev_*` | training runs and their development-set results |
| `manifests/` | record and question counts per source, split and task for each built suite; check your build against these |
| `training/` | the exact `kev.train` configuration and training metrics of every run |
| `studies/` | raw outputs of the latency, scaling, leak, transfer, baseline and LLM-scoring studies |

Model names in the JSON files: `v2_27` = RadKev-27B, `r9` = RadKev-9B, `b9` = the 9B plain-base arm, `*f10` = 10% data,
`v1med27` = RadKev-27B without report/teacher data, `v0open27` / `ft9` = pilots, `stock27` / `stock9` / `kev4` / `kev08` = released
Kev, `qwen38` = Qwen3.8-27B, `medgemma` = MedGemma pre-registered scoring, `medgemma_brief` = MedGemma with the thought channel
closed, `*_gen` = letter read where generated.
