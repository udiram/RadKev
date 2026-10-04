# Results

Aggregate results of the study. Nothing in this directory contains dataset text. Local paths are written as `$RADKEV_HOME/...`.

| Path | Contents |
|---|---|
| `numbers.csv` | every number in the manuscript (802 entries): key, value as printed, source file, field and note. Written by [`paper/build.py`](../paper/build.py); the manuscript reads these values through `\V{key}` macros, so no number is typed by hand |
| `manuscript/` | every table of the manuscript as CSV, named by its number: `Table1_data.csv` and `TableS1_teacher_labels.csv` to `TableS13_ablations.csv`; values exactly as typeset, with 95% confidence intervals in parentheses |
| `comparisons.json` | the scored comparison of every evaluated run on the held-out test split (`radkev.compare` output): per run overall, per decision family, per task, per label type and per wording, reliability bins, and paired bootstrap differences against Kev-27B (`vs_reference`) and against RadKev (`vs`) |
| `tables/data_counts.*` | records per source and split |
| `tables/training_runs.*`, `tables/dev_*` | training runs and their development-split results |
| `tables/orders_leak_sensitivity.*` | agreement on order questions whose clinical indication names an imaging examination, and on the remaining questions |
| `manifests/` | record and question counts per source, split and task for each built suite, to check a rebuild against |
| `training/` | the `kev.train` configuration and training metrics of every run |
| `studies/` | raw outputs of the latency, latency-scaling, leak, transfer, baseline and LLM-scoring studies |

The manuscript's intervals come from one shared bootstrap with 2,000 resamples (see [docs/EVALUATION.md](../docs/EVALUATION.md));
`comparisons.json`, `tables/` and `studies/` are the outputs of the individual scripts and use their own resampling (1,000
resamples in `radkev.compare`). Where the two differ in the last digit, the manuscript values in `numbers.csv` and `manuscript/`
are the reported ones.

Run names in the JSON files: `v2_27` = RadKev-27B; `r9` = RadKev-9B; `b9` = the 9B model fine-tuned from Qwen3.5-9B-Base;
`*f10` = trained on 10% of the training data; `v1med27` = RadKev-27B trained without the CT-RATE and teacher-labeled data;
`v0open27` and `ft9` = pilot models; `stock27`, `stock9`, `kev4`, `kev08` = the released Kev models; `qwen38` = Qwen3.8-27B;
`medgemma` = MedGemma-27B-text with the original prompt; `medgemma_brief` = MedGemma-27B-text with the revised prompt, before the
duplicated beginning-of-sequence token was removed (the manuscript's MedGemma-27B-text results use the final prompt);
`*_gen` = the answer letter read from generated text. Family names: `orders_protocols` = imaging orders; `human_keys` =
human-labeled questions.
