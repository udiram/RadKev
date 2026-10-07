# Results

Aggregate results of the study. Nothing in this directory contains dataset text. Local paths are written as `$RADKEV_HOME/...`.

## The manuscript

| Path | Contents |
|---|---|
| `numbers.csv`, `numbers_v3.csv` | every number in the manuscript: macro name, value as printed, source file, field and note. Written by [`paper/build.py`](../paper/build.py) (Introduction, Methods, Table 1, Figures 1 to 3) and [`paper/v3/build_v3.py`](../paper/v3/build_v3.py) (abstract, Results, Discussion, Conclusions, Figures 4 to 9, Tables S2 to S12); the manuscript reads these values through `\V{key}` macros, so no number is typed by hand |
| `manuscript/` | every table of the manuscript as CSV, named by its number: `Table1_data.csv`, `TableS1_teacher_labels.csv` and `TableS2_primary_by_task.csv` to `TableS12_external.csv`; values exactly as typeset, with 95% confidence intervals in parentheses |
| [`../paper/inputs/artifacts/`](../paper/inputs/artifacts/) | the aggregate job outputs from which both ledgers are computed, e.g. `eval_v3_prior/` (benchmark evaluation of every system and the reasoning-sample analysis, with the RadCases panel prior correction for RadKev), `radcases_prior/`, `latency_v3/`, `openai_dec/`, `blind_v3/`, `preread_v3/`, `answer_space3/`, `external_v3_analysis/`, `transfer_v3/` and `train_v3/` |

The manuscript's intervals come from one shared cluster bootstrap with 2,000 resamples, stratified by source.
