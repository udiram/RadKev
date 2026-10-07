# Experiments

| Path | Contents |
|---|---|
| [`final/`](final/) | the job scripts of the reported runs: data additions, training of RadKev-27B and RadKev-9B, the benchmark evaluation and every additional analysis in the manuscript, reproduced as they were run ([README](final/README.md)) |
| `teacher_labels.py`, `teacher_rescore.py`, `medgemma_rescore.py` | two-teacher labeling of the order, triage and follow-up questions, and the regenerated labels reported in Supplementary Note S2 |
| `figure_data.py` | the inputs of Figures 1 and 3 |
| other scripts | supporting runners shared by the final jobs (scoring of LLMs and decision baselines, latency, robustness and external-test helpers) |

[docs/REPRODUCE.md](../docs/REPRODUCE.md) describes the order in which they are run.
