| Model | ECE | Brier | Confident errors (p>=0.9 & wrong, %) | Coverage at 5% error (%) |
|---|---|---|---|---|
| Kev-27B (stock) | 0.026 | 0.225 | 0.7 | 76.1 |
| RadKev-27B (final) | 0.017 | 0.181 | 1.5 | 83.6 |
| RadKev-27B (no reports / teacher data) | 0.018 | 0.212 | 1.1 | 75.6 |
| Kev-9B (stock) | 0.016 | 0.260 | 1.1 | 70.5 |
| RadKev-9B (from Kev-9B) | 0.018 | 0.207 | 1.3 | 79.6 |
| RadKev-9B recipe from Qwen3.5-9B base | 0.016 | 0.220 | 0.8 | 77.6 |
| Qwen3.8-27B (zero-shot LLM) | 0.021 | 0.216 | 2.1 | 75.9 |
| MedGemma-27B-text (thought channel closed) | 0.135 | 0.326 | 9.5 | 66.6 |
| Laya typed-decisions (421M) | 0.036 | 0.567 | 0.0 | 0.0 |
