| Model | Kind | Params | Radiology training | Overall % | Human keys % | Radiology human keys % | Medical knowledge % | ECE | Confident errors % | Coverage at 5% error % | ms/record |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RadKev-27B (final) | decision, radiology fine-tune | 27B | yes | 87.2 | 82.4 | 95.4 | 68.1 | 0.017 | 1.5 | 83.6 | 193 |
| RadKev-27B (no reports / teacher data) | decision, medical-QA-only fine-tune | 27B | yes | 85.1 | 81.7 | 93.5 | 68.7 | 0.018 | 1.1 | 75.6 | 192 |
| Qwen3.8-27B (zero-shot LLM) | LLM, generalist | 27B | no | 85.0 | 80.3 | 94.9 | 64.2 | 0.021 | 2.1 | 75.9 | — |
| Qwen3.8-27B (letter read where generated) | LLM, generalist | 27B | no | 85.0 | 80.3 | 95.0 | 64.1 | 0.021 | 2.1 | 76.0 | — |
| RadKev-9B (from Kev-9B) | decision, radiology fine-tune | 9B | yes | 84.8 | 79.3 | 95.0 | 62.0 | 0.018 | 1.3 | 79.6 | 65 |
| RadKev-27B (pilot) | decision, pilot fine-tune | 27B | yes | 84.2 | 80.2 | 94.2 | 64.9 | 0.030 | 2.0 | 76.3 | 192 |
| RadKev-9B, 10% data (from Kev-9B) | decision, radiology fine-tune (10% data) | 9B | yes | 83.6 | 78.1 | 94.8 | 59.6 | 0.016 | 1.3 | 77.0 | 65 |
| RadKev-9B recipe, 10% data, from base | decision head + LoRA on a plain LM (10% data) | 9B | yes | 83.5 | 77.6 | 94.7 | 58.8 | 0.013 | 1.0 | 77.7 | 65 |
| RadKev-9B recipe from Qwen3.5-9B base | decision head + LoRA on a plain LM, radiology data | 9B | yes | 83.3 | 77.0 | 93.7 | 58.5 | 0.016 | 0.8 | 77.6 | 65 |
| Kev-27B (stock) | decision, generalist | 27B | no | 82.9 | 78.2 | 93.0 | 62.0 | 0.026 | 0.7 | 76.1 | 192 |
| RadKev-9B (pilot) | decision, pilot fine-tune | 9B | yes | 81.6 | 77.4 | 94.3 | 58.8 | 0.038 | 2.0 | 71.8 | 65 |
| Kev-9B (stock) | decision, generalist | 9B | no | 80.4 | 75.6 | 94.0 | 55.4 | 0.016 | 1.1 | 70.5 | 65 |
| MedGemma-27B-text (thought channel closed) | LLM, medical | 27B | no (medical) | 79.6 | 74.1 | 92.1 | 54.4 | 0.135 | 9.5 | 66.6 | — |
| Kev-4B (stock) | decision, generalist | 4B | no | 78.2 | 73.0 | 92.6 | 51.4 | 0.024 | 0.5 | 67.1 | 46 |
| MedGemma-27B-text (pre-registered scoring, reads its thought channel) | LLM, medical | 27B | no (medical) | 78.1 | 71.7 | 92.7 | 48.5 | 0.109 | 5.3 | 61.6 | — |
| MedGemma-27B (letter read where generated, 16 tokens) | LLM, medical | 27B | no (medical) | 78.0 | 71.6 | 92.7 | 48.3 | 0.110 | 5.5 | 60.9 | — |
| Kev-0.8B (stock) | decision, generalist | 0.8B | no | 68.2 | 60.2 | 85.7 | 32.1 | 0.066 | 0.3 | 45.9 | 31 |
| GLiNER2.5-Decide (340M) | decision, generalist encoder | 340M | no | 53.6 | 49.9 | 73.0 | 24.4 | 0.026 | 0.2 | 0.1 | — |
| Laya typed-decisions (421M) | decision, generalist encoder | 421M | no | 51.3 | 49.7 | 72.2 | 25.0 | 0.036 | 0.0 | 0.0 | — |
| Laya (stock) | decision, generalist encoder | 421M | no | 46.8 | 45.9 | 64.8 | 25.0 | 0.184 | 2.7 | 0.0 | — |
| Julia-1 (144M) | decision, generalist encoder | 144M | no | 45.9 | 44.3 | 62.5 | 24.2 | 0.379 | 25.6 | 0.0 | — |
