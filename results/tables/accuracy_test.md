| Decision family | n | Kev-27B (stock) | RadKev-27B (final) | RadKev-27B (no reports / teacher data) | Kev-9B (stock) | RadKev-9B (from Kev-9B) | RadKev-9B recipe from Qwen3.5-9B base | Qwen3.8-27B (zero-shot LLM) | MedGemma-27B-text (thought channel closed) | Laya typed-decisions (421M) |
|---|---|---|---|---|---|---|---|---|---|---|
| CXR report reading (human labels, IU) | 9218 | 93.7 | 95.8 | 93.7 | 95.3 | 95.6 | 94.3 | 95.6 | 93.2 | 74.2 |
| CXR report reading (CheXpert Plus / teacher) | 83 | 98.8 | 97.6 | 96.4 | 95.2 | 98.8 | 97.6 | 98.8 | 98.8 | 56.6 |
| CT report reading (CT-RATE) | 6950 | 95.3 | 98.7 | 94.9 | 93.2 | 98.4 | 98.9 | 95.9 | 92.7 | 55.3 |
| Case -> diagnosis (Eurorad) | 346 | 82.7 | 93.9 | 93.9 | 75.1 | 91.0 | 90.2 | 84.4 | 72.5 | 44.5 |
| Subspecialty routing | 185 | 84.3 | 83.2 | 85.9 | 77.8 | 82.7 | 81.6 | 87.0 | 80.0 | 39.5 |
| Radiology knowledge (MedMCQA) | 69 | 72.5 | 84.1 | 79.7 | 59.4 | 68.1 | 72.5 | 75.4 | 68.1 | 30.4 |
| Medical knowledge (MedQA, MedMCQA, MMLU, PubMedQA, MedXpertQA) | 8926 | 62.0 | 68.1 | 68.7 | 55.4 | 62.0 | 58.5 | 64.2 | 54.4 | 25.0 |
| Orders / protocols (teacher labels) | 520 | 74.0 | 94.6 | 71.2 | 72.5 | 91.7 | 91.9 | 98.5 | 87.9 | 43.7 |
| Triage / follow-up (teacher labels) | 422 | 91.5 | 97.2 | 88.9 | 87.9 | 96.9 | 95.5 | 98.8 | 95.3 | 61.1 |
| Overall (micro) | 26719 | 82.9 | 87.2 | 85.1 | 80.4 | 84.8 | 83.3 | 85.0 | 79.6 | 51.3 |
