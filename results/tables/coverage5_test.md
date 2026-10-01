| Model | CXR report reading (human labels, IU) | CXR report reading (CheXpert Plus / teacher) | CT report reading (CT-RATE) | Case -> diagnosis (Eurorad) | Subspecialty routing | Radiology knowledge (MedMCQA) | Medical knowledge (MedQA, MedMCQA, MMLU, PubMedQA, MedXpertQA) | Orders / protocols (teacher labels) | Triage / follow-up (teacher labels) |
|---|---|---|---|---|---|---|---|---|---|
| Kev-27B (stock) | 96.9 | 100.0 | 100.0 | 65.9 | 60.0 | 46.4 | 32.3 | 39.0 | 87.7 |
| RadKev-27B (final) | 100.0 | 100.0 | 100.0 | 96.5 | 76.8 | 65.2 | 33.4 | 95.0 | 100.0 |
| RadKev-27B (no reports / teacher data) | 96.0 | 100.0 | 99.6 | 96.2 | 77.8 | 68.1 | 33.4 | 31.2 | 80.8 |
| Kev-9B (stock) | 100.0 | 100.0 | 93.7 | 34.7 | 50.3 | 13.0 | 18.7 | 46.5 | 77.7 |
| RadKev-9B (from Kev-9B) | 100.0 | 100.0 | 100.0 | 89.6 | 72.4 | 31.9 | 26.3 | 88.5 | 100.0 |
| RadKev-9B recipe from Qwen3.5-9B base | 98.4 | 100.0 | 100.0 | 83.5 | 73.5 | 40.6 | 20.9 | 91.5 | 100.0 |
| Qwen3.8-27B (zero-shot LLM) | 100.0 | 100.0 | 100.0 | 57.8 | 68.1 | 43.5 | 26.5 | 100.0 | 100.0 |
| MedGemma-27B-text (thought channel closed) | 95.2 | 100.0 | 94.6 | 31.5 | 36.8 | 34.8 | 12.4 | 86.9 | 100.0 |
| Laya typed-decisions (421M) | 0.0 | 0.0 | 1.2 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 5.5 |
