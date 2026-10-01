| Data | Model | Test overall % | Test human keys % | Test medical knowledge % | Test ECE | General transfer acc % | General transfer ECE |
|---|---|---|---|---|---|---|---|
| 100% | RadKev-9B (from Kev-9B) | 84.8 | 79.3 | 62.0 | 0.018 | 83.7 | 0.040 |
| 100% | RadKev-9B recipe from Qwen3.5-9B base | 83.3 | 77.0 | 58.5 | 0.016 | 75.5 | 0.043 |
| 100% | Kev-init minus base-init [95% CI] | +1.5 [+1.2, +1.9] | +2.3 [+1.9, +2.8] | +3.5 [+2.6, +4.3] |  | +8.2 [+5.0, +11.1] |  |
| 10% | RadKev-9B, 10% data (from Kev-9B) | 83.6 | 78.1 | 59.6 | 0.016 | 82.0 | 0.061 |
| 10% | RadKev-9B recipe, 10% data, from base | 83.5 | 77.6 | 58.8 | 0.013 | 77.3 | 0.025 |
| 10% | Kev-init minus base-init [95% CI] | +0.1 [-0.2, +0.4] | +0.5 [+0.1, +0.8] | +0.9 [+0.1, +1.6] |  | +4.7 [+1.4, +7.8] |  |
| — | Kev-9B (stock) (no radiology training) | 80.4 | 75.6 | 55.4 | 0.016 | 82.0 | 0.041 |
