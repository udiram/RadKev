# Reproducing the study

Every number in the manuscript and in this repository came from the commands below, run on RTX A6000 (48 GB) GPUs. Run names (`v2mg`, `v2x9`, ...) are the
ones the results use, so a rerun lands next to the same labels. [`scripts/reproduce.sh`](../scripts/reproduce.sh) runs the main line
end to end.

## 0. Environment

```bash
scripts/setup.sh --laya            # Kev @ f2bb629 + its venv, the two-GPU patch, CUDA kernels, radkev; Laya for the baselines
source ~/.cache/radkev/kev/.venv/bin/activate
hf auth login                      # MedGemma-27B-text and CT-RATE are gated: accept their terms on the Hub first
```

Everything is written under `$RADKEV_HOME` (default `~/.cache/radkev`): `raw/`, `data/`, `runs/`. Pinned versions: Kev
`f2bb629d670f5b746f712fc05550a098526c836b`; Qwen3.8-27B `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; Qwen3.5-9B-Base
`68c46c4b3498877f3ef123c856ecfde50c39f404`; Kev-27B as initialized: Hub snapshot `01b81998019be550f0ae858727df49bac9511195`.

**Pinned Kev weights.** The Hugging Face repositories of Kev received new weights after this study. Every result used Kev-27B
revision `01b81998019be550f0ae858727df49bac9511195` and Kev-9B revision `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c`
(`radkev.paths.KEV_27B_PINNED`, `KEV_9B_PINNED`; the post hoc scripts load these). For training and for the held-out test, download
these snapshots and pass their local paths wherever the commands below name `jaredpalmer/kev-27b` or `jaredpalmer/kev-9b`:

```bash
hf download jaredpalmer/kev-27b --revision 01b81998019be550f0ae858727df49bac9511195 --local-dir ~/.cache/radkev/kev-27b-01b8199
hf download jaredpalmer/kev-9b  --revision 2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c --local-dir ~/.cache/radkev/kev-9b-2629c06
```

## 1. Data

```bash
python -m radkev.fetch                                  # open sources -> $RADKEV_HOME/raw (no login)
# gated: put CT-RATE's report and label CSVs under raw/ctrate/, ReXGradient-160K metadata under raw/rexgradient/,
# CheXpert Plus tables under raw/chexpert_plus/ (python -m radkev.fetch --list shows the layout)

python -m radkev.data --raw ~/.cache/radkev/raw --out ~/.cache/radkev/data/rad-open \
    --only iu,eurorad,medmcqa,medqa,mmlu_med,pubmedqa,medxpertqa
python -m radkev.data --raw ~/.cache/radkev/raw --out ~/.cache/radkev/data/rad-gated --only ctrate --cap ctrate=8000
```

Compare `data/*/manifest.json` with [`results/manifests/`](../results/manifests/). Hugging Face parquet conversions can change upstream;
the counts tell you whether yours match. (Our gated build was run with `--only ctrate,chexpert_plus,mimic --cap
mimic=12000,ctrate=8000,chexpert_plus=12000`, and only CT-RATE was present; the command above gives the identical records.)

## 2. Teacher labels (imaging orders, triage and follow-up, finding status)

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/teacher_labels.py --per-source 1500     # -> data/teacher, ~1.4 h
```

## 3. Training

| Run | Command | Result |
|---|---|---|
| RadKev-27B (primary) | `CUDA_VISIBLE_DEVICES=0,1 python experiments/train.py --models kev-27b --data rad-open,rad-gated,teacher --tag v2mg` | `runs/v2mg-kev-27b`, 25.7 h |
| RadKev-27B without report/teacher data (ablation) | `... --models kev-27b --data rad-open --tag v1med` | `runs/v1med-kev-27b`, 11.5 h |
| Pilot (ablation) | `... --models kev-27b,kev-9b --data rad-open --tag v0open --replay 4000 --max_steps kev-27b=400` | 1.0 h / 1.6 h |
| RadKev-9B + plain-base arm | `CUDA_VISIBLE_DEVICES=0,1 python experiments/train.py --models kev-9b,base-9b --data rad-open,rad-gated,teacher --tag v2x9` | 10.9 h each, in parallel |
| Same at 10% data | `... --models kev-9b,base-9b --data rad-open,rad-gated,teacher --tag v2x9f10 --train_fraction 0.1` | 1.2 h each |

Each run trains with `kev.train`, scores dev with the inherited temperature, fits a new temperature on dev
(`scripts/calibrate_checkpoint.py` from Kev), rescores dev, scores the released checkpoint on the same dev items, and runs Kev's
transfer suite for both. The exact `kev.train` arguments each run used are in [`results/training/`](../results/training/).

**Selection rule (prespecified).** `v2mg` stays primary unless its calibrated dev accuracy, macro over the tasks both runs share,
is more than 1.0 percentage point below `v1med`'s. It was 0.03 points below (86.71% vs 86.74%), so `v2mg` is RadKev-27B.

## 4. Held-out test

```bash
export T=rad-open,rad-gated,teacher
# every Kev-family model, Qwen3.8-27B and MedGemma-27B-text (original prompt), Laya
python experiments/final_test.py --tag final --data $T --no-compare \
    --runs stock27=jaredpalmer/kev-27b,v0open27=v0open-kev-27b,stock9=jaredpalmer/kev-9b,ft9=v0open-kev-9b,v1med27=v1med-kev-27b,v2_27=v2mg-kev-27b,r9=v2x9-kev-9b,b9=v2x9-base-9b,r9f10=v2x9f10-kev-9b,b9f10=v2x9f10-base-9b,kev08=jaredpalmer/kev-0.8b,kev4=jaredpalmer/kev-4b \
    --llms qwen38=Qwen/Qwen3.8-27B,medgemma=google/medgemma-27b-text-it --laya 1
python experiments/decision_baselines.py          # Laya-typed, Julia-1, GLiNER2.5-Decide (post hoc)
python experiments/llm_scoring.py                 # qwen38_gen, medgemma_gen, medgemma_brief (revised MedGemma prompt; post hoc)
# one comparison over everything already scored (no GPU needed)
python experiments/final_test.py --tag final --data $T --laya 1 --also v2_27,r9,r9f10 \
    --runs stock27=cached,v0open27=cached,stock9=cached,ft9=cached,v1med27=cached,v2_27=cached,r9=cached,b9=cached,r9f10=cached,b9f10=cached,kev08=cached,kev4=cached,laya_typed=cached,julia1=cached,gliner_decide=cached \
    --llms qwen38=cached,medgemma=cached,qwen38_gen=cached,medgemma_gen=cached,medgemma_brief=cached
```

`runs/test-final/comparisons.json` is the file [`results/comparisons.json`](../results/comparisons.json) was copied from.

## 5. Studies

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/latency.py                                         # 27B tier
CUDA_VISIBLE_DEVICES=0,1 python experiments/latency.py --kev stock9=jaredpalmer/kev-9b,radkev9=v2x9-kev-9b,base9=v2x9-base-9b --llms "" --out latency_9b.json
CUDA_VISIBLE_DEVICES=0,1 python experiments/latency_scaling.py
python experiments/leak_sensitivity.py
python experiments/transfer_paired.py
```

## 6. Post hoc analyses

These reproduce the analyses added after the primary results (manuscript, Section 2.6). Each writes aggregates or per-question
probabilities without text under `$RADKEV_HOME/runs/`.

| Analysis (manuscript) | Script | Hardware we used |
|---|---|---|
| Shared 2,000-resample bootstrap: intervals of every system, paired differences with Holm adjustment, withheld wordings, selective prediction | `experiments/robustness.py` | CPU |
| Primary comparison without the demonstration sample, calibration differences, specialization vs scale, recalibration, family average, prevalence baseline, distinctive words | `experiments/robustness2.py` | CPU |
| MedGemma-27B-text with the final prompt (the row reported in the manuscript) | `experiments/medgemma_rescore.py` | 2× A6000 |
| LLMs with reasoning on the 1,800-question sample | `experiments/llm_reasoning.py` (vLLM) | 2× A6000 |
| Options-only control | `experiments/options_only.py` | 2× or 4× A6000 |
| External test sets (RadCases, RadGraph-XL) | `experiments/external_tests.py` | 2× or 4× A6000 |
| Answer space: decision models and latency against the number of options | `experiments/answer_space.py` | 2× or 4× A6000 |
| Answer space: LLMs (at most 16 options) | `experiments/answer_space_llm.py` | 2× or 4× A6000 |
| Regenerated teacher labels and agreement of the decision models | `experiments/teacher_rescore.py` | 2× or 4× A6000 |
| Training-loss curves and the teacher examples of Figures 1 and 3 | `experiments/figure_data.py` | CPU |
| Latency (Table S7) and latency against questions per record | `experiments/latency.py`, `experiments/latency_scaling.py` | 2× A6000 (NVLink) |
| Leakage of order questions; Kev's transfer suite | `experiments/leak_sensitivity.py`, `experiments/transfer_paired.py` | CPU |

## 7. Manuscript

[`paper/build.py`](../paper/build.py) generates every number, table and figure of the manuscript from the aggregate outputs in
`paper/inputs/` and compiles it (Tectonic). Every number is written to [`results/numbers.csv`](../results/numbers.csv) with the file
and field it came from, and the manuscript reads it through `\V{key}` macros:

```bash
cd paper && python build.py            # numbers, tables, figures, main.pdf and an editable export
cd paper && python build.py --no-pdf   # without LaTeX
```

## Without a GPU

The data builders, the teacher merge, `radkev.evaluate --preds` and `radkev.compare` run on a laptop, and `pytest` exercises them
(including Kev's metrics and bootstrap) on synthetic fixtures. Kev-0.8B and Kev-4B also run on Apple Silicon through Kev's MLX
backend, which is enough to try `radkev.predict` and the examples.
