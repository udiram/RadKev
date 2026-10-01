# Reproducing RadKev

Every number in this repository came from the commands below, run on RTX A6000 (48 GB) GPUs. Run names (`v2mg`, `v2x9`, ...) are the
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
`68c46c4b3498877f3ef123c856ecfde50c39f404`; Kev-27B as initialised: Hub snapshot `01b81998019be550f0ae858727df49bac9511195`.

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

## 2. Teacher labels (orders, triage, follow-up, finding status)

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

**Selection rule (pre-registered).** `v2mg` stays primary unless its calibrated dev accuracy, macro over the tasks both runs share,
is more than 1.0 pp below `v1med`'s. It was 0.03 pp below, so `v2mg` is RadKev-27B.

## 4. Held-out test (read once)

```bash
export T=rad-open,rad-gated,teacher
# every Kev-family model, Qwen3.8-27B and MedGemma (pre-registered scoring), Laya
python experiments/final_test.py --tag final --data $T --no-compare \
    --runs stock27=jaredpalmer/kev-27b,v0open27=v0open-kev-27b,stock9=jaredpalmer/kev-9b,ft9=v0open-kev-9b,v1med27=v1med-kev-27b,v2_27=v2mg-kev-27b,r9=v2x9-kev-9b,b9=v2x9-base-9b,r9f10=v2x9f10-kev-9b,b9f10=v2x9f10-base-9b,kev08=jaredpalmer/kev-0.8b,kev4=jaredpalmer/kev-4b \
    --llms qwen38=Qwen/Qwen3.8-27B,medgemma=google/medgemma-27b-text-it --laya 1
python experiments/decision_baselines.py          # Laya-typed, Julia-1, GLiNER2.5-Decide (post hoc)
python experiments/llm_scoring.py                 # qwen38_gen, medgemma_gen, medgemma_brief (post hoc)
# one comparison over everything already scored (no GPU needed)
python experiments/final_test.py --tag final --data $T --laya 1 --also v2_27,r9,r9f10 \
    --runs stock27=cached,v0open27=cached,stock9=cached,ft9=cached,v1med27=cached,v2_27=cached,r9=cached,b9=cached,r9f10=cached,b9f10=cached,kev08=cached,kev4=cached,laya_typed=cached,julia1=cached,gliner_decide=cached \
    --llms qwen38=cached,medgemma=cached,qwen38_gen=cached,medgemma_gen=cached,medgemma_brief=cached
```

`runs/test-final/comparisons.json` is the file [`results/comparisons.json`](../results/comparisons.json) was copied from. Every table in
[`results/tables/`](../results/tables/) is a view of it, of the training summaries, or of the studies below.

## 5. Studies

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/latency.py                                         # 27B tier
CUDA_VISIBLE_DEVICES=0,1 python experiments/latency.py --kev stock9=jaredpalmer/kev-9b,radkev9=v2x9-kev-9b,base9=v2x9-base-9b --llms "" --out latency_9b.json
CUDA_VISIBLE_DEVICES=0,1 python experiments/latency_scaling.py
python experiments/leak_sensitivity.py
python experiments/transfer_paired.py
```

## Without a GPU

The data builders, the teacher merge, `radkev.evaluate --preds` and `radkev.compare` run on a laptop, and `pytest` exercises them
(including Kev's metrics and bootstrap) on synthetic fixtures. Kev-0.8B and Kev-4B also run on Apple Silicon through Kev's MLX
backend, which is enough to try `radkev.predict` and the examples.
