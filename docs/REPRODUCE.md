# Reproducing the study

The study runs in four stages: data construction, teacher labeling, training and evaluation, and the manuscript build. The first
two use the `radkev` package and `experiments/teacher_labels.py`; training and evaluation use the job scripts of the reported runs
in [`experiments/final/`](../experiments/final/); the manuscript build in [`paper/`](../paper/) regenerates every number, table and
figure from the aggregate outputs of those jobs. All runs used RTX A6000 (48 GB) GPUs.

## 0. Environment

```bash
scripts/setup.sh                   # Kev @ f2bb629 and its environment, the two-GPU patch, CUDA kernels, radkev
source ~/.cache/radkev/kev/.venv/bin/activate
hf auth login                      # MedGemma-27B-text, CT-RATE and the RadKev weights are gated: accept their terms first
```

Local data, runs and caches are written under `$RADKEV_HOME` (default `~/.cache/radkev`): `raw/`, `data/` and `runs/`.

**Pinned versions.**

| Component | Revision |
|---|---|
| Kev | `f2bb629d670f5b746f712fc05550a098526c836b` |
| Kev-27B (initialization) | `01b81998019be550f0ae858727df49bac9511195` |
| Kev-9B (initialization) | `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c` |
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Qwen3.5-9B-Base | `68c46c4b3498877f3ef123c856ecfde50c39f404` |

The Hugging Face repositories of Kev received new weights after this study. Every result used the revisions above
(`radkev.paths.KEV_27B_PINNED`, `KEV_9B_PINNED`):

```bash
hf download jaredpalmer/kev-27b --revision 01b81998019be550f0ae858727df49bac9511195 --local-dir ~/.cache/radkev/kev-27b-01b8199
hf download jaredpalmer/kev-9b  --revision 2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c --local-dir ~/.cache/radkev/kev-9b-2629c06
```

## 1. Data

```bash
python -m radkev.fetch                                  # openly licensed sources -> $RADKEV_HOME/raw (no login)
# access-controlled sources: place CT-RATE's report and label CSVs under raw/ctrate/, ReXGradient-160K metadata under
# raw/rexgradient/ and CheXpert Plus tables under raw/chexpert_plus/ (python -m radkev.fetch --list shows the layout)

python -m radkev.data --raw ~/.cache/radkev/raw --out ~/.cache/radkev/data/rad-open \
    --only iu,eurorad,medmcqa,medqa,mmlu_med,pubmedqa,medxpertqa
python -m radkev.data --raw ~/.cache/radkev/raw --out ~/.cache/radkev/data/rad-gated --only ctrate --cap ctrate=8000
```

RadCases, ReXErr and the radiology filter of the knowledge sources are added by
[`experiments/final/build_v3.py`](../experiments/final/build_v3.py), and the radiology-only training and development records by
[`experiments/final/build_v3r.py`](../experiments/final/build_v3r.py). The per-source counts in [DATA.md](DATA.md#counts) and the
state hashes in the released data allow a rebuild to be checked record by record.

## 2. Teacher labels (imaging orders, triage and follow-up)

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/teacher_labels.py --per-source 1500     # -> data/teacher, approximately 1.4 h
```

## 3. Training

[`experiments/final/train.py`](../experiments/final/train.py) trains each model with `kev.train` (one epoch, 1,000 replayed
records of Kev's training data), fits a temperature on the development split, and evaluates the development split and Kev's
out-of-domain transfer suite for the specialized and the released model. RadKev-27B was trained on four GPUs as two data-parallel
ranks, each holding the backbone split over one NVLink pair; RadKev-9B as four data-parallel ranks of one GPU each. The exact
`kev.train` configurations are in
[`paper/inputs/artifacts/train_v3/artifacts/`](../paper/inputs/artifacts/train_v3/artifacts/).

| Model | Training time |
|---|---|
| RadKev-27B | 13.1 h on four RTX A6000 GPUs |
| RadKev-9B | 3.4 h on four RTX A6000 GPUs |

## 4. Evaluation

| Analysis (manuscript) | Script in `experiments/final/` | Hardware |
|---|---|---|
| Every system on the 14,142 benchmark questions: accuracy, paired differences, calibration, wording | `eval_v3.py` | 4× A6000 |
| RadCases panel prior correction and the reported benchmark analysis (Methods 2.4, Results) | `radcases_prior.py`, `eval_v3_prior.py` | CPU |
| Qwen3.8-27B with reasoning on 1,675 questions (Results 3.2, Supplementary Note S4) | `llm_reasoning.py`, `reasoning_v3_fix.py` | 4× A6000 (vLLM); CPU for the analysis |
| OpenAI Decisions and Jev (Results 3.2) | `openai_decisions.py`, `jev_decisions.py` | CPU, API keys |
| Latency, one request at a time (Results 3.2) | `latency_bench.py` | 2× A6000 |
| Kev's transfer suite, paired (Results 3.3) | `transfer_paired.py` | CPU |
| Subspecialty classification before the read; options-only control (Results 3.5) | `preread_route.py`, `blind_v3.py` | 4× A6000 |
| Answer space (Results 3.6) | `answer_space3.py`, `answer_space_hosted.py` | 4× A6000; CPU for the hosted models |
| RadGraph-XL external test (Results 3.7) | `radgraph_xl_build.py`, `external_tests.py`, `external_analyse_node.py`, `radgraph_xl_errors_v3.py`, `radgraph_xl_definite.py`, `radgraph_xl_hosted.py` | 4× A6000; CPU for the analyses |

The job scripts are reproduced as they were run on the study's compute node: each reads `$XDG_CACHE_HOME/radkev` (the Kev
checkout, its environment, the built data and the trained runs) and writes aggregate outputs only to `$ZCB_OUTPUT_DIR`. To run one
elsewhere, set both variables. [`experiments/final/README.md`](../experiments/final/README.md) lists what each script produces.

## 5. Manuscript

```bash
cd paper
python3 build.py --no-pdf          # Introduction, Methods, Table 1, Figures 1 to 3, numbers.csv
python3 v3/build_v3.py --strict    # abstract, Results, Discussion, Figures 4 to 9, supplementary tables, numbers_v3.csv
python3 build.py                   # compiles main.pdf (Tectonic) and writes an editable export
```

Every number in the manuscript is a `\V{key}` macro whose value, source file and field are listed in
[`results/numbers.csv`](../results/numbers.csv) and [`results/numbers_v3.csv`](../results/numbers_v3.csv). The public build
reproduces both ledgers exactly; see [paper/README.md](../paper/README.md).

## Without a GPU

The data builders, the teacher merge, `radkev.evaluate --preds`, `radkev.compare` and the manuscript build run on a laptop, and
`pytest` exercises the library (including Kev's metrics and bootstrap) on synthetic fixtures. Kev-0.8B and Kev-4B also run on
Apple Silicon through Kev's MLX backend, which suffices to try `radkev.predict` and the examples.
