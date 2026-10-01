#!/usr/bin/env bash
# The main line, end to end: data -> teacher labels -> RadKev-27B -> held-out test vs Kev-27B and Qwen3.8-27B.
# Needs two 48 GB GPUs (or one 80 GB), the environment from scripts/setup.sh, and Hugging Face access to the gated repos
# (CT-RATE, MedGemma-27B-text). About 30 GPU-hours. Every step skips work that is already done, so it can be re-run.
# The ablations, baselines and studies are in docs/REPRODUCE.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RADKEV_HOME="${RADKEV_HOME:-$HOME/.cache/radkev}"; export RADKEV_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RAW="$RADKEV_HOME/raw"; DATA="$RADKEV_HOME/data"
T=rad-open,rad-gated,teacher

python -m radkev.fetch
[ -f "$DATA/rad-open/manifest.json" ] || python -m radkev.data --raw "$RAW" --out "$DATA/rad-open" --only iu,eurorad,medmcqa,medqa,mmlu_med,pubmedqa,medxpertqa
[ -f "$DATA/rad-gated/manifest.json" ] || python -m radkev.data --raw "$RAW" --out "$DATA/rad-gated" --only ctrate --cap ctrate=8000
[ -f "$DATA/teacher/manifest.json" ] || python experiments/teacher_labels.py --per-source 1500
python experiments/train.py --models kev-27b --data "$T" --tag v2mg
python experiments/final_test.py --tag final --data "$T" \
    --runs stock27=jaredpalmer/kev-27b,v2_27=v2mg-kev-27b --llms qwen38=Qwen/Qwen3.8-27B --also v2_27
python - <<EOF
import json
c = json.load(open("$RADKEV_HOME/runs/test-final/comparisons.json"))
for m, v in c["models"].items(): print(f"{m:10s} acc {v['overall']['acc']:.3f}  ece {v['overall']['ece']:.3f}")
d = c["vs_reference"]["v2_27"]["overall_macro_acc"]
print(f"RadKev-27B - Kev-27B, macro accuracy: {100 * d['delta']:+.1f} pp [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]   (paper: +6.8 [+5.8, +7.7])")
EOF
