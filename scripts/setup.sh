#!/usr/bin/env bash
# One-time environment setup. Everything lives under $RADKEV_HOME (default ~/.cache/radkev).
#
#   scripts/setup.sh            # Kev at the pinned commit + its venv, with radkev installed into it
#   scripts/setup.sh --laya     # also a separate Laya venv (only for the Laya baselines)
#
# Needs git and uv (https://docs.astral.sh/uv/). Training and the 27B evaluations need CUDA GPUs; scoring with
# --preds, comparisons and the data builders run anywhere.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RADKEV_HOME="${RADKEV_HOME:-$HOME/.cache/radkev}"
KEV_DIR="${KEV_DIR:-$RADKEV_HOME/kev}"
KEV_REPO="https://github.com/jaredpalmer/kev.git"
KEV_COMMIT="f2bb629d670f5b746f712fc05550a098526c836b"
LAYA_VERSION="0.3.20"
CAUSAL_CONV1D="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
mkdir -p "$RADKEV_HOME"

echo "==> Kev @ ${KEV_COMMIT:0:7} in $KEV_DIR"
[ -d "$KEV_DIR/.git" ] || git clone -q "$KEV_REPO" "$KEV_DIR"
git -C "$KEV_DIR" fetch -q origin
git -C "$KEV_DIR" checkout -q --detach "$KEV_COMMIT"
(cd "$KEV_DIR" && uv sync --frozen --extra serve)
KEV_PY="$KEV_DIR/.venv/bin/python"

echo "==> multi-GPU patch (lets Kev-27B split over two 48 GB cards: KEV_DEVICE_MAP=auto)"
if git -C "$KEV_DIR" apply --check --reverse "$REPO/patches/kev_multigpu.patch" 2>/dev/null; then
  echo "    already applied"
else
  git -C "$KEV_DIR" apply "$REPO/patches/kev_multigpu.patch"
fi

if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null; then
  echo "==> CUDA kernels for the Gated-DeltaNet layers (Kev's own pins)"
  uv pip install -q --python "$KEV_PY" flash-linear-attention || echo "    flash-linear-attention failed; Kev falls back to slower reference code"
  "$KEV_PY" -c "import causal_conv1d" 2>/dev/null || uv pip install -q --python "$KEV_PY" --no-deps "$CAUSAL_CONV1D" || echo "    causal-conv1d failed; slower fallback"
fi

echo "==> radkev into Kev's venv"
uv pip install -q --python "$KEV_PY" -e "$REPO"

if [ "${1:-}" = "--laya" ]; then
  echo "==> Laya $LAYA_VERSION venv (baselines only)"
  [ -x "$RADKEV_HOME/laya-venv/bin/python" ] || uv venv -q "$RADKEV_HOME/laya-venv" --python 3.12
  uv pip install -q --python "$RADKEV_HOME/laya-venv/bin/python" "laya==$LAYA_VERSION"
fi

"$KEV_PY" - <<'EOF'
import json, torch, transformers, peft, kev, radkev
print(json.dumps({"radkev": radkev.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                  "gpus": torch.cuda.device_count(), "transformers": transformers.__version__, "peft": peft.__version__}))
EOF
cat <<EOF

Done. Activate the environment with:
    source "$KEV_DIR/.venv/bin/activate"
Then see docs/REPRODUCE.md, or run examples/quickstart.py.
EOF
