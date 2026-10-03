"""Where RadKev keeps its working files: one root, $RADKEV_HOME (default ~/.cache/radkev).

    raw/     downloaded sources (radkev.fetch), plus any gated sources you place there yourself
    data/    built record suites: rad-open, rad-gated, teacher, and the training mixes mix-<tag>
    kev/     the Kev checkout at the pinned commit (scripts/setup.sh)
    runs/    training runs and scored evaluations

Experiments run inside the Kev environment (scripts/setup.sh installs radkev into it), so `sys.executable` is Kev's
Python. Gated Hugging Face repos (MedGemma, CT-RATE) read the usual HF_TOKEN or `hf auth login` credentials.
"""
import os
import sys
from pathlib import Path

HOME = Path(os.environ.get("RADKEV_HOME", Path.home() / ".cache" / "radkev")).expanduser()
RAW, DATA, RUNS = HOME / "raw", HOME / "data", HOME / "runs"
KEV_DIR = Path(os.environ.get("KEV_DIR", HOME / "kev")).expanduser()
KEV_PY = Path(os.environ.get("KEV_PYTHON", sys.executable))
LAYA_PY = Path(os.environ.get("LAYA_PYTHON", HOME / "laya-venv" / "bin" / "python"))
VLLM_PY = Path(os.environ.get("VLLM_PYTHON", HOME / "vllm-venv" / "bin" / "python"))   # experiments/llm_reasoning.py only

KEV_REPO = "https://github.com/jaredpalmer/kev.git"
KEV_COMMIT = "f2bb629d670f5b746f712fc05550a098526c836b"   # every result in this repo used this Kev commit

# released checkpoints and their pinned backbones
KEV_27B, KEV_9B = "jaredpalmer/kev-27b", "jaredpalmer/kev-9b"
# the Kev revisions scored in the paper (the Hub repos later received new weights); post hoc experiments load these
KEV_27B_REV, KEV_9B_REV = "01b81998019be550f0ae858727df49bac9511195", "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"
KEV_27B_PINNED, KEV_9B_PINNED = f"{KEV_27B}@{KEV_27B_REV}", f"{KEV_9B}@{KEV_9B_REV}"
QWEN38_27B, QWEN38_27B_REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN35_9B, QWEN35_9B_REV = "Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"
MEDGEMMA_27B = "google/medgemma-27b-text-it"


def resolve_run(name):
    """A run name ("v2mg-kev-27b") -> its checkpoint directory under runs/; Hub ids and paths pass through."""
    ck = RUNS / name / "checkpoint"
    return str(ck) if ck.exists() else str(name)
