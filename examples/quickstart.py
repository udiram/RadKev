"""Ask RadKev three kinds of radiology decisions and print each answer's probabilities.

    python examples/quickstart.py --run jaredpalmer/kev-27b                    # the released generalist, for comparison
    python examples/quickstart.py --run $RADKEV_HOME/runs/v2mg-kev-27b/checkpoint
    python examples/quickstart.py --run jaredpalmer/kev-4b                     # small enough for a laptop (MLX on Apple Silicon)

The example cases in this folder are invented for illustration; they are not from any dataset.
"""
import argparse
import json
from pathlib import Path

from radkev.predict import Predictor

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="Kev checkpoint: Hub id or local directory")
a = ap.parse_args()
predictor = Predictor(a.run)
for path in sorted(Path(__file__).parent.glob("*.json")):
    request = json.loads(path.read_text())
    result = predictor(request)
    print(f"\n{path.stem}  ({result['latency_ms']} ms, {result['input_tokens']} tokens)")
    for qid, ans in result["answers"].items():
        probs = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
        print(f"  {qid:14s} " + "  ".join(f"{k}={p:.2f}" for k, p in probs[:4]))
