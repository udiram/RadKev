"""Download the open sources into $RADKEV_HOME/raw (idempotent; standard library only).

    python -m radkev.fetch            # everything that needs no login
    python -m radkev.fetch --list     # what goes where, including the gated sources you fetch yourself

Open sources come from Open-i (IU reports) and the Hugging Face parquet conversions of public datasets. Gated or
credentialed sources (CT-RATE, ReXGradient-160K, CheXpert Plus, MIMIC-CXR) need you to accept their terms first; this
script only prints where radkev.data expects them. See docs/DATA.md.
"""
import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from radkev.paths import RAW

PQ = "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/{}/{}/0000.parquet"
OPEN = {
    "iu/NLMCXR_reports.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
    "eurorad/eurorad.parquet": PQ.format("wanglab/eurorad-reasoning", "default", "train"),
    "medmcqa/medmcqa_train.parquet": PQ.format("openlifescienceai/medmcqa", "default", "train"),
    "medmcqa/medmcqa_val.parquet": PQ.format("openlifescienceai/medmcqa", "default", "validation"),
    "medqa/medqa_train.parquet": PQ.format("GBaker/MedQA-USMLE-4-options", "default", "train"),
    "medqa/medqa_test.parquet": PQ.format("GBaker/MedQA-USMLE-4-options", "default", "test"),
    "pubmedqa/pqa_labeled.parquet": PQ.format("qiaojin/PubMedQA", "pqa_labeled", "train"),
    "medxpertqa/text_test.parquet": PQ.format("TsinghuaC3I/MedXpertQA", "Text", "test"),
}
for _sub in ("anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine", "college_biology"):
    for _sp in ("validation", "test"):
        OPEN[f"mmlu/{_sub}_{_sp}.parquet"] = PQ.format("cais/mmlu", _sub, _sp)

GATED = {
    "ctrate/": "CT-RATE (huggingface.co/datasets/ibrahimhamamci/CT-RATE): train_reports.csv, validation_reports.csv, "
               "train_predicted_labels.csv, valid_predicted_labels.csv",
    "rexgradient/": "ReXGradient-160K (huggingface.co/datasets/rajpurkarlab/ReXGradient-160K): train/valid/test_metadata.csv "
                    "(teacher questions only)",
    "chexpert_plus/": "CheXpert Plus (Stanford AIMI): df_chexpert_plus_*.csv and the CheXbert impression label file",
    "mimic/": "MIMIC-CXR (PhysioNet, credentialed): mimic-cxr-reports.zip, mimic-cxr-2.0.0-chexpert.csv.gz, "
              "mimic-cxr-2.0.0-split.csv.gz (not used for the released models)",
}


def fetch(url, dest, tries=4):
    for i in range(tries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "radkev-fetch/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while chunk := r.read(8 << 20): f.write(chunk)
            tmp.replace(dest); return
        except Exception:
            if i == tries - 1: raise
            time.sleep(10 * (i + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(RAW), help="destination (default: $RADKEV_HOME/raw)")
    ap.add_argument("--list", action="store_true", help="print the expected layout and exit")
    a = ap.parse_args()
    raw = Path(a.raw)
    if a.list:
        for rel, url in OPEN.items(): print(f"{raw / rel}\n    <- {url}")
        for rel, what in GATED.items(): print(f"{raw / rel}\n    <- gated: {what}")
        return
    info = {}
    for rel, url in OPEN.items():
        dest = raw / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"fetching {rel}", file=sys.stderr); fetch(url, dest)
        info[rel] = {"bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    if not (raw / "iu/ecgen-radiology").exists():
        with tarfile.open(raw / "iu/NLMCXR_reports.tgz") as t:
            t.extractall(raw / "iu", filter="data") if sys.version_info >= (3, 12) else t.extractall(raw / "iu")
    (raw / "open_manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v["bytes"] for k, v in info.items()}, indent=1))
    missing = [rel for rel in GATED if not (raw / rel).exists()]
    if missing: print(f"gated sources not present (optional, see docs/DATA.md): {', '.join(m.rstrip('/') for m in missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
