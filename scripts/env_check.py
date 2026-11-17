"""Environment verification (Prompt 08). Usage: python -m scripts.env_check

Expected output: library versions, the accelerator in use, data file checksums matching
data/README.md, partition sizes 500/80, and one scored example per model with three
label probabilities summing to 1. Any assertion error is a failed check.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

EXPECTED_SHA256 = {
    "pqa_labeled.parquet": "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c",
    "test_ground_truth.json": "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51",
}


def main(score_models: bool = "--no-models" not in sys.argv):
    print("python", platform.python_version(), "| torch", torch.__version__, "| transformers",
          transformers.__version__, "| numpy", np.__version__)
    print("platform", platform.platform())
    print("mps available", torch.backends.mps.is_available(), "| cuda available", torch.cuda.is_available())
    for f, h in EXPECTED_SHA256.items():
        got = hashlib.sha256((ROOT / "data/raw" / f).read_bytes()).hexdigest()
        assert got == h, f"checksum mismatch for {f}"
        print("checksum ok", f)
    from src.data import load_splits
    cfg = json.load(open(ROOT / "configs/experiment.json"))
    s = load_splits(cfg["dataset"]["dev_size"], cfg["dataset"]["dev_seed"])
    print("partitions", {k: len(v) for k, v in s.items()},
          "test labels", s["test"].label.value_counts().to_dict())
    if not score_models:
        return
    from src.prompts import render
    from src.scoring import LabelScorer
    r = s["dev"].iloc[0]
    for m in cfg["models"]:
        sc = LabelScorer(m["id"], m["revision"])
        p = sc.score([render("P0", r.context_text, r.question)])[0]
        assert abs(p[:3].sum() - 1) < 1e-5
        print(f"{m['short']}: device={sc.device} p(yes,no,maybe)={np.round(p[:3], 3)} label_mass={p[3]:.3f}")
        del sc
    print("ENVIRONMENT OK")


if __name__ == "__main__":
    main()
