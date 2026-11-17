"""Data loading and frozen partitions for PubMedQA (PQA-L, expert-labelled subset).

Source: Jin et al. (2019), PubMedQA. Files downloaded by scripts/download_data.sh:
  data/raw/pqa_labeled.parquet      (HF dataset qiaojin/PubMedQA, config pqa_labeled, 1000 rows)
  data/raw/test_ground_truth.json   (official 500-item test split, github.com/pubmedqa/pubmedqa)

Partitions (frozen in configs/experiment.json):
  test : the official 500 PubMedQA test ids (never used for any choice).
  dev  : 150 items sampled with a fixed seed from the remaining 500 (used only for
         dev-based prompt selection, comparator B2).
The abstract's own conclusion (long_answer) is never shown to the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ["yes", "no", "maybe"]


def format_context(ctx: dict) -> str:
    """Join abstract sections (without the conclusion) as 'LABEL: text' paragraphs."""
    parts = []
    for lab, txt in zip(ctx["labels"], ctx["contexts"]):
        parts.append(f"{str(lab).strip().upper()}: {str(txt).strip()}")
    return "\n".join(parts)


def load_pqal() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "pqa_labeled.parquet")
    df["pubid"] = df["pubid"].astype(str)
    df["context_text"] = df["context"].apply(format_context)
    df["label"] = df["final_decision"].str.strip().str.lower()
    assert set(df["label"]) <= set(LABELS), set(df["label"])
    assert df["pubid"].is_unique
    return df[["pubid", "question", "context_text", "long_answer", "label"]]


def load_splits(dev_size: int = 150, seed: int = 2026) -> dict[str, pd.DataFrame]:
    df = load_pqal()
    test_ids = json.load(open(RAW / "test_ground_truth.json"))
    test_mask = df["pubid"].isin(test_ids.keys())
    test = df[test_mask].copy()
    # Check our labels against the official ground-truth file (leakage / alignment check).
    mism = (test["label"] != test["pubid"].map(test_ids)).sum()
    assert mism == 0, f"{mism} test labels disagree with official file"
    rest = df[~test_mask].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    dev_idx = np.sort(rng.choice(len(rest), size=dev_size, replace=False))
    dev = rest.iloc[dev_idx].copy()
    assert not set(dev["pubid"]) & set(test["pubid"])
    return {"test": test.sort_values("pubid").reset_index(drop=True),
            "dev": dev.sort_values("pubid").reset_index(drop=True)}
