"""Baseline validation checks (Prompt 10). Usage: python -m scripts.validate_baseline --model qwen0.5b

1. Sanity questions with obvious answers must be scored correctly.
2. Scoring vs generation: for N dev abstracts under P0, greedy-decode up to 6 new tokens,
   take the first word, and compare it with the argmax of restricted scoring. If the scorer
   reads the right position and the right tokens, the two should agree whenever the
   generated word is a label.
3. Reproducibility: re-score 20 dev items and compare with the saved run file.
Output: results/validation/<model>.json and a printed summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import LABELS, load_splits  # noqa: E402
from src.prompts import render  # noqa: E402
from src.scoring import LabelScorer  # noqa: E402

CFG = json.load(open(ROOT / "configs" / "experiment.json"))
SANITY = [  # (context, question, expected)
    ("RESULTS: All 120 patients given the drug recovered fully; none of the 118 controls recovered.",
     "Does the drug lead to recovery?", "yes"),
    ("RESULTS: The intervention and placebo groups had identical outcomes (risk ratio 1.00, 95% CI 0.98-1.02) "
     "in a trial of 40,000 participants.", "Does the intervention improve outcomes compared with placebo?", "no"),
    ("RESULTS: Smoking was strongly associated with lung cancer (odds ratio 15.2, p<0.001).",
     "Is smoking associated with lung cancer?", "yes"),
    ("RESULTS: Vitamin X had no effect on blood pressure (mean difference 0.1 mmHg, p=0.94).",
     "Does vitamin X lower blood pressure?", "no"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    m = next(x for x in CFG["models"] if x["short"] == a.model)
    sc = LabelScorer(m["id"], m["revision"])
    out: dict = {"model": m["id"], "revision": m["revision"]}

    # 1. sanity
    probs = sc.score([render("P0", c, q) for c, q, _ in SANITY], batch_size=4)
    out["sanity"] = [{"question": q, "expected": e, "predicted": LABELS[int(p[:3].argmax())],
                      "p": [round(float(v), 3) for v in p[:3]]} for (c, q, e), p in zip(SANITY, probs)]
    out["sanity_correct"] = int(sum(r["expected"] == r["predicted"] for r in out["sanity"]))

    # 2. scoring vs greedy generation
    dev = load_splits(CFG["dataset"]["dev_size"], CFG["dataset"]["dev_seed"])["dev"].head(a.n)
    texts = [render("P0", c, q) for c, q in zip(dev.context_text, dev.question)]
    sc_probs = sc.score(texts, batch_size=4)
    # Greedy decoding twice: with the model's shipped generation_config (which may contain a
    # repetition_penalty that still applies when do_sample=False) and with repetition_penalty=1.0.
    def first_word(g):
        w = re.findall(r"[a-zA-Z]+", g.lower())
        return w[0] if w else ""

    out["shipped_generation_config"] = {k: getattr(sc.model.generation_config, k, None)
                                        for k in ("do_sample", "repetition_penalty", "temperature", "top_p", "top_k")}
    rows = []
    with torch.inference_mode():
        for t, p in zip(texts, sc_probs):
            enc = sc.tok(sc.chat_text(t), return_tensors="pt", add_special_tokens=False).to(sc.device)
            row = {"scored_argmax": LABELS[int(p[:3].argmax())], "label_mass": float(p[3])}
            for tag, kw in (("default", {}), ("rp1", {"repetition_penalty": 1.0})):
                ids = sc.model.generate(**enc, max_new_tokens=6, do_sample=False,
                                        pad_token_id=sc.tok.pad_token_id, **kw)[0, enc.input_ids.shape[1]:]
                g = sc.tok.decode(ids, skip_special_tokens=True)
                row[f"gen_{tag}"], row[f"word_{tag}"] = g.strip(), first_word(g)
            rows.append(row)
    df = pd.DataFrame(rows)
    gc = {"n": len(df)}
    for tag in ("default", "rp1"):
        lab = df[df[f"word_{tag}"].isin(LABELS)]
        gc[f"{tag}_generated_word_is_label"] = int(len(lab))
        gc[f"{tag}_agreement_with_scorer"] = float((lab[f"word_{tag}"] == lab.scored_argmax).mean()) if len(lab) else None
    gc["conclusion_changed_by_shipped_penalty"] = int((df.word_default != df.word_rp1).sum())
    gc["examples"] = df.head(8).to_dict(orient="records")
    out["generation_check"] = gc

    # 3. reproducibility vs saved run
    saved = pd.read_csv(ROOT / "results/raw" / a.model / "dev__P0.csv", dtype={"pubid": str}).head(20)
    assert (saved.pubid.values == dev.pubid.values[:20]).all()
    diff = np.abs(saved[["p_yes", "p_no", "p_maybe"]].values - sc_probs[:20, :3]).max()
    same = (saved[["p_yes", "p_no", "p_maybe"]].values.argmax(1) == sc_probs[:20, :3].argmax(1)).mean()
    out["rerun_vs_saved"] = {"max_abs_prob_diff": float(diff), "same_argmax": float(same)}

    (ROOT / "results/validation").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results/validation" / f"{a.model}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "generation_check"}, indent=1))
    gc = out["generation_check"]
    print("generation check:", {k: v for k, v in gc.items() if k != "examples"})
    print(df[["gen_default", "gen_rp1", "scored_argmax", "label_mass"]].head(8).to_string())


if __name__ == "__main__":
    main()
