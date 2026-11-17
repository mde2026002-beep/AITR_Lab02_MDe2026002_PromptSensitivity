"""Score every (model, split, prompt, abstract) and every content-free calibration input.

Usage:
    python -m scripts.run_scoring                      # all models in configs/experiment.json
    python -m scripts.run_scoring --models qwen0.5b    # one model
    python -m scripts.run_scoring --limit 10           # quick smoke run (writes to results/smoke)

Outputs (one file per model/split/prompt so runs can resume after interruption):
    results/raw/<model>/<split>__<prompt>.csv   columns: pubid,label,p_yes,p_no,p_maybe,label_mass
    results/raw/<model>/content_free.csv        columns: prompt,cf_input,p_yes,p_no,p_maybe,label_mass
    results/logs/run_<model>.jsonl              timing + environment per prompt
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_splits  # noqa: E402
from src.prompts import ALL_IDS, render  # noqa: E402
from src.scoring import LabelScorer  # noqa: E402

CFG = json.load(open(ROOT / "configs" / "experiment.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    splits = load_splits(CFG["dataset"]["dev_size"], CFG["dataset"]["dev_seed"])
    out_root = ROOT / ("results/smoke" if args.limit else "results/raw")
    bs = CFG["inference"]["batch_size"]

    for m in CFG["models"]:
        if args.models and m["short"] not in args.models:
            continue
        odir = out_root / m["short"]
        odir.mkdir(parents=True, exist_ok=True)
        log = open(ROOT / "results" / "logs" / f"run_{m['short']}{'_smoke' if args.limit else ''}.jsonl", "a")
        t0 = time.perf_counter()
        sc = LabelScorer(m["id"], m["revision"])
        env = {"python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "device": sc.device,
               "platform": platform.platform(), "label_token_ids": sc.label_token_ids}
        print(f"[{m['short']}] loaded in {time.perf_counter() - t0:.1f}s on {sc.device}", flush=True)

        # content-free inputs for contextual calibration
        cf_path = odir / "content_free.csv"
        if not cf_path.exists():
            rows = []
            for pid in ALL_IDS:
                cfs = CFG["content_free_inputs"]
                probs = sc.score([render(pid, c, c) for c in cfs], batch_size=len(cfs))
                for c, p in zip(cfs, probs):
                    rows.append({"prompt": pid, "cf_input": c, "p_yes": p[0], "p_no": p[1],
                                 "p_maybe": p[2], "label_mass": p[3]})
            pd.DataFrame(rows).to_csv(cf_path, index=False)

        for split in ("dev", "test"):
            df = splits[split]
            if args.limit:
                df = df.head(args.limit)
            for pid in ALL_IDS:
                path = odir / f"{split}__{pid}.csv"
                if path.exists():
                    continue
                t = time.perf_counter()
                texts = [render(pid, c, q) for c, q in zip(df.context_text, df.question)]
                probs = sc.score(texts, batch_size=bs)
                res = pd.DataFrame({"pubid": df.pubid, "label": df.label, "p_yes": probs[:, 0],
                                    "p_no": probs[:, 1], "p_maybe": probs[:, 2], "label_mass": probs[:, 3]})
                res.to_csv(path, index=False, float_format="%.6f")
                dt = time.perf_counter() - t
                rec = {"time_utc": datetime.now(timezone.utc).isoformat(), "model": m["id"],
                       "revision": m["revision"], "split": split, "prompt": pid, "n": len(df),
                       "seconds": round(dt, 2), "forward_passes": len(df), **env}
                log.write(json.dumps(rec) + "\n"); log.flush()
                acc = (res[["p_yes", "p_no", "p_maybe"]].values.argmax(1) ==
                       res.label.map({"yes": 0, "no": 1, "maybe": 2}).values).mean()
                print(f"[{m['short']}] {split} {pid}: n={len(df)} {dt:.0f}s acc={acc:.3f}", flush=True)
        del sc
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
