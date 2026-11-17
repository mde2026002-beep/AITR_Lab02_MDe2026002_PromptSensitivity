# Data

**Dataset:** PubMedQA, PQA-L subset (1,000 expert-annotated questions). Jin et al., EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1259.
**Licence:** MIT (PubMedQA GitHub repository and Hugging Face dataset card `qiaojin/PubMedQA`). Abstracts are from PubMed.
**Not committed:** `data/raw/` is git-ignored. Run `bash scripts/download_data.sh` to download and checksum-verify the files.

| File | Source | SHA-256 |
|---|---|---|
| `raw/pqa_labeled.parquet` | https://huggingface.co/api/datasets/qiaojin/PubMedQA/parquet/pqa_labeled/train/0.parquet | `3d56bd1a…7be231c` |
| `raw/test_ground_truth.json` | https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json | `939fe566…c44d51` |

**Record structure:** `pubid`, `question` (usually the article title phrased as a question), `context.contexts` / `context.labels` (abstract sections *without* the conclusion), `long_answer` (the abstract's conclusion, never given to the model), and `final_decision` ∈ {yes, no, maybe}.

**Partitions** (code: `src/data.py`; settings: `configs/experiment.json`)
- **test:** the 500 official PubMedQA test IDs (yes 276 / no 169 / maybe 55). Used only for final evaluation. The loader asserts that every label matches the official file.
- **dev:** 80 items sampled with seed 2026 from the remaining 500 (yes 39 / no 31 / maybe 10). Used only to select the prompt for comparator B2 and the majority class.
- The remaining 420 items are unused.

**Leakage controls:** test and dev are disjoint (asserted), the conclusion text is excluded from the input (unit-tested), and no prompt, setting or threshold was tuned on test.
