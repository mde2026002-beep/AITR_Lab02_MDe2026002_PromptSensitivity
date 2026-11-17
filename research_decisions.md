# Research decisions log

Each decision is listed with its reason and evidence, and the next action. Labels: VERIFIED / INFERRED / UNKNOWN.

| # | Date | Decision | Reason / evidence | Next action |
|---|---|---|---|---|
| D1 | 2026-11-17 | Stay with the review topic and narrow it to *conclusion stability* for single study abstracts | The review's §7.2 gap; gap G1 in `docs/01-03_…md`; the earlier Prompt 01/02 response (`MDe2026002_ATR`) ranked "prompt sensitivity in research-conclusion generation" #1 | Find a dataset with gold conclusions |
| D2 | 2026-11-17 | Use PubMedQA PQA-L rather than free-text conclusions from arbitrary abstracts (the original Prompt 02 plan) | Free-text conclusions would need a judge model or manual adjudication, which cannot be done reliably by one person in one week. PubMedQA gives expert yes/no/maybe conclusions for each abstract with the conclusion removed. That is a direct operationalisation of "the conclusion drawn from a study" (VERIFIED: dataset structure) | Download and checksum the data |
| D3 | 2026-11-17 | Use the official 500-item test split, plus an 80-item dev split (seed 2026) used only for B2 and MAJ | Comparable to published numbers; prevents tuning on test. Dev was cut from 150 to 80 after timing showed 1.7 s per item for the 1.5B model (measured) | Freeze the config |
| D4 | 2026-11-17 | Use restricted next-token scoring rather than free-text generation and parsing | Hua et al. 2025 show that parsing artifacts inflate sensitivity. Restricted scoring is deterministic and parse-free, and it allows contextual calibration | Record label mass as a compliance diagnostic |
| D5 | 2026-11-17 | Models: Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, with revisions pinned | Only models that fit 8 GB in fp16 and were completely cached. The Qwen2.5-3B snapshot had no weight shards (VERIFIED: `ls` of the cache) | — |
| D6 | 2026-11-17 | Three prompt families of 5 prompts each (paraphrase / format / underspecified), sharing P0 between paraphrase and format | Equal K makes stability metrics comparable across families. The families map onto mechanisms from the review (§3.1–3.3) | Freeze `src/prompts.py` |
| D7 | 2026-11-17 | Comparators MAJ, B1 and B2; mitigations CC (Zhao et al. 2021) and ENS | The assignment asks for ≥ 2 baselines; CC is a published method from the review, which gives the reproduction component | Unit-test the implementation |
| D8 | 2026-11-17 | Primary metric is macro-F1 rather than accuracy | maybe makes up 11% of test; accuracy rewards ignoring it | — |
| D9 | 2026-11-17 | Decision rules and hypotheses H1–H3 written (`docs/04-05_…md`) | Written **after** dev-split accuracies for qwen0.5b prompts P0–F3 had been printed by the runner, but **before any test-split prediction existed** (checked in `results/logs/run_all.out`). Dev results could in principle have influenced the 10% threshold. This is disclosed | Run test |
| D10 | 2026-11-17 | Novelty claim downgraded to "controlled course-level study" | The OpenAlex search found Hariprasad 2026 and Sadanandan 2026 already measuring prompt sensitivity on PubMedQA (VERIFIED: abstracts in `docs/verification/`) | Position against them in Related Work |
| D11 | 2026-11-17 | Qwen2.5-1.5B-Instruct dropped from the runs; the study reports Qwen2.5-0.5B and Llama-3.2-1B | Submission deadline. The 1.5B model needed about 4 h more on the M1 (measured 1.7–1.9 s per item). The decision was made **before any 1.5B output existed**, so it cannot be result-driven | Report as a compute limitation |

## Protocol change log (after freezing)

| Date | Change | Reason | Results affected? |
|---|---|---|---|
| 2026-11-17 | Model loading changed to local-snapshot paths with `local_files_only` first | transformers 4.57.3 makes a network call from `_patch_mistral_regex` even in offline mode, and the gated Llama repo returned 401 on a re-download attempt (see `ai_assistance_log.md`) | No. Same weights and same revision hashes |
| 2026-11-17 | `logits_to_keep=1` added to the forward pass | Speed only: computing the vocabulary projection at every position was wasteful | No. The last-position logits are mathematically identical (smoke test values unchanged to 3 decimals) |
| 2026-11-17 | Model set reduced from 3 to 2 (D11) | Deadline / compute | No results discarded: the 1.5B model was never scored. The analysis uses only models with complete raw outputs |
