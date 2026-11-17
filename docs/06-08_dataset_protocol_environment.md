# Phase II–III setup: dataset validation sheet, frozen protocol, environment (Prompts 06–08)

## Prompt 06: Dataset and baseline validation sheet

| Item | Value | Status |
|---|---|---|
| Primary dataset | PubMedQA PQA-L (Jin et al., EMNLP-IJCNLP 2019, pp. 2567–2577) | VERIFIED (OpenAlex) |
| Official source | HF `qiaojin/PubMedQA` (pqa_labeled); test IDs from github.com/pubmedqa/pubmedqa | VERIFIED (downloaded, SHA-256 in `data/README.md`) |
| File structure | 1000 rows: pubid, question, context{contexts, labels, meshes}, long_answer, final_decision | VERIFIED (inspected) |
| Label distribution | Full set yes 552 / no 338 / maybe 110; test yes 276 / no 169 / maybe 55 | VERIFIED |
| Benchmark protocol | "Reasoning-required" setting: question + context without the conclusion. Accuracy and macro-F1 on the 500 test items | VERIFIED from the dataset README / original paper |
| Licence | MIT | VERIFIED (dataset card) |
| Compute | Inference only; no training | — |
| Pitfalls | (1) `long_answer` leaks the conclusion and must be excluded (unit-tested); (2) maybe is rare (11%), so accuracy hides maybe errors and macro-F1 is primary; (3) free-text answers need parsing, which we avoid with restricted scoring; (4) the Llama chat template injects today's date, which we fix | Handled |
| Leakage risks | Document-level overlap between dev and test (asserted disjoint); no tuning on test; prompts frozen before scoring test | Handled |

**Comparators.** These are meaningful for a *stability* question:
- **MAJ:** majority-class baseline. It checks the reproduction of the published 55.2% majority figure.
- **B1:** the single canonical prompt, i.e. what a typical researcher does.
- **B2:** best-of-14 prompts selected on dev, i.e. "prompt engineering".

The **simplest credible baseline** is B1.

**Independent verification checklist:**
- [x] checksums match
- [x] 500 test IDs present and labels identical to the official file
- [x] test majority = 276/500 = 55.2%
- [x] conclusion text absent from the inputs
- [x] label tokens are single tokens and distinct (asserted in `LabelScorer`)

## Prompt 07: Frozen experiment blueprint

Frozen on 2026-11-17 in `configs/experiment.json` and `src/prompts.py`, before any test prediction was produced.

```mermaid
flowchart LR
  A[PubMedQA PQA-L<br/>1000 abstracts] --> B{split}
  B -->|official 500| T[test]
  B -->|80, seed 2026| D[dev]
  P[14 frozen prompts<br/>para / fmt / min] --> R
  T --> R[chat template<br/>+ prompt render]
  D --> R
  R --> S[next-token scoring<br/>p(yes,no,maybe)]
  CF[content-free inputs<br/>N/A, '', MASK] --> S
  S --> C[contextual calibration]
  S --> E[ensembles]
  S --> M1[B1 / B2 / MAJ]
  C --> E
  E & M1 & C --> EV[metrics: macro-F1, acc,<br/>flip rate, Errica sensitivity,<br/>Fleiss kappa, 2-analyst agreement]
  EV --> BS[bootstrap CIs + McNemar]
```

- **Input:** the abstract sections (BACKGROUND/METHODS/RESULTS…) as "LABEL: text" lines, plus the question.
- **Output:** a probability distribution over {yes, no, maybe}. The surface forms {yes, Yes, YES}, etc. are summed and renormalised. *Label mass* (the probability on any label token before renormalisation) is recorded as an instruction-following diagnostic.
- **Prompts:** 14 in total, all in `src/prompts.py`.

| Family | Prompts | What varies |
|---|---|---|
| Paraphrase | P0–P4 | Wording of the fully specified instruction |
| Format | P0, F1–F4 | Separators, upper case, question before abstract, ":::"-style fields; P0's wording is kept |
| Underspecified | M1–M5 | No label definitions and no output format |

- **Settings:** fp16 on MPS, batch size 4, padding on the left. There is no sampling in the main analysis (the argmax of the distribution).
- **Metrics, with direction:**
  - macro-F1 ↑ (primary) and accuracy ↑
  - flip rate ↓
  - Errica pairwise disagreement ↓
  - Fleiss κ ↑
  - accuracy/F1 range across prompts ↓
  - two-analyst agreement ↑
  - label mass ↑ (diagnostic)
- **Seeds:**
  - dev split: 2026
  - bootstrap: 2026 (2000 resamples over abstracts)
  - sampling analysis: 2026
  - two-analyst draws: 2026 (500 draws)
- **Budget:** about 7 h of MPS time and less than 1 GB of outputs.
- **Deep analyses:**
  - family ablation
  - calibration ablation
  - prompt vs sampling variance
  - cost curve for the number of prompts
  - flips by gold class and by P0 confidence
  - qualitative review of the most unstable abstracts
- **Planned tables and figures:**
  - per-prompt macro-F1 dot plot
  - flip rate by family (raw vs CC)
  - two-analyst agreement vs k
  - variance sources
  - predicted-label shares
  - methods table with CIs and McNemar
  - stability table

**Threats to validity:**
1. Small models (≤ 1.5B), so absolute accuracy is low and the results may not transfer to frontier models.
2. One dataset and a 3-way label.
3. The prompts were written by one author, so the paraphrase space is limited.
4. Restricted scoring differs from how researchers actually use chat output: it removes parsing artifacts but does not measure them.
5. fp16 MPS numerics: batch-vs-single differences of up to 0.007 in probability could flip near-ties.
6. Sampling variance is simulated from the restricted distribution, not from full free-text decoding.

**Pre-experiment checklist:**
- [x] primary metric fixed (macro-F1)
- [x] test split fixed (official 500)
- [x] prompts frozen
- [x] comparison protocol fixed
- [x] decision rules written (`04-05_…md`)
- [x] seeds fixed

**Protocol changes after freezing** are logged in `research_decisions.md` (table *Protocol change log*), with date, reason and whether earlier results were affected.

## Prompt 08: Environment verification log

Command: `python -m scripts.env_check`. Actual output: `results/logs/env_check.txt`.

```
python 3.13.5 | torch 2.9.1 | transformers 4.57.3 | numpy 2.1.3
platform macOS-26.7.1-arm64-arm-64bit-Mach-O
mps available True | cuda available False
checksum ok pqa_labeled.parquet
checksum ok test_ground_truth.json
partitions {'test': 500, 'dev': 80} test labels {'yes': 276, 'no': 169, 'maybe': 55}
```

The repository structure follows assignment §9. Dependencies are pinned in `requirements.txt` to the versions actually used. Unit tests: `python -m pytest -q tests` gives **10 passed**.
