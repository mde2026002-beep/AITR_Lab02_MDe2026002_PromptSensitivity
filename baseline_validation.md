# Baseline validation report (Prompt 10, Gate G2)

**Baseline under audit:**
- **B1:** single canonical prompt P0, restricted next-token scoring, argmax over {yes, no, maybe}.
- **MAJ:** majority-class comparator.
- **CC:** contextual calibration (Zhao et al. 2021), reproduced as a published method.

**Commands:**
- `python -m scripts.run_scoring` (raw outputs: `results/raw/`)
- `python -m scripts.validate_baseline --model <m>` (outputs: `results/validation/<m>.json`)
- `python -m scripts.analyze` (tables: `results/tables/`)
- `python -m pytest -q tests`

**Status:** Qwen2.5-0.5B is validated (§A–D). The Llama-3.2-1B and Qwen2.5-1.5B validations are added in §E once their scoring finishes. Validation is not run during scoring because it competes for the 8 GB of memory (see `ai_assistance_log.md`).

## A. Data and protocol: VERIFIED

| Check | Expected (source) | Measured | Status |
|---|---|---|---|
| PQA-L label counts | 552 / 338 / 110 (Jin et al. 2019, dataset card) | 552 / 338 / 110 | VERIFIED |
| Official test split | 500 IDs (`test_ground_truth.json`) | 500 found; 0 label mismatches (asserted in `load_splits`) | VERIFIED |
| Majority baseline on test | Accuracy 55.20, macro-F1 23.71 (Jin et al. 2019, Table 5, p. 2574; checked in the ACL Anthology PDF) | MAJ = **55.20** accuracy [51.0, 59.4], **23.71** macro-F1 | VERIFIED: exact reproduction of both published numbers |
| Dev/test disjoint | — | asserted; unit test | VERIFIED |
| Conclusion not in input | — | unit test over 50 test items | VERIFIED |
| Data integrity | — | SHA-256 matches; `download_data.sh` re-run gives OK | VERIFIED |

## B. Scorer correctness: VERIFIED, with one explained discrepancy

| Check | Result (Qwen2.5-0.5B) | Status |
|---|---|---|
| Label tokens are single and distinct | yes: 9454/9693/14004; no: 2152/2753/8996; maybe: 21390/36760 ("MAYBE" is multi-token and excluded) | VERIFIED (asserted) |
| Batch vs single-item scoring | max \|Δp\| = 0.0073 (fp16 MPS) | VERIFIED; small numerical noise |
| Rerun vs saved run (20 dev items) | max \|Δp\| = 0.010; same argmax 20/20 | VERIFIED: reproducible |
| Scoring vs greedy generation (40 dev items, P0) | 87.5% agreement with the model's **shipped** generation config; **97.5%** with `repetition_penalty=1.0` | VERIFIED after diagnosis (below) |
| Sanity questions with obvious answers (4) | 1/4 correct (smoking → yes ✓; the drug, placebo and vitamin questions ✗) | UNCERTAIN: points to model weakness, not a scorer bug (generation gives the same answers) |

**Diagnosis of the generation discrepancy.** Qwen2.5's `generation_config.json` ships `repetition_penalty: 1.1`. transformers applies it even with `do_sample=False`. The penalty lowers "maybe", which occurs in the prompt, and leaves "Yes", which does not, untouched. Under the canonical prompt it changed the generated conclusion for **6 of 40** dev abstracts. Our scorer reads raw logits and is unaffected. The single remaining mismatch is a near-tie that summing surface forms resolves the other way (yes + Yes = 0.445 vs maybe = 0.412). This is an **acceptable adaptation**, not a bug: the protocol deliberately scores the distribution rather than a decoded string. It does, however, show that "greedy" chat output depends on a hidden decoding default.

## C. Baseline behaviour: B1 is below the majority class (UNCERTAIN → explained)

| Metric (test, n = 500) | MAJ | B1 (P0) |
|---|---|---|
| Accuracy | 55.2 [51.0, 59.4] | **35.8** [31.6, 40.2] |
| Macro-F1 | 23.7 | 24.5 [21.7, 27.3] |
| Predicted shares (yes / no / maybe) | 100 / 0 / 0 | 53.8 / 0.4 / 45.8 |

B1 almost never predicts "no" (0.4%), although 33.8% of gold labels are "no". It over-predicts "maybe" (45.8% vs 11.0% gold). There is **no published Qwen2.5-0.5B PubMedQA number** to compare with (UNKNOWN; the Qwen2.5 report was not checked for one). We therefore do **not** demand numerical replication. Instead, we verified that the scorer reproduces the model's own greedy answers (B above). The low score is a model property, consistent with the 1/4 sanity result. **Consequence for the study:** results for the 0.5B model describe a near-chance model. Conclusions about stability are drawn mainly from the larger models, and every claim is reported per model.

## D. Reproduction of contextual calibration (Zhao et al. 2021): mechanism VERIFIED; claimed benefit NOT reproduced

- **Implementation:** W = diag(p̂_cf)⁻¹, b = 0, with p̂_cf averaged over the content-free inputs "N/A", "" and "[MASK]", exactly as in Zhao et al. Unit-tested: a content-free input maps to uniform, and a biased prediction is corrected.
- **Mechanism works as specified:** calibration removes the content-free bias. For example, under P0 p̂_cf = (yes 0.05, no 0.40, maybe 0.55), and after calibration the share of "maybe" predictions falls from 45.8% to 1.0%.
- **The paper's claimed benefit (higher accuracy, lower variance across prompts):**
  - Stability across prompts improves sharply. In the paraphrase family the flip rate falls from 63.4% to 15.6%, and the accuracy range from 11.2–41.6% to 50.2–55.8%.
  - However, calibrated predictions collapse onto "yes": 98.4% "yes" for P0, and 92.7% across all 14 prompts. Calibrated P0 accuracy (55.2%) equals the majority baseline, and its macro-F1 is unchanged (24.6 vs 24.5).
  - **The stability gain is largely degenerate.** That is why `stability_test.csv` reports the modal-label share next to every stability metric.
- **Discrepancy explanation (INFERRED, to be tested on the larger models):** Zhao et al. assume that the answer to a content-free input reveals pure label bias. In PubMedQA, "maybe" means *the evidence is insufficient*, which is the *correct* answer to "N/A". Calibration then treats correct behaviour as bias and over-boosts "yes". This is an **acceptable reproduction under different conditions** (instruction-tuned chat models, a 3-way task with an abstention-like class), not an implementation error.

| Item | Verdict |
|---|---|
| Data, split, leakage, majority-baseline reproduction | VERIFIED |
| Scorer reads the right position and tokens; reproducible | VERIFIED |
| Calibration implementation | VERIFIED |
| Low absolute accuracy of the 0.5B model | UNCERTAIN: no published reference; explained as model weakness |
| Calibration benefit | NOT REPRODUCED for the 0.5B model (degenerate stability); mechanism hypothesis INFERRED |
| NEEDS CORRECTION | None in code. The prompt-count description (13 → 14) was corrected in the docs |

**Additional checks required:**
1. Repeat §B–C for Llama-3.2-1B and Qwen2.5-1.5B after scoring finishes.
2. Test the "maybe is a legitimate content-free answer" hypothesis. Comparing calibration on the yes/no subset is a possible follow-up.

## E. Llama-3.2-1B (Qwen2.5-1.5B not run; see D11)

| Check | Result | Status |
|---|---|---|
| Label tokens | yes: 9642/9891/14331; no: 2201/2822/9173; maybe: 22105/37860 | VERIFIED |
| Batch vs single | max \|Δp\| = 0.0016 | VERIFIED |
| Rerun vs saved (20 dev items) | max \|Δp\| = 0.0071; same argmax 20/20 | VERIFIED |
| Scoring vs greedy generation (40 dev, P0) | **100%** agreement, both with the shipped config and with `repetition_penalty=1.0`. The shipped config has `repetition_penalty: 1.0`, so there is nothing to change. This is consistent with the Qwen diagnosis in §B | VERIFIED |
| Chat-template date | Fixed to "17 Nov 2026" (verified in the rendered template) | VERIFIED |
| Sanity questions | 1/4 (smoking ✓) | UNCERTAIN: weak model |
| B1 on test | Accuracy 55.8 [51.6, 60.0], macro-F1 25.7. Predicts "yes" for 98.2% of abstracts, so it is effectively the majority class (McNemar vs MAJ p = 0.45) | Explained |
| Contextual calibration | **Harmful**: P0 accuracy 55.8 → 11.4, and 99.6% of predictions become "maybe". The content-free distribution for P0 is (yes 0.27, no 0.73, maybe 0.003), so the calibration weight 1/p̂_cf is about 330 for "maybe", against 1.4 for "no" | NOT REPRODUCED. Supports the §D mechanism: CC over-boosts whichever label the model *correctly* avoids on empty input |

**Overall G2 verdict:**
- The data, split, scorer and metrics are verified, and the published majority baseline is reproduced exactly.
- Both models are weak in absolute terms (B1 is close to the majority class).
- Contextual calibration's published benefit does not transfer to this setting. In the two models it fails in two different directions (collapse to "yes" for Qwen, collapse to "maybe" for Llama), and the shared mechanism is identified.
- These findings are reported as results, not hidden.
