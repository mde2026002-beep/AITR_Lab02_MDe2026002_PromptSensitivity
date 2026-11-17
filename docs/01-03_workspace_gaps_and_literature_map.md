# Phase I: from review to experiment (Prompts 01–03)

Evidence labels: **VERIFIED** means checked against the original source or an actual execution. **INFERRED** means a plausible interpretation that needs testing. **UNKNOWN** means the evidence is missing.

## Prompt 01: Workspace and review scope

- **Review being converted:** *Prompt Sensitivity and Its Effect on the Stability of LLM-Generated Research Conclusions* (first paper, `../main.tex`, `../AI_Assisted_Literature_Review_Prompt_Sensitivity (1).pdf`).
- **Earlier Prompt 01/02 response:** `../MDe2026002_ATR (1).pdf`. It set up the workspace and the gap matrix summarised below.
- **Review scope (VERIFIED by reading `main.tex`):** a narrative synthesis covering formatting and example-order sensitivity, semantic paraphrase sensitivity, task underspecification, reasoning-strategy interactions, decoding stochasticity, and the "evaluation artifact" counter-argument. It also covers mitigations (contextual calibration, self-consistency, sensitivity indices, multi-prompt evaluation) and the gap between benchmark-level sensitivity and the stability of *research conclusions*.
- **Workspace:** this repository. Its structure follows Section 9 of the assignment brief. Decisions are recorded in `research_decisions.md`.

## Prompt 02: Gap matrix (source evidence → candidate question → possible experiment)

| # | Gap | Evidence in the review (section / reference) | Original limitation or my inference | Experimental question | Evidence that would test it |
|---|---|---|---|---|---|
| G1 | Benchmark-level sensitivity has not been translated into *conclusion-level* stability | §1, §3.2, §7.2; Mizrahi et al. 2024; Errica et al. 2025 | **Inference.** Those papers measure task accuracy, not whether a research conclusion drawn from one source stays the same | How often does the conclusion an LLM draws from one abstract change across equivalent prompts? | Flip rate and pairwise disagreement on a dataset of abstracts with gold conclusions |
| G2 | Relative contribution of paraphrase, format and underspecification | §3.1–3.3; Sclar et al. 2024; Pecher et al. 2026 | **Original.** Pecher et al. argue that underspecification drives much of the sensitivity. That has not been tested on a research-conclusion task | Which family of prompt change destabilises conclusions most? | Compare stability across prompt families of equal size |
| G3 | Whether published mitigations stabilise conclusions, not just accuracy | §4.1, §4.3; Zhao et al. 2021; Wang et al. 2022 | **Original.** Zhao et al. report that calibration reduces variance across prompts on classification tasks. Transfer to small instruction-tuned models on biomedical conclusions is **UNKNOWN** | Do contextual calibration and multi-prompt ensembling reduce conclusion flips? | Ablation: raw vs calibrated, single prompt vs ensemble |
| G4 | Prompt variation versus run-to-run variation | §3.5; Ouyang et al. 2025; Shyr et al. 2025 | **Inference** | Is the variation from rewording larger than the variation from repeated sampling? | Within-prompt sampling disagreement vs between-prompt disagreement |
| G5 | Evaluation artifacts inflate measured sensitivity | §3.6; Hua et al. 2025 | **Original.** Hua et al. argue that rigid answer parsing inflates sensitivity | Does the measured sensitivity survive when free-text parsing is removed? | Score options by restricted next-token probability, which needs no parsing |

**Ranking** (significance × feasibility on an 8 GB laptop):
1. G1, which addresses the review's central question directly.
2. G3, which reproduces a published mitigation, so there is a baseline to validate.
3. G2/G4, which come almost for free from the same runs.

G5 is handled *by design*: the main scoring method removes parsing altogether. Measuring the parsing artifact itself is left as follow-up work. Novelty is not claimed (see `04-05_feasibility_novelty_and_brief.md`).

## Prompt 03: Literature-to-experiment map

| Published method (from the review) | Algorithm | Dataset in original | Code/weights access | Metric | Known limitation | Compute here |
|---|---|---|---|---|---|---|
| Contextual calibration, Zhao et al. 2021 (ICML) | Estimate label bias from content-free inputs ("N/A", "", "[MASK]"); rescale by diag(p_cf)^-1 | SST-2, AGNews, TREC, … (GPT-2/GPT-3) | Algorithm fully specified in the paper; reimplemented in `src/metrics.py` (10 lines, unit-tested) | Accuracy; variance across prompts | Shown on few-shot base models; instruction-tuned chat models untested in the original | None beyond 3 extra passes per prompt |
| Sensitivity and consistency, Errica et al. 2025 (NAACL) | Pairwise prediction disagreement across rephrasings; per-class agreement | Classification tasks | Formula in the review (Eq. 1–2); reimplemented and unit-tested | Sensitivity ↓, consistency ↑ | Classification only | Negligible |
| Multi-prompt evaluation and ensembling, Mizrahi et al. 2024 (TACL); self-consistency, Wang et al. 2022 | Evaluate or aggregate over a set of paraphrased prompts | LMentry, BBH, … | Idea only; implemented as probability averaging | Accuracy spread; model rankings | K× inference cost | K forward passes per item |

**Study options compared:**
- **(A) Reproduction + extension:** reproduce contextual calibration and extend it to conclusion stability on PubMedQA. The smallest baseline is the single-prompt argmax.
- **(B) Controlled comparison:** compare 3 models × 3 prompt families × 2 calibration settings under one protocol.
- **(C) Targeted modification:** change one component (calibration) of the baseline pipeline.

**Chosen:** **B with an A component.** It is a controlled comparative study, and the published contextual calibration method is reproduced as one of the conditions. This fits an Apple M1 with 8 GB and about one week: roughly 24k forward passes, about 7 h.

**Mandatory bridge:** review §3.2/§4.1 → Errica et al. 2025 + Zhao et al. 2021 → restricted-scoring baseline with the single canonical prompt P0 on PubMedQA → planned test of flip rate, κ and macro-F1 across prompt families, with and without calibration and ensembling.

**Things I had to verify personally before coding** (all done):
- dataset licence and access (VERIFIED, MIT, downloaded and checksummed)
- the official test IDs (VERIFIED, 500)
- model weights available locally (VERIFIED for Qwen2.5-0.5B/1.5B and Llama-3.2-1B; the Qwen2.5-3B cache was incomplete, so that model was excluded)
- every reference's metadata (`docs/verification/openalex_reference_check.tsv`)
