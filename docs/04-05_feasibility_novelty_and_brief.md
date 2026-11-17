# Prompts 04–05: Feasibility, novelty and the one-page research brief (Gate G1)

> **Update after the runs (2026-11-17):** only Qwen2.5-0.5B and Llama-3.2-1B were run. Qwen2.5-1.5B was dropped for time before it produced any output (decision D11). The plan below is kept as originally frozen.

## Prompt 04: Feasibility and novelty verification

**Search strings run against OpenAlex on 2026-11-17** (raw results: `docs/verification/openalex_novelty_search.tsv`):
- "prompt sensitivity PubMedQA"
- "prompt sensitivity biomedical question answering"
- "contextual calibration prompt variance"
- "multi-prompt evaluation medical question answering"
- "prompt paraphrase stability systematic review large language model"
- "LLM conclusions reproducibility prompt wording scientific abstracts"

Abstracts of the closest neighbours were read from OpenAlex and saved in `docs/verification/neighbour_abstracts.txt`.

| Published work | VERIFIED contribution (source: abstract, OpenAlex) | Limitation relative to my question | Proposed difference | Evidence still missing |
|---|---|---|---|---|
| Hariprasad 2026, arXiv:2603.00917 | 5 small open models × MedQA/MedMCQA/PubMedQA × 5 prompt styles. Consistency and accuracy are largely independent, and models can be "reliably wrong" | Prompt *styles* (roleplay, formal, …) are mixed with parsing failures ("UNKNOWN" outputs); no mitigation tested | Restricted scoring removes parse failures; families are separated (paraphrase / format / underspecified); calibration and ensembles are tested; prompt and sampling variance are separated | — |
| Sadanandan 2026, Research Square / arXiv:2603.25960 | MedGemma on PubMedQA: CoT −5.7 points, few-shot −11.9 points; option shuffling flips 59.1% of predictions | Reasoning-strategy and option-order manipulations only | Equivalent-instruction families, conclusion-level agreement | — |
| Barrie et al. 2024, arXiv:2407.02039 | Prompt Stability Score built on intra- and inter-coder reliability | Text annotation in the social sciences | I use Fleiss κ across prompts in the same spirit, on biomedical conclusions | — |
| Cao et al. 2024, NeurIPS | Worst-prompt performance; existing consistency methods have limited impact | Open-ended generation (RobustAlpacaEval) | I report worst and best prompt (accuracy range) and test whether calibration or ensembles help | — |
| Zhao et al. 2021, ICML | Contextual calibration reduces variance across prompts | GPT-2/3 few-shot classification | Reproduced on instruction-tuned 0.5–1.5B models and a 3-way research-conclusion task | Whether the result transfers (to be tested) |

**Novelty statement:** Prompt sensitivity on PubMedQA has already been measured (Hariprasad 2026; Sadanandan 2026). This project therefore does **not** claim to be the first to measure it. The contribution is a *controlled, course-level* study with these features: conclusion-level stability metrics, families of prompt change kept separate, a reproduced mitigation, and a split between prompt and sampling variance, all under a parse-free scoring protocol.

**Access and licences (VERIFIED):**
- **PubMedQA:** MIT licence, downloaded and checksummed.
- **Qwen2.5-0.5B/1.5B-Instruct:** Apache-2.0.
- **Llama-3.2-1B-Instruct:** Llama 3.2 Community License, gated. It was already in the local cache, and the code never redistributes weights.

**Compute (measured, not estimated):**
- Hardware: Apple M1, 8 GB unified memory, MPS fp16.
- Throughput with ~450-token prompts: 0.58 s per item (0.5B), 1.3 s (Llama 1B), 1.7–1.9 s (1.5B).
- Total cost: 580 items × 14 prompts × 3 models = 24,360 passes, about 7 h.
- Fallback if this were too slow: a 300-item stratified test subset. It was not needed.

**Go/no-go:** **GO.**

## Prompt 05: Research problem, questions and success criteria

**Problem.** Researchers who use LLMs to read study abstracts and draw a conclusion fix one prompt, but many equally reasonable prompts exist. If the conclusion depends on that arbitrary choice, LLM-assisted evidence synthesis gains a hidden researcher degree of freedom (review §1, Gelman & Loken 2013).

| | RQ1: Conclusion stability | RQ2: Source of instability | RQ3: Mitigation |
|---|---|---|---|
| Question | How often does the yes/no/maybe conclusion for the same abstract change across semantically equivalent prompts? | Do paraphrase, surface-format and underspecification changes destabilise conclusions to different degrees, and how does prompt variation compare with sampling variation? | Do contextual calibration (Zhao et al. 2021) and multi-prompt ensembling make conclusions more stable and more accurate than a single prompt? |
| Input | 500 PubMedQA test abstracts (conclusion removed) + question | same | same (+ content-free inputs; dev set for B2) |
| Comparator | Within-model across prompts | Families of 5 prompts; sampled repeats of one prompt at T ∈ {0.7, 1.0} | MAJ, B1 (canonical prompt), B2 (dev-selected prompt) |
| Independent variables | Prompt (14), model (3) | Family; temperature | Calibration on/off; ensemble size |
| Dependent variables | Flip rate, pairwise disagreement (Errica), Fleiss κ, accuracy/macro-F1 range | same, per family | Macro-F1 (primary), accuracy, flip rate, two-analyst agreement |
| Decision rule | Conclusions count as *unstable* if the flip rate within the paraphrase family is ≥ 10% of abstracts, with the 95% bootstrap CI excluding 10% | A family differs if its flip-rate CIs do not overlap | A mitigation *helps* if macro-F1 improves over B1 (McNemar on accuracy, p < 0.05) **or** the flip rate falls with non-overlapping CIs without lowering macro-F1 |
| Observation that would contradict H | Flip rate < 10% for every model | Overlapping CIs for all families | Calibration or ensembling lowers macro-F1 or raises the flip rate |

**Hypotheses** (stated in advance):
- **H1:** the flip rate within the paraphrase family is ≥ 10% for every model.
- **H2:** underspecified prompts are less stable than fully specified paraphrases (following Pecher et al. 2026).
- **H3:** calibration reduces disagreement across prompts (following Zhao et al. 2021).

**Already demonstrated vs to be investigated:**
- Already demonstrated: that accuracy varies with the prompt (Sclar; Mizrahi; Hariprasad).
- To be investigated: conclusion-level agreement, the families compared, and the effect of the mitigations on stability.

## One-page research brief (Section 11 template)

| Field | Project information |
|---|---|
| Review topic and key papers | Prompt sensitivity and the stability of LLM-generated research conclusions. Key papers: Zhao et al. 2021 (ICML); Sclar et al. 2024 (ICLR); Mizrahi et al. 2024 (TACL); Errica et al. 2025 (NAACL); Pecher et al. 2026 (arXiv:2602.04297). All verified via OpenAlex/arXiv. |
| Research gap | The review (§7.2) finds that sensitivity is measured at benchmark level, not at the level of the conclusion drawn from one study. Mitigations are rarely evaluated for stability. |
| Research question | How often, and through which prompt changes, does an LLM's yes/no/maybe conclusion about a study abstract change, and do calibration or ensembling reduce this? |
| Study pathway | B: controlled comparative study, with a reproduction of contextual calibration (A component). |
| Dataset / measurement | PubMedQA PQA-L (MIT). Official 500-item test split; 80-item dev split (seed 2026). |
| Baseline / comparators | MAJ; B1 single canonical prompt; B2 dev-selected prompt. Mitigations: CC, ENS, ENS+CC. Models: Qwen2.5-0.5B/1.5B-Instruct, Llama-3.2-1B-Instruct (local weights, revisions pinned). |
| Primary metric and control | Macro-F1 (accuracy); flip rate, pairwise disagreement and Fleiss κ (stability). Held constant: abstracts, model revision, dtype, scoring method, label tokens, chat template (fixed date). |
| Computing and risks | M1, 8 GB, about 7 h. Risks: small models near chance; swap pressure; gated Llama weights. Fallbacks: 300-item subset; drop Llama. |
| First executable milestone | Done: data checksums, partitions (500/80), a smoke-run of the scorer on all 3 models, and batch-vs-single agreement (max difference 0.007). |
