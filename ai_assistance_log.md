# AI assistance log (Prompts 09, 11, 18)

**Tool:** Claude Code (Anthropic), model Claude Opus 5.5, running in a terminal on the project machine with file and shell access. The student's instruction was "complete the assignment with every phase". The AI then planned, wrote and ran the code, ran the experiments, and drafted the documents and paper. All numbers come from scripts that the AI executed on this machine; the logs are in `results/logs/`.

**Important for the student (ownership):** Assignment §2/§10 requires *you* to understand, verify and defend this work. Before submission:
- re-run `python -m pytest -q tests` and `python -m scripts.analyze`
- read `src/` end-to-end
- re-check the decisions in `research_decisions.md`
- add your own entries below wherever you change or reject something

## Significant AI-assisted activities

| # | Task | Tool / prompt | AI proposal | Human decision | Verification method | Code/result change | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | Read the assignment and earlier work | Claude Code; "read the assignmnent pdf" | Summarised the brief; found `main.tex` (review) and `MDe2026002_ATR` (Prompt 01/02) | Student asked to complete every phase | Manual reading of the PDFs | — | Scope fixed to the review topic |
| 2 | Choose the dataset | Claude Code (Prompts 02–06 reasoning) | PubMedQA PQA-L instead of free-text conclusions | *Pending student review* | Downloaded; label counts match the paper (552/338/110); 500 official test IDs found | `src/data.py` | VERIFIED |
| 3 | Scorer implementation | Claude Code (Prompt 09 style) | Restricted next-token scoring with summed surface forms | *Pending student review* | Smoke run on 3 models; batch-vs-single max difference 0.0073 (0.5B), 0.0024 (1.5B), 0.0016 (Llama) | `src/scoring.py` | VERIFIED |
| 4 | Metrics | Claude Code | Errica sensitivity, flip rate, Fleiss κ, CC, ensembles, bootstrap, McNemar | *Pending student review* | 10 unit tests, including Fleiss κ against `statsmodels` (abs 1e-9) | `src/metrics.py`, `tests/` | 10 passed |
| 5 | Reference verification | OpenAlex + arXiv APIs | Checked 18 references; ran a novelty search | — | Raw API output saved in `docs/verification/` | `paper/references.bib` | 16/18 matched by OpenAlex; Qwen2.5 and Llama 3 confirmed via arXiv; Gelman & Loken (2013 tech report) not indexed (UNKNOWN in OpenAlex) |
| 6 | Speed-up and deadline decision | Claude Code | Options presented: close apps; 300-item subset; MLX port; skip dev. Later, under deadline: drop Qwen2.5-1.5B, submit 1 model + partial, or run all 3 overnight | **Student chose to drop Qwen2.5-1.5B (D11)** | A watcher stopped the run right after Llama's last file; the 1.5B model only loaded and produced no output | `scripts/analyze.py` uses only complete models | 2 models reported |
| 7 | Baseline validation | `scripts/validate_baseline.py` | Sanity questions, scoring vs generation, rerun check | — | Majority baseline checked against Table 5 of the PubMedQA PDF (55.20 / 23.71, exact) | `baseline_validation.md` | VERIFIED; CC benefit not reproduced |
| 8 | Analysis and figures | `scripts/analyze.py`; dataviz guidance (validated palette slots 1–3) | Tables, 5 figures, LaTeX tables | — | Rendered every figure and paper page and inspected them; fixed legend overlap, an empty panel, a diagram arrow crossing text, and the `<` glyph | `results/`, `paper/` | Done |
| 9 | Manuscript, peer review, README | Claude Code (Prompts 15–17) | Drafted paper (abstract last), self-review, README | **Pending student review** | Each number in the text checked against `results/tables/*.csv` | `paper/`, `docs/16_…md`, `README.md` | Compiled PDF, 11 pages |

## Real AI limitations, corrections and verifications (Prompt 11 cases)

These all happened during this project. The tracebacks are reproduced from the actual terminal output.

1. **Offline-mode crash in transformers (a library bug the AI did not anticipate).**
   - The AI's first scorer passed the hub model ID with `HF_HUB_OFFLINE=1` set. It failed with `OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct`, raised from `transformers/tokenization_utils_base.py::_patch_mistral_regex → is_base_mistral → model_info`.
   - Confirmed cause: transformers 4.57.3 queries the hub for *every* tokenizer load to check whether it is a Mistral model.
   - Fix: resolve the pinned local snapshot directory and load from that path.
   - Verification: the rerun succeeded for both Qwen models.
2. **Gated-repo 401 for Llama-3.2-1B.**
   - The first fix used `snapshot_download(model_id, revision)`, which tried to fetch missing files (`original/consolidated.00.pth`) from the gated repo. That failed with `GatedRepoError: 401`, although the safetensors weights were cached.
   - Fix: try `local_files_only=True` first; if that fails, download only `*.json`, `*.safetensors` and `*.txt`.
   - Verification: Llama loaded and was scored.
3. **Performance mistake in the AI's first version.**
   - The first scorer computed full-vocabulary logits at every token position: about 450 positions × 151k vocab × batch. Measured speed was 0.98 s per item (0.5B) and 2.5 s per item (1.5B).
   - Fix: `logits_to_keep=1`.
   - Verification: probabilities unchanged to 3 decimals; speed 0.60 and 1.9 s per item.
   - The batch-size test then showed batch 1 is *not* slower than batch 8 on this M1 (1.70 vs 2.22 s per item for 1.5B). This was not what the AI expected; batch size 4 was kept.
4. **Hidden prompt drift found by inspecting the rendered template (purposeful verification).**
   - The Llama-3.2 chat template inserts `Today Date: <current date>` into the system header. Results would silently change from day to day.
   - Fix: pass a fixed `date_string="17 Nov 2026"`.
   - Verification: printed the rendered template (`…Cutting Knowledge Date: December 2023\nToday Date: 17 Nov 2026…`).
5. **Bibliography errors carried over from the first paper.**
   - Verifying the review's `references.bib` against arXiv found two problems:
     - The first author of *Calibrate Before Use* was given as "Zhao, Zihao". arXiv and OpenAlex list **Tony Z. Zhao**.
     - Sadanandan 2026 was entered as `@inproceedings` with an arXiv number as the "booktitle", and the first author's given name was abbreviated.
   - Both were corrected in `paper/references.bib`.
   - The original `../references.bib` (first paper) was **not** modified. The student should fix it there too.
6. **The AI's own counting error: 13 vs 14 prompts.**
   - The AI described the prompt set as "13 prompts" in the config, docs and analysis labels. The analysis of the first model reported 14 test prompts (7000 passes) and exposed the error.
   - Cause: P0–P4 (5) + F1–F4 (4) + M1–M5 (5) = 14 distinct prompts. P0 is shared by two families of 5 and was counted only once on one side.
   - The data and code were correct. Only the descriptions were wrong, and they were fixed everywhere.
   - Lesson: the AI's prose claims about its own code must be checked against the outputs.
7. **A wrong hypothesis rejected by evidence, and a hidden decoding default found.**
   - Validation compared restricted scoring with greedy `generate()` and found only 87.5% agreement (35/40).
   - The AI hypothesised that summing surface forms ("maybe" + "Maybe") was the cause. A direct check **rejected** this: in 5 of 6 mismatches the single top token at the scored position was "maybe", but `generate()` emitted "Yes".
   - Second hypothesis, **confirmed**: Qwen2.5's shipped `generation_config.json` has `repetition_penalty: 1.1`, which transformers applies even with `do_sample=False`. "maybe" occurs in the prompt and is penalised; "Yes" (capitalised) does not and is not penalised.
   - With `repetition_penalty=1.0`, agreement rises to 97.5% (39/40). The last case is a genuine near-tie (yes + Yes = 0.445 vs maybe = 0.412).
   - The shipped penalty changed the conclusion for 6 of 40 dev abstracts under the canonical prompt. That is a hidden harness parameter of exactly the kind the review warns about.
   - The main experiment is unaffected because it reads raw logits. Reported in the paper as a validation finding.
8. **Scope correction from the novelty search.**
   - The working framing was "conclusion stability on PubMedQA". The OpenAlex search found two 2026 papers already measuring prompt sensitivity on PubMedQA.
   - The novelty claim was downgraded and the differences were made explicit (D10).
9. **Resource contention caused by the AI.**
   - Running baseline validation (a second model) while the Llama scoring run was active pushed the 8 GB machine into swap (3.4 GB). Llama dev prompts slowed from about 105 s to 446–661 s.
   - Corrected by not running models concurrently. Results were unaffected (verified by the rerun check: same argmax 20/20).
10. **Imprecise wording caught when checking numbers.**
    - The AI first described calibration as boosting labels "about 20×" and "about 300×". These are the absolute weights 1/p̂_cf, not relative boosts, and the text was corrected.
    - "Pre-registered" was corrected to "pre-specified", because the protocol was frozen in the repository, not registered externally.

## What depended on human reasoning (to be completed by the student)

- [ ] Agreement with the dataset substitution (D2) and the prompt wording in `src/prompts.py`
- [ ] Interpretation of results in the paper's Discussion
- [ ] Final novelty positioning against Hariprasad 2026 and Sadanandan 2026 (read those papers in full)
- [ ] Author name, affiliation and enrollment number on the paper

## Lessons about vibe coding (from this project)

- Library behaviour (offline mode, gated repos, chat templates that inject dates) was the main source of failures, not the experiment logic. Each failure was caught only because the code was *run* and its output read.
- Smoke tests on 3–16 items, with a direct comparison of batch and single scoring, caught issues before the 7-hour run.
- Frozen configs and per-prompt checkpoint files made the long run resumable and auditable.

## Remaining risks

- The AI wrote most of the code. The student must be able to explain `LabelScorer.score`, `contextual_calibration` and `two_analyst_agreement` in the viva.
- The results come from small models. Conclusions should not be generalised to GPT-4-class systems without further runs.
