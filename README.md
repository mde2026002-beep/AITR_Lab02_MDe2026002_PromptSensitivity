# Same Abstract, Different Conclusion: prompt sensitivity of LLM-drawn research conclusions

This is the implementation study (AI Tools for Research, Lab 02) that follows the literature review *Prompt Sensitivity and Its Effect on the Stability of LLM-Generated Research Conclusions*.

**Research question.** When a small open LLM reads a biomedical abstract (with its conclusion removed) and states the study's conclusion (yes / no / maybe), how often does that conclusion change across semantically equivalent prompts? Which kind of prompt change causes it? Do contextual calibration (Zhao et al. 2021) or multi-prompt ensembles fix it?

**Pathway:** B, a controlled comparative study, with a reproduction of a published method (contextual calibration).

## Repository map

| Path | Contents |
|---|---|
| `src/data.py` | PubMedQA loading, official test split, dev split (seed 2026), leakage assertions |
| `src/prompts.py` | The 14 frozen prompts in 3 families (paraphrase, format, underspecified) |
| `src/scoring.py` | `LabelScorer`: chat template + restricted next-token scoring of yes/no/maybe |
| `src/metrics.py` | Macro-F1, Errica sensitivity, flip rate, Fleiss κ, contextual calibration, ensembles, two-analyst agreement, bootstrap, McNemar |
| `scripts/` | `download_data.sh`, `env_check.py`, `run_scoring.py`, `validate_baseline.py`, `analyze.py` |
| `configs/experiment.json` | Frozen protocol: models and revisions, splits, seeds, metrics |
| `tests/` | 10 unit tests |
| `results/raw/` | Per-item label probabilities for every model, split and prompt (genuine outputs) |
| `results/tables/`, `results/figures/`, `results/summary.json` | Everything reported in the paper |
| `results/logs/` | Run logs with timing and environment per prompt; environment check |
| `results/validation/` | Baseline-validation outputs |
| `paper/` | LaTeX source, bibliography, figures, compiled PDF |
| `docs/` | Phase documents for Prompts 01–08, 12–14 and 16; reference and novelty verification |
| `research_decisions.md`, `baseline_validation.md`, `ai_assistance_log.md` | Provenance (assignment §8D) |

## Setup

Tested on macOS 26.7.1, Apple M1 (8 GB), Python 3.13.5, with PyTorch MPS.

```bash
pip install -r requirements.txt          # pinned to the versions actually used
bash scripts/download_data.sh            # PubMedQA PQA-L + official test ids, checksum-verified
python -m scripts.env_check --no-models  # versions, checksums, partitions (500 test / 80 dev)
python -m pytest -q tests                # expected: 10 passed
```

**Models** (downloaded from Hugging Face on first use, with revisions pinned in `configs/experiment.json`):
- `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0)
- `meta-llama/Llama-3.2-1B-Instruct` (Llama 3.2 Community License; **gated**: accept the licence on Hugging Face and run `huggingface-cli login`)

Weights are not redistributed.

## Reproduce the results

```bash
python -m scripts.run_scoring --models qwen0.5b llama1b   # ~1 h + ~2.7 h on an M1; resumable per prompt
python -m scripts.validate_baseline --model qwen0.5b      # sanity, scoring-vs-generation, rerun checks
python -m scripts.validate_baseline --model llama1b
python -m scripts.analyze                                 # tables, figures, summary.json (~1 min)
cd paper && tectonic main.tex                             # paper PDF
```

- `run_scoring` writes one CSV per (model, split, prompt), skips files that already exist, and logs the time per prompt to `results/logs/run_<model>.jsonl`.
- To check reproducibility quickly, delete one file (for example `results/raw/qwen0.5b/test__P0.csv`) and rerun `run_scoring --models qwen0.5b`. About 5 minutes later the probabilities should match to within about 0.01 (fp16 MPS noise; argmax identical in our check).
- `python -m scripts.run_scoring --limit 3` is a 3-item smoke test that writes to `results/smoke/`.
- **Compute:** 580 items × 14 prompts per model. Measured on the M1: 0.52 s per pass (0.5B) and about 1.2 s per pass (Llama 1B). Swap pressure on 8 GB machines can double these times; do not run two models at once.
- The third configured model (Qwen2.5-1.5B) was **not run** because of time constraints (`research_decisions.md`, D11). `analyze.py` uses only models with complete outputs.

## Key design choices

- **Parse-free scoring.** The model's next-token probabilities for yes / Yes / YES (and so on) are summed; there is no free-text parsing.
- **No conclusion leakage.** The abstract's conclusion (`long_answer`) is never in the input (unit-tested).
- **Fixed chat-template date for Llama**, which otherwise injects today's date.
- **Primary metric:** macro-F1. Stability is measured by flip rate, Errica sensitivity, Fleiss κ and two-analyst agreement, always reported with the modal-label share so that a model that always answers the same way does not look "stable".

## Limitations

- Small models only (0.5B and 1B), one dataset, and a three-way label.
- The prompts were written by one author.
- Sampling variance is simulated from the restricted label distribution, not from full free-text decoding.
- See the paper, Section "Limitations".

## Credits and licence

- **Data:** PubMedQA (Jin et al., EMNLP-IJCNLP 2019), MIT.
- **Models:** Qwen Team (Apache-2.0); Meta (Llama 3.2 Community License).
- **Code:** MIT. AI assistance is documented in `ai_assistance_log.md`.

**Citation:** Sumanth, J. S. S. (2026). *Same Abstract, Different Conclusion: A Controlled Study of Prompt Sensitivity in LLM-Drawn Research Conclusions.* IIIT Allahabad, AI Tools for Research.

## Minimum reproduction checklist for a second student

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `bash scripts/download_data.sh` prints `OK` twice
- [ ] `python -m pytest -q tests` gives 10 passed
- [ ] `python -m scripts.env_check --no-models` shows partitions `{'test': 500, 'dev': 80}`
- [ ] `python -m scripts.analyze` regenerates `results/tables/methods_test.csv` identically from the shipped `results/raw/`
- [ ] Optional (about 5 min): delete and re-score `results/raw/qwen0.5b/test__P0.csv`; accuracy prints `acc=0.358`
