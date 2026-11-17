"""Sanity tests for metrics, prompts and data partitions. Run: python -m pytest -q tests"""
import numpy as np
import pytest

from src import metrics as M
from src.prompts import FAMILIES, PROMPTS, render


def test_pairwise_disagreement_hand_computed():
    preds = np.array([[0, 0, 1], [0, 1, 1], [0, 2, 1]])
    # item0: all agree -> 0; item1: all differ -> 1; item2: all agree -> 0
    assert np.allclose(M.pairwise_disagreement(preds), [0, 1, 0])
    preds = np.array([[0], [0], [1]])  # pairs: (0,1) same, (0,2) diff, (1,2) diff
    assert np.allclose(M.pairwise_disagreement(preds), [2 / 3])


def test_flip_rate():
    preds = np.array([[0, 0, 1, 2], [0, 1, 1, 2]])
    assert M.flip_rate(preds) == 0.25


def test_fleiss_kappa_perfect_and_matches_statsmodels():
    perfect = np.array([[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]])
    assert M.fleiss_kappa(perfect) == pytest.approx(1.0)
    from statsmodels.stats.inter_rater import fleiss_kappa as fk
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 3, (5, 200))
    table = np.stack([(preds == c).sum(0) for c in range(3)], 1)
    assert M.fleiss_kappa(preds) == pytest.approx(fk(table), abs=1e-9)


def test_contextual_calibration_removes_bias():
    # A prompt biased towards 'yes': the content-free input gets p=(.8,.1,.1).
    p_cf = np.array([0.8, 0.1, 0.1])
    q = M.contextual_calibration(p_cf[None], p_cf)
    assert np.allclose(q, 1 / 3)  # content-free input becomes uniform
    p = np.array([[0.7, 0.2, 0.1]])  # 'yes' argmax before, 'no' after calibration
    assert M.contextual_calibration(p, p_cf).argmax() == 1


def test_ensemble_and_macro_f1():
    probs = np.array([[[0.6, 0.4, 0]], [[0.2, 0.8, 0]]])
    assert M.ensemble(probs).argmax(-1)[0] == 1
    assert M.macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0


def test_sampling_low_temperature_is_argmax():
    rng = np.random.default_rng(0)
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    s = M.sample_predictions(probs, 0.01, 50, rng)
    assert (s == probs.argmax(-1)).all()
    s = M.sample_predictions(probs, 1.0, 20000, rng)
    assert np.allclose((s[:, 0] == 0).mean(), 0.5, atol=0.02)


def test_two_analyst_identical_prompts_agree():
    probs = np.repeat(np.random.default_rng(1).dirichlet([1, 1, 1], 30)[None], 6, 0)
    assert (M.two_analyst_agreement(probs, 2, 10, np.random.default_rng(0)) == 1).all()


def test_mcnemar():
    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    bb, c, p = M.mcnemar_exact(a, b)
    assert (bb, c) == (10, 0) and p < 0.01


def test_prompt_families_and_placeholders():
    assert all(len(v) == 5 for v in FAMILIES.values())
    for pid in PROMPTS:
        txt = render(pid, "CTX_MARK", "Q_MARK")
        assert txt.count("CTX_MARK") == 1 and txt.count("Q_MARK") == 1, pid


def test_splits_disjoint_and_official():
    from src.data import load_splits
    s = load_splits(80, 2026)
    assert len(s["test"]) == 500 and len(s["dev"]) == 80
    assert not set(s["test"].pubid) & set(s["dev"].pubid)
    # the conclusion text must not leak into the model input
    for _, r in s["test"].head(50).iterrows():
        assert r.long_answer.strip()[:60] not in r.context_text
