"""Compute every table and figure in the paper from results/raw/.

Usage: python -m scripts.analyze
Writes results/tables/*.csv, results/tables/*.tex, results/figures/*.pdf|png and
results/summary.json. Every number in the paper is read from these files.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics as M  # noqa: E402
from src.prompts import ALL_IDS, CANONICAL, FAMILIES  # noqa: E402

CFG = json.load(open(ROOT / "configs" / "experiment.json"))
RAW, TAB, FIG = ROOT / "results/raw", ROOT / "results/tables", ROOT / "results/figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = CFG["bootstrap"]["seed"]; NBOOT = CFG["bootstrap"]["n"]
def _complete(short: str) -> bool:
    d = ROOT / "results/raw" / short
    return all((d / f"{s}__{p}.csv").exists() for s in ("dev", "test") for p in ALL_IDS) and \
        (d / "content_free.csv").exists()


# Models with a complete set of raw outputs (Qwen2.5-1.5B was not run; see research_decisions.md D11).
MODELS = [m["short"] for m in CFG["models"] if _complete(m["short"])
          and (not os.environ.get("MODELS") or m["short"] in os.environ["MODELS"].split(","))]
NICE = {"qwen0.5b": "Qwen2.5-0.5B", "llama1b": "Llama-3.2-1B", "qwen1.5b": "Qwen2.5-1.5B"}
# Reference categorical palette, slots 1-3 (validated all-pairs, light mode)
COL = {"qwen0.5b": "#2a78d6", "llama1b": "#eb6834", "qwen1.5b": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300, "legend.frameon": False})


# ------------------------------------------------------------------ loading
def load(model: str, split: str):
    frames = [pd.read_csv(RAW / model / f"{split}__{p}.csv", dtype={"pubid": str}) for p in ALL_IDS]
    ids = frames[0].pubid.values
    for f in frames:
        assert (f.pubid.values == ids).all(), "pubid alignment broken"
        assert f[["p_yes", "p_no", "p_maybe"]].notna().all().all()
    probs = np.stack([f[["p_yes", "p_no", "p_maybe"]].values for f in frames])  # K,N,3
    mass = np.stack([f.label_mass.values for f in frames])
    y = frames[0].label.map(M.L2I).values
    cf = pd.read_csv(RAW / model / "content_free.csv", keep_default_na=False)
    p_cf = np.stack([cf[cf.prompt == p][["p_yes", "p_no", "p_maybe"]].values.mean(0) for p in ALL_IDS])
    return ids, probs, mass, y, p_cf


def calibrate_all(probs, p_cf):
    return np.stack([M.contextual_calibration(probs[k], p_cf[k]) for k in range(len(ALL_IDS))])


IDX = {p: i for i, p in enumerate(ALL_IDS)}
FAM_SETS = {**{f: [IDX[p] for p in ps] for f, ps in FAMILIES.items()}, "all14": list(range(len(ALL_IDS)))}


def fmt_ci(t, pct=True, d=1):
    s = 100 if pct else 1
    return f"{t[0]*s:.{d}f} [{t[1]*s:.{d}f}, {t[2]*s:.{d}f}]"


def main():
    D = {m: {s: load(m, s) for s in ("dev", "test")} for m in MODELS}
    summary: dict = {"n_test": int(len(D[MODELS[0]]["test"][0])), "n_dev": int(len(D[MODELS[0]]["dev"][0]))}
    y = D[MODELS[0]]["test"][3]
    for m in MODELS:
        assert (D[m]["test"][3] == y).all()
    summary["test_label_counts"] = {l: int((y == i).sum()) for i, l in enumerate(M.LABELS)}

    # ============================================================ T1 per-prompt results
    rows = []
    for m in MODELS:
        _, probs, mass, yy, p_cf = D[m]["test"]
        cal = calibrate_all(probs, p_cf)
        for p in ALL_IDS:
            k = IDX[p]
            pr, pc = probs[k].argmax(-1), cal[k].argmax(-1)
            rows.append({"model": m, "prompt": p, "family": "+".join(f for f in FAMILIES if p in FAMILIES[f]),
                         "acc": M.accuracy(pr, yy), "macro_f1": M.macro_f1(pr, yy),
                         "acc_cc": M.accuracy(pc, yy), "macro_f1_cc": M.macro_f1(pc, yy),
                         "share_yes": (pr == 0).mean(), "share_no": (pr == 1).mean(), "share_maybe": (pr == 2).mean(),
                         "share_yes_cc": (pc == 0).mean(), "share_no_cc": (pc == 1).mean(),
                         "share_maybe_cc": (pc == 2).mean(),
                         "label_mass": mass[k].mean(),
                         "cf_yes": p_cf[k, 0], "cf_no": p_cf[k, 1], "cf_maybe": p_cf[k, 2]})
    per_prompt = pd.DataFrame(rows)
    per_prompt.to_csv(TAB / "per_prompt_test.csv", index=False, float_format="%.4f")

    # ============================================================ T2 baselines & mitigations
    rng = np.random.default_rng(SEED)
    rows, mcn = [], []
    for m in MODELS:
        _, probs, _, yy, p_cf = D[m]["test"]
        _, dprobs, _, dy, dcf = D[m]["dev"]
        cal = calibrate_all(probs, p_cf)
        dev_f1 = [M.macro_f1(dprobs[k].argmax(-1), dy) for k in range(len(ALL_IDS))]
        best = int(np.argmax(dev_f1))
        maj = np.bincount(dy, minlength=3).argmax()
        methods = {
            "MAJ (majority class)": np.full_like(yy, maj),
            "B1 single canonical prompt (P0)": probs[IDX[CANONICAL]].argmax(-1),
            f"B2 dev-selected prompt": probs[best].argmax(-1),
            "CC calibrated P0": cal[IDX[CANONICAL]].argmax(-1),
            "ENS-para (5 prompts)": M.ensemble(probs[FAM_SETS["para"]]).argmax(-1),
            "ENS-all (14 prompts)": M.ensemble(probs).argmax(-1),
            "ENS+CC-para (5 prompts)": M.ensemble(cal[FAM_SETS["para"]]).argmax(-1),
            "ENS+CC-all (14 prompts)": M.ensemble(cal).argmax(-1),
        }
        b1_correct = methods["B1 single canonical prompt (P0)"] == yy
        for name, pred in methods.items():
            acc = M.bootstrap_ci(lambda p, t: M.accuracy(p, t), pred, yy, n=NBOOT, seed=SEED)
            f1 = M.bootstrap_ci(lambda p, t: M.macro_f1(p, t), pred, yy, n=NBOOT, seed=SEED)
            b, c, pv = M.mcnemar_exact(pred == yy, b1_correct)
            rows.append({"model": m, "method": name, "acc": acc[0], "acc_lo": acc[1], "acc_hi": acc[2],
                         "macro_f1": f1[0], "f1_lo": f1[1], "f1_hi": f1[2],
                         "pred_maybe_share": float((pred == 2).mean()),
                         "mcnemar_vs_B1_b": b, "mcnemar_vs_B1_c": c, "mcnemar_vs_B1_p": pv,
                         "forward_passes_per_item": {"MAJ": 0, "B1": 1, "B2": 1, "CC": 1}.get(name.split()[0],
                                                     5 if "para" in name else len(ALL_IDS)),
                         "selected_prompt": ALL_IDS[best] if name.startswith("B2") else ""})
    methods_df = pd.DataFrame(rows)
    methods_df.to_csv(TAB / "methods_test.csv", index=False, float_format="%.4f")

    # ============================================================ T3 stability by family (raw vs CC)
    rows = []
    for m in MODELS:
        _, probs, _, yy, p_cf = D[m]["test"]
        cal = calibrate_all(probs, p_cf)
        for cond, P in (("raw", probs), ("calibrated", cal)):
            for fam, ks in FAM_SETS.items():
                preds = P[ks].argmax(-1)
                sens = M.bootstrap_ci(lambda a: M.pairwise_disagreement(a).mean(), preds, n=NBOOT, seed=SEED)
                flip = M.bootstrap_ci(lambda a: M.flip_rate(a), preds, n=NBOOT, seed=SEED)
                accs = [M.accuracy(p, yy) for p in preds]
                f1s = [M.macro_f1(p, yy) for p in preds]
                cons = M.errica_consistency(preds, yy)
                rows.append({"model": m, "condition": cond, "family": fam, "K": len(ks),
                             "sensitivity": sens[0], "sens_lo": sens[1], "sens_hi": sens[2],
                             "flip_rate": flip[0], "flip_lo": flip[1], "flip_hi": flip[2],
                             "fleiss_kappa": M.fleiss_kappa(preds),
                             "acc_min": min(accs), "acc_max": max(accs), "acc_range": max(accs) - min(accs),
                             "f1_min": min(f1s), "f1_max": max(f1s), "f1_range": max(f1s) - min(f1s),
                             # degenerate-stability diagnostic: share of all predictions in the modal label
                             "modal_label": M.LABELS[int(np.bincount(preds.ravel(), minlength=3).argmax())],
                             "modal_label_share": float(np.bincount(preds.ravel(), minlength=3).max() / preds.size),
                             "mean_macro_f1": float(np.mean(f1s)),
                             **{f"consistency_{k}": v for k, v in cons.items()}})
    stab = pd.DataFrame(rows)
    stab.to_csv(TAB / "stability_test.csv", index=False, float_format="%.4f")

    # ============================================================ T4 prompt vs sampling variance
    rows = []
    for m in MODELS:
        _, probs, _, yy, _ = D[m]["test"]
        greedy = probs.argmax(-1)
        between = M.pairwise_disagreement(greedy[FAM_SETS["para"]]).mean()
        rows.append({"model": m, "source": "between paraphrases (greedy)", "temperature": 0,
                     "disagreement": between})
        for T in CFG["sampling_analysis"]["temperatures"]:
            r = np.random.default_rng(SEED)
            within = []
            for k in FAM_SETS["para"]:
                s = M.sample_predictions(probs[k], T, CFG["sampling_analysis"]["samples_per_prompt"], r)
                within.append(M.pairwise_disagreement(s).mean())
            rows.append({"model": m, "source": "within prompt (repeated sampling)", "temperature": T,
                         "disagreement": float(np.mean(within))})
            # combined: one sample from each paraphrase
            s_comb = np.stack([M.sample_predictions(probs[k], T, 1, r)[0] for k in FAM_SETS["para"]])
            rows.append({"model": m, "source": "between paraphrases (sampled)", "temperature": T,
                         "disagreement": M.pairwise_disagreement(s_comb).mean()})
    var_df = pd.DataFrame(rows)
    var_df.to_csv(TAB / "variance_sources_test.csv", index=False, float_format="%.4f")

    # ============================================================ T5 model-ranking instability
    f1tab = per_prompt.pivot(index="prompt", columns="model", values="macro_f1").loc[ALL_IDS, MODELS]
    rankings = f1tab.apply(lambda r: tuple(r.sort_values(ascending=False).index), axis=1)
    taus = [kendalltau(f1tab.iloc[i], f1tab.iloc[j]).statistic for i, j in combinations(range(len(ALL_IDS)), 2)]
    best_counts = f1tab.idxmax(axis=1).value_counts().to_dict()
    rank_info = {"distinct_rankings": int(rankings.nunique()),
                 "ranking_counts": {" > ".join(k): int(v) for k, v in rankings.value_counts().items()},
                 "best_model_counts": {k: int(v) for k, v in best_counts.items()},
                 "mean_pairwise_kendall_tau": float(np.nanmean(taus)),
                 "share_prompt_pairs_with_different_winner": float(np.mean(
                     [f1tab.idxmax(axis=1).iloc[i] != f1tab.idxmax(axis=1).iloc[j]
                      for i, j in combinations(range(len(ALL_IDS)), 2)]))}
    summary["model_ranking"] = rank_info

    # ============================================================ T6 two-analyst agreement vs k
    rows = []
    for m in MODELS:
        _, probs, _, yy, p_cf = D[m]["test"]
        cal = calibrate_all(probs, p_cf)
        pool = FAM_SETS["para"] + [IDX[p] for p in FAMILIES["fmt"] if p != CANONICAL]  # 9 specified prompts
        for cond, P in (("raw", probs), ("calibrated", cal)):
            for k in range(1, 5):
                ag = M.two_analyst_agreement(P[pool], k, 500, np.random.default_rng(SEED))
                rows.append({"model": m, "condition": cond, "k": k, "agreement_mean": ag.mean(),
                             "agreement_p05": np.quantile(ag, 0.05), "agreement_min": ag.min()})
    ta = pd.DataFrame(rows)
    ta.to_csv(TAB / "two_analyst_agreement_test.csv", index=False, float_format="%.4f")

    # ============================================================ T7 flips by gold class + examples
    rows, ex = [], []
    for m in MODELS:
        ids, probs, _, yy, _ = D[m]["test"]
        preds = probs[FAM_SETS["para"]].argmax(-1)
        sens = M.pairwise_disagreement(preds)
        flipped = (preds != preds[0]).any(0)
        for c in range(3):
            mask = yy == c
            rows.append({"model": m, "gold": M.LABELS[c], "n": int(mask.sum()),
                         "flip_rate_para": float(flipped[mask].mean()),
                         "sensitivity_para": float(sens[mask].mean()),
                         "acc_P0": float((preds[0][mask] == c).mean())})
        # margin analysis: canonical prompt confidence vs flipping
        conf = probs[IDX[CANONICAL]].max(-1)
        q = pd.qcut(conf, 4, labels=False, duplicates="drop")
        for b in np.unique(q):
            rows.append({"model": m, "gold": f"P0-confidence quartile {int(b)+1}", "n": int((q == b).sum()),
                         "flip_rate_para": float(flipped[q == b].mean()),
                         "sensitivity_para": float(sens[q == b].mean()), "acc_P0": float(
                             (preds[0][q == b] == yy[q == b]).mean())})
        for i in np.argsort(-sens)[:15]:
            ex.append({"model": m, "pubid": ids[i], "gold": M.LABELS[yy[i]],
                       **{p: M.LABELS[probs[IDX[p], i].argmax()] for p in ALL_IDS}})
    pd.DataFrame(rows).to_csv(TAB / "flips_by_class_test.csv", index=False, float_format="%.4f")
    pd.DataFrame(ex).to_csv(TAB / "most_unstable_examples_test.csv", index=False)
    # abstracts whose conclusion flips for ALL three models within the para family
    flip_all = np.ones(len(y), bool)
    for m in MODELS:
        pr = D[m]["test"][1][FAM_SETS["para"]].argmax(-1)
        flip_all &= (pr != pr[0]).any(0)
    summary["abstracts_flipping_for_all_models_para"] = int(flip_all.sum())

    # ============================================================ cost
    logs = []
    for m in MODELS:
        f = ROOT / "results/logs" / f"run_{m}.jsonl"
        if f.exists():
            logs += [json.loads(l) for l in open(f)]
    if logs:
        lg = pd.DataFrame(logs)
        lg = lg[lg.split == "test"]
        cost = lg.groupby("model").agg(seconds=("seconds", "sum"), passes=("forward_passes", "sum")).reset_index()
        cost["sec_per_pass"] = cost.seconds / cost.passes
        cost.to_csv(TAB / "cost_test.csv", index=False, float_format="%.4f")
        summary["cost"] = cost.to_dict(orient="records")
        summary["environment"] = {k: logs[0][k] for k in ("python", "torch", "transformers", "device", "platform")}

    # ============================================================ figures
    fig_per_prompt(per_prompt)
    fig_stability(stab)
    fig_two_analyst(ta)
    fig_variance(var_df)
    fig_label_shares(per_prompt)
    summary["per_prompt_f1_range"] = {m: float(per_prompt[per_prompt.model == m].macro_f1.max()
                                               - per_prompt[per_prompt.model == m].macro_f1.min()) for m in MODELS}
    json.dump(summary, open(ROOT / "results/summary.json", "w"), indent=2, default=float)
    write_latex_tables(per_prompt, methods_df, stab, var_df)
    print(json.dumps(summary, indent=2, default=float))


# ------------------------------------------------------------------ figures
def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_prompt(pp):
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    x = np.arange(len(ALL_IDS))
    off = {m: (j - (len(MODELS) - 1) / 2) * 0.22 for j, m in enumerate(MODELS)}
    for m in MODELS:
        d = pp[pp.model == m].set_index("prompt").loc[ALL_IDS]
        ax.plot(x + off[m], d.macro_f1 * 100, "o", ms=4.5, color=COL[m], label=NICE[m],
                markeredgecolor="white", markeredgewidth=0.6)
    for b in (4.5, 8.5):
        ax.axvline(b, color=INK2, lw=0.6, ls=":")
    for xx, lab in ((2, "paraphrase (P0–P4)"), (6.5, "format (F1–F4)"), (10.5, "underspecified (M1–M5)")):
        ax.text(xx, 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2, fontsize=7)
    ax.set_xticks(x, ALL_IDS)
    ax.set_ylabel("Macro-F1 on test (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    _save(fig, "fig_per_prompt_f1")


def fig_stability(st):
    fams = ["para", "fmt", "min", "all14"]
    lab = {"para": "paraphrase", "fmt": "format", "min": "underspec.", "all14": "all 14"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    w = 0.8 / len(MODELS)
    for ax, cond in zip(axes, ("raw", "calibrated")):
        for j, m in enumerate(MODELS):
            d = st[(st.model == m) & (st.condition == cond)].set_index("family").loc[fams]
            xs = np.arange(len(fams)) + (j - (len(MODELS) - 1) / 2) * w
            ax.bar(xs, d.flip_rate * 100, width=w - 0.03, color=COL[m], label=NICE[m])
            ax.errorbar(xs, d.flip_rate * 100, yerr=[(d.flip_rate - d.flip_lo) * 100, (d.flip_hi - d.flip_rate) * 100],
                        fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.set_xticks(np.arange(len(fams)), [lab[f] for f in fams])
        ax.set_title("raw probabilities" if cond == "raw" else "after contextual calibration", fontsize=8, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Abstracts with ≥1 flipped\nconclusion (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=7)
    _save(fig, "fig_flip_rate_by_family")


def fig_two_analyst(ta):
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for m in MODELS:
        for cond, ls, mk in (("raw", "-", "o"), ("calibrated", "--", "s")):
            d = ta[(ta.model == m) & (ta.condition == cond)]
            ax.plot(d.k, d.agreement_mean * 100, ls=ls, marker=mk, ms=4, lw=1.6, color=COL[m],
                    label=f"{NICE[m]} ({'raw' if cond == 'raw' else 'CC'})")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Prompts per analyst (k)")
    ax.set_ylabel("Same conclusion reached (%)")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    _save(fig, "fig_two_analyst_agreement")


def fig_variance(v):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    cats = [("within prompt (repeated sampling)", 0.7, "one prompt,\nT=0.7"),
            ("within prompt (repeated sampling)", 1.0, "one prompt,\nT=1.0"),
            ("between paraphrases (greedy)", 0, "5 paraphr.,\ngreedy"),
            ("between paraphrases (sampled)", 1.0, "5 paraphr.,\nT=1.0")]
    w = 0.8 / len(MODELS)
    for j, m in enumerate(MODELS):
        vals = [v[(v.model == m) & (v.source == s) & (v.temperature == t)].disagreement.iloc[0] * 100 for s, t, _ in cats]
        ax.bar(np.arange(len(cats)) + (j - (len(MODELS) - 1) / 2) * w, vals, width=w - 0.03, color=COL[m], label=NICE[m])
    ax.set_xticks(np.arange(len(cats)), [c[2] for c in cats], fontsize=6.5)
    ax.set_ylabel("Pairwise disagreement (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6, loc="upper left")
    _save(fig, "fig_variance_sources")


def fig_label_shares(pp):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.8, 2.2), sharey=True, squeeze=False)
    axes = axes[0]
    shades = {"share_yes": "#2a78d6", "share_no": "#9ec5f4", "share_maybe": "#52514e"}
    names = {"share_yes": "yes", "share_no": "no", "share_maybe": "maybe"}
    for ax, m in zip(axes, MODELS):
        d = pp[pp.model == m].set_index("prompt").loc[ALL_IDS]
        bottom = np.zeros(len(ALL_IDS))
        for c in shades:
            ax.bar(np.arange(len(ALL_IDS)), d[c] * 100, bottom=bottom, color=shades[c], width=0.8,
                   edgecolor="white", linewidth=0.8, label=names[c])
            bottom += d[c].values * 100
        ax.set_xticks(np.arange(len(ALL_IDS)), ALL_IDS, rotation=90, fontsize=6)
        ax.set_title(NICE[m], fontsize=8, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Predicted conclusions (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=7)
    _save(fig, "fig_label_shares")


# ------------------------------------------------------------------ latex tables
def write_latex_tables(pp, md, st, vd):
    # main methods table
    lines = []
    for m in MODELS:
        d = md[md.model == m]
        for i, r in enumerate(d.itertuples()):
            sel = f" ({r.selected_prompt})" if r.selected_prompt else ""
            p = "" if r.method.startswith("B1") or r.method.startswith("MAJ") else \
                (f"{r.mcnemar_vs_B1_p:.3f}" if r.mcnemar_vs_B1_p >= 0.001 else "$<$0.001")
            lines.append(f"{NICE[m] if i == 0 else ''} & {r.method.split(' ')[0]}{sel} & {r.forward_passes_per_item} & "
                         f"{r.acc*100:.1f} [{r.acc_lo*100:.1f}, {r.acc_hi*100:.1f}] & "
                         f"{r.macro_f1*100:.1f} [{r.f1_lo*100:.1f}, {r.f1_hi*100:.1f}] & {r.pred_maybe_share*100:.1f} & {p} \\\\")
        lines.append("\\midrule")
    (TAB / "methods_test.tex").write_text("\n".join(lines[:-1]) + "\n")
    # stability table
    lines = []
    lab = {"para": "Paraphrase", "fmt": "Format", "min": "Underspecified", "all14": "All 14"}
    for m in MODELS:
        for i, fam in enumerate(["para", "fmt", "min", "all14"]):
            r = st[(st.model == m) & (st.condition == "raw") & (st.family == fam)].iloc[0]
            c = st[(st.model == m) & (st.condition == "calibrated") & (st.family == fam)].iloc[0]
            lines.append(f"{NICE[m] if i == 0 else ''} & {lab[fam]} & {r.K} & {r.sensitivity*100:.1f} & "
                         f"{r.flip_rate*100:.1f} [{r.flip_lo*100:.1f}, {r.flip_hi*100:.1f}] & {r.fleiss_kappa:.2f} & "
                         f"{r.acc_min*100:.1f}--{r.acc_max*100:.1f} & {r.f1_range*100:.1f} & "
                         f"{c.flip_rate*100:.1f} & {c.fleiss_kappa:.2f} \\\\")
        lines.append("\\midrule")
    (TAB / "stability_test.tex").write_text("\n".join(lines[:-1]) + "\n")


if __name__ == "__main__":
    main()
