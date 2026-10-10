"""Report figures + table comparing the initial (v1) and retrained (v2) model, both top-5 weight
averages, on the held-out Day 7 session, from the per-bout logits of evaluate_raw_session.py.

    python plot_day7_comparison.py --eval-dir eval_day7_object1

Writes to <eval-dir>/figures/:
    day7_ethogram.pdf/.png        ground truth vs predicted behaviour over time (SIPEC-style)
    day7_confusion.pdf/.png       row-normalised confusion matrices, Run 1 | Run 2
    day7_metrics.pdf/.png         a overall scores and b paired differences with bootstrap CIs,
                                  c per-class F1, d share of annotated time per behaviour
    day7_metrics_table.tex        booktabs table for the thesis
    day7_metrics.csv              every number in the table, unrounded

Bouts are resampled with replacement (paired: the same resample for both runs) for the CIs, and
the per-bout correctness of the two runs is compared with an exact McNemar test. Classification
uses the annotated bout boundaries, so these are bout-classification scores, not detection.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score  # noqa: E402

FPS = 30
CM = 1 / 2.54
TEXT_WIDTH = 15.9 * CM  # thesis: A4, 1 in margins

# label_map order (ids 0..8) and the names the thesis uses ("Locomotion" in the label files is
# reported as Anogenital sniffing, see Methods)
LABEL_NAMES = ["Air sniffing", "Approaching/Following", "Body touch", "Exploring", "Grooming",
               "Investigation", "Anogenital sniffing", "Rearing", "Tail investigation"]
# display order: conspecific-directed classes, then the focal rat's own actions (Methods)
SOCIAL = ["Approaching/Following", "Body touch", "Investigation", "Anogenital sniffing", "Tail investigation"]
SOLITARY = ["Air sniffing", "Exploring", "Grooming", "Rearing"]
ORDER = SOCIAL + SOLITARY
ORDER_IDS = [LABEL_NAMES.index(n) for n in ORDER]

# validated categorical slots 1-3 (dataviz reference palette, all-pairs CVD pass on white)
# labels follow the thesis Results chapter ("initial" / "retrained" model; the code's v1 / v2)
RUNS = [("v1_avg5", "Initial model", "#2a78d6"), ("v2_avg5", "Retrained model", "#eb6834")]
SUPPLEMENTARY = [("v2_ep158", "Retrained, epoch 158", "#1baf7a")]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, TRUTH_BG = "#e1e0d9", "#c3c2b7", "#d6d5cf"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def load_run(eval_dir, name):
    df = pd.read_csv(Path(eval_dir) / f"logits_{name}.csv").drop_duplicates("segment_id", keep="last")
    return df.set_index("segment_id")


# ── metrics ──────────────────────────────────────────────────────────────────

def metrics(y, yhat, w):
    present = np.unique(y)
    return {
        "accuracy": float(np.mean(y == yhat)),
        "macro_f1": float(f1_score(y, yhat, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yhat, labels=present, average="weighted", zero_division=0)),
        "time_weighted_accuracy": float(np.sum(w * (y == yhat)) / np.sum(w)),
        "cohen_kappa": float(cohen_kappa_score(y, yhat)),
    }


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted F1",
    "time_weighted_accuracy": "Time-weighted accuracy",
    "cohen_kappa": "Cohen's κ",
}


def paired_bootstrap(y, preds, w, n_boot, seed=0):
    """preds: {run: yhat}. Returns {run: {metric: array}} over bout resamples (same indices for
    every run, so differences are paired)."""
    rng = np.random.default_rng(seed)
    out = {r: {m: np.empty(n_boot) for m in METRIC_LABELS} for r in preds}
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for r, yhat in preds.items():
            for m, v in metrics(y[idx], yhat[idx], w[idx]).items():
                out[r][m][b] = v
    return out


def ci(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5])  # kappa is undefined on a one-class resample
    return float(lo), float(hi)


# ── figures ──────────────────────────────────────────────────────────────────

def style_axis(ax, grid_x=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_x:
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def save(fig, out_dir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_ethogram(truth, runs, out_dir):
    t0 = truth["start_frame"].to_numpy() / FPS
    dur = (truth["end_frame"] - truth["start_frame"]).to_numpy() / FPS
    row_of = {lid: i for i, lid in enumerate(ORDER_IDS)}
    y_true = truth["true_label_id"].to_numpy()
    x_min, x_max = np.floor(t0.min()), np.ceil((t0 + dur).max())

    panels = [("Ground truth", None, INK2)] + [(lbl, df, col) for _, lbl, col, df in runs]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(TEXT_WIDTH, 4.2 * CM * len(panels)),
                             gridspec_kw={"hspace": 0.45})
    for ax, (title, df, color) in zip(axes, panels):
        if df is None:
            for s, d, lid in zip(t0, dur, y_true):
                ax.broken_barh([(s, d)], (row_of[lid] - 0.32, 0.64), facecolor=color, edgecolor="none")
            ax.set_title(f"{title} ({len(truth)} bouts, {dur.sum():.0f} s)", fontsize=8)
        else:
            y_pred = df.loc[truth.index, "pred_label_id"].to_numpy()
            for s, d, lid in zip(t0, dur, y_true):  # truth behind, as context
                ax.broken_barh([(s, d)], (row_of[lid] - 0.40, 0.80), facecolor=TRUTH_BG, edgecolor="none")
            for s, d, lid in zip(t0, dur, y_pred):
                ax.broken_barh([(s, d)], (row_of[lid] - 0.20, 0.40), facecolor=color, edgecolor="none")
            acc = np.mean(y_true == y_pred)
            ax.set_title(f"{title} prediction (accuracy {acc:.2f}; grey = ground truth)", fontsize=8)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER)
        ax.set_ylim(len(ORDER) - 0.5, -0.5)
        ax.axhline(len(SOCIAL) - 0.5, color=AXIS, linewidth=0.5)
        ax.set_xlim(x_min, x_max)
        style_axis(ax)
        ax.tick_params(axis="y", length=0)
        # group brackets on the right
        for (lo, hi), name in [((0, len(SOCIAL) - 1), "conspecific"),
                               ((len(SOCIAL), len(ORDER) - 1), "own actions")]:
            ax.annotate("", xy=(1.006, 1 - (lo - 0.3 + 0.5) / len(ORDER)), xycoords="axes fraction",
                        xytext=(1.006, 1 - (hi + 0.3 + 0.5) / len(ORDER)),
                        arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6))
            ax.text(1.015, 1 - ((lo + hi) / 2 + 0.5) / len(ORDER), name, transform=ax.transAxes,
                    rotation=90, va="center", ha="left", fontsize=6, color=MUTED)
    axes[-1].set_xlabel("Time in the Day 7 recording (s)")
    save(fig, out_dir, "day7_ethogram")


def fig_confusion(truth, runs, out_dir):
    y_true = truth["true_label_id"].to_numpy()
    support = {lid: int(np.sum(y_true == lid)) for lid in ORDER_IDS}
    fig, axes = plt.subplots(1, len(runs), figsize=(TEXT_WIDTH, 8.6 * CM), gridspec_kw={"wspace": 0.08})
    for k, (ax, (_, lbl, color, df)) in enumerate(zip(axes, runs)):
        y_pred = df.loc[truth.index, "pred_label_id"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=ORDER_IDS)
        frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(frac, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=90)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([f"{n} ({support[lid]})" for n, lid in zip(ORDER, ORDER_IDS)] if k == 0 else [])
        for v in (len(SOCIAL) - 0.5,):
            ax.axhline(v, color=INK2, linewidth=0.5)
            ax.axvline(v, color=INK2, linewidth=0.5)
        ax.set_title(f"{lbl} (accuracy {np.mean(y_true == y_pred):.2f})")
        ax.set_xlabel("Predicted")
        if k == 0:
            ax.set_ylabel("Ground truth (n bouts)")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Fraction of ground-truth bouts (row)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, out_dir, "day7_confusion")


def fig_quantitative(point, boots, truth, runs, mcnemar, out_dir):
    """a: overall scores, b: paired difference Run 2 - Run 1, c: per-class F1, d: time budget."""
    y_true = truth["true_label_id"].to_numpy()
    dur = (truth["end_frame"] - truth["start_frame"]).to_numpy().astype(float)
    (n0, l0, c0, df0), (n1, l1, c1, df1) = runs
    names = list(METRIC_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 13.0 * CM),
                             gridspec_kw={"width_ratios": [1, 0.8], "height_ratios": [5, 9],
                                          "wspace": 0.12, "hspace": 0.38})
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # a  overall scores with 95% bootstrap CI
    for i, m in enumerate(names):
        for k, (name, lbl, color, _) in enumerate(runs):
            lo, hi = ci(boots[name][m])
            y = i + (k - 0.5) * 0.3
            ax_a.plot([lo, hi], [y, y], color=color, linewidth=1.2, solid_capstyle="round")
            ax_a.plot(point[name][m], y, "o", ms=4.2, color=color, mec="white", mew=0.8, zorder=3)
    ax_a.set_yticks(range(len(names)))
    ax_a.set_yticklabels([METRIC_LABELS[m] for m in names])
    ax_a.set_ylim(len(names) - 0.5, -0.5)
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("Score, 95% bootstrap CI")
    ax_a.set_title("a  Overall scores")

    # b  paired difference
    for i, m in enumerate(names):
        d = boots[n1][m] - boots[n0][m]
        lo, hi = ci(d)
        delta = point[n1][m] - point[n0][m]
        ax_b.plot([lo, hi], [i, i], color=INK2, linewidth=1.2, solid_capstyle="round")
        ax_b.plot(delta, i, "o", ms=4.2, color=INK, mec="white", mew=0.8, zorder=3)
        ax_b.text(hi, i - 0.28, f"{delta:+.3f}", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax_b.axvline(0, color=AXIS, linewidth=0.8)
    lim = max(0.05, max(abs(v) for m in names for v in ci(boots[n1][m] - boots[n0][m]))) * 1.1
    ax_b.set_xlim(-lim, lim)
    ax_b.set_ylim(len(names) - 0.5, -0.5)
    ax_b.set_yticks(range(len(names)))
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("Retrained − initial, paired 95% CI")
    ax_b.set_title(f"b  Difference (McNemar p = {mcnemar['p']:.2f})")

    # c  per-class F1
    support = [int(np.sum(y_true == lid)) for lid in ORDER_IDS]
    f1s = {lbl: f1_score(y_true, df.loc[truth.index, "pred_label_id"].to_numpy(), labels=ORDER_IDS,
                         average=None, zero_division=0) for _, lbl, _, df in runs}
    for i in range(len(ORDER)):
        # small vertical offset so equal scores (e.g. both 0) stay visible as two dots
        ax_c.plot([f1s[l0][i], f1s[l1][i]], [i - 0.12, i + 0.12], color=AXIS, linewidth=1.0, zorder=1)
        ax_c.plot(f1s[l0][i], i - 0.12, "o", ms=4.2, color=c0, mec="white", mew=0.8, zorder=3)
        ax_c.plot(f1s[l1][i], i + 0.12, "o", ms=4.2, color=c1, mec="white", mew=0.8, zorder=3)
    ax_c.set_yticks(range(len(ORDER)))
    ax_c.set_yticklabels([f"{n} ({s})" for n, s in zip(ORDER, support)])
    for tick, s in zip(ax_c.get_yticklabels(), support):
        if s < 5:
            tick.set_color(MUTED)
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_xlabel("F1 (n bouts in brackets; grey: n < 5)")
    ax_c.set_title("c  Per class")

    # d  time budget
    series = [("Ground truth", INK2, y_true)] + [
        (lbl, col, df.loc[truth.index, "pred_label_id"].to_numpy()) for _, lbl, col, df in runs]
    shares = {lbl: np.array([dur[y == lid].sum() / dur.sum() for lid in ORDER_IDS]) for lbl, _, y in series}
    tvd = {lbl: 0.5 * np.abs(shares[lbl] - shares["Ground truth"]).sum() for lbl, _, _ in series[1:]}
    h = 0.24
    for k, (lbl, col, _) in enumerate(series):
        ax_d.barh(np.arange(len(ORDER)) + (k - 1) * (h + 0.03), shares[lbl] * 100, height=h,
                  color=col, edgecolor="none")
    ax_d.set_yticks(range(len(ORDER)))
    ax_d.set_yticklabels([])
    ax_d.set_xlabel("Share of annotated time (%)")
    ax_d.set_title("d  Time budget")
    ax_d.text(0.98, 0.02, "\n".join(f"{lbl}: {tvd[lbl] * 100:.0f}% of time misallocated" for lbl in tvd),
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)

    for ax in (ax_c, ax_d):
        ax.set_ylim(len(ORDER) - 0.5, -0.5)
        ax.axhline(len(SOCIAL) - 0.5, color=AXIS, linewidth=0.5)
    for ax in (ax_a, ax_b, ax_c, ax_d):
        style_axis(ax, grid_x=True)
        ax.tick_params(axis="y", length=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=INK2, label="Ground truth (d)")] + [
        plt.Line2D([], [], marker="o", ls="", ms=4.2, color=c, mec="white", mew=0.8, label=l)
        for _, l, c, _ in runs]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    save(fig, out_dir, "day7_metrics")
    return tvd


# ── table ────────────────────────────────────────────────────────────────────

def write_table(point, boots, specs, mcnemar, out_dir):
    cols = list(METRIC_LABELS)
    lines = [r"\small", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(METRIC_LABELS[c].replace("κ", r"$\kappa$").replace("Time-weighted accuracy", "Time-weighted acc.") for c in cols) + r" \\",
             r"\midrule"]
    for name, lbl in specs:
        cells = []
        for c in cols:
            lo, hi = ci(boots[name][c])
            cells.append(f"{point[name][c]:.3f} {{\\scriptsize [{lo:.2f}, {hi:.2f}]}}")
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")
    (n0, l0), (n1, l1) = specs[0], specs[1]
    lines.append(r"\midrule")
    cells = []
    for c in cols:
        d = boots[n1][c] - boots[n0][c]
        lo, hi = ci(d)
        cells.append(f"{point[n1][c] - point[n0][c]:+.3f} {{\\scriptsize [{lo:+.2f}, {hi:+.2f}]}}")
    lines.append(f"$\\Delta$ {l1} $-$ {l0} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"% exact McNemar on per-bout correctness: {l0} only correct {mcnemar['b']}, "
              f"{l1} only correct {mcnemar['c']}, p = {mcnemar['p']:.3g}"]
    (out_dir / "day7_metrics_table.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="eval_day7_object1")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = eval_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    all_specs = RUNS + [s for s in SUPPLEMENTARY if (eval_dir / f"logits_{s[0]}.csv").exists()]
    dfs = {name: load_run(eval_dir, name) for name, _, _ in all_specs}
    common = sorted(set.intersection(*(set(d.index) for d in dfs.values())),
                    key=lambda s: dfs[RUNS[0][0]].loc[s, "start_frame"])
    truth = dfs[RUNS[0][0]].loc[common, ["start_frame", "end_frame", "true_label_id"]]
    for name, d in dfs.items():
        if not (d.loc[common, "true_label_id"] == truth["true_label_id"]).all():
            raise ValueError(f"{name}: ground truth differs from {RUNS[0][0]}")
    print(f"{len(common)} bouts scored by every model")

    y = truth["true_label_id"].to_numpy()
    w = (truth["end_frame"] - truth["start_frame"]).to_numpy().astype(float)
    preds = {name: dfs[name].loc[common, "pred_label_id"].to_numpy() for name, _, _ in all_specs}
    point = {name: metrics(y, p, w) for name, p in preds.items()}
    boots = paired_bootstrap(y, preds, w, args.n_boot)

    (n0, l0, _), (n1, l1, _) = RUNS
    ok0, ok1 = preds[n0] == y, preds[n1] == y
    b, c = int(np.sum(ok0 & ~ok1)), int(np.sum(~ok0 & ok1))
    mcnemar = {"b": b, "c": c, "p": float(binomtest(c, b + c, 0.5).pvalue) if b + c else 1.0}

    runs = [(name, lbl, col, dfs[name]) for name, lbl, col in RUNS]
    fig_ethogram(truth, runs, out_dir)
    fig_confusion(truth, runs, out_dir)
    tvd = fig_quantitative(point, boots, truth, runs, mcnemar, out_dir)
    write_table(point, boots, [(name, lbl) for name, lbl, _ in all_specs], mcnemar, out_dir)

    rows = []
    for name, lbl, _ in all_specs:
        for m in METRIC_LABELS:
            lo, hi = ci(boots[name][m])
            rows.append({"model": lbl, "metric": m, "value": point[name][m], "ci_low": lo, "ci_high": hi})
    for m in METRIC_LABELS:
        d = boots[n1][m] - boots[n0][m]
        lo, hi = ci(d)
        rows.append({"model": f"{l1} - {l0}", "metric": m, "value": point[n1][m] - point[n0][m],
                     "ci_low": lo, "ci_high": hi, "p_boot_le_0": float(np.mean(d <= 0))})
    for lbl, v in tvd.items():
        rows.append({"model": lbl, "metric": "misallocated_time_share", "value": v})
    rows.append({"model": f"{l1} vs {l0}", "metric": "mcnemar_exact_p", "value": mcnemar["p"],
                 "note": f"{l0} only correct {b}, {l1} only correct {c}"})
    pd.DataFrame(rows).to_csv(out_dir / "day7_metrics.csv", index=False)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
