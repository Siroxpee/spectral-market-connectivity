"""Offline structural figures G09-G15 for the findings report.

The recovered baseline is unchanged. Node deletion is a sensitivity analysis;
Gaussian nulls and circular block resamples are explicitly separate experiments.
Every replicate is retained, including failures and undefined diagnostics.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from archive_catalog import load_snapshot
from spectral_core import analyze_window, analyze_correlation, build_positive_graph, clean_window


NAVY, TEAL, ORANGE = "#17324D", "#007F86", "#D57B27"
DATES = ("2020-03-20", "2022-10-21")
SEED = 20261008
METRICS = ("lambda2", "lambda2_scale_ratio", "normalized_lambda2", "fiedler_ipr")
SNAPSHOT = "data/original_notebook/720726a5d713c24f9cae262ef7ede3d619cf0d329b4ed5382941de4cfb64811f"


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.labelcolor": NAVY, "text.color": NAVY,
                         "xtick.color": NAVY, "ytick.color": NAVY,
                         "axes.edgecolor": "#CCD6DC", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def finish(fig, path, *, footer=None):
    axes = []
    for ax in fig.axes:
        color_scale = hasattr(ax, "_colorbar")
        item = {"kind": "color_scale" if color_scale else "measurement", "x": ax.get_xlabel(), "y": ax.get_ylabel(),
                "x_visible": ax.xaxis.get_visible(), "y_visible": ax.yaxis.get_visible()}
        if color_scale:
            if not (item["x"] or item["y"]):
                raise ValueError("Missing color-scale label: " + str(path))
        elif (item["x_visible"] and not item["x"]) or (item["y_visible"] and not item["y"]):
            raise ValueError(f"Missing measurement axis label in {path}: {item}")
        axes.append(item)
    path.with_suffix(".axes.json").write_text(json.dumps(axes, indent=2), encoding="utf-8")
    if footer:
        fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=9, color=NAVY)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def graph(identifier, title, figure, equations, explanation, method, finding, limitations, sources, stats, group="Structure and robustness"):
    return {"id": identifier, "title": title, "group": group, "figure": figure,
            "equations": equations, "explanation": explanation, "method": method,
            "finding": finding, "limitations": limitations,
            "sources": [str(Path(source).resolve()) for source in sources], "stats": stats}


def selected_windows(prices):
    returns = prices.pct_change(fill_method=None).iloc[1:]
    selected = {}
    for date in DATES:
        end = returns.index.get_loc(pd.Timestamp(date)) + 1
        raw = returns.iloc[end - 60:end]
        cleaned = clean_window(raw)
        baseline = analyze_window(cleaned, include_fiedler=True)
        correlation = cleaned.corr()
        weights, laplacian = build_positive_graph(correlation)
        weak = baseline["min_degree_ticker"]
        reduced = analyze_window(cleaned.drop(columns=weak))
        vector = baseline["fiedler_vector"]
        if vector is None:
            raise AssertionError(f"Selected window {date} lacks a separated Fiedler vector.")
        residual = float(np.linalg.norm(laplacian @ vector.to_numpy() - baseline["lambda2"] * vector.to_numpy()))
        if residual > 1e-8 or abs(vector.sum()) > 1e-8:
            raise AssertionError("Fiedler residual or zero-sum identity failed.")
        reference_path = Path(f"output/reproduction/selected_windows/{date}/diagnostics.json")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))["baseline"]
        for key in METRICS + ("min_degree", "mean_positive_weight"):
            np.testing.assert_allclose(baseline[key], reference[key], rtol=1e-9, atol=1e-11)
        selected[date] = {"returns": cleaned, "correlation": correlation, "weights": weights,
                          "baseline": baseline, "reduced": reduced, "weak": weak,
                          "degrees": weights.sum(axis=1), "vector": vector,
                          "residual": residual, "reference": reference_path}
    assert selected[DATES[0]]["returns"].shape == (60, 483)
    assert selected[DATES[1]]["returns"].shape == (60, 493)
    return selected


def ferg_chart(prices, selected, figures, analysis, source, *, write_tables=True):
    date = DATES[0]
    window = selected[date]["returns"]
    first_position = prices.index.get_loc(window.index[0]) - 1
    last_position = prices.index.get_loc(window.index[-1]) + 1
    raw_prices = prices.iloc[first_position:last_position]["FERG"]
    ferg_returns = window["FERG"]
    zeros = ferg_returns.eq(0)
    assert int(zeros.sum()) == 34
    frame = pd.DataFrame({"raw_saved_close": raw_prices,
                          "price_index": 100 * raw_prices / raw_prices.iloc[0],
                          "arithmetic_return": ferg_returns.reindex(raw_prices.index)})
    frame["in_60_return_window"] = frame.index.isin(window.index)
    frame["exact_zero_return"] = frame["in_60_return_window"] & frame["arithmetic_return"].eq(0)
    table = analysis / "G09_FERG_raw_window.csv"
    if write_tables:
        frame.to_csv(table, float_format="%.17g")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [1.2, 1]})
    axes[0].plot(frame.index, frame["price_index"], color=NAVY, lw=1.7)
    axes[0].scatter(ferg_returns.index[zeros], frame.loc[ferg_returns.index[zeros], "price_index"], color=ORANGE, s=21, label="Dates with exactly zero return", zorder=3)
    axes[0].set(ylabel="Indexed close (index points; start = 100)", xlabel="Close observation date (calendar date)", title="FERG: flat observations inside the low-connectivity window")
    axes[0].legend(loc="lower left", frameon=False)
    axes[0].text(.99, .96, "34 of 60 returns are exactly zero (56.7%)", transform=axes[0].transAxes, ha="right", va="top", fontsize=11,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    axes[1].vlines(ferg_returns.index, 0, 100 * ferg_returns, color=TEAL, lw=2)
    axes[1].scatter(ferg_returns.index[zeros], np.zeros(zeros.sum()), color=ORANGE, s=21, zorder=3)
    axes[1].axhline(0, color="#AAB8C2", lw=.8)
    axes[1].set(ylabel="Arithmetic return (%)", xlabel="Return observation date (2019-12-24 to 2020-03-20)")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in axes:
        ax.grid(axis="y", color="#E7EDF0", lw=.6)
    filename = "G09_ferg_flat_returns.png"
    finish(fig, figures / filename)
    return graph("G09", "FERG observations behind the March 2020 low", filename,
                 [r"r_t=P_t/P_{t-1}-1", r"I_t=100P_t/P_{t_0}"],
                 "The orange markers identify saved observations whose close-to-close arithmetic return is exactly zero. The price curve includes the preceding close so every one of the 60 returns can be checked.",
                 "Use the untouched frozen FERG closes and the exact 60-return window ending March 20, 2020; do not fill missing values. Index the first preceding close to 100 for display and retain raw closes in the companion CSV.",
                 f"FERG has 34/60 exact zero returns (56.7%) and sample return standard deviation {ferg_returns.std(ddof=1):.5f}; it survives the baseline cleaning threshold.",
                 "Flat saved observations require source/listing/corporate-action review; their economic cause is unverified.",
                 [source, table], {"zero_returns": 34, "return_observations": 60, "zero_fraction": 34 / 60,
                                  "sample_std_ddof1": float(ferg_returns.std(ddof=1)), "preceding_close_date": frame.index[0].isoformat(),
                                  "window_start": window.index[0].isoformat(), "window_end": window.index[-1].isoformat()})


def localization_chart(selected, figures, analysis, source, *, write_tables=True):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), layout="constrained")
    stats = {}
    for row, date in enumerate(DATES):
        item = selected[date]
        degree, mass = item["degrees"], item["vector"] ** 2
        weak, baseline = item["weak"], item["baseline"]
        low, concentrated = degree.nsmallest(10), mass.nlargest(10)
        table = pd.DataFrame({"degree": degree, "fiedler_squared_mass": mass})
        if write_tables:
            table.to_csv(analysis / f"G10_{date}_node_diagnostics.csv", float_format="%.17g")
        axes[row, 0].barh(low.index[::-1], low.values[::-1], color=[ORANGE if name == weak else NAVY for name in low.index[::-1]])
        axes[row, 0].set(title=f"{date}: ten smallest weighted degrees", xlabel=r"Weighted degree $d_i=\sum_j W_{ij}$")
        axes[row, 1].barh(concentrated.index[::-1], concentrated.values[::-1], color=[ORANGE if name == weak else TEAL for name in concentrated.index[::-1]])
        axes[row, 1].set_xscale("log")
        axes[row, 1].set(title=f"{date}: ten largest Fiedler masses", xlabel=r"Squared vector mass $v_i^2$ (log scale)", xlim=(float(concentrated.min()) / 3, 2))
        bound = len(degree) / (len(degree) - 1) * degree.min()
        fraction = baseline["lambda2"] / bound
        axes[row, 0].set_ylabel("Stock ticker (category)")
        axes[row, 1].set_ylabel("Stock ticker (category)")
        axes[row, 0].set_xlabel(r"Weighted degree $d_i=\sum_j W_{ij}$ (unitless)" +
                               f"\n{weak}: degree {degree[weak]:.4f}; single-node bound {bound:.4f}" +
                               f"\nActual lambda2 = {baseline['lambda2']:.4f} ({100 * fraction:.2f}% of bound)", fontsize=9)
        axes[row, 1].set_xlabel(r"Squared vector mass $v_i^2$ (unitless; log scale)" +
                               f"\n{weak} holds {100 * mass[weak]:.2f}% of unit-vector mass" +
                               f"\nIPR = {baseline['fiedler_ipr']:.5f}", fontsize=9)
        stats[date] = {"weak_node": weak, "min_degree": float(degree.min()), "lambda2": baseline["lambda2"],
                       "singleton_rayleigh_bound": float(bound), "lambda2_fraction_of_bound": float(fraction),
                       "weak_node_squared_mass": float(mass[weak]), "ipr": baseline["fiedler_ipr"],
                       "lambda3_gap": baseline["lambda3_gap"], "fiedler_residual_norm": item["residual"]}
    filename = "G10_weak_nodes_fiedler_mass.png"
    finish(fig, figures / filename)
    return graph("G10", "Both extrema are dominated by a weak singleton", filename,
                 [r"d_i=\sum_j W_{ij}", r"\lambda_2\leq\frac{n}{n-1}\min_i d_i", r"\mathrm{IPR}=\sum_i v_i^4,\quad\sum_i v_i^2=1"],
                 "The left panels locate the least-connected stocks; the right panels show where the second eigenvector puts its mass. A single-node contrast gives the displayed upper bound on algebraic connectivity.",
                 "Rebuild each selected graph on its original common cleaned sample, obtain the numerically separated unit Fiedler vector, and rank degrees and squared vector entries separately. Verify eigenvector residuals and the zero-sum identity.",
                 "FERG holds 99.79% of Fiedler mass in March 2020 and BMY 97.44% in October 2022; raw lambda2 reaches 99.94% and 99.90% of the respective singleton bounds.",
                 "Numerical localization describes these saved windows; it does not establish a stable economic partition or explain its cause.",
                 [source, selected[DATES[0]]["reference"], selected[DATES[1]]["reference"]], stats)


def removal_chart(selected, figures, source):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), layout="constrained")
    fig.suptitle("Node-removal sensitivity: compare relative changes within each metric", fontsize=12)
    names = [("lambda2", r"Raw $\lambda_2$"), ("lambda2_scale_ratio", r"Scale ratio $q$"), ("normalized_lambda2", r"Normalized $\lambda_2$")]
    positions = np.arange(2)
    stats = {}
    for ax, (metric, title) in zip(axes, names):
        original = [selected[date]["baseline"][metric] for date in DATES]
        reduced = [selected[date]["reduced"][metric] for date in DATES]
        ax.bar(positions - .18, original, width=.35, color=NAVY, label="Original baseline")
        ax.bar(positions + .18, reduced, width=.35, color=ORANGE, label="Remove weakest node")
        if metric in ("lambda2", "lambda2_scale_ratio"):
            ax.set_yscale("log")
            ax.set_ylim(min(original + reduced) * .6, max(original + reduced) * 2.1)
            ax.set_ylabel(title + " (unitless; log scale)")
        else:
            ax.set_ylim(0, 1.08)
            ax.set_ylabel(title + " (unitless)")
        for x, values in ((positions - .18, original), (positions + .18, reduced)):
            for xpos, value in zip(x, values):
                ax.annotate(f"{value:.5f}" if metric != "lambda2" else f"{value:.3f}", (xpos, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, rotation=35)
        ax.set(title=title, xlabel="Window end date (category)", xticks=positions, xticklabels=["2020-03-20\nFERG removed", "2022-10-21\nBMY removed"])
        ax.grid(axis="y", color="#E7EDF0", lw=.6)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    for date in DATES:
        item = selected[date]
        stats[date] = {"removed_node": item["weak"], "original": {key: item["baseline"][key] for key in METRICS},
                       "sensitivity": {key: item["reduced"][key] for key in METRICS},
                       "relative_percent_changes": {key: 100 * (item["reduced"][key] / item["baseline"][key] - 1) for key in METRICS},
                       "original_n": item["baseline"]["num_stocks"], "sensitivity_n": item["reduced"]["num_stocks"]}
    filename = "G11_node_removal_sensitivity.png"
    finish(fig, figures / filename)
    return graph("G11", "Node removal changes raw connectivity far more in percentage terms", filename,
                 [r"q=\lambda_2/(n\bar{w})", r"\mathcal{L}=I-D^{-1/2}WD^{-1/2}"],
                 "These paired bars compare the unchanged original graph with a separate graph that omits its weakest stock. The logarithmic raw and q axes make the small 2020 values visible.",
                 "Remove FERG or BMY only from the selected already-cleaned window, recompute all measurements on that reduced graph, and leave the full daily baseline untouched.",
                 "Removing FERG increases raw lambda2 from 1.300 to 4.431 (+241.0%) and q by 240.3%, while normalized lambda2 changes by 0.0083%. Removing BMY changes raw lambda2 by 1.25% and normalized lambda2 by 0.0592%.",
                 "Removing an asset changes composition, node count and edge scale; this sensitivity is not a corrected baseline or a forecast.",
                 [source, selected[DATES[0]]["reference"], selected[DATES[1]]["reference"]], stats)


def correlation_chart(selected, figures, analysis, source, *, write_tables=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("negative_to_positive", [ORANGE, "white", TEAL])
    stats = {}
    for ax, date in zip(axes, DATES):
        item = selected[date]
        order = [item["weak"]] + sorted(symbol for symbol in item["correlation"].columns if symbol != item["weak"])
        correlation = item["correlation"].loc[order, order]
        table = analysis / f"G12_{date}_full_correlation.csv.gz"
        if write_tables:
            correlation.to_csv(table, compression="gzip", float_format="%.17g")
        image = ax.imshow(correlation.to_numpy(), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ticks = [0, 100, 200, 300, 400]
        labels = [f"0: {item['weak']}", "100", "200", "300", "400"]
        ax.set(xticks=ticks, yticks=ticks, xticklabels=labels, yticklabels=labels,
               xlabel="Column stock position (index)", ylabel="Row stock position (index)",
               title=f"{date}: n = {len(order)}, T = 60\n{item['returns'].index[0].date()} to {item['returns'].index[-1].date()}")
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        stats[date] = {"n": len(order), "T": 60, "weak_node_first": item["weak"],
                       "ordering": "weak node first, then remaining ticker symbols alphabetical", "matrix_csv": str(table.resolve()),
                       "mean_raw_correlation": item["baseline"]["mean_raw_correlation"],
                       "mean_positive_weight": item["baseline"]["mean_positive_weight"]}
    fig.colorbar(image, ax=axes, shrink=.76, label=r"Pearson correlation $\rho$ (unitless)", ticks=[-1, -.5, 0, .5, 1])
    filename = "G12_selected_correlation_matrices.png"
    finish(fig, figures / filename)
    return graph("G12", "The low raw eigenvalue coexists with stronger average correlation", filename,
                 [r"C_{ij}=\mathrm{Corr}(r_i,r_j)", r"\bar{\rho}=\frac{2}{n(n-1)}\sum_{i<j} C_{ij}"],
                 "These are the entire retained correlation matrices, with the same color scale from minus one to one. Positions identify the weakest ticker first and all remaining tickers alphabetically, rather than an inferred sector grouping.",
                 "Compute Pearson correlation from the exact complete 60-return samples for each selected date and display every retained pair. Save both ordered matrices so the chart can be audited.",
                 "Mean raw correlation is 0.64195 in the March 2020 low and 0.52050 in the October 2022 high; mean clipped weights are 0.64385 and 0.52056. Thus the low raw lambda2 is not evidence of lower overall average dependence.",
                 "The universes differ (483 versus 493 assets), and estimates use only 60 observations; these heatmaps do not identify economic mechanisms.",
                 [source] + [stats[date]["matrix_csv"] for date in DATES], stats)


def toy_chart(figures, analysis, *, write_tables=True):
    levels = np.array([.05, .1, .2, .4, .6, .8])
    rows = []
    for c in levels:
        matrix = np.full((8, 8), c)
        np.fill_diagonal(matrix, 1)
        result = analyze_correlation(pd.DataFrame(matrix))
        np.testing.assert_allclose(result["lambda2"], 8 * c, atol=1e-12)
        np.testing.assert_allclose(result["lambda2_scale_ratio"], 1, atol=1e-12)
        rows.append({"uniform_edge_weight": float(c), "lambda2": result["lambda2"], "q": result["lambda2_scale_ratio"]})
    community = []
    for name, within, between in (("A", .8, .1), ("B", .4, .3)):
        matrix = np.full((4, 4), between)
        matrix[:2, :2] = within
        matrix[2:, 2:] = within
        np.fill_diagonal(matrix, 1)
        assert np.linalg.eigvalsh(matrix).min() > 0
        result = analyze_correlation(pd.DataFrame(matrix))
        np.testing.assert_allclose(result["mean_positive_weight"], 1 / 3, atol=1e-12)
        np.testing.assert_allclose(result["lambda2"], 4 * between, atol=1e-12)
        np.testing.assert_allclose(result["lambda2_scale_ratio"], .3 if name == "A" else .9, atol=1e-12)
        community.append({"name": name, "within": within, "between": between, "mean_weight": result["mean_positive_weight"], "lambda2": result["lambda2"], "q": result["lambda2_scale_ratio"]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axes[0].plot(levels, [row["lambda2"] for row in rows], "o-", color=NAVY, label=r"Raw $\lambda_2=8c$")
    axes[0].set(xlabel="Uniform edge weight c (unitless)", ylabel=r"Raw $\lambda_2$ (unitless)", title="SYNTHETIC: uniform complete graph (n = 8)", ylim=(0, 7))
    twin = axes[0].twinx()
    twin.plot(levels, [row["q"] for row in rows], "s--", color=TEAL, label="q = 1")
    twin.set(ylabel="Scale ratio q (unitless)", ylim=(0, 1.15))
    axes[0].legend(loc="upper left", frameon=False)
    twin.legend(loc="lower right", frameon=False)
    pos = np.arange(2)
    axes[1].bar(pos - .17, [row["lambda2"] for row in community], width=.34, color=NAVY, label=r"Raw $\lambda_2$")
    axes[1].bar(pos + .17, [row["q"] for row in community], width=.34, color=TEAL, label="q")
    for x, row in zip(pos, community):
        axes[1].text(x - .17, row["lambda2"] + .025, f"{row['lambda2']:.1f}", ha="center")
        axes[1].text(x + .17, row["q"] + .025, f"{row['q']:.1f}", ha="center")
    axes[1].set(xticks=pos, xticklabels=["A: within .8 / between .1", "B: within .4 / between .3"], xlabel="Synthetic graph (category)", ylabel="Metric value (unitless)",
                title="SYNTHETIC: same mean weight, different bottleneck", ylim=(0, 1.5))
    axes[1].text(.5, .98, "n = 4; mean weight = 1/3 in both graphs\n" + r"Here $a\geq b$, so $\lambda_2=4b$", transform=axes[1].transAxes, ha="center", va="top", fontsize=9)
    axes[1].legend(loc="center left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#E7EDF0", lw=.6)
    filename = "G13_synthetic_scale_and_bottlenecks.png"
    finish(fig, figures / filename)
    table = analysis / "G13_toy_identities.json"
    if write_tables:
        table.write_text(json.dumps({"uniform": rows, "community": community}, indent=2), encoding="utf-8")
    return graph("G13", "Synthetic controls separate uniform scaling from bottleneck structure", filename,
                 [r"\lambda_2(K_n,c)=nc,\quad q=1", r"n=4:\quad\bar{w}=(a+2b)/3,\quad\lambda_2=\min\{4b,2(a+b)\}", r"a\geq b\geq0\quad\Longrightarrow\quad\lambda_2=4b"],
                 "Uniformly raising every edge increases raw connectivity while q stays one. Two-community examples show that the same mean edge weight can still produce different algebraic connectivity.",
                 "Use valid positive-definite toy correlation matrices, apply the unchanged positive graph construction, and assert the exact spectra and q identities. These controls contain no market observations.",
                 "Both four-node graphs have mean weight 1/3: graph A gives lambda2 = 0.4 and q = 0.3; graph B gives lambda2 = 1.2 and q = 0.9. Every uniform complete-graph example gives lambda2 = 8c and q = 1.",
                 "Synthetic identities validate interpretation and arithmetic, not predictive usefulness in real markets.",
                 ["spectral_core.py", table], {"uniform": rows, "communities": community}, group="Synthetic interpretation controls")


def experiment_rows(selected, *, kind, repetitions, block_length=6):
    records = []
    for date_number, date in enumerate(DATES):
        window = selected[date]["returns"]
        n = window.shape[1]
        for replicate in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 14 if kind == "gaussian" else 15, date_number, replicate]))
            if kind == "gaussian":
                sampled = pd.DataFrame(rng.normal(0, .01, size=(60, n)), columns=window.columns)
            else:
                starts = rng.integers(0, 60, size=60 // block_length)
                indexes = ((starts[:, None] + np.arange(block_length)) % 60).ravel()
                assert len(indexes) == 60
                sampled = window.iloc[indexes].reset_index(drop=True)
            row = {"date": date, "replicate": replicate, "n": n, "T": 60, "kind": kind, "status": "error", "reason": ""}
            try:
                cleaned = clean_window(sampled)
                if cleaned.columns.tolist() != window.columns.tolist():
                    removed = sorted(set(window.columns) - set(cleaned.columns))
                    raise ValueError(f"Fixed universe invalid: constant/nonfinite sampled assets {removed}")
                result = analyze_window(cleaned)
                row.update({metric: result[metric] for metric in METRICS})
                row.update(status=result["status"], num_components=result["num_components"], fiedler_separated=result["fiedler_separated"], mean_positive_weight=result["mean_positive_weight"])
            except Exception as error:
                row.update({metric: np.nan for metric in METRICS})
                row["reason"] = f"{type(error).__name__}: {error}"
            records.append(row)
            if (replicate + 1) % 20 == 0:
                print(f"{kind} {date}: {replicate + 1}/{repetitions} replicates retained", flush=True)
    return pd.DataFrame(records)


def summaries(frame, selected):
    summary = {}
    for date in DATES:
        rows = frame.loc[frame["date"].eq(date)]
        metrics = {}
        for metric in METRICS:
            finite = rows.loc[np.isfinite(rows[metric]), metric].to_numpy()
            metrics[metric] = {"available_replicates": len(finite), "unavailable_replicates": len(rows) - len(finite),
                               "median": float(np.median(finite)) if len(finite) else None,
                               "central_90_percent": np.quantile(finite, [.05, .95]).tolist() if len(finite) else None,
                               "minimum": float(finite.min()) if len(finite) else None, "maximum": float(finite.max()) if len(finite) else None,
                               "observed": selected[date]["baseline"][metric]}
        errors = rows.loc[rows["status"].eq("error"), ["replicate", "reason"]].to_dict(orient="records")
        summary[date] = {"attempted_replicates": len(rows), "successful_computations": len(rows) - len(errors),
                         "statuses": dict(Counter(rows["status"])), "failed_replicates": errors,
                         "fiedler_unseparated_or_unavailable": int((~rows.get("fiedler_separated", pd.Series(False, index=rows.index)).fillna(False).astype(bool)).sum()),
                         "metrics": metrics}
    return summary


def distribution_panel(ax, frame, selected, metric, title, *, show_ranges=False):
    arrays = [frame.loc[frame["date"].eq(date) & np.isfinite(frame[metric]), metric].to_numpy() for date in DATES]
    positions = np.arange(1, 3)
    plotted = ax.boxplot(arrays, positions=positions, widths=.40, whis=(5, 95), patch_artist=True,
                         medianprops={"color": NAVY, "linewidth": 1.5}, boxprops={"facecolor": "#BFDDDF", "edgecolor": TEAL},
                         whiskerprops={"color": TEAL}, capprops={"color": TEAL},
                         flierprops={"marker": ".", "markersize": 4, "markerfacecolor": TEAL, "markeredgecolor": TEAL, "alpha": .5})
    observed = [selected[date]["baseline"][metric] for date in DATES]
    ax.scatter(positions, observed, marker="*", s=110, color=ORANGE, zorder=4, label="Observed original")
    maximum = max([float(array.max()) for array in arrays if len(array)] + observed)
    ax.set_ylim(0, max(maximum * 1.22, .02))
    for x, value in zip(positions, observed):
        offset = (18, -15) if metric == "fiedler_ipr" else (11, 3)
        ax.annotate(f"{value:.4g}", (x, value), xytext=offset, textcoords="offset points", fontsize=9, color=ORANGE,
                    va="top" if metric == "fiedler_ipr" else "bottom")
    if show_ranges:
        texts = []
        for date, array in zip(DATES, arrays):
            lo, hi = np.quantile(array, [.05, .95]) if len(array) else (np.nan, np.nan)
            texts.append(f"{date[:4]} central 90%: [{lo:.4g}, {hi:.4g}]")
        ax.text(.02, .96, "\n".join(texts), transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95})
    ax.set(title=title, xlabel="Reference window date (category)", xticks=positions,
           xticklabels=[f"{date}\nfinite = {len(array)}/{int(frame['date'].eq(date).sum())}" for date, array in zip(DATES, arrays)])
    ax.grid(axis="y", color="#E7EDF0", lw=.6)


def null_chart(selected, figures, analysis, source, repetitions, frame=None, *, write_tables=True):
    if frame is None:
        frame = experiment_rows(selected, kind="gaussian", repetitions=repetitions)
    csv = analysis / "G14_matched_gaussian_null_replicates.csv"
    if write_tables:
        frame.to_csv(csv, index=False, float_format="%.17g")
    stats = summaries(frame, selected)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), layout="constrained")
    distribution_panel(axes[0], frame, selected, "lambda2", r"SYNTHETIC null reference: raw $\lambda_2$", show_ranges=True)
    distribution_panel(axes[1], frame, selected, "lambda2_scale_ratio", r"SYNTHETIC null reference: scale ratio $q$", show_ranges=True)
    axes[0].set_ylabel(r"Raw $\lambda_2$ (unitless)")
    axes[1].set_ylabel("Scale ratio q (unitless)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    filename = "G14_matched_independent_gaussian_null.png"
    finish(fig, figures / filename)
    low_median = stats[DATES[0]]["metrics"]["lambda2_scale_ratio"]["median"]
    high_median = stats[DATES[1]]["metrics"]["lambda2_scale_ratio"]["median"]
    failures = sum(len(item["failed_replicates"]) for item in stats.values())
    return graph("G14", "Independent finite samples already create positive graph edges", filename,
                 [r"r_{ti}^{(b)}\sim\mathcal{N}(0,0.01^2)\ \mathrm{independently}", r"W_{ij}=\max(C_{ij},0),\quad W_{ii}=0"],
                 "The boxes summarize independent Gaussian samples after exactly the same Pearson correlation and positive clipping as the market graph. Orange stars show the original observed measurements on axes that retain all observed values.",
                 f"Generate {repetitions} independent 60-observation panels separately for n = 483 and n = 493, using explicit per-replicate seeds from {SEED}. Retain all outcomes; boxes show interquartile ranges, whiskers bound observations within the 5th and 95th percentiles, and printed ranges report the actual sample quantiles.",
                 f"Null median q is {low_median:.4f} for n=483 and {high_median:.4f} for n=493, versus observed q of 0.00418 and 0.43117; {failures} computations failed. Independent populations therefore do not imply zero clipped sample connectivity.",
                 "This strong Gaussian-independence reference matches n and T only; it is not a market model, p-value, causal explanation or financial edge test.",
                 [source, "spectral_core.py", csv], {"seed": SEED, "repetitions_per_dimension": repetitions, "reference": stats}, group="Exploratory uncertainty")


def resampling_chart(selected, figures, analysis, source, repetitions, frame=None, *, write_tables=True):
    if frame is None:
        frame = experiment_rows(selected, kind="block", repetitions=repetitions, block_length=6)
    csv = analysis / "G15_joint_circular_block_replicates.csv"
    if write_tables:
        frame.to_csv(csv, index=False, float_format="%.17g")
    stats = summaries(frame, selected)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    titles = [r"Joint circular block resamples: raw $\lambda_2$", r"Joint circular block resamples: scale ratio $q$",
              r"Joint circular block resamples: normalized $\lambda_2$", "Joint circular block resamples: Fiedler IPR"]
    for ax, metric, title in zip(axes.flat, METRICS, titles):
        distribution_panel(ax, frame, selected, metric, title, show_ranges=True)
        ax.set_ylabel({"lambda2": r"Raw $\lambda_2$", "lambda2_scale_ratio": "Scale ratio q", "normalized_lambda2": r"Normalized $\lambda_2$", "fiedler_ipr": "IPR"}[metric] + " (unitless)")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    filename = "G15_joint_circular_block_resampling.png"
    finish(fig, figures / filename)
    summary_text = []
    for date in DATES:
        item = stats[date]
        q_range = item["metrics"]["lambda2_scale_ratio"]["central_90_percent"]
        summary_text.append(f"{date}: {item['successful_computations']}/{repetitions} computations succeed, with q central 90% range [{q_range[0]:.4g}, {q_range[1]:.4g}]")
    return graph("G15", "Selected-window structure varies under joint block resampling", filename,
                 [r"s_k\sim\mathrm{Uniform}\{0,\ldots,59\},\quad t_{k,j}=(s_k+j)\ \mathrm{mod}\ 60", r"\ell=6,\quad k=1,\ldots,10,\quad j=0,\ldots,5"],
                 "Each resample uses the same selected row blocks for every stock, preserving their observed joint movements within six-row blocks. Orange stars mark the original windows; boxes and printed ranges show exploratory resampling variation.",
                 f"Draw {repetitions} circular block resamples per cleaned 60-row window, holding its original asset set fixed and using explicit seeds from {SEED}. Retain constant-asset failures, disconnected graphs and unavailable Fiedler diagnostics rather than silently changing the asset set or dropping outcomes.",
                 "; ".join(summary_text) + ".",
                 "These are exploratory central 90% resampling ranges, not formal confidence intervals; block length six and the chosen extrema are fixed retrospective choices.",
                 [source, "spectral_core.py", csv], {"seed": SEED, "repetitions_per_window": repetitions, "block_length": 6,
                                                      "joint_row_resampling": True, "fixed_asset_set": True, "resampling": stats}, group="Exploratory uncertainty")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=SNAPSHOT)
    parser.add_argument("--output-dir", default="output/findings")
    parser.add_argument("--null-replicates", type=int, default=80)
    parser.add_argument("--block-replicates", type=int, default=120)
    parser.add_argument("--render-only", action="store_true", help="Use saved replicates and preserve numerical metadata/tables")
    args = parser.parse_args()
    if args.null_replicates < 1 or args.block_replicates < 1:
        raise ValueError("Replicate counts must be positive.")
    root = Path(args.output_dir)
    figures, analysis = root / "figures", root / "analysis"
    figures.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    prices, manifest = load_snapshot(args.snapshot)
    source = Path(args.snapshot) / "prices.csv.gz"
    selected = selected_windows(prices)
    style()
    write_tables = not args.render_only
    graphs = [ferg_chart(prices, selected, figures, analysis, source, write_tables=write_tables), localization_chart(selected, figures, analysis, source, write_tables=write_tables),
              removal_chart(selected, figures, source), correlation_chart(selected, figures, analysis, source, write_tables=write_tables), toy_chart(figures, analysis, write_tables=write_tables)]
    print("G09-G13 saved and numerical identities verified", flush=True)
    gaussian = pd.read_csv(analysis / "G14_matched_gaussian_null_replicates.csv", float_precision="round_trip") if args.render_only else None
    blocks = pd.read_csv(analysis / "G15_joint_circular_block_replicates.csv", float_precision="round_trip") if args.render_only else None
    graphs.append(null_chart(selected, figures, analysis, source, args.null_replicates, gaussian, write_tables=write_tables))
    graphs.append(resampling_chart(selected, figures, analysis, source, args.block_replicates, blocks, write_tables=write_tables))
    if args.render_only:
        print("Refreshed G09-G15 from saved inputs/replicates; numerical tables and metadata unchanged", flush=True)
        return
    metadata = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_unchanged": True,
                "original_price_sha256": manifest["sha256"], "window_size": 60, "step_size": 1,
                "seed": SEED, "blas_threads": 1, "graphs": graphs}
    (root / "structure.json").write_text(json.dumps(json_ready(metadata), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(graphs)} structural figures and {root / 'structure.json'}", flush=True)


if __name__ == "__main__":
    main()
