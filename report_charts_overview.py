"""Eight descriptive overview figures from the frozen daily reproduction.

No download, outlier deletion, graph recalculation, forecast, or trade occurs.
Run: python report_charts_overview.py
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from archive_catalog import load_snapshot


ROOT = Path(__file__).resolve().parent
NAVY, TEAL, ORANGE = "#17324D", "#007F86", "#D57B27"
SELECTED = ("2020-03-20", "2022-10-21")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.edgecolor": "#B5C0C9", "axes.labelcolor": NAVY,
        "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.facecolor": "white",
    })


def decorate_time(axes, *, selected=True):
    for axis in np.atleast_1d(axes):
        if not axis.get_xlabel():
            axis.set_xlabel("Window end date (calendar year)")
        axis.grid(axis="y", color="#DCE3E8", linewidth=0.7, alpha=0.8)
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if selected:
            for date in SELECTED:
                axis.axvline(pd.Timestamp(date), color=ORANGE, linestyle="--", linewidth=0.8, alpha=0.7)


def footer(figure, text="Daily 60-observation windows; same cleaned sample; descriptive evidence only."):
    figure.text(0.01, 0.01, text, fontsize=9, color="#536978")


def save(figure, folder, filename):
    axes = [{"x": ax.get_xlabel(), "y": ax.get_ylabel(), "x_visible": ax.xaxis.get_visible(),
             "y_visible": ax.yaxis.get_visible(), "kind": "measurement"} for ax in figure.axes]
    for item in axes:
        if (item["x_visible"] and not item["x"]) or (item["y_visible"] and not item["y"]):
            raise ValueError(f"Missing measurement axis label in {filename}: {item}")
    (folder / (Path(filename).stem + ".axes.json")).write_text(json.dumps(axes, indent=2), encoding="utf-8")
    figure.savefig(folder / filename, dpi=160, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
    return filename


def entry(number, title, group, figure, equations, explanation, method, finding, limitations, sources, stats):
    return dict(id=f"G{number:02d}", title=title, group=group, figure=figure,
                equations=equations, explanation=explanation, method=method,
                finding=finding, limitations=limitations, sources=sources, stats=stats)


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_figures(features, prices, manifest, figures_dir, sources):
    dates = features["date"]
    total = len(features)
    graphs = []
    low = features.loc[features["lambda2"].idxmin()]
    high = features.loc[features["lambda2"].idxmax()]
    selected_rows = features.set_index("date").loc[pd.to_datetime(list(SELECTED))]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(dates, features["lambda2"], color=NAVY, linewidth=0.9)
    for position, label in ((low, "Minimum"), (high, "Maximum")):
        ax.scatter(position["date"], position["lambda2"], color=ORANGE, zorder=4)
        offset = (-185, 110) if label == "Minimum" else (-135, -35)
        ax.annotate(f"{label}: {position['date']:%Y-%m-%d}\n{position['lambda2']:.3f}",
                    (position["date"], position["lambda2"]), xytext=offset,
                    textcoords="offset points", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set(title="Raw algebraic connectivity over time", ylabel="Raw lambda2 (unitless)", xlabel="Window end date (calendar year)")
    decorate_time(ax)
    footer(fig)
    fig.subplots_adjust(bottom=0.17)
    graphs.append(entry(1, "Raw algebraic connectivity over time", "Daily graph overview",
        save(fig, figures_dir, "G01_raw_lambda2.png"),
        [r"W_{ij,t}=\max(C_{ij,t},0),\quad W_{ii,t}=0", r"L_t=\operatorname{diag}(W_t\mathbf{1})-W_t,\quad \lambda_{2,t}=\lambda_2(L_t)"],
        "Raw lambda2 is the second-smallest eigenvalue of the positive-correlation graph Laplacian. It measures connectivity on the graph's existing weight scale and depends on asset composition.",
        "Plot the reproduced lambda2 for each trailing 60-return window, labeled by its final date. Pearson correlations, clipping and cleaning use the established baseline.",
        f"Across {total:,} windows, lambda2 ranges from {low['lambda2']:.6f} on {low['date']:%Y-%m-%d} to {high['lambda2']:.6f} on {high['date']:%Y-%m-%d}; the median is {features['lambda2'].median():.6f}.",
        "Overlapping windows are dependent; extrema alone do not establish a market mechanism or prediction.", sources,
        {"windows": total, "minimum": low["lambda2"], "minimum_date": str(low["date"].date()),
         "maximum": high["lambda2"], "maximum_date": str(high["date"].date()), "median": features["lambda2"].median()}))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    axes[0].plot(dates, features["lambda2_scale_ratio"], color=TEAL, linewidth=0.9)
    axes[1].plot(dates, features["normalized_lambda2"], color=NAVY, linewidth=0.9)
    axes[0].set(title="Connectivity after two different scale adjustments", ylabel="Scale ratio q (unitless)")
    axes[1].set(ylabel="Normalized lambda2 (unitless)", xlabel="Window end date (calendar year)")
    decorate_time(axes)
    footer(fig, "Separate panels preserve each definition; no forecasting transformation is applied.")
    fig.subplots_adjust(hspace=0.28, bottom=0.12)
    graphs.append(entry(2, "Scale ratio and normalized connectivity", "Daily graph overview",
        save(fig, figures_dir, "G02_scaled_connectivity.png"),
        [r"q_t=\frac{\lambda_{2,t}}{n_t\overline{W}_t}", r"L_{\mathrm{norm},t}=I-D_t^{-1/2}W_tD_t^{-1/2}"],
        "q removes uniform multiplication of all edge weights, while the normalized Laplacian adjusts each edge using its endpoint degrees. Neither adjustment removes changing composition or correlation-estimation noise.",
        "Use the stored scale ratio and normalized lambda2 from the same cleaned window as raw lambda2. Normalized lambda2 is reported only for connected graphs with positive degrees.",
        f"q ranges from {features['lambda2_scale_ratio'].min():.6f} to {features['lambda2_scale_ratio'].max():.6f}, with median {features['lambda2_scale_ratio'].median():.6f}; normalized lambda2 ranges from {features['normalized_lambda2'].min():.6f} to {features['normalized_lambda2'].max():.6f}.",
        "These are different operators; their numerical values are not interchangeable risk scores.", sources,
        {"q_min": features["lambda2_scale_ratio"].min(), "q_max": features["lambda2_scale_ratio"].max(),
         "q_median": features["lambda2_scale_ratio"].median(), "normalized_min": features["normalized_lambda2"].min(),
         "normalized_max": features["normalized_lambda2"].max(), "normalized_available": int(features["normalized_lambda2"].notna().sum())}))

    delta = features["mean_positive_weight"] - features["mean_raw_correlation"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, features["mean_raw_correlation"], color=NAVY, linewidth=1, label="Mean raw Pearson rho (negative pairs retained)")
    ax.plot(dates, features["mean_positive_weight"], color=TEAL, linewidth=0.9, label="Mean clipped weight (zeros included)")
    ax.fill_between(dates, features["mean_raw_correlation"], features["mean_positive_weight"], color=ORANGE, alpha=0.2)
    ax.set(title="Negative clipping changes the dependence average", ylabel="Pair-average coefficient (unitless)", xlabel="Window end date (calendar year)")
    ax.legend(loc="upper right", fontsize=9)
    decorate_time(ax)
    footer(fig, "Each unordered pair is counted once; diagonal entries are excluded in both averages.")
    fig.subplots_adjust(bottom=0.17)
    graphs.append(entry(3, "Raw mean correlation and mean clipped weight", "Dependence and composition",
        save(fig, figures_dir, "G03_raw_vs_clipped_mean.png"),
        [r"\overline{\rho}_t=\frac{2}{n_t(n_t-1)}\sum_{i<j}C_{ij,t}",
         r"\overline{W}_t=\frac{2}{n_t(n_t-1)}\sum_{i<j}\max(C_{ij,t},0)"],
        "The raw average includes negative correlations, while the graph-weight average clips them to zero. Their difference is the average magnitude of discarded negative coefficients across all pairs.",
        "Read both averages from the reproduced table; each uses the identical retained assets and 60 observations. Count every unordered pair once, exclude diagonals and include zero graph weights.",
        f"The median raw average is {features['mean_raw_correlation'].median():.6f}, versus {features['mean_positive_weight'].median():.6f} for clipped weights; the median clipping increment is {delta.median():.6f} and its maximum is {delta.max():.6f}.",
        "Positive mean correlation does not mean every pair is positive; negative dependence is absent from the graph operator.", sources,
        {"raw_median": features["mean_raw_correlation"].median(), "clipped_median": features["mean_positive_weight"].median(),
         "clipping_increment_median": delta.median(), "clipping_increment_max": delta.max()}))

    daily_coverage = prices.notna().mean(axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(dates, features["num_stocks"], color=NAVY, linewidth=1)
    axes[0].axhline(prices.shape[1], color="#8698A5", linewidth=0.8, linestyle=":")
    axes[1].plot(dates, features["num_removed_stocks"], color=ORANGE, linewidth=1)
    axes[2].plot(prices.index, daily_coverage, color=TEAL, linewidth=1)
    axes[0].set(title="Available observations and retained graph assets", ylabel="Retained assets (count)")
    axes[1].set_ylabel("Removed assets (count)")
    axes[2].set(ylabel="Observed close coverage (%)", xlabel="Source close date (calendar year)")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    decorate_time(axes)
    footer(fig, "Daily price coverage and complete-window asset retention are different quantities.")
    fig.subplots_adjust(hspace=0.32, bottom=0.11)
    nonmissing = int(prices.notna().sum().sum())
    graphs.append(entry(4, "Changing asset count and observation coverage", "Dependence and composition",
        save(fig, figures_dir, "G04_retention_and_coverage.png"),
        [r"n_t=|\mathcal{S}_t|,\quad u_t=N-n_t", r"\mathrm{coverage}_s=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{P_{i,s}\ \mathrm{observed}\}"],
        "Retained assets must have a complete, nonconstant 60-return window. Daily close coverage measures only whether each price is present on a single source date.",
        "Plot reproduced retained/removed counts above source-price coverage across the 503-symbol panel. Missing observations stay missing and the entire original panel is one dataset.",
        f"Graphs retain {int(features['num_stocks'].min())}–{int(features['num_stocks'].max())} assets, with median {int(features['num_stocks'].median())}; {nonmissing:,} of {prices.size:,} price cells are observed ({nonmissing/prices.size:.2%}).",
        "Today's symbol universe viewed backward has selection bias; listing/history gaps are not corrected by the count plot.", sources,
        {"panel_symbols": prices.shape[1], "price_rows": len(prices), "nonmissing_price_cells": nonmissing,
         "missing_price_cells": int(prices.isna().sum().sum()), "overall_price_coverage": nonmissing/prices.size,
         "retained_min": int(features["num_stocks"].min()), "retained_max": int(features["num_stocks"].max()),
         "retained_median": features["num_stocks"].median(), "daily_coverage_min": daily_coverage.min()}))

    raw_corr = features["lambda2"].corr(features["mean_raw_correlation"])
    clipped_corr = features["lambda2"].corr(features["mean_positive_weight"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    for axis, column, color, xlabel, correlation in (
        (axes[0], "mean_raw_correlation", NAVY, "Mean raw Pearson rho", raw_corr),
        (axes[1], "mean_positive_weight", TEAL, "Mean clipped graph weight", clipped_corr),
    ):
        axis.scatter(features[column], features["lambda2"], s=10, color=color, alpha=0.28, edgecolors="none")
        axis.scatter(selected_rows[column], selected_rows["lambda2"], s=45, color=ORANGE, edgecolors="white", linewidth=0.8, zorder=4)
        axis.set_xlabel(xlabel + " (unitless)")
        axis.set_ylabel("Raw lambda2 (unitless)")
        axis.set_title(f"Pearson r = {correlation:.6f}", fontsize=12)
        axis.grid(alpha=0.2)
    fig.suptitle("Connectivity and average dependence are distinct summaries", fontsize=15, fontweight="bold", y=0.98)
    footer(fig, "Each dot is a trailing window; orange dots mark 2020-03-20 and 2022-10-21. No significance claim.")
    fig.subplots_adjust(top=0.83, bottom=0.17, wspace=0.1)
    graphs.append(entry(5, "Raw connectivity versus two dependence averages", "Dependence and composition",
        save(fig, figures_dir, "G05_connectivity_vs_dependence.png"),
        [r"r_{\lambda,\rho}=\operatorname{Corr}_t(\lambda_{2,t},\overline{\rho}_t)",
         r"r_{\lambda,W}=\operatorname{Corr}_t(\lambda_{2,t},\overline{W}_t)"],
        "Each point compares measurements from the same window rather than predicting a later outcome. Pearson association measures linear co-movement across these overlapping dates.",
        "Scatter reproduced lambda2 against raw and clipped pair averages separately, then calculate their pairwise Pearson correlations. The raw-average result is independently matched to the original notebook references.",
        f"The reproduced raw-average association is {raw_corr:.15f}; using the actual clipped graph weights gives {clipped_corr:.15f}. Neither correlation by itself proves incremental forecasting information.",
        "Adjacent windows share 59 returns, so ordinary independent-observation significance is inappropriate.", sources,
        {"lambda2_raw_mean_pearson": raw_corr, "lambda2_clipped_mean_pearson": clipped_corr, "paired_windows": total}))

    bound = features["num_stocks"] / (features["num_stocks"] - 1) * features["min_degree"]
    proximity = features["lambda2"] / bound
    within_one_percent = int((proximity >= 0.99).sum())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    axes[0].plot(dates, features["min_degree"], color=TEAL, linewidth=1.3, label="Minimum weighted degree")
    axes[0].plot(dates, features["lambda2"], color=NAVY, linewidth=0.8, label="Raw lambda2")
    axes[0].set(title="Connectivity frequently approaches the weak-node upper bound", ylabel="Degree / raw lambda2 (unitless)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].plot(dates, proximity, color=ORANGE, linewidth=0.9)
    axes[1].axhline(1, color=NAVY, linestyle=":", linewidth=1, label="Rayleigh upper bound")
    axes[1].set(ylabel="lambda2 / bound (unitless)", xlabel="Window end date (calendar year)", ylim=(0, 1.03))
    decorate_time(axes)
    footer(fig, "The upper bound follows from the graph operator; source quality and economic causes need separate investigation.")
    fig.subplots_adjust(hspace=0.28, bottom=0.12)
    graphs.append(entry(6, "Minimum degree and Rayleigh-bound proximity", "Localization and source quality",
        save(fig, figures_dir, "G06_min_degree_and_bound.png"),
        [r"d_{\min,t}=\min_i\sum_j W_{ij,t}", r"\lambda_{2,t}\leq B_t=\frac{n_t}{n_t-1}d_{\min,t},\quad p_t=\frac{\lambda_{2,t}}{B_t}"],
        "A contrast vector concentrating on the minimum-degree node gives the upper bound B. Near equality identifies proximity to that graph constraint, without establishing the economic cause of weak connectivity.",
        "Compute B from each window's retained count and minimum degree, then compare lambda2 and minimum degree over time and plot lambda2/B separately. The full baseline remains unchanged.",
        f"The median bound fraction is {proximity.median():.6f}; {within_one_percent:,}/{total:,} windows ({within_one_percent/total:.1%}) lie within 1% of the bound. Lambda2 and minimum degree have descriptive correlation {features['lambda2'].corr(features['min_degree']):.6f}.",
        "Near-bound behavior concerns this positive graph operator; it does not prove one stock caused a market event.", sources,
        {"bound_fraction_median": proximity.median(), "bound_fraction_min": proximity.min(),
         "bound_fraction_max": proximity.max(), "windows_within_one_percent": within_one_percent,
         "fraction_within_one_percent": within_one_percent/total,
         "lambda2_min_degree_correlation": features["lambda2"].corr(features["min_degree"])}))

    separated = features["fiedler_separated"].eq(True)
    ipr = features["fiedler_ipr"].where(separated)
    gap = features["lambda3_gap"].where(separated)
    localized = int((ipr > 0.9).sum())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.3), sharex=True)
    axes[0].plot(dates, ipr, color=NAVY, linewidth=0.8, label="Fiedler IPR")
    axes[0].plot(dates, 1/features["num_stocks"], color=TEAL, linewidth=1, linestyle="--", label="Uniform mass: 1/n")
    axes[0].set(title="Fiedler localization and numerical eigenvalue separation", ylabel="IPR / uniform mass (unitless)", ylim=(0, 1.03))
    axes[0].legend(fontsize=9, loc="lower left")
    axes[1].plot(dates, gap, color=TEAL, linewidth=0.8)
    axes[1].set(ylabel="lambda3 - lambda2 (unitless)", xlabel="Window end date (calendar year)")
    decorate_time(axes)
    footer(fig, "Only numerically separated vectors shown; a separated eigenspace is not an economically stable direction.")
    fig.subplots_adjust(hspace=0.28, bottom=0.12)
    graphs.append(entry(7, "Fiedler localization and eigengap", "Localization and source quality",
        save(fig, figures_dir, "G07_fiedler_localization.png"),
        [r"\mathrm{IPR}_t=\sum_i v_{i,t}^{4},\quad v_t^{\mathsf{T}}v_t=1", r"g_t=\lambda_{3,t}-\lambda_{2,t}"],
        "IPR approaches one when a unit Fiedler vector concentrates on very few assets; equal squared mass across n assets gives 1/n. The eigengap indicates numerical separation from the next eigenvalue.",
        "Read IPR and eigengap from the common-window analysis and show them only where its numerical separation flag is true. The graph uses no vector sign alignment or stability assumption.",
        f"Numerical separation holds in {int(separated.sum()):,}/{total:,} windows; median IPR is {ipr.median():.6f}, and {localized:,}/{int(separated.sum()):,} ({localized/int(separated.sum()):.1%}) exceed 0.9. The median eigengap is {gap.median():.6f}.",
        "Numerical separation does not establish resampling stability; the vector's sign is arbitrary and price artifacts may create localization.", sources,
        {"numerically_separated_windows": int(separated.sum()), "ipr_median": ipr.median(), "ipr_min": ipr.min(),
         "ipr_max": ipr.max(), "ipr_above_09_count": localized, "ipr_above_09_fraction": localized/int(separated.sum()),
         "eigengap_median": gap.median(), "eigengap_min": gap.min(), "eigengap_max": gap.max()}))

    coverage = prices.notna().mean(axis=0).sort_values(kind="stable").head(20)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    stacked = returns.stack(future_stack=True).dropna()
    finite_returns = stacked[np.isfinite(stacked)]
    flagged = finite_returns.loc[finite_returns.abs().nlargest(14).index]
    flags = [dict(date=str(date.date()), ticker=str(ticker), arithmetic_return=float(value),
                  previous_close=float(prices.loc[:date, ticker].iloc[-2]), close=float(prices.loc[date, ticker]))
             for (date, ticker), value in flagged.items()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.4), gridspec_kw={"width_ratios": [0.9, 1.35]})
    axes[0].barh(coverage.index, coverage.to_numpy(), color=TEAL, height=0.72)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Observed closes (%)", ylabel="Stock ticker (category)", title="Lowest 20 symbol coverages", xlim=(0, 1))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    labels = [f"{record['ticker']}  {record['date']}" for record in flags]
    axes[1].barh(labels, flagged.to_numpy(), color=[ORANGE if value < 0 else NAVY for value in flagged], height=0.72)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#8698A5", linewidth=0.8)
    axes[1].set(xlabel="Source arithmetic return (%)", ylabel="Stock / source date (category)", title="Largest 14 absolute return flags")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.grid(axis="x", color="#DCE3E8", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", labelsize=9)
    fig.suptitle("Original price coverage and source-review flags", fontsize=15, fontweight="bold", y=0.99)
    footer(fig, "Flags remain in the baseline. Listing history, adjustment and corporate actions have not been independently reconciled.")
    fig.subplots_adjust(top=0.89, bottom=0.12, wspace=0.66)
    graphs.append(entry(8, "Original price-quality coverage and return flags", "Localization and source quality",
        save(fig, figures_dir, "G08_price_quality_flags.png"),
        [r"\mathrm{coverage}_i=\frac{\sum_s\mathbf{1}\{P_{i,s}\ \mathrm{observed}\}}{T}", r"r_{i,s}=\frac{P_{i,s}}{P_{i,s-1}}-1"],
        "Low coverage can reflect listing dates or source-history limitations, while unusually large arithmetic returns flag observations for source review. A flag is not proof of a bad price or permission to remove it.",
        "Rank the lowest 20 observed-close fractions over all 2,703 source dates and the largest 14 absolute finite daily returns computed without filling. Preserve every flagged observation in the established baseline.",
        f"The lowest coverage is {coverage.index[0]} at {coverage.iloc[0]:.2%}; the largest absolute return flag is {flags[0]['ticker']} on {flags[0]['date']} at {flags[0]['arithmetic_return']:+.2%}. The panel has {int(prices.isna().sum().sum()):,} missing price cells.",
        "Original auto-adjustment and corporate-action handling remain unverified; no external event explanation or outlier deletion is asserted.", sources,
        {"lowest20_coverage": coverage.to_dict(), "largest14_absolute_return_flags": flags,
         "missing_price_cells": int(prices.isna().sum().sum()), "finite_return_cells": len(finite_returns),
         "nonfinite_nonmissing_return_cells": int((~np.isfinite(stacked)).sum()), "observations_deleted": 0,
         "original_adjustment_setting": manifest["request"]["adjustment"]}))
    return graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/findings")
    parser.add_argument("--render-only", action="store_true", help="Refresh figure labels without rewriting saved numerical metadata")
    args = parser.parse_args()
    style()
    feature_path = ROOT / "output/reproduction/features.csv"
    summary_path = ROOT / "output/reproduction/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    features = pd.read_csv(feature_path, parse_dates=["date"], float_precision="round_trip")
    if len(features) != 2643 or not features["date"].is_unique or not features["date"].is_monotonic_increasing:
        raise ValueError("The overview requires the complete ordered 2,643-window reproduction.")
    prices, manifest = load_snapshot(summary["snapshot_directory"])
    if manifest["sha256"] != summary["snapshot_price_sha256"]:
        raise ValueError("Frozen prices do not match the reproduction's recorded checksum.")
    figures_dir = args.output_dir.resolve() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sources = ["output/reproduction/features.csv", "output/reproduction/summary.json", "spectral_core.py",
               str(Path(summary["snapshot_directory"]).relative_to(ROOT) / "manifest.json").replace("\\", "/")]
    graphs = make_figures(features, prices, manifest, figures_dir, sources)
    if args.render_only:
        print(f"Refreshed {len(graphs)} figures; saved numerical metadata unchanged")
        return
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset="Original 503-symbol daily equity research panel; one frozen snapshot",
                  price_sha256=manifest["sha256"], graphs=graphs)
    (args.output_dir / "overview.json").write_text(
        json.dumps(_finite_json(result), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(graphs)} overview figures and {args.output_dir / 'overview.json'}")


if __name__ == "__main__":
    main()
