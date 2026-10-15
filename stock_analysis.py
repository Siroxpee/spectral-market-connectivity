"""Inspect local stock data or explicitly create descriptive spectral outputs.

Examples:
  python stock_analysis.py inspect --snapshot PATH
  python stock_analysis.py plot --metrics results/window_metrics.csv \
      --metadata data/dataset.json --output-dir NEW_DIRECTORY
  python stock_analysis.py analyze --snapshot PATH --output-dir NEW_DIRECTORY \
      --metrics results/window_metrics.csv --metadata data/dataset.json --date 2020-03-20
  python stock_analysis.py analyze --snapshot PATH --output-dir NEW_DIRECTORY --recompute

No acquisition, forecast, output-directory creation or analysis occurs on import.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_core import (
    analyze_window,
    build_positive_graph,
    clean_window,
    rolling_features,
)
from stock_data import arithmetic_returns, file_sha256, load_stock_prices


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_metrics(
    path, metadata, prices=None, price_metadata=None, window_size=60, step_size=1
):
    """Check a saved derivative's source/settings/hash and exact scheduled dates.

    This binds a previously calculated table; it does not recompute every
    eigenvalue. Requested selected windows are independently recalculated.
    """
    if (
        price_metadata is not None
        and metadata.get("price_sha256") != price_metadata["price_sha256"]
    ):
        raise ValueError("Saved metrics belong to a different price snapshot.")
    if (
        metadata.get("window_size") != window_size
        or metadata.get("step_size") != step_size
    ):
        raise ValueError("Saved metrics use different window/step settings.")
    if metadata.get("metrics_sha256") != file_sha256(path):
        raise ValueError("Saved metric-file checksum mismatch.")
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {
        "date",
        "lambda2",
        "num_stocks",
        "mean_raw_correlation",
        "mean_positive_weight",
        "min_degree",
        "fiedler_ipr",
        "fiedler_separated",
        "status",
    }
    if not required <= set(frame):
        raise ValueError("Saved metrics lack required baseline columns.")
    frame["date"] = pd.to_datetime(frame.date, format="ISO8601", errors="raise")
    dates = pd.DatetimeIndex(frame.date)
    if (
        dates.hasnans
        or dates.tz is not None
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Saved metric dates must be unique increasing daily source labels."
        )
    if prices is not None and not dates.equals(prices.index[window_size::step_size]):
        raise ValueError("Saved metric dates do not match the native rolling schedule.")
    if metadata.get("metrics_rows") != len(frame):
        raise ValueError("Saved metric row count mismatch.")
    counts = pd.to_numeric(frame.num_stocks, errors="raise")
    maximum = (
        len(prices.columns)
        if prices is not None
        else metadata.get("symbol_columns", metadata.get("symbol_count"))
    )
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, np.integer))
        or maximum < 1
    ):
        raise ValueError(
            "Supply a positive symbol_columns or symbol_count in metric metadata."
        )
    if (
        counts.isna().any()
        or not counts.between(0, maximum).all()
        or (counts % 1 != 0).any()
    ):
        raise ValueError("Saved retained-stock counts are invalid.")
    return add_bound_columns(frame)


def add_bound_columns(frame):
    """Algebraic summaries of existing metrics; no graph recomputation."""
    frame = frame.copy()
    bound = (frame.num_stocks / (frame.num_stocks - 1) * frame.min_degree).where(
        frame.num_stocks > 1
    )
    frame["weak_singleton_bound"] = bound
    frame["lambda2_over_singleton_bound"] = (frame.lambda2 / bound).where(bound > 0)
    return frame


def diagnose_window(returns, date, window_size=60, *, sensitivity=False):
    """Return (portable diagnostic dict, per-stock derived node table)."""
    stamp = pd.Timestamp(date)
    if stamp not in returns.index:
        raise ValueError("Selected date is not an observed source session.")
    end = returns.index.get_loc(stamp)
    if end + 1 < window_size:
        raise ValueError(
            "Selected date has insufficient preceding return observations."
        )
    window = returns.iloc[end - window_size + 1 : end + 1]
    baseline = analyze_window(window, include_fiedler=True)
    vector = baseline.pop("fiedler_vector", None)
    clean = clean_window(window)
    nodes = pd.DataFrame(
        {"symbol": clean.columns, "zero_returns": (clean == 0).sum().to_numpy()}
    )
    if clean.shape[1] >= 2:
        weights, _ = build_positive_graph(clean.corr())
        nodes["degree"] = weights.sum(axis=1).to_numpy()
    else:
        nodes["degree"] = np.nan
    nodes["fiedler_vector"] = (
        vector.reindex(clean.columns).to_numpy() if vector is not None else np.nan
    )
    nodes["fiedler_mass"] = nodes.fiedler_vector**2
    result = {
        "date": stamp.isoformat(),
        "window_start": window.index[0].isoformat(),
        "return_observations": len(window),
        "retained_symbols": list(clean.columns),
        "removed_symbols": [x for x in returns.columns if x not in clean],
        "baseline": serial(baseline),
        "fiedler_policy": "Only numerically separated connected modes; sign arbitrary, no economic stability claim.",
    }
    if sensitivity and clean.shape[1] >= 3:
        removed = baseline["min_degree_ticker"]
        result["remove_weakest_node_sensitivity"] = {
            "removed_symbol": removed,
            "label": "ALTERNATIVE composition, not a corrected baseline or forecast",
            "measurements": serial(analyze_window(window.drop(columns=[removed]))),
        }
    return result, nodes


def run_reference_experiment(
    clean, kind, date_number=0, repetitions=None, seed=20261008, block_length=6
):
    """Return every Gaussian/null or joint circular-block replicate, including failures.

    The seed is SeedSequence([seed, 14 or 15, date_number, replicate]), matching
    the retained controls. Defaults are 80 Gaussian or 120 block samples.
    Inputs are one already-cleaned fixed stock universe. Block length must
    divide its observation count; no sampled asset is silently removed.
    These descriptive reference ranges are not formal confidence intervals.
    """
    if kind not in ("gaussian", "block"):
        raise ValueError("Reference kind must be gaussian or block.")
    if not isinstance(clean, pd.DataFrame) or clean.shape[0] < 2 or clean.shape[1] < 2:
        raise ValueError(
            "Reference input needs at least two observations and two stocks."
        )
    if clean_window(clean).columns.tolist() != clean.columns.tolist():
        raise ValueError(
            "Reference input must already have a fixed finite, nonconstant stock universe."
        )
    repetitions = (
        (80 if kind == "gaussian" else 120) if repetitions is None else repetitions
    )
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, (int, np.integer))
        or repetitions < 1
    ):
        raise ValueError("Replicate count must be a positive integer.")
    if (
        isinstance(date_number, bool)
        or not isinstance(date_number, (int, np.integer))
        or date_number < 0
    ):
        raise ValueError("date_number must be a nonnegative seed index.")
    observations, stocks = clean.shape
    if kind == "block" and (
        isinstance(block_length, bool)
        or not isinstance(block_length, (int, np.integer))
        or block_length < 1
        or observations % block_length
    ):
        raise ValueError(
            "A positive block length must divide the observation count exactly."
        )
    metrics = ["lambda2", "lambda2_scale_ratio", "normalized_lambda2", "fiedler_ipr"]
    date = (
        str(clean.index[-1].date())
        if isinstance(clean.index, pd.DatetimeIndex)
        else "sample_" + str(date_number)
    )
    records = []
    for replicate in range(repetitions):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, 14 if kind == "gaussian" else 15, date_number, replicate]
            )
        )
        if kind == "gaussian":
            sampled = pd.DataFrame(
                rng.normal(0, 0.01, size=(observations, stocks)), columns=clean.columns
            )
        else:
            starts = rng.integers(0, observations, size=observations // block_length)
            indexes = (
                (starts[:, None] + np.arange(block_length)) % observations
            ).ravel()
            sampled = clean.iloc[indexes].reset_index(drop=True)
        row = {
            "date": date,
            "replicate": replicate,
            "n": stocks,
            "T": observations,
            "kind": kind,
            "status": "error",
            "reason": "",
        }
        try:
            retained = clean_window(sampled)
            if retained.columns.tolist() != clean.columns.tolist():
                removed = sorted(set(clean.columns) - set(retained.columns))
                raise ValueError(
                    "Fixed universe invalid: constant/nonfinite sampled assets "
                    + str(removed)
                )
            measured = analyze_window(retained)
            row.update({metric: measured[metric] for metric in metrics})
            row.update(
                status=measured["status"],
                num_components=measured["num_components"],
                fiedler_separated=measured["fiedler_separated"],
                mean_positive_weight=measured["mean_positive_weight"],
            )
        except Exception as error:
            row.update({metric: np.nan for metric in metrics})
            row["reason"] = f"{type(error).__name__}: {error}"
        records.append(row)
    return pd.DataFrame(records)


def plot_metrics(frame, destination):
    """Raw connectivity and pair means on two separate measured axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    for ax, field, label, color in zip(
        axes,
        ["lambda2", "mean_raw_correlation"],
        ["Raw lambda2 (dimensionless)", "Mean pair correlation\n(dimensionless)"],
        ["#17324D", "#007F86"],
    ):
        ax.plot(frame.date, frame[field], color=color, lw=1)
        ax.set(xlabel="Window end date (calendar year)", ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].lines[0].set_label("Signed raw pair mean")
    axes[1].plot(
        frame.date,
        frame.mean_positive_weight,
        color="#D57B27",
        lw=1,
        label="Clipped weight mean, including zeros",
    )
    axes[1].legend(fontsize=9)
    fig.suptitle("Trailing stock correlation graphs: descriptive measurements")
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(frame, destination):
    """Compare weakest degree, singleton-bound fraction and separated-mode IPR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = add_bound_columns(frame)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    fields = ["lambda2", "lambda2_over_singleton_bound", "fiedler_ipr"]
    labels = [
        "Raw lambda2 and minimum degree\n(dimensionless)",
        "lambda2 / singleton bound\n(dimensionless fraction)",
        "Separated Fiedler IPR\n(dimensionless)",
    ]
    for ax, field, label in zip(axes, fields, labels):
        values = (
            frame[field].where(frame.fiedler_separated.eq(True))
            if field == "fiedler_ipr"
            else frame[field]
        )
        ax.plot(
            frame.date,
            values,
            color="#17324D",
            lw=1,
            label="Raw lambda2" if field == "lambda2" else None,
        )
        ax.set(xlabel="Window end date (calendar year)", ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot(
        frame.date,
        frame.min_degree,
        color="#D57B27",
        lw=1,
        ls="--",
        label="Minimum weighted degree",
    )
    axes[0].legend(fontsize=9)
    axes[1].axhline(1, color="#D57B27", ls="--", lw=1)
    fig.suptitle("Graph bottleneck and localization diagnostics; descriptive only")
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nodes(nodes, date, destination):
    """Weakest degrees and largest separated-mode masses on categorical axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    low = nodes.nsmallest(12, "degree")
    mass = nodes.nlargest(12, "fiedler_mass").dropna(subset=["fiedler_mass"])
    axes[0].bar(low.symbol, low.degree, color="#007F86")
    axes[0].set(
        xlabel="Retained stock symbol (categorical)",
        ylabel="Weighted degree (dimensionless)",
        title="Smallest weighted degrees",
    )
    axes[1].bar(mass.symbol, mass.fiedler_mass, color="#D57B27")
    axes[1].set(
        xlabel="Retained stock symbol (categorical)",
        ylabel="Squared Fiedler entry (fraction)",
        title="Largest Fiedler masses; sign arbitrary",
        ylim=(0, 1),
    )
    if mass.empty:
        axes[1].text(
            0.5,
            0.5,
            "Separated Fiedler mode unavailable",
            ha="center",
            transform=axes[1].transAxes,
        )
    for ax in axes:
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle("Window ending " + str(pd.Timestamp(date).date()))
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze(args):
    snapshot = Path(args.snapshot).resolve()
    out = Path(args.output_dir).resolve()
    frozen = (
        snapshot
        if snapshot.is_dir()
        else snapshot.parent
        if (snapshot.parent / "manifest.json").exists()
        else None
    )
    if frozen is not None and (out == frozen or frozen in out.parents):
        raise ValueError("Output must be outside the frozen snapshot directory.")
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Choose a new or empty output directory; existing results are preserved."
        )
    prices, price_metadata = load_stock_prices(snapshot, args.expected_sha256)
    returns = arithmetic_returns(prices)
    if args.metrics:
        if not args.metadata:
            raise ValueError(
                "--metrics requires --metadata binding source/settings/CSV hashes."
            )
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        frame = load_metrics(
            args.metrics, metadata, prices, price_metadata, args.window, args.step
        )
        mode = (
            "Reused source/settings/hash-bound derivative; not all windows recomputed."
        )
    else:
        frame = add_bound_columns(
            rolling_features(returns, window_size=args.window, step_size=args.step)
        )
        mode = "Every scheduled window recalculated from supplied stock prices."
    diagnoses = {}
    for date in args.date:
        report, nodes = diagnose_window(
            returns, date, args.window, sensitivity=args.sensitivity
        )
        row = frame.loc[frame.date.eq(pd.Timestamp(date))]
        if not row.empty:
            for field in ["lambda2", "mean_raw_correlation"]:
                np.testing.assert_allclose(
                    row.iloc[0][field],
                    report["baseline"][field],
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True,
                )
            if int(row.iloc[0].num_stocks) != report["baseline"]["num_stocks"]:
                raise ValueError(
                    "Selected-window retained count differs from saved metrics."
                )
        diagnoses[str(pd.Timestamp(date).date())] = (report, nodes)
    out.mkdir(parents=True, exist_ok=True)
    metric_path = out / "window_metrics.csv"
    frame.to_csv(metric_path, index=False, float_format="%.17g")
    plot_metrics(frame, out / "rolling_metrics.png")
    plot_diagnostics(frame, out / "diagnostics.png")
    for date, (report, nodes) in diagnoses.items():
        nodes.to_csv(
            out / ("nodes_" + date + ".csv"), index=False, float_format="%.17g"
        )
        plot_nodes(nodes, date, out / ("nodes_" + date + ".png"))
    if args.controls:
        for date_number, date in enumerate(diagnoses):
            clean = clean_window(returns.loc[: pd.Timestamp(date)].tail(args.window))
            for kind, count in [
                ("gaussian", args.null_repetitions),
                ("block", args.block_repetitions),
            ]:
                samples = run_reference_experiment(
                    clean, kind, date_number, count, args.seed, args.block_length
                )
                samples.to_csv(
                    out / (kind + "_" + date + ".csv"),
                    index=False,
                    float_format="%.17g",
                )
    summary = {
        **price_metadata,
        "window_size": args.window,
        "step_size": args.step,
        "metrics_rows": len(frame),
        "metrics_sha256": file_sha256(metric_path),
        "analysis_mode": mode,
        "lambda2_mean_raw_correlation": float(
            frame.lambda2.corr(frame.mean_raw_correlation)
        ),
        "selected_windows": {date: report for date, (report, _) in diagnoses.items()},
        "claim": "Descriptive stock co-movement, not forecasts, causality or trading advantage.",
        "reference_controls": {
            "enabled": args.controls,
            "seed": args.seed,
            "date_seed_order": list(diagnoses),
            "gaussian_repetitions": args.null_repetitions,
            "block_repetitions": args.block_repetitions,
            "block_length": args.block_length,
        },
        "figure_axes": "History x: window end date, calendar year; y: dimensionless graph measurements. Node x: categorical symbols; y: weighted degree or Fiedler squared mass.",
    }
    (out / "summary.json").write_text(
        json.dumps(serial(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "window_rows": len(frame),
                "mode": mode,
                "selected_dates": list(diagnoses),
            },
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plot = commands.add_parser(
        "plot",
        help="Plot the hash-bound published metrics without loading raw stock data.",
    )
    plot.add_argument("--metrics", required=True)
    plot.add_argument("--metadata", required=True)
    plot.add_argument("--output-dir", required=True)
    for action in ["inspect", "analyze"]:
        command = commands.add_parser(action)
        command.add_argument("--snapshot", required=True)
        command.add_argument("--expected-sha256")
        if action == "analyze":
            command.add_argument("--output-dir", required=True)
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--metrics")
            source.add_argument("--recompute", action="store_true")
            command.add_argument("--metadata")
            command.add_argument("--window", type=int, default=60)
            command.add_argument("--step", type=int, default=1)
            command.add_argument("--date", action="append", default=[])
            command.add_argument("--sensitivity", action="store_true")
            command.add_argument(
                "--controls",
                action="store_true",
                help="Explicitly run Gaussian/block controls for requested dates; may be slow.",
            )
            command.add_argument("--null-repetitions", type=int, default=80)
            command.add_argument("--block-repetitions", type=int, default=120)
            command.add_argument("--seed", type=int, default=20261008)
            command.add_argument("--block-length", type=int, default=6)
    args = parser.parse_args(argv)
    if args.action == "inspect":
        print(
            json.dumps(
                load_stock_prices(args.snapshot, args.expected_sha256)[1], indent=2
            )
        )
    elif args.action == "plot":
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        frame = load_metrics(
            args.metrics,
            metadata,
            window_size=metadata["window_size"],
            step_size=metadata["step_size"],
        )
        out = Path(args.output_dir)
        if out.exists() and any(out.iterdir()):
            raise ValueError("Choose a new or empty output directory.")
        out.mkdir(parents=True, exist_ok=True)
        plot_metrics(frame, out / "rolling_metrics.png")
        plot_diagnostics(frame, out / "diagnostics.png")
        print(
            "Saved two descriptive figures from checked metrics; raw price observations were not loaded."
        )
    else:
        if args.window < 2 or args.step < 1:
            parser.error("Window must be at least 2 and step at least 1.")
        if args.controls and not args.date:
            parser.error("--controls requires explicit --date selections.")
        analyze(args)


if __name__ == "__main__":
    main()
