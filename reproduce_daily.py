"""Reproduce the recovered daily snapshot, offline, with separate diagnostics.

Example:
    python reproduce_daily.py --snapshot data/original_notebook/REQUEST_KEY \
        --output-dir results/daily --reference-dir data/session_reference --workers 4

The baseline remains 60 arithmetic-return observations, step 1. Selected-window
node removal is a sensitivity calculation; it never changes features.csv.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


WINDOW_SIZE = 60
EXPECTED_WINDOWS = 2643
SELECTED_DATES = ("2020-03-20", "2022-10-21")
EIG_RTOL = EIG_ATOL = 1e-9
CORRELATION_ATOL = 1e-12
_RETURNS = None


def _import_numerics():
    # main sets BLAS limits before this call. Windows workers inherit those
    # environment settings and import numerical libraries in their initializer.
    global np, pd, analyze_window, clean_window, build_positive_graph, load_snapshot
    import numpy as np
    import pandas as pd
    from spectral_core import analyze_window, clean_window, build_positive_graph
    from archive_catalog import load_snapshot


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_returns(snapshot):
    _import_numerics()
    prices, manifest = load_snapshot(snapshot)
    if manifest["request"]["interval"] != "1d":
        raise ValueError("This reproduction requires a declared daily (1d) snapshot.")
    if not prices.index.equals(prices.index.normalize()):
        raise ValueError("Daily prices require midnight date labels, without intraday timestamps.")
    # No padding, forward fill, zero fill, or alternative missing-data policy.
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if len(returns) - WINDOW_SIZE + 1 != EXPECTED_WINDOWS:
        raise ValueError(
            f"The recovered run requires {EXPECTED_WINDOWS} full windows; "
            f"this snapshot would yield {len(returns) - WINDOW_SIZE + 1}."
        )
    return returns, manifest


def _initialize_worker(snapshot):
    global _RETURNS
    _RETURNS, _ = _load_returns(snapshot)


def _window_row(end):
    window = _RETURNS.iloc[end - WINDOW_SIZE:end]
    row = analyze_window(window)
    row["date"] = window.index[-1]
    return row


def calculate_features(returns, snapshot, workers):
    """Ordered processes each load frozen returns once, rather than per task."""
    global _RETURNS
    ends = range(WINDOW_SIZE, len(returns) + 1)
    rows = []

    def collect(iterator):
        for number, row in enumerate(iterator, 1):
            rows.append(row)
            if number % 100 == 0 or number == EXPECTED_WINDOWS:
                print(f"Computed {number}/{EXPECTED_WINDOWS} daily windows", flush=True)

    if workers == 1:
        _RETURNS = returns
        collect(map(_window_row, ends))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker,
            initargs=(str(snapshot),),
        ) as executor:
            collect(executor.map(_window_row, ends, chunksize=8))
    features = pd.DataFrame(rows)
    columns = ["date", "lambda2", "num_stocks"]
    features = features[columns + [c for c in features if c not in columns]]
    expected_dates = returns.index[WINDOW_SIZE - 1:]
    if len(features) != EXPECTED_WINDOWS or not pd.DatetimeIndex(features["date"]).equals(expected_dates):
        raise AssertionError("Rolling features lost, duplicated, or shifted a daily window.")
    return features


def _read_reference(path, columns):
    reference = pd.read_csv(path, float_precision="round_trip")
    if not set(columns).issubset(reference):
        raise ValueError(f"{path} requires columns {columns}.")
    reference = reference.loc[:, columns].copy()
    reference["date"] = pd.to_datetime(reference["date"], format="ISO8601")
    dates = pd.DatetimeIndex(reference["date"])
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"{path} requires unique, nonmissing, chronologically ordered dates.")
    if not dates.equals(dates.normalize()):
        raise ValueError(f"{path} contains intraday timestamps rather than daily labels.")
    for column in columns[1:]:
        reference[column] = pd.to_numeric(reference[column], errors="raise")
    return reference


def _numeric_comparison(actual, reference, *, rtol, atol):
    left, right = actual.to_numpy(dtype=float), reference.to_numpy(dtype=float)
    close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
    finite = np.isfinite(left) & np.isfinite(right)
    return {
        "matches": bool(close.all()), "mismatched_rows": int((~close).sum()),
        "max_absolute_difference": float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else None,
        "rtol": rtol, "atol": atol,
    }


def compare_references(features, references, reference_dir):
    """Outer one-to-one joins expose missing dates, rather than dropping them."""
    if references is None:
        return {"status": "not_requested"}
    result = {"status": "matched", "reference_directory": str(reference_dir), "files": {}}
    for filename, reference in references.items():
        joined = features.merge(reference, on="date", how="outer", validate="one_to_one",
                                suffixes=("", "_reference"), indicator=True)
        same_dates = bool((joined["_merge"] == "both").all())
        report = {
            "sha256": _hash_file(reference_dir / filename),
            "reference_rows": len(reference), "exact_dates_match": same_dates,
            "baseline_only_dates": int((joined["_merge"] == "left_only").sum()),
            "reference_only_dates": int((joined["_merge"] == "right_only").sum()),
        }
        if filename == "results_df.csv":
            report["lambda2"] = _numeric_comparison(joined["lambda2"], joined["lambda2_reference"],
                                                      rtol=EIG_RTOL, atol=EIG_ATOL)
            counts_match = bool(np.array_equal(joined["num_stocks"].to_numpy(),
                                               joined["num_stocks_reference"].to_numpy()))
            report["stock_counts_match_exactly"] = counts_match
            passed = same_dates and counts_match and report["lambda2"]["matches"]
        else:
            report["mean_raw_correlation"] = _numeric_comparison(
                joined["mean_raw_correlation"], joined["avg_corr"], rtol=0, atol=CORRELATION_ATOL)
            passed = same_dates and report["mean_raw_correlation"]["matches"]
        report["matches"] = passed
        result["files"][filename] = report
        if not passed:
            result["status"] = "mismatch"
    original = references["results_df.csv"].merge(references["avg_corr_df.csv"], on="date",
                                                  how="outer", validate="one_to_one")
    original_correlation = float(original["lambda2"].corr(original["avg_corr"]))
    reproduced_correlation = float(features["lambda2"].corr(features["mean_raw_correlation"]))
    result["reference_lambda2_average_correlation"] = original_correlation
    result["reproduced_lambda2_average_correlation"] = reproduced_correlation
    result["correlation_difference"] = reproduced_correlation - original_correlation
    result["correlation_matches"] = bool(np.isclose(reproduced_correlation, original_correlation,
                                                   rtol=0, atol=CORRELATION_ATOL))
    if not result["correlation_matches"]:
        result["status"] = "mismatch"
    return result


def _json_value(value):
    """Use JSON null for unavailable diagnostics, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def selected_window(returns, features, date, output_dir):
    timestamp = pd.Timestamp(date, tz=returns.index.tz)
    if timestamp not in returns.index:
        raise ValueError(f"The selected daily date {date} is absent from the snapshot.")
    end = returns.index.get_loc(timestamp) + 1
    window = returns.iloc[end - WINDOW_SIZE:end]
    if len(window) != WINDOW_SIZE:
        raise ValueError(f"The selected date {date} has no full 60-observation window.")
    cleaned = clean_window(window)
    baseline = analyze_window(window, include_fiedler=True)
    vector = baseline.pop("fiedler_vector")
    row = features.loc[features["date"] == timestamp].iloc[0]
    if baseline["num_stocks"] != row["num_stocks"] or not np.isclose(
        baseline["lambda2"], row["lambda2"], rtol=EIG_RTOL, atol=EIG_ATOL, equal_nan=True
    ):
        raise AssertionError(f"Selected-window calculation differs from the full baseline on {date}.")
    weights, _ = build_positive_graph(cleaned.corr())
    degrees = weights.sum(axis=1)
    assets = pd.DataFrame({"ticker": cleaned.columns, "degree": degrees.to_numpy()})
    assets["fiedler_vector"] = vector.reindex(cleaned.columns).to_numpy() if vector is not None else np.nan
    assets["fiedler_squared_mass"] = assets["fiedler_vector"] ** 2
    n = len(cleaned.columns)
    weak_symbol = baseline["min_degree_ticker"]
    bound = n / (n - 1) * baseline["min_degree"]
    top = assets.sort_values("fiedler_squared_mass", ascending=False).head(10)
    sensitivity = analyze_window(cleaned.drop(columns=[weak_symbol]))
    report = {
        "date": timestamp.isoformat(), "window_first_return_date": window.index[0].isoformat(),
        "window_last_return_date": window.index[-1].isoformat(),
        "baseline": baseline, "retained_symbols": cleaned.columns.tolist(),
        "removed_symbols": [symbol for symbol in window.columns if symbol not in cleaned],
        "fiedler_vector_available": vector is not None,
        "fiedler_note": "Unit vector; sign is arbitrary. Numerical separation is not resampling evidence.",
        "top_localized_symbols": top.to_dict("records") if vector is not None else [],
        "min_degree_rayleigh_upper_bound": bound,
        "min_degree_rayleigh_formula": "lambda2 <= n/(n-1) * min_degree; single-node contrast vector",
        "lambda2_fraction_of_rayleigh_bound": baseline["lambda2"] / bound if bound > 0 else None,
        "remove_min_degree_node_sensitivity": {
            "label": "Separate descriptive sensitivity; the baseline is unchanged",
            "removed_symbol": weak_symbol, "metrics": sensitivity,
            "lambda2_change": sensitivity["lambda2"] - baseline["lambda2"],
            "limitation": "Removing an asset changes graph composition, n and edge scale; this is not a forecast or a corrected baseline.",
        },
    }
    if baseline["lambda2"] > bound + EIG_ATOL + EIG_RTOL * abs(bound):
        raise AssertionError(f"Rayleigh upper bound failed on {date}.")
    folder = output_dir / "selected_windows" / date
    folder.mkdir(parents=True, exist_ok=True)
    assets.to_csv(folder / "assets.csv", index=False, float_format="%.17g")
    _write_json(folder / "diagnostics.json", report)
    return report


def plot_features(features, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    labels = (("lambda2", "Raw lambda2"), ("lambda2_scale_ratio", "q = lambda2 / (n mean weight)"),
              ("mean_positive_weight", "Mean clipped weight (all pairs)"), ("min_degree", "Minimum weighted degree"))
    for axis, (column, label) in zip(axes, labels):
        axis.plot(features["date"], features[column], linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        for date in SELECTED_DATES:
            axis.axvline(pd.Timestamp(date, tz=features["date"].dt.tz), color="gray", linewidth=0.6, linestyle="--")
    axes[0].set_title("Recovered daily snapshot: descriptive 60-observation positive-correlation graph")
    axes[-1].set_xlabel("Window end date (measurement available after that close)")
    figure.savefig(output_dir / "daily_diagnostics.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="Frozen request-key directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path, help="Optional live-session reference CSV directory")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.workers > 1:
        for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[variable] = "1"
    _import_numerics()
    snapshot = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    # Results must not overwrite or add files to the frozen observations.
    if output_dir == snapshot or snapshot in output_dir.parents:
        raise ValueError("Choose an output directory outside the frozen snapshot.")
    returns, manifest = _load_returns(snapshot)
    references = None
    reference_dir = args.reference_dir.resolve() if args.reference_dir else None
    if reference_dir is not None:
        references = {
            "results_df.csv": _read_reference(reference_dir / "results_df.csv", ["date", "lambda2", "num_stocks"]),
            "avg_corr_df.csv": _read_reference(reference_dir / "avg_corr_df.csv", ["date", "avg_corr"]),
        }
    features = calculate_features(returns, snapshot, args.workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "features.csv", index=False, float_format="%.17g")
    comparison = compare_references(features, references, reference_dir)
    selected = {date: selected_window(returns, features, date, output_dir) for date in SELECTED_DATES}
    plot_features(features, output_dir)
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    correlation = float(features["lambda2"].corr(features["mean_raw_correlation"]))
    summary = {
        "status": "descriptive_reproduction" if comparison["status"] != "mismatch" else "reference_mismatch",
        "claim": "Descriptive market-network measurements; no predictive or trading result",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_directory": str(snapshot), "snapshot_key": snapshot.name,
        "snapshot_price_sha256": manifest["sha256"], "snapshot_manifest": manifest,
        "script_sha256": _hash_file(__file__), "package_versions": versions,
        "settings": {"window_size": WINDOW_SIZE, "step_size": 1, "workers": args.workers,
                     "returns": "arithmetic pct_change(fill_method=None), first row removed",
                     "cleaning": "replace infinities with NaN; drop any incomplete column; sample std > 1e-12",
                     "graph": "Pearson C; W=max(C,0), diagonal=0; L=D-W; all metrics share one cleaned sample",
                     "timing": "Trailing window ends on the label date; available after that close",
                     "blas_threads_per_process": 1 if args.workers > 1 else "environment default"},
        "price_rows": manifest["rows"], "return_rows": len(returns), "feature_rows": len(features),
        "first_feature_date": features["date"].iloc[0].isoformat(),
        "last_feature_date": features["date"].iloc[-1].isoformat(),
        "status_counts": features["status"].value_counts().to_dict(),
        "lambda2_mean_raw_correlation": correlation,
        "correlation_paired_rows": int(features[["lambda2", "mean_raw_correlation"]].dropna().shape[0]),
        "reference_comparison": comparison,
        "selected_dates": {date: {"baseline": report["baseline"],
                                   "remove_min_degree_node_sensitivity": report["remove_min_degree_node_sensitivity"]}
                           for date, report in selected.items()},
        "limitations": ["Original adjustment, universe scrape date, download time and download environment remain unverified.",
                        "Today's selected stock universe changes composition through time.",
                        "Adjacent windows share 59 observations; no independent-sample significance is inferred.",
                        "Fiedler localization and node-removal sensitivity are descriptive, without resampling validation."],
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Saved {len(features)} windows to {output_dir}; correlation={correlation:.15g}", flush=True)
    if comparison["status"] == "mismatch":
        raise SystemExit("Reference mismatch: inspect summary.json; no agreement claim is made.")


if __name__ == "__main__":
    main()
