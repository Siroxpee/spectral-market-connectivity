"""Standalone stock-graph numerical checks. Run: python checks.py

These verify equations and data-handling behavior, not predictive usefulness.
Only pandas, NumPy and SciPy are required; no testing framework is needed.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from spectral_core import (
    analyze_correlation,
    analyze_window,
    build_positive_graph,
    clean_window,
    rolling_features,
)


def labeled_correlation(values):
    names = [f"asset_{i}" for i in range(len(values))]
    return pd.DataFrame(values, index=names, columns=names)


def check_complete_uniform_graph():
    n, c = 8, 0.25
    correlation = np.full((n, n), c)
    np.fill_diagonal(correlation, 1.0)
    result = analyze_correlation(labeled_correlation(correlation), include_fiedler=True)
    np.testing.assert_allclose(result["lambda2"], n * c, atol=1e-12)
    np.testing.assert_allclose(result["normalized_lambda2"], n / (n - 1), atol=1e-12)
    np.testing.assert_allclose(result["lambda2_scale_ratio"], 1.0, atol=1e-12)
    # The repeated lambda2 has no unique Fiedler vector.
    assert not result["fiedler_separated"]
    assert result["fiedler_vector"] is None
    assert np.isnan(result["fiedler_ipr"])


def check_block_graph_and_identity_shrinkage():
    n, within, between = 6, 0.7, 0.1
    correlation = np.full((n, n), between)
    correlation[:3, :3] = within
    correlation[3:, 3:] = within
    np.fill_diagonal(correlation, 1.0)
    # This is a valid, positive-definite correlation matrix, not arbitrary edges.
    assert np.linalg.eigvalsh(correlation).min() > 0
    result = analyze_correlation(labeled_correlation(correlation), include_fiedler=True)
    np.testing.assert_allclose(result["lambda2"], n * between, atol=1e-12)
    vector = result["fiedler_vector"].to_numpy()
    assert np.all(vector[:3] * vector[0] > 0)
    assert np.all(vector[3:] * vector[0] < 0)
    np.testing.assert_allclose(result["fiedler_ipr"], 1 / n, atol=1e-12)

    alpha = 0.35
    shrunk = (1 - alpha) * correlation + alpha * np.eye(n)
    shrunk_result = analyze_correlation(labeled_correlation(shrunk))
    np.testing.assert_allclose(shrunk_result["lambda2"], (1 - alpha) * result["lambda2"])
    np.testing.assert_allclose(shrunk_result["normalized_lambda2"], result["normalized_lambda2"])
    np.testing.assert_allclose(shrunk_result["lambda2_scale_ratio"], result["lambda2_scale_ratio"])


def check_disconnected_and_zero_graphs():
    # Two perfectly separated, internally positive blocks: no isolated nodes.
    correlation = np.eye(4)
    correlation[0, 1] = correlation[1, 0] = 0.5
    correlation[2, 3] = correlation[3, 2] = 0.5
    result = analyze_correlation(labeled_correlation(correlation), include_fiedler=True)
    assert result["status"] == "disconnected" and result["num_components"] == 2
    assert result["lambda2"] == 0
    assert np.isnan(result["normalized_lambda2"])
    assert result["fiedler_vector"] is None

    isolated = labeled_correlation([[1, .4, 0], [.4, 1, 0], [0, 0, 1]])
    result = analyze_correlation(isolated, include_fiedler=True)
    assert result["num_components"] == 2 and result["min_degree"] == 0
    assert result["weak_singleton_bound"] == 0
    assert np.isnan(result["lambda2_over_singleton_bound"])
    assert np.isnan(result["normalized_lambda2"]) and result["fiedler_vector"] is None

    # Negative correlations are clipped, and all-zero edge weights have no ratio.
    result = analyze_correlation(labeled_correlation([[1.0, -0.4], [-0.4, 1.0]]))
    assert result["lambda2"] == 0 and result["mean_positive_weight"] == 0
    assert np.isnan(result["lambda2_scale_ratio"])


def check_cleaning_and_small_windows():
    window = pd.DataFrame({
        "keep": [0.1, -0.2, 0.3, -0.1],
        "missing": [0.1, np.nan, 0.3, -0.1],
        "constant": [0.2, 0.2, 0.2, 0.2],
        "infinite": [0.1, np.inf, 0.3, -np.inf],
        "near_constant": [0., 1e-14, -1e-14, 0.],
    })
    original = window.copy(deep=True)
    cleaned = clean_window(window)
    assert cleaned.columns.tolist() == ["keep"]
    pd.testing.assert_frame_equal(window, original)
    result = analyze_window(window)
    assert result["status"] == "insufficient_assets" and result["num_stocks"] == 1
    assert result["num_removed_stocks"] == 4 and np.isnan(result["lambda2"])
    assert analyze_window(window.iloc[:1])["num_stocks"] == 0
    assert analyze_window(pd.DataFrame(index=range(4)))["status"] == "insufficient_assets"


def check_seeded_independent_returns_and_original_agreement():
    rng = np.random.default_rng(20261008)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(60, 30)))
    result = analyze_window(returns)
    # Independent populations still create nonzero positive sample edges.
    assert abs(result["mean_raw_correlation"]) < 0.1
    assert result["mean_positive_weight"] > 0
    assert result["num_components"] == 1
    assert 0 <= result["lambda2_scale_ratio"] <= 1 + 1e-12

    # This reproduces the user's original calculation, independently of helpers.
    original_cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    original_cleaned = original_cleaned.loc[:, original_cleaned.std() > 1e-12]
    original_weights = original_cleaned.corr().clip(lower=0)
    original_weights = original_weights.mask(np.eye(len(original_weights), dtype=bool), 0)
    original_degrees = original_weights.sum(axis=1)
    original_laplacian = np.diag(original_degrees.to_numpy()) - original_weights.to_numpy()
    original_lambda2 = eigh(original_laplacian, subset_by_index=[0, 1], eigvals_only=True)[1]
    np.testing.assert_allclose(result["lambda2"], original_lambda2, atol=1e-12)

    # Verify the short-window rank issue belongs to C, not to clipped L.
    high_dimension = pd.DataFrame(rng.normal(size=(12, 25)))
    correlation = high_dimension.corr()
    assert np.linalg.matrix_rank(correlation.to_numpy(), tol=1e-10) <= 11
    weights, laplacian = build_positive_graph(correlation)
    assert np.linalg.matrix_rank(laplacian, tol=1e-10) == 24


def check_rolling_dates_and_consistent_cleaning():
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(size=(12, 4)),
        index=pd.bdate_range("2020-01-01", periods=12),
        columns=["A", "B", "C", "D"],
    )
    returns.iloc[0, 0] = np.nan
    result = rolling_features(returns, window_size=6, step_size=3)
    assert result["date"].tolist() == returns.index[[5, 8, 11]].tolist()
    assert result["num_stocks"].tolist() == [3, 4, 4]
    for position, end in enumerate([6, 9, 12]):
        direct = analyze_window(returns.iloc[end - 6:end])
        np.testing.assert_allclose(result.iloc[position]["lambda2"], direct["lambda2"])
        np.testing.assert_allclose(result.iloc[position]["mean_positive_weight"], direct["mean_positive_weight"])

    try:
        rolling_features(returns.iloc[::-1], 6)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsorted dates must be rejected.")


def check_real_types_nullable_correlation_and_date_labels():
    correlation = labeled_correlation([[1.0, 0.4], [0.4, 1.0]]).astype("Float64")
    np.testing.assert_allclose(analyze_correlation(correlation)["lambda2"], 0.8)
    for values in ([True, False, True], [1 + 1j, 2 + 2j, 3 + 3j]):
        try:
            analyze_window(pd.DataFrame({"A": values}))
        except TypeError:
            pass
        else:
            raise AssertionError("Boolean/complex returns must be rejected.")
    try:
        build_positive_graph(labeled_correlation([[1, 0.4j], [-0.4j, 1]]))
    except TypeError:
        pass
    else:
        raise AssertionError("Complex correlations must be rejected.")
    strings = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 1.0, 2.0]},
                           index=["2026-1-1", "2026-1-10", "2026-1-2"])
    try:
        rolling_features(strings, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("String sort order must not masquerade as chronological dates.")


def check_singleton_bound_modes_and_permutations():
    # A valid correlation matrix with one deliberately weak node. Its graph
    # has lambda2 = n*b and attains the singleton Rayleigh bound exactly.
    n, within, weak = 6, .5, .02
    values = np.full((n, n), within)
    values[0, :] = weak
    values[:, 0] = weak
    np.fill_diagonal(values, 1.)
    correlation = labeled_correlation(values)
    assert np.linalg.eigvalsh(values).min() > 0
    result = analyze_correlation(correlation, include_fiedler=True)
    weights, laplacian = build_positive_graph(correlation)
    np.testing.assert_allclose(laplacian.sum(axis=1), 0., atol=1e-14)
    np.testing.assert_allclose(np.trace(laplacian), weights.to_numpy().sum())
    assert np.linalg.eigvalsh(laplacian).min() > -1e-12
    np.testing.assert_allclose(result["lambda2"], n * weak, atol=1e-12)
    np.testing.assert_allclose(result["weak_singleton_bound"], n * weak, atol=1e-12)
    np.testing.assert_allclose(result["lambda2_over_singleton_bound"], 1., atol=1e-12)
    vector = result["fiedler_vector"].to_numpy()
    np.testing.assert_allclose(np.linalg.norm(vector), 1., atol=1e-12)
    np.testing.assert_allclose(vector.sum(), 0., atol=1e-12)
    np.testing.assert_allclose(laplacian @ vector, result["lambda2"] * vector, atol=1e-12)
    np.testing.assert_allclose(vector[0] ** 2, (n - 1) / n, atol=1e-12)
    np.testing.assert_allclose(result["fiedler_ipr"], .7, atol=1e-12)
    # A coordinate's squared mass and sum(v_i**4) are different quantities.
    assert not np.isclose(vector[0] ** 2, result["fiedler_ipr"])
    permutation = [3, 0, 5, 2, 1, 4]
    moved = analyze_correlation(correlation.iloc[permutation, permutation], include_fiedler=True)
    for name in ("lambda2", "normalized_lambda2", "lambda2_scale_ratio", "weak_singleton_bound", "fiedler_ipr"):
        np.testing.assert_allclose(result[name], moved[name], atol=1e-12)
    aligned = moved["fiedler_vector"].reindex(correlation.index).to_numpy()
    np.testing.assert_allclose(abs(vector @ aligned), 1., atol=1e-12)


def check_local_stock_pipeline():
    import spectral_core
    import stock_analysis
    import stock_data
    here = Path(__file__).resolve().parent
    assert all(Path(module.__file__).resolve().parent == here
               for module in (spectral_core, stock_analysis, stock_data))
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
                            "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"])
    prices = pd.DataFrame({"A": [100., 110., 105., 108., 103., 112., 107., 115.],
                           "B": [80., 82., np.nan, 85., 81., 83., 84., 90.],
                           "C": [50.] * 8, "D": [120., 119., 124., 123., 127., 125., 130., 129.]},
                          index=dates.rename("Date"))
    original = prices.copy(deep=True)
    changes = stock_data.arithmetic_returns(prices)
    pd.testing.assert_frame_equal(prices, original)
    np.testing.assert_allclose(changes.loc["2020-01-06", "A"], 105 / 110 - 1)
    assert changes["B"].isna().sum() == 2
    assert changes.loc[["2020-01-06", "2020-01-07"], "B"].isna().all()
    frame = rolling_features(changes, window_size=4)
    assert frame.date.tolist() == dates[4:].tolist()
    report, nodes = stock_analysis.diagnose_window(changes, dates[-1], window_size=4)
    assert report["baseline"]["num_stocks"] == 3 and report["removed_symbols"] == ["C"]
    assert set(nodes.symbol) == {"A", "B", "D"}

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        price_file = folder / "prices.csv.gz"
        prices.to_csv(price_file, float_format="%.17g")
        digest = stock_data.file_sha256(price_file)
        manifest = {"sha256": digest, "rows": len(prices), "actual_symbols": list(prices.columns),
                    "actual_first_timestamp": dates[0].isoformat(), "actual_last_timestamp": dates[-1].isoformat(),
                    "missing_cells": 1, "index_timezone": None, "request": {"source": "synthetic stocks"}}
        manifest_file = folder / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        restored, metadata = stock_data.load_stock_prices(folder, expected_sha256=digest)
        pd.testing.assert_frame_equal(prices, restored, check_freq=False, check_dtype=False, check_exact=True)
        assert metadata["manifest_checked"] and metadata["missing_price_cells"] == 1
        metric_file = folder / "window_metrics.csv"
        frame.to_csv(metric_file, index=False, float_format="%.17g")
        binding = {"price_sha256": digest, "window_size": 4, "step_size": 1,
                   "metrics_rows": len(frame), "metrics_sha256": stock_data.file_sha256(metric_file)}
        rebound = stock_analysis.load_metrics(metric_file, binding, restored, metadata, 4, 1)
        np.testing.assert_allclose(rebound.lambda2, frame.lambda2, equal_nan=True)
        for action in (lambda: stock_data.load_stock_prices(folder, expected_sha256="0" * 64),
                       lambda: stock_analysis.load_metrics(metric_file, dict(binding, window_size=5), restored, metadata, 4, 1)):
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched source/settings must reject a purported reproduction.")
        manifest_file.write_text(json.dumps(dict(manifest, rows=99)), encoding="utf-8")
        try:
            stock_data.load_stock_prices(folder)
        except ValueError:
            pass
        else:
            raise AssertionError("Manifest summaries must describe the actual observations.")


def check_equal_mean_four_stock_control():
    for within, between, expected_lambda2, expected_q in ((.8, .1, .4, .3), (.4, .3, 1.2, .9)):
        values = np.full((4, 4), between)
        values[:2, :2] = within
        values[2:, 2:] = within
        np.fill_diagonal(values, 1.)
        assert np.linalg.eigvalsh(values).min() > 0
        weights, laplacian = build_positive_graph(labeled_correlation(values))
        np.testing.assert_allclose(np.linalg.eigvalsh(laplacian),
                                   [0., 4 * between, 2 * (within + between), 2 * (within + between)], atol=1e-12)
        result = analyze_correlation(labeled_correlation(values))
        np.testing.assert_allclose(weights.to_numpy()[np.triu_indices(4, 1)].mean(), 1 / 3, atol=1e-12)
        np.testing.assert_allclose(result["lambda2"], expected_lambda2, atol=1e-12)
        np.testing.assert_allclose(result["lambda2_scale_ratio"], expected_q, atol=1e-12)
    # Reversing a >= b changes which analytic eigenvalue is second-smallest.
    values = np.full((4, 4), .4)
    values[:2, :2] = values[2:, 2:] = .1
    np.fill_diagonal(values, 1.)
    result = analyze_correlation(labeled_correlation(values), include_fiedler=True)
    np.testing.assert_allclose(result["lambda2"], min(4 * .4, 2 * (.1 + .4)), atol=1e-12)
    assert not result["fiedler_separated"] and result["fiedler_vector"] is None


if __name__ == "__main__":
    checks = [
        check_complete_uniform_graph,
        check_block_graph_and_identity_shrinkage,
        check_disconnected_and_zero_graphs,
        check_cleaning_and_small_windows,
        check_seeded_independent_returns_and_original_agreement,
        check_rolling_dates_and_consistent_cleaning,
        check_real_types_nullable_correlation_and_date_labels,
        check_singleton_bound_modes_and_permutations,
        check_local_stock_pipeline,
        check_equal_mean_four_stock_control,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"All {len(checks)} checks passed. No market data was downloaded.")
