"""Meaningful, offline numerical checks. Run: python checks.py

These verify equations and data-handling behavior, not predictive usefulness.
Only pandas, NumPy and SciPy are required; no testing framework is needed.
"""

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
    })
    original = window.copy(deep=True)
    cleaned = clean_window(window)
    assert cleaned.columns.tolist() == ["keep"]
    pd.testing.assert_frame_equal(window, original)
    result = analyze_window(window)
    assert result["status"] == "insufficient_assets" and result["num_stocks"] == 1
    assert result["num_removed_stocks"] == 3 and np.isnan(result["lambda2"])
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


if __name__ == "__main__":
    checks = [
        check_complete_uniform_graph,
        check_block_graph_and_identity_shrinkage,
        check_disconnected_and_zero_graphs,
        check_cleaning_and_small_windows,
        check_seeded_independent_returns_and_original_agreement,
        check_rolling_dates_and_consistent_cleaning,
        check_real_types_nullable_correlation_and_date_labels,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"All {len(checks)} checks passed. No market data was downloaded.")
