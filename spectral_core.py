"""Standalone stock-return correlation graphs and descriptive diagnostics.

No download, forecast, or trade is performed here. Every graph uses the same
stocks and observations for all its measurements. The main entry points are
``analyze_window`` (one window) and ``rolling_features`` (many windows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components


def clean_window(window: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame using your original per-window cleaning rule.

    Rows are return observations; columns are asset symbols. Infinity becomes
    NaN. Any column with any missing observation is removed. Finally, pandas'
    sample standard deviation (ddof=1) must exceed 1e-12. Missing returns are
    never replaced with zero. The input DataFrame is not modified.
    """
    if not isinstance(window, pd.DataFrame):
        raise TypeError("window must be a pandas DataFrame of returns.")
    if not window.columns.is_unique:
        raise ValueError("Asset column labels must be unique.")
    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
        for dtype in window.dtypes
    ):
        raise TypeError("Every asset column must contain real numeric returns.")

    window_clean = window.replace([np.inf, -np.inf], np.nan)
    window_clean = window_clean.dropna(axis=1)
    std = window_clean.std()
    window_clean = window_clean.loc[:, std > 1e-12]
    return window_clean


def build_positive_graph(
    correlation: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (W, L) from an already calculated, labeled correlation matrix.

    W is a DataFrame: negative correlations become zero, positive ones remain,
    and the diagonal is zero. L is a NumPy array, L = D - W. Labels and their
    order are retained in W; that order also applies to the rows/columns of L.
    """
    if not isinstance(correlation, pd.DataFrame):
        raise TypeError("correlation must be a pandas DataFrame.")
    n = len(correlation)
    if n < 2 or correlation.shape != (n, n):
        raise ValueError("A correlation graph needs a square matrix of at least two assets.")
    if not correlation.columns.is_unique or not correlation.index.equals(correlation.columns):
        raise ValueError("Correlation row and column labels must be identical and unique.")
    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
        for dtype in correlation.dtypes
    ):
        raise TypeError("Correlation coefficients must be real numeric values.")
    values = correlation.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The correlation matrix must contain only finite values.")
    if not np.allclose(values, values.T, atol=1e-12, rtol=0):
        raise ValueError("The correlation matrix must be symmetric.")
    if not np.allclose(np.diag(values), 1.0, atol=1e-12, rtol=0):
        raise ValueError("The correlation matrix must have ones on its diagonal.")
    if np.max(np.abs(values)) > 1.0 + 1e-12:
        raise ValueError("Correlation coefficients must lie between -1 and 1.")

    # Ordinary float arrays also support pandas' nullable Float64 input safely.
    weights = pd.DataFrame(values, index=correlation.index, columns=correlation.columns).clip(lower=0)
    diagonal_mask = np.eye(n, dtype=bool)
    weights = weights.mask(diagonal_mask, 0)
    degrees = weights.sum(axis=1)
    degree_matrix = np.diag(degrees.to_numpy())
    laplacian = degree_matrix - weights.to_numpy()
    return weights, laplacian


def analyze_correlation(
    correlation: pd.DataFrame, *, include_fiedler: bool = False
) -> dict:
    """Measure one valid correlation graph without cleaning returns again.

    ``mean_positive_weight`` averages ALL off-diagonal weights, including
    zero weights, counting each unordered pair once. It is not the average
    of just the positive edges.

    ``lambda2_scale_ratio`` = lambda2 / (n * mean_positive_weight), or NaN
    when all weights are zero. It removes common edge scaling, but is not a
    universal risk measure or a guarantee of comparability across universes.
    It is a scalar ratio, not an eigenvalue of the degree-normalized Laplacian.

    ``weak_singleton_bound`` = n/(n-1) * min_degree is the Rayleigh upper
    bound from separating one minimum-degree node from all other nodes.
    ``lambda2_over_singleton_bound`` reports proximity to that constraint;
    it is undefined when the bound is zero. Near equality does not identify
    why that stock has weak measured correlations or prove an economic cause.

    This starter reports normalized lambda2 ONLY for connected graphs with
    strictly positive degrees. Other graphs get NaN, with components/status
    reported explicitly, rather than silently choosing an isolated-node
    convention. Their unnormalized lambda2 is exactly zero.

    Fiedler diagnostics are reported only when lambda2 is numerically
    separated from adjacent eigenvalues. A repeated eigenvalue has no unique
    vector. The separation flag is a numerical check, not bootstrap evidence
    of economic stability. The optional vector is a labeled pandas Series.
    """
    weights, laplacian = build_positive_graph(correlation)
    n = len(weights)
    values = weights.to_numpy()
    degrees = weights.sum(axis=1)
    upper_triangle = np.triu_indices(n, k=1)
    mean_raw_correlation = float(correlation.to_numpy(dtype=float)[upper_triangle].mean())
    mean_positive_weight = float(values[upper_triangle].mean())

    # This uses the exact positive edges of the prototype, without thresholding.
    num_components = int(connected_components(values, directed=False, return_labels=False))
    connected = num_components == 1

    # Three eigenvalues let us check whether the second is nearly repeated.
    last_index = min(2, n - 1)
    fiedler_vector = None
    if connected:
        eigenvalues, eigenvectors = eigh(laplacian, subset_by_index=[0, last_index])
        lambda2 = max(0.0, float(eigenvalues[1]))
    else:
        eigenvalues = eigh(laplacian, subset_by_index=[0, last_index], eigvals_only=True)
        lambda2 = 0.0

    lambda3_gap = float(eigenvalues[2] - eigenvalues[1]) if n >= 3 else np.nan
    # A small tolerance relative to the Laplacian's scale identifies degeneracy.
    tolerance = 1e-10 * max(1.0, 2.0 * float(degrees.max()))
    fiedler_separated = connected and lambda2 > tolerance
    if n >= 3:
        fiedler_separated = fiedler_separated and lambda3_gap > tolerance

    fiedler_ipr = np.nan
    if fiedler_separated:
        vector = eigenvectors[:, 1]
        fiedler_ipr = float(np.sum(vector**4))
        if include_fiedler:
            fiedler_vector = pd.Series(vector, index=weights.index, name="fiedler_vector")

    normalized_lambda2 = np.nan
    if connected and (degrees > 0).all():
        inverse_sqrt_degree = 1.0 / np.sqrt(degrees.to_numpy())
        normalized_laplacian = (
            np.eye(n)
            - inverse_sqrt_degree[:, None] * values * inverse_sqrt_degree[None, :]
        )
        normalized_eigenvalues = eigh(
            normalized_laplacian, subset_by_index=[0, 1], eigvals_only=True
        )
        normalized_lambda2 = max(0.0, float(normalized_eigenvalues[1]))

    scale_ratio = lambda2 / (n * mean_positive_weight) if mean_positive_weight > 0 else np.nan
    singleton_bound = n / (n - 1) * float(degrees.min())
    result = {
        "status": "ok" if connected else "disconnected",
        "num_stocks": n,
        "num_components": num_components,
        "lambda2": lambda2,
        "normalized_lambda2": normalized_lambda2,
        "mean_raw_correlation": mean_raw_correlation,
        "mean_positive_weight": mean_positive_weight,
        "positive_edge_fraction": float((values[upper_triangle] > 0).mean()),
        "lambda2_scale_ratio": float(scale_ratio),
        "mean_degree": float(degrees.mean()),
        "min_degree": float(degrees.min()),
        "min_degree_ticker": str(degrees.idxmin()),
        "weak_singleton_bound": float(singleton_bound),
        "lambda2_over_singleton_bound": float(lambda2 / singleton_bound) if singleton_bound > 0 else np.nan,
        "lambda3_gap": lambda3_gap,
        "fiedler_separated": bool(fiedler_separated),
        "fiedler_ipr": fiedler_ipr,
    }
    if include_fiedler:
        result["fiedler_vector"] = fiedler_vector
    return result


def analyze_window(window: pd.DataFrame, *, include_fiedler: bool = False) -> dict:
    """Clean once, calculate Pearson correlations once, and measure one graph.

    If fewer than two assets remain, return status='insufficient_assets' and
    NaN measurements. That window is retained in a rolling result; no graph
    eigenvalue is invented and no missing-data window silently disappears.
    """
    cleaned = clean_window(window)
    n = cleaned.shape[1]
    if n >= 2:
        correlation = cleaned.corr()
        result = analyze_correlation(correlation, include_fiedler=include_fiedler)
    else:
        result = {
            "status": "insufficient_assets",
            "num_stocks": n,
            "num_components": np.nan,
            "lambda2": np.nan,
            "normalized_lambda2": np.nan,
            "mean_raw_correlation": np.nan,
            "mean_positive_weight": np.nan,
            "positive_edge_fraction": np.nan,
            "lambda2_scale_ratio": np.nan,
            "mean_degree": np.nan,
            "min_degree": np.nan,
            "min_degree_ticker": None,
            "weak_singleton_bound": np.nan,
            "lambda2_over_singleton_bound": np.nan,
            "lambda3_gap": np.nan,
            "fiedler_separated": False,
            "fiedler_ipr": np.nan,
        }
        if include_fiedler:
            result["fiedler_vector"] = None
    result["num_input_stocks"] = window.shape[1]
    result["num_removed_stocks"] = window.shape[1] - n
    return result


def rolling_features(
    returns: pd.DataFrame, window_size: int = 60, step_size: int = 1
) -> pd.DataFrame:
    """Return one row for each full rolling window, dated at its LAST row.

    Example: window_size=60, step_size=5 analyzes rows 0:60, 5:65, 10:70,
    and so on. The unit is observations (usually trading days), not calendar
    days. The last incomplete window is not calculated. Vectors and full
    matrices are not stored in this table; reconstruct an individual window
    with analyze_window(..., include_fiedler=True) when needed.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    for name, value in (("window_size", window_size), ("step_size", step_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
    if window_size < 2 or step_size < 1:
        raise ValueError("window_size must be at least 2 and step_size at least 1.")
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.hasnans:
        raise ValueError("Rolling return rows require a nonmissing DatetimeIndex.")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("Return rows must have unique dates in increasing order.")
    if len(returns) < window_size:
        raise ValueError("There are not enough return observations for one full window.")

    results = []
    for end in range(window_size, len(returns) + 1, step_size):
        window = returns.iloc[end - window_size : end]
        row = analyze_window(window)
        row["date"] = window.index[-1]
        results.append(row)
    results_df = pd.DataFrame(results)
    first_columns = ["date", "lambda2", "num_stocks"]
    other_columns = [column for column in results_df.columns if column not in first_columns]
    return results_df[first_columns + other_columns]
