"""Inspectable Python-only crypto network calculations on one common sample.

No acquisition, filling, forecast fitting, price selection or plotting occurs
here. Public functions return ordinary Python dictionaries/lists for JSON and
presentation. Signed lowest modes are deliberately not ordinary Fiedler modes.
"""
from __future__ import annotations

import math
import numbers
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.sparse.csgraph import connected_components

OPERATORS = ("positive", "negative", "signed", "shifted")
STD_MIN = 1e-12
MAX_ROLLING_WINDOWS = 120


def serial(value: Any) -> Any:
    """Convert NumPy scalars/arrays and missing floating values to JSON types."""
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(item) for item in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _labels(labels):
    if isinstance(labels, (str, bytes)):
        raise TypeError("Labels must be a sequence of complete variable IDs.")
    result = list(labels)
    if any(not isinstance(label, str) or not label.strip() for label in result):
        raise ValueError("Variable labels must be nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError("Variable labels must be unique.")
    return result


def _rows(values, width=None):
    rows = list(values)
    if width is None:
        width = len(rows[0]) if rows else 0
    for row in rows:
        if len(row) != width:
            raise ValueError("Every observation must have the same number of variables.")
        if any(value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)) for value in row):
            raise TypeError("Values must be numeric or missing, not strings/booleans.")
    array = np.asarray(rows, dtype=float).reshape(len(rows), width)
    return np.where(np.isfinite(array), array, np.nan)


def _times(dates, length):
    dates = list(dates)
    if len(dates) != length:
        raise ValueError("Dates must match observation rows.")
    stamps = []
    for value in dates:
        if not isinstance(value, (str, pd.Timestamp)) and not hasattr(value, "isoformat"):
            raise TypeError("Dates must be ISO strings or datetime objects.")
        if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?", value):
            raise ValueError("Use valid ISO date labels.")
        stamp = pd.Timestamp(str(value) if isinstance(value, str) else value)
        if pd.isna(stamp):
            raise ValueError("Date labels cannot be missing.")
        stamps.append(stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC"))
    result = pd.DatetimeIndex(stamps)
    if len(result) > 1 and np.any(np.diff(result.asi8) <= 0):
        raise ValueError("Date labels must be unique and increasing.")
    return dates, result


def prepare(prices, dates, bar_seconds=None):
    """Strict-positive arithmetic returns, with every native gap left missing.

    Supply one preceding close to obtain a return at the first study label.
    Missing/nonfinite prices remain missing; finite nonpositive prices raise.
    """
    array = _rows(prices)
    labels, times = _times(dates, len(array))
    if bar_seconds is not None and (isinstance(bar_seconds, bool) or not isinstance(bar_seconds, numbers.Real) or not math.isfinite(bar_seconds) or bar_seconds <= 0):
        raise ValueError("bar_seconds must be positive or None.")
    if np.any(array[np.isfinite(array)] <= 0):
        raise ValueError("Arithmetic returns require strictly positive prices.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        changes = array[1:] / array[:-1] - 1.0
    changes[~np.isfinite(changes)] = np.nan
    gap = np.zeros(max(0, len(array) - 1), dtype=bool)
    if bar_seconds is not None and len(array) > 1:
        gap = np.diff(times.asi8) != int(bar_seconds * 1_000_000_000)
        changes[gap] = np.nan
    return {"dates": labels[1:], "values": serial(changes), "gapRows": np.flatnonzero(gap).tolist(),
            "transform": "arithmetic_return", "timestampPolicy": "Original labels retained; naive ISO labels treated as UTC only for elapsed-time checks."}


def common_complete(values):
    """Research-cohort guard: all selected columns must survive unchanged."""
    array = np.asarray(values, dtype=float)
    return bool(array.ndim == 2 and len(array) >= 2 and np.isfinite(array).all() and np.all(array.std(axis=0, ddof=1) > STD_MIN))


def _eigen(matrix):
    eigenvalues, vectors = linalg.eigh(matrix, check_finite=True)
    scale = max(1.0, float(linalg.norm(matrix, "fro")))
    residual = float(linalg.norm(matrix @ vectors - vectors * eigenvalues, "fro") / scale)
    orthogonality = float(linalg.norm(vectors.T @ vectors - np.eye(len(matrix)), "fro"))
    if residual > 1e-9 or orthogonality > 1e-9:
        raise ArithmeticError("Symmetric eigenvalue residual/orthogonality check failed.")
    return eigenvalues, vectors, {"method": "scipy.linalg.eigh", "converged": True,
                                "relativeResidual": residual, "orthogonalityError": orthogonality}


def graph_measurements(correlation, operator="positive"):
    """Four distinct Laplacians on a supplied finite symmetric correlation C.

    Wminus stores negative-edge magnitudes. Signed W stores signed C, while
    its degree sums absolute weights: Ls=Lplus+(Dminus+Wminus), not Lplus-Lminus.
    """
    c = np.asarray(correlation, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or len(c) < 2 or not np.isfinite(c).all():
        raise ValueError("A finite square correlation matrix with >=2 nodes is required.")
    if not np.allclose(c, c.T, atol=1e-12, rtol=0) or not np.allclose(np.diag(c), 1, atol=1e-12, rtol=0) or np.max(np.abs(c)) > 1 + 1e-12:
        raise ValueError("C must be symmetric, bounded by1, with unit diagonal.")
    if operator not in OPERATORS:
        raise ValueError("Unknown graph operator.")
    n = len(c)
    s = c.copy(); np.fill_diagonal(s, 0)
    signed = operator == "signed"
    w = np.maximum(s, 0) if operator == "positive" else np.maximum(-s, 0) if operator == "negative" else s.copy() if signed else (c + 1) / 2
    np.fill_diagonal(w, 0)
    degrees = np.abs(w).sum(axis=1) if signed else w.sum(axis=1)
    lap = np.diag(degrees) - w
    count, component_ids = connected_components(np.abs(w) > 0, directed=False, return_labels=True)
    eigenvalues, vectors, solver = _eigen(lap)
    tolerance = 1e-10 * max(1.0, 2 * float(degrees.max(initial=0)))
    if eigenvalues[0] < -tolerance:
        raise ArithmeticError("Expected positive-semidefinite graph operator failed.")
    eigenvalues = np.maximum(eigenvalues, 0)
    if not signed:
        for i in range(count):
            if eigenvalues[i] <= tolerance:
                eigenvalues[i] = 0
    index = 0 if signed else 1
    forward = float(eigenvalues[index + 1] - eigenvalues[index]) if index + 1 < n else None
    backward = float(eigenvalues[index] - eigenvalues[index - 1]) if index > 0 else None
    separation = min(gap for gap in (forward, backward) if gap is not None)
    reason = None
    if count != 1:
        reason = "Absolute signed-edge support is disconnected; no global lowest mode is displayed." if signed else "Graph is disconnected or contains isolated coins; classical Fiedler mode is not displayed."
    elif not signed and eigenvalues[1] <= tolerance:
        reason = "Lambda2 is numerically indistinguishable from the zero mode."
    elif separation <= tolerance:
        reason = "Selected eigenvalue is repeated or not numerically separated; a unique vector cannot be displayed."
    mode = vectors[:, index].copy() if reason is None else None
    if mode is not None and mode[np.argmax(np.abs(mode))] < 0:
        mode *= -1
    ipr = float(np.sum(mode ** 4)) if mode is not None else None
    lambda2 = (float(eigenvalues[1]) if count == 1 else 0.0) if not signed else None
    mu1 = float(eigenvalues[0]) if signed else None
    mean_weight = float(w[np.triu_indices(n, 1)].mean()) if not signed else None
    normalized_lambda2 = None
    if not signed and count == 1 and np.all(degrees > 0):
        inv = 1 / np.sqrt(degrees)
        normalized_lambda2 = float(max(0, linalg.eigh(np.eye(n) - inv[:, None] * w * inv[None, :], eigvals_only=True)[1]))
    trace_expected = float(degrees.sum()); trace_observed = float(eigenvalues.sum())
    if abs(trace_expected - trace_observed) > 1e-8 * max(1, trace_expected):
        raise ArithmeticError("Graph trace identity failed.")
    diagnostics = {"PSD": True, "minEigenvalue": float(eigenvalues[0]), "traceExpected": trace_expected,
                   "traceObserved": trace_observed, "traceError": abs(trace_expected - trace_observed),
                   "rowSumMax": float(np.abs(lap.sum(axis=1)).max()), "zeroRowSumsExpected": not signed,
                   "modeResidual": float(linalg.norm(lap @ mode - eigenvalues[index] * mode)) if mode is not None else None,
                   "modeUnitNorm": float(mode @ mode) if mode is not None else None, "solver": solver}
    definition = {"positive": "Wij=max(Cij,0) off diagonal; W_ii=0; L=D-W.",
                  "negative": "Wij=max(-Cij,0) off diagonal; W_ii=0; L=D-W. Negative-edge magnitude layer.",
                  "signed": "Ls=diag(sum_j|Sij|)-S=Lplus+Qminus, Qminus=Dminus+Wminus; not Lplus-Lminus.",
                  "shifted": "Wij=(Cij+1)/2 off diagonal; W_ii=0; L=D-W. Zero correlation becomes weight0.5."}[operator]
    return serial({"name": operator, "operator": operator, "W": w, "weights": w, "L": lap, "laplacian": lap,
                   "degrees": degrees, "eigenvalues": eigenvalues, "lambda2": lambda2, "mu1": mu1,
                   "metric": mu1 if signed else lambda2, "metricName": "mu1" if signed else "lambda2",
                   "smallestEigenvalue": eigenvalues[0], "largestEigenvalue": eigenvalues[-1],
                   "modeKind": "signed_lowest" if signed else "fiedler", "mode": mode, "vector": mode,
                   "lowestMode": mode if signed else None, "fiedler": mode if not signed else None,
                   "modeReason": reason, "reason": reason, "modeIPR": ipr, "IPR": ipr, "ipr": ipr,
                   "eligible": reason is None, "modeEligible": reason is None, "mode_unique": reason is None,
                   "fiedlerEligible": not signed and reason is None, "gap": forward if forward is not None else backward,
                   "separationGap": separation, "separationTolerance": tolerance, "tolerance": tolerance,
                   "components": count, "support_components": count, "componentIds": component_ids,
                   "q": lambda2 / (n * mean_weight) if mean_weight is not None and mean_weight > 0 else None,
                   "normalized_lambda2": normalized_lambda2, "modeTurnover": None, "turnover": None,
                   "turnoverReason": "Two comparable windows are required.", "diagnostics": diagnostics, "definition": definition,
                   "metricExplanation": "Smallest signed-Laplacian eigenvalue mu1; not ordinary algebraic connectivity and not automatically zero." if signed else "Ordinary algebraic connectivity lambda2; exactly zero for a disconnected nonnegative-weight graph."})


def _options(options):
    result = {"operator": "positive", "correlation": "pearson", "barsPerYear": None, "btcLabel": None, **(options or {})}
    if result["operator"] not in OPERATORS or result["correlation"] not in ("pearson", "spearman"):
        raise ValueError("Unknown operator or correlation rule.")
    annual = result["barsPerYear"]
    if annual is not None and (isinstance(annual, bool) or not isinstance(annual, numbers.Real) or not math.isfinite(annual) or annual <= 0):
        raise ValueError("barsPerYear must be positive or None.")
    if result["btcLabel"] is not None and not isinstance(result["btcLabel"], str):
        raise TypeError("btcLabel must be an exact variable ID or None.")
    return result


def analyze(prepared_rows, labels, options=None):
    """Clean whole columns once, then calculate all four graphs and indicators."""
    labels = _labels(labels); o = _options(options); rows = _rows(prepared_rows, len(labels))
    kept = []; removed = []
    for j, label in enumerate(labels):
        column = rows[:, j]
        reason = "incomplete_or_nonfinite" if not np.isfinite(column).all() else "fewer_than_two_observations" if len(rows) < 2 else "sample_std_at_or_below_1e-12" if np.std(column, ddof=1) <= STD_MIN else None
        if reason:
            removed.append({"index": j, "label": label, "reason": reason})
        else:
            kept.append(j)
    n = len(kept)
    warnings = ([f"{len(removed)} of {len(labels)} selected coins fail this window; selected coin list was not replaced."] if removed else []) + [f"Removed {item['label']}: {item['reason']}." for item in removed]
    result = {"status": "ok" if n >= 2 else "insufficient_assets", "reason": None if n >= 2 else "Fewer than two variables survive the common-window cleaning rule.",
              "n": n, "inputN": len(labels), "kept": kept, "keptIndices": kept.copy(), "labels": [labels[i] for i in kept],
              "selectedLabels": labels, "removed": removed, "warnings": warnings, "observations": len(rows), "options": o,
              "C": None, "correlation": None, "Wplus": None, "Wminus": None, "S": None, "Qminus": None, "z": None,
              "operators": dict.fromkeys(OPERATORS), "indicators": None, "selectedOperator": o["operator"],
              "currentWeights": None, "currentLaplacian": None, "currentSpectrum": None,
              "visualizationMode": "shifted_similarity" if o["operator"] == "shifted" else "correlation_two_layers",
              "colorMidpoint": .5, "correlationEigenvalues": None, "correlationRank": None, "rankUpperBound": min(n, max(0, len(rows) - 1))}
    if n < 2:
        return result
    clean = rows[:, kept]
    ranks = stats.rankdata(clean, axis=0, method="average") if o["correlation"] == "spearman" else clean
    c = np.corrcoef(ranks, rowvar=False); c = np.clip((c + c.T) / 2, -1, 1); np.fill_diagonal(c, 1)
    ceig, _, _ = _eigen(c)
    if ceig[0] < -1e-9 * max(1, float(ceig[-1])):
        raise ArithmeticError("Common-sample correlation PSD check failed.")
    ceig = np.maximum(ceig, 0)
    operators = {name: graph_measurements(c, name) for name in OPERATORS}
    upper = c[np.triu_indices(n, 1)]
    positive = operators["positive"]
    btc = result["labels"].index(o["btcLabel"]) if o["btcLabel"] in result["labels"] else -1
    if o["btcLabel"] is None:
        matches = [i for i, label in enumerate(result["labels"]) if label.split("::")[-1].upper() in ("BTC", "BTCUSDT", "BTC-USD", "BTC/USD", "BTC-USD:USD")]
        btc = matches[0] if len(matches) == 1 else -1
    vols = clean.std(axis=0, ddof=1); basket_vol = float(clean.mean(axis=1).std(ddof=1))
    annual = math.sqrt(o["barsPerYear"]) if o["barsPerYear"] is not None else None
    momentum = None; momentum_reason = None
    if btc < 0:
        momentum_reason = "Exact BTC variable is absent from this cleaned window; no proxy substituted."
    elif len(clean) < 20:
        momentum_reason = "Twenty prepared BTC returns are required."
    elif np.any(clean[-20:, btc] <= -1):
        momentum_reason = "A prepared BTC return is incompatible with strictly positive prices."
    else:
        with np.errstate(over="ignore"):
            value = float(np.expm1(np.log1p(clean[-20:, btc]).sum()))
        momentum = value if math.isfinite(value) else None
        if momentum is None:
            momentum_reason = "Compounded BTC momentum exceeded finite numeric range."
    indicators = {"meanCorrelation": float(upper.mean()), "meanAbsCorrelation": float(np.abs(upper).mean()),
                  "negativeFraction": float((upper < 0).mean()), "minDegree": min(positive["degrees"]),
                  "positiveEdgeDensityAt03": float((upper >= .3).mean()), "pcaShare": float(ceig[-1] / ceig.sum()),
                  "basketVolAnnual": basket_vol * annual if annual is not None else None,
                  "btcVolAnnual": float(vols[btc] * annual) if annual is not None and btc >= 0 else None,
                  "averageCoinVolAnnual": float(vols.mean() * annual) if annual is not None else None,
                  "basketVol": basket_vol, "btcVol": float(vols[btc]) if btc >= 0 else None, "averageCoinVol": float(vols.mean()),
                  "btcMomentum20": momentum, "btcLabel": result["labels"][btc] if btc >= 0 else None, "btcMomentumReason": momentum_reason,
                  "barsPerYear": o["barsPerYear"], "volatilityUnits": "Sample standard deviation of decimal arithmetic returns; annualized fields multiply by sqrt(barsPerYear). Multiply by100 for percent.",
                  "volatilityReason": "No barsPerYear supplied; annualized volatility is intentionally unavailable." if annual is None else None,
                  "basketDefinition": "Equal-weight arithmetic-return basket of the same retained coins and common observations.",
                  "momentumDefinition": "Product of(1+r) minus1 across last20 prepared BTC observations; not necessarily20days."}
    minus = np.asarray(operators["negative"]["W"]); qminus = np.diag(operators["negative"]["degrees"]) + minus
    residual = float(np.max(np.abs(np.asarray(operators["signed"]["L"]) - np.asarray(positive["L"]) - qminus)))
    if residual > 1e-10:
        raise ArithmeticError("Signed decomposition Ls=Lplus+Qminus failed.")
    result.update(C=c, correlation=c, operators=operators, indicators=indicators,
                  Wplus=positive["W"], Wminus=minus, S=operators["signed"]["W"], Qminus=qminus,
                  z=(c + 1) / 2, signedDecompositionResidual=residual, correlationEigenvalues=ceig,
                  correlationRank=int(np.sum(ceig > 1e-10 * max(1, float(ceig[-1])))),
                  currentWeights=operators[o["operator"]]["W"], currentLaplacian=operators[o["operator"]]["L"], currentSpectrum=operators[o["operator"]]["eigenvalues"])
    return serial(result)


def turnover(previous, current, operator="positive"):
    """Sign-invariant unit-mode change after aligning exactly the same IDs."""
    if operator not in OPERATORS:
        raise ValueError("Unknown turnover operator.")
    if previous is None or current is None:
        return {"value": None, "reason": "Two comparable windows are required."}
    a, b = previous["operators"][operator], current["operators"][operator]
    if a is None or b is None or not a["eligible"] or not b["eligible"]:
        return {"value": None, "reason": "Both windows need connected support and a numerically simple selected mode."}
    if set(previous["labels"]) != set(current["labels"]):
        return {"value": None, "reason": "Retained coin identity changed; no vector comparison is made."}
    positions = {label: i for i, label in enumerate(current["labels"])}
    dot = sum(value * b["mode"][positions[label]] for label, value in zip(previous["labels"], a["mode"]))
    return {"value": float(np.clip(1 - abs(dot), 0, 1)), "reason": None}


compare_modes = turnover


def rolling(prepared_rows, dates, labels, options=None):
    """Up to120 scheduled windows; turnover always uses preceding NATIVE bar."""
    labels = _labels(labels); o = _options(options); rows = _rows(prepared_rows, len(labels))
    dates, _ = _times(dates, len(rows))
    window, step = o.get("window", 60), o.get("step", 1)
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in (window, step)) or window < 2 or step < 1:
        raise ValueError("Window must be an integer>=2 and step an integer>=1.")
    candidates = list(range(window, len(rows) + 1, step))
    if candidates and candidates[-1] != len(rows):
        candidates.append(len(rows))
    capped = len(candidates) > MAX_ROLLING_WINDOWS
    # Match historical JavaScript Math.round for nonnegative schedule indices.
    ends = [candidates[math.floor(i * (len(candidates) - 1) / (MAX_ROLLING_WINDOWS - 1) + .5)] for i in range(MAX_ROLLING_WINDOWS)] if capped else candidates
    results = [analyze(rows[end-window:end], labels, o) for end in ends]
    saved = dict(zip(ends, results)); changes = []
    for end, current in zip(ends, results):
        previous = None if end <= window else saved.get(end - 1)
        if previous is None and end > window:
            previous = analyze(rows[end-window-1:end-1], labels, o)
        per_operator = {}
        for name in OPERATORS:
            change = turnover(previous, current, name); per_operator[name] = change
            op = current["operators"][name]
            if op is not None:
                op.update(modeTurnover=change["value"], turnover=change["value"], turnoverReason=change["reason"],
                          turnoverHorizon="1 prepared native bar", turnoverReferenceIndex=end - 2 if end > window else None)
        changes.append(per_operator)
    selected = [change[o["operator"]] for change in changes]
    return {"dates": [dates[end - 1] for end in ends], "indices": [end - 1 for end in ends], "results": results,
            "window": int(window), "step": int(step), "capped": capped, "candidateCount": len(candidates), "turnovers": changes,
            "turnover": [change["value"] for change in selected], "turnoverReasons": [change["reason"] for change in selected],
            "turnoverHorizon": "1 prepared native bar", "turnoverReferenceIndices": [end - 2 if end > window else None for end in ends],
            "schedule": "Requested regular step plus final endpoint; at most120 evenly spaced scheduled endpoints including final.",
            "interpretation": "Descriptive overlapping windows; annualization and empirical distributions are not forecasts."}
