"""Small, inspectable disk cache for closing-price matrices.

The first successful request writes prices.csv.gz plus a JSON manifest.
Identical later requests read those files without calling the downloader.
No database service, API account, or network call is built into this module.
For a large intraday archive, use partitioned Parquet instead of a wide CSV.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUEST_FIELDS = {
    "source", "symbols", "start", "end", "interval", "adjustment",
    "feed", "timezone_policy", "universe_as_of",
}

# CSV can preserve these values exactly when its reader is given the dtype.
# In particular, a nullable integer must not be inferred as float because NaN
# is present: large integers would then lose precision.
NUMERIC_DTYPES = {
    "float16", "float32", "float64", "Float32", "Float64",
    *[f"{prefix}{bits}" for prefix in ("int", "uint", "Int", "UInt")
      for bits in (8, 16, 32, 64)],
}


def _publish_snapshot(staging, target):
    """Publish within one cache root; briefly retry Windows sharing denials.

    Seven attempts wait at most 1.575 seconds in total. Existing destinations,
    non-Windows permission errors and every other failure propagate unchanged.
    """
    staging, target = Path(staging), Path(target)
    delays = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)
    for attempt in range(len(delays) + 1):
        # lexists also refuses a dangling symlink. Recheck after every wait.
        if os.path.lexists(target):
            raise FileExistsError(f"Snapshot destination already exists: {target}")
        if staging.resolve().parent != target.resolve().parent:
            raise ValueError("Snapshot publication must stay within one cache root.")
        try:
            os.rename(staging, target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def validate_prices(prices):
    """Reject ambiguous indexes/labels; keep genuine missing observations."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Expected a nonempty closing-price DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("The price index must be a DatetimeIndex.")
    if prices.index.hasnans or not prices.index.is_unique:
        raise ValueError("Price timestamps must be nonmissing and unique.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Sort timestamps before saving; do not silently reorder.")
    if prices.index.tz is not None:
        try:
            pd.DatetimeIndex([], tz=str(prices.index.tz))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("Convert this timezone to UTC before saving.") from error
    if not prices.columns.is_unique or not all(isinstance(c, str) for c in prices.columns):
        raise ValueError("Use unique string ticker labels and a single column level.")
    for dtype in prices.dtypes:
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            raise ValueError("Closing-price columns must contain real numeric values.")
        if str(dtype) not in NUMERIC_DTYPES:
            raise ValueError(
                f"Closing-price dtype {dtype} cannot be round-tripped by this cache. "
                "Use a NumPy integer/float dtype or pandas nullable numeric dtype."
            )
    if np.isinf(prices.to_numpy(dtype="float64", na_value=np.nan)).any():
        raise ValueError("Closing prices must not contain infinity; genuine NaN values are allowed.")
    return prices


def _canonical_request(request):
    if set(request) != REQUEST_FIELDS:
        raise ValueError(f"Request must contain exactly: {sorted(REQUEST_FIELDS)}")
    normalized = dict(request)
    symbols = normalized["symbols"]
    if not isinstance(symbols, (list, tuple)) or not symbols:
        raise ValueError("symbols must be a nonempty list of ticker strings.")
    if not all(isinstance(s, str) and s for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("Ticker strings must be nonempty and unique.")
    normalized["symbols"] = sorted(symbols)
    for key, value in normalized.items():
        if key != "symbols" and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a nonempty descriptive string.")
    # API credentials are deliberately not part of this schema or manifest.
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return normalized, hashlib.sha256(encoded.encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions():
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "yfinance"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def _check_manifest_summary(prices, manifest):
    """Check descriptive fields against the observations just loaded."""
    actual = {
        "rows": len(prices),
        "actual_first_timestamp": prices.index[0].isoformat(),
        "actual_last_timestamp": prices.index[-1].isoformat(),
        "missing_cells": int(prices.isna().sum().sum()),
    }
    for field, value in actual.items():
        if manifest.get(field) != value:
            raise ValueError(f"Cache manifest {field} no longer matches the price data.")


def cached_closes(request, fetch=None, cache_dir="data_cache"):
    """Return (prices, manifest, cache_hit) for one explicitly described request.

    fetch is a zero-argument function returning a closing-price DataFrame.
    It runs only on a cache miss. With fetch=None, this function is offline.
    The entire requested universe must be represented with some observations;
    missing cells are preserved, but silently failed tickers are rejected.
    A cached request is frozen: use a different explicit end date for new data.
    Do not delete a valid snapshot merely to refresh a chart.
    """
    normalized, key = _canonical_request(request)
    root = Path(cache_dir)
    target = root / key
    price_path = target / "prices.csv.gz"
    manifest_path = target / "manifest.json"

    if target.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["request"] != normalized or manifest["sha256"] != _file_hash(price_path):
            raise ValueError("Cache metadata or price checksum mismatch.")
        schema = manifest.get("schema_version")
        if schema not in (1, 2):
            raise ValueError(f"Unsupported cache schema version: {schema}")
        column_dtypes = None
        if schema == 2:
            column_dtypes = manifest.get("column_dtypes")
            if (
                not isinstance(column_dtypes, dict)
                or set(column_dtypes) != set(manifest["actual_symbols"])
                or not all(isinstance(dtype, str) and dtype in NUMERIC_DTYPES
                           for dtype in column_dtypes.values())
            ):
                raise ValueError("Cache numeric dtype metadata is missing or invalid.")
        prices = pd.read_csv(
            price_path, index_col=0, float_precision="round_trip", dtype=column_dtypes
        )
        saved_timezone = manifest["index_timezone"]
        prices.index = pd.to_datetime(
            prices.index, format="ISO8601", utc=saved_timezone is not None
        )
        if saved_timezone is not None:
            prices.index = prices.index.tz_convert(saved_timezone)
        prices.index.name = manifest["index_name"]
        prices.columns.name = manifest["columns_name"]
        validate_prices(prices)
        if list(prices.columns) != manifest["actual_symbols"]:
            raise ValueError("Cache ticker labels no longer match the manifest.")
        _check_manifest_summary(prices, manifest)
        return prices, manifest, True

    if fetch is None:
        raise FileNotFoundError("No matching snapshot. Supply fetch once, or copy the cache folder.")
    prices = validate_prices(fetch()).copy()
    missing_symbols = set(normalized["symbols"]) - set(prices.columns)
    extra_symbols = set(prices.columns) - set(normalized["symbols"])
    empty_symbols = prices.columns[prices.isna().all()].tolist()
    if missing_symbols or extra_symbols or empty_symbols:
        raise ValueError(
            f"Universe mismatch: missing={sorted(missing_symbols)}, "
            f"extra={sorted(extra_symbols)}, entirely empty={empty_symbols}. "
            "Resolve download failures or explicitly redefine the request."
        )
    root.mkdir(parents=True, exist_ok=True)
    # Publish the complete folder only after both files have been written.
    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=root) as temporary:
        staging = Path(temporary)
        staged_prices = staging / "prices.csv.gz"
        prices.to_csv(staged_prices, compression="gzip", float_format="%.17g")
        manifest = {
            "schema_version": 2,
            "request": normalized,
            "snapshot_saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "actual_first_timestamp": prices.index[0].isoformat(),
            "actual_last_timestamp": prices.index[-1].isoformat(),
            "actual_symbols": prices.columns.tolist(),
            "rows": len(prices),
            "missing_cells": int(prices.isna().sum().sum()),
            "index_timezone": str(prices.index.tz) if prices.index.tz is not None else None,
            "index_name": prices.index.name,
            "columns_name": prices.columns.name,
            "column_dtypes": {column: str(dtype) for column, dtype in prices.dtypes.items()},
            "package_versions": _versions(),
            "sha256": _file_hash(staged_prices),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _publish_snapshot(staging, target)
    return prices, manifest, False
