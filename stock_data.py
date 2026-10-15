"""Read local DAILY stock prices and form arithmetic returns, without downloads.

Input is a CSV/CSV.gz with dates in its first column, or a frozen snapshot
directory containing prices.csv.gz and manifest.json. Missing prices remain
missing. Date labels represent source sessions, not verified publication times.
"""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stock_prices(prices):
    """Require ordered daily dates, unique symbols and positive observed prices."""
    if (
        not isinstance(prices, pd.DataFrame)
        or len(prices) < 2
        or not len(prices.columns)
    ):
        raise ValueError("Supply at least two price rows and one stock column.")
    index = prices.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or index.tz is not None
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError(
            "Use unique increasing timezone-naive daily session-date labels."
        )
    if not prices.columns.is_unique or not all(
        isinstance(x, str) and x.strip() for x in prices.columns
    ):
        raise ValueError("Stock labels must be unique nonempty strings.")
    for dtype in prices.dtypes:
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError(
                "Prices must be real numeric values or genuine missing observations."
            )
    array = prices.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any() or ((array <= 0) & np.isfinite(array)).any():
        raise ValueError(
            "Observed stock prices must be finite and strictly positive; missing values may remain NaN."
        )
    return prices


def load_stock_prices(path, expected_sha256=None):
    """Return (prices, portable metadata); validate supplied snapshot metadata.

    A matching hash verifies bytes, not price accuracy, adjustment history or
    historical membership. A plain user CSV has no implied provider provenance.
    """
    path = Path(path)
    price_file = path / "prices.csv.gz" if path.is_dir() else path
    manifest_file = price_file.parent / "manifest.json"
    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.is_file()
        else None
    )
    actual_hash = file_sha256(price_file)
    for expected in (expected_sha256, manifest.get("sha256") if manifest else None):
        if expected is not None and actual_hash != expected:
            raise ValueError("Stock price-file checksum mismatch.")
    opener = gzip.open if price_file.suffix == ".gz" else open
    with opener(price_file, "rt", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    symbols = header[1:]
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not x.strip() for x in symbols)
    ):
        raise ValueError("CSV headers must declare unique nonempty stock symbols.")
    prices = pd.read_csv(price_file, index_col=0, float_precision="round_trip")
    prices.index = pd.to_datetime(prices.index, format="ISO8601", errors="raise")
    validate_stock_prices(prices)
    if manifest:
        expected = {
            "rows": len(prices),
            "actual_symbols": list(prices.columns),
            "actual_first_timestamp": prices.index[0].isoformat(),
            "actual_last_timestamp": prices.index[-1].isoformat(),
            "missing_cells": int(prices.isna().sum().sum()),
        }
        for key, value in expected.items():
            if key in manifest and manifest[key] != value:
                raise ValueError("Snapshot manifest mismatch: " + key)
        if manifest.get("index_timezone") is not None:
            raise ValueError("Expected daily session dates, not timezone-aware bars.")
    request = manifest.get("request", {}) if manifest else {}
    metadata = {
        "price_sha256": actual_hash,
        "file_name": price_file.name,
        "price_rows": len(prices),
        "symbols": list(prices.columns),
        "symbol_count": len(prices.columns),
        "first_price_date": prices.index[0].isoformat(),
        "last_price_date": prices.index[-1].isoformat(),
        "observed_price_cells": int(prices.notna().sum().sum()),
        "missing_price_cells": int(prices.isna().sum().sum()),
        "source": request.get(
            "source", "User-supplied local CSV; source not documented"
        ),
        "adjustment": request.get("adjustment", "Not documented"),
        "universe_as_of": request.get("universe_as_of", "Not documented"),
        "timestamp_policy": "Source-native daily session-date labels; not verified publication timestamps.",
        "fill_policy": "No zero/forward/interpolating fill",
        "manifest_checked": manifest is not None,
        "rights": "Raw inputs remain local; obtain appropriate provider permission for your own use.",
    }
    return prices, metadata


def arithmetic_returns(prices):
    """Adjacent native sessions, including Friday/Monday; no calendar-day mask."""
    validate_stock_prices(prices)
    return (
        prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).iloc[1:]
    )
