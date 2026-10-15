"""Small, explicit adapters for three public research-data sources.

Nothing downloads on import. Every fetch requires symbols and fixed bounds;
all adapter intervals use [start, end). Missing observations stay missing.
Prices from different venues, numeraires, sessions or adjustments belong in
separate panels. These observations alone do not establish a financial edge.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from data_cache import cached_closes, validate_prices


ECB_HISTORY_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
COINBASE_URL = "https://api.exchange.coinbase.com"
YAHOO_INTERVALS = {"1d", "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
COINBASE_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}


def _symbols(symbols):
    if not isinstance(symbols, (list, tuple)) or not symbols:
        raise ValueError("Pass a nonempty list of explicit symbols.")
    if not all(isinstance(symbol, str) and symbol.strip() == symbol and symbol for symbol in symbols):
        raise ValueError("Symbols must be nonempty strings without surrounding spaces.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Symbols must be unique.")
    return list(symbols)


def _bounds(start, end, *, daily):
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(first) or pd.isna(last):
        raise ValueError("Explicit nonmissing start and end are required.")
    if daily:
        if first.tzinfo is not None or last.tzinfo is not None:
            raise ValueError("Daily bounds must be timezone-naive calendar dates.")
        if first != first.normalize() or last != last.normalize():
            raise ValueError("Daily bounds must be midnight calendar dates.")
    else:
        if first.tzinfo is None or last.tzinfo is None:
            raise ValueError("Intraday bounds require explicit timezone offsets, such as Z.")
        first, last = first.tz_convert("UTC"), last.tz_convert("UTC")
    if first >= last:
        raise ValueError("start must precede the exclusive end.")
    return first, last


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _coverage(prices):
    """Count observed cells, rather than treating absent prices as zeros."""
    return {
        "rows": len(prices),
        "observed_cells": int(prices.notna().sum().sum()),
        "missing_cells": int(prices.isna().sum().sum()),
        "observations_by_symbol": {symbol: int(prices[symbol].notna().sum()) for symbol in prices},
        "missing_by_symbol": {symbol: int(prices[symbol].isna().sum()) for symbol in prices},
        "first_label": prices.index[0].isoformat(),
        "last_label": prices.index[-1].isoformat(),
    }


def _finish(prices, provenance):
    validate_prices(prices)
    observed = prices.to_numpy(dtype=float, na_value=np.nan)
    if (observed[np.isfinite(observed)] <= 0).any():
        raise ValueError("Observed prices/rates must be strictly positive.")
    provenance["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["package_versions"] = {name: _version(name) for name in ("pandas", "numpy", "requests")}
    provenance["coverage"] = _coverage(prices)
    encoded = prices.to_csv(float_format="%.17g").encode("utf-8")
    provenance["price_matrix_sha256"] = hashlib.sha256(encoded).hexdigest()
    return prices, provenance


def _get_bytes(url, *, params=None, timeout=30, get=None):
    """A failed HTTP response is an error, never a missing-price observation."""
    if get is None:
        import requests
        get = requests.get
    response = get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def _preserve_body(body, evidence_dir, suffix):
    """Keep the exact bytes under their SHA256 name when requested."""
    if evidence_dir is None:
        return None
    root = Path(evidence_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body).hexdigest()
    path = root / f"{digest}{suffix}"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("An existing evidence file does not match its hash filename.")
    else:
        with path.open("xb") as handle:
            handle.write(body)
    return str(path)


def fetch_yahoo_closes(symbols, start, end, *, auto_adjust, interval="1d",
                       daily_timezone="America/New_York", download=None, evidence_dir=None):
    """Fetch one Yahoo close panel with declared adjustment and fixed bounds.

    Daily labels are dates in one declared common market timezone. Use
    separate panels for exchanges with different session calendars/timezones.
    Intraday labels are UTC instants. Returned rows need not form a complete
    exchange calendar; the coverage summary counts provider-returned rows.
    A wholly failed symbol raises rather than silently changing the universe.
    """
    symbols = _symbols(symbols)
    if interval not in YAHOO_INTERVALS:
        raise ValueError(f"Supported Yahoo intervals: {sorted(YAHOO_INTERVALS)}")
    if not isinstance(auto_adjust, bool):
        raise TypeError("Declare auto_adjust as True or False.")
    daily = interval == "1d"
    first, last = _bounds(start, end, daily=daily)
    if daily:
        # Validate this before requesting data, even if a mock returns naive dates.
        pd.DatetimeIndex([], tz=daily_timezone)
    if download is None:
        import yfinance as yf
        download = yf.download
    settings = {
        "tickers": symbols, "start": first, "end": last, "interval": interval,
        "auto_adjust": auto_adjust, "back_adjust": False, "prepost": False,
        "repair": False, "keepna": True, "ignore_tz": False, "actions": False,
        "rounding": False, "threads": False, "progress": False,
        "group_by": "column", "multi_level_index": True,
    }
    data = download(**settings)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Yahoo returned no observations; inspect the download failure.")
    returned_frame = data.to_csv(float_format="%.17g").encode("utf-8")
    returned_frame_path = _preserve_body(returned_frame, evidence_dir, ".yfinance-frame.csv")
    if isinstance(data.columns, pd.MultiIndex):
        if data.columns.nlevels != 2 or "Close" not in data.columns.get_level_values(0):
            raise ValueError("Expected Yahoo columns in (price field, ticker) order.")
        prices = data["Close"].copy()
    elif len(symbols) == 1 and "Close" in data:
        prices = data[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        raise ValueError("Yahoo returned an unexpected Close-column schema.")
    if set(prices.columns) != set(symbols) or not prices.columns.is_unique:
        raise ValueError("Yahoo returned a different or duplicate symbol universe.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Yahoo timestamps must be a nonmissing DatetimeIndex.")
    if daily:
        if prices.index.tz is not None:
            prices.index = prices.index.tz_convert(daily_timezone).tz_localize(None)
        prices.index = prices.index.normalize()
    else:
        if prices.index.tz is None:
            raise ValueError("Yahoo intraday timestamps lost their timezone.")
        prices.index = prices.index.tz_convert("UTC")
    prices = prices.loc[(prices.index >= first) & (prices.index < last), symbols]
    if prices.empty or prices.isna().all().any():
        raise ValueError("Yahoo has no in-range observations for at least one requested symbol.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Yahoo returned duplicate or unsorted price timestamps.")
    prices.index.name, prices.columns.name = "date" if daily else "timestamp", "symbol"
    provenance = {
        "source": "Yahoo via yfinance", "source_url": "https://query1.finance.yahoo.com/v8/finance/chart/",
        "symbols": symbols, "start": first.isoformat(), "end_exclusive": last.isoformat(),
        "interval": interval, "auto_adjust": auto_adjust,
        "adjustment": "yfinance auto_adjust=True" if auto_adjust else "provider Close; auto_adjust=False",
        "feed": "Yahoo public interface", "prepost": False, "repair": False,
        "keepna": True, "ignore_tz": False,
        "timezone_policy": f"naive daily dates in {daily_timezone}" if daily else "UTC instants",
        "timestamp_semantics": "daily session-date labels" if daily else "provider bar timestamps; verify session/bucket convention",
        "raw_response_sha256": None,
        "raw_response_note": "yfinance.download does not expose raw HTTP bodies; price_matrix_sha256 hashes its returned Close panel",
        "returned_frame_sha256": hashlib.sha256(returned_frame).hexdigest(),
        "returned_frame_path": returned_frame_path,
        "returned_frame_note": "Full library-returned price-field/ticker DataFrame serialized as CSV; not raw HTTP evidence.",
        "yfinance_version": _version("yfinance"),
        "limitations": "Unofficial personal-research interface; recent intraday retention and coverage depend on Yahoo and package version.",
    }
    return _finish(prices, provenance)


def fetch_ecb_rates(currencies, start, end, *, get=None, evidence_dir=None):
    """Filter the official small daily archive to currencies and [start, end).

    Values are units of foreign currency per EUR, not transaction quotes.
    EUR itself is the numeraire and is deliberately not a constant asset column.
    TARGET holidays/weekends have no fabricated observations.
    """
    currencies = _symbols(currencies)
    if any(len(currency) != 3 or not currency.isalpha() or currency != currency.upper() for currency in currencies):
        raise ValueError("Use explicit three-letter uppercase currency codes.")
    if "EUR" in currencies:
        raise ValueError("EUR is the base numeraire, not a varying reference-rate series.")
    first, last = _bounds(start, end, daily=True)
    body = _get_bytes(ECB_HISTORY_URL, get=get)
    evidence_path = _preserve_body(body, evidence_dir, ".ecb.zip")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        expected = "eurofxref-hist.csv"
        if expected not in archive.namelist():
            raise ValueError("The ECB ZIP lacks eurofxref-hist.csv.")
        csv_body = archive.read(expected)
    table = pd.read_csv(io.BytesIO(csv_body), na_values=["N/A", "-"])
    table.columns = table.columns.str.strip()
    if not table.columns.is_unique or "Date" not in table:
        raise ValueError("The ECB archive has invalid column labels.")
    missing = set(currencies) - set(table.columns)
    if missing:
        raise ValueError(f"Currencies missing from ECB archive: {sorted(missing)}")
    dates = pd.to_datetime(table["Date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("ECB reference dates must be nonmissing and unique.")
    prices = table[currencies].apply(pd.to_numeric, errors="raise")
    prices.index = pd.DatetimeIndex(dates, name="date")
    prices = prices.loc[(prices.index >= first) & (prices.index < last)].sort_index()
    if prices.empty:
        raise ValueError("The ECB archive contains no reference dates inside the request.")
    prices.columns.name = "currency"
    return _finish(prices, {
        "source": "European Central Bank reference rates", "source_url": ECB_HISTORY_URL,
        "symbols": currencies, "start": first.isoformat(), "end_exclusive": last.isoformat(),
        "interval": "1d", "base_currency": "EUR", "quote_convention": "foreign-currency units per EUR",
        "adjustment": "not applicable: official reference rates", "feed": "ECB eurofxref-hist.csv",
        "timezone_policy": "naive reference dates; publication around 16:00 CET, not midnight quotes",
        "timestamp_semantics": "reference date; usual concertation around 14:10 CET",
        "raw_response_sha256": hashlib.sha256(body).hexdigest(),
        "raw_evidence_path": evidence_path,
        "archive_member_sha256": hashlib.sha256(csv_body).hexdigest(),
        "limitations": "Informational daily rates, not executable quotes; no intraday observations; series coverage can differ.",
    })


def _parse_coinbase_candles(body, granularity):
    """Validate one raw response before distinguishing absent buckets from errors."""
    payload = json.loads(body)
    if not isinstance(payload, list) or len(payload) > 300:
        raise ValueError("Expected a Coinbase candle array with at most 300 rows.")
    records = []
    for candle in payload:
        if not isinstance(candle, list) or len(candle) != 6:
            raise ValueError("A Coinbase candle must contain time, low, high, open, close, volume.")
        if any(isinstance(value, bool) for value in candle):
            raise ValueError("Boolean values are not valid candle numbers.")
        values = np.asarray(candle, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Coinbase candle values must be finite.")
        second, low, high, opening, close, volume = values
        if second != int(second) or int(second) % granularity:
            raise ValueError("Coinbase candle starts must align to the declared UTC grid.")
        if min(low, high, opening, close) <= 0 or volume < 0:
            raise ValueError("Coinbase prices must be positive and volume nonnegative.")
        if not low <= min(opening, close) <= max(opening, close) <= high:
            raise ValueError("Coinbase OHLC values violate low/high bounds.")
        records.append((int(second), close))
    if records:
        seconds = [record[0] for record in records]
        if len(set(seconds)) != len(seconds):
            raise ValueError("Coinbase returned duplicate candle starts.")
        if seconds != sorted(seconds) and seconds != sorted(seconds, reverse=True):
            raise ValueError("Coinbase returned inconsistent candle ordering.")
    return records


def fetch_coinbase_closes(products, start, end, *, granularity=300, get=None, max_requests=1000, evidence_dir=None):
    """Fetch public Exchange candles; retain absent buckets on a declared UTC grid.

    Bounds must align to whole candle buckets. Labels are candle START times;
    a close is available only after its bucket ends. Each request spans at
    most 300 buckets. Products and the full range must be explicit; no endpoint
    falls back to another venue or silently reduces the product set.
    """
    products = _symbols(products)
    if any(not product or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for char in product) for product in products):
        raise ValueError("Use explicit uppercase Coinbase Exchange product IDs.")
    if isinstance(granularity, bool) or not isinstance(granularity, (int, np.integer)) or granularity not in COINBASE_GRANULARITIES:
        raise ValueError(f"Coinbase granularity must be one of {sorted(COINBASE_GRANULARITIES)} seconds.")
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
        raise ValueError("max_requests must be a positive integer.")
    first, last = _bounds(start, end, daily=False)
    bucket_ns = int(granularity) * 1_000_000_000
    if first.value % bucket_ns or last.value % bucket_ns:
        raise ValueError("Coinbase start and end must align to the declared UTC candle grid.")
    slots = (last.value - first.value) // bucket_ns
    request_count = len(products) * ((slots + 299) // 300)
    if request_count > max_requests:
        raise ValueError(f"This request needs {request_count} calls, exceeding max_requests={max_requests}.")
    grid = pd.date_range(first, last, freq=pd.Timedelta(seconds=int(granularity)), inclusive="left", name="timestamp")
    prices = pd.DataFrame(np.nan, index=grid, columns=products)
    responses, combined_hash = [], hashlib.sha256()
    for product in products:
        cursor = first
        while cursor < last:
            stop = min(last, cursor + pd.Timedelta(seconds=300 * int(granularity)))
            url = f"{COINBASE_URL}/products/{quote(product, safe='')}/candles"
            params = {"start": cursor.isoformat(), "end": stop.isoformat(), "granularity": int(granularity)}
            body = _get_bytes(url, params=params, get=get)
            evidence_path = _preserve_body(body, evidence_dir, ".coinbase.json")
            records = _parse_coinbase_candles(body, int(granularity))
            # The API may return candles before start. Trim each disjoint block
            # exactly, including the exclusive end, before writing to the grid.
            for second, close in records:
                timestamp = pd.Timestamp(second, unit="s", tz="UTC")
                if cursor <= timestamp < stop:
                    prices.loc[timestamp, product] = close
            responses.append({"url": url, "parameters": params, "sha256": hashlib.sha256(body).hexdigest(), "raw_evidence_path": evidence_path})
            combined_hash.update(len(body).to_bytes(8, "big"))
            combined_hash.update(body)
            cursor = stop
    prices.columns.name = "product"
    return _finish(prices, {
        "source": "Coinbase Exchange public candles", "source_url": COINBASE_URL,
        "symbols": products, "start": first.isoformat(), "end_exclusive": last.isoformat(),
        "interval": f"{int(granularity)}s", "granularity_seconds": int(granularity),
        "adjustment": "unadjusted venue spot candles", "feed": "Coinbase Exchange",
        "timezone_policy": "UTC", "timestamp_semantics": "candle START; close available after start plus granularity",
        "raw_response_sha256": combined_hash.hexdigest(),
        "raw_hash_policy": "request-ordered response bodies, each prefixed by its 8-byte big-endian length",
        "raw_responses": responses, "expected_slots_per_product": len(grid),
        "missing_interval_semantics": "absent from response; no ticks and incomplete history cannot be distinguished here",
        "limitations": "Venue-specific spot data; historical candles may be incomplete; absent no-tick buckets remain NaN.",
    })


def make_source_request(provenance, *, universe_as_of):
    """Build the existing cache's nine fields from actual adapter semantics.

    universe_as_of describes the explicit research list, not a fabricated
    historical index-membership date. This is useful after a bounded fetch;
    an identical dictionary can later read that frozen cache without a fetch.
    """
    return {
        "source": provenance["source"], "symbols": list(provenance["symbols"]),
        "start": provenance["start"], "end": provenance["end_exclusive"],
        "interval": provenance["interval"], "adjustment": provenance["adjustment"],
        "feed": provenance["feed"], "timezone_policy": provenance["timezone_policy"],
        "universe_as_of": universe_as_of,
    }


def _validate_source_request(request, provenance):
    if not isinstance(provenance, dict):
        raise ValueError("The source adapter must return a provenance dictionary.")
    if set(provenance.get("symbols", [])) != set(request["symbols"]):
        raise ValueError("Source provenance differs from the requested universe.")
    expected = make_source_request(provenance, universe_as_of=request["universe_as_of"])
    for field in ("source", "interval", "adjustment", "feed", "timezone_policy"):
        if request[field] != expected[field]:
            raise ValueError(f"Source provenance differs from request field: {field}.")
    for field in ("start", "end"):
        if pd.Timestamp(request[field]) != pd.Timestamp(expected[field]):
            raise ValueError(f"Source provenance differs from request bound: {field}.")


def save_source_snapshot(request, fetch=None, *, cache_dir="data_cache"):
    """Reuse cached_closes and keep adapter evidence in provenance.json beside it.

    fetch is a zero-argument function returning (prices, provenance). On a
    frozen cache hit it is never called. Return prices, manifest, provenance,
    cache_hit. Missing or mismatched companion evidence raises explicitly.
    """
    captured = {}

    def fetch_prices():
        prices, provenance = fetch()
        _validate_source_request(request, provenance)
        # Check serializability before cached_closes publishes a price folder.
        json.dumps(provenance, allow_nan=False)
        captured.update(provenance)
        return prices

    prices, manifest, hit = cached_closes(request, fetch_prices if fetch is not None else None, cache_dir)
    encoded = json.dumps(manifest["request"], sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(encoded.encode()).hexdigest()
    companion = Path(cache_dir) / key / "provenance.json"
    if hit:
        if not companion.exists():
            raise ValueError("This frozen cache lacks provenance.json; recover its evidence rather than redownloading silently.")
        provenance = json.loads(companion.read_text(encoding="utf-8"))
        if provenance.get("cache_request") != manifest["request"] or provenance.get("price_file_sha256") != manifest["sha256"]:
            raise ValueError("Source provenance no longer matches the frozen price snapshot.")
        _validate_source_request(manifest["request"], provenance)
    else:
        provenance = dict(captured, cache_request=manifest["request"], price_file_sha256=manifest["sha256"])
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=companion.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(provenance, handle, indent=2, allow_nan=False)
        try:
            os.replace(temporary, companion)
        finally:
            if temporary.exists():
                temporary.unlink()
    return prices, manifest, provenance, hit
