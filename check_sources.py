"""Mocked adapter checks; no provider downloads. Run: python check_sources.py."""

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from market_sources import fetch_coinbase_closes, fetch_ecb_rates, fetch_yahoo_closes, make_source_request, save_source_snapshot


class Response:
    def __init__(self, content, *, failed=False):
        self.content = content
        self.failed = failed

    def raise_for_status(self):
        if self.failed:
            raise RuntimeError("Mock HTTP failure")


def must_raise(call, expected=ValueError):
    try:
        call()
    except expected:
        return
    raise AssertionError(f"Expected {expected.__name__}")


def check_yahoo_dates_timezone_and_flags():
    calls = []

    def download(**settings):
        calls.append(settings)
        index = pd.date_range("2025-01-02", periods=3, tz="America/New_York").tz_convert("UTC")
        columns = pd.MultiIndex.from_product([["Close"], ["SPY", "QQQ"]])
        return pd.DataFrame([[1., 2.], [3., np.nan], [5., 6.]], index=index, columns=columns)

    daily, evidence = fetch_yahoo_closes(["QQQ", "SPY"], "2025-01-02", "2025-01-04", auto_adjust=True, download=download)
    assert daily.index.tz is None and daily.columns.tolist() == ["QQQ", "SPY"]
    assert daily.index.tolist() == pd.date_range("2025-01-02", periods=2).tolist()
    assert np.isnan(daily.iloc[1, 0]) and evidence["coverage"]["missing_cells"] == 1
    assert evidence["raw_response_sha256"] is None
    for field, value in {"auto_adjust": True, "prepost": False, "repair": False, "keepna": True, "ignore_tz": False}.items():
        assert calls[0][field] is value

    def intraday(**settings):
        return pd.DataFrame({"Close": [10., 11., 12.]}, index=pd.date_range("2025-01-02 09:30", periods=3, freq="5min", tz="America/New_York"))

    prices, _ = fetch_yahoo_closes(["SPY"], "2025-01-02T14:30:00Z", "2025-01-02T14:40:00Z", interval="5m", auto_adjust=False, download=intraday)
    assert str(prices.index.tz) == "UTC" and len(prices) == 2
    assert prices.iloc[-1, 0] == 11.
    must_raise(lambda: fetch_yahoo_closes(["SPY"], "2025-01-02", "2025-01-03", interval="5m", auto_adjust=False, download=intraday))


def check_ecb_reference_dates_numeraire_and_checksum():
    stream = io.BytesIO()
    csv = b"Date,USD,GBP,\n2025-01-06,1.04,0.83,\n2025-01-03,1.03,N/A,\n2025-01-02,1.02,0.82,\n"
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("eurofxref-hist.csv", csv)
    body = stream.getvalue()
    get = lambda *args, **kwargs: Response(body)
    prices, provenance = fetch_ecb_rates(["GBP", "USD"], "2025-01-02", "2025-01-06", get=get)
    assert prices.index.tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert prices.loc["2025-01-02", "USD"] == 1.02  # Units per EUR, not its inverse.
    assert np.isnan(prices.loc["2025-01-03", "GBP"])
    assert provenance["base_currency"] == "EUR"
    assert provenance["raw_response_sha256"] == hashlib.sha256(body).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        _, evidence = fetch_ecb_rates(["USD"], "2025-01-02", "2025-01-06", get=get, evidence_dir=directory)
        assert Path(evidence["raw_evidence_path"]).read_bytes() == body
    must_raise(lambda: fetch_ecb_rates(["EUR"], "2025-01-02", "2025-01-06", get=get))
    must_raise(lambda: fetch_ecb_rates(["XXX"], "2025-01-02", "2025-01-06", get=get))


def candle(second, close=10.):
    return [second, 9., 12., 10., close, 5.]


def check_coinbase_grid_missing_buckets_and_invalid_responses():
    first = pd.Timestamp("2025-01-01T00:00:00Z")
    second = int(first.timestamp())
    # Descending is a valid provider order. End and pre-start candles are trimmed.
    payload = [candle(second + 900), candle(second + 600, 11.), candle(second), candle(second - 300)]
    get = lambda *args, **kwargs: Response(json.dumps(payload).encode())
    prices, provenance = fetch_coinbase_closes(["BTC-USD"], first, first + pd.Timedelta(minutes=15), get=get)
    assert len(prices) == 3 and str(prices.index.tz) == "UTC"
    assert prices.iloc[0, 0] == 10. and np.isnan(prices.iloc[1, 0]) and prices.iloc[2, 0] == 11.
    assert provenance["coverage"]["missing_cells"] == 1
    assert "START" in provenance["timestamp_semantics"]
    empty_get = lambda *args, **kwargs: Response(b"[]")
    empty, _ = fetch_coinbase_closes(["BTC-USD"], first, first + pd.Timedelta(minutes=5), get=empty_get)
    assert empty.isna().all().all()  # Valid no-tick response, not a manufactured zero return.
    invalid_payloads = [
        {"message": "provider failure"},
        [candle(second), candle(second)],
        [candle(second), candle(second + 600), candle(second + 300)],
        [[second, 9., 12., 10., 13., 5.]],
        [[second, 9., 12., 10., 10., -1.]],
        [[second + 1, 9., 12., 10., 10., 5.]],
    ]
    for payload in invalid_payloads:
        mock = lambda *args, **kwargs: Response(json.dumps(payload).encode())
        must_raise(lambda: fetch_coinbase_closes(["BTC-USD"], first, first + pd.Timedelta(minutes=15), get=mock))
    failed = lambda *args, **kwargs: Response(b"[]", failed=True)
    must_raise(lambda: fetch_coinbase_closes(["BTC-USD"], first, first + pd.Timedelta(minutes=5), get=failed), RuntimeError)


def check_coinbase_request_splitting_and_cache_companion():
    calls = []
    first = pd.Timestamp("2025-01-01T00:00:00Z")
    last = first + pd.Timedelta(minutes=301)

    def get(url, *, params, timeout):
        calls.append(params)
        start, end = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
        assert (end - start).total_seconds() / params["granularity"] <= 300
        payload = [candle(int(start.timestamp()))]
        return Response(json.dumps(payload).encode())

    request = {
        "source": "Coinbase Exchange public candles", "symbols": ["BTC-USD"],
        "start": first.isoformat(), "end": last.isoformat(), "interval": "60s",
        "adjustment": "unadjusted venue spot candles", "feed": "Coinbase Exchange",
        "timezone_policy": "UTC", "universe_as_of": "explicit pilot list",
    }
    fetch = lambda: fetch_coinbase_closes(["BTC-USD"], first, last, granularity=60, get=get)
    with tempfile.TemporaryDirectory() as directory:
        prices, _, provenance, hit = save_source_snapshot(request, fetch, cache_dir=directory)
        assert not hit and len(calls) == 2 and len(prices) == 301
        assert prices.notna().sum().iloc[0] == 2 and len(provenance["raw_responses"]) == 2
        assert make_source_request(provenance, universe_as_of="explicit pilot list") == request
        reused, _, repeated, hit = save_source_snapshot(request, cache_dir=directory)
        assert hit and len(calls) == 2 and repeated == provenance
        pd.testing.assert_frame_equal(prices, reused, check_freq=False)
        companion = next(Path(directory).glob("*/provenance.json"))
        for field, value in (("feed", "another venue"), ("end_exclusive", (last + pd.Timedelta(minutes=1)).isoformat())):
            tampered = dict(provenance, **{field: value})
            companion.write_text(json.dumps(tampered), encoding="utf-8")
            must_raise(lambda: save_source_snapshot(request, cache_dir=directory))
            companion.write_text(json.dumps(provenance), encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        wrong_request = dict(request, end=(last + pd.Timedelta(minutes=1)).isoformat())
        must_raise(lambda: save_source_snapshot(wrong_request, fetch, cache_dir=directory))
    must_raise(lambda: fetch_coinbase_closes(["BTC-USD"], first, last, granularity=60, get=get, max_requests=1))


if __name__ == "__main__":
    checks = [check_yahoo_dates_timezone_and_flags, check_ecb_reference_dates_numeraire_and_checksum,
              check_coinbase_grid_missing_buckets_and_invalid_responses, check_coinbase_request_splitting_and_cache_companion]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"All {len(checks)} source checks passed. No provider data was downloaded.")
