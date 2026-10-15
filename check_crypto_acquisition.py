"""Offline meaningful adapter, bounds, identity and resampling checks."""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

import crypto_acquisition as ca


def fails(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Expected explicit validation failure")


def synthetic_archive(folder, month, interval="1h", bad_duration=False, outside=False, one_partial=False):
    unit = "us" if month >= "2025-01" else "ms"
    quantum = pd.Timedelta(microseconds=1) if unit == "us" else pd.Timedelta(milliseconds=1)
    start = pd.Timestamp(month + "-01", tz="UTC")
    if outside:
        start -= pd.Timedelta(days=1)
    rows = []
    for i in range(3):
        opened = start + i * ca.INTERVALS[interval]
        closed = opened + ca.INTERVALS[interval] - quantum
        if bad_duration or (one_partial and i == 1):
            closed -= pd.Timedelta(seconds=1)
        divisor = 1000 if unit == "us" else 1_000_000
        rows.append([opened.value // divisor, 10, 12, 9, 11 + .1 * i, 2,
                     closed.value // divisor, 20, 3, 1, 10, 0])
    path = folder / f"BTCUSDT-{interval}-{month}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", "\n".join(",".join(map(str, r)) for r in rows))
    return path


def check_units_and_bounds():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for month in ("2024-12", "2025-01"):
            path = synthetic_archive(folder, month)
            prices, metadata = ca.parse_archive(path, "BTCUSDT", "1h", month)
            assert prices.index[0] == pd.Timestamp(month + "-01 01:00", tz="UTC")
            assert metadata["timestamp_unit"] == ("us" if month >= "2025-01" else "ms")
            assert prices.tolist() == [11., 11.1, 11.2]
        path = synthetic_archive(folder, "2025-01", bad_duration=True)
        fails(ca.parse_archive, path, "BTCUSDT", "1h", "2025-01")
        path = synthetic_archive(folder, "2025-01", outside=True)
        fails(ca.parse_archive, path, "BTCUSDT", "1h", "2025-01")
        path = synthetic_archive(folder, "2025-01", one_partial=True)
        prices, metadata = ca.parse_archive(path, "BTCUSDT", "1h", "2025-01")
        assert prices.iloc[0] == 11 and pd.isna(prices.iloc[1]) and prices.iloc[2] == 11.2
        assert metadata["nonstandard_duration_bars_masked"] == 1
    print("PASS archive timestamp units, bar end, duration and month bounds")


def check_daily():
    grid = pd.date_range("2025-01-01 01:00", periods=48, freq="h", tz="UTC")
    hourly = pd.DataFrame({"A": np.arange(48, dtype=float) + 10,
                           "B": np.arange(48, dtype=float) + 20}, index=grid)
    hourly.iloc[3, 0] = np.nan
    daily = ca.hourly_to_daily(hourly)
    assert daily.index.tolist() == [pd.Timestamp("2025-01-02", tz="UTC"),
                                   pd.Timestamp("2025-01-03", tz="UTC")]
    assert pd.isna(daily.iloc[0, 0]) and daily.iloc[0, 1] == 43
    assert daily.iloc[1].tolist() == [57., 67.]
    assert not hourly.isna().sum().sum() == 0
    print("PASS daily UTC resampling requires all 24 observed hours without fill")


def check_identity():
    ranking = [{"id": f"coin{i}", "symbol": f"S{i}", "name": f"Coin {i}",
                "market_cap_rank": i, "market_cap": 1000-i,
                "last_updated": "2026-10-09T00:00:00Z"} for i in range(1, 101)]
    ranking[0]["symbol"] = "USDT"
    ranking[1]["symbol"] = "USDC"
    ranking[2]["symbol"] = ranking[3]["symbol"] = "DUP"
    exchange = {"symbols": [{"baseAsset": "USDC", "quoteAsset": "USDT", "symbol": "USDCUSDT",
                             "isSpotTradingAllowed": True, "status": "TRADING"},
                            {"baseAsset": "DUP", "quoteAsset": "USDT", "symbol": "DUPUSDT",
                             "isSpotTradingAllowed": True, "status": "TRADING"}]}
    universe = ca.map_universe(ranking, exchange)
    assert len(universe) == 100
    assert universe[0]["status"] == "quote_asset_excluded"
    assert universe[1]["status"] == "eligible" and universe[1]["stablecoin_symbol_flag"]
    assert universe[2]["status"] == universe[3]["status"] == "ambiguous_symbol_excluded"
    assert universe[4]["status"] == "no_exact_spot_usdt_pair"
    fails(ca.map_universe, ranking[:99], exchange)
    print("PASS exact ranked IDs, unavailable distinctions, quote and ambiguous symbols")


def check_ten_minutes():
    index = pd.date_range("2025-01-01 00:05", periods=6, freq="5min", tz="UTC")
    frame = pd.DataFrame({"A": [10., 11., np.nan, 13., 14., 15.]}, index=index)
    ten = ca.five_to_ten_minutes(frame)
    assert ten.index.tolist() == list(pd.date_range("2025-01-01 00:10", periods=3, freq="10min", tz="UTC"))
    assert ten.iloc[0, 0] == 11 and pd.isna(ten.iloc[1, 0]) and ten.iloc[2, 0] == 15
    assert frame.iloc[2].isna().all()
    print("PASS ten-minute close requires both child bars, preserves a missing child")


def check_frozen_hash():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "raw.json"
        blob = b'{"prices":[1,2]}'
        meta = {"url": "https://example.org/public", "sha256": hashlib.sha256(blob).hexdigest()}
        with patch.object(ca, "public_get", return_value=(blob, meta)) as getter:
            assert ca.evidence(path, meta["url"])[0] == blob
            assert ca.evidence(path, meta["url"])[0] == blob
            assert getter.call_count == 1
            path.write_bytes(b"corrupted")
            fails(ca.evidence, path, meta["url"])
        plan = {"listing_failures": [], "jobs": [{"size": 100}]}
        fails(lambda: ca.acquire(Path(directory), plan, max_raw_bytes=50))
    print("PASS frozen raw reuse, checksum corruption rejection and byte budget guard")


def check_cross_provider_identity():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        ca.write_json(folder / "universe.json", {"coins": [
            {"status": "eligible", "venue_symbol": "SKYUSDT", "coin_id": "sky", "symbol": "SKY", "name": "Sky"},
            {"status": "eligible", "venue_symbol": "LITUSDT", "coin_id": "lighter", "symbol": "LIT", "name": "Lighter"}]})
        calls = []
        def mock_evidence(path, url):
            calls.append(url)
            if "coinpaprika" in url:
                payload = [{"pair": "SKY/USDT", "category": "Spot", "base_currency_name": "Skycoin", "base_currency_id": "sky-skycoin"}]
            else:
                payload = {"name": "Sky", "tickers": [{"base": "SKY", "target": "USDT", "coin_id": "sky", "market": {"identifier": "binance"}}]}
            return json.dumps(payload).encode(), {"url": url, "sha256": "mock"}
        with patch.object(ca, "evidence", side_effect=mock_evidence), patch.object(ca.time, "sleep"):
            result = ca.verify_identities(folder, {})
        assert result["trusted_symbols"] == ["SKYUSDT"]
        assert len(calls) == 2  # Skycoin mismatch needed a distinct current coin check.
        assert next(x for x in result["records"] if x["venue_symbol"] == "LITUSDT")["status"] == "different_historical_asset_excluded"
    print("PASS cross-provider name mismatch requires identity evidence; legacy LIT excluded")


def check_reused_ticker_bound():
    index = pd.DatetimeIndex(["2019-04-01 01:00+00:00", "2026-04-09 08:00+00:00", "2026-04-09 09:00+00:00"])
    source = pd.Series([1., 1., 1.], index=index)
    result, count = ca.apply_identity_bounds(source, "USDSUSDT", "1h")
    assert result.iloc[:2].isna().all() and result.iloc[2] == 1 and count == 2
    assert source.notna().all()  # Frozen original observations are preserved.
    result, count = ca.apply_identity_bounds(source, "BTCUSDT", "1h")
    assert result.equals(source) and count == 0
    print("PASS documented USDS ticker reuse excludes legacy closes at exact bar-start bound")


def check_final_registry():
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        root = project / "data" / "expanded" / "crypto"
        ca.write_json(root / "identity_review.json", {"trusted_symbols": ["BTCUSDT"]})
        datasets = []
        for interval in ("1m", "5m", "10m", "1h", "1d"):
            request = {"source": "binance_vision_spot_archive", "symbols": ["BTCUSDT"],
                       "start": "2025-01-01", "end": "2025-01-02", "interval": interval,
                       "adjustment": "none; legacy-USDS bars masked v4", "feed": "spot",
                       "timezone_policy": "UTC bar END", "universe_as_of": "2025-01-01"}
            folder = root / "snapshots" / ca.cache_key(request)
            ca.write_json(folder / "manifest.json", {"request": request})
            ca.write_json(root / f"coverage_{interval}.json",
                          {"request": request, "snapshot_dir": str(folder),
                           "source_failures": [], "exact_cache_reload_verified": True})
            datasets.append({"dataset_id": "crypto_binance_current_top100_" + interval,
                             "interval": interval, "snapshot_path": str(folder), "notes": []})
        old_request = {**request, "adjustment": "none; bars masked v3"}
        old_folder = root / "snapshots" / ca.cache_key(old_request)
        ca.write_json(old_folder / "manifest.json", {"request": old_request})
        index_path = project / "output" / "expansion" / "crypto_results.json"
        ca.write_json(index_path, {"datasets": datasets})
        with patch.object(ca, "ROOT", project):
            result = ca.finalize_index(root)
            assert result["finalized"]
            assert {x["dataset_id"] for x in result["datasets"] if x["interval"] in ("1h", "1d")} == {
                "crypto_binance_current_top100_1h_identity_v4", "crypto_binance_current_top100_1d_identity_v4"}
            assert result["superseded"][0]["snapshot_path"] == str(old_folder)
            result["datasets"][0]["snapshot_path"] = str(old_folder)
            ca.write_json(index_path, result)
            result = ca.finalize_index(root)
            assert not result["finalized"] and not result["datasets"][0]["finalized"]
    print("PASS corrected input IDs, immutable superseded references and request/path finality binding")


if __name__ == "__main__":
    check_units_and_bounds()
    check_daily()
    check_identity()
    check_ten_minutes()
    check_frozen_hash()
    check_cross_provider_identity()
    check_reused_ticker_bound()
    check_final_registry()
    print("8 crypto acquisition check groups passed; no network or market downloads")
