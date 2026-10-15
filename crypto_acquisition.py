"""Free, resumable Binance spot archives for a frozen current top-100 universe.

No accounts, API keys, trades, cross-venue substitution or missing-price fills.
Raw ZIPs, publisher checksums and HTTP evidence stay on disk. Cached prices are
indexed by bar END (minimum time the close can be known), not candle start.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from data_cache import cached_closes, _canonical_request

ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "data" / "expanded" / "crypto"
VISION = "https://data.binance.vision/"
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
RANKING_URL = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
               "&order=market_cap_desc&per_page=100&page=1&sparkline=false")
EXCHANGE_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
INTERVALS = {"1h": pd.Timedelta(hours=1), "5m": pd.Timedelta(minutes=5), "1m": pd.Timedelta(minutes=1)}
DEFAULT_INTERVALS = ("1h", "5m")
STABLE_SYMBOLS = {"USDT", "USDC", "DAI", "USDE", "USD1", "FDUSD", "USDD",
                  "TUSD", "PYUSD", "USDS", "FRAX", "RLUSD", "BUSD", "GUSD",
                  "U", "BFUSD", "USDG", "USDF", "USDGO", "OUSD", "GHO"}

# Exact LIT archive names contain legacy Litentry, not today's ranked Lighter.
# Keep those raw objects, but never put them in the top-100 price panel.
IDENTITY_EXCLUSIONS = {"LITUSDT": {
    "coin_id": "lighter", "reason": "historical LIT spot is Litentry; current ranked Lighter has no verified Binance spot listing",
    "sources": ["https://www.binance.com/en/support/announcement/detail/2a9feaa556f74dcdaa2192366f0e247c",
                "https://www.binance.com/en-IA/how-to-buy/lighter"]}}

DOCUMENTED_IDENTITIES = {
    "ATOMUSDT": ("cosmos", "atom-cosmos", "https://www.binance.com/en/research/projects/cosmos-network"),
    "INJUSDT": ("injective-protocol", "inj-injective-protocol", "https://www.binance.com/en/support/announcement/detail/e9bf1b713c5249fa8839d7d52d12e9d8"),
    "JUPUSDT": ("jupiter-exchange-solana", "jup-jupiter-exchange-token", "https://www.binance.com/en/support/announcement/detail/7b5c643c3d8a4c9a9d443b1ceefb0015"),
    "POLUSDT": ("polygon-ecosystem-token", "pol-polygon-ecosystem-token", "https://www.binance.com/en/support/announcement/detail/619c4929fc3f4a0d9df7f9ae1d4519a5"),
    "WLFIUSDT": ("world-liberty-financial", "wlfi-official-world-liberty-financial", "https://www.binance.com/en/support/announcement/detail/503b79756ba04a1cac0dfd6216958422"),
}

IDENTITY_BOUNDS = {"USDSUSDT": {
    "first_current_asset_bar_start": "2026-04-09T08:00:00+00:00",
    "reason": "2019-2020 USDS is StableUSD; current ranked USDS is Sky/upgraded DAI",
    "sources": ["https://www.binance.com/en-IN/support/announcement/detail/360022747791",
                "https://www.binance.com/en/support/announcement/detail/df2ce9c43b1d406ca2c4b14e4e34610a"]}}


def apply_identity_bounds(series: pd.Series, symbol: str, interval: str) -> tuple[pd.Series, int]:
    """Mask a documented reused-ticker period; never splice predecessor tokens."""
    bound = IDENTITY_BOUNDS.get(symbol)
    if not bound:
        return series, 0
    minimum_end = pd.Timestamp(bound["first_current_asset_bar_start"]) + INTERVALS[interval]
    invalid = series.index < minimum_end
    count = int(series.iloc[np.flatnonzero(invalid)].count())
    result = series.copy()
    result.iloc[np.flatnonzero(invalid)] = np.nan
    return result, count


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def cache_key(request: dict) -> str:
    return _canonical_request(request)[1]


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def public_get(url: str, attempts: int = 3) -> tuple[bytes, dict]:
    """Bounded retries; a missing object (404) remains distinct from an error."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (local noncommercial research)",
                "Accept": "*/*"})
            with urllib.request.urlopen(request, timeout=45) as response:
                blob = response.read()
                return blob, {"url": url, "retrieved_at_utc": now(),
                              "http_status": response.status,
                              "etag": response.headers.get("ETag"),
                              "last_modified": response.headers.get("Last-Modified"),
                              "bytes": len(blob), "sha256": digest(blob)}
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt + 1 == attempts:
                raise
        except (TimeoutError, OSError):
            if attempt + 1 == attempts:
                raise
        time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("unreachable")


def evidence(path: Path, url: str) -> tuple[bytes, dict]:
    """Fetch once; an existing raw file must match its saved hash and URL."""
    meta_path = Path(str(path) + ".http.json")
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        blob = path.read_bytes()
        if meta["url"] != url or meta["sha256"] != digest(blob):
            raise ValueError(f"Existing raw evidence failed validation: {path}")
        return blob, meta
    blob, meta = public_get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_bytes(blob)
    temporary.replace(path)
    write_json(meta_path, meta)
    return blob, meta


def map_universe(ranking: list[dict], exchange: dict) -> list[dict]:
    """Exact, unique symbol matches only; preserve all 100 IDs and their status.

    Symbol matching is inspectable but is not a verified historical contract
    identity. No ticker rename, token migration or quote conversion is applied.
    """
    if len(ranking) != 100 or len({x["id"] for x in ranking}) != 100:
        raise ValueError("Ranking must contain exactly 100 unique coin IDs")
    ranks = [x["market_cap_rank"] for x in ranking]
    if ranks != list(range(1, 101)):
        raise ValueError("Ranking is not the current ordered ranks 1 through 100")
    counts = pd.Series([x["symbol"].upper() for x in ranking]).value_counts()
    pairs = {x["baseAsset"]: x for x in exchange["symbols"]
             if x["quoteAsset"] == "USDT" and x.get("isSpotTradingAllowed", False)}
    result = []
    for coin in ranking:
        base = coin["symbol"].upper()
        pair = pairs.get(base)
        status = ("quote_asset_excluded" if base == "USDT" else
                  "ambiguous_symbol_excluded" if counts[base] != 1 else
                  "no_exact_spot_usdt_pair" if pair is None else "eligible")
        result.append({"rank": coin["market_cap_rank"], "coin_id": coin["id"],
                       "name": coin["name"], "symbol": base,
                       "market_cap_usd": coin["market_cap"],
                       "ranking_last_updated": coin["last_updated"],
                       "stablecoin_symbol_flag": base in STABLE_SYMBOLS,
                       "status": status,
                       "venue_symbol": pair["symbol"] if status == "eligible" else None,
                       "venue_status": pair["status"] if pair else None,
                       "identity_method": "unique exact current symbol; contracts/history unverified"})
    return result


def verify_identities(root: Path, plan: dict, workers: int = 1) -> dict:
    """Cross-check ranked coin IDs against provider's current Binance spot pairs.

    Unresolved candidates retain raw archive history separately and are omitted
    explicitly from trusted panels. A documented TON/GRAM rename is the sole
    ID alias; old TON observations are not concatenated with GRAM observations.
    """
    output = root / "identity_review.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    paprika_blob, paprika_meta = evidence(root / "evidence" / "coinpaprika_binance_markets.json",
                "https://api.coinpaprika.com/v1/exchanges/binance/markets")
    paprika = json.loads(paprika_blob)
    universe = json.loads((root / "universe.json").read_text(encoding="utf-8"))["coins"]
    candidates = [c for c in universe if c["status"] == "eligible"]
    records = []

    def check(coin):
        symbol = coin["venue_symbol"]
        if symbol in IDENTITY_EXCLUSIONS:
            return {"venue_symbol": symbol, "coin_id": coin["coin_id"],
                    "status": "different_historical_asset_excluded", **IDENTITY_EXCLUSIONS[symbol]}
        matching = [m for m in paprika if m.get("pair") == coin["symbol"] + "/USDT"
                    and m.get("category") == "Spot"
                    and m.get("base_currency_name", "").casefold() == coin["name"].casefold()]
        documented = DOCUMENTED_IDENTITIES.get(symbol)
        if documented and coin["coin_id"] == documented[0]:
            named_pairs = [m for m in paprika if m.get("pair") == coin["symbol"] + "/USDT"
                           and m.get("category") == "Spot" and m.get("base_currency_id") == documented[1]]
            if len(named_pairs) == 1:
                return {"venue_symbol": symbol, "coin_id": coin["coin_id"],
                        "status": "confirmed_documented_name_alias", "coinpaprika_coin_id": documented[1],
                        "evidence": paprika_meta, "sources": [documented[2]],
                        "identity_method": "Official venue project/listing confirms named asset; provider IDs retained separately",
                        "historical_identity_limit": "Exact-symbol history only; no predecessor-symbol splice"}
        if len(matching) == 1:
            return {"venue_symbol": symbol, "coin_id": coin["coin_id"],
                    "status": "confirmed_current_coin_id",
                    "identity_method": "CoinGecko ranked name equals Coinpaprika Binance spot base name; exact unique symbol",
                    "coinpaprika_coin_id": matching[0]["base_currency_id"],
                    "evidence": paprika_meta,
                    "historical_identity_limit": "Exact-symbol history only; no automatic token migrations or renamed-series splice"}
        url = f"https://api.coingecko.com/api/v3/coins/{coin['coin_id']}/tickers?exchange_ids=binance"
        path = root / "evidence" / "identity" / f"{coin['coin_id']}.json"
        if not path.exists():
            time.sleep(6)  # Keyless provider limits: keep this metadata route bounded.
        blob, meta = evidence(path, url)
        payload = json.loads(blob)
        tickers = payload["tickers"]
        pairs = [t for t in tickers if t["base"] == coin["symbol"] and t["target"] == "USDT"
                 and t.get("market", {}).get("identifier") == "binance"]
        ids = {t.get("coin_id") for t in pairs}
        confirmed = ids == {coin["coin_id"]} or (len(ids) == 1 and
                           payload.get("name", "").casefold() == coin["name"].casefold())
        alias = coin["coin_id"] == "the-open-network" and coin["symbol"] == "GRAM" and ids == {"gram"}
        return {"venue_symbol": symbol, "coin_id": coin["coin_id"],
                "status": "confirmed_documented_current_alias" if alias else
                          "confirmed_current_coin_id" if confirmed else "unresolved_current_mapping_excluded",
                "ticker_coin_ids": sorted(ids), "evidence": meta,
                "sources": (["https://www.binance.com/en/support/announcement/detail/8177254072cf4da3b9d8fcaacceb0a2c"] if alias else []),
                "historical_identity_limit": "Exact-symbol history only; no automatic token migrations or renamed-series splice"}

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(check, coin): coin for coin in candidates}
        for future in as_completed(futures):
            coin = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append({"venue_symbol": coin["venue_symbol"], "coin_id": coin["coin_id"],
                                "status": "identity_source_error_excluded",
                                "error": f"{type(exc).__name__}: {exc}"})
    review = {"reviewed_at_utc": now(), "requested_count": 100, "candidate_count": len(candidates),
              "trusted_symbols": sorted(x["venue_symbol"] for x in records if x["status"].startswith("confirmed_")),
              "records": sorted(records, key=lambda x: x["venue_symbol"]),
              "policy": "Only provider-coin-ID-confirmed current pairs enter trusted panels; unresolved raw archives remain separate"}
    write_json(output, review)
    return review


def list_archives(root: Path, symbol: str, interval: str) -> list[dict]:
    """Frozen S3 listings enumerate actual archives, avoiding prelisting 404s."""
    prefix = f"data/spot/monthly/klines/{symbol}/{interval}/"
    objects = []
    marker = ""
    page = 0
    while True:
        query = {"prefix": prefix, "max-keys": "1000"}
        if marker:
            query["marker"] = marker
        url = BUCKET + "?" + urllib.parse.urlencode(query)
        raw, _ = evidence(root / "evidence" / "listings" / f"{symbol}_{interval}_{page}.xml", url)
        tree = ET.fromstring(raw)
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in tree.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns)
            if key and key.endswith(".zip"):
                objects.append({"key": key, "size": int(item.findtext("s:Size", namespaces=ns)),
                                "last_modified": item.findtext("s:LastModified", namespaces=ns),
                                "etag": item.findtext("s:ETag", namespaces=ns),
                                "symbol": symbol, "interval": interval,
                                "month": key.rsplit("-", 2)[-2] + "-" + key.rsplit("-", 1)[-1][:-4]})
        if tree.findtext("s:IsTruncated", namespaces=ns) != "true":
            return objects
        marker = tree.findtext("s:NextMarker", namespaces=ns)
        if not marker:
            contents = tree.findall("s:Contents", ns)
            if not contents:
                raise ValueError("Truncated archive listing has no continuation")
            marker = contents[-1].findtext("s:Key", namespaces=ns)
        page += 1


def prepare(root: Path, workers: int = 6) -> dict:
    """Freeze ranking, venue metadata and archive plan before any bulk transfer."""
    path = root / "plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rank_blob, rank_meta = evidence(root / "evidence" / "current_top100.json", RANKING_URL)
    exchange_blob, exchange_meta = evidence(root / "evidence" / "exchange_info.json", EXCHANGE_URL)
    universe = map_universe(json.loads(rank_blob), json.loads(exchange_blob))
    write_json(root / "universe.json", {"retrieved_at_utc": rank_meta["retrieved_at_utc"],
               "ranking_source": "CoinGecko current market cap ranks", "requested_count": 100,
               "quote": "USDT (venue prices, not USD-converted consolidated prices)",
               "stablecoin_policy": "Keep available stablecoins; exclude USDT self-quote only",
               "historical_universe": "Current survivors; NOT a point-in-time rolling top100",
               "coins": universe})
    jobs, failures = [], []
    eligible = [x["venue_symbol"] for x in universe if x["status"] == "eligible"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(list_archives, root, s, i): (s, i)
                   for s in eligible for i in DEFAULT_INTERVALS}
        for future in as_completed(futures):
            symbol, interval = futures[future]
            try:
                for job in future.result():
                    lower = "2017-08" if interval == "1h" else "2024-10"
                    if lower <= job["month"] < "2026-10":
                        jobs.append(job)
            except Exception as exc:
                failures.append({"symbol": symbol, "interval": interval,
                                 "error": f"{type(exc).__name__}: {exc}"})
    plan = {"prepared_at_utc": now(), "ranking_evidence": rank_meta,
            "exchange_evidence": exchange_meta, "requested_count": 100,
            "eligible_symbols": eligible, "listing_failures": failures,
            "intended_bounds": {"1h": ["2017-08-01", "2026-10-01"],
                                "5m": ["2024-10-01", "2026-10-01"]},
            "jobs": sorted(jobs, key=lambda j: (j["interval"], j["symbol"], j["month"])),
            "archive_files": len(jobs), "expected_raw_bytes": sum(x["size"] for x in jobs),
            "disk_free_bytes_at_start": shutil.disk_usage(root).free}
    write_json(path, plan)
    return plan


def prepare_minutes(root: Path, base_plan: dict, workers: int = 6) -> dict:
    """Additional one-minute year for the already verified asset identities."""
    path = root / "minute_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    identities = verify_identities(root, base_plan)
    jobs, failures = [], []
    symbols = identities["trusted_symbols"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(list_archives, root, symbol, "1m"): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                jobs.extend(j for j in future.result() if "2025-10" <= j["month"] < "2026-10")
            except Exception as exc:
                failures.append({"symbol": symbol, "interval": "1m", "error": f"{type(exc).__name__}: {exc}"})
    plan = {"prepared_at_utc": now(), "ranking_evidence": base_plan["ranking_evidence"],
            "exchange_evidence": base_plan["exchange_evidence"], "requested_count": 100,
            "eligible_symbols": symbols, "listing_failures": failures,
            "intended_bounds": {"1m": ["2025-10-01", "2026-10-01"]},
            "jobs": sorted(jobs, key=lambda j: (j["symbol"], j["month"])),
            "archive_files": len(jobs), "expected_raw_bytes": sum(j["size"] for j in jobs),
            "plan_path": str(path), "checkpoint_name": "minute_checkpoint.json",
            "prior_raw_planned_bytes": base_plan["expected_raw_bytes"],
            "disk_free_bytes_at_start": shutil.disk_usage(root).free}
    write_json(path, plan)
    return plan


def download_archive(root: Path, job: dict) -> dict:
    path = root / "raw" / job["key"]
    checksum_blob, _ = evidence(Path(str(path) + ".CHECKSUM"), VISION + job["key"] + ".CHECKSUM")
    checksum = checksum_blob.decode("ascii").strip().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Publisher checksum is malformed")
    blob, meta = evidence(path, VISION + job["key"])
    if digest(blob) != checksum:
        raise ValueError(f"Publisher SHA-256 mismatch: {path}")
    return {**job, **meta, "publisher_sha256": checksum,
            "status": "verified", "path": str(path.relative_to(root))}


def acquire(root: Path, plan: dict, *, workers: int = 6,
            max_raw_bytes: int = 10_000_000_000, sample: bool = False) -> dict:
    """Download listed objects with bounded workers; checkpoint every 25 results."""
    if plan["listing_failures"]:
        raise ValueError("Archive listing errors must be resolved explicitly before acquisition")
    jobs = plan["jobs"]
    if sample:
        jobs = [j for j in jobs if j["symbol"] == "BTCUSDT" and
                j["month"] in ("2024-12", "2025-01", "2026-09")]
    total = sum(x["size"] for x in jobs)
    if total + plan.get("prior_raw_planned_bytes", 0) > max_raw_bytes:
        raise ValueError(f"Archive plan {total:,} bytes exceeds explicit limit {max_raw_bytes:,}")
    if shutil.disk_usage(root).free < total * 3 + 2_000_000_000:
        raise ValueError("Insufficient local working disk space")
    checkpoint = root / ("sample_checkpoint.json" if sample else plan.get("checkpoint_name", "checkpoint.json"))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_archive, root, j): j for j in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({**job, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            if len(results) % 25 == 0 or len(results) == len(jobs):
                write_json(checkpoint, {"updated_at_utc": now(), "planned_files": len(jobs),
                           "completed_files": len(results), "results": results})
                print(json.dumps({"completed": len(results), "planned": len(jobs),
                                  "verified": sum(x["status"] == "verified" for x in results),
                                  "raw_bytes": sum(x.get("bytes", 0) for x in results)}), flush=True)
    return {"results": results, "checkpoint": str(checkpoint)}


def parse_archive(path: Path, symbol: str, interval: str, month: str) -> tuple[pd.Series, dict]:
    """Validate 12-field klines, ms/us switch, bounds, OHLC and exact bar duration."""
    if interval not in INTERVALS:
        raise ValueError("Unsupported interval")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        expected = f"{symbol}-{interval}-{month}.csv"
        if len(members) != 1 or members[0].filename != expected or members[0].file_size > 50_000_000:
            raise ValueError("Unexpected ZIP contents or uncompressed size")
        raw = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(raw), header=None)
    if frame.shape[1] != 12 or frame.empty:
        raise ValueError("Unexpected kline shape")
    unit = "us" if month >= "2025-01" else "ms"
    opens = pd.to_datetime(pd.to_numeric(frame[0], errors="raise"), unit=unit, utc=True)
    ends = pd.to_datetime(pd.to_numeric(frame[6], errors="raise"), unit=unit, utc=True)
    quantum = pd.Timedelta(microseconds=1) if unit == "us" else pd.Timedelta(milliseconds=1)
    duration = INTERVALS[interval]
    elapsed = ends - opens
    standard_duration = elapsed == duration - quantum
    start = pd.Timestamp(month + "-01", tz="UTC")
    stop = start + pd.offsets.MonthBegin(1)
    if not ((opens >= start) & (opens < stop)).all():
        raise ValueError("Kline outside archive month")
    on_grid = pd.Series(opens.array.as_unit("ns").astype("int64") % duration.value == 0,
                        index=opens.index)
    if not (standard_duration & on_grid).any():
        raise ValueError("Archive has no standard, UTC-grid candles; check timestamp units")
    numeric = frame[[1, 2, 3, 4, 5, 7, 8, 9, 10]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all() or not (numeric[[1, 2, 3, 4]] > 0).all().all():
        raise ValueError("Nonfinite or nonpositive prices")
    if not ((numeric[2] >= numeric[[1, 3, 4]].max(axis=1)) &
            (numeric[3] <= numeric[[1, 2, 4]].min(axis=1))).all():
        raise ValueError("OHLC ordering fails")
    if not (numeric[[5, 7, 8, 9, 10]] >= 0).all().all():
        raise ValueError("Negative volume or trade count")
    index = pd.DatetimeIndex(opens + duration, name="bar_end_utc")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted klines")
    series = pd.Series(numeric[4].to_numpy(dtype=float), index=index, name=symbol)
    # Legacy partial candles and inconsistent inclusive endpoints are retained
    # in raw ZIPs and individually masked, without discarding the good month.
    series.iloc[np.flatnonzero(~standard_duration.to_numpy())] = np.nan
    series = series.iloc[np.flatnonzero(on_grid.to_numpy())]
    return series, {"rows": len(series), "timestamp_unit": unit,
                    "first_bar_start": opens.iloc[0].isoformat(),
                    "last_bar_start": opens.iloc[-1].isoformat(),
                    "zero_trade_bars": int((numeric[8] == 0).sum()),
                    "zero_volume_bars": int((numeric[5] == 0).sum()),
                    "nonstandard_duration_bars_masked": int((~standard_duration).sum()),
                    "nonstandard_bar_starts": opens[~standard_duration].astype(str).tolist(),
                    "off_grid_bars_excluded": int((~on_grid).sum()),
                    "off_grid_bar_starts": opens[~on_grid].astype(str).tolist(),
                    "uncompressed_csv_sha256": digest(raw)}


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Last hourly close only when a UTC day has all 24 observed hourly closes.

    Input bar ends run 01:00..24:00; output is next midnight (minimum availability).
    Neither missing hours nor missing prices are filled.
    """
    starts = hourly.copy()
    starts.index = starts.index - pd.Timedelta(hours=1)
    counts = starts.resample("1D").count()
    daily = starts.resample("1D").last().where(counts == 24)
    daily.index = daily.index + pd.Timedelta(days=1)
    daily.index.name = "bar_end_utc"
    return daily


def five_to_ten_minutes(prices: pd.DataFrame) -> pd.DataFrame:
    """A ten-minute close exists only if both child five-minute bars are observed."""
    starts = prices.copy()
    starts.index = starts.index - pd.Timedelta(minutes=5)
    counts = starts.resample("10min").count()
    result = starts.resample("10min").last().where(counts == 2)
    result.index = result.index + pd.Timedelta(minutes=10)
    result.index.name = "bar_end_utc"
    return result


def freeze_panels(root: Path, plan: dict, intervals=None) -> dict:
    """Parse verified raw sources; save explicit coverage and immutable caches."""
    checkpoint = json.loads((root / plan.get("checkpoint_name", "checkpoint.json")).read_text(encoding="utf-8"))
    intervals = list(plan["intended_bounds"]) if intervals is None else intervals
    if not intervals or not set(intervals) <= set(plan["intended_bounds"]):
        raise ValueError("Requested interval is not part of this acquisition plan")
    identity_review = verify_identities(root, plan)
    identity_by_symbol = {x["venue_symbol"]: x for x in identity_review["records"]}
    results = checkpoint["results"]
    completed = {x["key"] for x in results}
    expected = {x["key"] for x in plan["jobs"] if x["interval"] in intervals}
    if not expected <= completed:
        raise ValueError("Requested interval download is incomplete; resume acquisition first")
    panels, summaries = {}, []
    for interval, bounds in plan["intended_bounds"].items():
        if interval not in intervals:
            continue
        start, end = [pd.Timestamp(x, tz="UTC") for x in bounds]
        grid = pd.date_range(start + INTERVALS[interval], end, freq=INTERVALS[interval],
                             name="bar_end_utc")
        columns, coverage, failures = {}, [], []
        for position, symbol in enumerate(plan["eligible_symbols"]):
            if position % 10 == 0:
                print(json.dumps({"stage": "parse", "interval": interval,
                                  "candidate": position + 1, "candidates": len(plan["eligible_symbols"])}), flush=True)
            if symbol not in identity_review["trusted_symbols"]:
                coverage.append({"symbol": symbol, "status": "identity_excluded",
                                 "observations": 0, "identity": identity_by_symbol[symbol]})
                continue
            parts, raw_stats = [], []
            identity_masked = 0
            jobs = [x for x in results if x["symbol"] == symbol and x["interval"] == interval]
            for job in jobs:
                if job["status"] != "verified":
                    failures.append(job)
                    continue
                path = root / job["path"]
                if digest(path.read_bytes()) != job["publisher_sha256"]:
                    raise ValueError(f"Raw hash changed: {path}")
                try:
                    series, stats = parse_archive(path, symbol, interval, job["month"])
                    series, removed_identity_rows = apply_identity_bounds(series, symbol, interval)
                    identity_masked += removed_identity_rows
                    stats["legacy_other_asset_closes_masked"] = removed_identity_rows
                    parts.append(series)
                    raw_stats.append({"month": job["month"], **stats})
                except Exception as exc:
                    failures.append({"symbol": symbol, "interval": interval,
                                     "month": job["month"], "status": "parse_error",
                                     "error": f"{type(exc).__name__}: {exc}"})
            if not parts:
                coverage.append({"symbol": symbol, "status": "no_valid_archive_observations",
                                 "observations": 0, "archive_months": len(jobs)})
                continue
            series = pd.concat(parts)
            if series.index.has_duplicates:
                raise ValueError(f"Overlapping raw archive timestamps: {symbol}")
            series = series.sort_index().reindex(grid)
            columns[symbol] = series
            available = series.dropna()
            coverage.append({"symbol": symbol, "status": "covered", "observations": len(available),
                             "missing_slots": int(series.isna().sum()),
                             "first_bar_end": available.index.min().isoformat(),
                             "last_bar_end": available.index.max().isoformat(),
                             "archive_months": len(parts), "raw_statistics": raw_stats,
                             "legacy_other_asset_closes_masked": identity_masked,
                             "identity_bounds": IDENTITY_BOUNDS.get(symbol)})
        if not columns:
            raise ValueError(f"No valid observations for {interval}")
        prices = pd.DataFrame(columns, index=grid)
        panels[interval] = prices
        summaries.append(save_panel(root, plan, interval, prices, bounds, coverage, failures))
    for child, interval, convert, derivation in [
        ("1h", "1d", hourly_to_daily, "UTC daily close from all 24 observed hourly bars; incomplete day NaN"),
        ("5m", "10m", five_to_ten_minutes, "Ten-minute close requires both observed five-minute child bars; incomplete bucket NaN")]:
        if child in panels:
            derived = convert(panels[child])
            derived = derived.loc[:, derived.notna().any()]
            child_failures = next(x["source_failures"] for x in summaries if x["interval"] == child)
            summaries.append(save_panel(root, plan, interval, derived, plan["intended_bounds"][child],
                             [{"symbol": c, "observations": int(derived[c].count()),
                               "missing_slots": int(derived[c].isna().sum())} for c in derived], child_failures,
                             derivation=derivation))
    summary_path = root / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    written_intervals = {x["interval"] for x in summaries}
    summaries = [x for x in previous.get("panels", []) if x["interval"] not in written_intervals] + summaries
    all_raw = {}
    for checkpoint_path in (root / "checkpoint.json", root / "minute_checkpoint.json"):
        if checkpoint_path.exists():
            for item in json.loads(checkpoint_path.read_text(encoding="utf-8"))["results"]:
                all_raw[item["key"]] = item
    all_results = list(all_raw.values())
    summary = {"completed_at_utc": now(), "spend_usd": 0, "requested_universe_count": 100,
               "raw_archive_files": len(all_results), "verified_archive_files": sum(x["status"] == "verified" for x in all_results),
               "raw_archive_bytes": sum(x.get("bytes", 0) for x in all_results),
               "panels": summaries, "catalog_written": False,
               "known_limits": ["Current ranking creates survivor/look-ahead selection for historical tests",
                   "Single Binance spot venue and USDT numeraire; not global USD prices",
                   "Current coin IDs cross-checked; exact-symbol historical contract continuity remains unverified beyond documented exclusions",
                   "Bar-end timestamp is minimum availability; publication latency unmeasured",
                   "Publisher archives may later be revised; frozen hashes preserve this acquisition"]}
    write_json(root / "summary.json", summary)
    write_json(ROOT / "output" / "expansion" / "crypto_acquisition.json", summary)
    index_path = ROOT / "output" / "expansion" / "crypto_results.json"
    existing = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"datasets": []}
    intervals_written = {x["interval"] for x in summaries}
    datasets = [x for x in existing["datasets"] if x["interval"] not in intervals_written]
    for item in summaries:
        datasets.append({"dataset_id": "crypto_binance_current_top100_" + item["interval"],
                         "family": "crypto", "snapshot_path": item["snapshot_dir"],
                         "interval": item["interval"], "transform": "arithmetic_return",
                         "bar_seconds": {"1m": 60, "5m": 300, "10m": 600, "1h": 3600, "1d": 86400}[item["interval"]],
                         "continuous_utc": True, "units": "USDT per base token, Binance spot close",
                         "source_url": "https://data.binance.vision/",
                         "notes": summary["known_limits"] + ([item["derivation"]] if item["derivation"] else []),
                         "covered_symbols": item["covered_symbols"],
                         "nonmissing_observations": item["nonmissing_observations"]})
    write_json(index_path, {"updated_at_utc": now(), "finalized": False, "datasets": datasets,
                           "requested_universe_path": str(root / "universe.json"),
                           "identity_exclusions": IDENTITY_EXCLUSIONS, "spend_usd": 0})
    finalize_index(root)
    return summary


def finalize_index(root: Path) -> dict:
    """Make completeness explicit; never expose preliminary snapshots as final."""
    path = ROOT / "output" / "expansion" / "crypto_results.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    review = json.loads((root / "identity_review.json").read_text(encoding="utf-8"))
    trusted = set(review["trusted_symbols"])
    for dataset in index["datasets"]:
        detail = json.loads((root / f"coverage_{dataset['interval']}.json").read_text(encoding="utf-8"))
        policy = detail["request"]["adjustment"]
        final_policy = ("masked v4" in policy if dataset["interval"] in ("1h", "1d") else
                        "masked v3" in policy or "masked v4" in policy or "masked v2" in policy)
        bound_to_detail = (Path(dataset["snapshot_path"]) == Path(detail["snapshot_dir"]) and
                           Path(detail["snapshot_dir"]).name == cache_key(detail["request"]))
        dataset["finalized"] = (final_policy and bound_to_detail and not detail["source_failures"] and
                                 set(detail["request"]["symbols"]) == trusted and
                                 detail["exact_cache_reload_verified"])
        dataset["validation_policy"] = policy
        dataset["status"] = "finalized" if dataset["finalized"] else "preliminary"
        if dataset["interval"] in ("1h", "1d") and "masked v4" in policy:
            dataset["dataset_id"] = "crypto_binance_current_top100_" + dataset["interval"] + "_identity_v4"
            dataset["notes"] = list(dict.fromkeys(dataset["notes"] + [
                "Historical 2019-2020 USDS was StableUSD; masked before current Sky USDS listing on 2026-04-09 08:00 UTC"] ))
    current_paths = {Path(d["snapshot_path"]).resolve() for d in index["datasets"]}
    superseded = []
    for manifest_path in sorted((root / "snapshots").glob("*/manifest.json")):
        if manifest_path.parent.resolve() in current_paths:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request = manifest["request"]
        if request.get("source") != "binance_vision_spot_archive":
            continue
        superseded.append({"dataset_id": "crypto_binance_current_top100_" + request["interval"],
                           "snapshot_path": str(manifest_path.parent),
                           "snapshot_key": manifest_path.parent.name,
                           "interval": request["interval"], "status": "superseded",
                           "validation_policy": request["adjustment"],
                           "symbol_count": len(request["symbols"]),
                           "reason": "Earlier identity or timestamp validation policy; preserved for audit, excluded from final registry"})
    index["superseded"] = superseded
    index["historical_identity_bounds"] = IDENTITY_BOUNDS
    index["candidate_identity_review_path"] = str(root / "identity_review.json")
    index["requested_universe_count"] = 100
    index["selected_identity_checked_symbols"] = len(trusted)
    index["identity_unresolved_or_excluded"] = [x for x in review.get("records", []) if x["venue_symbol"] not in trusted]
    expected = {"1m", "5m", "10m", "1h", "1d"}
    final = {d["interval"] for d in index["datasets"] if d["finalized"]}
    index["finalized"] = expected <= final
    index["expected_intervals"] = sorted(expected)
    index["updated_at_utc"] = now()
    write_json(path, index)
    return index


def save_panel(root, plan, interval, prices, bounds, coverage, failures, derivation=None):
    print(json.dumps({"stage": "freeze", "interval": interval, "rows": len(prices),
                      "symbols": len(prices.columns), "nonmissing": int(prices.count().sum()),
                      "source_failures": len(failures)}), flush=True)
    request = {"source": "binance_vision_spot_archive", "symbols": prices.columns.tolist(),
               "start": bounds[0], "end": bounds[1], "interval": interval,
               "adjustment": "none; venue spot close; no migration adjustment; nonstandard/off-grid/legacy-USDS bars masked v4",
               "feed": "Binance spot klines USDT monthly archives" +
                       (("; daily derived" if interval == "1d" else f"; {interval} derived") if derivation else ""),
               "timezone_policy": "UTC bar END; earliest close availability; latency unmeasured",
               "universe_as_of": plan["ranking_evidence"]["retrieved_at_utc"]}
    _, manifest, hit = cached_closes(request, fetch=lambda: prices, cache_dir=root / "snapshots")
    restored, reloaded_manifest, reloaded_hit = cached_closes(request, fetch=None, cache_dir=root / "snapshots")
    assert reloaded_hit and reloaded_manifest["sha256"] == manifest["sha256"]
    pd.testing.assert_frame_equal(prices, restored, check_freq=False, check_exact=True)
    folder = root / "snapshots" / cache_key(request)
    detail = {"interval": interval, "snapshot_dir": str(folder), "cache_key": cache_key(request),
              "request": request, "requested_universe_count": 100, "covered_symbols": len(prices.columns),
              "rows": len(prices), "nonmissing_observations": int(prices.count().sum()),
              "missing_cells": int(prices.isna().sum().sum()), "first_bar_end": prices.index.min().isoformat(),
              "last_bar_end": prices.index.max().isoformat(), "sha256": manifest["sha256"],
              "first_observed_bar_end": prices.dropna(how="all").index.min().isoformat(),
              "last_observed_bar_end": prices.dropna(how="all").index.max().isoformat(),
              "coverage": coverage, "source_failures": failures, "derivation": derivation,
              "close_available_no_earlier_than_index": True, "publication_latency_measured": False,
              "exact_cache_reload_verified": True, "identity_review_path": str(root / "identity_review.json"),
              "raw_plan_path": plan.get("plan_path", str(root / "plan.json")),
              "raw_checkpoint_path": str(root / plan.get("checkpoint_name", "checkpoint.json")),
              "documented_identity_bounds": IDENTITY_BOUNDS}
    path = root / f"coverage_{interval}.json"
    write_json(path, detail)
    return {k: v for k, v in detail.items() if k not in ("coverage",)}


def audit_historical_identities(root: Path) -> dict:
    """Inventory all selected histories and focused, primary listing boundaries.

    This is an explicit audit of evidence already acquired, not a claim that
    every historical contract event or issuer price is independently verified.
    """
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    details = json.loads((root / "coverage_1h.json").read_text(encoding="utf-8"))
    universe = {x["venue_symbol"]: x for x in json.loads((root / "universe.json").read_text(encoding="utf-8"))["coins"] if x.get("venue_symbol")}
    rules = {
        "SKYUSDT": ("2025-09-17T08:00:00+00:00", "MKR to SKY at 1:24000; no MKR prices spliced", "https://www.binance.com/en-TR/support/announcement/detail/77f45eea5e7e482497153c7536781a8d"),
        "USDSUSDT": ("2026-04-09T08:00:00+00:00", "DAI to current Sky USDS at 1:1; legacy StableUSD USDS excluded", IDENTITY_BOUNDS["USDSUSDT"]["sources"][1]),
        "GRAMUSDT": ("2026-07-02T08:00:00+00:00", "TON to GRAM at 1:1; no TON prices spliced", "https://www.binance.com/en/support/announcement/detail/8177254072cf4da3b9d8fcaacceb0a2c"),
        "POLUSDT": ("2024-09-13T10:00:00+00:00", "MATIC to POL at 1:1; no MATIC prices spliced", DOCUMENTED_IDENTITIES["POLUSDT"][2]),
        "RENDERUSDT": ("2024-07-26T08:00:00+00:00", "RNDR to RENDER at 1:1; no RNDR prices spliced", "https://www.binance.com/en/support/announcement/detail/d1f2ae8d99b24439a7a900caa9bb6b3b"),
        "JUPUSDT": ("2024-01-31T16:00:00+00:00", "Jupiter Exchange Token spot start postponed from 15:30 to 16:00 UTC; current name and coin ID checked", "https://www.binance.com/es-LA/support/announcement/detail/45be0f381ae3476e85908dbea8586564"),
        "WLFIUSDT": ("2025-09-01T13:00:00+00:00", "World Liberty Financial spot listing; contract addresses specified by venue", DOCUMENTED_IDENTITIES["WLFIUSDT"][2]),
        "ATOMUSDT": ("2019-04-29T04:00:00+00:00", "Cosmos ATOM first spot listing", "https://www.binance.com/en/support/announcement/detail/360027707351"),
        "BCHUSDT": ("2019-11-28T10:00:00+00:00", "BCHABC renamed BCH at 1:1; no BCHABC or older BCC prices spliced", "https://www.binance.com/en/support/announcement/detail/360037219511"),
        "AAVEUSDT": ("2020-10-15T03:00:00+00:00", "100 LEND to 1 AAVE; no LEND prices spliced", "https://www.binance.com/en/support/announcement/detail/17595222999545348c8acdd5dc41c00a"),
        "VETUSDT": ("2018-07-25T04:00:00+00:00", "1 VEN to 100 VET; no VEN prices spliced", "https://www.binance.com/en/support/announcement/detail/360007439672"),
        "DOTUSDT": ("2020-08-18T23:00:00+00:00", "Venue explicitly listed only post-redenominated DOT units from first trading, before the on-chain 100:1 redenomination", "https://www.binance.com/en/support/announcement/detail/0c0e20f3e1584d579448a000fe5a0350"),
        "STXUSDT": ("2019-10-25T04:00:00+00:00", "Blockstack listing; venue research describes current Stacks as formerly Blockstack, same STX ticker", "https://www.binance.com/en/support/announcement/detail/360035477851"),
    }
    inventory, flags = [], []
    for row in details["coverage"]:
        if row["status"] != "covered":
            continue
        symbol = row["symbol"]
        months = sorted({x["month"] for x in plan["jobs"] if x["symbol"] == symbol and x["interval"] == "1h"})
        missing_runs = [{"last_available_month": a, "next_available_month": b,
                         "unavailable_months_between": pd.Period(b, "M").ordinal - pd.Period(a, "M").ordinal - 1}
                        for a, b in zip(months, months[1:]) if pd.Period(b, "M").ordinal - pd.Period(a, "M").ordinal > 1]
        rule = rules.get(symbol)
        first_open = pd.Timestamp(row["first_bar_end"]) - pd.Timedelta(hours=1)
        primary = None if rule is None else {
            "announced_first_current_pair_open": rule[0], "actual_first_valid_hour_open": first_open.isoformat(),
            "exact_boundary_match": first_open == pd.Timestamp(rule[0]), "conclusion": rule[1], "source_url": rule[2]}
        inventory.append({"symbol": symbol, "current_coin_id": universe[symbol]["coin_id"],
                          "current_coin_name": universe[symbol]["name"], "rank": universe[symbol]["rank"],
                          "first_raw_archive_month": months[0], "last_raw_archive_month": months[-1],
                          "raw_archive_months": len(months), "missing_archive_month_runs": missing_runs,
                          "first_current_valid_hour_open": first_open.isoformat(),
                          "observations": row["observations"],
                          "legacy_other_asset_closes_masked": row["legacy_other_asset_closes_masked"],
                          "primary_boundary_check": primary,
                          "full_historical_contract_audit_completed": False})
        bound = IDENTITY_BOUNDS.get(symbol)
        cutoff = pd.Timestamp(bound["first_current_asset_bar_start"]) if bound else None
        nonstandard, offgrid = set(), set()
        zero_all = zero_legacy = zero_volume_all = zero_volume_legacy = 0
        for raw in row["raw_statistics"]:
            nonstandard.update(x for x in raw["nonstandard_bar_starts"] if cutoff is None or pd.Timestamp(x) >= cutoff)
            offgrid.update(x for x in raw["off_grid_bar_starts"] if cutoff is None or pd.Timestamp(x) >= cutoff)
            zero_all += raw["zero_trade_bars"]
            zero_volume_all += raw["zero_volume_bars"]
            if cutoff is not None and pd.Timestamp(raw["last_bar_start"]) < cutoff:
                zero_legacy += raw["zero_trade_bars"]
                zero_volume_legacy += raw["zero_volume_bars"]
            elif cutoff is not None and pd.Timestamp(raw["first_bar_start"]) < cutoff:
                raise ValueError("Mixed old/current asset month requires per-row zero-trade attribution")
        flags.append({"symbol": symbol, "nonstandard_duration_bars_current_asset": len(nonstandard),
                      "off_grid_bars_current_asset": len(offgrid),
                      "unique_masked_or_excluded_raw_bars_current_asset": len(nonstandard | offgrid),
                      "zero_trade_raw_bars_current_asset": zero_all - zero_legacy,
                      "zero_trade_raw_bars_legacy_other_asset": zero_legacy,
                      "zero_volume_raw_bars_current_asset": zero_volume_all - zero_volume_legacy,
                      "zero_volume_raw_bars_legacy_other_asset": zero_volume_legacy,
                      "legacy_other_asset_closes_masked": row["legacy_other_asset_closes_masked"]})
    usdc_raw = next(x["raw_statistics"] for x in details["coverage"] if x["symbol"] == "USDCUSDT")
    stop_month = next(x for x in usdc_raw if x["month"] == "2022-09")
    resume_month = next(x for x in usdc_raw if x["month"] == "2023-03")
    usdc_gap = {"symbol": "USDCUSDT", "announced_trading_stop": "2022-09-26T03:00:00+00:00",
                "last_raw_hour_open_before_gap": stop_month["last_bar_start"],
                "announced_trading_resume": "2023-03-11T14:00:00+00:00",
                "first_raw_hour_open_after_gap": resume_month["first_bar_start"],
                "exact_stop_and_resume_match":
                    pd.Timestamp(stop_month["last_bar_start"]) + pd.Timedelta(hours=1) == pd.Timestamp("2022-09-26T03:00Z") and
                    pd.Timestamp(resume_month["first_bar_start"]) == pd.Timestamp("2023-03-11T14:00Z"),
                "conclusion": "Venue removed USDC/USDT during BUSD auto-conversion and later relisted it; existing calendar NaN gap retained without BUSD replacement",
                "sources": ["https://www.binance.com/en/support/announcement/detail/e62f703604a94538a1f1bc803b2d579f",
                            "https://www.binance.com/en/support/announcement/detail/dd331bed50bb44f485e47aed132bfc02"]}
    audit = {"as_of_utc": now(), "selected_symbols": len(inventory),
             "current_identity_evidence": str(root / "identity_review.json"), "inventory": inventory,
             "primary_boundary_checks": len(rules),
             "all_primary_boundary_checks_match": all(x["primary_boundary_check"]["exact_boundary_match"] for x in inventory if x["primary_boundary_check"]),
             "identity_bounds": IDENTITY_BOUNDS, "excluded_legacy_candidate": IDENTITY_EXCLUSIONS,
             "documented_venue_gaps": [usdc_gap],
             "additional_continuity_evidence": [
                 {"symbol": "STXUSDT", "fact": "Stacks is formerly Blockstack", "source_url": "https://www.binance.com/en/research/projects/stacks"},
                 {"symbol": "BCHUSDT", "fact": "After the November 2020 fork Binance assigned BCH name, existing pairs and historic klines to majority BCHN; BCHA was a separately distributed token", "source_url": "https://www.binance.com/en/support/announcement/detail/6d0b3851f6934ec58d4f87ac37fd19a5"}],
             "limitations": ["No monthly archive gap is not proof of contract continuity", "Current coin-ID/name checks and focused listing checks do not verify every historical migration, fork, rebase or distribution", "Spot closes exclude fork distributions and are not total investment return", "Current top-100 selection is not historical rolling membership"]}
    write_json(ROOT / "output" / "expansion" / "crypto_identity_audit.json", audit)
    quality_path = ROOT / "output" / "expansion" / "crypto_quality_flags.json"
    preserved = quality_path.with_name("crypto_quality_flags_initial_v3.json")
    if quality_path.exists() and not preserved.exists():
        preserved.write_bytes(quality_path.read_bytes())
    quality = {"as_of_utc": now(), "interval": "1h", "snapshot_path": details["snapshot_dir"],
               "policy": "Raw archives preserved; nonstandard/off-grid candles excluded or masked; old USDS StableUSD separated from current Sky USDS; no operator fills",
               "summary": {key: sum(row[key] for row in flags) for key in flags[0] if key != "symbol"},
               "assets": flags,
               "zero_trade_interpretation": "Provider-reported zero-trade bars retained when standard and on-grid; no assertion of independent traded prices"}
    write_json(quality_path, quality)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "identity", "audit", "sample", "acquire", "freeze", "prepare-minutes", "acquire-minutes", "freeze-minutes"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-raw-gb", type=float, default=10)
    parser.add_argument("--interval", choices=list(INTERVALS), help="Freeze one completed interval while other downloads continue")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1 through 8")
    args.root.mkdir(parents=True, exist_ok=True)
    plan = prepare(args.root, args.workers)
    if args.action.endswith("-minutes"):
        plan = prepare_minutes(args.root, plan, args.workers)
    if args.action in ("prepare", "prepare-minutes"):
        print(json.dumps({k: v for k, v in plan.items() if k != "jobs"}, indent=2))
    elif args.action == "identity":
        print(json.dumps(verify_identities(args.root, plan), indent=2))
    elif args.action == "audit":
        print(json.dumps(audit_historical_identities(args.root), indent=2))
    elif args.action in ("sample", "acquire", "acquire-minutes"):
        result = acquire(args.root, plan, workers=args.workers,
                         max_raw_bytes=int(args.max_raw_gb * 1e9), sample=args.action == "sample")
        print(json.dumps({"checkpoint": result["checkpoint"], "errors": [x for x in result["results"] if x["status"] != "verified"]}, indent=2))
    else:
        summary = freeze_panels(args.root, plan, intervals=[args.interval] if args.interval else None)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
