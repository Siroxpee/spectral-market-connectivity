"""Collect an explicitly bounded public-data snapshot and register it locally.

No network access occurs on import. Existing matching snapshots are loaded
offline. Example:
python collect_data.py yahoo --symbols SPY QQQ --start 2026-10-01T00:00:00-04:00 \
    --end 2026-10-03T00:00:00-04:00 --interval 5m --auto-adjust true \
    --cache-dir data/pilots/yahoo_5m --catalog data/archive_catalog.sqlite
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from archive_catalog import load_snapshot, register_snapshot
from market_sources import fetch_coinbase_closes, fetch_ecb_rates, fetch_yahoo_closes, save_source_snapshot


def collect_requested(arguments):
    """Fetch once, freeze prices/evidence, verify exact reload and register."""
    first, last = pd.Timestamp(arguments.start), pd.Timestamp(arguments.end)
    evidence_dir = arguments.evidence_dir or str(Path(arguments.cache_dir) / "raw_evidence")
    if arguments.provider == "yahoo":
        daily = arguments.interval == "1d"
        if not daily:
            if first.tzinfo is None or last.tzinfo is None:
                raise ValueError("Intraday bounds require explicit timezone offsets.")
            first, last = first.tz_convert("UTC"), last.tz_convert("UTC")
        auto_adjust = arguments.auto_adjust == "true"
        source = "Yahoo via yfinance"
        interval = arguments.interval
        adjustment = "yfinance auto_adjust=True" if auto_adjust else "provider Close; auto_adjust=False"
        feed = "Yahoo public interface"
        timezone_policy = f"naive daily dates in {arguments.daily_timezone}" if daily else "UTC instants"
        fetch = lambda: fetch_yahoo_closes(arguments.symbols, first, last, interval=interval,
                                         auto_adjust=auto_adjust, daily_timezone=arguments.daily_timezone, evidence_dir=evidence_dir)
    elif arguments.provider == "ecb":
        source, interval = "European Central Bank reference rates", "1d"
        adjustment, feed = "not applicable: official reference rates", "ECB eurofxref-hist.csv"
        timezone_policy = "naive reference dates; publication around 16:00 CET, not midnight quotes"
        fetch = lambda: fetch_ecb_rates(arguments.symbols, first, last, evidence_dir=evidence_dir)
    else:
        if first.tzinfo is None or last.tzinfo is None:
            raise ValueError("Intraday bounds require explicit timezone offsets.")
        first, last = first.tz_convert("UTC"), last.tz_convert("UTC")
        source, interval = "Coinbase Exchange public candles", f"{arguments.granularity}s"
        adjustment, feed, timezone_policy = "unadjusted venue spot candles", "Coinbase Exchange", "UTC"
        fetch = lambda: fetch_coinbase_closes(arguments.symbols, first, last,
                                             granularity=arguments.granularity, max_requests=arguments.max_requests, evidence_dir=evidence_dir)
    request = {
        "source": source, "symbols": arguments.symbols,
        "start": first.isoformat(), "end": last.isoformat(), "interval": interval,
        "adjustment": adjustment, "feed": feed, "timezone_policy": timezone_policy,
        "universe_as_of": arguments.universe_as_of,
    }
    prices, manifest, provenance, hit = save_source_snapshot(request, fetch, cache_dir=arguments.cache_dir)
    canonical = json.dumps(manifest["request"], sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(canonical.encode()).hexdigest()
    snapshot_path = (Path(arguments.cache_dir) / key).resolve()
    # Registration itself validates the cache; the extra comparison confirms
    # numerical equality with the panel returned by this bounded acquisition.
    metadata = register_snapshot(snapshot_path, arguments.catalog)
    reloaded, _ = load_snapshot(key, arguments.catalog)
    pd.testing.assert_frame_equal(prices, reloaded, check_exact=True, check_freq=False)
    return {
        "provider": arguments.provider, "source": source,
        "snapshot_key": key, "snapshot_path": str(snapshot_path),
        "catalog": str(Path(arguments.catalog).resolve()), "cache_hit": hit,
        "exact_reload_verified": True, "request": manifest["request"],
        "coverage": provenance["coverage"],
        "index_timezone": str(prices.index.tz) if prices.index.tz is not None else None,
        "source_url": provenance["source_url"], "incremental_spend_usd": 0,
        "evidence_dir": str(Path(evidence_dir).resolve()),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    providers = result.add_subparsers(dest="provider", required=True)
    for name in ("yahoo", "ecb", "coinbase"):
        provider = providers.add_parser(name)
        provider.add_argument("--symbols", nargs="+", required=True, help="Exact symbols/currencies/product IDs; space-separated.")
        provider.add_argument("--start", required=True, help="Inclusive start; dates for daily, timezone-aware instants for intraday.")
        provider.add_argument("--end", required=True, help="Exclusive end; no open-ended acquisition.")
        provider.add_argument("--cache-dir", required=True)
        provider.add_argument("--catalog", required=True, help="Local SQLite catalog path.")
        provider.add_argument("--evidence-dir", help="Preserve hashed raw bodies/library frames; default: CACHE_DIR/raw_evidence.")
        provider.add_argument("--universe-as-of", default="explicit caller-selected research list; historical membership not asserted")
        if name == "yahoo":
            provider.add_argument("--auto-adjust", choices=("true", "false"), required=True)
            provider.add_argument("--interval", choices=("1d", "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"), required=True)
            provider.add_argument("--daily-timezone", default="America/New_York")
        elif name == "coinbase":
            provider.add_argument("--granularity", type=int, choices=(60, 300, 900, 3600, 21600, 86400), required=True)
            provider.add_argument("--max-requests", type=int, default=1000)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    summary = collect_requested(arguments)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
