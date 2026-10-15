"""Export bounded offline dashboard samples; never acquire or modify sources.

Raw panel samples contain the latest consecutive source rows, not dates chosen
by a graph metric. Comparison features are computed on each native calendar
before an explicit descriptive calendar-label join. No missing values are filled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

from data_cache import _canonical_request, _file_hash, cached_closes
from expanded_spectral import transform_panel, measure_window, mathematical_battery, finite_json

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "dashboard" / "local_data"
GROUPS = ["stocks", "etfs", "forex", "crypto", "commodities", "electricity", "grid_operations"]
TITLES = {
    "original_stocks_daily": "Original stocks · daily, recovered notebook",
    "yahoo_original_stocks_daily_long": "Yahoo stocks · daily, long history",
    "yahoo_original_stocks_5m_20260817_20261009": "Yahoo stocks · five minutes",
    "yahoo_original_stocks_1m_20261002_20261009": "Yahoo stocks · one minute",
    "yahoo_etf_daily_long": "Yahoo ETFs · daily, long history",
    "yahoo_etf_5m_20260817_20261009": "Yahoo ETFs · five minutes",
    "yahoo_etf_1m_20261002_20261009": "Yahoo ETFs · one minute",
    "ecb_fx_all_history_daily": "ECB currency reference rates · daily",
    "yahoo_fx_usd_daily_usd_numeraire": "Yahoo currencies per US dollar · daily",
    "yahoo_fx_usd_5m_20260817_20261009": "Yahoo currencies per US dollar · five minutes",
    "yahoo_fx_usd_1m_20261002_20261009": "Yahoo currencies per US dollar · one minute",
    "fred_eia_energy_spot_daily": "Oil and natural gas spot assessments · daily",
    "worldbank_commodity_monthly_1960": "World Bank physical commodity prices · monthly",
    "yahoo_commodity_futures_daily_27ticker": "Yahoo commodity futures proxies · daily",
    "yahoo_commodity_futures_5m_20260817_20261009": "Yahoo commodity futures proxies · five minutes",
    "yahoo_commodity_futures_1m_20261002_20261009": "Yahoo commodity futures proxies · one minute",
    "eia_wholesale_gas_daily_2014_2017": "Wholesale natural gas · daily, 2014–2017",
    "eia_wholesale_electric_daily_2001_202609": "Wholesale electricity contracts · daily",
    "nyiso_dam_hourly_2020_202609": "New York electricity · day-ahead hourly",
    "caiso_dam_hourly_202308_202609": "California electricity · day-ahead hourly",
    "caiso_rtm_5min_202609": "California electricity · real-time five minutes",
    "ercot_dam_hourly_2020_202609": "Texas electricity · day-ahead hourly",
    "isone_da_hourly_2025_202609": "New England electricity · day-ahead hourly",
    "isone_rt_hourly_2025_202609": "New England electricity · real-time hourly",
    "eia_state_retail_total_monthly": "State retail electricity · all sectors monthly",
    "eia_state_retail_residential_monthly": "State retail electricity · residential monthly",
    "eia_state_retail_commercial_monthly": "State retail electricity · commercial monthly",
    "eia_state_retail_industrial_monthly": "State retail electricity · industrial monthly",
    "eia_grid_demand_hourly_2019": "US grid demand · hourly",
    "eia_grid_generation_hourly_2019": "US grid generation · hourly",
    "eia_grid_interchange_hourly_2019": "US grid interchange · hourly",
    "crypto_binance_current_top100_1m": "Binance crypto · one minute, 12 months",
    "crypto_binance_current_top100_5m": "Binance crypto · five minutes, 24 months",
    "crypto_binance_current_top100_10m": "Binance crypto · derived ten minutes",
    "crypto_binance_current_top100_1h_identity_v4": "Binance crypto · hourly, long history",
    "crypto_binance_current_top100_1d_identity_v4": "Binance crypto · derived daily",
}
PREFERRED = {
    "stocks": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM"],
    "etfs": ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA"],
    "forex": ["EUR_per_USD", "GBP_per_USD", "JPY_per_USD", "CHF_per_USD", "CAD_per_USD", "AUD_per_USD", "USD", "JPY", "GBP", "CHF", "CAD", "AUD"],
    "crypto": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"],
    "commodities": ["Crude oil, WTI", "Natural gas, US", "Copper", "Gold", "Maize", "Coffee, Arabica", "CL=F", "NG=F", "GC=F", "HG=F", "ZC=F", "KC=F", "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"],
    "electricity": ["CA", "NY", "TX", "FL", "WA", "MA"],
    "grid_operations": [],
}
PAIR_SPECS = [
    ("stocks_etfs_daily", "Stocks and ETFs · daily changes", "yahoo_original_stocks_daily_long", "yahoo_etf_daily_long", "daily_calendar"),
    ("etfs_forex_daily", "ETFs and currencies · daily changes", "yahoo_etf_daily_long", "yahoo_fx_usd_daily_usd_numeraire", "daily_calendar"),
    ("commodities_retail_electricity_monthly", "Physical commodities and retail electricity · monthly changes", "worldbank_commodity_monthly_1960", "eia_state_retail_total_monthly", "reference_month"),
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(finite_json(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Hosted assets must not leak the source machine's paths or credentials.
    if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", encoded) or re.search(r"/(?:Users|home)/", encoded):
        raise ValueError("Absolute local path in hosted asset")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(encoded.encode()).hexdigest()


def public_url(value):
    if not isinstance(value, str) or urlparse(value).scheme not in ("https", "http"):
        return None
    secret_keys = {"api_key", "apikey", "token", "access_token", "secret", "password", "authorization"}
    if secret_keys & {k.lower() for k in parse_qs(urlparse(value).query)}:
        raise ValueError("Credential-like URL parameter in source metadata")
    return value


def group_for(spec):
    identifier = spec["dataset_id"]
    if identifier.startswith("eia_grid_"): return "grid_operations"
    if "stocks" in identifier: return "stocks"
    if "etf" in identifier: return "etfs"
    if "crypto" in identifier: return "crypto"
    if "fx" in identifier: return "forex"
    if any(t in identifier for t in ("caiso", "nyiso", "ercot", "isone", "retail", "wholesale_electric")): return "electricity"
    return "commodities"


def active_inventory(project=ROOT):
    document = json.loads((project / "output/expansion/inventory.json").read_text(encoding="utf-8"))
    superseded = set()
    for filename in ("financial.json", "electricity_results.json", "crypto_results.json"):
        source = project / "output/expansion" / filename
        if source.exists():
            data = json.loads(source.read_text(encoding="utf-8"))
            for entry in data.get("superseded_datasets", []) + data.get("superseded_snapshots", []) + data.get("superseded", []):
                if isinstance(entry, dict) and "snapshot_path" in entry and entry.get("replacement_snapshot_path") != entry["snapshot_path"]:
                    superseded.add(Path(entry["snapshot_path"]).name)
    active = [x for x in document["datasets"] if not x["specification"]["dataset_id"].startswith("legacy_") and x["snapshot"]["snapshot_key"] not in superseded]
    if len(active) != 36 or len({x["specification"]["dataset_id"] for x in active}) != 36:
        raise ValueError("Expected the declared 36 unique active frozen panels")
    return active


def read_frozen_tail(item, rows=481):
    """Validate every source row with bounded RAM, retain only a genuine tail."""
    folder = Path(item["specification"]["snapshot_path"])
    manifest_path, price_path = folder / "manifest.json", folder / "prices.csv.gz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _canonical_request(manifest["request"])[1] != folder.name:
        raise ValueError("Frozen request/path binding mismatch")
    if _file_hash(price_path) != manifest["sha256"] or manifest["sha256"] != item["snapshot"]["sha256"]:
        raise ValueError("Frozen price hash mismatch")
    tail, observed_tail, count, missing, first, last, previous = None, None, 0, 0, None, None, None
    first_observed, last_observed, last_observed_row_number = None, None, None
    discarded_observed, observed_tail_predecessor = None, None
    for chunk in pd.read_csv(price_path, index_col=0, float_precision="round_trip", dtype=manifest.get("column_dtypes"), chunksize=4096):
        if list(chunk.columns) != manifest["actual_symbols"]:
            raise ValueError("Frozen column ordering mismatch")
        chunk.index = pd.to_datetime(chunk.index, format="ISO8601", utc=manifest["index_timezone"] is not None)
        if manifest["index_timezone"] is not None:
            chunk.index = chunk.index.tz_convert(manifest["index_timezone"])
        if chunk.index.hasnans or not chunk.index.is_unique or not chunk.index.is_monotonic_increasing or (previous is not None and chunk.index[0] <= previous):
            raise ValueError("Ambiguous or unsorted frozen source timestamps")
        if np.isinf(chunk.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Infinite frozen price observation")
        observed_positions = np.flatnonzero(chunk.notna().any(axis=1).to_numpy())
        if len(observed_positions):
            first_observed = chunk.index[observed_positions[0]] if first_observed is None else first_observed
            last_position = int(observed_positions[-1])
            last_observed = chunk.index[last_position]
            last_observed_row_number = count + last_position + 1
            observed_prefix = chunk.iloc[:last_position+1]
            observed_candidate = observed_prefix if tail is None else pd.concat([tail, observed_prefix])
            before_tail = observed_candidate.iloc[:-rows] if len(observed_candidate) > rows else observed_candidate.iloc[:0]
            before_observed = before_tail.loc[before_tail.notna().any(axis=1)]
            observed_tail_predecessor = before_observed.iloc[-1].copy() if len(before_observed) else discarded_observed
            observed_tail = observed_candidate.tail(rows)
        count += len(chunk)
        missing += int(chunk.isna().sum().sum())
        first = chunk.index[0] if first is None else first
        last = previous = chunk.index[-1]
        candidate = chunk if tail is None else pd.concat([tail, chunk])
        discarded = candidate.iloc[:-rows] if len(candidate) > rows else candidate.iloc[:0]
        discarded_with_values = discarded.loc[discarded.notna().any(axis=1)]
        if len(discarded_with_values): discarded_observed = discarded_with_values.iloc[-1].copy()
        tail = candidate.tail(rows)
    if count != manifest["rows"] or missing != manifest["missing_cells"] or first.isoformat() != manifest["actual_first_timestamp"] or last.isoformat() != manifest["actual_last_timestamp"]:
        raise ValueError("Frozen coverage summary mismatch")
    if observed_tail is None:
        raise ValueError("Frozen panel has no observed values")
    observed_tail.index.name = manifest["index_name"]
    observed_tail.columns.name = manifest["columns_name"]
    observed_bounds = {"first_observed": first_observed.isoformat(), "last_observed": last_observed.isoformat(),
                       "trailing_all_missing_rows_excluded": count - last_observed_row_number}
    return observed_tail, manifest, hashlib.sha256(manifest_path.read_bytes()).hexdigest(), observed_bounds, observed_tail_predecessor


def select_columns(frame, group, max_variables):
    counts = frame.notna().sum()
    eligible = {str(c) for c in frame.columns if counts[c] > 0}
    if not eligible: eligible = set(frame.columns)  # Preserve an all-missing latest interval honestly.
    preferred = [c for c in PREFERRED[group] if c in eligible and counts[c] >= .9 * len(frame)][:6]
    ranked = sorted(eligible, key=lambda c: (-int(counts[c]), str(c)))
    return (preferred + [c for c in ranked if c not in preferred])[:max_variables]


def columns_metadata(item, columns):
    spec = item["specification"]
    group, units = group_for(spec), spec.get("units", "source units")
    return [{"id": spec["dataset_id"] + "::" + c, "label": c, "group": group,
             "unit": units.get(c, "source units unverified") if isinstance(units, dict) else units,
             "source_variable": c, "transform": spec["transform"]} for c in columns]


def values_json(frame):
    array = frame.to_numpy(dtype=float, na_value=np.nan)
    return [[float(v) if np.isfinite(v) else None for v in row] for row in array]


def notes_for(item, manifest):
    notes = item["specification"].get("notes", [])
    notes = [notes] if isinstance(notes, str) else list(notes)
    for field in ("session_policy", "availability_policy", "timestamp_semantics", "missing_policy"):
        if item["specification"].get(field):
            notes.append(item["specification"][field])
    notes += ["Frozen acquisition batch from October 2026; these are saved observations, not live data.",
              "A bounded variable subset is used for browser interaction; full-panel saved analyses use the complete declared universe.",
              "Missing observations remain null; neither prices nor changes are filled."]
    if item["specification"]["dataset_id"].startswith("eia_grid_"):
        notes.append("Source UTC labels are HOUR ENDING. The old CSV index name says interval_start_utc incorrectly; actual interval start is source label minus one hour. No cross-market operational-hour join is exported.")
    return notes


def panel_export(item, output, max_rows=480, max_variables=16):
    spec = item["specification"]
    tail, manifest, manifest_hash, observed_bounds, observed_tail_predecessor = read_frozen_tail(item, max_rows + 1)
    sample = tail.tail(max_rows)
    source_row_observed = sample.notna().any(axis=1).tolist()
    selected = select_columns(sample, group_for(spec), max_variables)
    sample = sample[selected]
    previous = tail.iloc[-len(sample)-1] if len(tail) > len(sample) else None
    timestamp_policy = manifest["request"]["timezone_policy"]
    if spec["dataset_id"].startswith("eia_grid_"):
        timestamp_policy = "UTC source HOUR ENDING; actual interval start = source timestamp minus one hour; legacy CSV index name is incorrect"
    links = list(dict.fromkeys(url for url in [public_url(spec.get("source_url")), public_url(manifest["request"]["source"])] if url))
    if "Yahoo" in manifest["request"]["source"]: links += ["https://finance.yahoo.com/"]
    if "ECB" in manifest["request"].get("feed", ""): links += ["https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"]
    data = {"id": spec["dataset_id"], "title": TITLES[spec["dataset_id"]], "group": group_for(spec),
            "interval": spec["interval"], "transform": spec["transform"], "prepared": False,
            "units": spec.get("units", "source units"), "timestamp_policy": timestamp_policy,
            "source_notes": notes_for(item, manifest), "source_links": list(dict.fromkeys(links)),
            "coverage": {"full_rows": manifest["rows"], "full_variables": len(manifest["actual_symbols"]),
                "observed_cells": manifest["rows"] * len(manifest["actual_symbols"]) - manifest["missing_cells"],
                "missing_cells": manifest["missing_cells"], "first": observed_bounds["first_observed"], "last": observed_bounds["last_observed"],
                "first_requested_label": manifest["actual_first_timestamp"], "last_requested_label": manifest["actual_last_timestamp"], **observed_bounds},
            "sample": {"dates": [d.isoformat() for d in sample.index], "columns": columns_metadata(item, selected),
                "values": values_json(sample), "max_rows": max_rows,
                "selection_note": "Consecutive source rows ending at the latest row with any observed value. Trailing all-missing requested rows are counted separately; gaps inside the sample are preserved. Up to six familiar variables with at least 90% coverage, then highest nonmissing count; alphabetical ties. Selection never uses eigenvalues or outcomes.",
                "observed_cells": int(sample.notna().sum().sum()), "full_variable_count": len(manifest["actual_symbols"])},
            "preceding_row": None if previous is None else {"date": previous.name.isoformat(), "values": values_json(previous[selected].to_frame().T)[0]},
            "bar_seconds": spec.get("bar_seconds"),
            "availability_policy": spec.get("availability_policy", "Source labels are not an independently verified historical publication-time database."),
            "provenance": {"source": manifest["request"]["source"], "feed": manifest["request"]["feed"],
                "adjustment": manifest["request"]["adjustment"], "universe_as_of": manifest["request"]["universe_as_of"],
                "snapshot_key": folder_name(spec["snapshot_path"]), "prices_sha256": manifest["sha256"],
                "manifest_sha256": manifest_hash, "saved_at_utc": manifest["snapshot_saved_at_utc"]},
            "effective_count_note": "Transform after date selection using the preceding source row if present. Gap-spanning intraday changes and missing inputs stay null. Display usable observations and retained variables explicitly."}
    if spec["interval"] in ("1d", "1d_trade_dates", "1mo"):
        data["sample"]["source_row_observed"] = source_row_observed
        data["sample"]["all_source_missing_rows"] = sum(not v for v in source_row_observed)
        data["sample"]["source_row_observed_scope"] = "Boolean based on every variable in the full native source panel, before the displayed subset is selected. Partly missing rows remain observed."
        calendar_previous = previous if previous is not None and previous.notna().any() else observed_tail_predecessor
        data["observed_calendar_preceding_row"] = None if calendar_previous is None else {
            "date": calendar_previous.name.isoformat(), "values": values_json(calendar_previous[selected].to_frame().T)[0]}
        data["observed_calendar_note"] = "Optional alternative: omit only native rows where ALL source variables are missing, then compute changes on that observed-row calendar. This changes the observation time base and is not reproduction of the native-row baseline. No filling, interpolation, or removal of partly missing rows."
    if group_for(spec) == "crypto":
        data["license"] = {"name": "CC BY-NC-SA 4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                           "attribution": "Binance Vision", "terms_url": "https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md",
                           "note": "Derived crypto excerpts for noncommercial research; no Binance sponsorship or endorsement."}
    relative = "panels/" + spec["dataset_id"] + ".json"
    write_json(output / relative, data)
    return data, relative


def folder_name(value):
    return Path(value).name


def read_selected_prices(item, columns):
    folder = Path(item["specification"]["snapshot_path"])
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    header = pd.read_csv(folder / "prices.csv.gz", nrows=0).columns[0]
    frame = pd.read_csv(folder / "prices.csv.gz", usecols=[header] + columns, index_col=0,
                        float_precision="round_trip", dtype={k:v for k,v in manifest.get("column_dtypes", {}).items() if k in columns})
    frame.index = pd.to_datetime(frame.index, format="ISO8601", utc=manifest["index_timezone"] is not None)
    if manifest["index_timezone"] is not None: frame.index = frame.index.tz_convert(manifest["index_timezone"])
    return frame[columns]


def comparison_export(pair, by_id, panels, output, max_rows=480):
    identifier, title, first_id, second_id, alignment = pair
    groups, frames, metadata, full_features, full_observed_cells = [], [], [], [], 0
    for source_id in (first_id, second_id):
        item = by_id[source_id]
        panel = panels[source_id]
        # These are small daily/monthly panels. Transform on native labels first
        # so removed join dates do not create artificial multi-day changes.
        candidates = [c["source_variable"] for c in panel["sample"]["columns"]]
        prices = read_selected_prices(item, candidates)
        changes, _ = transform_panel(prices, item["specification"])
        if prices.index.tz is not None:
            raise ValueError("Comparison requires declared calendar/reference labels, not implicit timezone conversion")
        changes.index = pd.DatetimeIndex([d.to_period("M").start_time if alignment == "reference_month" else d.normalize() for d in changes.index])
        if not changes.index.is_unique:
            raise ValueError("Ambiguous comparison calendar labels")
        full_features.append(changes)
        groups.append(group_for(item["specification"]))
    common = full_features[0].index.intersection(full_features[1].index).sort_values()
    dates = common[-max_rows:]
    for source_id, feature in zip((first_id, second_id), full_features):
        item = by_id[source_id]
        selected = select_columns(feature.loc[dates], group_for(item["specification"]), 8)
        selected_metadata = columns_metadata(item, selected)
        for column in selected_metadata:
            column["feature_unit"] = "dimensionless arithmetic change in quote" if column["transform"] == "arithmetic_return" else "change in " + column["unit"]
        metadata += selected_metadata
        full_observed_cells += int(feature.loc[common, selected].notna().sum().sum())
        frames.append(feature.loc[dates, selected].rename(columns={c: source_id + "::" + c for c in selected}))
    joined = pd.concat(frames, axis=1)
    timing = ("Matched daily calendar labels for descriptive comparison. ETF/stock closing clocks and FX quote sessions differ; labels do not prove simultaneous availability." if alignment == "daily_calendar" else
              "Matched reference months for descriptive comparison. Monthly commodity assessments and retail electricity averages are published later and may be revised; reference months are not availability times.")
    notes = [timing, "Features were transformed on each full native source calendar before joining; missing changes remain null.",
             "Up to eight variables per group selected by coverage and declared familiar defaults, without graph-metric selection.",
             "Correlation indicates shared variation, not a causal route, tradable transfer or forecast."]
    if alignment == "reference_month": notes += ["Retail electricity is an average billed price, not an executable wholesale quote. Commodity aggregate/component overlaps are not independent physical markets."]
    data = {"id": identifier, "title": title, "group": "comparison", "groups": groups,
            "interval": "1mo" if alignment == "reference_month" else "1d", "transform": "prepared", "prepared": True,
            "units": "mixed per-column native features", "timestamp_policy": timing,
            "source_notes": notes, "source_links": list(dict.fromkeys(url for source_id in (first_id, second_id) for url in panels[source_id]["source_links"])),
            "source_datasets": [first_id, second_id],
            "provenance": [{"dataset_id": source_id, **panels[source_id]["provenance"]} for source_id in (first_id, second_id)],
            "coverage": {"full_rows": len(common), "full_variables": len(metadata),
                         "observed_cells": full_observed_cells, "missing_cells": len(common)*len(metadata)-full_observed_cells,
                         "source_full_variables": sum(panels[i]["coverage"]["full_variables"] for i in (first_id, second_id)),
                         "first": common[0].isoformat(), "last": common[-1].isoformat(),
                         "observed_cells_scope": "all common native feature labels, for the selected comparison variables only"},
            "sample": {"dates": [d.isoformat() for d in joined.index], "columns": metadata, "values": values_json(joined),
                       "max_rows": max_rows, "selection_note": "Latest common native feature labels; at most eight variables per group. Features are already prepared and must not be transformed again."},
            "preceding_row": None, "bar_seconds": None, "alignment": alignment,
            "effective_count_note": "These values are already native-source changes; select dates and columns directly, then clean one common complete sample."}
    relative = "comparisons/" + identifier + ".json"
    write_json(output / relative, data)
    return data, relative


def validate_export(data):
    sample = data["sample"]
    dates, columns, values = sample["dates"], sample["columns"], sample["values"]
    assert 0 < len(dates) == len(values) <= 480
    assert len(dates) == len(set(dates)) and dates == sorted(dates)
    assert len({c["id"] for c in columns}) == len(columns) <= 16
    assert all(len(row) == len(columns) for row in values)
    assert all(v is None or isinstance(v, (float, int)) and np.isfinite(v) for row in values for v in row)
    assert all(c["group"] in GROUPS and "::" in c["id"] for c in columns)
    if data["preceding_row"] is not None:
        assert data["preceding_row"]["date"] < dates[0]
        assert len(data["preceding_row"]["values"]) == len(columns)
    if data["prepared"]: assert data["transform"] == "prepared"
    if "source_row_observed" in sample:
        assert len(sample["source_row_observed"]) == len(dates)
        assert all(isinstance(v, bool) for v in sample["source_row_observed"])
        assert sample["all_source_missing_rows"] == sum(not v for v in sample["source_row_observed"])
        preceding = data["observed_calendar_preceding_row"]
        if preceding is not None:
            assert preceding["date"] < dates[0] and len(preceding["values"]) == len(columns)


def frame_from_export(data):
    sample = data["sample"]
    frame = pd.DataFrame(sample["values"], columns=[c["id"] for c in sample["columns"]], index=pd.to_datetime(sample["dates"], format="ISO8601"))
    if data["prepared"]: return frame
    previous = data["preceding_row"]
    if previous is not None:
        leading = pd.DataFrame([previous["values"]], columns=frame.columns, index=pd.to_datetime([previous["date"]], format="ISO8601"))
        frame = pd.concat([leading, frame])
    return transform_panel(frame, data)[0]


def check_contracts():
    """Meaningful offline checks of export boundaries and sample preparation."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        dates = pd.date_range("2025-01-01", periods=10000, freq="min", tz="UTC")
        source = pd.DataFrame({"A": np.arange(10000, dtype=float)+10., "B": np.arange(10000, dtype=float)+20.}, index=dates)
        source.iloc[4301:] = np.nan
        source.iloc[4295, 0] = np.nan
        source.iloc[3820] = np.nan  # Immediate predecessor to a 480-row sample is blank.
        request = {"source": "synthetic offline check", "symbols": ["A", "B"], "start": "2025-01-01",
                   "end": "2025-01-08", "interval": "1m", "adjustment": "none", "feed": "synthetic",
                   "timezone_policy": "UTC", "universe_as_of": "synthetic test"}
        _, manifest, _ = cached_closes(request, fetch=lambda: source, cache_dir=temporary)
        folder = temporary / _canonical_request(request)[1]
        item = {"specification": {"snapshot_path": str(folder)}, "snapshot": {"sha256": manifest["sha256"]}}
        tail, _, _, bounds, predecessor = read_frozen_tail(item, rows=481)
        assert len(tail) == 481 and tail.index[-1] == dates[4300]
        assert bounds["trailing_all_missing_rows_excluded"] == 5699
        assert pd.isna(tail.loc[dates[4295], "A"])
        assert tail.index.equals(dates[3820:4301])
        assert predecessor.name == dates[3819]
        assert tail.iloc[0].isna().all() and predecessor.notna().all()
        asset_path = temporary / "public.json"
        write_json(asset_path, {"source": "https://example.org/official"})
        try:
            write_json(asset_path, {"private": "C:\\Users\\private\\file"})
        except ValueError:
            pass
        else:
            raise AssertionError("Local path was not rejected")
        try:
            public_url("https://example.org/?api_key=example")
        except ValueError:
            pass
        else:
            raise AssertionError("Credential-like URL was not rejected")
        print("PASS bounded cross-chunk latest-observed tail, internal gap preservation, hosted path/URL exclusion")
    stocks = pd.DataFrame({"A": [10., 12., 14.]}, index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04"]))
    fx = pd.DataFrame({"B": [20., 24., 30., 33.]}, index=pd.date_range("2025-01-01", periods=4))
    stock_features = transform_panel(stocks, {"transform": "arithmetic_return"})[0]
    fx_features = transform_panel(fx, {"transform": "arithmetic_return"})[0]
    common = stock_features.index.intersection(fx_features.index)
    assert np.isclose(fx_features.loc[common[0], "B"], .25)
    assert not np.isclose(fx_features.loc[common[0], "B"], .5)
    data = {"prepared": False, "transform": "arithmetic_return", "bar_seconds": 60,
            "sample": {"dates": ["2025-01-01T00:01:00+00:00", "2025-01-01T00:03:00+00:00"],
                       "columns": [{"id": "test::A"}], "values": [[12.], [18.]]},
            "preceding_row": {"date": "2025-01-01T00:00:00+00:00", "values": [10.]}}
    prepared = frame_from_export(data)
    assert len(prepared) == 2 and np.isclose(prepared.iloc[0, 0], .2) and pd.isna(prepared.iloc[1, 0])
    data["prepared"], data["transform"], data["preceding_row"] = True, "prepared", None
    assert frame_from_export(data).iloc[0, 0] == 12.
    selected = select_columns(pd.DataFrame({"ZZZ": [1., 2.], "AAPL": [1., 2.], "BBB": [1., np.nan]}), "stocks", 2)
    assert selected == ["AAPL", "ZZZ"]
    print("PASS native-calendar return lag, preceding price, strict gap mask, prepared bypass and deterministic coverage selection")


def refresh_daily_metadata(output=DEFAULT_OUTPUT):
    """Refresh daily/monthly samples only, preserving other exports and catalog."""
    catalog_path = output / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = {x["id"]: x for x in catalog["datasets"]}
    changed = []
    for item in active_inventory():
        if item["specification"]["interval"] not in ("1d", "1d_trade_dates", "1mo"):
            continue
        path = output / "panels" / (item["specification"]["dataset_id"] + ".json")
        previous = json.loads(path.read_text(encoding="utf-8"))
        data, _ = panel_export(item, output)
        validate_export(data)
        assert previous["sample"]["values"] == data["sample"]["values"]
        assert previous["sample"]["dates"] == data["sample"]["dates"]
        assert previous["sample"]["columns"] == data["sample"]["columns"]
        entries[data["id"]]["notes"] = data["source_notes"]
        entries[data["id"]]["all_source_missing_sample_rows"] = data["sample"]["all_source_missing_rows"]
        changed.append({"id": data["id"], "all_source_missing_sample_rows": data["sample"]["all_source_missing_rows"]})
    catalog["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(catalog_path, catalog)
    return changed


def export_all(output=DEFAULT_OUTPUT):
    active = active_inventory()
    by_id = {x["specification"]["dataset_id"]: x for x in active}
    panels, entries, pairs, cases = {}, [], [], []
    for item in active:
        data, relative = panel_export(item, output)
        validate_export(data)
        panels[data["id"]] = data
        entries.append({"id": data["id"], "title": data["title"], "group": data["group"], "interval": data["interval"],
                        "transform": data["transform"], "path": relative, "coverage": data["coverage"],
                        "sample_rows": len(data["sample"]["dates"]), "sample_variables": len(data["sample"]["columns"]), "notes": data["source_notes"]})
        # Publish the catalog after each completed export so the UI can integrate
        # while larger sources stream. Completeness is explicit throughout.
        catalog = {"as_of": "2026-10-09", "exported_at_utc": datetime.now(timezone.utc).isoformat(), "groups": GROUPS,
                   "datasets": entries, "comparisons": pairs, "spend_usd": 0,
                   "finalized": False, "expected_panels": 36, "data_status": "Frozen October 2026 acquisition batch; not live"}
        write_json(output / "catalog.json", catalog)
        print(json.dumps({"exported": data["id"], "rows": len(data["sample"]["dates"]), "variables": len(data["sample"]["columns"])}), flush=True)
    comparison_data = []
    for pair in PAIR_SPECS:
        data, relative = comparison_export(pair, by_id, panels, output)
        validate_export(data)
        comparison_data.append(data)
        pairs.append({"id": data["id"], "title": data["title"], "path": relative, "groups": data["groups"], "interval": data["interval"],
                      "notes": data["source_notes"], "sample_rows": len(data["sample"]["dates"]), "sample_variables": len(data["sample"]["columns"])})
    selected_references = ["yahoo_original_stocks_daily_long", "yahoo_etf_5m_20260817_20261009", "caiso_dam_hourly_202308_202609", "worldbank_commodity_monthly_1960", "eia_grid_interchange_hourly_2019", "isone_da_hourly_2025_202609"]
    for data in [panels[i] for i in selected_references] + comparison_data:
        features = frame_from_export(data).tail(60)
        metric, spectra = measure_window(features)
        battery = mathematical_battery(features)
        cases.append({"id": data["id"], "path": ("comparisons/" if data["prepared"] else "panels/") + data["id"] + ".json",
                      "selection": "Last 60 available feature rows in the exported sample; all exported columns; no metric-dependent endpoint selection",
                      "feature_rows": len(features), "expected_metrics": metric, "expected_spectra": spectra,
                      "expected_battery": battery})
    write_json(output / "reference_cases.json", {"implementation": "Python pandas/scipy and established spectral_core common-sample baseline", "cases": cases, "absolute_tolerance": 1e-8})
    catalog["datasets"], catalog["comparisons"], catalog["finalized"] = entries, pairs, True
    catalog["reference_cases_path"] = "reference_cases.json"
    catalog["total_sample_cells"] = sum(e["sample_rows"]*e["sample_variables"] for e in entries + pairs)
    write_json(output / "catalog.json", catalog)
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Run synthetic offline export-contract checks only")
    parser.add_argument("--refresh-daily-metadata", action="store_true", help="Refresh daily/monthly masks without repeating large minute exports")
    arguments = parser.parse_args()
    if arguments.check:
        check_contracts()
    elif arguments.refresh_daily_metadata:
        print(json.dumps(refresh_daily_metadata(arguments.output), indent=2))
    else:
        result = export_all(arguments.output)
        print(json.dumps({"finalized": result["finalized"], "datasets": len(result["datasets"]), "comparisons": len(result["comparisons"]), "sample_cells": result["total_sample_cells"]}, indent=2))
