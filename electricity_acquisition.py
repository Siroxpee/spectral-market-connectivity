"""Bounded official CAISO/NYISO/ERCOT/ISO-NE and EIA wholesale observations.

Prices are signed USD/MWh observations, not positive asset closes. UTC interval
starts identify observations. Raw official ZIPs and component signs are retained;
no network call occurs on import and no trade/physical-network inference is made.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urljoin
import zipfile

import numpy as np
import pandas as pd
import requests

from data_cache import cached_closes, _canonical_request

CAISO_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
NYISO_URL = "https://mis.nyiso.com/public/csv/damlbmp/"
CAISO_HUBS = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]
NYISO_ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
EIA_PAGE = "https://www.eia.gov/electricity/wholesale/"
EIA_POWER_PRODUCTS = ["ERCOT North 345KV Peak", "Indiana Hub RT Peak", "Mid C Peak", "NP15 EZ Gen DA LMP Peak", "Nepool MH DA LMP Peak", "PJM WH Real Time Peak", "Palo Verde Peak", "SP15 EZ Gen DA LMP Peak"]
EIA_GAS_PRODUCTS = ["Algonquin Citygates", "Chicago Citygates", "Henry", "Malin", "PG&E - Citygate", "Socal-Citygate", "Socal-Ehrenberg", "TETCO-M3"]
EIA_LEGACY_URL = "https://www.eia.gov/electricity/wholesale/xls/archive/ice_electric-historical.zip"
EIA_LEGACY_PRODUCTS = ["legacy|" + name for name in ["Palo Verde Hub", "PJM West Hub", "SP 15_1 Hub", "SP 15_2 Hub", "Indiana Hub", "Mass Hub", "NP 15 Hub", "MID-C Hub"]]
ERCOT_HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]
ERCOT_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
ERCOT_DOWNLOAD = "https://www.ercot.com/misdownload/servlets/mirDownload"
ISONE_ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA"]
ISONE_PAGE = "https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info"
ISONE_LIST = "https://www.iso-ne.com/isoexpress/web/reports/download/docWidgetGetMore"
DEFAULT_ROOT = Path("data/expanded/electricity")
MAX_BATCH_BYTES = 500_000_000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_bounds(start, end, interval="1h"):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tz is None or end.tz is None:
        raise ValueError("Use explicit timezone offsets or UTC Z on both bounds.")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    step = pd.Timedelta(interval)
    if not start < end or start.floor(interval) != start or end.floor(interval) != end:
        raise ValueError("Bounds must be increasing and aligned to the declared interval.")
    if (end - start) % step:
        raise ValueError("The requested span must contain whole intervals.")
    return start, end


def locations_checked(locations):
    if not locations or not all(isinstance(item, str) and item.strip() == item and item for item in locations):
        raise ValueError("Declare nonempty location names explicitly.")
    if len(locations) != len(set(locations)):
        raise ValueError("Duplicate requested locations.")
    return sorted(locations)


def raw_download(url, params, directory, *, max_response_bytes=32_000_000, attempts=3):
    """Freeze a request's first successful body; cached bodies are hash checked."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = requests.Request("GET", url, params=params).prepare().url
    request_key = hashlib.sha256(prepared.encode()).hexdigest()
    record_path = directory / (request_key + ".json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        path = directory / record["body_file"]
        if record["requested_url"] != prepared or sha256(path) != record["body_sha256"]:
            raise ValueError("Frozen raw response URL/checksum mismatch.")
        return path.read_bytes(), record
    existing_bytes = sum(path.stat().st_size for path in DEFAULT_ROOT.rglob("*") if path.is_file()) if DEFAULT_ROOT.exists() else 0
    remaining = MAX_BATCH_BYTES - existing_bytes
    if remaining <= 0:
        raise ValueError("Initial electricity archive has reached its 500 MB acquisition bound.")
    last_error = None
    for attempt in range(attempts):
        try:
            with requests.get(url, params=params, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(128 * 1024):
                    size += len(chunk)
                    if size > min(max_response_bytes, remaining):
                        raise ValueError("Response exceeds explicit archive/download size limit.")
                    chunks.append(chunk)
                body = b"".join(chunks)
                digest = hashlib.sha256(body).hexdigest()
                extension = ".zip" if body.startswith(b"PK") else ".body"
                filename = digest + extension
                (directory / filename).write_bytes(body)
                record = {"requested_url": prepared, "resolved_url": response.url,
                          "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                          "http_status": response.status_code, "content_type": response.headers.get("Content-Type"),
                          "last_modified_header": response.headers.get("Last-Modified"),
                          "body_file": filename, "body_sha256": digest, "bytes": len(body),
                          "evidence_path": str((directory / filename).resolve())}
                record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                return body, record
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    raise last_error


def zip_csvs(body):
    if not body.startswith(b"PK"):
        raise ValueError("Expected official ZIP archive; received a different body.")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        infos = archive.infolist()
        if sum(item.file_size for item in infos) > 200_000_000:
            raise ValueError("Uncompressed response exceeds 200 MB safety bound.")
        names = sorted(item.filename for item in infos if item.filename.lower().endswith(".csv"))
        if not names:
            details = archive.read(infos[0]).decode("utf-8", errors="replace")[:1000] if infos else "empty ZIP"
            raise ValueError("Official ZIP contains no CSV data: " + details)
        return [(name, pd.read_csv(io.BytesIO(archive.read(name)))) for name in names]


def parse_caiso(body, start, end, locations, interval="1h", market="DAM"):
    start, end = utc_bounds(start, end, interval)
    frames = [frame.assign(raw_member=name) for name, frame in zip_csvs(body)]
    frame = pd.concat(frames, ignore_index=True)
    value_column = "MW" if market == "DAM" else "VALUE"
    required = {"INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT", "NODE", "LMP_TYPE", value_column, "MARKET_RUN_ID", "OPR_DT", "OPR_HR", "OPR_INTERVAL"}
    if not required <= set(frame.columns):
        raise ValueError("CAISO CSV schema is missing required fields.")
    frame["interval_start_utc"] = pd.to_datetime(frame["INTERVALSTARTTIME_GMT"], utc=True)
    frame["interval_end_utc"] = pd.to_datetime(frame["INTERVALENDTIME_GMT"], utc=True)
    frame = frame.loc[frame["interval_start_utc"].ge(start) & frame["interval_start_utc"].lt(end) & frame.NODE.isin(locations)].copy()
    if frame.empty or not frame.MARKET_RUN_ID.eq(market).all():
        raise ValueError("Empty CAISO result or unexpected settlement market.")
    if not frame["interval_end_utc"].sub(frame["interval_start_utc"]).eq(pd.Timedelta(interval)).all():
        raise ValueError("CAISO reported interval lengths disagree with request.")
    frame["price_usd_per_mwh"] = pd.to_numeric(frame[value_column], errors="raise")
    if not np.isfinite(frame["price_usd_per_mwh"]).all():
        raise ValueError("CAISO response includes nonfinite prices.")
    if frame.duplicated(["interval_start_utc", "NODE", "LMP_TYPE"]).any():
        raise ValueError("Duplicate CAISO price/component observations.")
    return frame.rename(columns={"NODE": "location", "LMP_TYPE": "component"})


def parse_nyiso(body, start, end, locations):
    """Resolve the two fall-back hours from their source row order, per zone."""
    start, end = utc_bounds(start, end)
    frames = [frame.assign(raw_member=name) for name, frame in zip_csvs(body)]
    frame = pd.concat(frames, ignore_index=True)
    mapping = {"LBMP ($/MWHr)": "LBMP", "Marginal Cost Losses ($/MWHr)": "reported_marginal_cost_losses", "Marginal Cost Congestion ($/MWHr)": "reported_marginal_cost_congestion"}
    if not {"Time Stamp", "Name", "PTID", *mapping} <= set(frame.columns):
        raise ValueError("NYISO CSV schema is missing required fields.")
    frame = frame.loc[frame.Name.isin(locations)].copy()
    frame["interval_start_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for _, group in frame.groupby("Name", sort=False):
        naive = pd.DatetimeIndex(pd.to_datetime(group["Time Stamp"], format="mixed", errors="raise"))
        # Archive members and their rows remain in original chronological order.
        # Ambiguous single hours or impossible spring-forward labels must fail.
        localized = naive.tz_localize("America/New_York", ambiguous="infer", nonexistent="raise")
        frame.loc[group.index, "interval_start_utc"] = localized.tz_convert("UTC")
    frame = frame.loc[frame["interval_start_utc"].ge(start) & frame["interval_start_utc"].lt(end)].copy()
    if frame.empty or frame.duplicated(["interval_start_utc", "Name"]).any():
        raise ValueError("Empty/duplicate NYISO UTC observation.")
    for column in mapping:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError("NYISO response includes nonfinite prices.")
    frame["interval_end_utc"] = frame["interval_start_utc"] + pd.Timedelta(hours=1)
    frame = frame.rename(columns={"Name": "location", "Time Stamp": "source_local_label"})
    long = frame.melt(id_vars=["interval_start_utc", "interval_end_utc", "location", "PTID", "source_local_label", "raw_member"], value_vars=list(mapping), var_name="component", value_name="price_usd_per_mwh")
    long["component"] = long["component"].map(mapping)
    return long


def panel_from_components(components, start, end, locations, interval, total_component):
    start, end = utc_bounds(start, end, interval)
    values = components.loc[components.component.eq(total_component)]
    if values.duplicated(["interval_start_utc", "location"]).any():
        raise ValueError("Duplicate total-price observations across request chunks.")
    if set(values.location.unique()) != set(locations):
        raise ValueError("Actual location universe differs from the explicit request.")
    panel = values.pivot(index="interval_start_utc", columns="location", values="price_usd_per_mwh")
    panel = panel.reindex(index=pd.date_range(start, end, freq=interval, inclusive="left"), columns=sorted(locations)).astype(float)
    panel.index.name, panel.columns.name = "interval_start_utc", "location"
    return panel


def coverage(panel, units="USD/MWh"):
    unit_suffix = "mmbtu" if units == "USD/MMBtu" else "mwh"
    return {"expected_intervals": len(panel), "location_count": len(panel.columns),
            "nonmissing_total_prices": int(panel.notna().sum().sum()), "missing_total_prices": int(panel.isna().sum().sum()),
            "negative_prices": int(panel.lt(0).sum().sum()), "exact_zero_prices": int(panel.eq(0).sum().sum()),
            "minimum_usd_per_" + unit_suffix: float(panel.min().min()), "maximum_usd_per_" + unit_suffix: float(panel.max().max()),
            "per_location": {name: {"nonmissing": int(panel[name].notna().sum()), "missing": int(panel[name].isna().sum()), "negative": int(panel[name].lt(0).sum()), "zero": int(panel[name].eq(0).sum())} for name in panel}}


def cache_request(provider, start, end, locations, interval, market, units="USD/MWh", daily=False):
    return {"source": provider + " official public wholesale electricity", "symbols": locations_checked(locations),
            "start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(), "interval": interval,
            "adjustment": f"signed wholesale {units} levels; no financial return assumption; no fill",
            "feed": market + " latest historical archive retrieved; revision vintages not reconstructed",
            "timezone_policy": "naive trade-date labels, not instants; inclusive start and exclusive end" if daily else "UTC interval starts; exact inclusive start and exclusive end",
            "universe_as_of": "fixed caller-declared hubs/zones; no inferred physical connectivity"}


def freeze_observations(panel, components, provenance, cache_dir):
    request = provenance["cache_request"]
    _, key = _canonical_request(request)
    target = Path(cache_dir) / key
    if target.exists():
        return load_observations(target)
    saved, manifest, _ = cached_closes(request, fetch=lambda: panel, cache_dir=cache_dir)
    component_path = target / "price_components.csv.gz"
    components.sort_values(["interval_start_utc", "location", "component"]).to_csv(component_path, index=False, compression="gzip", float_format="%.17g")
    provenance = dict(provenance, price_file_sha256=manifest["sha256"], components_file_sha256=sha256(component_path), coverage=coverage(saved, provenance["units"]))
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    # Descriptive level/dollar-difference correlations deliberately avoid pct_change.
    saved.corr().to_csv(target / "price_level_correlations.csv", float_format="%.17g")
    saved.diff().corr().to_csv(target / "first_difference_correlations.csv", float_format="%.17g")
    loaded, _, _ = load_observations(target)
    pd.testing.assert_frame_equal(saved, loaded, check_freq=False, check_exact=True)
    return loaded, provenance, target.resolve()


def load_observations(snapshot_dir):
    target = Path(snapshot_dir)
    provenance = json.loads((target / "provenance.json").read_text())
    request = provenance["cache_request"]
    _, expected_key = _canonical_request(request)
    if target.name != expected_key:
        raise ValueError("Observation snapshot key disagrees with its request.")
    panel, manifest, _ = cached_closes(request, cache_dir=target.parent)
    if provenance["price_file_sha256"] != manifest["sha256"] or sha256(target / "price_components.csv.gz") != provenance["components_file_sha256"]:
        raise ValueError("Observation/component provenance checksum mismatch.")
    for record in provenance["raw_responses"]:
        if sha256(record["evidence_path"]) != record["body_sha256"]:
            raise ValueError("Official raw evidence checksum mismatch.")
    if provenance["coverage"] != coverage(panel, provenance["units"]):
        raise ValueError("Observation coverage summary mismatch.")
    return panel, provenance, target.resolve()


def parse_eia_workbook(body, kind, start, end, products, *, legacy_location=None):
    """Keep trade and delivery dates distinct; use explicitly named ICE products."""
    if kind not in ("electric", "natgas"):
        raise ValueError("EIA product kind must be electric or natgas.")
    engine = "openpyxl" if body.startswith(b"PK") else "xlrd"
    if engine == "xlrd":
        # Optional parser for official legacy BIFF XLS; no install/network here.
        tooling = DEFAULT_ROOT / "tooling"
        if tooling.is_dir() and str(tooling.resolve()) not in sys.path:
            sys.path.insert(0, str(tooling.resolve()))
    book = pd.ExcelFile(io.BytesIO(body), engine=engine)
    unit = "MMBtu" if kind == "natgas" else "MWh"
    weighted = f"Wtd avg price $/{unit}"
    frames = []
    for sheet in book.sheet_names:
        frame = book.parse(sheet)
        frame.columns = [" ".join(str(column).split()) for column in frame.columns]
        required = {"Price hub", "Trade date", "Delivery start date", "Delivery end date", weighted, f"High price $/{unit}", f"Low price $/{unit}"}
        canonical = {name.casefold(): name for name in required}
        frame = frame.rename(columns={name: canonical.get(name.casefold(), name) for name in frame.columns})
        if not required <= set(frame.columns):
            if legacy_location is not None and (frame.empty or not required.intersection(frame.columns)):
                continue
            raise ValueError(f"EIA {sheet} schema missing required columns: {sorted(required - set(frame.columns))}")
        if legacy_location is not None:
            frame["source_price_hub"] = frame["Price hub"]
            frame["Price hub"] = legacy_location
        frame = frame.loc[frame["Price hub"].isin(products)].copy()
        for column in ["Trade date", "Delivery start date", "Delivery end date"]:
            frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
        if frame["Trade date"].isna().any():
            raise ValueError("EIA quote has no trade-date label.")
        frame = frame.loc[frame["Trade date"].ge(start) & frame["Trade date"].lt(end)].copy()
        for column in [weighted, f"High price $/{unit}", f"Low price $/{unit}"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if np.isinf(frame[column]).any():
                raise ValueError("Infinite EIA transaction price.")
        frame["missing_weighted_average_price"] = frame[weighted].isna()
        frame["invalid_weighted_price_range"] = (frame[f"Low price $/{unit}"] > frame[weighted] + 0.01) | (frame[f"High price $/{unit}"] < frame[weighted] - 0.01) | frame[f"Low price $/{unit}"].gt(frame[f"High price $/{unit}"])
        frame["invalid_delivery_date_order"] = frame["Delivery end date"].lt(frame["Delivery start date"])
        frame["source_sheet"] = sheet
        frame = frame.rename(columns={"Price hub": "location", "Trade date": "trade_date", weighted: "weighted_average_price"})
        frames.append(frame)
    if not frames:
        raise ValueError("No EIA quote sheet matched the declared schema.")
    return pd.concat(frames, ignore_index=True)


def acquire_eia(start, end, *, kind="electric", products=None, root=DEFAULT_ROOT):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tz is not None or end.tz is not None or start.normalize() != start or end.normalize() != end or start >= end:
        raise ValueError("EIA bounds must be increasing naive trade-date labels at midnight.")
    if kind not in ("electric", "natgas"):
        raise ValueError("EIA product kind must be electric or natgas.")
    include_legacy = kind == "electric" and start.year < 2014
    products = locations_checked(products or ((EIA_POWER_PRODUCTS + (EIA_LEGACY_PRODUCTS if include_legacy else [])) if kind == "electric" else EIA_GAS_PRODUCTS))
    units = "USD/MWh" if kind == "electric" else "USD/MMBtu"
    request = cache_request("EIA ICE " + kind, start, end, products, "1d_trade_dates", "physical peak electricity index; mixed named DA/RT peak products" if kind == "electric" else "physical regional natural-gas weighted index", units, daily=True)
    cache_dir = Path(root) / ("eia_" + kind + "_daily")
    _, key = _canonical_request(request)
    if (cache_dir / key).exists():
        return load_observations(cache_dir / key)
    page_body, page_record = raw_download(EIA_PAGE, None, Path(root) / "raw" / "eia")
    links = sorted(set(urljoin(EIA_PAGE, link) for link in re.findall(r'href=["\x27]([^"\x27]+\.xlsx?)', page_body.decode("utf-8", errors="replace"))))
    matching = []
    for url in links:
        match = re.search(r"ice_" + kind + r"-(\d{4})(?:final)?\.xlsx?$", url)
        if match and start.year <= int(match.group(1)) <= (end - pd.Timedelta(days=1)).year:
            matching.append((int(match.group(1)), url))
    if not matching and not include_legacy:
        raise ValueError("Official EIA page supplies no workbook for this product/date span.")
    frames, records, supplied_years = [], [page_record], []
    if include_legacy:
        body, record = raw_download(EIA_LEGACY_URL, None, Path(root) / "raw" / "eia")
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            for member in sorted(archive.namelist()):
                if not member.lower().endswith(".xls"):
                    continue
                location = "legacy|" + Path(member).stem
                if location not in products:
                    continue
                parsed = parse_eia_workbook(archive.read(member), kind, start, end, products, legacy_location=location)
                parsed["source_workbook"] = EIA_LEGACY_URL + "#" + member
                frames.append(parsed)
                print(f"EIA legacy: {member} retained ({len(parsed)} records)", flush=True)
        records.append(record)
        supplied_years.extend(range(max(start.year, 2001), min(end.year, 2014)))
    for year, url in sorted(matching):
        body, record = raw_download(url, None, Path(root) / "raw" / "eia")
        parsed = parse_eia_workbook(body, kind, start, end, products)
        parsed["source_workbook"] = url
        frames.append(parsed); records.append(record); supplied_years.append(year)
        print(f"EIA {kind}: official {year} workbook retained ({len(parsed)} records)", flush=True)
    components = pd.concat(frames, ignore_index=True)
    # Annual sheets can overlap, and one hub/trade date can quote distinct
    # delivery contracts. Preserve every source record and avoid choosing or
    # averaging different contracts into a fictitious single quote.
    identity = ["trade_date", "location"]
    material = [column for column in components if column not in ("source_sheet", "source_workbook")]
    unique_quotes = components.drop_duplicates(material, keep="first")
    ambiguous = unique_quotes.loc[unique_quotes.duplicated(identity, keep=False), identity].drop_duplicates()
    ambiguous_keys = set(map(tuple, ambiguous.to_numpy()))
    components["ambiguous_hub_trade_date"] = [tuple(row) in ambiguous_keys for row in components[identity].to_numpy()]
    usable_quotes = unique_quotes.loc[~unique_quotes[identity].apply(tuple, axis=1).isin(ambiguous_keys) & ~unique_quotes["invalid_weighted_price_range"]]
    if set(components.location.unique()) != set(products):
        raise ValueError("EIA requested product universe is missing entirely from supplied workbooks.")
    panel = usable_quotes.pivot(index="trade_date", columns="location", values="weighted_average_price").reindex(index=pd.DatetimeIndex(sorted(components.trade_date.unique())), columns=products).astype(float)
    panel.index.name, panel.columns.name = "trade_date", "ICE_product"
    # Generic frozen signed-observation cache preserves all source trade dates.
    components["interval_start_utc"] = components["trade_date"]  # compatibility sort only; labels remain naive trade dates
    components["component"] = "volume_weighted_transaction_index"
    requested_years = list(range(start.year, (end - pd.Timedelta(days=1)).year + 1))
    provenance = {"provider": "EIA republishing ICE", "units": units, "settlement_market": "Named ICE physical product transaction-weighted index", "interval": "daily source trade-date observations",
                  "timestamp_label": "Naive TRADE DATE, distinct from delivery start/end; no invented UTC clock time",
                  "source_operating_timezone": "not assigned to date-only labels; source describes electricity qualifying trades 06:00-11:00 CST",
                  "dst_policy": "Date labels only; no localization or DST conversion",
                  "component_policy": "Retain high/low, weighted average, traded volume, trade/counterparty counts and delivery start/end; these are transaction indices, not hourly nodal LMP components",
                  "publication_timing": "EIA republishes ICE data biweekly; original publication vintages not reconstructed and delivery dates are not information times",
                  "declared_locations": products, "actual_locations": panel.columns.tolist(), "cache_request": request, "raw_responses": records,
                  "supplied_workbook_years": supplied_years, "requested_years_without_public_link": sorted(set(requested_years) - set(supplied_years)),
                  "observation_grid": "Union of actual supplied trade dates; absent hub quotes stay NaN; no calendar/weekend padding or fill; the rolling60 meaning is60 observed trade dates",
                  "network_interpretation": "Regional physical-product price observation panel, not executable transfer prices or measured pipeline/transmission edges",
                  "legacy_parser": "xlrd2.0.2 fromPyPI; optional local dependency under data/expanded/electricity/tooling",
                  "invalid_delivery_date_order_records": int(components["invalid_delivery_date_order"].sum()),
                  "missing_weighted_average_price_records": int(components["missing_weighted_average_price"].sum()),
                  "invalid_weighted_price_range_records": int(components["invalid_weighted_price_range"].sum()),
                  "price_quality_policy": "Raw weighted indices outside reported low/high (0.01 rounding allowance) or low above high are flagged, retained in evidence, and excluded from the wide price matrix; no source correction invented",
                  "legacy_contract_policy": "Each historical ZIP workbook has a separate legacy| column; SP15 pre/post March2009 and other named historical contracts are never silently spliced into current named peak products",
                  "ambiguous_hub_trade_date_cells": len(ambiguous_keys),
                  "ambiguous_quote_policy": "Different delivery contracts or conflicting records at one hub/trade date remain fully preserved in source observation table; wide one-price matrix leaves that cell NaN rather than selecting or averaging contracts",
                  "delivery_quality_policy": "Preserve and flag source delivery-end-before-start inconsistencies; retain valid trade-date price observation without correcting delivery dates; exclude these records from any future physical delivery model",
                  "source_links": [EIA_PAGE] + ([EIA_LEGACY_URL] if include_legacy else []), "artifact_notes": "price_components.csv.gz retains source observations; compatibility interval_start_utc field contains naive trade-date labels, not UTC instants"}
    return freeze_observations(panel, components, provenance, cache_dir)


def acquire_caiso(start, end, locations=CAISO_HUBS, *, interval="1h", market="DAM", root=DEFAULT_ROOT):
    if (interval, market) not in (("1h", "DAM"), ("5min", "RTM")):
        raise ValueError("Supported explicit markets: DAM hourly or RTM five-minute.")
    start, end = utc_bounds(start, end, interval)
    locations = locations_checked(locations)
    request = cache_request("CAISO", start, end, locations, interval, market)
    cache_dir = Path(root) / ("caiso_" + market.lower() + "_" + interval)
    _, key = _canonical_request(request)
    if (cache_dir / key).exists():
        return load_observations(cache_dir / key)
    frames, records = [], []
    cursor = start
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=30 if market == "DAM" else 7), end)
        params = {"queryname": "PRC_LMP" if market == "DAM" else "PRC_INTVL_LMP", "version": 12 if market == "DAM" else 3,
                  "market_run_id": market, "node": ",".join(locations), "resultformat": 6,
                  "startdatetime": cursor.strftime("%Y%m%dT%H:%M-0000"), "enddatetime": stop.strftime("%Y%m%dT%H:%M-0000")}
        body, record = raw_download(CAISO_URL, params, Path(root) / "raw" / "caiso")
        frames.append(parse_caiso(body, cursor, stop, locations, interval, market)); records.append(record)
        print(f"CAISO {market}: {cursor.isoformat()} through {stop.isoformat()} retained", flush=True)
        cursor = stop
    components = pd.concat(frames, ignore_index=True)
    panel = panel_from_components(components, start, end, locations, interval, "LMP")
    provenance = {"provider": "CAISO", "units": "USD/MWh", "settlement_market": market, "interval": interval,
                  "timestamp_label": "UTC interval START; explicit official interval end retained", "source_operating_timezone": "America/Los_Angeles",
                  "dst_policy": "Use source UTC interval bounds; retain OPR_DT/OPR_HR/OPR_INTERVAL labels verbatim",
                  "component_policy": "Official DAM MW / RTM VALUE column contains this report's USD/MWh prices; retain LMP, energy MCE, congestion MCC, loss MCL and GHG MGHG with original signs",
                  "publication_timing": "Day-ahead prices relate to future operating intervals; retrieval timestamps do not reconstruct original publication or correction vintages",
                  "declared_locations": locations, "actual_locations": panel.columns.tolist(),
                  "cache_request": request, "raw_responses": records, "network_interpretation": "Observation-price network only; no transport capacities, losses or topology inferred",
                  "source_links": ["https://oasis.caiso.com/mrioasis/logon.do", "https://www.caiso.com/todays-outlook/prices"]}
    return freeze_observations(panel, components, provenance, cache_dir)


def acquire_nyiso(start, end, locations=NYISO_ZONES, *, root=DEFAULT_ROOT):
    start, end = utc_bounds(start, end)
    locations = locations_checked(locations)
    request = cache_request("NYISO", start, end, locations, "1h", "DAM zonal LBMP")
    cache_dir = Path(root) / "nyiso_dam_1h"
    _, key = _canonical_request(request)
    if (cache_dir / key).exists():
        return load_observations(cache_dir / key)
    first_month = start.tz_convert("America/New_York").tz_localize(None).to_period("M")
    last_month = (end - pd.Timedelta(nanoseconds=1)).tz_convert("America/New_York").tz_localize(None).to_period("M")
    frames, records = [], []
    for month in pd.period_range(first_month, last_month, freq="M"):
        url = NYISO_URL + month.strftime("%Y%m") + "01damlbmp_zone_csv.zip"
        body, record = raw_download(url, None, Path(root) / "raw" / "nyiso")
        frames.append(parse_nyiso(body, start, end, locations)); records.append(record)
        print(f"NYISO DAM: official {month} month archive retained", flush=True)
    components = pd.concat(frames, ignore_index=True)
    if components.duplicated(["interval_start_utc", "location", "component"]).any():
        raise ValueError("Duplicate NYISO component across monthly archives.")
    panel = panel_from_components(components, start, end, locations, "1h", "LBMP")
    provenance = {"provider": "NYISO", "units": "USD/MWh", "settlement_market": "Day-ahead zonal LBMP", "interval": "1h",
                  "timestamp_label": "Source local START label converted to UTC; interval end=start+1hour", "source_operating_timezone": "America/New_York",
                  "dst_policy": "Repeated fall-back hour resolved per zone by original ordered rows (first DST, second standard); missing/ambiguous/impossible labels fail; 23/25-hour source days retained",
                  "component_policy": "Retain source LBMP, marginal cost losses and marginal cost congestion verbatim; reported congestion sign is not silently harmonized; no energy component invented",
                  "publication_timing": "DAM schedules/prices are posted before the operating day; retrieved corrected archive is not a publication-time vintage. Source posting/correction logs must be used before forecasting",
                  "declared_locations": locations, "actual_locations": panel.columns.tolist(),
                  "location_ptids": {name: sorted(components.loc[components.location.eq(name), "PTID"].unique().tolist()) for name in locations},
                  "cache_request": request, "raw_responses": records, "network_interpretation": "Eleven domestic price zones are an observation universe, not eleven transport nodes; external proxies excluded by explicit selection",
                  "source_links": ["https://mis.nyiso.com/public/P-2Alist.htm", "https://mis.nyiso.com/public/menu.htm", "https://mis.nyiso.com/public/pdf/corrlogs/CorrLogDescription.pdf"]}
    return freeze_observations(panel, components, provenance, cache_dir)


def parse_ercot_frame(frame, start, end, locations):
    """Use official hour-ending labels and repeated-hour flags, retaining both."""
    required = {"Delivery Date", "Hour Ending", "Repeated Hour Flag", "Settlement Point", "Settlement Point Price"}
    if not required <= set(frame):
        raise ValueError("ERCOT annual workbook schema is missing required fields.")
    frame = frame.loc[frame["Settlement Point"].isin(locations)].copy()
    labels = frame["Hour Ending"].astype(str)
    if not labels.str.fullmatch(r"(?:0[1-9]|1[0-9]|2[0-4]):00").all():
        raise ValueError("Unexpected ERCOT hour-ending label.")
    repeated = frame["Repeated Hour Flag"].astype(str)
    if not repeated.isin(["N", "Y"]).all():
        raise ValueError("Unexpected ERCOT repeated-hour flag.")
    naive = pd.DatetimeIndex(pd.to_datetime(frame["Delivery Date"], format="mixed").dt.normalize() + pd.to_timedelta(labels.str[:2].astype(int) - 1, unit="h"))
    stamps = naive.tz_localize("America/Chicago", ambiguous=repeated.eq("N").to_numpy(), nonexistent="raise").tz_convert("UTC")
    # A Y marker must identify an actual repeated local hour, not an ordinary hour.
    ambiguous = naive.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="raise").isna()
    if (repeated.eq("Y").to_numpy() & ~ambiguous).any():
        raise ValueError("ERCOT repeated-hour flag occurs outside the fall-back hour.")
    frame["interval_start_utc"] = stamps
    frame["interval_end_utc"] = stamps + pd.Timedelta(hours=1)
    frame = frame.loc[frame.interval_start_utc.ge(start) & frame.interval_start_utc.lt(end)].copy()
    frame["price_usd_per_mwh"] = pd.to_numeric(frame["Settlement Point Price"], errors="raise")
    if not np.isfinite(frame.price_usd_per_mwh).all() or frame.duplicated(["interval_start_utc", "Settlement Point"]).any():
        raise ValueError("Nonfinite/duplicate ERCOT observation.")
    frame["component"] = "SPP"
    return frame.rename(columns={"Settlement Point": "location"})


def acquire_ercot(start, end, locations=ERCOT_HUBS, *, root=DEFAULT_ROOT):
    start, end = utc_bounds(start, end)
    locations = locations_checked(locations)
    request = cache_request("ERCOT", start, end, locations, "1h", "DAM hub settlement point price")
    cache_dir = Path(root) / "ercot_dam_1h"
    _, key = _canonical_request(request)
    if (cache_dir / key).exists():
        return load_observations(cache_dir / key)
    body, record = raw_download(ERCOT_LIST, {"reportTypeId": 13060}, Path(root) / "raw" / "ercot")
    docs = [item["Document"] for item in json.loads(body)["ListDocsByRptTypeRes"]["DocumentList"]]
    frames, records, publications = [], [record], []
    first_year = start.tz_convert("America/Chicago").year
    last_year = (end - pd.Timedelta(nanoseconds=1)).tz_convert("America/Chicago").year
    for year in range(first_year, last_year + 1):
        selected = [item for item in docs if item["FriendlyName"] == f"DAMLZHBSPP_{year}" and item["Extension"] == "zip" and item["SecurityStatus"] == "P"]
        if len(selected) != 1:
            raise ValueError(f"ERCOT annual public archive selection not unique for {year}.")
        doc = selected[0]
        body, record = raw_download(ERCOT_DOWNLOAD, {"doclookupId": doc["DocID"]}, Path(root) / "raw" / "ercot")
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = [name for name in archive.namelist() if name.endswith(".xlsx")]
            if len(members) != 1:
                raise ValueError("ERCOT annual ZIP must contain exactly one XLSX workbook.")
            book = pd.ExcelFile(io.BytesIO(archive.read(members[0])), engine="openpyxl")
            for sheet in book.sheet_names:
                frame = parse_ercot_frame(book.parse(sheet), start, end, locations)
                frame["source_sheet"], frame["source_member"] = sheet, members[0]
                if not frame.empty:
                    frames.append(frame)
        records.append(record); publications.append(doc)
        print(f"ERCOT DAM: official {year} annual archive retained", flush=True)
    components = pd.concat(frames, ignore_index=True)
    panel = panel_from_components(components, start, end, locations, "1h", "SPP")
    provenance = {"provider": "ERCOT", "units": "USD/MWh", "settlement_market": "DAM hub SPP", "interval": "1h",
                  "timestamp_label": "UTC interval START derived from source local hour-ending label", "source_operating_timezone": "America/Chicago",
                  "dst_policy": "HE minus one gives local start; N is the first daylight occurrence and Y the second standard occurrence during repeated hour; impossible labels fail; source flags retained",
                  "component_policy": "Total settlement-point prices retained; no energy/congestion/loss split supplied by this annual hub report",
                  "publication_timing": "Archive updated weekly; official per-archive PublishDate retained; original DAM price-publication vintages not reconstructed",
                  "declared_locations": locations, "actual_locations": panel.columns.tolist(), "cache_request": request, "raw_responses": records,
                  "official_document_publications": publications, "network_interpretation": "Hub price observation panel; no measured transport edges inferred",
                  "source_links": ["https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER"]}
    return freeze_observations(panel, components, provenance, cache_dir)


def parse_isone_zone(frame, start, end, zone, market="DA"):
    """Map this archive's 01/03 spring and 02/02X fall labels to UTC slots."""
    fields = [market + suffix for suffix in ["_LMP", "_EC", "_CC", "_MLC"]]
    if not {"Date", "Hr_End", *fields} <= set(frame):
        raise ValueError("ISO-NE zonal workbook schema is missing required fields.")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    labels = frame.Hr_End.astype(str).str.upper().str.zfill(2)
    frame["interval_start_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for date, group in frame.groupby("Date", sort=False):
        midnight = date.tz_localize("America/New_York")
        following = (date + pd.Timedelta(days=1)).tz_localize("America/New_York")
        grid = pd.date_range(midnight.tz_convert("UTC"), following.tz_convert("UTC"), freq="1h", inclusive="left")
        expected = [f"{hour:02}" for hour in range(1, 25)]
        if len(grid) == 23:
            expected.remove("02")
        elif len(grid) == 25:
            expected.insert(2, "02X")
        elif len(grid) != 24:
            raise ValueError("Unexpected ISO-NE operating-day length.")
        mapping = dict(zip(expected, grid))
        actual = labels.loc[group.index]
        if not actual.isin(mapping).all() or actual.duplicated().any():
            raise ValueError(f"Illegal/duplicate ISO-NE operating-day hour-ending marker: {zone} {date.date()} {actual.tolist()} vs {expected}.")
        frame.loc[group.index, "interval_start_utc"] = pd.DatetimeIndex(actual.map(mapping))
    frame = frame.loc[frame.interval_start_utc.ge(start) & frame.interval_start_utc.lt(end)].copy()
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if np.isinf(frame[field]).any():
            raise ValueError("Infinite ISO-NE price component.")
    frame["interval_end_utc"] = frame.interval_start_utc + pd.Timedelta(hours=1)
    frame["location"] = zone
    return frame.melt(id_vars=[column for column in frame if column not in fields], value_vars=fields, var_name="component", value_name="price_usd_per_mwh")


def acquire_isone(start, end, locations=ISONE_ZONES, *, market="DA", root=DEFAULT_ROOT):
    start, end = utc_bounds(start, end)
    locations = locations_checked(locations)
    if market not in ("DA", "RT") or not set(locations) <= set(ISONE_ZONES):
        raise ValueError("Declare DA/RT and actual ISO-NE eight zone worksheet names.")
    request = cache_request("ISO-NE", start, end, locations, "1h", market + " zonal hourly LMP")
    cache_dir = Path(root) / ("isone_" + market.lower() + "_1h")
    _, key = _canonical_request(request)
    if (cache_dir / key).exists():
        return load_observations(cache_dir / key)
    body, record = raw_download(ISONE_LIST, {"start": 0, "treenode": "zone-info"}, Path(root) / "raw" / "isone")
    docs = json.loads(body)["data"]
    frames, records, publications = [], [record], []
    first_year = start.tz_convert("America/New_York").year
    last_year = (end - pd.Timedelta(nanoseconds=1)).tz_convert("America/New_York").year
    for year in range(first_year, last_year + 1):
        selected = [item for item in docs if Path(item["path"]).name == f"{year}_smd_hourly.xlsx"]
        if len(selected) != 1:
            raise ValueError(f"ISO-NE annual archive selection not unique in first public40 records for {year}.")
        doc = selected[0]
        body, record = raw_download(urljoin(ISONE_PAGE, doc["path"]), None, Path(root) / "raw" / "isone")
        book = pd.ExcelFile(io.BytesIO(body), engine="openpyxl")
        for zone in locations:
            frame = parse_isone_zone(book.parse(zone), start, end, zone, market)
            frame["source_sheet"], frame["source_workbook"] = zone, doc["path"]
            frames.append(frame)
        records.append(record); publications.append(doc)
        print(f"ISO-NE {market}: official {year} annual archive retained", flush=True)
    components = pd.concat(frames, ignore_index=True)
    panel = panel_from_components(components, start, end, locations, "1h", market + "_LMP")
    provenance = {"provider": "ISO New England", "units": "USD/MWh", "settlement_market": market + " zonal hourly LMP", "interval": "1h",
                  "timestamp_label": "UTC interval START mapped from source Date and Hr_End", "source_operating_timezone": "America/New_York",
                  "dst_policy": "Archive spring day omits02; fall day uses02 then02X; each marker maps to an actual23/24/25-hour operating-day UTC slot; missing observations remainNaN",
                  "component_policy": "Retain source LMP, energy EC, congestion CC and marginal loss MLC; other workbook demand/weather fields preserved alongside prices",
                  "publication_timing": "Annual archive includes latest re-settled values; reported archive publication metadata retained; original available-at-time vintages not reconstructed",
                  "real_time_aggregation": "Source notes: from2017-03-01 RT_LMP is hourly average of five-minuteLMPs; requestedRT is an hourly settlement average, not a selected5mquote",
                  "declared_locations": locations, "actual_locations": panel.columns.tolist(), "cache_request": request, "raw_responses": records,
                  "official_document_publications": publications, "network_interpretation": "Eight-zone price observation panel; no measured transport edges inferred",
                  "source_links": [ISONE_PAGE]}
    return freeze_observations(panel, components, provenance, cache_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=["caiso", "nyiso", "ercot", "isone", "eia-electric", "eia-natgas"])
    parser.add_argument("--start", required=True, help="Inclusive, explicit UTC/offset timestamp")
    parser.add_argument("--end", required=True, help="Exclusive, explicit UTC/offset timestamp")
    parser.add_argument("--locations", nargs="+")
    parser.add_argument("--interval", default="1h", choices=["1h", "5min"])
    parser.add_argument("--market", default="DAM", choices=["DAM", "RTM", "DA", "RT"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    if args.provider == "caiso":
        panel, provenance, target = acquire_caiso(args.start, args.end, args.locations or CAISO_HUBS, interval=args.interval, market=args.market, root=args.root)
    elif args.provider == "nyiso":
        if args.interval != "1h" or args.market != "DAM":
            raise ValueError("NYISO adapter explicitly implements hourly DAM only.")
        panel, provenance, target = acquire_nyiso(args.start, args.end, args.locations or NYISO_ZONES, root=args.root)
    elif args.provider == "ercot":
        if args.interval != "1h" or args.market != "DAM":
            raise ValueError("ERCOT annual adapter implements hourlyDAM only.")
        panel, provenance, target = acquire_ercot(args.start, args.end, args.locations or ERCOT_HUBS, root=args.root)
    elif args.provider == "isone":
        if args.interval != "1h":
            raise ValueError("ISO-NE annual adapter implements hourly only.")
        panel, provenance, target = acquire_isone(args.start, args.end, args.locations or ISONE_ZONES, market=args.market, root=args.root)
    else:
        panel, provenance, target = acquire_eia(args.start, args.end, kind="electric" if args.provider == "eia-electric" else "natgas", products=args.locations, root=args.root)
    summary = {"snapshot_path": str(target), "snapshot_key": target.name, "cache_request": provenance["cache_request"],
               "coverage": coverage(panel, provenance["units"]), "price_file_sha256": provenance["price_file_sha256"], "raw_response_count": len(provenance["raw_responses"]),
               "raw_bytes": sum(record["bytes"] for record in provenance["raw_responses"]), "incremental_spend_usd": 0,
               "analysis_transform": "Signed " + provenance["units"] + " levels or price first differences; no arithmetic or log returns",
               "source_links": provenance["source_links"], "retrieved_at_utc": max(record["retrieved_at_utc"] for record in provenance["raw_responses"])}
    output = Path(args.summary); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
