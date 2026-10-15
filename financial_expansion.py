"""Bounded public financial panels, frozen with their declared universes.

Nothing downloads on import. The CLI acquires fixed daily and recent five-minute
panels; it does not refresh an existing snapshot, buy data, or write the catalog.
Raw HTTP bytes are retained for ECB/World Bank/FRED. Yahoo evidence is the full
library-returned frame, explicitly not raw HTTP. Missing observations stay NaN.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from archive_catalog import load_snapshot
from data_cache import cached_closes, validate_prices
from market_sources import make_source_request, save_source_snapshot


PROJECT = Path(__file__).resolve().parent
ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
WORLD_BANK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ORIGINAL = PROJECT / "data/original_notebook/720726a5d713c24f9cae262ef7ede3d619cf0d329b4ed5382941de4cfb64811f"
ECB_RAW = PROJECT / "data/pilots/ecb_daily/raw_evidence/e22b97b9fb3c9022e4f957692797dbf515e5f44f043e158251717a3b460777f1.ecb.zip"

# All are US-listed ETF exposures. Countries describe holdings, not bar venues.
ETF_SYMBOLS = """SPY QQQ DIA IWM VTI RSP MDY IJH IJR VB VXF VT VXUS VEA VWO EFA EEM ACWI
IWF IWD IWB IWS IWP IWN IWO MTUM QUAL USMV VLUE DGRO VIG VYM SCHD
XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY
SMH SOXX XBI IBB KRE KBE XRT ITB XHB IYT JETS IYR VNQ IYZ XME XOP OIH GDX GDXJ
EWC EWW EWZ EWU EWG EWQ EWI EWP EWJ EWA EWH EWS EWY EWT INDA MCHI FXI
BND AGG TLT IEF SHY TIP LQD HYG JNK EMB BNDX
GLD IAU SLV DBC PDBC USO UNG DBA CORN WEAT SOYB CPER""".split()

# The subset is declared before fetching; fill remaining slots alphabetically
# from the original universe. This is not a historical membership reconstruction.
INTRADAY_STOCK_CANDIDATES = """AAPL MSFT AMZN GOOGL GOOG META NVDA TSLA AVGO BRK-B JPM BAC WFC C GS
MS V MA AXP PYPL UNH JNJ PFE MRK ABBV BMY LLY ABT TMO DHR AMGN GILD
XOM CVX COP SLB EOG OXY MPC PSX VLO CAT DE GE RTX LMT BA HON UPS FDX UNP
CSX NSC WM ETN EMR ITW MMM DIS NFLX CMCSA T VZ TMUS WMT COST TGT HD LOW
MCD SBUX NKE KO PEP PG CL KMB PM MO ADM GIS KHC MDLZ IBM ORCL CRM ADBE
AMD INTC QCOM TXN AMAT LRCX MU KLAC PANW NOW INTU ISRG BSX MDT NEE DUK SO""".split()

FUTURES_SYMBOLS = """CL=F BZ=F NG=F RB=F HO=F GC=F SI=F HG=F PL=F PA=F
ZC=F ZW=F ZS=F ZM=F ZL=F ZO=F ZR=F KC=F SB=F CC=F CT=F OJ=F LE=F HE=F GF=F LBS=F LBR=F""".split()
FUTURES_UNITS = {
    "CL=F": "USD/barrel", "BZ=F": "USD/barrel", "NG=F": "USD/MMBtu",
    "RB=F": "USD/gallon", "HO=F": "USD/gallon", "GC=F": "USD/troy ounce",
    "SI=F": "USD/troy ounce", "HG=F": "USD/pound", "PL=F": "USD/troy ounce",
    "PA=F": "USD/troy ounce", "ZC=F": "US cents/bushel", "ZW=F": "US cents/bushel",
    "ZS=F": "US cents/bushel", "ZM=F": "USD/short ton", "ZL=F": "US cents/pound",
    "ZO=F": "US cents/bushel", "ZR=F": "USD/hundredweight", "KC=F": "US cents/pound",
    "SB=F": "US cents/pound", "CC=F": "USD/metric tonne", "CT=F": "US cents/pound",
    "OJ=F": "US cents/pound", "LE=F": "US cents/pound", "HE=F": "US cents/pound",
    "GF=F": "US cents/pound", "LBS=F": "USD/thousand board feet; legacy ticker coverage uncertain",
    "LBR=F": "USD/thousand board feet; separate current lumber ticker, not spliced to LBS=F",
}
FX_PAIRS = {
    "EURUSD=X": ("EUR", True), "GBPUSD=X": ("GBP", True),
    "AUDUSD=X": ("AUD", True), "NZDUSD=X": ("NZD", True),
    "JPY=X": ("JPY", False), "CHF=X": ("CHF", False),
    "CAD=X": ("CAD", False), "CNY=X": ("CNY", False),
    "HKD=X": ("HKD", False), "SGD=X": ("SGD", False),
    "INR=X": ("INR", False), "MXN=X": ("MXN", False),
    "BRL=X": ("BRL", False), "ZAR=X": ("ZAR", False),
    "TRY=X": ("TRY", False), "SEK=X": ("SEK", False),
    "NOK=X": ("NOK", False), "DKK=X": ("DKK", False),
    "PLN=X": ("PLN", False), "KRW=X": ("KRW", False),
}
FRED_SERIES = ["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def versions():
    import importlib.metadata
    result = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "yfinance", "requests", "openpyxl"):
        result[package] = importlib.metadata.version(package)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(body)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class StorageBudget:
    """Stop before publishing evidence that exceeds the declared local cap."""

    def __init__(self, root, max_bytes):
        self.root, self.max_bytes = Path(root), int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def check(self, additional):
        if self.used() + int(additional) > self.max_bytes:
            raise RuntimeError("Financial acquisition storage cap reached; no further publication.")

    def preserve(self, body, suffix):
        path = self.root / "raw_evidence" / f"{sha(body)}{suffix}"
        if path.exists():
            if sha(path.read_bytes()) != sha(body):
                raise ValueError("Existing source evidence checksum mismatch.")
        else:
            self.check(len(body))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(body)
        return {"path": str(path.resolve()), "sha256": sha(body), "bytes": len(body)}


def coverage(prices):
    result = {}
    for symbol in prices:
        series = prices[symbol].dropna()
        result[symbol] = {
            "observations": len(series), "missing": int(prices[symbol].isna().sum()),
            "first": series.index[0].isoformat() if len(series) else None,
            "last": series.index[-1].isoformat() if len(series) else None,
        }
    values = prices.to_numpy(dtype=float)
    return {
        "rows": len(prices), "declared_symbols": len(prices.columns),
        "observed_symbols": int(prices.notna().any().sum()),
        "observed_cells": int(np.isfinite(values).sum()),
        "missing_cells": int(np.isnan(values).sum()),
        "negative_cells": int((values < 0).sum()), "zero_cells": int((values == 0).sum()),
        "first_label": prices.index[0].isoformat(), "last_label": prices.index[-1].isoformat(),
        "per_symbol": result,
    }


def request_key(request):
    normalized = dict(request, symbols=sorted(request["symbols"]))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return normalized, sha(encoded)


def freeze_panel(prices, provenance, dataset, budget, *, positive):
    """Freeze the observed universe and disclose the complete source request.

    Positive panels use save_source_snapshot; signed panels use cached_closes.
    A wholly failed symbol is explicitly recorded in requested_universe and
    unavailable_symbols, while the core cache request contains observed symbols.
    Missing cells within observed series remain NaN; no shared API is modified.
    """
    validate_prices(prices)
    if set(prices.columns) != set(provenance["symbols"]):
        raise ValueError("Prepared columns differ from the declared universe.")
    values = prices.to_numpy(dtype=float)
    if positive and (values[np.isfinite(values)] <= 0).any():
        raise ValueError("A supposedly positive panel contains nonpositive observations.")
    unavailable = prices.columns[prices.isna().all()].tolist()
    requested_universe = list(prices.columns)
    requested_coverage = coverage(prices)
    prices = prices.loc[:, prices.notna().any()].copy()
    if prices.shape[1] == 0:
        raise ValueError("No observed symbols; the failed request has no research snapshot.")
    provenance = dict(provenance, symbols=list(prices.columns), coverage=requested_coverage,
                      cached_coverage=coverage(prices), package_versions=versions(),
                      requested_universe=requested_universe, unavailable_symbols=unavailable,
                      observed_universe_policy="Cache request contains observed symbols; full original source request and wholly unavailable symbols remain explicit in companion provenance.")
    request = make_source_request(provenance, universe_as_of=dataset["universe_as_of"])
    normalized, key = request_key(request)
    target = budget.root / "snapshots" / key
    if target.exists():
        loaded, manifest = load_snapshot(target)
        pd.testing.assert_frame_equal(loaded, prices, check_exact=True, check_freq=False)
        existing = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
        if existing.get("requested_universe", list(loaded.columns)) != requested_universe or existing.get("unavailable_symbols", []) != unavailable:
            raise ValueError("Frozen source-request universe/failures differ; preserve its original disclosure.")
    else:
        # Reserve the actual compressed observations before publishing the cache.
        serialized = prices.to_csv(float_format="%.17g").encode("utf-8")
        budget.check(len(gzip.compress(serialized)) + 200_000)
        provenance["dataset_id"] = dataset["dataset_id"]
        if positive:
            _, manifest, _, _ = save_source_snapshot(request, lambda: (prices, provenance), cache_dir=target.parent)
        else:
            _, manifest, _ = cached_closes(request, lambda: prices, cache_dir=target.parent)
            write_json(target / "provenance.json", dict(provenance, cache_request=normalized, price_file_sha256=manifest["sha256"]))
        loaded, _ = load_snapshot(target)
        pd.testing.assert_frame_equal(loaded, prices, check_exact=True, check_freq=False)
    result = dict(dataset, snapshot_path=str(target.resolve()), source_url=provenance["source_url"],
                  coverage=requested_coverage, cached_coverage=coverage(loaded), requested_universe=requested_universe,
                  actual_universe=list(loaded.columns),
                  unavailable_symbols=unavailable, sha256=manifest["sha256"], status="partial" if unavailable else "acquired")
    result["source_snapshot_dataset_id"] = json.loads((target / "provenance.json").read_text(encoding="utf-8"))["dataset_id"]
    print(f"{dataset['dataset_id']}: {len(prices):,} rows, {result['coverage']['observed_symbols']}/{len(requested_universe)} series, {result['coverage']['observed_cells']:,} observations", flush=True)
    return result


def parse_ecb(body):
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        csv_body = archive.read("eurofxref-hist.csv")
    table = pd.read_csv(io.BytesIO(csv_body), na_values=["N/A", "-"])
    table.columns = table.columns.str.strip()
    currencies = [column for column in table if re.fullmatch(r"[A-Z]{3}", column)]
    prices = table[currencies].apply(pd.to_numeric, errors="raise").astype(float)
    prices.index = pd.DatetimeIndex(pd.to_datetime(table["Date"], format="%Y-%m-%d"), name="date")
    prices = prices.sort_index()
    prices.columns.name = "currency"
    validate_prices(prices)
    return prices, sha(csv_body)


def parse_world_bank(body):
    """Read names/units from the workbook; accept only explicit blank markers."""
    workbook = pd.ExcelFile(io.BytesIO(body))
    table = pd.read_excel(workbook, "Monthly Prices", header=None)
    name_rows = table.index[table.iloc[:, 1].eq("Crude oil, average")].tolist()
    if len(name_rows) != 1:
        raise ValueError("Cannot identify the World Bank monthly price header.")
    header = name_rows[0]
    names = [str(value).strip() for value in table.iloc[header, 1:]]
    if len(set(names)) != len(names) or any(name == "nan" for name in names):
        raise ValueError("World Bank price column labels are invalid.")
    units = dict(zip(names, [str(value).strip() for value in table.iloc[header + 1, 1:]]))
    mask = table.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}M\d{2}")
    labels = table.loc[mask].iloc[:, 0].astype(str)
    dates = pd.to_datetime(labels.str.replace("M", "-", regex=False) + "-01", format="%Y-%m-%d")
    data = table.loc[mask].iloc[:, 1:].copy()
    data.columns = names
    data = data.mask(data.isin(["\u2026", "..."]))
    prices = data.apply(pd.to_numeric, errors="raise").astype(float)
    prices.index = pd.DatetimeIndex(dates, name="month")
    # Preserve the entire workbook, but avoid treating its gas index as a price.
    prices = prices.drop(columns=["Natural gas index"])
    units.pop("Natural gas index")
    prices.columns.name = "commodity"
    validate_prices(prices)
    mismatch_rows = len(pd.read_excel(workbook, "Mismatch Details", header=3)) if "Mismatch Details" in workbook.sheet_names else 0
    return prices, units, {
        "workbook_sheets": workbook.sheet_names, "update_label": str(table.iloc[3, 0]),
        "mismatch_details_rows": mismatch_rows, "excluded_index": "Natural gas index (2010=100)",
        "descriptions_sheet": "Description; preserved verbatim in raw workbook",
    }


def parse_fred(body):
    table = pd.read_csv(io.BytesIO(body), na_values=["."])
    if list(table.columns) != ["observation_date", *FRED_SERIES]:
        raise ValueError("FRED returned an unexpected series universe.")
    prices = table[FRED_SERIES].apply(pd.to_numeric, errors="raise").astype(float)
    prices.index = pd.DatetimeIndex(pd.to_datetime(table["observation_date"], format="%Y-%m-%d"), name="date")
    prices.columns.name = "series"
    validate_prices(prices)
    return prices


def extract_yahoo_closes(frame, symbols, *, daily):
    if frame.empty:
        return pd.DataFrame(columns=symbols, dtype=float)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2 or "Close" not in frame.columns.get_level_values(0):
        raise ValueError("Yahoo full frame lacks (field, ticker) Close columns.")
    prices = frame["Close"].reindex(columns=symbols).astype(float)
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Yahoo returned invalid timestamps.")
    if daily:
        # ignore_tz=True requests provider-local calendar labels, not UTC instants.
        if prices.index.tz is not None:
            raise ValueError("Yahoo ignored the explicit daily ignore_tz setting.")
        prices.index = prices.index.normalize()
    else:
        if prices.index.tz is None:
            raise ValueError("Yahoo intraday timestamps are missing their offsets.")
        prices.index = prices.index.tz_convert("UTC")
    prices.index.name = "date" if daily else "timestamp"
    prices.columns.name = "symbol"
    validate_prices(prices)
    return prices


def yahoo_panel(symbols, start, end, interval, budget, *, auto_adjust, batch_size=25, download=None):
    """Fetch finite batches; log wholly unavailable symbols and preserve all columns."""
    if download is None:
        import yfinance as yf
        download = yf.download
    daily = interval == "1d"
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if daily != (first.tzinfo is None and last.tzinfo is None):
        raise ValueError("Daily bounds are naive; intraday bounds need explicit offsets.")
    settings = {
        "start": first, "end": last, "interval": interval, "auto_adjust": auto_adjust,
        "back_adjust": False, "prepost": False, "repair": False, "keepna": True,
        "ignore_tz": daily, "actions": False, "rounding": False,
        "threads": 4, "progress": False, "group_by": "column", "multi_level_index": True,
        "timeout": 20,
    }
    frames, evidence, errors = [], [], []
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset:offset + batch_size]
        print(f"Yahoo {interval}: batch {offset // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}, {len(batch)} symbols", flush=True)
        retrieved = utc_now()
        try:
            frame = download(tickers=batch, **settings)
            csv_bytes = frame.to_csv(float_format="%.17g").encode("utf-8")
            stored = budget.preserve(gzip.compress(csv_bytes, mtime=0), ".yfinance-frame.csv.gz")
            evidence.append(dict(stored, symbols=batch, retrieved_at_utc=retrieved, returned_csv_sha256=sha(csv_bytes)))
            prices = extract_yahoo_closes(frame, batch, daily=daily)
            if len(prices):
                frames.append(prices)
            empty = batch if prices.empty else [symbol for symbol in batch if prices[symbol].isna().all()]
            errors.extend({"symbol": symbol, "reason": "No in-range Close observations returned by yfinance; provider stderr may have details."} for symbol in empty)
        except (RuntimeError, MemoryError):
            raise
        except Exception as error:
            errors.extend({"symbol": symbol, "reason": f"{type(error).__name__}: {error}"} for symbol in batch)
    if not frames:
        raise ValueError(f"No Yahoo observations for declared universe ({len(symbols)} symbols).")
    prices = pd.concat(frames, axis=1).reindex(columns=symbols).sort_index()
    prices = prices.loc[(prices.index >= first) & (prices.index < last)]
    prices.index.name = "date" if daily else "timestamp"
    prices.columns.name = "symbol"
    validate_prices(prices)
    return prices, {
        "source": "Yahoo via yfinance", "source_url": YAHOO_URL, "symbols": symbols,
        "start": first.isoformat(), "end_exclusive": last.isoformat(), "interval": interval,
        "adjustment": "yfinance auto_adjust=True" if auto_adjust else "provider Close; auto_adjust=False",
        "feed": "Yahoo public interface", "timezone_policy": "provider-local daily calendar labels" if daily else "UTC instants",
        "retrieved_at_utc": utc_now(), "settings": {key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in settings.items()},
        "raw_response_sha256": None, "returned_frames": evidence, "unavailable_log": errors,
        "evidence_limit": "Full yfinance-returned frames are preserved; yfinance does not expose raw HTTP responses.",
        "missing_policy": "No forward/zero filling; union of actual returned timestamps, declared symbols retained.",
    }


def normalize_fx(prices):
    output = pd.DataFrame(index=prices.index)
    for ticker, (currency, invert) in FX_PAIRS.items():
        series = prices[ticker]
        if (series.dropna() <= 0).any():
            raise ValueError("Cannot invert nonpositive FX quotes.")
        output[f"{currency}_per_USD"] = 1.0 / series if invert else series
    output.columns.name = "currency_per_USD"
    return output


def acquire(args):
    import requests
    budget = StorageBudget(args.data_dir, args.max_mb * 1_000_000)
    results = {"created_at_utc": utc_now(), "incremental_spend_usd": 0,
               "storage_cap_bytes": budget.max_bytes, "datasets": [], "failed_datasets": [],
               "dataset_definition": "Declared full universe, interval and intended span; files/tickers/windows are not datasets."}
    output = Path(args.output)

    def record(dataset_id, family, interval, transform, units, notes, prepare, positive, bar_seconds=None, requested_universe=None, intended_start=None, intended_end=None):
        if args.only and dataset_id not in args.only:
            return
        dataset = dict(dataset_id=dataset_id, family=family, interval=interval, transform=transform,
                       units=units, notes=notes, universe_as_of="explicit research list fixed 2026-10-08; not historical membership")
        if bar_seconds:
            dataset["bar_seconds"] = bar_seconds
        if requested_universe is not None:
            dataset["requested_universe"] = list(requested_universe)
        if intended_start is not None:
            dataset["intended_start"] = intended_start
            dataset["intended_end_exclusive"] = intended_end
        try:
            prices, provenance = prepare()
            dataset["intended_start"] = provenance["start"]
            dataset["intended_end_exclusive"] = provenance["end_exclusive"]
            if isinstance(units, str) and units == "from source workbook":
                dataset["units"] = provenance["units_by_symbol"]
            results["datasets"].append(freeze_panel(prices, provenance, dataset, budget, positive=positive))
        except Exception as error:
            results["failed_datasets"].append(dict(dataset, error=f"{type(error).__name__}: {error}"))
            print(f"{dataset_id} FAILED: {type(error).__name__}: {error}", flush=True)
        results["stored_bytes"] = budget.used()
        write_json(output, results)

    def ecb():
        body = ECB_RAW.read_bytes()
        evidence = budget.preserve(body, ".ecb.zip")
        prices, csv_hash = parse_ecb(body)
        return prices, dict(source="European Central Bank reference rates", source_url=ECB_URL,
            symbols=prices.columns.tolist(), start="1999-01-04", end_exclusive=args.end,
            interval="1d", adjustment="not applicable: official reference rates", feed="ECB eurofxref-hist.csv",
            timezone_policy="naive reference dates; usual publication around 16:00 CET, not midnight quotes",
            retrieved_at_utc="2026-10-07 (original pilot retrieval; reused frozen raw archive)", prepared_at_utc=utc_now(),
            raw_evidence=evidence, archive_member_sha256=csv_hash, base_currency="EUR",
            quote_convention="foreign-currency units per EUR", units_by_symbol={symbol: f"{symbol}/EUR" for symbol in prices},
            notes="Includes retired/suspended currencies with partial coverage. EUR is the numeraire, not a constant asset. Reference quotes are informational; no intraday claim.")

    record("ecb_fx_all_history_daily", "fx_reference", "1d", "arithmetic_return", "foreign-currency units/EUR",
           "All 41 currencies present in frozen archive; retired/suspended series stay missing. Same EUR numeraire creates common exposure.", ecb, True)

    def world_bank():
        page_response = requests.get(WORLD_BANK_PAGE, timeout=30)
        page_response.raise_for_status()
        page_evidence = budget.preserve(page_response.content, ".worldbank-source-page.html")
        links = re.findall(r"href=[\"']([^\"']*CMO-Historical-Data-Monthly.xlsx[^\"']*)", page_response.text)
        if not links or len(set(links)) != 1 or not links[0].startswith("https://thedocs.worldbank.org/"):
            raise ValueError("Official World Bank page does not identify one monthly workbook.")
        response = requests.get(links[0], timeout=45)
        response.raise_for_status()
        evidence = budget.preserve(response.content, ".worldbank-monthly.xlsx")
        prices, units, details = parse_world_bank(response.content)
        return prices, dict(source="World Bank Pink Sheet monthly prices", source_url=links[0],
            symbols=prices.columns.tolist(), start="1960-01-01", end_exclusive=args.end,
            interval="1mo", adjustment="nominal source levels; no deflation", feed="Monthly Prices workbook sheet",
            timezone_policy="naive first-of-month period labels, not availability timestamps",
            retrieved_at_utc=utc_now(), raw_evidence=evidence, discovery_page=page_evidence,
            units_by_symbol=units, **details,
            notes="Mixed physical assessments, averages and some futures inputs; definitions can change. Crude average and tea average overlap component series. Monthly aggregates are known after the period; revised history is not point-in-time. Workbook mismatch-details entries are source quality flags, not corrected here.")

    record("worldbank_commodity_monthly_1960", "physical_commodity_assessments", "1mo", "first_difference", "from source workbook",
           "70 source price columns, excluding gas index; source units differ. First differences avoid denominator issues. Standardized Pearson dependence is unit-scale invariant, but aggregate/component overlap can inflate it.", world_bank, False)

    def fred():
        parameters = {"id": ",".join(FRED_SERIES), "cosd": "1980-01-01", "coed": (pd.Timestamp(args.end) - pd.Timedelta(days=1)).date().isoformat()}
        response = requests.get(FRED_URL, params=parameters, timeout=40)
        response.raise_for_status()
        evidence = budget.preserve(response.content, ".fred-energy.csv")
        prices = parse_fred(response.content)
        return prices, dict(source="EIA spot prices via FRED", source_url=response.url, symbols=FRED_SERIES,
            start="1980-01-01", end_exclusive=args.end, interval="1d", adjustment="not seasonally adjusted; nominal spot assessments",
            feed="FRED CSV / EIA", timezone_policy="naive observation-date labels; publication occurs later",
            retrieved_at_utc=utc_now(), raw_evidence=evidence, request_parameters=parameters,
            units_by_symbol={"DCOILWTICO": "USD/barrel", "DCOILBRENTEU": "USD/barrel", "DHHNGSP": "USD/MMBtu"},
            series_urls=[f"https://fred.stlouisfed.org/series/{series}" for series in FRED_SERIES],
            notes="WTI can be negative. Missing holidays/source gaps remain NaN; current revised history is not point-in-time. EIA series pages identify daily spot assessments, not futures.")

    record("fred_eia_energy_spot_daily", "energy_spot", "1d", "first_difference",
           {"DCOILWTICO": "USD/barrel", "DCOILBRENTEU": "USD/barrel", "DHHNGSP": "USD/MMBtu"},
           "Three long daily physical spot assessments; signed levels retained, differences in source units. Oil and gas are separate commodity exposures.", fred, False)

    stock_prices, _ = load_snapshot(ORIGINAL)
    stock_symbols = sorted(stock_prices.columns.tolist())
    subset = [symbol for symbol in INTRADAY_STOCK_CANDIDATES if symbol in stock_symbols]
    subset += [symbol for symbol in stock_symbols if symbol not in subset][:max(0, 100 - len(subset))]
    subset = subset[:100]
    if len(stock_symbols) != 503:
        raise ValueError("Expected the unchanged original 503-symbol universe.")

    for prefix, family, symbols, units, adjusted, positive, notes in [
        ("yahoo_etf", "us_listed_etf", ETF_SYMBOLS, "USD/share; yfinance adjusted close levels", True, True,
         "US-listed funds across equity styles/sectors/countries, bonds and commodity exposures; overlapping holdings and proxies are not independent physical markets. Regular US session bars only."),
        ("yahoo_original_stocks", "us_equity_original_universe", stock_symbols, "USD/share; yfinance adjusted close levels", True, True,
         "New download of original 503-symbol list, not reconstructed historical membership; corporate actions/listing changes and failed symbols remain explicit. Original snapshot is unchanged. Regular US session bars only."),
        ("yahoo_fx_usd", "fx_indicative", list(FX_PAIRS), "foreign-currency units/USD after explicit inversion where needed", False, True,
         "Indicative Yahoo FX, separate from ECB and equities. Four USD-per-foreign quote series are inverted to one USD numeraire; 24-hour quote sessions differ from US equities."),
        ("yahoo_commodity_futures", "commodity_futures_proxy", FUTURES_SYMBOLS, FUTURES_UNITS, False, False,
         "Provider nearby/continuous futures proxies, not physical spot; contract selection/roll/back-adjustment is not independently verified. Signed levels retained. Quote units are conventional contract units, pending provider contract reconciliation."),
    ]:
        for interval in args.intervals:
            chosen = subset if prefix == "yahoo_original_stocks" and interval != "1d" else symbols
            start = args.daily_start if interval == "1d" else (args.minute_start if interval == "1m" else args.intraday_start) + "T00:00:00Z"
            end = args.end if interval == "1d" else args.end + "T00:00:00Z"
            transform = "first_difference" if family == "commodity_futures_proxy" else "arithmetic_return"

            def prepare(chosen=chosen, start=start, end=end, interval=interval, adjusted=adjusted, family=family, notes=notes):
                prices, provenance = yahoo_panel(chosen, start, end, interval, budget, auto_adjust=adjusted)
                provenance["notes"] = notes
                if family == "fx_indicative":
                    prices = normalize_fx(prices)
                    provenance["provider_symbols"] = chosen
                    provenance["symbols"] = prices.columns.tolist()
                    provenance["adjustment"] += "; explicit quote inversion to foreign-currency/USD"
                    provenance["quote_transformation"] = {ticker: {"output": f"{currency}_per_USD", "inverse": invert} for ticker, (currency, invert) in FX_PAIRS.items()}
                    provenance["unavailable_log"] = [dict(item, output_symbol=f"{FX_PAIRS[item['symbol']][0]}_per_USD") for item in provenance["unavailable_log"]]
                if family == "commodity_futures_proxy":
                    provenance["units_by_symbol"] = FUTURES_UNITS
                return prices, provenance

            daily_id = "daily_usd_numeraire" if family == "fx_indicative" else "daily_27ticker" if family == "commodity_futures_proxy" else "daily_long"
            interval_id = daily_id if interval == "1d" else interval + "_" + start[:10].replace("-", "") + "_" + end[:10].replace("-", "")
            record(prefix + "_" + interval_id, family, interval,
                   transform, units, notes, prepare, positive, 60 if interval == "1m" else 300 if interval == "5m" else None,
                   [f"{currency}_per_USD" for currency, _ in FX_PAIRS.values()] if family == "fx_indicative" else chosen,
                   start, end)

    results["completed_at_utc"] = utc_now()
    results["stored_bytes"] = budget.used()
    results["dataset_count"] = len(results["datasets"])
    results["total_observed_cells"] = sum(row["coverage"]["observed_cells"] for row in results["datasets"])
    write_json(output, results)
    print(f"Completed {results['dataset_count']} panels; {results['stored_bytes'] / 1e6:.2f} MB stored; additional spend $0", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/expanded/financial")
    parser.add_argument("--output", type=Path, default=PROJECT / "output/expansion/financial.json")
    parser.add_argument("--daily-start", default="1950-01-01")
    parser.add_argument("--intraday-start", default="2026-08-17")
    parser.add_argument("--minute-start", default="2026-10-02")
    parser.add_argument("--intervals", nargs="+", choices=["1d", "5m", "1m"], default=["1d", "5m", "1m"])
    parser.add_argument("--end", default="2026-10-09", help="Fixed exclusive date; no automatic rolling refresh.")
    parser.add_argument("--max-mb", type=int, default=500)
    parser.add_argument("--only", nargs="+", help="Acquire only these declared dataset IDs; use a separate output JSON for supplementary runs.")
    args = parser.parse_args()
    if args.max_mb <= 0 or pd.Timestamp(args.daily_start) >= pd.Timestamp(args.end) or pd.Timestamp(args.intraday_start) >= pd.Timestamp(args.end) or pd.Timestamp(args.minute_start) >= pd.Timestamp(args.end):
        parser.error("Positive storage budget and increasing explicit bounds are required.")
    acquire(args)


if __name__ == "__main__":
    main()
