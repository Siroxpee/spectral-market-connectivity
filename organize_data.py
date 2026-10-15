"""Organize frozen active panels and materialize every baseline-transformed row.

No acquisition, catalog writes, source edits, global column deletion or filling.
Run ``python organize_data.py --self-test`` for small policy checks, then run
without arguments to prepare the validated active inventory into seven groups.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

# Import the established transform, including its single-thread BLAS policy,
# before importing NumPy/Pandas. Do not duplicate the baseline mathematics.
from expanded_spectral import transform_panel
from archive_catalog import load_snapshot
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GROUPS = ("stocks", "etfs", "forex", "crypto", "commodities", "electricity", "grid_operations")
LABELS = dict(zip(GROUPS, ("Stocks", "ETFs", "Forex", "Crypto", "Commodities", "Electricity", "Grid operations")))
CHUNK_ROWS = 20_000
VERSION = 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"project_relative_path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def group_for(spec):
    family = spec["family"]
    if family in ("stocks", "us_equity_original_universe"): return "stocks"
    if family in ("etfs", "us_listed_etf"): return "etfs"
    if family in ("forex", "fx_reference", "fx_indicative"): return "forex"
    if family == "crypto": return "crypto"
    if family in ("energy_spot", "commodity_futures_proxy", "physical_commodity_assessments", "natural_gas"): return "commodities"
    if family in ("electricity", "electricity_retail"): return "electricity"
    if family == "grid_operations": return "grid_operations"
    raise ValueError(f"Unknown family must be assigned explicitly: {family}")


def active_inventory(expected_panels):
    """The report CSV selects active descriptors; inventory also retains pilots."""
    inventory_path = ROOT / "output/expansion/inventory.json"
    active_path = ROOT / "output/expansion/Dataset_Inventory.csv"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    with active_path.open(encoding="utf-8-sig", newline="") as handle:
        active = list(csv.DictReader(handle))
    wanted = {row["dataset"]: row["snapshot_key"] for row in active}
    if len(wanted) != len(active) or len(active) != expected_panels:
        raise ValueError("Active inventory has duplicate IDs or unexpected panel count.")
    items = [item for item in inventory["datasets"] if item["specification"]["dataset_id"] in wanted]
    if len(items) != len(wanted): raise ValueError("Active report IDs do not match frozen inventory descriptors.")
    for item in items:
        spec = item["specification"]
        if not re.fullmatch(r"[a-z0-9_-]+", spec["dataset_id"]): raise ValueError("Unsafe dataset ID.")
        if Path(spec["snapshot_path"]).name != wanted[spec["dataset_id"]]:
            raise ValueError("Active dataset points to a different frozen snapshot.")
    return items, inventory, [reference(inventory_path), reference(active_path)]


def timestamp_policy(spec, manifest, index):
    """Source labels stay intact; only verified interval boundaries are derived."""
    seconds = spec.get("bar_seconds")
    policy = dict(source_index_name=manifest.get("index_name"),
                  source_timezone=str(index.tz) if index.tz is not None else None,
                  source_policy=manifest["request"]["timezone_policy"],
                  csv_source_column="source_timestamp", source_labels_preserved=True,
                  publication_or_latency_history="not reconstructed", bar_seconds=seconds,
                  interval_boundary_kind="unverified", interval_start_rule=None, interval_end_rule=None)
    if spec["family"] == "grid_operations":
        if seconds != 3600 or index.tz is None: raise ValueError("EIA grid needs explicit UTC hourly labels.")
        policy.update(interval_boundary_kind="source_hour_ending_utc",
                      interval_start_rule="source_timestamp - 1 hour", interval_end_rule="source_timestamp",
                      legacy_index_name_correction="The frozen interval_start_utc name is incorrect; source labels mean HOUR ENDING. Native files remain unchanged.")
    elif spec["family"] == "crypto" and "bar END" in manifest["request"]["timezone_policy"]:
        if not seconds or index.tz is None: raise ValueError("Crypto bar-end labels need explicit UTC interval length.")
        policy.update(interval_boundary_kind="source_bar_end_utc",
                      interval_start_rule="source_timestamp - bar_seconds", interval_end_rule="source_timestamp",
                      earliest_close_availability="bar end; publication latency unmeasured")
    elif manifest.get("index_name") == "interval_start_utc" and spec["family"] == "electricity":
        if not seconds or index.tz is None: raise ValueError("Interval-start electricity needs explicit UTC interval length.")
        policy.update(interval_boundary_kind="source_interval_start_utc",
                      interval_start_rule="source_timestamp", interval_end_rule="source_timestamp + bar_seconds")
    elif index.tz is None:
        policy.update(interval_boundary_kind="calendar_or_reference_label",
                      calendar_label_warning="Naive dates/months are reference labels, not invented midnight UTC quotes or publication times.")
    else:
        policy["unverified_boundary_warning"] = "Provider UTC labels retained; start/end bucket convention is not independently reconciled. No interval boundaries are invented."
    return policy


def boundaries(index, policy):
    seconds = pd.Timedelta(seconds=policy["bar_seconds"] or 0)
    kind = policy["interval_boundary_kind"]
    if kind in ("source_hour_ending_utc", "source_bar_end_utc"):
        return index - seconds, index
    if kind == "source_interval_start_utc": return index, index + seconds
    return None, None


def rights_for(spec):
    if spec["family"] == "crypto":
        return {"source": "Binance Vision", "scope": "local noncommercial research",
                "derivative_redistribution": "Binance Vision attribution and CC BY-NC-SA 4.0 required by the recorded dataset terms; commercial use requires a separate agreement.",
                "terms_url": "https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md",
                "evidence": "output/expansion/crypto_sources.json"}
    return {"scope": "local research organization only; no new source permissions granted",
            "redistribution_and_publication": "Preserve source attribution and consult native provenance/provider reviews; permission is not inferred from free download access.",
            "evidence": "native raw/reference.json and output/expansion/financial_provider_review.json where applicable"}


def entity_identifier(spec, source_request, symbol):
    """An established Yahoo instrument label, not a guessed legal/token identity."""
    financial = spec["family"] in ("stocks", "us_equity_original_universe", "etfs", "us_listed_etf")
    source = source_request["source"].lower()
    if financial and ("yahoo" in source or "yfinance" in source):
        return "yahoo:ticker:" + symbol
    return None


def supplementary_paths(spec, native):
    paths = list(native.glob("*.json"))
    family = spec["family"]
    if family == "crypto":
        names = ("crypto_sources.json", "crypto_results.json", "crypto_acquisition.json", "crypto_identity_audit.json", "crypto_quality_flags.json", "crypto_completion.json")
        paths += [ROOT / "output/expansion" / name for name in names]
        paths += [ROOT / "data/expanded/crypto" / name for name in ("universe.json", "checkpoint.json", "minute_checkpoint.json", "identity_review.json")]
    elif family == "grid_operations": paths.append(ROOT / "data/expanded/grid_operations/series_metadata.json")
    elif family == "electricity_retail": paths.append(ROOT / "output/expansion/retail_electricity.json")
    elif family in ("electricity", "natural_gas"): paths.append(ROOT / "output/expansion/electricity_results.json")
    elif spec["dataset_id"] != "original_stocks_daily":
        paths += [ROOT / "output/expansion" / name for name in ("financial.json", "financial_provider_review.json", "financial_validation.json")]
    else: paths += [ROOT / "source_files.json", ROOT / "output/reproduction/summary.json"]
    return sorted({p.resolve() for p in paths if p.is_file()})


def matrix_quality(frame):
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any(): raise ValueError("Infinity in organized numeric matrix.")
    observed = frame.notna().sum()
    return dict(rows=len(frame), variables=len(frame.columns), observed_cells=int(observed.sum()),
                missing_cells=int(frame.isna().sum().sum()), negative_cells=int((values < 0).sum()),
                zero_cells=int((values == 0).sum()), all_missing_columns=observed.index[observed == 0].tolist(),
                first_label=frame.index[0].isoformat() if len(frame) else None,
                last_label=frame.index[-1].isoformat() if len(frame) else None,
                per_variable={str(symbol): {"observations": int(count), "missing": int(len(frame)-count)} for symbol, count in observed.items()})


def write_prepared(path, changes, variables, policy, identifier):
    """Deterministic gzip; precision sufficient for exact float64 reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0, compresslevel=3) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                for start in range(0, len(changes), CHUNK_ROWS):
                    block = changes.iloc[start:start+CHUNK_ROWS].rename(columns=variables, copy=False)
                    first, last = boundaries(block.index, policy)
                    if first is not None:
                        block = block.copy()
                        block.insert(0, "interval_end_utc", last)
                        block.insert(0, "interval_start_utc", first)
                    block.to_csv(text, index=True, index_label="source_timestamp", header=start == 0,
                                 float_format="%.17g", na_rep="", lineterminator="\n")
                    if start and start % 100_000 == 0:
                        print(f"{identifier}: wrote {min(start+CHUNK_ROWS,len(changes)):,}/{len(changes):,} prepared rows", flush=True)
    temporary.replace(path)


def verify_prepared(path, changes, variables, policy):
    """Verify every stored numeric value, timestamp and interval boundary."""
    ids = list(variables.values())
    position = observed = 0
    for block in pd.read_csv(path, chunksize=CHUNK_ROWS, dtype={key: "float64" for key in ids}, float_precision="round_trip"):
        expected = changes.iloc[position:position+len(block)]
        if not np.array_equal(block[ids].to_numpy(), expected.to_numpy(dtype=float, na_value=np.nan), equal_nan=True):
            raise AssertionError("Prepared numeric values changed on CSV round trip.")
        actual_index = pd.DatetimeIndex(pd.to_datetime(block["source_timestamp"], utc=changes.index.tz is not None))
        if not np.array_equal(actual_index.as_unit("ns").asi8, expected.index.as_unit("ns").asi8):
            raise AssertionError("Prepared source timestamps changed.")
        first, last = boundaries(expected.index, policy)
        if first is not None:
            for name, expected_index in (("interval_start_utc", first), ("interval_end_utc", last)):
                actual = pd.DatetimeIndex(pd.to_datetime(block[name], utc=True))
                if not np.array_equal(actual.as_unit("ns").asi8, expected_index.as_unit("ns").asi8):
                    raise AssertionError("Incorrect prepared interval boundaries.")
        observed += int(block[ids].notna().sum().sum())
        position += len(block)
    if position != len(changes): raise AssertionError("Prepared output row count changed.")
    return {"status": "pass", "rows_checked": position, "observed_values_checked": observed,
            "numeric_comparison": "exact float64 equality including NaN", "source_timestamps_and_verified_boundaries": "exact"}


def organize_one(item, groups_root):
    spec = item["specification"]
    identifier, group = spec["dataset_id"], group_for(spec)
    native = Path(spec["snapshot_path"]).resolve()
    print(f"Preparing {group}/{identifier}", flush=True)
    prices, manifest = load_snapshot(native)
    if manifest["sha256"] != item["snapshot"]["sha256"]: raise ValueError("Native input differs from inventory hash.")
    policy = timestamp_policy(spec, manifest, prices.index)
    variables = {symbol: identifier + "::" + symbol for symbol in prices.columns}
    folder = groups_root / group / identifier
    for stage in ("raw", "clean", "prepared", "analysis"): (folder / stage).mkdir(parents=True, exist_ok=True)
    native_refs = [reference(p) for p in sorted(native.iterdir()) if p.is_file()]
    metadata_refs = [reference(p) for p in supplementary_paths(spec, native)]
    write_json(folder / "raw/reference.json", dict(stage="raw", native_snapshot_project_relative=native.relative_to(ROOT).as_posix(),
               snapshot_key=native.name, price_file_sha256=manifest["sha256"], native_artifacts=native_refs,
               source_request=manifest["request"], source_metadata=metadata_refs,
               raw_evidence_policy="Original raw files and source receipts are referenced in native provenance/acquisition ledgers; broader raw coverage is not additional prepared data.",
               original_data_modified=False, rights=rights_for(spec)))
    clean_quality = matrix_quality(prices)
    changes, gap_rows = transform_panel(prices, spec)
    if len(changes) != len(prices)-1: raise AssertionError("Unexpected baseline transformation row count.")
    prepared_quality = matrix_quality(changes)
    path = folder / "prepared/features.csv.gz"
    write_prepared(path, changes, variables, policy, identifier)
    validation = verify_prepared(path, changes, variables, policy)
    if sha256(native / "prices.csv.gz") != manifest["sha256"]: raise AssertionError("Frozen source changed while organizing.")
    prepared_ref = reference(path)
    write_json(folder / "clean/reference.json", dict(stage="clean", representation="validated native numeric level matrix; referenced to avoid duplicating frozen observations",
               native_prices=reference(native / "prices.csv.gz"), native_manifest=reference(native / "manifest.json"),
               source_timestamp_policy=policy, modifications="None. Signed/zero physical levels, missing observations and previously declared provider-quality masks remain unchanged.",
               unresolved_source_quality="Native-cache integrity does not certify economic prices, identities, corporate actions or original publication vintages."))
    write_json(folder / "clean/quality.json", dict(stage="clean", native=clean_quality, prepared=prepared_quality,
               transformation=spec["transform"], gap_spanning_rows_masked=gap_rows,
               native_hash_unchanged=True, infinity_policy="Native infinity rejected; baseline transform replaces nonfinite changes with NaN.",
               filling="none", global_column_removal="none", baseline_window_policy="For each 60-observation window, remove incomplete/nonfinite variables and sample std(ddof=1) <= 1e-12; use one common sample for all operators.",
               quality_evidence=metadata_refs, validation=validation))
    summary = item.get("analysis")
    analysis_folder = Path(summary["output_directory"]) if summary else None
    analysis_refs = [reference(p) for p in sorted(analysis_folder.iterdir()) if p.is_file()] if analysis_folder else []
    write_json(folder / "analysis/reference.json", dict(stage="analysis", status="existing descriptive screen" if summary else "not analyzed",
               artifacts=analysis_refs, input_price_sha256=manifest["sha256"],
               schedule={key: summary.get(key) for key in ("window_size", "step_size", "scheduled_windows", "status_counts")} if summary else None,
               prepared_scope="Every transformed row exists here; prior analysis used its declared scheduled subset of windows.",
               timestamp_note="Prior analysis date labels retain source timestamps; use explicit interval boundaries before calendar/physical joins.",
               claim_scope="descriptive; no scored holdout, significance or executed arbitrage implied"))
    units = spec.get("units", "unknown source units")
    variable_records = [{"dataset_id": identifier, "variable_id": variable_id, "source_variable": symbol,
                         "entity_id": entity_identifier(spec, manifest["request"], symbol),
                         "entity_id_scope": "provider ticker identifier only; historical legal-entity/listing continuity is not established" if entity_identifier(spec, manifest["request"], symbol) else "unknown/not established; no token-contract or physical-commodity identity guessed",
                         "units": units.get(symbol, "unknown source units") if isinstance(units, dict) else units,
                         "prepared_units": "dimensionless arithmetic return" if spec["transform"] == "arithmetic_return" else (units.get(symbol, "unknown source units") if isinstance(units, dict) else units),
                         "join_policy": "Canonical variable ID within this panel only; same source label across panels does not establish historical identity, currency, timing or aggregation compatibility."}
                        for symbol, variable_id in variables.items()]
    dataset = dict(schema_version=VERSION, dataset_id=identifier, group=group, family=spec["family"],
                   interval=spec["interval"], transform=spec["transform"], snapshot_key=native.name,
                   source_price_sha256=manifest["sha256"], source_specification=spec, variables=variable_records,
                   timestamp_policy=policy, rights=rights_for(spec),
                   stages={name: f"{name}/reference.json" for name in ("raw", "clean", "analysis")},
                   prepared=dict(file="prepared/features.csv.gz", sha256=prepared_ref["sha256"], bytes=prepared_ref["bytes"],
                       format="gzip UTF-8 CSV, %.17g numeric precision, empty cells are NaN", source_timestamp_column="source_timestamp",
                       interval_boundary_columns=["interval_start_utc", "interval_end_utc"] if boundaries(changes.index[:1], policy)[0] is not None else [],
                       numeric_columns=list(variables.values()), rows=len(changes),
                       observed_cells=prepared_quality["observed_cells"], missing_cells=prepared_quality["missing_cells"],
                       row_policy="All baseline-transformed rows, without scheduling/subsampling. The first native row is omitted by transform_panel because its preceding observation is unavailable.",
                       meaning="Wide numeric changes supplied to spectral windows; these are input measurements, not eigenvalue diagnostics or predictive features."))
    write_json(folder / "dataset.json", dataset)
    print(f"Ready {identifier}: {len(changes):,} rows, {prepared_quality['observed_cells']:,} observed changes, {prepared_ref['bytes']/1e6:.1f} MB", flush=True)
    return {"dataset_id": identifier, "group": group, "path": folder.relative_to(ROOT).as_posix(),
            "variables": len(variables), "native_rows": len(prices), "native_observed_cells": clean_quality["observed_cells"],
            "prepared_rows": len(changes), "prepared_observed_cells": prepared_quality["observed_cells"],
            "prepared_missing_cells": prepared_quality["missing_cells"], "prepared_bytes": prepared_ref["bytes"],
            "prepared_sha256": prepared_ref["sha256"], "gap_spanning_rows_masked": gap_rows,
            "snapshot_key": native.name, "source_price_sha256": manifest["sha256"], "validation": validation}, variable_records


def documentation(groups_root, records):
    common = """Each dataset directory has four stages: `raw/reference.json` points to unchanged native/raw provenance and hashes; `clean/reference.json` and `clean/quality.json` describe the validated frozen level matrix; `prepared/features.csv.gz` contains every arithmetic return or declared first difference; `analysis/reference.json` points to the existing spectral outputs. Clean prices are references rather than duplicated gzip files to keep disk use reasonable.

Missing cells stay missing. No zero/forward fill, outlier deletion, global complete-case reduction or calendar interpolation is applied. Provider-declared invalid bars/identities remain masked. `expanded_spectral.transform_panel` performs the exact baseline change and interval-gap masking, then omits the first undefined change. The subsequent 60-observation cleaner removes incomplete/nonfinite and near-constant variables within each window; no single globally cleaned universe is promised.

Canonical IDs are unchanged `dataset_id` values and `dataset_id::source_variable` values. CSV measurement columns use the namespaced variable IDs. Metadata time columns are never graph nodes. An identical ticker, coin symbol, zone name or date across panels is not permission to join blindly: reconcile entity/listing continuity, quote currency/numeraire, units and adjustment, interval boundaries, aggregation hierarchy and information availability first. No cross-panel entity continuity or point-in-time availability is invented. Daily/monthly naive labels remain calendar/reference periods; crypto labels are bar ends; electricity interval labels are starts; EIA grid labels are HOUR ENDING, with a separate verified `interval_start_utc = source_timestamp - 1 hour` column. Unverified Yahoo bucket boundaries remain unfilled metadata.

`source_variable` preserves the provider label. Optional `entity_id = yahoo:ticker:<symbol>` identifies the established Yahoo stock/ETF provider instrument across its daily/minute panels, including original yfinance labels; it does not prove historical legal-entity continuity, adjustments, currency or information timing. Other entity IDs remain null where not established. For example, common Yahoo tickers still require specification checks; an ECB EUR-based quote and a Yahoo USD-based quote need an explicit numeraire mapping; EIA hour-ending energy must be aligned to verified electricity interval starts rather than identical timestamp text. Canonical joins use the source, timing and specification together.

The daily-futures panel additionally exposes `prepared/observed_calendar_features.csv.gz` with `prepared/variant.json`. This is an explicitly ALTERNATIVE representation, off by default, that omits only native rows with every price missing before computing the same first difference. Partly missing rows remain; nothing is filled. Its 60-observation windows follow an irregular observed-source calendar, not 60 calendar days. It is not an additional dataset and never replaces the strict native-row baseline.

Numeric reload: read `dataset.json` for `prepared.numeric_columns`, parse only the measurement columns as floats with `float_precision='round_trip'`, and parse `source_timestamp` according to its declared timezone. Empty numeric cells represent NaN. Verified interval boundary columns appear only where their source convention is established. Source rights/attribution accompany each dataset; the organization grants no new redistribution permission. Preparation and later rolling windows are not additional datasets.
"""
    links = "\n".join(f"- [{LABELS[group]}]({group}/README.md): {sum(r['group']==group for r in records)} active datasets" for group in GROUPS)
    (groups_root / "README.md").write_text("# Frozen market data in seven groups\n\n" + links + "\n\n" + common + "\nRebuild offline from the project root with `python organize_data.py`. The active selection is checked against `output/expansion/Dataset_Inventory.csv` and `inventory.json`; earlier pilots/superseded snapshots stay at their existing paths and are excluded here. The organization receipts and canonical ID registries are under `output/organization/`.\n", encoding="utf-8")
    for group in GROUPS:
        folder = groups_root / group
        folder.mkdir(parents=True, exist_ok=True)
        links = "\n".join(f"- [{r['dataset_id']}]({r['dataset_id']}/dataset.json): {r['variables']} variables; {r['prepared_rows']:,} prepared rows" for r in records if r["group"] == group)
        (folder / "README.md").write_text(f"# {LABELS[group]}\n\n{links}\n\n{common}", encoding="utf-8")


def self_test():
    dates = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    signed = pd.DataFrame({"flow": [-2., 0., 3., np.nan]}, index=dates)
    changed, count = transform_panel(signed, {"transform": "first_difference", "bar_seconds": 3600})
    np.testing.assert_array_equal(changed["flow"].to_numpy(), [2., 3., np.nan])
    assert count == 0
    try: transform_panel(signed, {"transform": "arithmetic_return"})
    except ValueError: pass
    else: raise AssertionError("Nonpositive financial prices must be rejected.")
    gap = signed.iloc[[0, 2, 3]]
    changed, count = transform_panel(gap, {"transform": "first_difference", "bar_seconds": 3600})
    assert count == 1 and changed.iloc[0].isna().all()
    policy = timestamp_policy({"family": "grid_operations", "bar_seconds": 3600},
        {"index_name": "interval_start_utc", "request": {"timezone_policy": "source UTC hourly labels"}}, dates)
    first, last = boundaries(dates, policy)
    assert first[0] == dates[0]-pd.Timedelta(hours=1) and last.equals(dates)
    assert group_for({"family": "natural_gas"}) == "commodities"
    assert entity_identifier({"family": "us_listed_etf"}, {"source": "Yahoo via yfinance"}, "SPY") == "yahoo:ticker:SPY"
    assert entity_identifier({"family": "crypto"}, {"source": "binance_vision_spot_archive"}, "USDSUSDT") is None
    assert entity_identifier({"family": "commodity_futures_proxy"}, {"source": "Yahoo via yfinance"}, "CL=F") is None
    partial = pd.DataFrame({"a": [1., np.nan, 2., np.nan, 3.], "b": [10., np.nan, np.nan, 11., 12.]},
                           index=pd.date_range("2026-01-01", periods=5))
    retained = partial.loc[~partial.isna().all(axis=1)]
    assert len(retained) == 4 and retained.index[1] == partial.index[2]
    alternate, _ = transform_panel(retained, {"transform": "first_difference"})
    assert alternate.iloc[0]["a"] == 1. and pd.isna(alternate.iloc[0]["b"])
    print("PASS: signed values, financial positivity, gap masking, EIA mapping, IDs and whole-row-only alternative policy", flush=True)


def observed_calendar_alternative(groups_root, records):
    """One disclosed representation; strict native-row baseline stays intact."""
    identifier = "yahoo_commodity_futures_daily_27ticker"
    record = next(item for item in records if item["dataset_id"] == identifier)
    folder = ROOT / record["path"]
    dataset = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    spec = dataset["source_specification"]
    native = Path(spec["snapshot_path"])
    prices, manifest = load_snapshot(native)
    if spec["transform"] != "first_difference" or len(prices.columns) != 26:
        raise ValueError("The declared daily-futures alternative needs the original 26-variable signed panel.")
    if spec.get("bar_seconds"): raise ValueError("Daily-source alternative must not invent fixed UTC bar spacing.")
    baseline = folder / "prepared/features.csv.gz"
    if sha256(baseline) != record["prepared_sha256"]: raise ValueError("Baseline preparation hash changed.")
    missing = prices.isna().all(axis=1)
    omitted = prices.index[missing]
    retained = prices.loc[~missing]
    partly_missing = prices.isna().any(axis=1) & ~missing
    if not retained.index.equals(prices.index[~missing]): raise AssertionError("Unexpected calendar reordering.")
    changes, gap_rows = transform_panel(retained, spec)
    if gap_rows != 0: raise AssertionError("No fixed-spacing daily mask belongs to this alternative.")
    variables = {row["source_variable"]: row["variable_id"] for row in dataset["variables"]}
    policy = timestamp_policy(spec, manifest, retained.index)
    target = folder / "prepared/observed_calendar_features.csv.gz"
    write_prepared(target, changes, variables, policy, identifier + " [ALTERNATIVE]")
    validation = verify_prepared(target, changes, variables, policy)
    if sha256(baseline) != record["prepared_sha256"] or sha256(native / "prices.csv.gz") != manifest["sha256"]:
        raise AssertionError("The alternative changed baseline/native observations.")
    quality = matrix_quality(changes)
    variant = dict(schema_version=VERSION, dataset_id=identifier, variant_id="observed_source_calendar",
        stage="prepared representation", alternative=True, enabled_by_default=False,
        counts_as_additional_dataset=False, source_snapshot_key=native.name,
        raw_input_price_sha256=manifest["sha256"], native_price_rows=len(prices),
        omitted_price_rows=int(missing.sum()), omitted_dates=[date.isoformat() for date in omitted],
        omitted_weekday_counts=dict(Counter(date.day_name() for date in omitted)),
        retained_native_price_rows=len(retained), partly_missing_price_rows_preserved=int(partly_missing.sum()),
        partly_missing_rows_removed=0, prepared_rows=len(changes), prepared_quality=quality,
        calendar="irregular observed-source calendar; 60 observations do not mean 60 calendar days",
        policy="Before the same first_difference transform, drop only native rows for which ALL 26 prices are NaN. No partly missing rows are removed, no filling, no elapsed-time division, no outlier deletion.",
        source_timestamp_policy=policy, baseline_preserved=True,
        baseline_prepared_file=reference(baseline), alternative_prepared_file=reference(target),
        validation=validation, organizer_code=reference(Path(__file__)),
        limitations="Provider date/session/roll/settlement and futures physical identity remain unverified. Window cleaning still applies per 60-observation sample; this alternative does not certify source prices or establish a trading signal.")
    write_json(folder / "prepared/variant.json", variant)
    dataset["prepared_alternatives"] = [{"variant_id": "observed_source_calendar", "metadata": "prepared/variant.json",
        "file": "prepared/observed_calendar_features.csv.gz", "enabled_by_default": False,
        "counts_as_additional_dataset": False, "sha256": variant["alternative_prepared_file"]["sha256"]}]
    write_json(folder / "dataset.json", dataset)
    return {key: variant[key] for key in ("dataset_id", "variant_id", "omitted_price_rows", "omitted_weekday_counts", "partly_missing_price_rows_preserved", "prepared_rows", "baseline_preserved", "validation")} | {
        "path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size,
        "sha256": variant["alternative_prepared_file"]["sha256"], "counts_as_additional_dataset": False}


def finalize_metadata(output, groups_root):
    """Refresh identifiers and add the requested small alternative, no baseline rerun."""
    receipt = json.loads((output / "registry.json").read_text(encoding="utf-8"))
    if receipt["status"] != "pass" or receipt["failures"]: raise ValueError("Finalize only completed baseline preparations.")
    variables = []
    for record in receipt["datasets"]:
        folder = ROOT / record["path"]
        dataset = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        raw = json.loads((folder / "raw/reference.json").read_text(encoding="utf-8"))
        for variable in dataset["variables"]:
            variable["entity_id"] = entity_identifier(dataset["source_specification"], raw["source_request"], variable["source_variable"])
            variable["entity_id_scope"] = "provider ticker identifier only; historical legal-entity/listing continuity is not established" if variable["entity_id"] else "unknown/not established; no token-contract or physical-commodity identity guessed"
            variables.append(variable)
        dataset["schema_version"] = VERSION
        write_json(folder / "dataset.json", dataset)
        quality_path = folder / "clean/quality.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["per_variable_key_policy"] = "source_variable keys; dataset.json maps these to the canonical namespaced variable_id"
        write_json(quality_path, quality)
    fields = ("dataset_id", "variable_id", "source_variable", "entity_id", "entity_id_scope", "units", "prepared_units", "join_policy")
    with (output / "Variable_Registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(variables)
    if len({row["variable_id"] for row in variables}) != len(variables): raise AssertionError("Duplicate canonical variable ID.")
    receipt["alternative_representations"] = [observed_calendar_alternative(groups_root, receipt["datasets"])]
    prior_code = receipt["organizer_code"]
    current_code = reference(Path(__file__))
    if prior_code["sha256"] != current_code["sha256"]:
        historical = output / "organize_data_preparation_v1.py"
        if not historical.exists() or sha256(historical) != prior_code["sha256"]:
            raise ValueError("Preserve the exact executed preparation code before editing the organizer.")
        receipt["organizer_code"] = reference(historical)
    receipt["metadata_organizer_code"] = current_code
    receipt["metadata_finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["schema_version"] = VERSION
    receipt["provider_qualified_entity_rows"] = sum(row["entity_id"] is not None for row in variables)
    receipt["unknown_entity_rows"] = sum(row["entity_id"] is None for row in variables)
    documentation(groups_root, receipt["datasets"])
    write_json(output / "registry.json", receipt)
    for _ in range(4):
        receipt["new_storage_bytes"] = sum(p.stat().st_size for folder in (groups_root, output) for p in folder.rglob("*") if p.is_file())
        write_json(output / "registry.json", receipt)
    actual = sum(p.stat().st_size for folder in (groups_root, output) for p in folder.rglob("*") if p.is_file())
    if actual != receipt["new_storage_bytes"]: raise AssertionError("Storage receipt did not stabilize.")
    print(json.dumps({key: receipt[key] for key in ("active_datasets", "canonical_variable_count", "provider_qualified_entity_rows", "prepared_rows", "prepared_bytes", "new_storage_bytes", "alternative_representations")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-panels", type=int, default=36)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--finalize-metadata", action="store_true", help="Add IDs/observed-calendar alternative to completed preparations without rewriting baseline CSVs")
    args = parser.parse_args()
    if args.self_test: self_test(); return
    if args.finalize_metadata:
        finalize_metadata(ROOT / "output/organization", ROOT / "data/groups"); return
    items, inventory, inputs = active_inventory(args.expected_panels)
    groups_root, output = ROOT / "data/groups", ROOT / "output/organization"
    output.mkdir(parents=True, exist_ok=True)
    groups_root.mkdir(parents=True, exist_ok=True)
    records, variables, failures = [], [], []
    for item in items:
        try:
            record, ids = organize_one(item, groups_root)
            records.append(record); variables.extend(ids)
        except Exception as error:
            failures.append({"dataset_id": item["specification"]["dataset_id"], "type": type(error).__name__, "message": str(error)})
            print(f"FAILED {failures[-1]}", flush=True)
        write_json(output / "progress.json", {"completed": records, "failures": failures})
    documentation(groups_root, records)
    with (output / "Variable_Registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset_id", "variable_id", "source_variable", "entity_id", "entity_id_scope", "units", "prepared_units", "join_policy"))
        writer.writeheader(); writer.writerows(variables)
    receipt = dict(schema_version=VERSION, created_at_utc=datetime.now(timezone.utc).isoformat(),
        status="pass" if not failures else "incomplete", active_datasets=len(records),
        group_counts={group: sum(r["group"]==group for r in records) for group in GROUPS},
        canonical_dataset_ids=[r["dataset_id"] for r in records], canonical_variable_count=len(variables),
        native_observed_cells_including_overlap=sum(r["native_observed_cells"] for r in records),
        prepared_rows=sum(r["prepared_rows"] for r in records),
        prepared_observed_cells_including_overlap=sum(r["prepared_observed_cells"] for r in records),
        prepared_bytes=sum(r["prepared_bytes"] for r in records), source_inputs=inputs,
        transformation_code=reference(ROOT / "expanded_spectral.py"), organizer_code=reference(Path(__file__)),
        datasets=records, failures=failures, incremental_spend_usd=0,
        counting_note="36 panels organized; changes/files/windows are stages, not additional datasets. Overlap and derived/aggregate observations remain disclosed.",
        originals_policy="Frozen original/native files unchanged; source hashes checked during load and after preparation.",
        join_policy="No currency/timing/aggregation-blind joins. Namespaced IDs identify measurements; cross-panel entity/time compatibility must be explicitly established.")
    write_json(output / "registry.json", receipt)
    write_json(output / "preserved_reference.json", {"excluded_pilots_or_superseded": [item["specification"] for item in inventory["datasets"] if item["specification"]["dataset_id"] not in receipt["canonical_dataset_ids"]],
        "earlier_snapshots": inventory.get("preserved_superseded_snapshots", []), "policy": "Read-only original locations; not included in active group counts."})
    print(json.dumps({key: receipt[key] for key in ("status", "active_datasets", "group_counts", "native_observed_cells_including_overlap", "prepared_rows", "prepared_observed_cells_including_overlap", "prepared_bytes", "failures")}, indent=2), flush=True)
    if failures: raise SystemExit(1)
    finalize_metadata(output, groups_root)


if __name__ == "__main__": main()
