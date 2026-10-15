"""Focused offline checks for signed levels, failures, provenance and parsers."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from archive_catalog import load_snapshot
from financial_expansion import (StorageBudget, extract_yahoo_closes, freeze_panel,
                                 normalize_fx, FX_PAIRS, parse_ecb, parse_fred,
                                 parse_world_bank, yahoo_panel)


def expect_error(operation, error=ValueError):
    try:
        operation()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def check_signed_partial_snapshot(root):
    budget = StorageBudget(root / "cache", 2_000_000)
    prices = pd.DataFrame({"oil": [-37.63, 10.01, 20.4], "unavailable": [np.nan] * 3},
                          index=pd.date_range("2020-04-20", periods=3, name="date"))
    prices.columns.name = "symbol"
    provenance = dict(source="mock signed source", source_url="https://example.invalid", symbols=list(prices),
                      start="2020-04-20", end_exclusive="2020-04-23", interval="1d",
                      adjustment="none", feed="mock", timezone_policy="naive dates")
    dataset = dict(dataset_id="signed_partial", family="commodity", interval="1d", transform="first_difference",
                   units="USD/barrel", notes="mock", universe_as_of="fixed mock universe")
    result = freeze_panel(prices, provenance, dataset, budget, positive=False)
    loaded, manifest = load_snapshot(result["snapshot_path"])
    pd.testing.assert_frame_equal(prices[["oil"]], loaded, check_exact=True, check_freq=False)
    assert loaded.iloc[0, 0] == -37.63 and list(loaded) == ["oil"]
    assert manifest["request"]["symbols"] == ["oil"]
    assert result["unavailable_symbols"] == ["unavailable"]
    assert result["coverage"]["negative_cells"] == 1
    evidence = json.loads((Path(result["snapshot_path"]) / "provenance.json").read_text())
    assert evidence["price_file_sha256"] == manifest["sha256"]
    assert evidence["requested_universe"] == ["oil", "unavailable"] and evidence["unavailable_symbols"] == ["unavailable"]
    expect_error(lambda: freeze_panel(prices, provenance, dataset, budget, positive=True))
    changed = prices.copy()
    changed.iloc[0, 0] += 1
    expect_error(lambda: freeze_panel(changed, provenance, dataset, budget, positive=False), AssertionError)


def check_yahoo_missing_and_time(root):
    budget = StorageBudget(root / "yahoo", 2_000_000)
    settings_seen = []

    def mock_download(tickers, **settings):
        settings_seen.append(settings)
        columns = pd.MultiIndex.from_product([["Close", "Open"], tickers])
        index = pd.date_range("2026-09-01 13:30", periods=3, freq="5min", tz="UTC")
        frame = pd.DataFrame(10.0, index=index, columns=columns)
        if "B" in tickers:
            frame.loc[:, pd.IndexSlice[:, "B"]] = np.nan
        return frame

    prices, provenance = yahoo_panel(["A", "B", "C"], "2026-09-01T13:30:00Z", "2026-09-01T13:40:00Z",
                                     "5m", budget, auto_adjust=True, batch_size=2, download=mock_download)
    assert len(prices) == 2 and list(prices) == ["A", "B", "C"] and prices["B"].isna().all()
    assert str(prices.index.tz) == "UTC" and not settings_seen[0]["ignore_tz"]
    assert provenance["unavailable_log"][0]["symbol"] == "B" and len(provenance["returned_frames"]) == 2
    assert provenance["raw_response_sha256"] is None
    naive = pd.DataFrame([[1.0]], index=pd.DatetimeIndex(["2026-09-01"]),
                         columns=pd.MultiIndex.from_tuples([("Close", "A")]))
    expect_error(lambda: extract_yahoo_closes(naive, ["A"], daily=False))
    bad = naive.copy()
    bad.iloc[0, 0] = np.inf
    expect_error(lambda: extract_yahoo_closes(bad, ["A"], daily=True))
    fx = pd.DataFrame({ticker: [2.0] for ticker in FX_PAIRS}, index=pd.DatetimeIndex(["2026-09-01"], name="date"))
    transformed = normalize_fx(fx)
    assert transformed["EUR_per_USD"].iloc[0] == 0.5 and transformed["JPY_per_USD"].iloc[0] == 2.0


def check_official_parsers():
    content = b"Date,USD,EEK, \n2026-10-07,1.1,N/A,\n1999-01-04,1.2,15.6,\n"
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr("eurofxref-hist.csv", content)
    prices, _ = parse_ecb(body.getvalue())
    assert list(prices) == ["USD", "EEK"] and prices.index.is_monotonic_increasing
    assert pd.isna(prices.loc["2026-10-07", "EEK"])
    fred = parse_fred(b"observation_date,DCOILWTICO,DCOILBRENTEU,DHHNGSP\n2020-04-20,-36.98,20,1.7\n2020-04-21,.,21,1.8\n")
    assert fred.iloc[0, 0] == -36.98 and pd.isna(fred.iloc[1, 0])
    rows = [["Title", None, None, None], ["Nominal", None, None, None], ["Note", None, None, None],
            ["Updated on October 02, 2026", None, None, None],
            [None, "Crude oil, average", "Natural gas index", "Gold"],
            [None, "($/bbl)", "(2010=100)", "($/troy oz)"],
            ["1960M01", 1.6, 5.0, "\u2026"], ["1960M02", 1.7, 5.1, "..."], ["1960M03", 1.8, 5.2, 35.0]]
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Monthly Prices", index=False, header=False)
    monthly, units, details = parse_world_bank(workbook.getvalue())
    assert list(monthly) == ["Crude oil, average", "Gold"]
    assert monthly["Gold"].isna().sum() == 2 and units["Gold"] == "($/troy oz)"
    assert details["excluded_index"] == "Natural gas index (2010=100)"
    assert monthly.index[0] == pd.Timestamp("1960-01-01")
    rows[-1][-1] = "unrecognized marker"
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Monthly Prices", index=False, header=False)
    expect_error(lambda: parse_world_bank(workbook.getvalue()))


def check_storage_limit(root):
    budget = StorageBudget(root / "small", 5)
    first = budget.preserve(b"abc", ".txt")
    assert budget.preserve(b"abc", ".txt") == first
    expect_error(lambda: budget.preserve(b"def", ".txt"), RuntimeError)
    assert budget.used() == 3


def main():
    with tempfile.TemporaryDirectory(prefix="financial-check-") as directory:
        root = Path(directory)
        for name, operation in [
            ("exact signed reload and full source-failure disclosure", lambda: check_signed_partial_snapshot(root)),
            ("Yahoo missing symbols, evidence and explicit timestamps", lambda: check_yahoo_missing_and_time(root)),
            ("official source parsing and units", check_official_parsers),
            ("bounded evidence storage", lambda: check_storage_limit(root)),
        ]:
            operation()
            print("PASS:", name)
    print("4 financial expansion check groups passed (offline, mocked source responses).")


if __name__ == "__main__":
    main()
