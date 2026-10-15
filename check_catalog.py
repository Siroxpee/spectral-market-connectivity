"""Focused offline checks for the local snapshot metadata catalog.

Run: python check_catalog.py
Only synthetic observations and temporary folders are used.
"""

from pathlib import Path
import shutil
import tempfile

import pandas as pd

from archive_catalog import list_snapshots, load_snapshot, register_snapshot
from data_cache import cached_closes


def _request(source="synthetic daily", interval="1d"):
    return {
        "source": source, "symbols": ["A", "B"],
        "start": "2020-01-01", "end": "2020-02-01", "interval": interval,
        "adjustment": "none: synthetic", "feed": "synthetic",
        "timezone_policy": "preserve supplied labels", "universe_as_of": "synthetic",
    }


def _prices(intraday=False):
    index = (
        pd.date_range("2020-01-02T14:30:00Z", periods=3, freq="5min", name="Date")
        if intraday else pd.bdate_range("2020-01-01", periods=3, name="Date")
    )
    return pd.DataFrame({"A": [20.0, 20.1, 20.2], "B": [30.0, 30.2, 30.1]}, index=index)


def _snapshot(root, request, prices):
    cached_closes(request, lambda: prices, cache_dir=root)
    # Each test root contains exactly one snapshot, so its directory is explicit.
    return next(Path(root).glob("*/manifest.json")).parent


def _expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected the invalid snapshot or operation to be rejected.")


def check_registration_query_and_loading():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        db = root / "catalog.sqlite"
        daily_prices = _prices()
        daily = _snapshot(root / "daily", _request(), daily_prices)
        before = {path.name: path.read_bytes() for path in daily.iterdir()}
        entry = register_snapshot(daily, db)
        assert register_snapshot(daily, db) == entry
        assert len(list_snapshots(db)) == 1
        assert entry["row_count"] == 3 and entry["symbol_count"] == 2
        assert entry["actual_first_timestamp"] == daily_prices.index[0].isoformat()
        assert entry["actual_last_timestamp"] == daily_prices.index[-1].isoformat()
        assert entry["index_timezone"] is None
        loaded, _ = load_snapshot(entry["snapshot_key"], db)
        direct, _ = load_snapshot(daily)
        pd.testing.assert_frame_equal(loaded, daily_prices, check_freq=False, check_exact=True)
        pd.testing.assert_frame_equal(direct, loaded, check_exact=True)
        assert before == {path.name: path.read_bytes() for path in daily.iterdir()}

        intraday = _snapshot(root / "intraday", _request("synthetic intraday", "5m"), _prices(True))
        second = register_snapshot(intraday, db)
        assert second["index_timezone"] == "UTC"
        assert len(list_snapshots(db)) == 2
        assert list_snapshots(db, source="synthetic daily") == [entry]
        assert list_snapshots(db, interval="5m") == [second]
        assert list_snapshots(db, source="synthetic intraday", interval="5m") == [second]
        assert list_snapshots(db, source="missing") == []


def check_corruption_rejection():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        db = root / "catalog.sqlite"
        snapshot = _snapshot(root / "raw", _request(), _prices())
        entry = register_snapshot(snapshot, db)
        with (snapshot / "prices.csv.gz").open("ab") as handle:
            handle.write(b"corruption")
        _expect_value_error(lambda: register_snapshot(snapshot, db))
        _expect_value_error(lambda: load_snapshot(entry["snapshot_key"], db))
        _expect_value_error(lambda: load_snapshot(snapshot))
        assert list_snapshots(db) == [entry]


def check_explicit_relocation_and_immutable_key():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        db = root / "catalog.sqlite"
        snapshot = _snapshot(root / "original", _request(), _prices())
        entry = register_snapshot(snapshot, db)
        relocated = root / "relocated" / snapshot.name
        shutil.copytree(snapshot, relocated)
        _expect_value_error(lambda: register_snapshot(relocated, db))
        assert list_snapshots(db) == [entry]
        moved = register_snapshot(relocated, db, relocate=True)
        assert moved["snapshot_path"] == str(relocated.resolve())
        assert len(list_snapshots(db)) == 1
        assert register_snapshot(relocated, db) == moved
        loaded, _ = load_snapshot(entry["snapshot_key"], db)
        pd.testing.assert_frame_equal(loaded, _prices(), check_freq=False, check_exact=True)

        # A new download of the same request is a different frozen artifact,
        # even if its local files are internally consistent.
        changed = _prices() + 1.0
        replacement = _snapshot(root / "different", _request(), changed)
        _expect_value_error(lambda: register_snapshot(replacement, db, relocate=True))
        assert list_snapshots(db) == [moved]


if __name__ == "__main__":
    check_registration_query_and_loading()
    print("PASS: idempotent registration, exact source/interval queries, frozen key/path loads")
    check_corruption_rejection()
    print("PASS: checksum corruption rejects registration and key/path loads")
    check_explicit_relocation_and_immutable_key()
    print("PASS: explicit relocation, unchanged row count, immutable request-key contents")
    print("All catalog checks passed. No market data was downloaded.")
