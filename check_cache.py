"""Offline cache checks: no network, credentials, or real market data.

Run: python check_cache.py
"""

import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_cache import cached_closes, _publish_snapshot
from export_existing import export_current_prices


def request_for(symbols):
    return {
        "source": "synthetic test",
        "symbols": symbols,
        "start": "2020-01-01",
        "end": "2020-12-31",
        "interval": "1d",
        "adjustment": "none: synthetic",
        "feed": "synthetic",
        "timezone_policy": "preserve supplied labels",
        "universe_as_of": "synthetic",
    }


def check_roundtrip_and_download_reuse():
    rng = np.random.default_rng(21)
    prices = pd.DataFrame(
        rng.uniform(20, 100, size=(6, 2)),
        index=pd.bdate_range("2020-01-01", periods=6, name="Date"),
        columns=pd.Index(["A", "B"], name="Ticker"),
    )
    prices.iloc[2, 0] = np.nan
    calls = []

    def fake_download():
        calls.append(1)
        return prices

    with tempfile.TemporaryDirectory() as temporary:
        first, manifest, hit = cached_closes(request_for(["A", "B"]), fake_download, temporary)
        assert hit is False
        second, second_manifest, hit = cached_closes(request_for(["B", "A"]), fake_download, temporary)
        assert hit is True and len(calls) == 1
        pd.testing.assert_frame_equal(first, second, check_freq=False, check_exact=True)
        assert manifest == second_manifest and manifest["missing_cells"] == 1
        assert list(Path(temporary).glob("*/manifest.json"))
        # A checksum catches accidental corruption before loading altered prices.
        price_path = next(Path(temporary).glob("*/prices.csv.gz"))
        with price_path.open("ab") as handle:
            handle.write(b"corrupt")
        try:
            cached_closes(request_for(["A", "B"]), cache_dir=temporary)
        except ValueError:
            pass
        else:
            raise AssertionError("A corrupted cache must be rejected.")


def check_timezone_and_invalid_download():
    # Cross a daylight-saving change; absolute instants and named timezone survive.
    index = pd.date_range("2020-10-31", periods=4, freq="D", tz="America/New_York")
    prices = pd.DataFrame({"A": [1.1, 1.2, 1.3, 1.4]}, index=index)
    with tempfile.TemporaryDirectory() as temporary:
        cached_closes(request_for(["A"]), lambda: prices, temporary)
        restored, _, hit = cached_closes(request_for(["A"]), cache_dir=temporary)
        pd.testing.assert_frame_equal(prices, restored, check_freq=False, check_exact=True)
        assert hit is True

    with tempfile.TemporaryDirectory() as temporary:
        try:
            cached_closes(request_for(["A", "FAILED"]), lambda: prices, temporary)
        except ValueError:
            assert not list(Path(temporary).iterdir())
        else:
            raise AssertionError("Failed symbols must not silently create a complete snapshot.")
        try:
            cached_closes(request_for(["A"]), cache_dir=temporary)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Offline cache misses must be explicit.")


def check_intraday_precision_and_type_rejection():
    # ISO labels can mix whole seconds with nanosecond precision.
    index = pd.to_datetime(
        ["2020-01-02T14:30:00Z", "2020-01-02T14:31:00.123456789Z"],
        format="ISO8601",
    )
    prices = pd.DataFrame({"A": [20.0, 20.1]}, index=index)
    with tempfile.TemporaryDirectory() as temporary:
        cached_closes(request_for(["A"]), lambda: prices, temporary)
        restored, _, _ = cached_closes(request_for(["A"]), cache_dir=temporary)
        pd.testing.assert_frame_equal(prices, restored, check_freq=False, check_exact=True)
    from dateutil.tz import tzoffset
    unsupported_index = pd.date_range("2020-01-01", periods=2, tz=tzoffset(None, 19800))
    cases = [
        pd.DataFrame({"A": [1 + 2j, 3 + 4j]}, index=index),
        pd.DataFrame({"A": [True, False]}, index=index),
        pd.DataFrame({"A": [1.1, 1.2]}, index=unsupported_index),
    ]
    for case in cases:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                cached_closes(request_for(["A"]), lambda: case, temporary)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsupported price types/timezones must fail before saving.")


def check_export_conflicts_and_infinity():
    prices = pd.DataFrame(
        {"A": [10.0, 11.0], "B": [20.0, 21.0]},
        index=pd.date_range("2020-01-01", periods=2, name="Date"),
    )
    with tempfile.TemporaryDirectory() as temporary:
        with redirect_stdout(io.StringIO()):
            export_current_prices(prices, temporary)
            export_current_prices(prices.copy(), temporary)
        price_path = next(Path(temporary).glob("*/prices.csv.gz"))
        frozen_bytes = price_path.read_bytes()
        changed_values = prices.copy()
        changed_values.iloc[0, 0] = 100.0
        changed_name = prices.rename_axis("different_date_name")
        changed_columns = prices.loc[:, ["B", "A"]]
        for changed in (changed_values, changed_name, changed_columns):
            try:
                export_current_prices(changed, temporary)
            except ValueError as error:
                assert "frozen snapshot differs" in str(error)
            else:
                raise AssertionError("An export must not silently discard supplied changes.")
            assert price_path.read_bytes() == frozen_bytes
            assert len(list(Path(temporary).glob("*/manifest.json"))) == 1
        invalid = prices.copy()
        invalid.iloc[0, 0] = np.inf
        try:
            export_current_prices(invalid, temporary)
        except ValueError as error:
            assert "infinity" in str(error)
        else:
            raise AssertionError("Export input must be validated even on a cache hit.")

    # A parser-selected datetime storage unit is not a changed observation.
    second_labels = prices.copy()
    second_labels.index = second_labels.index.as_unit("s")
    with tempfile.TemporaryDirectory() as temporary:
        with redirect_stdout(io.StringIO()):
            export_current_prices(second_labels, temporary)
            export_current_prices(second_labels, temporary)

    for values in ([np.inf, np.inf], [1.0, -np.inf], [np.nan, np.inf]):
        invalid = pd.DataFrame({"A": values}, index=prices.index)
        with tempfile.TemporaryDirectory() as temporary:
            try:
                cached_closes(request_for(["A"]), lambda: invalid, temporary)
            except ValueError as error:
                assert "infinity" in str(error)
                assert not list(Path(temporary).iterdir())
            else:
                raise AssertionError("Infinite prices must fail before any snapshot is saved.")


def check_numeric_dtypes_and_legacy_schema():
    prices = pd.DataFrame({
        "large_signed": pd.array([9007199254740993, None, -9007199254740993], dtype="Int64"),
        "large_unsigned": pd.array([18446744073709551615, None, 9007199254740993], dtype="UInt64"),
        "nullable_float": pd.array([np.nextafter(1.0, 2.0), None, 1.2345678901234567], dtype="Float64"),
        "small_float": np.array([1.1, np.nan, 3.25], dtype="float32"),
        "integer": np.array([9007199254740993, 2, -9007199254740993], dtype="int64"),
    }, index=pd.date_range("2020-01-01", periods=3))
    with tempfile.TemporaryDirectory() as temporary:
        _, manifest, _ = cached_closes(request_for(prices.columns.tolist()), lambda: prices, temporary)
        assert manifest["schema_version"] == 2
        assert manifest["column_dtypes"] == {column: str(dtype) for column, dtype in prices.dtypes.items()}
        restored, _, _ = cached_closes(request_for(prices.columns.tolist()), cache_dir=temporary)
        pd.testing.assert_frame_equal(prices, restored, check_freq=False, check_exact=True)

    # Existing schema-1 float64 snapshots remain usable; fixtures alone are edited.
    legacy = pd.DataFrame({"A": [1.1, np.nan, 1.3]}, index=prices.index)
    with tempfile.TemporaryDirectory() as temporary:
        _, manifest, _ = cached_closes(request_for(["A"]), lambda: legacy, temporary)
        manifest["schema_version"] = 1
        manifest.pop("column_dtypes")
        manifest_path = next(Path(temporary).glob("*/manifest.json"))
        manifest_path.write_text(json.dumps(manifest))
        restored, _, hit = cached_closes(request_for(["A"]), cache_dir=temporary)
        assert hit is True
        pd.testing.assert_frame_equal(legacy, restored, check_freq=False, check_exact=True)


def check_manifest_summaries():
    prices = pd.DataFrame({"A": [1.0, np.nan, 1.2]}, index=pd.date_range("2020-01-01", periods=3))
    changes = {
        "rows": 99,
        "actual_first_timestamp": "2019-01-01T00:00:00",
        "actual_last_timestamp": "2021-01-01T00:00:00",
        "missing_cells": 0,
    }
    with tempfile.TemporaryDirectory() as temporary:
        _, original_manifest, _ = cached_closes(request_for(["A"]), lambda: prices, temporary)
        manifest_path = next(Path(temporary).glob("*/manifest.json"))
        for field, value in changes.items():
            altered = dict(original_manifest, **{field: value})
            manifest_path.write_text(json.dumps(altered))
            try:
                cached_closes(request_for(["A"]), cache_dir=temporary)
            except ValueError as error:
                assert field in str(error)
            else:
                raise AssertionError("Manifest summaries must match the saved observations.")


def check_bounded_windows_publication():
    def windows_error(code):
        error = PermissionError(13, "synthetic sharing denial")
        error.winerror = code
        return error

    real_rename = os.rename
    for code in (5, 32, 33):
        with tempfile.TemporaryDirectory() as temporary:
            staging, target = Path(temporary) / "staging", Path(temporary) / "published"
            staging.mkdir()
            (staging / "marker").write_text("complete synthetic snapshot")
            calls = []

            def transient_then_publish(source, destination):
                calls.append(1)
                if len(calls) <= 2:
                    raise windows_error(code)
                return real_rename(source, destination)

            with patch("data_cache.os.rename", side_effect=transient_then_publish), patch("data_cache.time.sleep") as sleep:
                _publish_snapshot(staging, target)
            assert not staging.exists() and (target / "marker").read_text() == "complete synthetic snapshot"
            assert len(calls) >= 3 and sum(call.args[0] for call in sleep.call_args_list) <= 3

    with tempfile.TemporaryDirectory() as temporary:
        staging, target = Path(temporary) / "staging", Path(temporary) / "published"
        staging.mkdir()
        permanent = windows_error(5)
        with patch("data_cache.os.rename", side_effect=permanent) as rename, patch("data_cache.time.sleep") as sleep:
            try:
                _publish_snapshot(staging, target)
            except PermissionError as error:
                assert error is permanent
            else:
                raise AssertionError("Persistent sharing denials must exhaust a finite retry budget.")
        assert 1 < rename.call_count <= 8 and sum(call.args[0] for call in sleep.call_args_list) <= 3
        assert staging.is_dir() and not target.exists()

        def concurrent_destination(source, destination):
            target.mkdir()
            (target / "existing").write_text("must remain unchanged")
            raise windows_error(32)

        with patch("data_cache.os.rename", side_effect=concurrent_destination) as rename, patch("data_cache.time.sleep"):
            try:
                _publish_snapshot(staging, target)
            except FileExistsError:
                pass
            else:
                raise AssertionError("A destination created between retries must never be overwritten.")
        assert rename.call_count == 1 and (target / "existing").read_text() == "must remain unchanged"
        with patch("data_cache.os.rename") as rename, patch("data_cache.time.sleep") as sleep:
            try:
                _publish_snapshot(staging, target)
            except FileExistsError:
                pass
            else:
                raise AssertionError("An initially existing destination must be refused.")
        rename.assert_not_called()
        sleep.assert_not_called()

    for failure in (FileExistsError(17, "synthetic existing destination"),
                    PermissionError(13, "synthetic non-Windows permission failure"),
                    OSError(28, "synthetic full disk")):
        with tempfile.TemporaryDirectory() as temporary:
            staging, target = Path(temporary) / "staging", Path(temporary) / "published"
            staging.mkdir()
            with patch("data_cache.os.rename", side_effect=failure) as rename, patch("data_cache.time.sleep") as sleep:
                try:
                    _publish_snapshot(staging, target)
                except OSError as error:
                    assert error is failure
                else:
                    raise AssertionError("Unrelated publication errors must propagate.")
            assert rename.call_count == 1
            sleep.assert_not_called()

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary) / "staging"
        staging.mkdir()
        with patch("data_cache.os.rename") as rename:
            try:
                _publish_snapshot(staging, Path(temporary) / "other_root" / "published")
            except ValueError:
                pass
            else:
                raise AssertionError("Publication cannot move a snapshot outside its cache root.")
        rename.assert_not_called()


if __name__ == "__main__":
    check_roundtrip_and_download_reuse()
    print("PASS: exact numeric round-trip, missing cells, one download, corruption rejection")
    check_timezone_and_invalid_download()
    print("PASS: DST timezone round-trip, failed-symbol rejection, explicit offline miss")
    check_intraday_precision_and_type_rejection()
    print("PASS: mixed intraday timestamp precision, unsupported type/timezone rejection")
    check_export_conflicts_and_infinity()
    print("PASS: conflicting exports preserve frozen snapshots; infinite prices rejected")
    check_numeric_dtypes_and_legacy_schema()
    print("PASS: exact large/nullable numeric dtype round-trips; legacy schema readable")
    check_manifest_summaries()
    print("PASS: manifest row counts, timestamp bounds and missing-cell totals verified")
    check_bounded_windows_publication()
    print("PASS: bounded Windows sharing retries; permanent/unrelated errors propagate; destinations never replaced")
    print("All cache checks passed. No market data was downloaded.")
