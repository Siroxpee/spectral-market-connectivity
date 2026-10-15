"""Bounded, separately frozen daily extension for the fixed-ten trajectory study.

No account, trading, filling, feed substitution or mutation of earlier snapshots.
Run ``python crypto_trajectory_data.py --acquire`` for the authorized October
1–14, 2026 batch. ``load_trajectory_panel()`` returns (prices, metadata), with
UTC interval-END labels and one native predecessor before the Jan 2021 study.
The newest day must have 24 valid source hours for all ten coins; outages
inside the retained span remain missing. Without a finalized update, the
loader explicitly falls back to the older September 30 frozen source.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from crypto_mvp_data import load_crypto_panel, VENUE_SYMBOLS, cohort, sha256

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'output/crypto_trajectory/data'
BASE = 'https://data.binance.vision/data/spot/daily/klines/'
TERMS = 'https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md'
README = 'https://github.com/binance/binance-public-data'
FIRST = pd.Timestamp('2026-10-01', tz='UTC')
AUTHORIZED_LAST = pd.Timestamp('2026-10-14', tz='UTC')
_rate_lock = threading.Lock()
_last_request = 0.0


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fetch_once(path, url, max_bytes=262144):
    """Resume immutable raw evidence; <=4 requests/sec globally, two attempts."""
    global _last_request
    sidecar = path.with_suffix(path.suffix + '.http.json')
    if path.exists():
        meta = read_json(sidecar)
        blob = path.read_bytes()
        if meta['url'] != url or digest(blob) != meta['sha256']:
            raise ValueError('Existing raw evidence hash/URL mismatch')
        return blob, meta
    for attempt in range(2):
        with _rate_lock:
            delay = 0.25 - (time.monotonic() - _last_request)
            if delay > 0:
                time.sleep(delay)
            _last_request = time.monotonic()
        request = urllib.request.Request(url, headers={
            'User-Agent': 'Noncommercial academic research; bounded historical daily update', 'Accept': '*/*'})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                blob = response.read(max_bytes + 1)
                if len(blob) > max_bytes:
                    raise ValueError('Response exceeds bounded daily archive allowance')
                meta = dict(url=url, retrieved_at_utc=now(), http_status=response.status,
                            bytes=len(blob), sha256=digest(blob),
                            etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'),
                            server_date=response.headers.get('Date'))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            write_json(sidecar, meta)
            return blob, meta
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 1:
                raise
        except (TimeoutError, OSError):
            if attempt == 1:
                raise
        time.sleep(1)
    raise RuntimeError('Unreachable retry state')


def parse_hourly_day(blob, symbol, day):
    """Official 12-field microsecond klines; exact UTC hours, inclusive close."""
    expected = f'{symbol}-1h-{day}.csv'
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != expected or members[0].file_size > 131072:
            raise ValueError('Unexpected daily ZIP member or uncompressed size')
        raw = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(raw), header=None, float_precision='round_trip')
    if frame.empty or frame.shape[1] != 12 or len(frame) > 24:
        raise ValueError('Unexpected daily hourly-kline shape')
    numeric = frame.apply(pd.to_numeric, errors='raise')
    opened = pd.DatetimeIndex(pd.to_datetime(numeric[0], unit='us', utc=True))
    closed = pd.DatetimeIndex(pd.to_datetime(numeric[6], unit='us', utc=True))
    start = pd.Timestamp(day, tz='UTC')
    if not ((opened >= start) & (opened < start + pd.Timedelta(days=1))).all():
        raise ValueError('Kline outside its declared UTC archive day')
    if not opened.is_unique or not opened.is_monotonic_increasing:
        raise ValueError('Duplicate/unsorted native kline timestamp')
    if (start > AUTHORIZED_LAST or start >= pd.Timestamp.now(tz='UTC').normalize()):
        raise ValueError('Future or unauthorized source day rejected')
    standard = (closed - opened) == pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)
    on_grid = opened.asi8 % pd.Timedelta(hours=1).value == 0
    value_columns = numeric[[1, 2, 3, 4, 5, 7, 8, 9, 10]]
    finite = np.isfinite(value_columns.to_numpy(dtype=float)).all(axis=1)
    positive = (numeric[[1, 2, 3, 4]] > 0).all(axis=1).to_numpy()
    ordered = ((numeric[2] >= numeric[[1, 3, 4]].max(axis=1)) &
               (numeric[3] <= numeric[[1, 2, 4]].min(axis=1))).to_numpy()
    volumes = (numeric[[5, 7, 8, 9, 10]] >= 0).all(axis=1).to_numpy()
    valid = np.asarray(standard) & on_grid & finite & positive & ordered & volumes
    values = numeric[4].to_numpy(dtype=float)
    values[~valid] = np.nan
    series = pd.Series(values, index=opened + pd.Timedelta(hours=1), name=symbol)
    series = series.iloc[np.flatnonzero(on_grid)]
    expected_grid = pd.date_range(start + pd.Timedelta(hours=1), periods=24, freq='h')
    series = series.reindex(expected_grid)
    series.index.name = 'bar_end_utc'
    return series, dict(raw_rows=len(frame), valid_hourly_closes=int(series.notna().sum()),
        missing_hour_slots=int(series.isna().sum()), timestamp_unit='us',
        nonstandard_duration_bars=int((~np.asarray(standard)).sum()), off_grid_bars=int((~on_grid).sum()),
        invalid_price_or_volume_bars=int((~(finite & positive & ordered & volumes)).sum()),
        zero_trade_bars=int((numeric[8] == 0).sum()), zero_volume_bars=int((numeric[5] == 0).sum()),
        zero_trade_policy='Standard valid zero-trade bars retained; not independent trade verification.',
        uncompressed_csv_sha256=digest(raw))


def acquire_day(symbol, day):
    name = f'{symbol}-1h-{day}.zip'
    folder = DATA / 'raw' / symbol
    url = f'{BASE}{symbol}/1h/{name}'
    try:
        checksum, checksum_meta = fetch_once(folder / (name + '.CHECKSUM'), url + '.CHECKSUM', 4096)
        text = checksum.decode('ascii').strip()
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?' + re.escape(name), text)
        if not match:
            raise ValueError('Publisher checksum has unexpected hash/filename convention')
        blob, zip_meta = fetch_once(folder / name, url)
        if digest(blob) != match.group(1).lower():
            raise ValueError('Publisher checksum mismatch')
        series, quality = parse_hourly_day(blob, symbol, day)
        return series, dict(symbol=symbol, interval_start_day=day, status='verified',
            archive_path=(folder / name).relative_to(ROOT).as_posix(),
            zip_http=zip_meta, checksum_http=checksum_meta, publisher_checksum_matched=True,
            quality=quality)
    except Exception as exc:
        return None, dict(symbol=symbol, interval_start_day=day, status='unavailable_or_invalid',
            source_url=url, error_type=type(exc).__name__, error=str(exc), checked_at_utc=now())


def _prices_coverage(prices):
    return dict(rows_excluding_seed=len(prices) - 1, columns=len(prices.columns),
        first_interval_end=prices.index[1].isoformat(), last_interval_end=prices.index[-1].isoformat(),
        observed_cells_excluding_seed=int(prices.iloc[1:].notna().sum().sum()),
        missing_cells_excluding_seed=int(prices.iloc[1:].isna().sum().sum()),
        complete_rows_excluding_seed=int(prices.iloc[1:].notna().all(axis=1).sum()))


def _freeze(prices, metadata):
    raw_csv = prices.to_csv(index_label='bar_end_utc', float_format='%.17g').encode('utf-8')
    provenance = json.dumps(dict(prices_csv_sha256=digest(raw_csv),
        base_prices_sha256=metadata['base_snapshot']['prices_sha256'],
        requested_end=metadata['requested_latest_interval_start_day'],
        selected_end=metadata['latest_complete_interval_start_day']), sort_keys=True).encode()
    key = digest(provenance)
    folder = DATA / 'snapshots' / key
    folder.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(raw_csv, mtime=0)
    target = folder / 'prices.csv.gz'
    if target.exists() and target.read_bytes() != blob:
        raise ValueError('Frozen trajectory snapshot key conflict')
    if not target.exists():
        target.write_bytes(blob)
    manifest = dict(metadata, snapshot_key=key, snapshot_path=folder.relative_to(ROOT).as_posix(),
                    prices_sha256=digest(blob), prices_csv_sha256=digest(raw_csv),
                    coverage=_prices_coverage(prices))
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if prior['prices_sha256'] != manifest['prices_sha256']:
            raise ValueError('Frozen trajectory manifest conflict')
        manifest = prior
    else:
        write_json(manifest_path, manifest)
    check = pd.read_csv(target, index_col=0, float_precision='round_trip')
    check.index = pd.to_datetime(check.index, utc=True)
    pd.testing.assert_frame_equal(prices, check, check_freq=False, check_names=False, check_exact=True)
    write_json(DATA / 'latest.json', dict(snapshot_path=folder.relative_to(ROOT).as_posix(),
        snapshot_key=key, prices_sha256=manifest['prices_sha256'],
        manifest_sha256=sha256(manifest_path), latest_interval_end=manifest['coverage']['last_interval_end'],
        as_of_utc=manifest['acquired_at_utc'], exact_price_reload_verified=True))
    return manifest


def acquire_extension(workers=2):
    DATA.mkdir(parents=True, exist_ok=True)
    actual = pd.Timestamp.now(tz='UTC')
    latest = min(actual.normalize() - pd.Timedelta(days=1), AUTHORIZED_LAST)
    base_prices, base_meta = load_crypto_panel('1d')
    base_copy = base_prices.copy(deep=True)
    term_meta = None
    term_error = None
    try:
        _, term_meta = fetch_once(DATA / 'source_evidence/dataset_terms.md', TERMS, 65536)
    except Exception as exc:
        term_error = f'{type(exc).__name__}: {exc}'
    days = pd.date_range(FIRST, latest, freq='D') if latest >= FIRST else []
    jobs = [(symbol, day.date().isoformat()) for day in days for symbol in VENUE_SYMBOLS]
    results, series_by_key = [], {}
    if term_error is None:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
            futures = {pool.submit(acquire_day, symbol, day): (symbol, day) for symbol, day in jobs}
            for future in as_completed(futures):
                series, result = future.result()
                results.append(result)
                if series is not None:
                    series_by_key[(result['symbol'], result['interval_start_day'])] = series
                if len(results) % 20 == 0 or len(results) == len(jobs):
                    print(json.dumps(dict(completed=len(results), planned=len(jobs),
                        publisher_verified=sum(x['status'] == 'verified' for x in results))), flush=True)
    else:
        results = [dict(symbol=symbol, interval_start_day=day, status='not_requested',
                        error='Current terms could not be retrieved; bounded update stopped: ' + term_error)
                   for symbol, day in jobs]
    daily_rows, day_quality = [], []
    for day in days:
        text = day.date().isoformat()
        row = {}
        count = {}
        for symbol in VENUE_SYMBOLS:
            hours = series_by_key.get((symbol, text))
            valid_count = int(hours.notna().sum()) if hours is not None else 0
            count[symbol] = valid_count
            row[symbol] = float(hours.iloc[-1]) if valid_count == 24 else np.nan
        daily_rows.append(pd.Series(row, name=day + pd.Timedelta(days=1)))
        day_quality.append(dict(interval_start_day=text, valid_hourly_children=count,
                                all_ten_complete=all(n == 24 for n in count.values())))
    extension = pd.DataFrame(daily_rows).reindex(columns=list(VENUE_SYMBOLS))
    extension.index.name = 'bar_end_utc'
    complete = extension.notna().all(axis=1) if len(extension) else pd.Series(dtype=bool)
    end = extension.index[np.flatnonzero(complete.to_numpy())[-1]] if complete.any() else base_prices.index[-1]
    selected = extension.loc[extension.index <= end] if len(extension) else extension
    prices = pd.concat([base_prices, selected]) if len(selected) else base_prices.copy()
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Base/extension timestamp collision or ordering error')
    pd.testing.assert_frame_equal(prices.loc[base_copy.index], base_copy, check_exact=True, check_freq=False)
    metadata = dict(schema_version=1, dataset_id='crypto_fixed10_daily_trajectory_20261015',
        acquired_at_utc=now(), clock_at_request_utc=actual.isoformat(),
        requested_latest_interval_start_day=latest.date().isoformat(),
        latest_complete_interval_start_day=(end - pd.Timedelta(days=1)).date().isoformat(),
        latest_complete_interval_end=end.isoformat(), extension_first_interval_start_day=FIRST.date().isoformat(),
        extension_days_retained=len(selected), new_complete_daily_rows=int(selected.notna().all(axis=1).sum()),
        extension_missing_cells_retained=int(selected.isna().sum().sum()),
        status='extended' if len(selected) else 'fallback_existing_frozen_2026_09_30',
        source_url='https://data.binance.vision/', source_archive_pattern=BASE + '{symbol}/1h/{symbol}-1h-{day}.zip',
        universe=cohort(), symbols=list(VENUE_SYMBOLS), interval='1d', bar_seconds=86400,
        units='USDT per base token; Binance spot close', transform='arithmetic_return',
        timestamp_policy=base_meta['timestamp_policy'], daily_policy=base_meta['daily_policy'],
        prior_row_date=base_prices.index[0].isoformat(), prior_row_role='Native predecessor before Jan 1, 2021 interval-start study; not a research observation.',
        requested_start='2021-01-01', requested_end_inclusive=(end-pd.Timedelta(days=1)).date().isoformat(),
        request_convention=base_meta['request_convention'],
        base_snapshot={k:base_meta[k] for k in ['snapshot_key','prices_sha256','source_dataset_id']},
        base_daily_values_unchanged=True, no_fill=True, terms_http=term_meta, terms_error=term_error,
        rights=base_meta['rights'], notes=base_meta['notes'] + [
            'October update is a separate frozen version; earlier sources and original research results remain unchanged.',
            'Latest retained day is the last available day with 24 valid hourly children for every fixed coin; missing intermediate days remain missing.',
            'Publisher checksum verifies archive bytes, not independent price accuracy; no market or trading mechanism inferred.',
            'New October observations are already inspected data for this project, not an untouched holdout.'],
        request_count=len(jobs), verified_archives=sum(x['status']=='verified' for x in results),
        failure_count=sum(x['status']!='verified' for x in results),
        requested_day_quality=day_quality, unavailable_or_invalid=[x for x in results if x['status']!='verified'],
        raw_receipt_path='output/crypto_trajectory/data/acquisition.json',
        local_raw_only=True, source_cost_usd=0)
    write_json(DATA / 'acquisition.json', dict(acquired_at_utc=now(), results=sorted(results,
        key=lambda r:(r['interval_start_day'],r['symbol'])), request_days=len(days), fixed_symbols=list(VENUE_SYMBOLS),
        terms_url=TERMS, terms_http=term_meta, readme_url=README, requests_per_second_cap=4,
        max_attempts_per_request=2, workers=max(1,min(workers,4)), maximum_archive_days=14,
        scope='October 1–14, 2026 only; actual UTC clock disallows incomplete/future dates; no other provider.', source_cost_usd=0))
    manifest = _freeze(prices, metadata)
    print(json.dumps(dict(status=manifest['status'], snapshot_path=manifest['snapshot_path'],
        latest_complete_interval_end=manifest['latest_complete_interval_end'], coverage=manifest['coverage'],
        verified_archives=manifest['verified_archives'], failures=manifest['failure_count'])), flush=True)
    return prices, manifest


def load_trajectory_panel():
    """Read newest finalized local snapshot; fallback is explicitly September 30.

    Returns positive raw daily closes, UTC END, fixed ordered venue-symbol
    columns, with Jan 1, 2021 END predecessor included. Never acquires data.
    """
    pointer = DATA / 'latest.json'
    if not pointer.exists():
        prices, metadata = load_crypto_panel('1d')
        metadata = dict(metadata, status='fallback_existing_frozen_2026_09_30',
                        latest_complete_interval_start_day='2026-09-30',
                        latest_complete_interval_end=prices.index[-1].isoformat(),
                        raw_data_hosted=False, no_fill=True)
        return prices, metadata
    record = read_json(pointer)
    folder = ROOT / record['snapshot_path']
    if sha256(folder / 'manifest.json') != record['manifest_sha256']:
        raise ValueError('Trajectory manifest hash mismatch')
    metadata = read_json(folder / 'manifest.json')
    if sha256(folder / 'prices.csv.gz') != metadata['prices_sha256']:
        raise ValueError('Trajectory prices hash mismatch')
    prices = pd.read_csv(folder / 'prices.csv.gz', index_col=0, float_precision='round_trip')
    prices.index = pd.to_datetime(prices.index, utc=True)
    if (list(prices.columns) != list(VENUE_SYMBOLS) or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing or len(prices) != metadata['coverage']['rows_excluding_seed']+1):
        raise ValueError('Trajectory snapshot labels/shape mismatch')
    return prices, metadata


def self_test():
    """Exercise missing/partial/off-grid/no-trade and frozen extension bounds."""
    def archive(omit=None, partial=None, off_grid=None, zero_trade=None):
        start = pd.Timestamp('2026-10-01', tz='UTC')
        lines = []
        for hour in range(24):
            if hour == omit:
                continue
            opened = start + pd.Timedelta(hours=hour) + (pd.Timedelta(minutes=1) if hour == off_grid else pd.Timedelta(0))
            ended = opened + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)
            if hour == partial:
                ended -= pd.Timedelta(seconds=1)
            trades = 0 if hour == zero_trade else 2
            lines.append(','.join(map(str, [opened.value//1000, 10, 12, 9, 11, 1,
                                           ended.value//1000, 11, trades, 1, 11, 0])))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zipped:
            zipped.writestr('BTCUSDT-1h-2026-10-01.csv', '\n'.join(lines))
        return buffer.getvalue()
    normal, stats = parse_hourly_day(archive(zero_trade=3), 'BTCUSDT', '2026-10-01')
    assert normal.notna().sum() == 24 and stats['zero_trade_bars'] == 1
    for variant in ('omit', 'partial', 'off_grid'):
        prices, stats = parse_hourly_day(archive(**{variant: 3}), 'BTCUSDT', '2026-10-01')
        assert len(prices) == 24 and prices.notna().sum() == 23 and pd.isna(prices.iloc[3])
    prices, metadata = load_trajectory_panel()
    older, _ = load_crypto_panel('1d')
    pd.testing.assert_frame_equal(prices.loc[older.index], older, check_freq=False, check_names=False, check_exact=True)
    if metadata.get('status') == 'extended':
        assert prices.index[-1] <= pd.Timestamp.now(tz='UTC').normalize()
        assert metadata['latest_complete_interval_start_day'] <= '2026-10-14'
        assert prices.iloc[-1].notna().all()  # Intermediate source outages may remain missing.
    print('PASS: strict24hourchildren; missing/partial/offgrid staymissing; zerotraderetained; oldpricesexact; nofutureday', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquire', action='store_true', help='Run the authorized bounded October update')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--self-test', action='store_true', help='Run focused offline source/extension checks')
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.acquire:
        acquire_extension(arguments.workers)
    else:
        prices, metadata = load_trajectory_panel()
        print(json.dumps(dict(status=metadata.get('status'), rows=len(prices),
                              last_interval_end=prices.index[-1].isoformat()), indent=2))
