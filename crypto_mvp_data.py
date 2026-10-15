"""Frozen, fixed-cohort crypto prices for the separate research MVP.

``load_crypto_panel('1d', start='2021-01-01', end='2026-09-30')``
returns (prices, metadata).  Start/end are inclusive UTC *interval-start
calendar dates*.  Prices are labeled by interval END; the last September
daily price is therefore labeled October 1 at 00:00 UTC.  By default one
native predecessor is included and explicitly identified in metadata.

This module reads existing frozen files only.  It never downloads or fills.
Run this file to write full local research files and bounded local previews.
Raw-price exports live outside the Site's dist/ publication directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'output' / 'crypto_mvp'
PREVIEW = ROOT / 'crypto_mvp' / 'local_data'
SYMBOLS = ('BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'LTC', 'LINK', 'DOT')
COIN_IDS = ('bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano', 'solana',
            'dogecoin', 'litecoin', 'chainlink', 'polkadot')
VENUE_SYMBOLS = tuple(x + 'USDT' for x in SYMBOLS)
PANELS = {
    '1d': dict(id='crypto_daily', title='Daily crypto', bar_seconds=86400,
               bars_per_year=365, start='2021-01-01', end='2026-09-30', cap=2100),
    '1h': dict(id='crypto_hourly', title='Hourly crypto', bar_seconds=3600,
               bars_per_year=8760, start='2024-10-01', end='2026-09-30', cap=5000),
    '5m': dict(id='crypto_5m', title='Five-minute crypto', bar_seconds=300,
               bars_per_year=105120, start='2026-07-03', end='2026-09-30', cap=3000),
}
SOURCE_URL = 'https://data.binance.vision/'
TERMS_URL = 'https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md'
NOTES = [
    'Frozen Binance spot prices, quoted in USDT per base token; USDT is not USD.',
    'The fixed proposed cohort is selected by identity/coverage, not historical performance.',
    'Current-ranking source membership causes survivor/look-ahead selection; this is not a point-in-time universe.',
    'Current coin IDs are checked; complete historical contract continuity has not been audited.',
    'UTC interval-END is the earliest possible availability; actual publication latency is unmeasured.',
    'No zero or forward fill; source-invalid or missing prices remain missing.',
    'Publisher checksum agreement verifies archived bytes, not independent price accuracy.',
    'Standard on-grid zero-trade bars are retained by the source policy; no independent trade validation is claimed.',
]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _assert_portable_public_json(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_portable_public_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable_public_json(item)
    elif isinstance(value, str) and (re.match(r'^[A-Za-z]:[\\/]', value) or '\\Users\\' in value):
        raise AssertionError('Absolute local path leaked into portable preview data')


def _iso(value) -> str | None:
    return None if value is None else pd.Timestamp(value).isoformat().replace('+00:00', 'Z')


def _descriptors():
    rows = _read_json(ROOT / 'output/expansion/crypto_results.json')['datasets']
    result = {}
    for row in rows:
        interval = row['interval']
        if interval in ('1d', '1h') and not row['dataset_id'].endswith('_identity_v4'):
            continue
        if interval in result:
            raise ValueError(f'Duplicate finalized source for {interval}')
        result[interval] = row
    return result


def cohort() -> list[dict]:
    review = _read_json(ROOT / 'data/expanded/crypto/identity_review.json')
    records = {row['venue_symbol']: row for row in review['records']}
    common = set.intersection(*(set(_read_json(Path(row['snapshot_path']) / 'manifest.json')['actual_symbols'])
                                for row in _descriptors().values()))
    result = []
    for symbol, coin_id, venue in zip(SYMBOLS, COIN_IDS, VENUE_SYMBOLS):
        record = records.get(venue, {})
        if (venue not in common or record.get('status') != 'confirmed_current_coin_id'
                or record.get('coin_id') != coin_id):
            raise ValueError(f'Proposed fixed cohort member is not archive/identity verified: {venue}')
        result.append(dict(id=coin_id, label=symbol, venue_symbol=venue,
                           identity_status=record['status'],
                           historical_identity_limit=record['historical_identity_limit']))
    return result


@lru_cache(maxsize=5)
def _native(interval: str):
    descriptor = _descriptors()[interval]
    folder = Path(descriptor['snapshot_path'])
    manifest = _read_json(folder / 'manifest.json')
    price_path = folder / 'prices.csv.gz'
    if sha256(price_path) != manifest['sha256']:
        raise ValueError(f'Frozen source hash mismatch: {descriptor["dataset_id"]}')
    label = manifest['index_name']
    prices = pd.read_csv(price_path, usecols=[label, *VENUE_SYMBOLS],
                         index_col=label, float_precision='round_trip')
    prices.index = pd.to_datetime(prices.index, utc=True)
    prices = prices.loc[:, list(VENUE_SYMBOLS)]
    if (len(prices) != manifest['rows'] or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing):
        raise ValueError(f'Frozen source shape/order mismatch: {interval}')
    if (prices.index[0] != pd.Timestamp(manifest['actual_first_timestamp'])
            or prices.index[-1] != pd.Timestamp(manifest['actual_last_timestamp'])):
        raise ValueError(f'Frozen source timestamp bounds mismatch: {interval}')
    arr = prices.to_numpy(dtype=float)
    if np.isinf(arr).any() or ((arr <= 0) & np.isfinite(arr)).any():
        raise ValueError(f'Nonpositive/infinite selected crypto close: {interval}')
    return prices, descriptor, manifest


def _utc_calendar_date(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')
    if timestamp != timestamp.normalize():
        raise ValueError('start/end must be UTC calendar dates at midnight')
    return timestamp


def _coverage(prices: pd.DataFrame, seconds: int) -> dict:
    values = prices.to_numpy(dtype=float)
    steps = prices.index.to_series().diff().dt.total_seconds()
    return dict(rows=len(prices), columns=prices.shape[1],
                first_interval_end=_iso(prices.index[0]) if len(prices) else None,
                last_interval_end=_iso(prices.index[-1]) if len(prices) else None,
                observed_cells=int(np.isfinite(values).sum()), missing_cells=int(np.isnan(values).sum()),
                complete_rows=int(np.isfinite(values).all(axis=1).sum()),
                timestamp_gap_count=int((steps.iloc[1:] != seconds).sum()),
                missing_by_symbol={str(k): int(v) for k, v in prices.isna().sum().items()})


def load_crypto_panel(interval: str, *, start: str | None = None,
                      end: str | None = None, include_preceding: bool = True):
    """Return (raw_prices, metadata) for inclusive UTC interval-start dates.

    The default requested span is research daily 2021+, hourly two years,
    or five-minute 90 days, all ending September 30, 2026.  An included
    native predecessor is a price seed, not an extra research observation.
    Returns are always calculated natively before trimming/joining.
    """
    if interval not in PANELS:
        raise ValueError(f'interval must be one of {tuple(PANELS)}')
    spec = PANELS[interval]
    start = start or spec['start']
    end = end or spec['end']
    lower = _utc_calendar_date(start)
    upper = _utc_calendar_date(end) + pd.Timedelta(days=1)
    if lower >= upper:
        raise ValueError('end precedes start')
    delta = pd.Timedelta(seconds=spec['bar_seconds'])
    prices, descriptor, manifest = _native(interval)
    if lower < prices.index[0] - delta or upper > prices.index[-1]:
        raise ValueError('Requested calendar span extends beyond the frozen native timestamp bounds')
    choose = ((prices.index - delta) >= lower) & (prices.index <= upper)
    positions = np.flatnonzero(choose)
    if not len(positions):
        raise ValueError('No frozen observations in requested interval-start dates')
    first, last = int(positions[0]), int(positions[-1])
    predecessor = first - 1 if first > 0 else None
    begin = predecessor if include_preceding and predecessor is not None else first
    result = prices.iloc[begin:last + 1].copy()
    observed = prices.iloc[first:last + 1]
    metadata = dict(
        id=spec['id'], interval=interval, bar_seconds=spec['bar_seconds'],
        bars_per_year=spec['bars_per_year'], annualization='365 UTC days per year',
        universe=cohort(), symbols=list(VENUE_SYMBOLS),
        snapshot_key=Path(descriptor['snapshot_path']).name,
        source_dataset_id=descriptor['dataset_id'], prices_sha256=manifest['sha256'],
        requested_start=lower.date().isoformat(), requested_end_inclusive=_utc_calendar_date(end).date().isoformat(),
        request_convention='Inclusive UTC interval-start calendar dates; source labels are interval END.',
        requested_end_exclusive=_iso(upper), first_interval_end=_iso(observed.index[0]),
        last_interval_end=_iso(observed.index[-1]),
        preceding_row_included=bool(begin < first),
        preceding_row_date=_iso(prices.index[predecessor]) if predecessor is not None else None,
        preceding_row_role='native predecessor used only to seed returns',
        timestamp_policy='UTC interval-END; bar-start = label minus bar_seconds; actual publication latency unmeasured',
        transform='arithmetic_return; finite positive endpoints only; mask every gap-spanning change',
        daily_policy='Derived daily close requires all 24 observed standard, on-grid hourly children; incomplete day is NaN.',
        units='USDT per base token', venue='Binance spot', quote_currency='USDT', source_url=SOURCE_URL,
        research_coverage=_coverage(observed, spec['bar_seconds']),
        native_coverage=_coverage(prices, spec['bar_seconds']),
        rights=dict(license='CC BY-NC-SA 4.0 with additional Binance dataset terms',
                    license_url='https://creativecommons.org/licenses/by-nc-sa/4.0/',
                    terms_url=TERMS_URL,
                    attribution='Binance Public Data, frozen archives acquired October 2026; noncommercial educational research derivative.',
                    caution='Repository MIT code license is separate from dataset rights; commercial/redistribution permissions require terms review.',
                    project_publication_policy='Raw close datasets stay local because dataset hosting language in section 4.3 is ambiguous alongside the CC grant. Only attributed derivative analytics/figures are prepared for the Site, with CC BY-NC-SA under section 4.5; this is a conservative project choice, not legal certainty.'),
        notes=NOTES,
    )
    return result, metadata


def transform_crypto_prices(prices: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Use the shared panel transformation, including native-gap masking."""
    from expanded_spectral import transform_panel
    changes, _ = transform_panel(prices, dict(transform='arithmetic_return', bar_seconds=PANELS[interval]['bar_seconds']))
    return changes


def _nullable_rows(frame: pd.DataFrame):
    return [[float(v) if np.isfinite(v) else None for v in row]
            for row in frame.to_numpy(dtype=float)]


def _quality_context(interval: str, start: str, end: str) -> dict:
    # Native daily rows are derived from hourly raw bars. Monthly summaries do
    # not locate zero-trade bars individually; never pretend these are exact
    # study-window flag dates/counts for a partial month.
    source_interval = '1h' if interval == '1d' else interval
    coverage = _read_json(ROOT / f'data/expanded/crypto/coverage_{source_interval}.json')
    by_symbol = {row['symbol']: row for row in coverage['coverage']}
    lower, upper = start[:7], end[:7]
    rows = []
    for symbol in VENUE_SYMBOLS:
        months = [x for x in by_symbol[symbol].get('raw_statistics', [])
                  if lower <= x['month'] <= upper]
        counts = {key: (sum(int(x[key]) for x in months)
                        if months and all(key in x for x in months) else None) for key in
                  ('zero_trade_bars', 'zero_volume_bars', 'nonstandard_duration_bars_masked',
                   'off_grid_bars_excluded')}
        rows.append(dict(venue_symbol=symbol, raw_months=len(months), **counts))
    return dict(source_interval=source_interval, first_raw_month=lower, last_raw_month=upper,
                scope='Whole source months intersecting the research interval; not exact partial-month study counts.',
                policy='Recorded invalid raw bars masked/excluded; standard zero-trade/volume bars retained. Null flag counts mean not recorded in that interval receipt, not zero.',
                flag_dates_available_in_price_export=False, assets=rows)


def _compare_frames(derived, source, description: str) -> dict:
    if not derived.index.equals(source.index) or not derived.columns.equals(source.columns):
        raise AssertionError(f'Cross-interval label mismatch: {description}')
    a, b = derived.to_numpy(dtype=float), source.to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    mask_disagreements = int((np.isnan(a) != np.isnan(b)).sum())
    differences = np.abs(a[both] - b[both])
    result = dict(method=description, compared_cells=int(both.sum()),
                  exact_equal_cells=int((a[both] == b[both]).sum()),
                  missing_mask_disagreements=mask_disagreements,
                  maximum_absolute_difference=float(differences.max()) if len(differences) else None)
    result['status'] = 'exact_match' if mask_disagreements == 0 and np.array_equal(a[both], b[both]) else 'differences'
    return result


def cross_interval_checks() -> dict:
    hourly, _, _ = _native('1h')
    daily, _, _ = _native('1d')
    # Match the acquisition's strict UTC all-24-child rule, including missing
    # values. A derived value is never reconstructed from fewer children.
    starts = hourly.copy()
    starts.index = starts.index - pd.Timedelta(hours=1)
    resampler = starts.resample('1D')
    derived_daily = resampler.last().where(resampler.count() == 24)
    derived_daily.index += pd.Timedelta(days=1)
    derived_daily = derived_daily.loc['2021-01-02':'2026-10-01']
    daily_check = _compare_frames(derived_daily, daily.reindex(derived_daily.index),
                                 'All ten assets, Jan 2021–Sep 2026: daily close from all 24 valid hourly children.')
    minute, _, _ = _native('1m')
    five, _, _ = _native('5m')
    minute = minute.loc['2026-09-01 00:01':'2026-10-01 00:00']
    starts = minute.copy()
    starts.index -= pd.Timedelta(minutes=1)
    resampler = starts.resample('5min')
    derived_five = resampler.last().where(resampler.count() == 5)
    derived_five.index += pd.Timedelta(minutes=5)
    five_check = _compare_frames(derived_five, five.reindex(derived_five.index),
                                'All ten assets, Sep 2026 only: published five-minute close vs final one-minute close with all five children valid.')
    five_starts = five.loc['2026-09-01 00:05':'2026-10-01 00:00'].copy()
    five_starts.index -= pd.Timedelta(minutes=5)
    resampler = five_starts.resample('1h')
    derived_hour = resampler.last().where(resampler.count() == 12)
    derived_hour.index += pd.Timedelta(hours=1)
    hourly_check = _compare_frames(derived_hour, hourly.reindex(derived_hour.index),
                                  'All ten assets, Sep 2026 only: published hourly close vs final five-minute close with all twelve children valid.')
    stored = _read_json(ROOT / 'output/expansion/crypto_cross_interval_check.json')
    stored_selected = [row for row in stored['results'] if row['symbol'] in VENUE_SYMBOLS]
    return dict(checks=[daily_check, five_check, hourly_check],
                stored_monthly_evidence=dict(source='output/expansion/crypto_cross_interval_check.json',
                    source_sha256=sha256(ROOT / 'output/expansion/crypto_cross_interval_check.json'),
                    checked_at_utc=stored['checked_at_utc'], selected_results=stored_selected,
                    scope='Previously saved Sep 2026 check covers BTC, ETH, SOL and BNB among the fixed cohort.'),
                interpretation='Same-venue interval aggregation checks; neither independent venue reconciliation nor exhaustive five-minute history verification.')


def audit_native_sources() -> dict:
    rows = []
    identities = {x['symbol']: x for x in _read_json(ROOT / 'output/expansion/crypto_identity_audit.json')['inventory']}
    for interval, descriptor in _descriptors().items():
        print(f'Audit frozen {interval} source...', flush=True)
        prices, _, manifest = _native(interval)
        seconds = {'1m': 60, '5m': 300, '10m': 600, '1h': 3600, '1d': 86400}[interval]
        rows.append(dict(interval=interval, source_dataset_id=descriptor['dataset_id'],
                         snapshot_key=Path(descriptor['snapshot_path']).name,
                         snapshot_prices_sha256=manifest['sha256'], hash_verified_now=True,
                         source_universe_columns=len(manifest['actual_symbols']),
                         source_all_universe_rows=manifest['rows'],
                         source_all_universe_observed_cells=manifest['rows'] * len(manifest['actual_symbols']) - manifest['missing_cells'],
                         source_all_universe_count_basis='Frozen manifest summary; all ten selected columns are read and counted now.',
                         selected_fixed_cohort_coverage=_coverage(prices, seconds),
                         derivation=descriptor.get('notes', [])))
    completion = _read_json(ROOT / 'output/expansion/crypto_completion.json')
    return dict(cohort=cohort(), proposed_candidates=list(SYMBOLS), unavailable_candidates=[],
                sources=rows, historical_identity=[identities[x] for x in VENUE_SYMBOLS],
                recorded_acquisition_checks=dict(raw_archive_files=completion['raw_archive_files'],
                    publisher_checksums_matched=completion['verified_archive_files'],
                    checked_at_utc=completion['completed_at_utc'],
                    scope='Recorded acquisition receipt; raw ZIP hashes were not all recomputed during this export.'),
                zero_trade_full_hourly_history_context=[x for x in _read_json(ROOT / 'output/expansion/crypto_quality_flags.json')['assets']
                                                        if x['symbol'] in VENUE_SYMBOLS],
                source_claim_limit=NOTES)


def _write_research(prices: pd.DataFrame, metadata: dict) -> dict:
    path = LOCAL / f'data_full_{metadata["interval"]}.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label='bar_end_utc', float_format='%.17g',
                  compression={'method': 'gzip', 'mtime': 0})
    reloaded = pd.read_csv(path, index_col=0, float_precision='round_trip')
    reloaded.index = pd.to_datetime(reloaded.index, utc=True)
    pd.testing.assert_frame_equal(prices, reloaded, check_freq=False, check_names=False, check_exact=True)
    receipt = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path), bytes=path.stat().st_size,
                   rows_including_seed=len(prices), exact_price_reload_verified=True,
                   columns=list(prices.columns),
                   research_change_coverage=_coverage(transform_crypto_prices(prices, metadata['interval']), metadata['bar_seconds']),
                   metadata=metadata)
    _write_json(LOCAL / f'data_full_{metadata["interval"]}.json', receipt)
    return receipt


def _preview_panel(prices: pd.DataFrame, metadata: dict) -> dict:
    spec = PANELS[metadata['interval']]
    # The loader always supplies the native seed for these default spans.
    if not metadata['preceding_row_included']:
        raise AssertionError('Default research panel is missing its predecessor')
    observations = prices.iloc[1:]
    sample = observations.iloc[-spec['cap']:]
    sample_begin = prices.index.get_loc(sample.index[0])
    seed = prices.iloc[[sample_begin - 1]]
    seeded_sample = pd.concat([seed, sample])
    changes = transform_crypto_prices(seeded_sample, metadata['interval'])
    if not changes.index.equals(sample.index):
        raise AssertionError('Sample transform/seed alignment mismatch')
    columns = [{**row, 'variable_id': f'{spec["id"]}::{row["venue_symbol"]}'} for row in metadata['universe']]
    payload = dict(id=spec['id'], title=spec['title'], interval=metadata['interval'],
        bar_seconds=spec['bar_seconds'], bars_per_year=spec['bars_per_year'],
        sample=dict(dates=[_iso(x) for x in sample.index], values=_nullable_rows(sample), columns=columns),
        preceding_row=dict(date=_iso(seed.index[0]), values=_nullable_rows(seed)[0],
                           role=metadata['preceding_row_role']),
        coverage=dict(sample=_coverage(sample, spec['bar_seconds']), research=metadata['research_coverage'],
                      native=metadata['native_coverage']),
        masks=dict(missing_price_cells=np.argwhere(sample.isna().to_numpy()).tolist(),
                   invalid_return_cells=np.argwhere(changes.isna().to_numpy()).tolist(),
                   cell_coordinates='[zero-based sample row, zero-based column]',
                   timestamp_gap_rows=[int(x) for x in np.flatnonzero(
                       seeded_sample.index.to_series().diff().dt.total_seconds().iloc[1:].ne(spec['bar_seconds']).to_numpy())],
                   policy='JSON null is missing; no zero/forward fill; returns require positive valid endpoints and exactly one native interval.'),
        cohort=metadata['universe'], metadata=metadata,
        source_quality_context=_quality_context(metadata['interval'], metadata['requested_start'], metadata['requested_end_inclusive']),
        research_file_policy='All raw prices remain local, including bounded previews. Hosted Site contains only attributed derivative research analytics.')
    return payload


def source_review() -> dict:
    return dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), incremental_spend_usd=0,
        recommendation='Existing frozen data are sufficient for the fixed-ten descriptive MVP and daily chronological research; no new acquisition is needed.',
        sufficiency_limits='Adequate history is not evidence of forecast skill. Missing days, single-venue USDT exposure, current-cohort selection, identity continuity and latency remain limits.',
        sources=[
            dict(provider='Binance Public Data', url='https://github.com/binance/binance-public-data',
                 status='Existing frozen source; no new download.', cost='Official dataset terms state access is free of charge.',
                 coverage='Public daily/monthly spot archives; timestamps use microseconds from January 2025; publisher may revise archives.',
                 rights='Dataset CC BY-NC-SA 4.0 plus additional terms; academic noncommercial research described. Section 4.3 hosting language is ambiguous alongside section 3.3 CC rights; raw closes remain local. Only attributed same-license derivative analytics/figures are prepared for the Site under section 4.5. This project choice is not a legal determination. Repository MIT code license does not govern market data.', terms_url=TERMS_URL),
            dict(provider='Kraken', url='https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data',
                 status='Alternative reviewed only; no download or fee incurred.',
                 coverage='Official public bulk OHLCVT CSVs: full history through June 30, 2026 plus listed 2026Q3 incremental archive. Intervals include 1/5/60/1440 minutes; no-trade intervals are absent.',
                 cost='Public direct downloads listed; no paid purchase requested or made.',
                 rights='Public download availability alone does not establish redistribution/commercial rights; review applicable Kraken terms before reuse.'),
            dict(provider='Coinbase Exchange', url='https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles',
                 status='Alternative API reviewed only; no request batching or account created.',
                 coverage='Candle API supports 1m/5m/1h/1d, at most 300 candles per request. Missing no-tick buckets and incomplete history are documented; timestamps label bucket starts.',
                 cost='No paid market-data product was purchased; this endpoint documentation is not a full pricing/rights grant.',
                 rights='Review Coinbase market-data terms before redistribution. Venue/quote differences require separate provenance, not silent substitution.'),
        ])


def build_exports() -> dict:
    LOCAL.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    audit = audit_native_sources()
    _write_json(LOCAL / 'data_audit.json', audit)
    checks = cross_interval_checks()
    if any(row['status'] != 'exact_match' for row in checks['checks']):
        raise AssertionError('Same-source cross-interval consistency check found differences; inspect before export')
    _write_json(LOCAL / 'data_cross_interval.json', checks)
    review = source_review()
    _write_json(LOCAL / 'data_sources.json', review)
    _write_json(PREVIEW / 'sources.json', review)
    panels, full_receipts = [], []
    for interval, spec in PANELS.items():
        print(f'Export {interval} fixed-ten research/sample...', flush=True)
        prices, metadata = load_crypto_panel(interval)
        expected_rows = int((_utc_calendar_date(spec['end']) + pd.Timedelta(days=1)
                             - _utc_calendar_date(spec['start'])).total_seconds() / spec['bar_seconds'])
        if len(prices) != expected_rows + 1:
            raise AssertionError(f'Research rows/seed differ from declared frozen UTC grid: {interval}')
        receipt = _write_research(prices, metadata)
        full_receipts.append({k: v for k, v in receipt.items() if k != 'metadata'})
        payload = _preview_panel(prices, metadata)
        payload['cross_interval_validation'] = checks
        path = PREVIEW / f'{spec["id"]}.json'
        _write_json(path, payload)
        # JSON float/null round trip is exact for observed source values.
        loaded = _read_json(path)
        expected = prices.iloc[1:].iloc[-spec['cap']:]
        actual = np.array(loaded['sample']['values'], dtype=float)
        if not np.array_equal(actual, expected.to_numpy(dtype=float), equal_nan=True):
            raise AssertionError(f'Local preview exact value reload failed: {interval}')
        _assert_portable_public_json(loaded)
        panels.append(dict(id=spec['id'], title=spec['title'], interval=interval,
                           bar_seconds=spec['bar_seconds'], bars_per_year=spec['bars_per_year'],
                           path=path.name, coverage=payload['coverage'], cohort=metadata['universe'],
                           sha256=sha256(path), sample_rows=len(expected),
                           first_interval_end=_iso(expected.index[0]), last_interval_end=_iso(expected.index[-1])))
    catalog = dict(schema_version=1, frozen_as_of='2026-09-30 UTC interval-start date',
                   generated_at_utc=datetime.now(timezone.utc).isoformat(), panels=panels,
                   cohort=cohort(), requested_candidates=list(SYMBOLS), missing_candidates=[],
                   units='USDT per base token', notes=NOTES,
                   source_review_path='sources.json', full_research_hosted=False, raw_prices_hosted=False,
                   local_preview_directory=PREVIEW.relative_to(ROOT).as_posix(),
                   sample_policy='Daily complete requested research span; recent hourly <=5000 and five-minute <=3000 observations; one native seed separate.')
    _write_json(PREVIEW / 'catalog.json', catalog)
    for path in PREVIEW.glob('*.json'):
        _assert_portable_public_json(_read_json(path))
    receipt = dict(status='pass', generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   panel_count=len(panels), fixed_assets=len(cohort()),
                   full_research_files=full_receipts,
                   local_preview_panels=panels, source_checks=checks,
                   portable_preview_json_checked=True,
                   local_preview_directory=PREVIEW.relative_to(ROOT).as_posix(),
                   raw_prices_hosted=False,
                   cross_interval_note='Aggregation agreements are descriptive same-source checks, not independent market-price verification.',
                   local_preview_bytes=sum(x.stat().st_size for x in PREVIEW.glob('*') if x.is_file()),
                   local_artifact_bytes_excluding_validation_receipt=sum(
                       x.stat().st_size for x in LOCAL.glob('data_*') if x.is_file() and x.name != 'data_validation.json'),
                   source_downloads=0, spend_usd=0)
    _write_json(LOCAL / 'data_validation.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k in ('status', 'panel_count', 'fixed_assets', 'local_preview_bytes', 'local_artifact_bytes_excluding_validation_receipt')}, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_exports()
