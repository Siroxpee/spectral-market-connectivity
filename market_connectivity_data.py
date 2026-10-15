"""Frozen fixed-12 ETF market selection and native-session preparation.

load_panel() returns (prepared_return_DataFrame, metadata). Columns are the
ordered bare ETF symbols; rows are source-native US session DATE labels,
not UTC publication instants. Friday/Monday and holidays are valid adjacent
source sessions. No crypto-style 86,400-second mask, filling or download.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'output/market_selection/selected_panel.json'
SNAPSHOT = 'data/expanded/financial/snapshots/6803319d2d24e5a4ff01ecb483f1bd1872e0c69972cbbbfa1d5a69c47b75a7a5'
PRICE_HASH = 'd279458ba10b2cb9e30986b073055c2d1912d85eda5e7b42aa9e1331153de235'
COHORT = ('XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','GLD','TLT','DBC')
EXPOSURES = ('Materials','Energy equities','Financials','Industrials','Technology',
             'Consumer staples','Utilities','Health care','Consumer discretionary',
             'Gold trust','Long-term US Treasury bonds','Diversified commodity futures fund')


def _hash(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _source():
    folder=ROOT/SNAPSHOT
    manifest=_json(folder/'manifest.json')
    if manifest['sha256']!=PRICE_HASH or _hash(folder/'prices.csv.gz')!=PRICE_HASH:
        raise ValueError('Chosen immutable ETF price-file hash changed')
    index_name=manifest['index_name']
    prices=pd.read_csv(folder/'prices.csv.gz',usecols=[index_name,'SPY',*COHORT],
                       index_col=index_name,float_precision='round_trip')
    prices.index=pd.to_datetime(prices.index,format='ISO8601')
    if prices.index.tz is not None or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Expected unique increasing naive provider session-date labels')
    # Restrict to the first common inception support, never drop any later row.
    first=prices.loc[:,list(COHORT)].notna().all(axis=1)
    beginning=prices.index[np.flatnonzero(first.to_numpy())[0]]
    prices=prices.loc[beginning:]
    if prices['SPY'].isna().any():
        raise ValueError('Parent source-session support series has a missing quote')
    session_dates=prices.index
    selected=prices.loc[:,list(COHORT)].copy()
    selected.index.name='session_label'
    values=selected.to_numpy(dtype=float)
    if np.isinf(values).any() or ((values<=0)&np.isfinite(values)).any():
        raise ValueError('Arithmetic-return source contains nonpositive/infinite level')
    features=selected.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan).iloc[1:]
    return selected,features,manifest,session_dates


def _frame_hash(frame):
    return hashlib.sha256(frame.to_csv(float_format='%.17g').encode('utf-8')).hexdigest()


def freeze_selection():
    """Freeze identity/coverage choice; no target, model, score or fit is made."""
    prices,features,manifest,sessions=_source()
    choice=dict(schema_version=1,panel_id='fixed12_diversified_etf_daily_v1',
        selection_frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_basis='Economic exposure diversity, common historical coverage and first1000-session descriptive connection audit; no forecast target or model score used.',
        source_dataset_id='yahoo_etf_daily_long',source_snapshot_path=SNAPSHOT,
        snapshot_key=Path(SNAPSHOT).name,prices_sha256=PRICE_HASH,
        manifest_sha256=_hash(ROOT/SNAPSHOT/'manifest.json'),
        source_url='https://finance.yahoo.com/',source='Yahoo via yfinance',
        adjustment=manifest['request']['adjustment'],feed=manifest['request']['feed'],
        ordered_symbols=list(COHORT),
        variables=[dict(symbol=s,source_variable=s,variable_id='etf12_daily::'+s,entity_id='yahoo:ticker:'+s,
                        exposure=e,group='equity_sector' if i<9 else 'diversifier')
                   for i,(s,e) in enumerate(zip(COHORT,EXPOSURES))],
        interval='1d',transform='arithmetic_return',feature_units='dimensionless fractional return',
        source_units='USD/share; provider adjusted close levels',annualization=252,bars_per_year=252,
        price_rows=len(prices),feature_rows=len(features),
        first_price_session=prices.index[0].isoformat(),last_price_session=prices.index[-1].isoformat(),
        first_feature_session=features.index[0].isoformat(),last_feature_session=features.index[-1].isoformat(),
        prices_missing_cells=int(prices.isna().sum().sum()),features_missing_cells=int(features.isna().sum().sum()),
        feature_matrix_sha256=_frame_hash(features),
        source_session_calendar_sha256=hashlib.sha256('\n'.join(d.isoformat() for d in sessions).encode()).hexdigest(),
        calendar=dict(type='source_native_session_dates',support='Full frozen ETF panel date grid, with SPY observed on every retained source row.',
            timestamp_policy='Naive provider-local session DATE labels; not UTC instants or verified publication timestamps.',
            adjacent_rule='Consecutive rows on this fixed source-session grid are adjacent sessions; Friday-Monday and holiday breaks are valid.',
            gap_policy='Retain every source-calendar row after first common inception; a missing selected quote invalidates both adjacent returns. Never delete missing dates or fill values.',
            elapsed_seconds_mask=False,independent_exchange_calendar_verified=False,
            limitation='An entirely absent source-wide trading session cannot be detected from this provider grid alone; no independent exchange-calendar library is installed.'),
        missing_policy='No zero/forward/interpolating fill. Fixed cohort remains ordered; incomplete/nonconstant windows must be marked invalid, not shrink the universe.',
        allowed_windows=[20,40,60,120],allowed_histories=[20,60,100],allowed_horizons=[20,60,100],
        early_descriptive_selection_period=dict(start='2006-02-07',end='2010-01-27',returns=1000,
            purpose='Compare connection diversity only, not future targets, model selection or strategy returns.'),
        notes=['The cohort was chosen in2026 after seeing data availability and early descriptive relationships; it is not a historically preselected or point-in-time universe.',
            'The source uses auto-adjusted prices frozen in2026, not a point-in-time corporate-action/publication database.',
            'Sector holdings/classifications changed, notably real estate2016 and communication services2018; fund identity does not freeze economic composition.',
            'GLD is a gold trust; TLT is long-duration Treasury exposure; DBC is a commodity futures fund including roll/collateral effects, not spot commodity prices.',
            'Shared equity beta remains; diversified exposures do not guarantee negative correlations or a connected negative graph in every window.',
            'Values end atOct8 2026 and are not live/current quotes. No newly unseen forecast holdout is implied.',
            'Research choice, not an investment recommendation. Costs, executable prices, latency and practical profit have not been established.'],
        raw_source_unchanged=True,downloads=0,spend_usd=0)
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    if CONFIG.exists():
        prior=_json(CONFIG)
        for key in ['panel_id','source_snapshot_path','prices_sha256','ordered_symbols','feature_matrix_sha256']:
            if prior[key]!=choice[key]:
                raise ValueError('Frozen market selection conflict; use a new selection version')
        return prior
    CONFIG.write_text(json.dumps(choice,indent=2)+'\n',encoding='utf-8')
    return choice


def load_panel():
    """Return prepared native-session returns and frozen selection metadata.

    This loader performs no acquisition, filling, target generation or scoring.
    Column order is COHORT; do not apply a second price transformation or an
    elapsed-24-hour gap mask. W/history/H counts mean native observations.
    """
    if not CONFIG.exists():
        raise FileNotFoundError('Freeze the approved source choice before modeling')
    metadata=_json(CONFIG)
    prices,features,manifest,sessions=_source()
    if (metadata['ordered_symbols']!=list(features.columns) or len(features)!=metadata['feature_rows']
            or _frame_hash(features)!=metadata['feature_matrix_sha256']):
        raise ValueError('Prepared selected panel differs from frozen selection receipt')
    metadata=dict(metadata,loader='market_connectivity_data.load_panel',prepared=True,
                  features_already_transformed=True,source_price_rows_include_first_return_seed=True)
    return features,metadata


if __name__=='__main__':
    choice=freeze_selection()
    frame,metadata=load_panel()
    print(json.dumps(dict(panel=metadata['panel_id'],shape=list(frame.shape),first=frame.index[0].isoformat(),
                         last=frame.index[-1].isoformat(),missing=int(frame.isna().sum().sum()),
                         feature_matrix_sha256=metadata['feature_matrix_sha256']),indent=2))
