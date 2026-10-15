"""Freeze EIA's monthly state/sector retail electricity prices, 1990 onward."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_cache import cached_closes

ROOT = Path(__file__).resolve().parent
STATES = 'AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split()


def parse_workbook(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    sectors = {str(raw.iloc[0, col]).lower(): col for col in range(4, raw.shape[1], 4)}
    data = raw.iloc[3:].copy()
    year = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    month = pd.to_numeric(data.iloc[:, 1], errors='coerce')
    valid = year.between(1990, 2100) & month.between(1, 12) & data.iloc[:, 2].isin(STATES)
    data = data.loc[valid]
    date = pd.to_datetime(dict(year=year[valid].astype(int), month=month[valid].astype(int), day=1))
    records = []
    checks = {}
    for sector in ('residential', 'commercial', 'industrial', 'total'):
        col = sectors[sector]
        revenue = pd.to_numeric(data.iloc[:, col], errors='coerce')
        sales = pd.to_numeric(data.iloc[:, col+1], errors='coerce')
        price = pd.to_numeric(data.iloc[:, col+3], errors='coerce')
        # A reported zero with no sales is an undefined average, not a free price.
        observed = (sales > 0) & price.notna()
        difference = (price - 100*revenue/sales).abs().where(observed)
        checks[sector] = dict(comparable_rows=int(difference.notna().sum()),
            rounding_difference_over_0011=int((difference > .011).sum()),
            largest_difference=float(difference.max()), zero_sales_rows=int((sales == 0).sum()))
        records.append(pd.DataFrame(dict(date=date, state=data.iloc[:, 2], sector=sector,
            price=price.where(observed), sales_mwh=sales, revenue_thousand_usd=revenue,
            publication_status=data.iloc[:, 3])))
    result = pd.concat(records, ignore_index=True)
    if result.duplicated(['date', 'state', 'sector']).any():
        raise ValueError('Duplicate state/month/sector observations in workbook')
    return result, checks


def main():
    root = ROOT/'data/expanded/retail_electricity'
    raw = root/'raw_evidence'
    probes = [json.loads((raw/name).read_text()) for name in ('historical_probe.json', 'probe.json')]
    pieces, checks = [], []
    for probe, sheet in zip(probes, ('Monthly', 'Monthly-States')):
        path = Path(probe['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != probe['sha256']:
            raise ValueError('Frozen workbook checksum changed')
        frame, check = parse_workbook(path, sheet)
        pieces.append(frame)
        checks.append(dict(source=probe, sheet=sheet, validation=check))
    records = pd.concat(pieces, ignore_index=True)
    if records.duplicated(['date', 'state', 'sector']).any():
        raise ValueError('Historical/current workbook spans overlap; resolve explicitly')
    records.to_csv(root/'state_sector_observations.csv.gz', index=False, compression='gzip')
    datasets = []
    for sector in ('total', 'residential', 'commercial', 'industrial'):
        selected = records[records.sector == sector]
        prices = selected.pivot(index='date', columns='state', values='price').sort_index().reindex(columns=sorted(STATES))
        prices = prices.reindex(pd.date_range(prices.index[0], prices.index[-1], freq='MS'))
        prices.index.name = 'reference_month'
        request = dict(source='EIA-861M state retail electricity', symbols=prices.columns.tolist(),
            start=prices.index[0].date().isoformat(), end=(prices.index[-1]+pd.offsets.MonthBegin()).date().isoformat(),
            interval='1mo', adjustment='reported average retail price; no sales means missing',
            feed=sector, timezone_policy='naive reference month start; not release timestamp',
            universe_as_of='50 US states plus DC; fixed state identities')
        _, manifest, _ = cached_closes(request, lambda:prices, root/'snapshots')
        from data_cache import _canonical_request
        path = (root/'snapshots'/_canonical_request(request)[1]).resolve()
        spec = dict(dataset_id='eia_state_retail_'+sector+'_monthly', family='electricity_retail',
            interval='1mo', transform='first_difference', units='cents/kWh', snapshot_path=str(path),
            source_url='https://www.eia.gov/electricity/data/eia861m/',
            notes='Monthly sales-weighted retail averages, '+sector+' sector; prices are not wholesale tradable quotes. Reference months are not publication dates; preliminary revisions are frozen.',
            coverage=dict(rows=len(prices), symbols=len(prices.columns), observed_cells=int(prices.notna().sum().sum()),
                missing_cells=int(prices.isna().sum().sum()), first=manifest['actual_first_timestamp'], last=manifest['actual_last_timestamp']))
        (path/'source_provenance.json').write_text(json.dumps(dict(workbooks=checks,
            publication_status_counts=selected.publication_status.value_counts().to_dict(),
            price_rule='Use reported cents/kWh only where sales_MWh > 0; no forward fill.',
            sector=sector), indent=2))
        datasets.append(spec)
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), incremental_spend_usd=0,
        datasets=datasets, source_validation=checks,
        excluded_sectors='Transportation has sparse active states; historic Other category is not consistent after 2003. Raw workbook fields remain archived.')
    output = ROOT/'output/expansion/retail_electricity.json'
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps([s['coverage'] for s in datasets], indent=2), flush=True)


if __name__ == '__main__':
    main()
