"""Stream the frozen EIA EBA JSONL archive into UTC operating-data panels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import pandas as pd
from data_cache import cached_closes, _canonical_request

ROOT = Path(__file__).resolve().parent


def main():
    root = ROOT/'data/expanded/grid_operations'
    receipt = json.loads((root/'raw_evidence/download.json').read_text())
    path = Path(receipt['path'])
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024), b''): digest.update(chunk)
    if digest.hexdigest() != receipt['sha256']: raise ValueError('Bulk archive changed')
    groups = {'demand':{}, 'generation':{}, 'interchange':{}}
    metadata = []
    uncompressed = 0
    mark = 0
    with zipfile.ZipFile(path) as archive, archive.open('EBA.txt') as handle:
        for line in handle:
            uncompressed += len(line)
            item = json.loads(line)
            sid = item.get('series_id','')
            group = 'demand' if sid.endswith('.D.H') else 'generation' if sid.endswith('.NG.H') else 'interchange' if sid.endswith('.ID.H') else None
            if group and item.get('f') == 'H' and item.get('units') == 'megawatthours':
                pairs = item.get('data',[])
                if pairs:
                    values = pd.DataFrame(pairs,columns=['date','value'])
                    values['date'] = pd.to_datetime(values.date, format='%Y%m%dT%H', utc=True, errors='raise')
                    values['value'] = pd.to_numeric(values.value,errors='coerce')
                    if values.date.duplicated().any(): raise ValueError('Duplicate UTC hours: '+sid)
                    series = values.set_index('date').value.sort_index()
                    series = series.loc[(series.index >= pd.Timestamp('2019-01-01',tz='UTC')) & (series.index <= pd.Timestamp.now(tz='UTC'))]
                    if series.notna().any():
                        groups[group][sid] = series
                        metadata.append({k:v for k,v in item.items() if k != 'data'})
            if uncompressed-mark > 500_000_000:
                mark = uncompressed
                print(f'EIA grid parsed {uncompressed/1e9:.2f} GB; selected {sum(len(g) for g in groups.values())} hourly series',flush=True)
    (root/'series_metadata.json').write_text(json.dumps(metadata,indent=2))
    datasets = []
    for group, series in groups.items():
        # Regional totals and both reported interchange directions remain explicit
        # labeled variables. They are never presented as independent locations.
        frame = pd.DataFrame(series).sort_index()
        frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq='h'))
        frame.index.name = 'source_hour_ending_utc'
        if frame.empty: raise ValueError('No hourly observations: '+group)
        request = dict(source='EIA public EBA bulk operating data',symbols=frame.columns.tolist(),
            start=frame.index[0].isoformat(),end=(frame.index[-1]+pd.Timedelta(hours=1)).isoformat(),
            interval='1h',adjustment='as reported; signed flows preserved; no filling',
            feed=group,timezone_policy='source UTC hour-ending labels; not publication times',
            universe_as_of='all nonempty hourly '+group+' series in frozen 2026-10-09 bulk archive')
        _, manifest, _ = cached_closes(request,lambda:frame,root/'snapshots')
        snapshot = (root/'snapshots'/_canonical_request(request)[1]).resolve()
        note = 'Reported hourly energy in MWh, not prices or transmission efficiency. Source UTC labels represent hour ending, not publication times. '+('Variables include nested regional totals and balancing authorities; account for aggregate overlap.' if group != 'interchange' else 'Variables are directional reported interchanges, including reverse-direction counterparts; spectral correlation is a graph of flow variables, not a physical route/capacity model.')
        if group=='generation':note+=' Prepared variables are total generation plus natural-gas subclasses selected by the NG.H suffix; other fuel subclasses are not prepared.'
        spec = dict(dataset_id='eia_grid_'+group+'_hourly_2019', family='grid_operations',
            interval='1h',bar_seconds=3600,transform='first_difference',units='MWh per hourly interval',
            snapshot_path=str(snapshot),source_url='https://www.eia.gov/opendata/bulkfiles.php',notes=note,
            coverage=dict(rows=len(frame),symbols=len(frame.columns),observed_cells=int(frame.notna().sum().sum()),missing_cells=int(frame.isna().sum().sum()),
                first=manifest['actual_first_timestamp'],last=manifest['actual_last_timestamp']))
        (snapshot/'source_provenance.json').write_text(json.dumps(dict(raw=receipt,
            metadata_path=str((root/'series_metadata.json').resolve()),notes=note),indent=2))
        datasets.append(spec)
        print(group,spec['coverage'],flush=True)
    (ROOT/'output/expansion/grid_operations.json').write_text(json.dumps(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),incremental_spend_usd=0,datasets=datasets,
        raw_bytes=receipt['bytes'], raw_uncompressed_bytes=uncompressed,
        warnings=['A bulk download is a revised snapshot, not a historical publication-time database.',
            'Regional sums overlap individual balancing-authority series.',
            'Metered interchanges do not supply transmission capacity, losses, schedules, wheeling fees or tradable access.']),indent=2))


if __name__ == '__main__':
    main()
