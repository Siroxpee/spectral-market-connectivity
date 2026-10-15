"""Independent saved-output identities and original-input preservation checks."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/expansion'


def main():
    records=[]
    for path in sorted((OUT/'spectra').glob('*/summary.json')):
        summary=json.loads(path.read_text())
        features=pd.read_csv(path.parent/'features.csv')
        assert len(features)==summary['scheduled_windows']
        assert features.date.is_unique
        failures=[];values=0
        with gzip.open(path.parent/'spectra.jsonl.gz','rt',encoding='utf-8') as handle:
            for number,line in enumerate(handle):
                item=json.loads(line);row=features.iloc[number]
                n=len(item['retained_symbols'])
                assert row['date']==item['date'] and n==row['num_stocks']
                if row['status']=='insufficient_assets':
                    assert n<2 and not item['C'] and not item['L_positive']
                    continue
                c=np.array(item['C']);lap=np.array(item['L_positive'])
                assert len(c)==n and len(lap)==n
                assert np.all(np.diff(c)>=-1e-10) and np.all(np.diff(lap)>=-1e-10)
                assert abs(c.sum()-n)<1e-8*max(1,n)
                assert c.min()>-1e-9*max(1,c.max())
                assert lap.min()>-1e-9*max(1,lap.max())
                assert abs(lap.sum()-row['laplacian_trace'])<1e-8*max(1,lap.sum())
                rank=int((c>1e-10*max(1,c.max())).sum())
                assert rank==row['correlation_rank'] and rank<=row['rank_upper_bound']
                assert abs(lap[0])<1e-8*max(1,lap.max())
                values+=2*n
        assert number+1==len(features)
        battery=json.loads((path.parent/'mathematical_battery.json').read_text())
        for item in battery:
            if item['status']!='ok':continue
            n=len(item['retained_symbols'])
            assert len(item['geometry_H0']['finite_H0_deaths'])==n-1
            assert item['permutation_group_check']['max_L_spectrum_difference']<1e-7
            signed=np.array(item['operator_spectra']['L_signed'])
            assert signed.min()>-1e-9*max(1,signed.max())
            for graph in item['threshold_graphs']:
                assert graph['closed_walk_trace_A2']==2*graph['edges']
                assert graph['closed_walk_trace_A3']==6*graph['triangles']
                assert graph['GF2_graph_cycle_dimension']>=0
                homology=graph['subset_clique_homology']
                assert 0<=homology['betti1_clique']<=homology['betti1_graph']
        records.append(dict(dataset_id=summary['specification']['dataset_id'],windows=len(features),
            snapshot_key=summary['snapshot_key'],saved_eigenvalues=values,selected_dates=len(battery),status='pass'))
    sources=json.loads((ROOT/'source_files.json').read_text())
    for source in sources:
        assert hashlib.sha256((ROOT/'originals'/source['filename']).read_bytes()).hexdigest()==source['sha256']
    original=ROOT/'data/original_notebook/720726a5d713c24f9cae262ef7ede3d619cf0d329b4ed5382941de4cfb64811f/prices.csv.gz'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='8fc96b8b4b98d3e099a7506ab11b86b89aca042baa97ee2b37bfdf3cf38fde6b'
    reference=pd.read_csv(ROOT/'output/reproduction/features.csv')
    full=pd.read_csv(OUT/'spectra/original_stocks_daily/features.csv')
    assert pd.to_datetime(reference.date,utc=True).tolist()==pd.to_datetime(full.date,utc=True).tolist()
    assert np.array_equal(reference.num_stocks,full.num_stocks)
    delta=float((reference.lambda2-full.lambda2).abs().max())
    assert delta<1e-9
    result=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),datasets=records,
        total_saved_eigenvalues=sum(r['saved_eigenvalues'] for r in records),
        original_files_and_prices_unchanged=True,baseline_lambda2_max_difference=delta,
        identities='C trace/PSD/rank bound; L PSD/zero/trace; saved full spectrum sizes; H0 merge counts; permutation invariance; signed-L PSD; walks/triangles; GF2 homology dimensions')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='datasets'},indent=2),flush=True)


if __name__=='__main__':main()
