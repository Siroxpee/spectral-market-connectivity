"""Register frozen acquisitions serially and schedule only new spectral panels."""
import json
from pathlib import Path
from archive_catalog import register_snapshot

ROOT=Path(__file__).resolve().parent
SOURCES=('initial_registry.json','financial.json','financial_worldbank.json',
    'electricity_results.json','retail_electricity.json','grid_operations.json','crypto_results.json')


def synchronize():
    output=ROOT/'output/expansion'
    done={}
    for path in (output/'spectra').glob('*/summary.json'):
        summary=json.loads(path.read_text())
        done[summary['snapshot_key']]=summary
    specs={}
    documents={}
    for name in SOURCES:
        path=output/name
        if not path.exists():continue
        document=json.loads(path.read_text())
        documents[name]=document
        for spec in document.get('datasets',[]):
            if name=='crypto_results.json' and not (document.get('finalized') or spec.get('finalized')):
                continue
            spec=dict(spec)
            spec['dataset_id']=spec['dataset_id'].lower()
            key=Path(spec['snapshot_path']).name
            specs[key]=spec
    pending=[]
    inventory=[]
    for key,spec in specs.items():
        metadata=register_snapshot(spec['snapshot_path'],ROOT/'data/archive_catalog.sqlite')
        summary=done.get(key)
        if summary is not None and summary['input_sha256'] != metadata['sha256']:
            raise ValueError('Frozen input changed for analyzed request: '+key)
        if summary is None:pending.append(spec)
        inventory.append(dict(specification=spec,snapshot=metadata,analysis=summary))
    (output/'pending_registry.json').write_text(json.dumps(dict(datasets=pending),indent=2))
    (output/'all_registry.json').write_text(json.dumps(dict(datasets=list(specs.values())),indent=2))
    archives={}
    for document in documents.values():
        for entry in document.get('superseded_datasets',[])+document.get('superseded_snapshots',[])+document.get('superseded',[]):
            if not isinstance(entry,dict) or 'snapshot_path' not in entry:continue
            path=Path(entry['snapshot_path'])
            # A renamed descriptor for the same frozen data is an alias, not a
            # separate obsolete panel and never demotes the active request.
            if path.name in specs or not (path/'manifest.json').exists():continue
            archives[path.name]=dict(entry,analysis=done.get(path.name))
    (output/'inventory.json').write_text(json.dumps(dict(datasets=inventory,preserved_superseded_snapshots=list(archives.values()),source_documents=list(documents),
        incremental_spend_usd=0,counting_rule='One universe + interval + intended span constitutes a dataset. Raw files/windows/features are not datasets. Overlapping/derived panels are not independent raw observations.'),indent=2))
    print('Frozen panels:',len(specs),'completed:',len(specs)-len(pending),'pending:',len(pending),flush=True)
    for spec in pending:print('Pending:',spec['dataset_id'],flush=True)
    return inventory


if __name__ == '__main__':synchronize()
