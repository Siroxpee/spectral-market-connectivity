"""Download public EIA bulk grid operating archive without an API account."""
import hashlib
import json
from pathlib import Path
import requests
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parent/'data/expanded/grid_operations/raw_evidence'
URL = 'https://www.eia.gov/opendata/bulk/EBA.zip'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt = ROOT/'download.json'
    if receipt.exists():
        print('Archive already frozen:', receipt.read_text(), flush=True)
        return
    temporary = ROOT/'EBA.zip.partial'
    digest = hashlib.sha256()
    size = 0
    mark = 0
    with requests.get(URL, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        headers = dict(response.headers)
        with temporary.open('wb') as handle:
            for chunk in response.iter_content(1024*1024):
                if not chunk: continue
                size += len(chunk)
                if size > 1_500_000_000:
                    raise ValueError('Public bulk archive exceeded 1.5 GB acquisition bound')
                digest.update(chunk)
                handle.write(chunk)
                if size - mark > 50_000_000:
                    mark = size
                    print(f'EIA grid archive: {size/1e6:.1f} MB received', flush=True)
    final = ROOT/(digest.hexdigest()+'.zip')
    temporary.rename(final)
    with zipfile.ZipFile(final) as archive:
        print([(f.filename, f.file_size, f.compress_size) for f in archive.infolist()], flush=True)
        with archive.open(archive.namelist()[0]) as handle:
            sample = [json.loads(handle.readline()) for _ in range(3)]
        print(json.dumps([{k:v for k,v in item.items() if k != 'data'} for item in sample],indent=2), flush=True)
    receipt.write_text(json.dumps(dict(url=URL, path=str(final), sha256=digest.hexdigest(), bytes=size,
        response_headers=headers, downloaded_at_utc=datetime.now(timezone.utc).isoformat()), indent=2))


if __name__ == '__main__':
    main()
