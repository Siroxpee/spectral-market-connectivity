"""Run archive-free, offline synthetic/integrity checks with this interpreter."""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).resolve().parent
COMMANDS=[['checks.py'],['check_cache.py'],['check_catalog.py'],['check_expanded_spectral.py'],
    ['crypto_mvp_research.py','--self-test'],['crypto_trajectory.py','--self-test'],
    ['market_connectivity.py','--self-test']]

if __name__=='__main__':
    for args in COMMANDS:
        print('CHECK '+' '.join(args),flush=True)
        subprocess.run([sys.executable,'-B',*args],cwd=ROOT,check=True)
    print('All 7 offline check commands passed. No market-data archive or network acquisition was used.')
