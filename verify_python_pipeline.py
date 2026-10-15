"""Verify the Python research / display-only JavaScript boundary.

Capture legacy source hashes before retiring implementations. Frozen scientific
outputs and reference fixtures are kept for reproducibility; source Git history
retains superseded implementation bytes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/python_migration"
LEGACY=("crypto_mvp/export_analytics.mjs","crypto_mvp/build_story_data.mjs","crypto_mvp/build_beginner.mjs","crypto_mvp/render_qa.mjs","crypto_mvp/check_crypto_math.mjs","crypto_mvp/check_dashboard.mjs","crypto_mvp/dist/crypto_math.js","crypto_mvp/dist/vendor/spectral_math.js","dashboard/dist/math.js","dashboard/check_math.mjs","dashboard/check_dashboard.mjs")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,obj):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
def capture():
    target=OUT/"legacy_source_hashes.json"
    if target.exists():return
    items=[{"path":name,"sha256":sha(ROOT/name),"bytes":(ROOT/name).stat().st_size} for name in LEGACY if (ROOT/name).exists()]
    write(target.name,{"captured_at_utc":datetime.now(timezone.utc).isoformat(),"reason":"Scientific logic is being migrated to Python by explicit user request; source version history and frozen numerical fixtures retain recovery/evidence.","files":items})
def check():
    remaining=[p for p in LEGACY if (ROOT/p).exists()]
    assert not remaining,"Retired scientific JavaScript files still exist: "+str(remaining)
    clients=[]
    forbidden=re.compile(r"eigenSymmetric|function\s+pearson|Math\.(?:sqrt|cos|sin|hypot|pow)|createElementNS|<svg[ >]|linearRegression|function\s+histogram")
    for folder in (ROOT/"crypto_mvp/dist",ROOT/"dashboard/dist"):
        for path in folder.rglob("*.js"):
            source=path.read_text(encoding="utf-8")
            assert not forbidden.search(source),"Analytical/chart geometry detected in browser client: "+str(path)
            clients.append({"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"scope":"presentation/selection of precomputed Python outputs"})
    for name in ("crypto_core.py","crypto_trajectory.py","crypto_trajectory_data.py","crypto_mvp/export_analytics.py","crypto_mvp/build_story_data.py","crypto_mvp/build_views.py","export_python_dashboard.py"):
        assert (ROOT/name).is_file(),"Required Python entry missing: "+name
    for name in ("crypto_mvp/dist/learning_manifest.json","crypto_mvp/dist/lab_manifest.json"):
        manifest=json.loads((ROOT/name).read_text(encoding="utf-8"));assert manifest.get("language")=="Python"
    assert not (ROOT/"crypto_mvp/dist/data").exists(),"Crypto raw data must stay local"
    result={"status":"pass","checked_at_utc":datetime.now(timezone.utc).isoformat(),"retired_scientific_javascript_files":len(LEGACY),"display_only_clients":clients,"python_sources":"Acquisition, transforms, graph construction, spectra/modes, features, models/evaluation, scientific figures and scientific checks.","limitations":"This boundary audit supplements numerical and browser tests; a lexical client check alone is not proof of mathematical correctness."}
    write("language_boundary.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--capture-legacy",action="store_true");args=parser.parse_args()
    if args.capture_legacy:capture()
    else:check()
