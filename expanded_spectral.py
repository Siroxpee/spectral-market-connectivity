"""Offline cross-market spectral screening on explicit frozen dataset panels.

Financial closes use arithmetic returns; signed physical prices use first
differences. The original sixty-observation positive graph stays the baseline.
Full positive-Laplacian and correlation spectra are retained for each scheduled
window, with a broader mathematical battery at declared selected dates.
"""

from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

from data_cache import cached_closes
from spectral_core import clean_window, build_positive_graph

ROOT = Path(__file__).resolve().parent


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def transform_panel(prices, specification):
    """Keep missing observations and prevent interval-gap/overnight returns."""
    mode = specification["transform"]
    if mode == "arithmetic_return":
        observed = prices.to_numpy(dtype=float, na_value=np.nan)
        if np.any(observed[np.isfinite(observed)] <= 0):
            raise ValueError("Arithmetic-return panels require strictly positive observations.")
        changes = prices.pct_change(fill_method=None)
    elif mode == "first_difference":
        changes = prices.diff()
    else:
        raise ValueError(f"Unknown transformation: {mode}")
    seconds = specification.get("bar_seconds")
    gap_rows = 0
    if seconds:
        elapsed = prices.index.to_series().diff().dt.total_seconds()
        gap = elapsed.notna() & elapsed.ne(seconds)
        changes.loc[gap] = np.nan
        gap_rows = int(gap.sum())
    return changes.replace([np.inf, -np.inf], np.nan).iloc[1:], gap_rows


def measure_window(window):
    """Baseline metrics and two full spectra from one common cleaned sample."""
    cleaned = clean_window(window)
    n = len(cleaned.columns)
    row = {"date": window.index[-1].isoformat(), "num_stocks": n,
           "num_input_stocks": window.shape[1], "num_removed_stocks": window.shape[1]-n}
    if n < 2:
        row.update(status="insufficient_assets")
        return row, {"retained_symbols": cleaned.columns.tolist(), "C": [], "L_positive": []}
    corr = cleaned.corr()
    weights, lap = build_positive_graph(corr)
    c = corr.to_numpy()
    w = weights.to_numpy()
    degree = w.sum(axis=1)
    eig, vectors = eigh(lap, check_finite=True)
    ceig = eigh(c, eigvals_only=True, check_finite=True)
    components = int(connected_components(w, directed=False, return_labels=False))
    lam = max(0.0, float(eig[1])) if components == 1 else 0.0
    tri = np.triu_indices(n, 1)
    mean_w = float(w[tri].mean())
    norm2 = np.nan
    if components == 1 and np.all(degree > 0):
        inverse = 1 / np.sqrt(degree)
        normal = np.eye(n) - inverse[:,None]*w*inverse[None,:]
        norm2 = max(0., float(eigh(normal, subset_by_index=[0,1], eigvals_only=True)[1]))
    tolerance = 1e-10 * max(1., 2.*degree.max())
    gap = eig[2]-eig[1] if n >= 3 else np.nan
    separated = components == 1 and lam > tolerance and (n < 3 or gap > tolerance)
    positive = np.maximum(ceig, 0.)
    share = positive / positive.sum()
    entropy = -np.sum(share[share > 0]*np.log(share[share > 0]))
    row.update(status="ok" if components == 1 else "disconnected",
        num_components=components, lambda2=lam, normalized_lambda2=norm2,
        mean_raw_correlation=float(c[tri].mean()), mean_positive_weight=mean_w,
        lambda2_scale_ratio=lam/(n*mean_w) if mean_w > 0 else np.nan,
        positive_edge_fraction=float((w[tri] > 0).mean()),
        mean_degree=float(degree.mean()), min_degree=float(degree.min()),
        min_degree_ticker=str(weights.index[np.argmin(degree)]),
        lambda3_gap=float(gap), fiedler_separated=bool(separated),
        fiedler_ipr=float(np.sum(vectors[:,1]**4)) if separated else np.nan,
        correlation_rank=int((ceig > 1e-10*max(1.,ceig[-1])).sum()),
        rank_upper_bound=min(n,len(cleaned)-1), correlation_trace=float(ceig.sum()),
        principal_component_share=float(share[-1]), effective_rank=float(np.exp(entropy)),
        participation_rank=float(1/np.sum(share**2)),
        laplacian_trace=float(eig.sum()),
        log_weighted_spanning_tree=float(np.log(eig[1:]).sum()-np.log(n)) if components==1 and np.all(eig[1:]>0) else np.nan,
        kirchhoff_index=float(n*np.sum(1/eig[1:])) if components==1 and np.all(eig[1:]>0) else np.nan)
    return row, {"retained_symbols": cleaned.columns.tolist(), "C": ceig, "L_positive": eig}


def gf2_rank(columns):
    """Rank over the two-element field using integer bit vectors and XOR."""
    pivots = {}
    for column in columns:
        value = int(column)
        while value:
            pivot = value.bit_length()-1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def clique_homology(adjacency):
    """H0/H1 over GF(2) of a clique complex's two-skeleton, not its graph alone."""
    n = len(adjacency)
    edges = [(i,j) for i in range(n) for j in range(i+1,n) if adjacency[i,j]]
    lookup = {edge:k for k,edge in enumerate(edges)}
    boundary1 = [(1 << i) | (1 << j) for i,j in edges]
    boundary2 = []
    for i,j in edges:
        for k in range(j+1,n):
            if adjacency[i,k] and adjacency[j,k]:
                boundary2.append((1<<lookup[(i,j)]) | (1<<lookup[(i,k)]) | (1<<lookup[(j,k)]))
    r1, r2 = gf2_rank(boundary1), gf2_rank(boundary2)
    return {"vertices":n,"edges":len(edges),"triangles":len(boundary2),
            "boundary1_rank_GF2":r1,"boundary2_rank_GF2":r2,
            "betti0":n-r1,"betti1_clique":len(edges)-r1-r2,
            "betti1_graph":len(edges)-r1}


def h0_filtration(correlation):
    """Kruskal H0 deaths include zero-distance merges of identical series."""
    n = len(correlation)
    distance = np.sqrt(np.maximum(0.,2*(1-correlation)))
    np.fill_diagonal(distance, 0)
    pairs = [(distance[i,j],i,j) for i in range(n) for j in range(i+1,n)]
    pairs.sort()
    parent = list(range(n))
    size = [1]*n
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    deaths, tree = [], []
    for d,i,j in pairs:
        a,b = find(i),find(j)
        if a == b:
            continue
        if size[a] < size[b]: a,b=b,a
        parent[b] = a
        size[a] += size[b]
        deaths.append(float(d))
        tree.append([int(i),int(j),float(d)])
        if len(deaths) == n-1: break
    return {"distance_definition":"sqrt(2*(1-Cij)) on same cleaned Pearson sample",
            "finite_H0_deaths":deaths,"essential_H0_classes":1,
            "MST_edges":tree,"MST_total_length":float(sum(deaths))}


def mathematical_battery(window):
    cleaned = clean_window(window)
    n = cleaned.shape[1]
    if n < 2:
        return {"status":"insufficient_assets","retained_symbols":cleaned.columns.tolist()}
    correlation = cleaned.corr()
    c = correlation.to_numpy()
    wframe, lap = build_positive_graph(correlation)
    w = wframe.to_numpy()
    signed = c.copy()
    np.fill_diagonal(signed, 0)
    absolute = np.abs(signed)
    degree = w.sum(axis=1)
    spectra = {"C":eigh(c,eigvals_only=True),"W_positive":eigh(w,eigvals_only=True),
               "L_positive":eigh(lap,eigvals_only=True),
               "L_absolute":eigh(np.diag(absolute.sum(axis=1))-absolute,eigvals_only=True),
               "L_signed":eigh(np.diag(absolute.sum(axis=1))-signed,eigvals_only=True)}
    if int(connected_components(w,directed=False,return_labels=False))==1 and np.all(degree>0):
        inv=1/np.sqrt(degree)
        spectra["L_normalized"]=eigh(np.eye(n)-inv[:,None]*w*inv[None,:],eigvals_only=True)
    else:
        spectra["L_normalized"]=None
    thresholds=[]
    # Higher-dimensional topology is explicitly a deterministic <=30-node subset.
    take=np.argsort(np.array(cleaned.columns,dtype=str))[:30]
    for threshold in (.2,.4,.6,.8):
        a=(c >= threshold).astype(np.int64)
        np.fill_diagonal(a,0)
        components=int(connected_components(a,directed=False,return_labels=False))
        deg=a.sum(axis=1)
        edges=int(a.sum()//2)
        triangles=int(np.trace(a@a@a)//6)
        wedges=int(np.sum(deg*(deg-1))//2)
        thresholds.append({"positive_correlation_threshold":threshold,"edges":edges,
            "density":edges/(n*(n-1)/2),"components":components,
            "triangles":triangles,"wedges":wedges,
            "global_transitivity":3*triangles/wedges if wedges else None,
            "GF2_graph_cycle_dimension":edges-n+components,
            "closed_walk_trace_A2":int(np.trace(a@a)),"closed_walk_trace_A3":int(np.trace(a@a@a)),
            "subset_clique_homology":clique_homology(a[np.ix_(take,take)])})
    permutation=np.random.default_rng(20261009).permutation(n)
    permuted_lap=lap[np.ix_(permutation,permutation)]
    perm_eigen=eigh(permuted_lap,eigvals_only=True)
    return finite_json({"status":"ok","retained_symbols":cleaned.columns.tolist(),
        "operator_spectra":spectra,"threshold_graphs":thresholds,
        "clique_subset_symbols":cleaned.columns[take].tolist(),
        "clique_subset_policy":"first <=30 retained symbols alphabetically; exploratory, not representative",
        "geometry_H0":h0_filtration(c),
        "permutation_group_check":{"group":"S_n acts by simultaneous row/column relabeling",
             "seed":20261009,"max_L_spectrum_difference":float(np.max(np.abs(spectra['L_positive']-perm_eigen)))},
        "matrix_algebra":"A^k counts closed walks via trace; GF(2) boundary ranks compute cycle/clique homology. These algebraic identities are not independent financial signals.",
        "D_W_commutator_frobenius":float(np.linalg.norm((degree[:,None]-degree[None,:])*w)),
        "C_PSD_min_eigenvalue":float(spectra['C'][0])})


def run_dataset(specification, output_root, *, window_size=60, max_windows=3000, selected_count=12):
    identifier = specification["dataset_id"]
    if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in identifier):
        raise ValueError("dataset_id must be a simple lowercase path component.")
    snapshot=Path(specification['snapshot_path']).resolve()
    manifest=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    prices,manifest,_=cached_closes(manifest['request'],fetch=None,cache_dir=snapshot.parent)
    observed_rows=prices.notna().any(axis=1)
    observed_bounds=prices.index[observed_rows]
    changes,gap_rows=transform_panel(prices,specification)
    output=Path(output_root)/identifier
    output.mkdir(parents=True,exist_ok=True)
    previous=output/'summary.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))['input_sha256']!=manifest['sha256']:
        raise ValueError('This analysis ID already describes a different frozen input; use a new dataset_id to preserve it.')
    candidate=max(0,len(changes)-window_size+1)
    step=max(1,math.ceil(candidate/max_windows))
    ends=list(range(window_size,len(changes)+1,step))
    if ends and ends[-1]!=len(changes): ends.append(len(changes))
    selected=set(ends[int(i)] for i in np.linspace(0,len(ends)-1,min(selected_count,len(ends)),dtype=int)) if ends else set()
    rows,battery=[],[]
    with gzip.open(output/'spectra.jsonl.gz','wt',encoding='utf-8') as handle:
        for number,end in enumerate(ends,1):
            window=changes.iloc[end-window_size:end]
            row,spectra=measure_window(window)
            rows.append(row)
            handle.write(json.dumps(finite_json(dict(date=row['date'],**spectra)),allow_nan=False)+'\n')
            if end in selected:
                battery.append(dict(date=row['date'],**mathematical_battery(window)))
            if number%250==0:
                print(f"{identifier}: {number}/{len(ends)} windows",flush=True)
    features=pd.DataFrame(rows)
    features.to_csv(output/'features.csv',index=False,float_format='%.17g')
    (output/'mathematical_battery.json').write_text(json.dumps(finite_json(battery),indent=2,allow_nan=False),encoding='utf-8')
    summary=dict(specification=specification,snapshot_key=snapshot.name,input_sha256=manifest['sha256'],
        price_rows=len(prices),symbols=len(prices.columns),observed_values=int(prices.notna().sum().sum()),
        missing_values=int(prices.isna().sum().sum()),nonpositive_values=int((prices<=0).sum().sum()),
        first_label=prices.index[0].isoformat(),last_label=prices.index[-1].isoformat(),
        first_observed_label=observed_bounds[0].isoformat(),last_observed_label=observed_bounds[-1].isoformat(),
        transform=specification['transform'],gap_spanning_rows_masked=gap_rows,
        window_size=window_size,step_size=step,scheduled_windows=len(ends),selected_battery_windows=len(battery),
        schedule_note="regular step plus final complete window; selected battery dates uniformly spaced in this schedule",
        full_spectra_every_scheduled_window=['C','L_positive'],
        alternative_operator_spectra_at_selected_windows=['W_positive','L_absolute','L_signed','L_normalized'],
        status_counts=features.status.value_counts().to_dict() if len(features) else {},
        created_at_utc=datetime.now(timezone.utc).isoformat(),output_directory=str(output),
        claim_scope="exploratory descriptive screen; no scored holdout, no p-values, no executable arbitrage")
    if len(features) and 'lambda2' in features:
        valid=features.dropna(subset=['lambda2'])
        for column in ('lambda2','lambda2_scale_ratio','fiedler_ipr','principal_component_share','effective_rank'):
            if column in valid:
                summary[column]=dict(min=float(valid[column].min()),median=float(valid[column].median()),max=float(valid[column].max()))
        summary['lambda2_min_degree_correlation']=float(valid.lambda2.corr(valid.min_degree))
        summary['lambda2_raw_correlation_association']=float(valid.lambda2.corr(valid.mean_raw_correlation))
    summary=finite_json(summary)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(f"Completed {identifier}: {len(ends)} windows",flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'output/expansion/spectra')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-windows',type=int,default=3000)
    parser.add_argument('--selected-count',type=int,default=12)
    parser.add_argument('--batch-name',default='batch_summary',help='Unique receipt name when independent batches run concurrently')
    args=parser.parse_args()
    if args.max_windows < 1 or args.selected_count < 1 or args.workers < 1:
        parser.error('workers, max-windows and selected-count must be positive')
    specifications=json.loads(args.registry.read_text(encoding='utf-8'))['datasets']
    summaries=[]
    failures=[]
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(run_dataset,spec,args.output,max_windows=args.max_windows,selected_count=args.selected_count):spec['dataset_id'] for spec in specifications}
        for task in as_completed(tasks):
            try: summaries.append(task.result())
            except Exception as error:
                failures.append(dict(dataset_id=tasks[task],error_type=type(error).__name__,error=str(error)))
                print(f"FAILED {tasks[task]}: {error}",flush=True)
            (args.output/(args.batch_name+'.json')).write_text(json.dumps({'datasets':summaries,'failures':failures},indent=2,allow_nan=False),encoding='utf-8')
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
