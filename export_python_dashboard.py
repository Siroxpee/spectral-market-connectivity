"""Offline Python-only general dashboard analytics and Matplotlib figures.

Input raw previews stay in dashboard/local_data. Output contains descriptive
derived metrics/matrices/PNG figures only. JavaScript selects already-computed
assets. Frozen sources, source calendars and missing values never change.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.linalg import eigh

from expanded_spectral import finite_json, measure_window
from spectral_core import clean_window, build_positive_graph
from export_dashboard_data import frame_from_export, write_json

ROOT = Path(__file__).resolve().parent
DASH = ROOT / 'dashboard'
INPUT = DASH / 'local_data'
OUTPUT = DASH / 'dist/analytics'
WINDOWS = (20, 60, 120)
POSITIONS = ('first', 'middle', 'latest')
INK, TEAL, ORANGE = '#17324D', '#007F86', '#D57B27'
GROUP_COLORS = dict(stocks=TEAL, etfs=ORANGE, forex='#426EA6', crypto='#7665A2',
                    commodities='#A75577', electricity='#62865F', grid_operations='#A34D49')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':9,
                     'axes.titlesize':11, 'figure.facecolor':'white', 'axes.spines.top':False,
                     'axes.spines.right':False, 'savefig.facecolor':'white'})


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_number(value, digits=4):
    return f'{float(value):.{digits}g}' if value is not None and np.isfinite(value) else 'undefined'


def result_for(window):
    """One common cleaned window; established Python baseline and full spectra."""
    cleaned = clean_window(window)
    metrics, spectra = measure_window(window)
    kept = cleaned.columns.tolist()
    missing = window.replace([np.inf,-np.inf],np.nan).isna().any()
    removed = [dict(id=str(column), reason=('missing_or_nonfinite' if missing[column]
                   else 'sample_std_at_or_below_1e-12')) for column in window.columns if column not in kept]
    if len(kept) < 2:
        return dict(metrics=metrics, spectra=spectra, retained=kept, removed=removed,
                    correlation=[], weights=[], laplacian=[], finite_window_cells=int(np.isfinite(window).sum().sum()))
    corr = cleaned.corr()
    weights, lap = build_positive_graph(corr)
    eigenvalues = np.asarray(spectra['L_positive'], dtype=float)
    if eigenvalues.min() < -1e-9 or not np.isclose(eigenvalues.sum(), np.trace(lap), atol=1e-9):
        raise AssertionError('Spectrum PSD/trace invariant failed')
    return dict(metrics=metrics, spectra=spectra, retained=kept, removed=removed,
                correlation=corr.to_numpy(), weights=weights.to_numpy(), laplacian=lap,
                finite_window_cells=int(np.isfinite(window).sum().sum()))


def save_figure(fig, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=115, bbox_inches='tight')
    plt.close(fig)
    return target.relative_to(DASH/'dist').as_posix()


def short_label(value, maximum=13):
    value = str(value)
    return value if len(value) <= maximum else value[:maximum-1]+'…'


def snapshot_figure(data, result, window_size, position, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.4,5.1), gridspec_kw={'width_ratios':[1,1.12,0.95]}, layout='constrained')
    n = len(result['retained'])
    date = result['metrics']['date']
    fig.suptitle(f'{data["title"]}\n{window_size} consecutive native changes · {position} endpoint: {date}', fontsize=11, color=INK)
    axis_records = []
    if n < 2:
        for ax, title in zip(axes, ['Network','Correlation','Laplacian spectrum']):
            ax.axis('off'); ax.set_title(title)
            ax.text(.5,.5,'Insufficient complete, nonconstant series\nNo graph or quantitative axes defined',
                    ha='center',va='center',transform=ax.transAxes,color=INK,fontsize=10)
        return save_figure(fig,output), [dict(panel='Unavailable graph', quantitative=False,
            explanation='No graph quantities defined; missing/constant series are not filled.')]
    by_id = {x['id']:x for x in data['sample']['columns']}
    columns = [by_id[x] for x in result['retained']]
    labels = [short_label(x['label']) for x in columns]
    numbered = data.get('group') == 'grid_operations'
    network_labels = [str(i+1) for i in range(n)] if numbered else labels
    if numbered:
        labels = [f'{i+1}: {short_label(x["label"],11)}' for i,x in enumerate(columns)]
    theta = np.linspace(0,2*np.pi,n,endpoint=False)+np.pi/2
    xy = np.column_stack([np.cos(theta),np.sin(theta)])
    weights = np.asarray(result['weights'])
    segments, colors, widths = [], [], []
    for i in range(n):
        for j in range(i+1,n):
            if weights[i,j] > 0:
                segments.append([xy[i],xy[j]])
                colors.append((0,.498,.525,.08+.6*weights[i,j]))
                widths.append(.3+2.3*weights[i,j])
    if segments:
        axes[0].add_collection(LineCollection(segments,colors=colors,linewidths=widths))
    axes[0].scatter(xy[:,0],xy[:,1],s=45,c=[GROUP_COLORS.get(x['group'],TEAL) for x in columns],edgecolors='white',zorder=3)
    for i,label in enumerate(network_labels):
        x,y=1.19*xy[i]
        axes[0].text(x,y,label,ha='center',va='center',fontsize=6.9,color=INK)
    axes[0].set(xlim=(-1.62,1.62),ylim=(-1.5,1.5),aspect='equal')
    axes[0].axis('off'); axes[0].set_title('Positive correlation network')
    note = 'Node numbers match heatmap order; full IDs in source table.' if numbered else 'Line width = positive correlation; all positive edges drawn.'
    axes[0].text(.5,-.03,'Layout has no measured axes.\n'+note,
                 ha='center',va='top',transform=axes[0].transAxes,fontsize=8)
    axis_records.append(dict(panel='Network', quantitative=False,
        explanation='Circular layout positions measure nothing; nodes are retained series, line width indicates positive correlation.' +
                    (' Node numbers match heatmap order; full source IDs remain in the variable table.' if numbered else '')))
    image=axes[1].imshow(result['correlation'],vmin=-1,vmax=1,cmap='BrBG',aspect='equal')
    axes[1].set_xticks(range(n),labels,rotation=65,ha='right',fontsize=6.7)
    axes[1].set_yticks(range(n),labels,fontsize=6.7)
    axes[1].set(xlabel='Retained series (categorical columns)',ylabel='Retained series (categorical rows)',title='Common-window Pearson correlation')
    fig.colorbar(image,ax=axes[1],fraction=.046,pad=.04,label='Pearson correlation (dimensionless)')
    axis_records.append(dict(panel='Correlation heatmap',quantitative=True,x='Retained series, categorical',
        y='Retained series, categorical',color='Pearson correlation, dimensionless [-1,1]',
        explanation='A cell compares the row and column series using exactly the same cleaned native changes.'))
    eig=np.asarray(result['spectra']['L_positive'])
    axes[2].bar(np.arange(1,n+1),eig,color=[ORANGE if k==1 else TEAL for k in range(n)])
    axes[2].set(xlabel='Eigenvalue index k (ascending order)',ylabel='Positive-Laplacian eigenvalue (dimensionless)',title='Spectrum of L = D − W')
    axes[2].set_xticks(np.arange(1,n+1));axes[2].tick_params(axis='x',labelsize=7);axes[2].grid(axis='y',alpha=.2)
    axis_records.append(dict(panel='Laplacian spectrum',quantitative=True,x='Ascending eigenvalue index k, count',
        y='Eigenvalue, dimensionless',explanation='Bars show all eigenvalues; orange marks the second-smallest, lambda2.'))
    return save_figure(fig,output),axis_records


def history_figure(data, rows, window_size, output):
    fig,axes=plt.subplots(3,1,figsize=(11,7.2),layout='constrained')
    dates=pd.to_datetime([r['date'] for r in rows],format='ISO8601')
    fields=[('lambda2','Positive λ₂ (dimensionless)'),('mean_raw_correlation','Mean pair correlation (dimensionless)'),('num_stocks','Retained series (count)')]
    records=[]
    for ax,(field,label) in zip(axes,fields):
        values=[r.get(field,np.nan) for r in rows]
        ax.plot(dates,values,color=TEAL,lw=1.3,marker='.',ms=2)
        ax.set(xlabel='Window end (source date/time or reference month)',ylabel=label)
        ax.grid(alpha=.2);ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        records.append(dict(panel=field,quantitative=True,x='Window end, original source labels',y=label,
            explanation='Horizontal positions preserve elapsed source time; gaps represent undefined measurements. Windows overlap; association is descriptive.'))
    axes[0].set_title(f'{data["title"]} · {window_size} native changes per window\nPredeclared regular preview endpoints; unavailable windows remain gaps',color=INK)
    return save_figure(fig,output),records


def parity_checks():
    reports=[]
    reference=read_json(INPUT/'reference_cases.json')
    for expected in reference['cases']:
        panel=read_json(INPUT/expected['path'])
        frame=frame_from_export(panel).tail(expected['feature_rows'])
        result=result_for(frame);metrics=result['metrics']
        assert result['retained']==expected['expected_spectra']['retained_symbols']
        errors={}
        for field in ['lambda2','normalized_lambda2','mean_raw_correlation','mean_positive_weight',
                      'lambda2_scale_ratio','num_stocks','num_components','principal_component_share','effective_rank']:
            a,b=metrics.get(field),expected['expected_metrics'].get(field)
            if a is None and b is None:continue
            if b is None:
                assert not np.isfinite(a);continue
            np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9,equal_nan=True)
            errors[field]=float(abs(a-b))
        for field in ['C','L_positive']:
            np.testing.assert_allclose(result['spectra'][field],expected['expected_spectra'][field],rtol=1e-9,atol=1e-9)
        reports.append(dict(id=expected['id'],maximum_metric_difference=max(errors.values(),default=0),status='pass'))
    fixture=read_json(INPUT/'fixtures/original_stock_window.json')
    prices=pd.DataFrame(fixture['prices'],index=pd.to_datetime(fixture['dates'],format='ISO8601'),columns=fixture['labels'])
    from expanded_spectral import transform_panel
    result=result_for(transform_panel(prices,dict(transform=fixture['transform'],bar_seconds=fixture['barSeconds']))[0])
    np.testing.assert_allclose(result['correlation'],fixture['expected']['correlation'],atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(result['weights'],fixture['expected']['weights'],atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(result['spectra']['L_positive'],fixture['expected']['eigenvalues'],atol=1e-9,rtol=1e-9)
    return dict(status='pass',real_reference_windows=reports,original_stock_fixture='pass',
                source='Preserved Python/SciPy fixtures, plus seven legacy JS parity groups executed before removal.',
                maximum_metric_difference=max(x['maximum_metric_difference'] for x in reports))


def build_one(entry):
    data=read_json(INPUT/entry['path'])
    features=frame_from_export(data)
    # Exact input source labels and native transforms are retained. Prepared
    # cross-market comparisons must not be transformed a second time.
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError('Unsorted/nonunique feature labels')
    panel=dict(id=data['id'],title=data['title'],group=data['group'],interval=data['interval'],
        transform=data['transform'],prepared=data['prepared'],columns=data['sample']['columns'],
        coverage=data['coverage'],preview_rows=len(features),preview_variables=len(features.columns),
        timestamp_policy=data['timestamp_policy'],source_notes=data['source_notes'],source_links=data['source_links'],
        provenance=data['provenance'],license=data.get('license'),
        input_preview_sha256=file_hash(INPUT/entry['path']),windows=[],
        sample_policy='Existing bounded subset selected by coverage/familiar defaults; all exported columns used, no outcome/metric selection.',
        observation_calendar='Native source rows only; no empty-row omission or filling.',
        unavailable_policy='Retain scheduled endpoint and explicit insufficient status; never seek a more favorable nearby window.')
    records=[]
    for window_size in WINDOWS:
        if len(features)<window_size:
            raise ValueError('Preview too short for declared window choice')
        last=len(features)-1;first=window_size-1;middle=(first+last)//2
        endpoint_indices=dict(zip(POSITIONS,[first,middle,last]))
        scheduled=sorted(set(np.linspace(first,last,min(80,last-first+1),dtype=int).tolist()+list(endpoint_indices.values())))
        calculated={end:result_for(features.iloc[end-window_size+1:end+1]) for end in scheduled}
        rows=[dict(calculated[end]['metrics'],endpoint_index=end) for end in scheduled]
        history_path,history_axes=history_figure(data,rows,window_size,OUTPUT/data['id']/f'w{window_size}'/'history.png')
        cases=[]
        for position,end in endpoint_indices.items():
            result=calculated[end]
            figure,axes=snapshot_figure(data,result,window_size,position,OUTPUT/data['id']/f'w{window_size}'/f'{position}.png')
            metrics=result['metrics'];labels={c['id']:c['label'] for c in panel['columns']}
            cross=[]
            if data['prepared'] and len(result['retained'])>=2:
                groups={c['id']:c['group'] for c in panel['columns']}
                corr=np.asarray(result['correlation'])
                cross=[corr[i,j] for i in range(len(result['retained'])) for j in range(i+1,len(result['retained']))
                       if groups[result['retained'][i]]!=groups[result['retained'][j]]]
            case=dict(id=position,date=features.index[end].isoformat(),
                start=features.index[end-window_size+1].isoformat(),metrics=metrics,figure=figure,axes=axes,
                retained_columns=[dict(id=k,label=labels[k]) for k in result['retained']],removed=result['removed'],
                finite_window_cells=result['finite_window_cells'],requested_window_cells=window_size*features.shape[1],
                matrices=dict(correlation=result['correlation'],weights=result['weights'],laplacian=result['laplacian']),
                eigenvalues=result['spectra']['L_positive'],
                cross_group=dict(pairs=len(cross),mean_correlation=float(np.mean(cross)) if cross else None),
                reading='Correlation compares variation of changes, not price direction or a causal effect. Lambda2 depends on node count and positive weight scale.')
            cases.append(case)
        panel['windows'].append(dict(size=window_size,history_figure=history_path,history_axes=history_axes,
            history=rows,scheduled_endpoints=len(rows),usable_endpoints=sum(r.get('num_stocks',0)>=2 for r in rows),cases=cases))
        records.append(dict(id=data['id'],window=window_size,cases=len(cases),scheduled=len(rows),
            usable=sum(r.get('num_stocks',0)>=2 for r in rows)))
    path=OUTPUT/'panels'/f'{data["id"]}.json'
    write_json(path,panel)
    return dict(id=data['id'],title=data['title'],group=data['group'],interval=data['interval'],
        path=path.relative_to(DASH/'dist').as_posix(),source_price_hashes=data['provenance'],
        preview_rows=len(features),preview_variables=len(features.columns)),records


def validate_artifacts(expected_inputs=None):
    catalog=read_json(OUTPUT/'catalog.json')
    assert len(catalog['datasets'])==36 and len(catalog['comparisons'])==3
    if expected_inputs is None:
        expected_inputs=read_json(DASH/'validation/python_dashboard.json')['input_files_unchanged']
    current_inputs={p.relative_to(INPUT).as_posix():file_hash(p) for p in INPUT.rglob('*') if p.is_file()}
    assert current_inputs==expected_inputs, 'Local raw preview/fixture input changed since generation'
    cases,figures=0,[]
    for entry in catalog['datasets']+catalog['comparisons']:
        panel=read_json(DASH/'dist'/entry['path'])
        assert [w['size'] for w in panel['windows']]==list(WINDOWS)
        assert not any(key in panel for key in ('sample','prices','returns','preceding_row'))
        for window in panel['windows']:
            figures.append(window['history_figure'])
            assert [x['id'] for x in window['cases']]==list(POSITIONS)
            for case in window['cases']:
                cases+=1;figures.append(case['figure'])
                assert len(case['axes'])>=1 and case['date']>=case['start']
                assert all(a.get('explanation') for a in case['axes'])
    for relative in figures:
        assert (DASH/'dist'/relative).is_file(),relative
    assert not (DASH/'dist/math.js').exists() and not (DASH/'dist/data').exists()
    source=(DASH/'dist/app.js').read_text(encoding='utf-8')
    assert not any(token in source for token in ('Math.','<svg','eigenSymmetric','SpectralMath','pearson(', 'covariance'))
    return dict(status='pass',panels=36,comparison_pairs=3,cases=cases,figure_files=len(set(figures)),
        all_figure_paths_exist=True,all_axis_guides_present=True,raw_panels_outside_dist=True,
        scientific_javascript_not_loaded=True,source_values_unchanged=True,
        capabilities='Precomputed positive Pearson combinatorial-Laplacian cases only: windows20/60/120, first/middle/latest; no arbitrary live math, variable subsets or calendar/operator alternatives.')


def build():
    before={p.relative_to(INPUT).as_posix():file_hash(p) for p in INPUT.rglob('*') if p.is_file()}
    parity=parity_checks()
    old=read_json(INPUT/'catalog.json')
    datasets,comparisons,statistics=[],[],[]
    for number,entry in enumerate(old['datasets']+old['comparisons'],1):
        exported,records=build_one(entry);statistics+=records
        (comparisons if entry in old['comparisons'] else datasets).append(exported)
        print(json.dumps(dict(completed=number,total=39,id=entry['id'])),flush=True)
    catalog=dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),
        groups=old['groups'],datasets=datasets,comparisons=comparisons,windows=list(WINDOWS),positions=list(POSITIONS),
        source_as_of=old['as_of'],calculation='Python: expanded_spectral.transform_panel/measure_window and spectral_core.clean_window/build_positive_graph',
        plot_generation='Python Matplotlib; browser only selects saved images and metadata.',raw_data_hosted=False,
        policy='Native rows; no fill; original per-window common-sample cleaning. Cross-market features prepared natively before descriptive label alignment.',
        capability_change='Fixed bounded cases replace arbitrary browser math, variable selection, weighting/correlation/normalization toggles and calendar alternatives.',
        science_claim='Descriptive saved previews; no forecast, causal, synchronized-availability or executable-profit claim.',spend_usd=0)
    write_json(OUTPUT/'catalog.json',catalog)
    after={p.relative_to(INPUT).as_posix():file_hash(p) for p in INPUT.rglob('*') if p.is_file()}
    assert before==after
    receipt=dict(validate_artifacts(before),checked_at_utc=datetime.now(timezone.utc).isoformat(),
        input_files_unchanged=before,parity=parity,statistics=statistics,
        figure_bytes=sum(p.stat().st_size for p in OUTPUT.rglob('*.png')))
    write_json(DASH/'validation/python_dashboard.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','panels','comparison_pairs','cases','figure_files','figure_bytes']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Validate existing generated assets and Python parity only')
    args=parser.parse_args()
    if args.check:
        print(json.dumps(dict(validate_artifacts(),parity=parity_checks()),indent=2))
    else:
        build()
