"""Build one offline inventory, spectral atlas and acquisition-gap report."""
import base64
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sync_expansion import synchronize

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/expansion'
FIG=OUT/'figures'
AXIS_AUDIT=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})


def image(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def save(fig, name):
    path=FIG/(name+'.png')
    axes=[]
    for number,ax in enumerate(fig.axes,1):
        if not ax.axison:continue
        if not ax.get_xlabel() or not ax.get_ylabel():
            raise ValueError(f'Quantitative axes need explicit x/y labels: {name}, panel {number}')
        axes.append(dict(figure=name,panel=number,x_label=ax.get_xlabel(),y_label=ax.get_ylabel(),
            x_scale=ax.get_xscale(),y_scale=ax.get_yscale(),quantitative=True))
    fig.savefig(path,dpi=145,bbox_inches='tight')
    plt.close(fig)
    (FIG/(name+'__axes.json')).write_text(json.dumps(axes,indent=2),encoding='utf-8')
    AXIS_AUDIT.extend(axes)
    return path


def nice(identifier):
    return identifier.replace('_',' ')


def prose(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return ' '.join(str(v) for v in value)
    return json.dumps(value,ensure_ascii=False)


def market_group(spec):
    identifier=spec['dataset_id']
    if identifier.startswith('eia_grid_'):return 'Grid operations'
    if 'stocks' in identifier:return 'Stocks'
    if 'etf' in identifier:return 'ETFs'
    if 'crypto' in identifier:return 'Crypto'
    if 'fx' in identifier:return 'Forex'
    if any(token in identifier for token in ('caiso','nyiso','ercot','isone','retail','wholesale_electric')):return 'Electricity'
    return 'Commodities'


def source_time_axis(spec,summary):
    """Window endpoints are source labels, not invented publication times."""
    if pd.Timestamp(summary['first_label']).tzinfo is not None:
        label='Window-end source timestamp (UTC)'
        meaning='UTC source label of the latest observation in each trailing 60-observation window; this is not a reconstructed publication timestamp.'
        if spec['family']=='grid_operations':
            meaning+=' EIA labels mean HOUR ENDING despite the legacy native column name; physical interval start is this label minus one hour.'
        elif spec['family']=='crypto':
            meaning+=' Crypto labels mark bar ends; publication latency remains unmeasured.'
        elif spec['family']=='electricity':
            meaning+=' These source labels mark electricity operating-interval starts.'
        else:
            meaning+=' Provider bucket start/end conventions must be checked separately.'
        return dict(label=label,meaning=meaning,units='UTC source timestamps')
    if spec['interval']=='1mo':
        return dict(label='Window-end reference month',units='reference months',
            meaning='Reference month of the latest observation in the window. First-of-month dates are period labels, not midnight quotes or release dates.')
    return dict(label='Window-end source date',units='source calendar dates',
        meaning='Calendar/reference date of the latest observation in the window. Naive daily labels do not establish a midnight UTC quote or original information availability.')


def figure_axes(spec,summary):
    time=source_time_axis(spec,summary)
    def time_panel(panel,y_label,y_meaning,y_units):
        return dict(panel=panel,x_label=time['label'],x_meaning=time['meaning'],x_units=time['units'],
            y_label=y_label,y_meaning=y_meaning,y_units=y_units)
    rolling=[
        time_panel('Upper left','Raw λ₂ (dimensionless)',
            'Second-smallest eigenvalue of L=D−W. It measures connectivity of the positive-correlation graph and depends on its weights and retained variable count; it is not a price or return.', 'dimensionless correlation-weight eigenvalue'),
        time_panel('Upper right','q (dimensionless ratio)',
            'Raw λ₂ divided by n times the mean off-diagonal clipped weight. This removes a uniform edge-weight scale; q is undefined when the mean weight is zero.', 'dimensionless ratio'),
        time_panel('Lower left','Largest PCA share (fraction)',
            'Largest correlation eigenvalue divided by trace(C), the fraction of standardized variation in its leading component. A value of 0.25 means 25%.', 'fraction of total standardized variation'),
        time_panel('Lower right','Retained variables n (count)',
            'Number of variables with complete finite changes and sample standard deviation above 10⁻¹² in that same window. This count can change between windows.', 'variable count')]
    spectra=[dict(panel='Left',x_label='Descending eigenvalue rank fraction (0–1)',
        x_meaning='Position (k−1)/(n−1) after sorting correlation eigenvalues largest first: 0 is the largest and 1 the smallest. Fractional rank permits different retained sizes; it is not time or a market distance.',x_units='dimensionless rank fraction',
        y_label='Correlation eigenvalue share (fraction)',y_meaning='One correlation eigenvalue divided by trace(C). The shares sum to approximately one; tiny negative numerical eigenvalues are clipped only for this display.',y_units='dimensionless fraction'),
        dict(panel='Right',x_label='Ascending eigenvalue rank fraction (0–1)',
        x_meaning='Position (k−1)/(n−1) after sorting positive-Laplacian eigenvalues smallest first: 0 is the smallest and 1 the largest. Each colored curve is one selected window.',x_units='dimensionless rank fraction',
        y_label='L eigenvalue / mean degree (ratio)',y_meaning='One eigenvalue of L divided by trace(L)/n, the mean weighted degree of that graph. Zero-mean-degree graphs have undefined scaling and their curves are omitted.',y_units='dimensionless ratio')]
    topology=[
        time_panel('Left','Threshold edge density (fraction)',
            'Fraction of all n(n−1)/2 unordered retained-variable pairs with Cᵢⱼ at least the legend threshold (0.2, 0.4, 0.6 or 0.8). Each pair is counted once; 0.25 means 25% of possible pairs.', 'fraction of unordered pairs'),
        time_panel('Right','Independent cycles β₁ (count)',
            'Dimension of the cycle space over GF(2) at Cᵢⱼ≥0.4, before and after filling triangles. Both curves use the same first at most 30 alphabetically retained variables; these are algebraic cycle counts, not geographic loops.', 'integer cycle-space dimension')]
    return [rolling,spectra,topology]


def label_axes(ax,description,*,time=False,integer=False):
    ax.set_xlabel(description['x_label'],fontsize=9)
    ax.set_ylabel(description['y_label'],fontsize=9)
    if time:
        locator=mdates.AutoDateLocator(minticks=3,maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if integer:ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def axis_caption(descriptions):
    return '<div class="axis-guide">'+''.join(
        '<p><b>'+html.escape(item['panel'])+' x-axis:</b> '+html.escape(item['x_meaning'])+
        ' <span class="muted">Units: '+html.escape(item['x_units'])+'.</span> <b>Y-axis:</b> '+
        html.escape(item['y_meaning'])+' <span class="muted">Units: '+html.escape(item['y_units'])+'.</span></p>'
        for item in descriptions)+'</div>'


def charts(spec, summary):
    identifier=spec['dataset_id']
    paths=[FIG/(identifier+'__'+suffix+'.png') for suffix in ('metrics','spectra','topology')]
    stamp=FIG/(identifier+'__chart_input.json')
    fingerprint=dict(input_sha256=summary['input_sha256'],window_size=summary['window_size'],
        step_size=summary['step_size'],scheduled_windows=summary['scheduled_windows'],
        chart_version=4)
    if stamp.exists() and all(p.exists() and p.with_name(p.stem+'__axes.json').exists() for p in paths) and json.loads(stamp.read_text())==fingerprint:
        for path in paths:
            AXIS_AUDIT.extend(json.loads(path.with_name(path.stem+'__axes.json').read_text(encoding='utf-8')))
        return paths
    folder=Path(summary['output_directory'])
    features=pd.read_csv(folder/'features.csv')
    dates=pd.to_datetime(features.date,utc=True)
    descriptions=figure_axes(spec,summary)
    fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
    plots=[('lambda2','Raw algebraic connectivity λ₂'),('lambda2_scale_ratio','Edge-scale-adjusted q = λ₂/(n w̄)'),
        ('principal_component_share','Largest correlation eigenvalue / trace C'),('num_stocks','Retained variables in each 60-observation window')]
    for number,(ax,(column,label)) in enumerate(zip(axes.flat,plots)):
        if column in features: ax.plot(dates,features[column],color='#087c94',linewidth=.9)
        if column not in features or not features[column].notna().any():
            ax.text(.5,.5,'No defined measurements on this schedule',ha='center',va='center',
                transform=ax.transAxes,fontsize=8,wrap=True)
        if len(dates)>1:ax.set_xlim(dates.min(),dates.max())
        ax.set_title(label,loc='left',fontsize=10)
        label_axes(ax,descriptions[0][number],time=True,integer=column=='num_stocks')
        ax.tick_params(axis='x',rotation=18,labelsize=8)
    fig.suptitle(nice(identifier)+' · '+spec['interval']+' · '+spec['transform'].replace('_',' '),fontsize=13)
    series=save(fig,identifier+'__metrics')
    battery=json.loads((folder/'mathematical_battery.json').read_text())
    usable=[item for item in battery if item['status']=='ok']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    colors=plt.cm.viridis(np.linspace(.1,.9,max(1,len(usable))))
    undefined_scale=0
    for item,color in zip(usable,colors):
        c=np.array(item['operator_spectra']['C'])
        lap=np.array(item['operator_spectra']['L_positive'])
        axes[0].plot(np.linspace(0,1,len(c)),np.maximum(c[::-1],0)/max(c.sum(),1e-15),color=color,linewidth=1,
            label=pd.Timestamp(item['date']).strftime('%m-%d %H:%MZ') if spec['interval'] in ('1m','5m','10m','5min','300s') else item['date'][:10])
        mean=lap.sum()/len(lap)
        if mean>0:axes[1].plot(np.linspace(0,1,len(lap)),lap/mean,color=color,linewidth=1)
        else:undefined_scale+=1
    axes[0].set_title('Full correlation spectra · descending',fontsize=10)
    axes[1].set_title('Full positive-Laplacian spectra · ascending',fontsize=10)
    for ax,description in zip(axes,descriptions[1]):label_axes(ax,description)
    if usable: axes[0].legend(fontsize=7,ncol=3,loc='upper right')
    if undefined_scale:
        axes[1].text(.02,.02,f'{undefined_scale} selected graph(s): mean degree zero; scaling undefined',
            transform=axes[1].transAxes,fontsize=8,color='#984600')
    if not usable:
        for ax in axes:ax.text(.5,.5,'Selected windows have insufficient complete variables',ha='center',transform=ax.transAxes)
    fig.suptitle(nice(identifier)+' · selected scheduled windows',fontsize=13)
    spectra=save(fig,identifier+'__spectra')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for threshold in (.2,.4,.6,.8):
        values=[];x=[]
        for item in usable:
            graph=next(g for g in item['threshold_graphs'] if g['positive_correlation_threshold']==threshold)
            x.append(pd.Timestamp(item['date']));values.append(graph['density'])
        axes[0].plot(x,values,'o-',markersize=3,label=f'C ≥ {threshold:g}',linewidth=1)
    if usable:
        x=[pd.Timestamp(item['date']) for item in usable]
        graph=[next(g for g in item['threshold_graphs'] if g['positive_correlation_threshold']==.4)['subset_clique_homology'] for item in usable]
        axes[1].plot(x,[g['betti1_graph'] for g in graph],'o-',markersize=3,label='Graph cycle space')
        axes[1].plot(x,[g['betti1_clique'] for g in graph],'o-',markersize=3,label='Clique-complex H₁')
    axes[0].set(title='Threshold-graph density · complete retained universe',ylim=(-.03,1.03))
    axes[1].set_title('GF(2) cycles at C ≥ 0.4 · ≤30-variable subset')
    for number,ax in enumerate(axes):label_axes(ax,descriptions[2][number],time=True,integer=number==1)
    for ax in axes:
        if len(dates)>1:ax.set_xlim(dates.min(),dates.max())
        if not usable:ax.text(.5,.5,'Selected windows have insufficient complete variables',ha='center',va='center',
            transform=ax.transAxes,fontsize=8,wrap=True)
        ax.tick_params(axis='x',rotation=18,labelsize=8)
        if usable:ax.legend(fontsize=8)
    fig.suptitle(nice(identifier)+' · exploratory selected-date structure',fontsize=13)
    topology=save(fig,identifier+'__topology')
    stamp.write_text(json.dumps(fingerprint,indent=2))
    return [series,spectra,topology]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-validated-inventory',action='store_true',
        help='Reassemble existing validated panels; verify frozen hashes without repeating CSV parsing')
    args=parser.parse_args()
    AXIS_AUDIT.clear()
    FIG.mkdir(parents=True,exist_ok=True)
    if args.reuse_validated_inventory:
        inventory=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))['datasets']
        from data_cache import _file_hash
        for item in inventory:
            path=Path(item['specification']['snapshot_path'])
            if _file_hash(path/'prices.csv.gz')!=item['snapshot']['sha256']:
                raise ValueError('Validated frozen input hash changed: '+str(path))
    else:inventory=synchronize()
    # Source adapters use several explicit superseded-list key names. Refresh
    # this display-only list without reloading or scheduling obsolete prices.
    record=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
    archived_versions={Path(e['snapshot_path']).name:e for e in record['preserved_superseded_snapshots']}
    current_keys={e['snapshot']['snapshot_key'] for e in inventory}
    for name in ('financial.json','electricity_results.json','crypto_results.json'):
        path=OUT/name
        if not path.exists():continue
        doc=json.loads(path.read_text(encoding='utf-8'))
        for entry in doc.get('superseded_datasets',[])+doc.get('superseded_snapshots',[])+doc.get('superseded',[]):
            if isinstance(entry,dict) and 'snapshot_path' in entry:
                key=Path(entry['snapshot_path']).name
                if key not in current_keys:archived_versions[key]=entry
    preserved=list(archived_versions.values())
    source_docs={}
    for name in ('financial.json','electricity_results.json','crypto_results.json'):
        path=OUT/name
        if path.exists():source_docs[name]=json.loads(path.read_text())
    superseded=set()
    for document in source_docs.values():
        for spec in document.get('superseded_datasets',[])+document.get('superseded_snapshots',[])+document.get('superseded',[]):
            if isinstance(spec,dict):
                if 'snapshot_path' in spec and spec.get('replacement_snapshot_path')!=spec['snapshot_path']:
                    superseded.add(Path(spec['snapshot_path']).name)
    active=[];archived=[]
    for item in inventory:
        identifier=item['specification']['dataset_id']
        # Earlier one/two-day pilots remain preserved, but are not advertised as
        # another long research panel in the current acquisition count.
        if identifier.startswith('legacy_') or item['snapshot']['snapshot_key'] in superseded:archived.append(item)
        else:active.append(item)
    analyzed=[item for item in active if item['analysis'] is not None]
    observed=sum(item['analysis']['observed_values'] for item in analyzed)
    windows=sum(item['analysis']['scheduled_windows'] for item in analyzed)
    valid=sum(sum(v for k,v in item['analysis']['status_counts'].items() if k!='insufficient_assets') for item in analyzed)
    selected=sum(item['analysis']['selected_battery_windows'] for item in analyzed)
    storage=sum(p.stat().st_size for p in (ROOT/'data').rglob('*') if p.is_file())
    cards=[];table=[];export=[];graph_index=[];axis_index=[]
    equations=[('Rolling measurements','C = corr(X); Wᵢⱼ = max(Cᵢⱼ,0), Wᵢᵢ = 0; L = diag(W1) − W; w̄ = 2Σᵢ<ⱼ Wᵢⱼ/[n(n−1)]; q = λ₂/(n w̄) requires w̄ > 0.',
        'Shows connectivity, edge-scale-adjusted connectivity, concentration of correlation variation and retained sample size. Inputs are actual archived closes transformed to returns, or archived price/energy observations transformed using the per-panel declared first difference.'),
        ('Full eigenvalue curves','C vₖ = μₖvₖ; L uₖ = λₖuₖ. Curves show μₖ/tr(C) and λₖ/mean degree.',
        'Shows every eigenvalue at selected dates so spectral shape is visible. Mean-degree-zero graphs have undefined Laplacian scaling and are omitted from the normalized curves. Full C and positive-L spectra are also saved for every scheduled usable window, not just these curves.'),
        ('Threshold graphs and topology','Aᵢⱼ(t) = 1[Cᵢⱼ ≥ t], i≠j; β₁(graph) = m−n+β₀; β₁(clique) = m−rank ∂₁−rank ∂₂ over GF(2).',
        'Shows how positive-edge density changes with threshold and how many cycles remain after filling triangles. Full-universe densities use all retained variables; clique homology uses the first ≤30 alphabetically, explicitly an exploratory subset.')]
    for item in active:
        spec=item['specification'];summary=item['analysis'];identifier=spec['dataset_id']
        meta=item['snapshot'];missing='pending'
        if summary:
            missing=f"{summary['missing_values']:,}"
            fraction=summary['missing_values']/(summary['price_rows']*summary['symbols'])
            row=dict(dataset=identifier,family=spec['family'],market_group=market_group(spec),interval=spec['interval'],symbols=summary['symbols'],rows=summary['price_rows'],
                start=summary.get('first_observed_label',summary['first_label']),end=summary.get('last_observed_label',summary['last_label']),observed=summary['observed_values'],missing=summary['missing_values'],
                scheduled_windows=summary['scheduled_windows'],usable_windows=sum(v for k,v in summary['status_counts'].items() if k!='insufficient_assets'),
                step=summary['step_size'],snapshot_key=meta['snapshot_key'],sha256=meta['sha256'])
            export.append(row)
            paths=charts(spec,summary)
            descriptions=figure_axes(spec,summary)
            for path,(label,formula,explanation),axes_description in zip(paths,equations,descriptions):
                graph_index.append(dict(dataset=identifier,graph=label,equation=formula,
                    explanation_and_input_method=explanation,figure_file=str(path.resolve()),
                    x_axis_labels='; '.join(item['panel']+': '+item['x_label'] for item in axes_description),
                    y_axis_labels='; '.join(item['panel']+': '+item['y_label'] for item in axes_description),
                    axis_meanings_and_units=json.dumps(axes_description,ensure_ascii=False)))
                axis_index.extend(dict(dataset=identifier,graph=label,figure_file=str(path.resolve()),
                    quantitative=True,**item) for item in axes_description)
            plots=''.join(f'<figure><img class="zoom" src="{image(path)}" alt="{html.escape(nice(identifier)+": "+equations[number][0])}"><figcaption><b>{number+1}. {label}</b><p class="formula">{formula}</p><p>{explanation}</p>{axis_caption(descriptions[number])}</figcaption></figure>' for number,(path,(label,formula,explanation)) in enumerate(zip(paths,equations)))
            line=f"{summary['observed_values']:,} observed cells · {summary['missing_values']:,} missing ({fraction:.1%}) · {summary['scheduled_windows']:,} scheduled windows · step {summary['step_size']} · {summary['selected_battery_windows']} selected dates"
            notes=html.escape(prose(spec.get('notes','')))
            unavailable=spec.get('unavailable_symbols',[])
            if unavailable:notes+=' Unavailable requested symbols: '+html.escape(', '.join(unavailable))+'.'
            units=spec.get('units','source quote units')
            if not isinstance(units,str):units=json.dumps(units,ensure_ascii=False)
            cards.append(f'<section class="dataset" id="{identifier}"><h2>{html.escape(nice(identifier))}</h2><p>{html.escape(line)}</p><p>{notes}</p><p>Source: <a href="{html.escape(spec.get("source_url","#"),quote=True)}">{html.escape(spec.get("source_url","frozen local archive"))}</a> · Transform: {spec["transform"].replace("_"," ")} · Units: {html.escape(units)}</p>{plots}<details><summary>Analysis, provenance and caveats</summary><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k not in ("specification",)},indent=2))}</pre></details></section>')
        table.append(f'<tr><td><a href="#{identifier}">{html.escape(nice(identifier))}</a></td><td>{html.escape(spec["family"])}</td><td>{spec["interval"]}</td><td>{meta["symbol_count"]}</td><td>{row["start"][:10]}<br>{row["end"][:10]}</td><td>{summary["observed_values"]:,}</td><td>{missing}</td><td>{summary["scheduled_windows"]:,} / {row["usable_windows"]:,}</td></tr>' if summary else f'<tr><td>{identifier}</td><td colspan="7">Frozen, analysis pending</td></tr>')
    pd.DataFrame(export).to_csv(OUT/'Dataset_Inventory.csv',index=False)
    pd.DataFrame(graph_index).to_csv(OUT/'Graph_Index.csv',index=False)
    overview=dict(as_of_utc=datetime.now(timezone.utc).isoformat(),active_datasets=len(active),analyzed_datasets=len(analyzed),
        preserved_pilot_or_superseded_datasets=len(archived)+len(preserved),prepared_observed_cells_including_overlap=observed,scheduled_windows=windows,
        usable_windows=valid,selected_battery_dates=selected,local_data_bytes=storage,incremental_spend_usd=0,
        dataset_scientific_figures=len(graph_index),
        generalized_group_dataset_counts=pd.DataFrame(export).market_group.value_counts().sort_index().to_dict(),
        counting_note='Cell sum includes overlap between old/new stock histories, derived crypto intervals, sector totals and nested grid aggregates. It is not a count of unique independent raw observations.')
    (OUT/'overview.json').write_text(json.dumps(overview,indent=2))
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    families=pd.DataFrame(export).groupby('market_group')['observed'].sum().sort_values()
    ax.barh([s.replace('_',' ') for s in families.index],families.values,color='#087c94')
    coverage_axes=[dict(panel='Coverage overview',x_label='Observed archived level cells (count, log scale)',
        x_meaning='Count of nonmissing variable-by-source-timestamp cells in the archived level matrices, summed within each market group. The scale is logarithmic: each major power-of-ten step means ten times as many cells. Overlapping/derived panels and aggregates are included; these are not unique independent observations.',
        x_units='observed cell count on a base-10 logarithmic scale',y_label='Market group (category)',
        y_meaning='One of the seven declared market groups, ordered by its observed-cell count. Vertical position separates categories and has no numeric distance interpretation.',y_units='market-group categories')]
    ax.set_xscale('log');ax.set_title('Acquired research panels by market group')
    label_axes(ax,coverage_axes[0])
    family_figure=save(fig,'family_coverage')
    axis_index.extend(dict(dataset='__overview__',graph='Observed level-cell coverage',figure_file=str(family_figure.resolve()),
        quantitative=True,**item) for item in coverage_axes)
    pd.DataFrame(axis_index).to_csv(OUT/'Axis_Index.csv',index=False)
    expected={}
    for item in axis_index:
        name=Path(item['figure_file']).stem
        expected.setdefault(name,[]).append(item)
    for name,descriptions in expected.items():
        actual=[item for item in AXIS_AUDIT if item['figure']==name]
        if len(actual)!=len(descriptions):raise ValueError('Missing axis audit for '+name)
        for actual_axis,description in zip(actual,descriptions):
            if any(actual_axis[key]!=description[key] for key in ('x_label','y_label')):
                raise ValueError('Rendered axes and caption metadata disagree: '+name)
    (OUT/'axis_validation.json').write_text(json.dumps(dict(status='pass',
        dataset_figures=len(graph_index),coverage_figures=1,quantitative_axes=len(axis_index),
        every_rendered_axis_has_explicit_x_and_y_label=True,every_axis_has_plain_meaning_and_units=True,
        rendered_labels_match_caption_and_index=True,
        diagram_policy='Workflow box positions, sizes, spacing and arrows are layout/procedure only; there are no quantitative axes.',
        numeric_policy='Only plotting, labels and explanatory text changed; frozen observations and eigenanalyses were read without recomputation.',
        axes=AXIS_AUDIT),indent=2),encoding='utf-8')
    findings=[]
    for item in analyzed:
        summary=item['analysis']
        pca=summary.get('principal_component_share',{}).get('median')
        q=summary.get('lambda2_scale_ratio',{}).get('median')
        ipr=summary.get('fiedler_ipr',{}).get('median')
        if pca is not None:findings.append(f'<li><b>{html.escape(nice(item["specification"]["dataset_id"]))}</b>: median largest-PCA share {pca:.1%}, '+(f'median q {q:.3f}' if q is not None else 'q undefined')+(f', median separated Fiedler IPR {ipr:.3f}.' if ipr is not None else '; no separated Fiedler vectors available.')+'</li>')
    # Current-source documents remain linked as inspectable evidence, including
    # prices/limits that cannot be condensed into a single vendor endorsement.
    gaps='''<ul><li><b>Stocks / ETFs, long intraday:</b> free Yahoo minute data are recent and current-stock history is subject to survivorship. Try a free Alpaca historical-data account first; feed coverage and permissions must be checked. Verified Massive alternatives are $29 for one month/5 years or $79 for one month/10 years of minute history. No account or subscription has been created.</li><li><b>Crypto rolling top 100:</b> the frozen current ranking is not historical membership. Need point-in-time market caps, identities and delisted assets. Missing exchange listings and uncertain symbol identities stay excluded. CoinGecko Basic is a checked $35/month limited-history option; it does not establish minute-history or rolling-universe entitlement. Binance venue quotes are USDT, not consolidated USD.</li><li><b>FX:</b> recent indicative Yahoo quotes and ECB daily reference rates are acquired. The official free Dukascopy export API returned an access challenge, so no price data were acquired. A manual official CSV export is a possible next free route before considering EODHD $29.99/month or Massive $49/month. Bid/ask, venue, timezone and entitlement need verification.</li><li><b>Commodities:</b> physical monthly prices and selected daily spot series differ from continuous futures proxies. Contract-level futures require expiry, rolls, settlement, multiplier and volume metadata. Databento is a usage-priced candidate with $125 posted signup credit; an exact historical selection quote is needed before any purchase.</li><li><b>Electricity physical opportunity:</b> prices and metered interchange are available, but a tradable route needs topology, time-varying transfer limits, measured losses, wheeling/rights fees, schedules, lead time, market eligibility and transaction costs. Retail averages are not executable wholesale quotes. No profitable arbitrage or historical exploitation has been established.</li></ul>'''
    tree='''<div class="tree"><b>1. Acquire & freeze</b><br>Stocks · ETFs · FX · crypto · commodities · electricity prices · grid operations<br>↓ official files / explicit price feed → raw checksums → identity, units, interval checks<br><b>2. Define datasets</b><br>Universe × interval × intended span → observed coverage and failures → SQLite catalog<br>↓ financial arithmetic returns / declared price/energy first differences → complete 60-observation samples<br><b>3. Spectral core</b><br>Pearson C → positive W → degree D → L = D−W → full eigenvalues<br>↓ scale, weak-node, localization and rank diagnostics<br><b>4. Mathematical screens</b><br>Alternative operators · threshold combinatorics · matrix algebra · Sₙ relabeling · GF(2) homology · correlation geometry / H₀<br>↓ descriptive candidates and quality flags<br><b>5. Evidence required next</b><br>Source reconciliation → longer-window / common-factor sensitivity → chronological holdout → physical route feasibility → cost-adjusted opportunity</div>'''
    workflow_path=FIG/'expanded_process_tree.png'
    if workflow_path.exists():tree='<figure><img class="zoom" src="'+image(workflow_path)+'" alt="Grouped visual tree for data acquisition and spectral research"><figcaption>Grouped workflow: frozen sources, separate transformations, common-sample spectra, mathematical screens and required future validation. This is a process diagram with no quantitative x/y axes: box positions, sizes and spacing are layout choices, and arrows show the procedure rather than measured time, physical distance or causal magnitude.</figcaption></figure>'+tree
    methods='''<h2 id="cleaning-method">From frozen source to each graph</h2><ol>
<li><b>Locate and validate the source:</b> each dataset fixes its universe, interval, intended span, provider, units, adjustment and timestamp policy. Native manifests and SHA-256 hashes identify the exact saved price/energy matrix. Requested failures and earlier snapshots remain documented; integrity checks do not independently certify economic prices or identities.</li>
<li><b>Keep the level matrix intact:</b> invalid or ambiguous bars, contracts and identities already excluded by the recorded acquisition audits stay missing; their raw evidence remains preserved. No zero/forward filling, interpolation, global complete-case deletion or discretionary outlier deletion is added. <a href="../../data/groups/README.md">The seven-group guide</a> links raw/clean references and per-variable quality counts.</li>
<li><b>Transform every source row:</b> strictly positive financial prices use rₜ=Pₜ/Pₜ₋₁−1; declared price/energy panels use ΔXₜ=Xₜ−Xₜ₋₁, preserving signed/zero levels separately. The first undefined change is omitted. If bar_seconds is declared, a change spanning another elapsed interval is set to NaN. Daily/monthly source labels are not silently regularized. The full prepared files preserve every remaining transformed row; the plotted screen uses its disclosed schedule.</li>
<li><b>Clean within each 60-observation window:</b> replace nonfinite changes with NaN, remove any variable incomplete in that window, and remove sample standard deviation s (ddof=1) ≤10⁻¹². All operators and diagnostics use this one common retained sample. The retained count n can change. Windows with fewer than two variables remain recorded as insufficient, with no invented eigenvalues.</li>
<li><b>Construct the baseline:</b> compute Pearson C, Wᵢⱼ=max(Cᵢⱼ,0) with zero diagonal, weighted degree D=diag(W1), and L=D−W. Each unordered pair is counted once for mean weights/density. Save all C/L eigenvalues for every usable scheduled window; up to 12 selected scheduled endpoints add alternative operators, threshold graphs, subset homology and full-universe correlation-distance geometry. Selected endpoints are spaced by schedule position, not promised equal elapsed calendar time.</li>
<li><b>Report conditions and axes:</b> disconnected graphs have raw λ₂=0; normalized connectivity, connected-tree/resistance quantities and Fiedler localization are reported only when their operator/separation conditions hold. q requires positive mean clipped weight; spectral mean-degree scaling also requires a positive denominator. Missing/undefined values stay absent. Every quantitative subplot below names x/y and explains both meanings and units. Process-diagram layouts have no measured axes.</li>
<li><b>Align before comparing:</b> source dates/months are not publication timestamps. Crypto labels are bar ends; electricity operating labels are starts; EIA grid labels are HOUR ENDING despite their legacy native column name, so interval start is label minus one hour. Same symbol text does not prove identity, quote-currency, timing or aggregation compatibility. <a href="../organization/registry.json">The organization registry</a> records stable dataset/variable IDs, native references and full-row preparation; <a href="../organization/validation.json">its validation receipt</a> records exact reload checks.</li>
</ol><p><b>Explicit alternative, outside the figures below:</b> the daily-futures <a href="../../data/groups/commodities/yahoo_commodity_futures_daily_27ticker/prepared/variant.json">observed-source-calendar preparation</a> drops only 1,477 dates where all 26 native prices are missing, retains all partly missing rows and applies the same first difference. It is off by default, counts irregular observed dates rather than calendar days and is not another dataset. The static figures here continue to show the strict native-row baseline.</p>
<h2>EIA operating-data time and universe clarification</h2><p>EIA grid labels are UTC hour ENDING, not the interval starts implied by a legacy CSV index name; subtract one hour only when aligning physical interval starts. Separate-panel eigenvalues are unchanged by this label clarification. The generation panel has 80 total-generation and 67 natural-gas subclass variables; other fuel types remain raw. <a href="https://www.eia.gov/survey/form/eia_930/instructions.pdf">Official EIA-930 instructions</a>.</p>'''
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market data and eigenvalue atlas</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#182d3f;background:#eef3f6;margin:0}}main{{max-width:1180px;margin:auto;padding:32px 24px;background:white}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:30px}}a{{color:#00687f}}.stats{{display:flex;gap:14px;flex-wrap:wrap}}.stats div{{background:#e3f2f5;border-radius:9px;padding:15px;min-width:155px}}.stats b{{display:block;font-size:26px}}.muted{{color:#526879}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:9px;border-bottom:1px solid #d8e1e6;text-align:left;vertical-align:top}}th{{background:#e3f2f5}}figure{{margin:26px 0;border:1px solid #d8e1e6;padding:14px;border-radius:7px}}img{{max-width:100%;display:block}}.zoom{{cursor:zoom-in}}.axis-guide{{border-left:3px solid #b8dce1;padding-left:12px;font-size:14px}}.formula{{font-family:Cambria,Georgia,serif;background:#edf4f7;padding:12px}}.dataset{{border-top:3px solid #087c94;margin-top:45px}}.tree{{background:#102f44;color:white;padding:24px;line-height:1.8;border-radius:8px}}.tree b{{color:#a1e6ee}}pre{{white-space:pre-wrap;font-size:12px}}li{{margin:12px 0}}select,input{{font:inherit;padding:9px;max-width:100%}}dialog{{width:94vw;max-width:none;border:0;padding:12px}}dialog::backdrop{{background:#000b}}dialog img{{width:100%}}button{{font:inherit;padding:8px;cursor:pointer}}.hidden{{display:none}}@media print{{main{{padding:0}}section,figure{{break-inside:avoid}}.controls,dialog{{display:none}}}}
</style></head><body><main><h1>Market data and eigenvalue atlas</h1><p class="muted">Offline report rebuild · {overview['as_of_utc']} · $0 spent</p><p>Open this report to see what data are actually saved, their coverage, the full eigenvalue work and the next acquisition gaps. Original findings remain in <a href="../findings/Findings_Report.html">the 18-graph findings report</a>.</p><div class="stats"><div><b>{len(active)}</b>active dataset panels</div><div><b>{observed:,}</b>observed panel cells</div><div><b>{valid:,}</b>usable eigenvalue windows</div><div><b>{storage/1e9:.2f} GB</b>local archived data</div></div><p><b>Counting:</b> a dataset means one declared universe, interval and intended span. Raw archive files, individual symbols and analysis windows are not additional datasets. The cell sum includes derived/overlapping intervals, old/new stock histories and aggregate grid series; it is not a unique raw-observation count. {len(archived)+len(preserved)} earlier pilot/superseded snapshots remain archived separately.</p><h2>Process tree</h2>{tree}<h2>What was acquired</h2><figure><img class="zoom" src="{image(family_figure)}" alt="Observed panel cells by market family"><figcaption>Observed archived level cells per market group, summed from validated frozen snapshots; overlaps included.{axis_caption(coverage_axes)}</figcaption></figure><p>CSV inventory: <a href="Dataset_Inventory.csv">Dataset_Inventory.csv</a>. Catalog: <code>data/archive_catalog.sqlite</code>. Exact manifests, hashes and source metadata sit beside each frozen <code>prices.csv.gz</code>.</p><div class="tablewrap"><table><thead><tr><th>Dataset</th><th>Family</th><th>Interval</th><th>Variables</th><th>First / last observed label</th><th>Observed cells</th><th>Missing cells</th><th>Scheduled / usable</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>{methods}<h2>Mathematics actually computed</h2><p>All panels use 60 observations, so a five-minute window contains 300 minutes of bars, an hourly window contains 60 hourly observations, and daily/monthly windows contain 60 trading dates/months. Missing/gap masking can invalidate a window. The regular step is disclosed per dataset and limits this initial screen to about 3,000 windows per panel; it is not every possible window in the archive. Incomplete columns and near-constant columns are removed, with no missing-value filling. Intraday gaps are masked; this may leave insufficient-variable windows.</p><ul><li><b>Every usable scheduled window:</b> full correlation and positive-Laplacian eigenvalues, raw/normalized λ₂, q, rank (at most 59 in a complete 60-observation Pearson sample), largest-PCA share, effective rank, spanning-tree determinant identity, Kirchhoff resistance sum and separated Fiedler localization.</li><li><b>Up to 12 uniformly spaced selected windows per panel:</b> positive adjacency, absolute/signed/normalized Laplacian spectra; threshold densities, triangles, transitivity, graph cycles and closed-walk identities; Sₙ permutation invariance; GF(2) clique homology on the declared ≤30-variable subset; full-universe correlation-distance MST and H₀ merge lengths.</li><li><b>Geometry:</b> dᵢⱼ = √[2(1−Cᵢⱼ)] on the common complete sample. <b>Matrix algebra:</b> tr(A²)=2m, tr(A³)=6 triangles. <b>Group action:</b> P L Pᵀ has the same spectrum. These identities validate the calculation; they are not separate market signals.</li><li><b>Limits:</b> this is a defined exploratory battery, not an exhaustive application of every theorem. Clique H₁ is a subset screen; H₀ merge lengths come from the full distance filtration. There is no scored holdout, multiplicity-adjusted significance, causal explanation or executed arbitrage.</li></ul><h2>Measured descriptive screen</h2><p>These medians summarize this exact window schedule. Different intervals, units, universes, sample sizes and shared numeraires limit cross-market comparisons.</p><ul>{''.join(findings)}</ul><h2>Data gaps and lowest-cost checked candidates</h2>{gaps}<p>Primary pricing references: <a href="https://docs.alpaca.markets/us/docs/about-market-data-api">Alpaca</a>, <a href="https://massive.com/pricing?product=stocks">Massive</a>, <a href="https://eodhd.com/pricing">EODHD</a>, <a href="https://www.coingecko.com/en/api/pricing">CoinGecko</a>, <a href="https://databento.com/pricing">Databento</a>. Prices were checked for this batch; Massive research/publication permissions and retention after downgrade/termination are unresolved and require written clarification; EODHD intraday quotes are indicative and not retrospectively split/dividend adjusted. Feed, coverage and archive rights require checking before purchase. The project budget remains $100 total, with no money spent.</p><h2>Graph atlas</h2><p>There are {len(graph_index)} dataset scientific figures. <a href="Axis_Index.csv">Axis_Index.csv</a> gives the meanings and units of all {len(axis_index)} quantitative subplots, including coverage; workflow layouts have no quantitative axes. <a href="Graph_Index.csv">Graph_Index.csv</a> lists every dataset figure, its equation, its explanation/input method, explicit axis labels/meanings/units and its saved image path. Each dataset has three numbered figures: rolling metrics, full selected-date spectra, and combinatorics/topology. Equations and input methods are printed below each graph. Click a figure to enlarge it.</p><div class="controls"><label for="dataset">Show dataset: </label><select id="dataset"><option value="all">All datasets</option>{''.join(f'<option value="{s["specification"]["dataset_id"]}">{html.escape(nice(s["specification"]["dataset_id"]))}</option>' for s in analyzed)}</select><button id="top">Back to top</button></div>{''.join(cards)}<h2>Preserved earlier panels</h2><p>{len(preserved)} additional superseded snapshots are listed in inventory.json with exact replacement relationships; descriptor aliases for the same snapshot are not counted again.</p><ul>{''.join('<li>'+html.escape(nice(i['specification']['dataset_id']))+' · frozen snapshot '+i['snapshot']['snapshot_key']+'</li>' for i in archived)}</ul><p class="muted">Rebuild offline: python build_expansion_dashboard.py. The collection scripts have explicit network operations; analysis reads frozen local snapshots. Crypto chart data attribution: Binance Vision. These crypto derivative figures are provided for noncommercial research under CC BY-NC-SA 4.0; see <a href="https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md">Binance Vision dataset terms</a>. Other data-source redistribution conditions apply; see archived provenance and crypto_sources.json. No Binance sponsorship or endorsement is implied.</p></main><dialog id="zoom"><button id="close">Close figure</button><img id="large" alt="Enlarged scientific figure"></dialog><script>
const choice=document.getElementById('dataset');choice.addEventListener('change',()=>document.querySelectorAll('.dataset').forEach(s=>s.classList.toggle('hidden',choice.value!=='all'&&s.id!==choice.value)));document.getElementById('top').onclick=()=>window.scrollTo({{top:0,behavior:'smooth'}});const modal=document.getElementById('zoom');document.querySelectorAll('img.zoom').forEach(i=>i.onclick=()=>{{document.getElementById('large').src=i.src;document.getElementById('large').alt=i.alt;modal.showModal();}});document.getElementById('close').onclick=()=>modal.close();document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',()=>{{choice.value='all';choice.dispatchEvent(new Event('change'));}}));
</script></body></html>'''
    (OUT/'Data_and_Eigenvalues.html').write_text(text,encoding='utf-8')
    print(json.dumps(overview,indent=2),flush=True)


if __name__=='__main__':main()
