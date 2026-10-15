"""Python teaching figures for the fixed-12 ETF connectivity study.
No forecast fitting or scientific plotting occurs in the slide-layout helper.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/market_connectivity'
FIG=OUT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
NAVY='#16324B'; TEAL='#007A75'; ORANGE='#B65E29'; GREY='#657780'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':NAVY,'ytick.color':NAVY,'axes.titleweight':'bold','savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(FIG/name,dpi=140,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return str((FIG/name).relative_to(ROOT))

def timing_figure():
    fig,ax=plt.subplots(figsize=(13,4.6))
    for y,start,width,color,label in [(3,-158,60,GREY,'Oldest 60-return graph in history'),(2,-59,60,TEAL,'Current 60-return graph'),(1,-158,159,'#AED3CF','100 overlapping graph scores: 159 returns'),(0,-39,60,ORANGE,'Graph at the 20-session endpoint')]:
        ax.broken_barh([(start-.5,width)],(y-.24,.48),facecolors=color)
        ax.text(-158,y+.32,label,fontsize=12)
    ax.axvline(.5,color=NAVY,ls='--');ax.axvspan(.5,20.5,color=ORANGE,alpha=.08)
    ax.text(2,3.5,'Unknown future',color=ORANGE,fontsize=12)
    ax.set(ylim=(-.6,4),xlim=(-162,23),yticks=[],xlabel='Return-session position relative to today (0 = latest observed return)')
    ax.set_title('Window timing example: graph W60, history K100, horizon H20',loc='left',pad=20)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    return save(fig,'R01_window_timing.png')

def pathway_figure():
    fig,ax=plt.subplots(figsize=(13,4.8));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
    steps=[(.2,4.1,'Frozen adjusted prices','5,201 rows × 12 funds'),(4.2,4.1,'Arithmetic returns','5,200 source sessions; no filling'),(8.1,4.1,'Rolling correlations','Same 12 funds; W sessions'),(.2,2.4,'Positive / negative graphs','Keep both signs separately'),(4.2,2.4,'Laplacians and scores','L = degree − adjacency'),(8.1,2.4,'History and earlier training','K scores → model input matrix'),(.2,.6,'Earlier tuning + calibration','Train only completed outcomes'),(4.2,.6,'Future average / endpoint','H sessions; score change'),(8.1,.6,'Compare with baselines','Later outcomes; pending stays unknown')]
    for x,y,title,sub in steps:
        ax.text(x,y,title,fontsize=14,weight='bold',va='top');ax.text(x,y-.42,sub,fontsize=11.5,va='top')
    for a,b in [((3.25,3.9),(3.85,3.9)),((7.25,3.9),(7.85,3.9)),((3.25,2.2),(3.85,2.2)),((7.25,2.2),(7.85,2.2)),((3.25,.4),(3.85,.4)),((7.25,.4),(7.85,.4))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
    ax.text(.2,-.15,'Read each row left to right, then move down. Layout positions have no measured axes.',fontsize=11,color=GREY)
    fig.tight_layout();return save(fig,'R02_cleaning_pathway.png')


MODELS=('persistence','local_mean60','AR1','current','trajectory_selected')
MODEL_LABELS={'persistence':'No change','local_mean60':'Recent mean','AR1':'Mean reversion','current':'Current state','trajectory_selected':'Add history'}
CHANNEL_LABELS={'positive_lambda2_over_n':'Positive bottleneck connectivity','negative_mean_weight':'Opposing-move strength','negative_lambda2_over_n':'Negative-graph connectivity'}
COLORS=['#8A969C','#C0B392','#526B91','#69ACA5',TEAL]

def load_results():
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    cases=index['results']
    if isinstance(cases,dict):cases=list(cases.values())
    if cases and isinstance(cases[0],str):cases=[json.loads((OUT/'cases'/x).read_text()) for x in cases]
    return index,cases

def case_at(cases,channel,W=60,H=20,kind='future_mean'):
    return next(c for c in cases if c['channel']==channel and c['W']==W and c['H']==H and c['target_kind']==kind)

def score_history_figure():
    import pandas as pd
    frame=pd.read_csv(OUT/'graph_scores.csv.gz');frame=frame[frame.W.eq(60)].copy();frame['date']=pd.to_datetime(frame.date)
    fig,axs=plt.subplots(1,3,figsize=(14.5,4.3))
    for ax,ch,color in zip(axs,CHANNEL_LABELS,(TEAL,ORANGE,GREY)):
        ax.plot(frame.date,frame[ch],color=color,lw=.8)
        ax.set(title=CHANNEL_LABELS[ch].replace(' ','\n',1),xlabel='Observed session date',ylabel='Score (dimensionless)')
        ax.tick_params(axis='x',rotation=35,labelsize=11);ax.grid(alpha=.17)
    fig.tight_layout();return save(fig,'R03_score_history.png')

def comparison_figure(case,name):
    fig,ax=plt.subplots(figsize=(12.8,4.3))
    values=[case['models'][m]['mae'] for m in MODELS]
    bars=ax.barh([MODEL_LABELS[m] for m in MODELS],values,color=COLORS)
    ax.invert_yaxis();ax.set_xlabel('Mean absolute prediction error (dimensionless score units; lower is better)')
    ax.set_ylabel('Forecast method (categorical)');ax.set_xlim(0,max(values)*1.23)
    for b,v in zip(bars,values):ax.text(v+max(values)*.02,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=14)
    ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True);fig.tight_layout();return save(fig,name)

def grid_figure(cases):
    fig,axs=plt.subplots(1,2,figsize=(13.5,4.5))
    arrays=[]
    for ch in list(CHANNEL_LABELS)[:2]:
        a=np.array([[(1-case_at(cases,ch,W,H)['models']['trajectory_selected']['mae']/case_at(cases,ch,W,H)['models']['current']['mae'])*100 for H in (20,60,100)] for W in (20,40,60,120)])
        arrays.append(a)
    vmax=max(5,max(np.max(np.abs(a)) for a in arrays))
    for ax,ch,a in zip(axs,list(CHANNEL_LABELS)[:2],arrays):
        im=ax.imshow(a,cmap='BrBG',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set(xticks=range(3),xticklabels=(20,60,100),yticks=range(4),yticklabels=(20,40,60,120),xlabel='Forecast horizon H (trading sessions)',ylabel='Graph window W (return observations)',title=CHANNEL_LABELS[ch])
        for (r,c),v in np.ndenumerate(a):ax.text(c,r,f'{v:+.1f}%',ha='center',va='center',color='white' if abs(v)>.65*vmax else NAVY,fontsize=15)
    fig.subplots_adjust(wspace=.38,left=.07,right=.83,bottom=.20,top=.84)
    cax=fig.add_axes([.89,.20,.022,.64])
    fig.colorbar(im,cax=cax,label='Reduction in MAE versus current state (%)')
    return save(fig,'R06_window_horizon_grid.png')

def calibration_figure(positive,negative):
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for ax,case in zip(axs,(positive,negative)):
        ax.plot([0,1],[0,1],ls='--',color=GREY,lw=1,label='Perfect agreement')
        for label,col in (('down',ORANGE),('flat',GREY),('up',TEAL)):
            bins=[b for b in case['models']['trajectory_selected']['calibration'] if b['class']==label]
            ax.plot([b['mean_prediction'] for b in bins],[b['observed_frequency'] for b in bins],marker='o',color=col,label=label)
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel='Estimated category probability (fraction)',ylabel='Observed category frequency (fraction)',title=CHANNEL_LABELS[case['channel']]);ax.grid(alpha=.17)
    axs[1].legend(fontsize=10,loc='lower right');fig.tight_layout();return save(fig,'R07_probability_calibration.png')

def distribution_figure():
    import pandas as pd
    frame=pd.read_csv(OUT/'graph_scores.csv.gz');frame=frame[frame.W.eq(60)].dropna(subset=['positive_lambda2_over_n','negative_mean_weight'])
    fig,axs=plt.subplots(1,2,figsize=(13,4.4))
    n=len(frame);cuts=[(0,1000,'First 1,000 windows',GREY),((n-1000)//2,(n-1000)//2+1000,'Middle 1,000',TEAL),(n-1000,n,'Last 1,000',ORANGE)]
    for ax,ch in zip(axs,list(CHANNEL_LABELS)[:2]):
        for start,stop,label,color in cuts:
            x=np.sort(frame[ch].iloc[start:stop].to_numpy());y=np.arange(1,len(x)+1)/len(x);ax.step(x,y,where='post',label=label,color=color,lw=2)
        ax.set(xlabel='Observed graph score (dimensionless)',ylabel='Fraction of windows at or below score',title=CHANNEL_LABELS[ch],ylim=(0,1));ax.grid(alpha=.17)
    axs[1].legend(fontsize=11,loc='lower right');fig.tight_layout();return save(fig,'R08_empirical_distributions.png')

def endpoint_figure(cases):
    fig,axs=plt.subplots(1,2,figsize=(13,4.2))
    for ax,ch in zip(axs,list(CHANNEL_LABELS)[:2]):
        values=[]
        for kind in ('future_mean','endpoint'):
            c=case_at(cases,ch,kind=kind);values.append([(1-c['models'][m]['mae']/c['models']['persistence']['mae'])*100 for m in ('AR1','current','trajectory_selected')])
        loc=np.arange(3)
        ax.bar(loc-.18,values[0],width=.35,color=TEAL,label='Average over next 20');ax.bar(loc+.18,values[1],width=.35,color=ORANGE,label='20-session endpoint')
        ax.axhline(0,color=GREY,lw=1);ax.set(xticks=loc,xticklabels=['Mean\nreversion','Current\nstate','Add\nhistory'],ylabel='MAE reduction versus no change (%)',xlabel='Forecast method (categorical)',title=CHANNEL_LABELS[ch]);ax.grid(axis='y',alpha=.17)
    axs[1].legend(fontsize=11);fig.tight_layout();return save(fig,'R09_average_vs_endpoint.png')


def early_heatmap_figure():
    from market_connectivity_data import load_panel
    frame,_=load_panel();early=frame.iloc[:1000]
    c=early.corr().to_numpy()
    fig,ax=plt.subplots(figsize=(13,4.5));im=ax.imshow(c,cmap='BrBG',vmin=-1,vmax=1,aspect='auto')
    ax.set(xticks=range(12),yticks=range(12),xticklabels=early.columns,yticklabels=early.columns,xlabel='Fund symbol (categorical columns)',ylabel='Fund symbol (categorical rows)')
    ax.tick_params(labelsize=11);fig.colorbar(im,ax=ax,pad=.02,label='Return correlation (dimensionless)',ticks=[-1,-.5,0,.5,1])
    ax.set_title('Early-history check only: first 1,000 common returns, 2006–2010',loc='left',pad=15)
    fig.tight_layout();return save(fig,'R00_early_correlation_wide.png')


def build_presentation_content():
    from datetime import datetime, timezone
    import base64
    import html
    import pandas as pd
    index,cases=load_results()
    if index['status']!='complete' or len(cases)!=50:
        raise RuntimeError('All 50 frozen cases must finish before presenting final findings')
    positive=case_at(cases,'positive_lambda2_over_n');negative=case_at(cases,'negative_mean_weight')
    literal=case_at(cases,'negative_lambda2_over_n')
    assets=[early_heatmap_figure(),timing_figure(),pathway_figure(),score_history_figure(),
            comparison_figure(positive,'R04_positive_forecast_error.png'),comparison_figure(negative,'R05_negative_forecast_error.png'),
            grid_figure(cases),calibration_figure(positive,negative),distribution_figure(),endpoint_figure(cases)]
    sources=json.loads((ROOT/'output/market_selection/external_sources.json').read_text())
    source_notes='\n'.join(s['title']+': '+s['url'] for s in sources['sources'])
    reproducibility='Research: market_connectivity.py; data: market_connectivity_data.py; plots/content: market_connectivity_report.py. Protocol SHA256 '+index['protocol_sha256']+'. Frozen source price SHA256 d279458ba10b2cb9e30986b073055c2d1912d85eda5e7b42aa9e1331153de235. Full inputs and output metrics remain local.\n'
    slides=[]
    def add(title,**kw):
        kw['notes']=reproducibility+kw.get('notes','')
        slides.append({'title':title,**kw})
    def figure(title,name,caption,detail):
        add(title,figure=str((FIG/name).relative_to(ROOT)),caption=caption,alt=title,notes=detail)
    add('Can market connections\npredict their direction?',layout='cover',subtitle='12 ETFs across stock sectors, gold, bonds and commodities',caption='Exploratory findings · daily observations, 2006–2026',notes='Question: what do past rolling graph windows tell us about future average connectivity and endpoint direction? This is not a price forecast or a trading result.')
    add('A practical next market: diversified ETFs',table=[['Candidate','Usable long history','Main limitation'],['12 diversified ETFs','5,200 complete returns','Fixed funds; holdings evolve'],['9 stock-sector ETFs','Longer, complete history','Much less early opposition'],['Wholesale electricity','No strict W120 windows*','Missing quotes/product changes'],['Commodity futures','Long observed-row variant','Roll/contract mapping unverified']],columnWidths=[290,375,455],tableFontSize=24,tableHeight=355,caption='Selected by coverage and early structure—not forecast scores. *Audited fixed cohorts; not all electricity data everywhere.',notes=source_notes+'\nSelection: output/market_selection/audit.json and SUITABILITY.md. Selected funds XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY/GLD/TLT/DBC. 5,201 positive adjusted close rows Feb6,2006–Oct8,2026 yield5,200 complete return rows. Electricity legacy4/modern4/modern5 cohorts all zero W120. Futures10 observed-calendar variant has4,055 W120 windows but continuous rolls remain unverified. Different calendars/counts are not independent sample sizes.')
    figure('Different exposures produce different links','R00_early_correlation_wide.png','Rows/columns identify funds. Color is return correlation: green together, brown opposite. This early check does not prove predictability.',source_notes+'\nPearson correlation over first1,000 common returns only, Feb7,2006–Jan27,2010. Separate W60 early audit: median negative-pair fraction18.18% in12funds, versus0% in9sector-only funds on the same dates. Current fund selection is retrospective; no historical point-in-time universe claim.')
    add('What the connection scores mean',paragraphs=['Positive λ₂/12 measures the weakest bridge in the graph of same-direction moves. Higher means harder to split into separate groups.','Opposing-move strength averages all 66 negative edge magnitudes, including zeros. Higher means more or stronger opposing links.','Negative-graph λ₂/12 is a separate connectivity score. It is zero in '+f"{literal['negative_lambda2_zero_fraction']:.1%}"+' of W60 windows; zero does not mean no opposing pairs.'],bodySize=28,paragraphHeight=120,paragraphStep=143,notes='W+=max(C,0); W−=max(−C,0), off-diagonal only. L±=diag(row sums W±)−W±. Normalization by12 changes scale, not market composition. Both scores are dimensionless and are not probabilities. Falling positive connectivity need not be market damage: it can mean an asset moves differently. Literal negative-lambda2 models are evaluated separately in two W60/H20 cases and retained in complete results.')
    figure('The structure changes through time','R03_score_history.png','x: observed session date. y: each dimensionless score, on its own scale. These describe relationships—not whether prices rose or fell.','All valid fixed-cohort W60 windows, no source imputation. Graph histories use the same12 identities. A zero Laplacian connectivity score indicates disconnected support. Do not read coincident extrema as proof of an economic cause. Underlying paths: graph_scores.csv.gz.')
    figure('Three lengths answer different questions','R01_window_timing.png','x: return sessions relative to today. Rows show observation spans. W60 × K100 uses 159 returns; H20 adds 20 future sessions.','Schematic timing illustration, not measured market data. W is the return count inside each graph; K is the count of overlapping graph-score endpoints; H is the forecast horizon. K100 with W60 needs W+K−1=159 returns,160 prices. W60/H20 future-average windows contain82.5% known return slots on average; this is not explained variance.')
    figure('The complete data-to-forecast pathway','R02_cleaning_pathway.png','Read rows left to right, then down. Positions are not measured axes. No prices or returns were filled.','Source Yahoo/yfinance frozen auto_adjust=True snapshot. Validate hash/order/duplicates/finite positive closes. Preserve native source/SPY-supported sessions; weekends are not missing crypto bars. Check all12 finite returns and sampleSD>1e-12 within every window. All975 Friday/Monday gaps retain correct adjacent-session returns. No independent full exchange-calendar reconciliation. Source date is not a publication timestamp. All scientific logic and plots are Python.')
    span=positive['evaluation_span']
    add('How we tested the forecasts',paragraphs=['Choose W = 20/40/60/120; history K = 20/60/100; horizon H = 20/60/100 trading sessions. W and H define different questions.','At each forecast block, use only completed earlier outcomes. Earlier data choose history length and ridge strength; earlier forecast errors calibrate probabilities.','Compare the same dates with no change, recent mean, mean reversion and current-state regression. Primary W60/H20: '+str(span['n'])+' matched dates.'],bodySize=28,paragraphHeight=120,paragraphStep=143,notes='Evaluation primary '+span['first_asof']+' to '+span['last_asof']+'. Final25% chronological span, outer20-row blocks, earlier60/20/20 fit/tune/calibration span with H-row purges. Calibration errors come from causal refits with labels matured before their own forecast origins. Current regression includes outgoing/retained known-window summaries, simple mean correlation/minimum degree/volatility and current scores. History adds change,mean-minus-current,SD,slope; not K independent lag coefficients. Archive already inspected, all retrospective exploratory. 48 main estimands +2 negative-lambda2 diagnostics; conditional block intervals not familywise corrected.')
    for case,name,title in [(positive,'R04_positive_forecast_error.png','Positive connectivity: history is not the best model'),(negative,'R05_negative_forecast_error.png','Opposing moves: a modest measured improvement')]:
        m=case['models'];gain=1-m['trajectory_selected']['mae']/m['persistence']['mae']
        pair=next(p for p in case['paired_differences'] if p['first']=='trajectory_selected' and p['second']=='current' and p['block_length']==79)
        figure(title,name,f"x: average error; smaller is better. y: forecast method. Added history reduced error {gain:.1%} versus no change; extra value beyond current state is uncertain.",f"Target: mean of next20 W60 graph scores minus current score; n={span['n']}. MAE is in dimensionless score units, not price percentage. Models: "+json.dumps(m)+'. Paired trajectory−current MAE difference '+str(pair['mean'])+'; conditional79-block95% interval '+str(pair['percentile_95'])+'. Both primary history-versus-persistence intervals include zero, as do both history-versus-current intervals. Observed lower MAE alone is not an established advantage.')
    figure('More history does not help every setting','R06_window_horizon_grid.png','Rows: graph length W. Columns: horizon H. Color/number: error reduction versus current state. Positive helps; negative hurts. All cells are exploratory.','Future-average targets only. Each cell earlier-selects K and alpha within its own fixed W/H/channel. Changing W/H changes the estimand, so never select an optimum by raw cross-cell MAE. All12 cells/channel shown, not selected winners. Conditional uncertainty and full endpoint results remain in comparison.csv/cases. This heatmap is descriptive, not multiplicity-adjusted discovery.')
    history_rows=[['Primary channel','K20 dates','K60 dates','K100 dates']]
    for case in (positive,negative):
        counts={k:sum(r['dates'] for r in case['selection_counts'] if r['K']==k) for k in (20,60,100)}
        history_rows.append([CHANNEL_LABELS[case['channel']],*[str(counts[k]) for k in (20,60,100)]])
    add('There is no universal best history length',table=history_rows,columnWidths=[520,200,200,200],tableHeight=245,tableFontSize=24,caption='Earlier validation chose different histories for different questions. K counts score endpoints; these are not independent observations.',notes='Primary W60/H20 future-average selection counts, one selected setting per outerblock propagated to each forecast date. Alpha chosen from .1/1/10/100. Selection frequency is not evidence of predictive importance; in the literalnegativeλ2 case K100 selected on all dates but its outer error is worse than no change. No derivative-based universal optimal W; W/K are discrete and W changes target.')
    figure('A curve describes the past, not a guarantee','R08_empirical_distributions.png','x: observed W60 score. y: fraction of windows at or below it. Shifting curves show changing distributions; jumps at zero mark disconnected graphs.','Empirical CDFs from first/middle/last1,000 chronological usable windows, not performance extrema. Overlapping windows are dependent. These are distributions of historical levels, not future conditional probabilities. Different regimes can have different distributions; no bell/Gamma curve is forced. Raw-window membership does not confer independent samples.')
    poscov=positive['models']['trajectory_selected']['nominal90_coverage'];negcov=negative['models']['trajectory_selected']['nominal90_coverage']
    figure('Probability estimates still need caution','R07_probability_calibration.png',f"x: estimated chance. y: observed frequency. Closer to diagonal is better. Nominal 90% bands covered {poscov:.1%} positive / {negcov:.1%} opposing outcomes.",'Selected-history models; categories down/flat/up use ±0.5 initial-fitting SD of target changes, frozen before evaluation. Each point groups predictions within fixed20%-wide probability bins; counts retained in case metrics. No iid confidence bars implied. Probabilities derive from earlier causal residuals shifted around bounded point forecasts, one pseudocount per physically feasible class. Dependence/drift/location-shift assumptions limit reliability; neither plot establishes dependable future probabilities.')
    figure('Future average and endpoint are separate tests','R09_average_vs_endpoint.png','x: forecast methods. y: error reduction versus no change. Green averages the next 20 scores; brown predicts only the final score.','W60/H20. Endpoint target and future-average target have separate model fits, tuning, calibration and outcome errors. Raw error levels need not be comparable because targets differ. Averages can be smoother but contain greater known-window overlap. Matched disjoint nextH versus previousH raw-return graph changes are descriptive diagnostics only, not a successfully tested disjoint prediction model.')
    latest_rows=[['Score (dimensionless)','Observed','Projected mean','Down / flat / up']]
    for ch in ('positive_lambda2_over_n','negative_mean_weight'):
        latest=next(v for v in index['latest'] if v['channel']==ch and v['W']==60 and v['H']==20 and v['target_kind']=='future_mean')
        m=next(m for m in latest['models'] if m['model']=='trajectory_selected');p=m['probabilities']
        latest_rows.append([CHANNEL_LABELS[ch],f"{m['current_score']:.4f}",f"{m['predicted_future_level']:.4f}",f"{p['down']:.0%} / {p['flat']:.0%} / {p['up']:.0%}"])
    add('The archived October 8 projection',table=latest_rows,columnWidths=[425,180,235,280],tableHeight=245,tableFontSize=24,caption='October 8 data; generated October 15. Target partly elapsed and unscored here—not live. Probabilities remain unvalidated. Flat uses a training-defined band; rounded shares may not total 100%.',notes='Selected-history model for futuremean. Exact targetdate absent because no independently supplied future trading calendar. Flat is a training-defined score-change band, not exactly unchanged. Zero down-probability for the near-zero positive score is a physical category restriction, not confidence that connection strength cannot decrease at all. Models/probabilities differ from best retrospectively observed model; not chosen by final test scores. Latest records include physicallevel clipping, intervals, actualissuetime/sourceasof, traininglabels and fitparameters. '+json.dumps([v for v in index['latest'] if v['W']==60 and v['H']==20 and v['target_kind']=='future_mean'])[:12000])
    add('Equations behind the presentation',paragraphs=['Returns: r = P / previous P − 1. Correlations: C = Pearson(r). Edge weights: W⁺ = max(C, 0), W⁻ = max(−C, 0); diagonals are zero.','Laplacian: L = D − W. Positive score = λ₂(L⁺)/12. Opposing strength = mean W⁻ across all 66 pairs. Negative λ₂(L⁻)/12 stays separate.','Average target = mean of next H scores − current score.\nEndpoint target = score at +H − current score.\nRidge uses current features and earlier history summaries.'],bodySize=26,paragraphHeight=125,paragraphStep=143,notes='D is diagonal of weighted degrees. Ridge minimizes squared target error plus alpha*sum(beta_j²) after train-only median/flags/standardization with unpenalized intercept. X rows are dated past feature vectors; columns are current/known-window measurements plus four summaries for selected K. This uses linear algebra for a testable forecast, not an extra unexplained eigenvalue of a window-of-windows matrix. All scientific arrays, formulas and figures generated in Python; slide helper only lays out completed content.')
    add('What we learned and what comes next',paragraphs=['This ETF panel is a better next research setting: long complete history and genuinely different exposures. That does not guarantee easier prediction.','The history models lowered measured error versus no change, but neither primary improvement is established by the block intervals. Extra history adds little beyond current state.','Next: freeze a fresh future test, then evaluate directed lagged links or a better documented power panel separately. Current forecasts do not tell us which prices rise.'],bodySize=28,paragraphHeight=120,paragraphStep=143,notes='Do not claim a validated financial edge, causal mechanism, universal optimal window or physical arbitrage result. A current-state model can exploit mechanical effects of known observations rolling out; W60/H20 futuremean retains82.5%knownreturnslots. New primary retrospectivemarketselectedfollowingcrypto negativefindings; no fresh untouched market-discoveryclaim. Source snapshot endsOct8; futureexperiment needsfreshasofdataandactualissuance beforetarget. Official filings/news still notused; any later layer must usepublicrelease/asofversions. Spend0USD. '+source_notes)
    content={'title':'Market connectivity findings','created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':index['protocol_sha256'],'slides':slides}
    build=OUT/'presentation_build';build.mkdir(exist_ok=True)
    (build/'content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')
    # Fully offline, accessible companion. Figure bytes are embedded; all math is already Python-computed.
    blocks=[]
    for i,s in enumerate(slides,1):
        body=''
        if s.get('subtitle'):body+='<p class="lead">'+html.escape(s['subtitle'])+'</p>'
        if s.get('figure'):
            data=base64.b64encode((ROOT/s['figure']).read_bytes()).decode();body+='<img alt="'+html.escape(s.get('alt',s['title']))+'" src="data:image/png;base64,'+data+'">'
        if s.get('table'):
            body+='<div class="scroll"><table>'+''.join('<tr>'+''.join(('<th>' if r==0 else '<td>')+html.escape(str(v))+('</th>' if r==0 else '</td>') for v in row)+'</tr>' for r,row in enumerate(s['table']))+'</table></div>'
        for p in s.get('paragraphs',[]):body+='<p>'+html.escape(p)+'</p>'
        if s.get('caption'):body+='<p class="caption">'+html.escape(s['caption'])+'</p>'
        body+='<details><summary>Methods, equations and evidence</summary><p>'+html.escape(s.get('notes','')).replace('\n','<br>')+'</p></details>'
        blocks.append('<section id="slide'+str(i)+'"><p class="number">'+str(i)+' / '+str(len(slides))+'</p><h1>'+html.escape(s['title']).replace('\n','<br>')+'</h1>'+body+'</section>')
    table=pd.read_csv(OUT/'comparison.csv').to_html(index=False,classes='all-results',border=0,float_format=lambda v:f'{v:.5g}')
    appendix='<section><h1>Complete experiment table</h1><p>All 50 cases, five models each. W/H change the target. Secondary comparisons are exploratory; intervals are conditional, not familywise adjusted. Relative MAE below 1 means lower error than the named baseline.</p><details><summary>Show all 250 model rows</summary><div class="scroll">'+table+'</div></details><p>Data rights: Yahoo/provider research-use caveats apply; no raw price feed is embedded. Spending: $0.</p></section>'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Market connectivity findings</title><style>body{margin:0;background:#edf3f1;color:#16324b;font:20px/1.55 Arial,sans-serif}main{max-width:1120px;margin:auto}section{background:white;padding:42px;margin:24px 0;break-after:page}h1{font-size:36px;line-height:1.2;margin:14px 0 30px}.lead{font-size:28px;color:#007a75}img{width:100%;height:auto}.caption{font-size:20px;color:#334c5c}p{max-width:1000px}table{width:100%;border-collapse:collapse;font-size:20px}th,td{text-align:left;padding:14px;border-bottom:1px solid #d5e2df}th{background:#16324b;color:white}.scroll{overflow:auto}details{font-size:16px;margin-top:20px;color:#526570;overflow-wrap:anywhere}summary{cursor:pointer}.number{font-size:16px;color:#657780}.all-results{font-size:12px}@media(max-width:650px){section{padding:22px}h1{font-size:28px}body{font-size:18px}table{font-size:16px}}@media print{body{background:white}section{margin:0;padding:24px}details{display:none}h1{font-size:30px}}</style><main>'+''.join(blocks)+appendix+'</main></html>'
    (OUT/'Connectivity_Findings.html').write_text(page,encoding='utf-8')
    manifest={'status':'complete','slides':len(slides),'figures':assets,'all_scientific_logic':'Python','plots_raster_in_pptx':'User requires all scientific graph-making in Python; slide layout places existing PNGs. Data and model tables remain separately available.','protocol_sha256':index['protocol_sha256'],'complete_cases':len(cases),'source_end':'2026-10-08','is_live_forecast':False}
    (OUT/'presentation_build'/'content_receipt.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--teaching-only',action='store_true');args=parser.parse_args()
    if args.teaching_only:
        print([timing_figure(),pathway_figure(),early_heatmap_figure()])
    else:build_presentation_content()
