"""Offline, seeded SYNTHETIC demonstration. No market observations or network.

Run from any directory: python synthetic_demo.py [--output-dir demo_output]
The weekday index is a toy calendar, not an exchange-session database.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectral_core import clean_window,rolling_features


def main(output_dir=None):
    out=Path(output_dir) if output_dir else Path(__file__).resolve().parent/'demo_output'
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261015);n=240
    factor=rng.normal(0,.008,n);noise=rng.normal(0,.006,(n,7))
    values=noise+factor[:,None]*np.array([1.,.9,.8,-.65,.15,.6,0.])
    values[:,-1]=0.
    names=['equity_A','equity_B','equity_C','rates_A','metal_A','missing_example','constant_example']
    prices=pd.DataFrame(100*np.cumprod(1+values,axis=0),index=pd.bdate_range('2020-01-01',periods=n),columns=names)
    prices.iloc[120,5]=np.nan
    returns=prices.pct_change(fill_method=None).iloc[1:]
    assert returns.iloc[119:121,5].isna().all(), 'A missing price invalidates both adjacent returns'
    features=rolling_features(returns,window_size=60)
    clean=clean_window(returns.iloc[-60:]);corr=clean.corr()
    assert 'constant_example' not in clean and features.num_stocks.min()==5
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='BrBG')
    ax.set(xticks=range(len(clean.columns)),yticks=range(len(clean.columns)),xticklabels=clean.columns,yticklabels=clean.columns,
           xlabel='Synthetic variable (categorical columns)',ylabel='Synthetic variable (categorical rows)',title='SYNTHETIC: final common 60-return correlation')
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
    fig.colorbar(im,ax=ax,label='Pearson correlation (dimensionless)');fig.tight_layout()
    fig.savefig(out/'synthetic_correlation.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=False)
    axes[0].plot(features.date,features.lambda2,color='#007F86')
    axes[0].set(xlabel='Synthetic weekday label',ylabel='Raw lambda2 (dimensionless)',title='SYNTHETIC: rolling positive-graph connectivity')
    axes[1].plot(features.date,features.mean_raw_correlation,color='#D57B27')
    axes[1].set(xlabel='Synthetic weekday label',ylabel='Mean pair correlation\n(dimensionless)',title='A different summary of the same cleaned windows')
    for ax in axes:plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    fig.tight_layout();fig.savefig(out/'synthetic_diagnostics.png',dpi=140);plt.close(fig)
    summary={'data_kind':'SYNTHETIC, not market evidence','seed':20261015,'price_rows':len(prices),'return_rows':len(returns),
        'input_variables':len(names),'window':60,'windows':len(features),'retained_min':int(features.num_stocks.min()),
        'retained_max':int(features.num_stocks.max()),'missing_price_cells':int(prices.isna().sum().sum()),
        'missing_example_invalid_adjacent_returns':2,'constant_series_removed':True,'fill_policy':'none',
        'x_axes':'Toy weekday labels for histories; categorical variable names for heatmap.',
        'y_axes':'Dimensionless graph/correlation measurements; categorical variable names for heatmap.',
        'color_axis':'Pearson correlation from -1 to1, opposite through same-direction relationships.',
        'interpretation':'The planted factor creates positive/negative relationships. Graph scores do not forecast returns.',
        'source':'NumPy seeded random generator, with one deliberately missing price and one constant series.',
        'downloads':0}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir')
    main(parser.parse_args().output_dir)
