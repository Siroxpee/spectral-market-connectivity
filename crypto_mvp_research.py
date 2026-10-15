"""Frozen, fixed-cohort crypto graph research and provisional chronological test.

No acquisition, price filling, cohort selection on performance or trading.
Unsigned connectivity and signed balance spectra have different meanings.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/crypto_mvp/research"
SITE = ROOT / "crypto_mvp/dist/research"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT"]
OPERATORS = ("positive", "negative", "signed", "shifted")
WINDOW, HORIZON, SEED = 60, 20, 20261015
COLORS = {"positive":"#17324D", "negative":"#D57B27", "signed":"#7665a2", "shifted":"#007F86"}
MODEL_COLORS = {"constant":"#62747b", "indicators":"#007F86", "indicators_spectral":"#D57B27"}
REFERENCES = [
    {"title":"Kunegis et al., Spectral Analysis of Signed Graphs for Clustering, Prediction and Visualization", "url":"https://epubs.siam.org/doi/10.1137/1.9781611972801.49", "use":"Signed Laplacian with absolute degrees; signed spectral interpretation differs from ordinary connectivity."},
    {"title":"Kunegis, Applications of Structural Balance in Signed Social Networks", "url":"https://arxiv.org/abs/1402.6865", "use":"Signed structural balance and Laplacian context; market interpretation here is descriptive, not a claim about social mechanisms."},
    {"title":"SciPy gamma distribution", "url":"https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html", "use":"Gamma MLE with fixed zero location; pdf/logpdf/CDF."},
    {"title":"SciPy lognormal distribution", "url":"https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html", "use":"Lognormal MLE with fixed zero location; positive-support parameterization."},
    {"title":"Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation", "url":"https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf", "use":"Proper probability scoring; this implementation minimizes Brier and negative log scores."},
    {"title":"Kunsch, The Jackknife and the Bootstrap for General Stationary Observations", "url":"https://doi.org/10.1214/aos/1176347265", "use":"Block resampling for dependent observations; our conditional circular-block sensitivity is approximate and does not establish stationarity."},
    {"title":"SciPy L-BFGS-B minimizer", "url":"https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html", "use":"Explicit regularized logistic objective with analytic gradient; no scikit-learn installation."},
]


def serial(value):
    if isinstance(value, dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [serial(x) for x in value]
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, (pd.Timestamp,datetime)): return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return value.as_posix()
    return value


def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    text = json.dumps(serial(obj),ensure_ascii=False,allow_nan=False,separators=(",", ":"))
    temporary=path.with_suffix(".tmp"); temporary.write_text(text,encoding="utf-8"); temporary.replace(path)


def public_metadata(value):
    """Whitelist provenance; no machine paths/credentials reach Site JSON."""
    allowed = ("source", "source_url", "snapshot_key", "prices_sha256", "input_sha256", "interval", "units", "timestamp_policy", "notes", "universe", "coins", "requested_start", "requested_end", "requested_end_inclusive", "request_convention", "requested_end_exclusive", "first_interval_end", "last_interval_end", "preceding_row_date", "preceding_row_role", "venue", "quote_currency", "start", "end", "availability_policy", "daily_policy")
    return {key:value[key] for key in allowed if key in value}


def graph_measurements(correlation, operator):
    c=np.asarray(correlation,dtype=float)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not np.isfinite(c).all() or not np.allclose(c,c.T,atol=1e-12):
        raise ValueError("One symmetric finite common-sample correlation matrix is required")
    n=len(c); s=c.copy(); np.fill_diagonal(s,0.)
    if operator=="positive": w=np.maximum(s,0.); degree=w.sum(axis=1)
    elif operator=="negative": w=np.maximum(-s,0.); degree=w.sum(axis=1)
    elif operator=="shifted": w=(c+1.)/2.; np.fill_diagonal(w,0.); degree=w.sum(axis=1)
    elif operator=="signed": w=s; degree=np.abs(s).sum(axis=1)
    else: raise ValueError("Unknown graph operator")
    lap=np.diag(degree)-w
    eigenvalues,vectors=linalg.eigh(lap,check_finite=True)
    tolerance=1e-10*max(1.,2.*float(degree.max(initial=0.)))
    if eigenvalues[0] < -tolerance: raise ValueError("PSD graph operator failed")
    eigenvalues[np.abs(eigenvalues)<tolerance]=0.
    support=np.abs(s) if operator=="signed" else w
    components=int(connected_components(support,directed=False,return_labels=False))
    result={"operator":operator,"weights":w,"degrees":degree,"laplacian":lap,"eigenvalues":eigenvalues,
            "support_components":components,"lambda2":None,"mu1":None,"q":None,"normalized_lambda2":None,
            "mode_unique":False,"mode":None,"ipr":None,"gap":None,"tolerance":tolerance}
    if operator=="signed":
        result["mu1"]=float(eigenvalues[0]); gap=float(eigenvalues[1]-eigenvalues[0])
        unique=components==1 and gap>tolerance
        mode=vectors[:,0] if unique else None
    else:
        result["lambda2"]=float(eigenvalues[1]) if components==1 else 0.
        mean_weight=float(w[np.triu_indices(n,1)].mean())
        result["q"]=result["lambda2"]/(n*mean_weight) if mean_weight>0 else None
        gap=float(min(eigenvalues[1]-eigenvalues[0],eigenvalues[2]-eigenvalues[1])) if n>2 else float(eigenvalues[1]-eigenvalues[0])
        unique=components==1 and gap>tolerance
        mode=vectors[:,1] if unique else None
        if components==1 and np.all(degree>0):
            inv=1./np.sqrt(degree)
            normal=np.eye(n)-inv[:,None]*w*inv[None,:]
            result["normalized_lambda2"]=float(max(0.,linalg.eigh(normal,eigvals_only=True)[1]))
    result.update(mode_unique=bool(unique),mode=mode,ipr=float(np.sum(mode**4)) if mode is not None else None,gap=gap)
    return result


def common_complete(array):
    array=np.asarray(array,dtype=float)
    return bool(array.ndim==2 and array.shape[0]>=2 and np.isfinite(array).all() and np.all(array.std(axis=0,ddof=1)>1e-12))


def compute_features(prices):
    """Fixed identity cohort: invalidate a window rather than shrink its nodes."""
    if list(prices.columns)!=SYMBOLS or prices.index.tz is None: raise ValueError("Research cohort/order and UTC labels must match fixed declaration")
    observed=prices.to_numpy(dtype=float)
    if np.any(observed[np.isfinite(observed)]<=0): raise ValueError("Crypto return inputs must be positive")
    finite_prices=prices.where(np.isfinite(prices))
    returns=finite_prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan).iloc[1:]
    elapsed=prices.index.to_series().diff().dt.total_seconds().iloc[1:]
    returns.loc[elapsed.ne(86400)]=np.nan
    rows=[]; spectra=[]; correlations={}; previous={name:None for name in OPERATORS}; prev_pos=None
    for end in range(WINDOW,len(returns)+1):
        window=returns.iloc[end-WINDOW:end]
        row={"date":window.index[-1].isoformat(),"interval_start":(window.index[-1]-pd.Timedelta(days=1)).isoformat(),
             "source_position":end-1,"status":"incomplete_fixed_cohort","n":len(SYMBOLS),"observations":WINDOW}
        if not common_complete(window):
            row["invalid_variables"]=";".join(window.columns[window.isna().any() | (window.std(ddof=1)<=1e-12)])
            rows.append(row); previous={name:None for name in OPERATORS}; prev_pos=None; continue
        c=window.corr().to_numpy(); correlations[end-1]=c
        upper=c[np.triu_indices(len(c),1)]; ceig=linalg.eigh(c,eigvals_only=True)
        btc=window.iloc[:,0].to_numpy(); basket=window.to_numpy().mean(axis=1)
        row.update(status="ok",invalid_variables="",mean_pearson=float(upper.mean()),mean_abs_c=float(np.abs(upper).mean()),
                   negative_fraction=float((upper<0).mean()),positive_density03=float((upper>=.3).mean()),
                   pca_share=float(ceig[-1]/ceig.sum()),btc_return=float(btc[-1]),basket_return=float(basket[-1]),
                   btc_realized_vol60=float(btc.std(ddof=1)*np.sqrt(365)),basket_realized_vol60=float(basket.std(ddof=1)*np.sqrt(365)),
                   btc_momentum20=float(np.prod(1.+returns.iloc[end-20:end,0].to_numpy())-1.))
        for name in OPERATORS:
            g=graph_measurements(c,name)
            prefix=name+"_"
            row[prefix+("mu1" if name=="signed" else "lambda2")]=g["mu1"] if name=="signed" else g["lambda2"]
            for key in ("q","normalized_lambda2","ipr","gap","support_components","mode_unique"):
                row[prefix+key]=g[key]
            row[prefix+"min_degree"]=float(g["degrees"].min())
            mode=g["mode"]
            row[prefix+"turnover"]=(float(max(0.,min(1.,1.-abs(np.dot(previous[name],mode)))))
                if mode is not None and previous[name] is not None and prev_pos==end-2 else None)
            previous[name]=mode
            spectra.append({"date":row["date"],"operator":name,"eigenvalues":g["eigenvalues"].tolist(),"mode_unique":g["mode_unique"]})
        prev_pos=end-1; rows.append(row)
    frame=pd.DataFrame(rows)
    event=[]
    for _,row in frame.iterrows():
        position=int(row["source_position"])
        future=returns.iloc[position+1:position+1+HORIZON]
        if row["status"]!="ok" or len(future)!=HORIZON or not common_complete(future):
            event.append(np.nan); continue
        future_c=future.corr().to_numpy()
        difference=(future_c-correlations[position])[np.triu_indices(len(SYMBOLS),1)]
        event.append(float(np.sqrt(np.mean(difference**2))))
    frame["future_correlation_rms20"]=event
    return frame,returns,spectra


def score(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-12,1.-1e-12)
    return {"n":len(y),"events":int(y.sum()),"event_rate":float(y.mean()),"mean_probability":float(p.mean()),
            "brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))}


def logit_objective(beta,design,y,C):
    """Summed logistic loss, unpenalized intercept and explicit L2 penalty."""
    eta=design@beta
    value=float(np.sum(np.logaddexp(0.,eta)-y*eta)+np.dot(beta[1:],beta[1:])/(2.*C))
    gradient=design.T@(special.expit(eta)-y)
    gradient[1:]+=beta[1:]/C
    return value,gradient


def fit_logit(x,y,C):
    """Train-only median/scaling; sum log loss + ||beta||²/(2C), bias free."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if len(np.unique(y))<2: raise ValueError("Training needs both event classes")
    all_missing=np.isnan(x).all(axis=0)
    medians=np.array([float(np.nanmedian(x[:,j])) if not all_missing[j] else 0. for j in range(x.shape[1])])
    missing=np.isnan(x); imputed=np.where(missing,medians,x)
    indicator_columns=np.flatnonzero(missing.any(axis=0) & ~all_missing)
    expanded=np.column_stack([imputed,missing[:,indicator_columns].astype(float)])
    mean=expanded.mean(axis=0); scale=expanded.std(axis=0,ddof=0)
    active=scale>1e-12
    scale=np.where(active,scale,1.)
    z=(expanded[:,active]-mean[active])/scale[active]
    design=np.column_stack([np.ones(len(z)),z])
    def objective(beta): return logit_objective(beta,design,y,C)
    initial=np.zeros(design.shape[1]); initial[0]=special.logit(np.clip(y.mean(),1e-6,1.-1e-6))
    solution=optimize.minimize(objective,initial,jac=True,method="L-BFGS-B",options={"gtol":1e-8,"ftol":1e-12,"maxiter":3000})
    if not solution.success or not np.isfinite(solution.fun): raise RuntimeError("Logistic fit failed: "+str(solution.message))
    return {"C":float(C),"beta":solution.x,"medians":medians,"indicator_columns":indicator_columns,"mean":mean,"scale":scale,"active":active,
            "all_missing_training_columns":np.flatnonzero(all_missing),"converged":True,"iterations":int(solution.nit),"gradient_max":float(np.max(np.abs(solution.jac)))}


def predict_logit(model,x):
    x=np.asarray(x,dtype=float); missing=np.isnan(x)
    imputed=np.where(missing,model["medians"],x)
    expanded=np.column_stack([imputed,missing[:,model["indicator_columns"]].astype(float)])
    standardized=(expanded[:,model["active"]]-model["mean"][model["active"]])/model["scale"][model["active"]]
    return special.expit(np.column_stack([np.ones(len(x)),standardized])@model["beta"])


def calibration(y,p):
    edges=np.array([0.,.05,.1,.2,.3,.5,.75,1.])
    bins=[]
    for i in range(len(edges)-1):
        keep=(p>=edges[i]) & (p<=edges[i+1] if i==len(edges)-2 else p<edges[i+1])
        if keep.any(): bins.append({"lower":float(edges[i]),"upper":float(edges[i+1]),"n":int(keep.sum()),"mean_prediction":float(p[keep].mean()),"observed_frequency":float(y[keep].mean())})
    return bins


def paired_block_interval(differences,block,replicates=2000,seed=SEED):
    differences=np.asarray(differences,dtype=float); n=len(differences)
    rng=np.random.default_rng(seed); count=int(np.ceil(n/block))
    starts=rng.integers(0,n,size=(replicates,count))
    indices=(starts[:,:,None]+np.arange(block)[None,None,:])%n
    samples=differences[indices.reshape(replicates,-1)[:,:n]].mean(axis=1)
    return {"block_length":block,"replicates":replicates,"seed":seed,"mean_difference":float(differences.mean()),
            "percentile_95":[float(x) for x in np.quantile(samples,[.025,.975])],"negative_favors_first_model":True}


def chronological_forecast(frame):
    available=len(frame)-HORIZON
    cut1,cut2=int(.6*available),int(.8*available)
    position=np.arange(len(frame)); metric=frame["future_correlation_rms20"].to_numpy(dtype=float)
    valid=(frame["status"].eq("ok").to_numpy() & np.isfinite(metric))
    masks={"train":valid & (position<cut1-HORIZON),
           "validation":valid & (position>=cut1) & (position<cut2-HORIZON),
           "test":valid & (position>=cut2) & (position<available)}
    threshold=float(np.quantile(metric[masks["train"]],.9))
    y=(metric>threshold).astype(float)
    # Exact source-position assertions guard fully realized horizon boundaries.
    assert np.all(frame.loc[masks["train"],"source_position"].to_numpy()+HORIZON < frame.iloc[cut1]["source_position"])
    assert np.all(frame.loc[masks["validation"],"source_position"].to_numpy()+HORIZON < frame.iloc[cut2]["source_position"])
    simple=["mean_pearson","mean_abs_c","negative_fraction","positive_density03","pca_share","positive_min_degree",
            "btc_realized_vol60","basket_realized_vol60","btc_momentum20","btc_return","basket_return"]
    spectral=[name for name in frame.columns if any(name.startswith(prefix+"_") for prefix in OPERATORS) and
              any(name.endswith("_"+suffix) for suffix in ("lambda2","mu1","q","normalized_lambda2","ipr","turnover"))]
    models={}; predictions={}
    baseline=float(y[masks["train"]].mean())
    for stage,mask in masks.items(): predictions[stage]={"constant":np.full(mask.sum(),baseline)}
    models["constant"]={"training_frequency":baseline,"validation":score(y[masks["validation"]],predictions["validation"]["constant"]),
                        "test":score(y[masks["test"]],predictions["test"]["constant"])}
    for name,fields in [("indicators",simple),("indicators_spectral",simple+spectral)]:
        x=frame[fields].to_numpy(dtype=float)
        candidates=[]
        for C in (.01,.1,1.,10.):
            fit=fit_logit(x[masks["train"]],y[masks["train"]],C)
            metrics=score(y[masks["validation"]],predict_logit(fit,x[masks["validation"]]))
            candidates.append((metrics["brier"],C,fit,metrics))
        _,C,fit,val_score=min(candidates,key=lambda item:(item[0],item[1]))
        for stage,mask in masks.items(): predictions[stage][name]=predict_logit(fit,x[mask])
        models[name]={"fields":fields,"selected_C":C,"C_grid":[.01,.1,1.,10.],
                      "validation_grid":[{"C":item[1],**item[3]} for item in candidates],"validation":val_score,
                      "test":score(y[masks["test"]],predictions["test"][name]),"fit":fit,
                      "calibration_test":calibration(y[masks["test"]],predictions["test"][name])}
    models["constant"]["calibration_test"]=calibration(y[masks["test"]],predictions["test"]["constant"])
    intervals=[]; target=y[masks["test"]]
    for first,second in [("indicators","constant"),("indicators_spectral","constant"),("indicators_spectral","indicators")]:
        diff=(predictions["test"][first]-target)**2-(predictions["test"][second]-target)**2
        for block in (20,60): intervals.append({"first":first,"second":second,**paired_block_interval(diff,block)})
    nonoverlap=[]
    for offset in range(HORIZON):
        keep=np.arange(offset,len(target),HORIZON)
        nonoverlap.append({"offset":offset,"n":len(keep),"event_rate":float(target[keep].mean()),
                           "scores":{name:score(target[keep],pred[keep]) for name,pred in predictions["test"].items()}})
    splits={}
    for stage,mask in masks.items():
        part=frame.loc[mask]
        splits[stage]={"n":int(mask.sum()),"first_asof":part.iloc[0]["date"],"last_asof":part.iloc[-1]["date"],
                       "last_label_available":(pd.Timestamp(part.iloc[-1]["date"])+pd.Timedelta(days=HORIZON)).isoformat(),"events":int(y[mask].sum()),"event_rate":float(y[mask].mean())}
    records=[]
    for stage in ("train","validation","test"):
        for j,i in enumerate(np.flatnonzero(masks[stage])):
            records.append({"date":frame.iloc[i]["date"],"stage":stage,"event_rms":metric[i],"event":int(y[i]),
                            **{name:float(pred[j]) for name,pred in predictions[stage].items()}})
    result={"status":"provisional chronological historical test; no live forecast","event":{"past_observations":WINDOW,"future_observations":HORIZON,
            "formula":"sqrt(mean_{i<j}((C_future20_ij-C_past60_ij)^2)) > training_90th_percentile", "threshold":threshold,"threshold_training_labels_only":True,
            "fixed_identity_cohort":True,"future_labels_never_predictor":True},
            "split_policy":"First60% / next20% / final20% of label-eligible scheduled as-of dates, chronologically;20 endpoints purged before each later split",
            "raw_label_eligible_scheduled_dates":available,"cut_positions":[cut1,cut2],"purged_per_boundary":HORIZON,"splits":splits,
            "models":models,"predictions":records,"uncertainty":{"method":"paired circular moving-block bootstrap of test Brier-loss differences, conditional on fixed fitted models; approximate percentile intervals, not IID p-values",
                "intervals":intervals,"limitations":"Overlapping windows, regime drift and model-selection/training uncertainty are not fully captured by conditional test-loss resampling."},
            "nonoverlap_future_label_sensitivity":{"stride":HORIZON,"all_offsets":nonoverlap,"note":"All20 offsets shown, not a favorable phase selected; future20 labels do not overlap within each offset, but past60 predictors still overlap and examples are few."},
            "implementation":{"library":"SciPy L-BFGS-B; sklearn not installed","objective":"sum(log(1+exp(eta))-y*eta)+sum(beta_j^2,j>0)/(2*C); intercept unpenalized",
                "preprocessing":"Train-only feature medians, missingness indicators and population scaling; all-constant expanded columns dropped. Source prices/returns are never imputed.",
                "tuning":"Fixed C grid selected by validation Brier, tie to smaller C; no refit on validation or test; training fitted transforms/models remain frozen."}}
    return result,masks


def distribution_study(frame):
    x=frame.loc[frame.status.eq("ok"),"positive_lambda2"].to_numpy(dtype=float)
    dates=frame.loc[frame.status.eq("ok"),"date"].tolist(); cut1,cut2=int(.6*len(x)),int(.8*len(x))
    subsets={"train":x[:cut1],"validation":x[cut1:cut2],"test":x[cut2:]}
    train=subsets["train"]; positive=train[train>0]
    p0=float((train==0).mean()); models={}
    for name,dist in (("gamma",stats.gamma),("lognormal",stats.lognorm)):
        parameters=dist.fit(positive,floc=0.)
        evaluation={}
        for stage,values in subsets.items():
            pos=np.sort(values[values>0]); cdf=dist.cdf(pos,*parameters)
            discrepancy=float(max(np.max(np.abs(cdf-np.arange(1,len(pos)+1)/len(pos))),np.max(np.abs(cdf-np.arange(len(pos))/len(pos)))))
            score_positive=float(np.mean(dist.logpdf(pos,*parameters)))
            zero_count=int((values==0).sum())
            mixture=None if zero_count and p0==0 else float(((np.log(p0)*zero_count if zero_count else 0.)+
                np.sum(np.log1p(-p0)+dist.logpdf(pos,*parameters)))/len(values))
            evaluation[stage]={"n":len(values),"positive_n":len(pos),"zero_n":zero_count,"zero_fraction":float(zero_count/len(values)),
                "mean_log_density_positive":score_positive,"mean_mixture_log_score":mixture,"cdf_max_discrepancy_descriptive":discrepancy}
        models[name]={"parameters":{"shape":float(parameters[0]),"loc":0.,"scale":float(parameters[2])},"evaluation":evaluation,
                      "mass_above_theoretical_lambda2_bound_n":float(dist.sf(len(SYMBOLS),*parameters))}
    best=min(models,key=lambda name:-models[name]["evaluation"]["validation"]["mean_log_density_positive"])
    result={"target":"Positive-graph lambda2 of fixed10,60 daily-return windows; not a distribution of asset returns","split":"Chronological60/20/20 of usable windows; fit first60% only; select family on validation log density; report untouched final20%",
            "cut_positions":[cut1,cut2],"train_first":dates[0],"train_last":dates[cut1-1],"validation_last":dates[cut2-1],"test_last":dates[-1],
            "zero_mass":{"training_empirical_probability":p0,"policy":"Point mass at zero reported separately; continuous fit conditional on positive lambda2. No silently smoothed zero mass."},
            "models":models,"selected_by_validation":best,"train_only_fitted":True,
            "limitations":["Overlapping windows are dependent; no IID KS p-value or universal distribution claim.","Gamma/lognormal have unbounded support, while this fixed-n positive graph has lambda2<=n; reported impossible-tail mass quantifies an approximation limitation.","Regime and cohort/interval/operator changes can change the empirical distribution."]}
    return result,subsets


def export_fixture(prices,metadata):
    selected=prices.tail(WINDOW+1)
    changes=selected.pct_change(fill_method=None).iloc[1:]
    if not common_complete(changes): raise ValueError("Latest declared fixture window incomplete")
    c=changes.corr().to_numpy()
    graphs={name:graph_measurements(c,name) for name in OPERATORS}
    fixture={"schema_version":1,"source":public_metadata(metadata),"symbols":SYMBOLS,"btc_symbol":"BTCUSDT",
        "price_dates":[x.isoformat() for x in selected.index],"prices":selected.to_numpy(),"returns":changes.to_numpy(),
        "correlation":c,"graphs":graphs,"absolute_tolerance":1e-8,
        "signed_note":"Signed mu1 is lowest eigenvalue/unique lowest mode; lambda2/q/Fiedler connectivity are deliberately null."}
    write_json(OUT/"python_fixture.json",fixture)
    # Raw per-coin prices/returns stay local. Hosted test evidence is only
    # the derived common correlation and graph operators, never a price feed.
    hosted={key:value for key,value in fixture.items() if key not in ("prices","returns","price_dates")}
    hosted["window_end"]=selected.index[-1].isoformat()
    hosted["observations"]=WINDOW
    hosted["publication_note"]="Matrices and spectra are analytic derivatives. Full raw-price/return test fixture is retained locally only."
    write_json(SITE/"python_fixture.json",hosted)
    return fixture


def self_test():
    c=np.array([[1.,.7,-.4],[.7,1.,-.2],[-.4,-.2,1.]])
    for operator in OPERATORS:
        g=graph_measurements(c,operator)
        assert np.allclose(g["laplacian"],g["laplacian"].T) and g["eigenvalues"].min()>=-1e-10
        assert np.isclose(g["eigenvalues"].sum(),np.trace(g["laplacian"]))
    signed=graph_measurements(np.array([[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]]),"signed")
    assert np.isclose(signed["mu1"],1.) and signed["lambda2"] is None and signed["q"] is None
    negative=graph_measurements(np.ones((3,3)),"negative")
    assert negative["support_components"]==3 and negative["ipr"] is None and negative["lambda2"]==0.
    x=np.column_stack([np.linspace(-2.,2.,100),np.full(100,np.nan)]); y=(x[:,0]>0).astype(float)
    fit=fit_logit(x,y,.1); assert fit["converged"] and predict_logit(fit,x)[0]<predict_logit(fit,x)[-1]
    medians=fit["medians"].copy(); predict_logit(fit,np.full((20,2),500.)); assert np.array_equal(fit["medians"],medians)
    interval=paired_block_interval(np.full(100,-.02),20,200)
    np.testing.assert_allclose(interval["percentile_95"],[-.02,-.02],atol=1e-15)
    rng=np.random.default_rng(SEED)
    design=np.column_stack([np.ones(17),rng.normal(size=(17,3))]); target=rng.integers(0,2,size=17)
    beta=rng.normal(size=4)
    numerical=optimize.approx_fprime(beta,lambda b:logit_objective(b,design,target,.1)[0],1e-7)
    np.testing.assert_allclose(logit_objective(beta,design,target,.1)[1],numerical,atol=2e-6,rtol=2e-6)
    assert np.isclose(logit_objective(np.array([1.,0.,0.,0.]),design,target,.1)[0],logit_objective(np.array([1.,0.,0.,0.]),design,target,10.)[0])
    synthetic=pd.DataFrame(np.exp(rng.normal(0,.01,size=(65,10)).cumsum(axis=0)),columns=SYMBOLS,index=pd.date_range("2020-01-01",periods=65,tz="UTC"))
    synthetic.iloc[20,0]=np.inf
    _,guarded,_=compute_features(synthetic)
    assert guarded.iloc[19:21,0].isna().all(),"Nonfinite current or previous price must invalidate its return"
    print("PASS graph spectra/signed semantics/disconnection; train-only logit state; dependent-loss resampling; logistic gradient/unpenalized intercept; nonfinite price adjacency")


def make_figures(prices,frame,spectra,distribution,forecast):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.18,"figure.facecolor":"white","savefig.facecolor":"white"})
    figures=[]; dates=pd.to_datetime(frame["date"],utc=True); source_dates=prices.index[1:]
    valid=frame.loc[frame.status.eq("ok")]
    def finish(fig,identifier,title,equation,method,axes,finding,limits):
        fig.suptitle(title,fontsize=14,x=.04,ha="left")
        fig.tight_layout(rect=(0,0,1,.955))
        folder=OUT/"figures"; folder.mkdir(parents=True,exist_ok=True)
        target=folder/(identifier+".png"); fig.savefig(target,dpi=155,bbox_inches="tight"); plt.close(fig)
        destination=SITE/"figures"; destination.mkdir(parents=True,exist_ok=True); shutil.copy2(target,destination/target.name)
        figures.append({"id":identifier,"title":title,"path":"research/figures/"+target.name,"equation":equation,"method":method,
                        "axis_explanation":axes,"finding":finding,"limitations":limits,"source":"Frozen Binance spot USDT, fixed10; source hashes recorded in index provenance."})
    fig,ax=plt.subplots(2,1,figsize=(12,8),gridspec_kw={"height_ratios":[1.35,1]})
    study_prices=prices.iloc[1:]
    normalized=study_prices.divide(study_prices.iloc[0]).mul(100)
    for symbol in SYMBOLS: ax[0].plot(source_dates,normalized[symbol],lw=1,label=symbol.removesuffix("USDT"))
    ax[0].set_yscale("log"); ax[0].set(xlabel="Interval end (UTC bar-end date)",ylabel="Price index (first study close = 100; log axis)")
    ax[0].legend(ncol=5,fontsize=8,loc="upper left")
    observed=study_prices.notna().to_numpy().T.astype(int)
    image=ax[1].imshow(observed,aspect="auto",interpolation="nearest",origin="upper",extent=[mdates.date2num(source_dates[0])-.5,mdates.date2num(source_dates[-1])+.5,9.5,-.5],cmap="Greens",vmin=0,vmax=1)
    ax[1].set_yticks(range(10),[s.removesuffix("USDT") for s in SYMBOLS]); ax[1].xaxis_date(); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[1].set(xlabel="Interval end (UTC bar-end date)",ylabel="Fixed coin identity (categorical)")
    bar=fig.colorbar(image,ax=ax[1],ticks=[0,1],pad=.02); bar.set_label("Source price observed (0 = missing, 1 = observed)")
    missing=int(study_prices.isna().sum().sum())
    finish(fig,"R01_prices_coverage","Fixed crypto cohort: price paths and honest coverage","I_t=100 P_t/P_first",
           "Ten identities fixed before analysis; no filling. First study observation normalizes each path; missing source prices remain gaps.",
           "Top: horizontal UTC interval-end date, vertical price index on a logarithmic scale. Bottom: time against categorical coins; the color scale marks observed/missing cells.",
           f"{len(study_prices):,} study daily observations across 10 coins; {missing} missing price cells. {len(valid):,}/{len(frame):,} scheduled 60-return windows retain the entire fixed cohort.",
           "Single venue and USDT numeraire; normalized price growth excludes costs and does not establish an investable historical selection.")

    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    for axis,name in zip(axes.flat,OPERATORS):
        field=name+("_mu1" if name=="signed" else "_lambda2")
        axis.plot(dates,frame[field],color=COLORS[name],lw=1)
        axis.set(title=name+(" · signed lowest mode" if name=="signed" else " · ordinary Laplacian"),xlabel="Window end (UTC bar-end date)",ylabel=("Signed μ₁" if name=="signed" else "λ₂")+" (dimensionless)")
        axis.set_ylim(bottom=0.)
        if frame[field].dropna().eq(0.).all():
            axis.set_ylim(0.,.05)
            axis.text(.04,.88,"λ₂ = 0 at every usable endpoint\nDisconnected support; Fiedler mode undefined",transform=axis.transAxes,fontsize=9,color=COLORS[name])
    finish(fig,"R02_operators","Four graph definitions produce different spectral summaries","L_+=D_+-max(S,0); L_-=D_--max(-S,0); L_s=D_|S|-S; W_shift=(C+1)/2 off diagonal",
           "Same complete 60-return sample and 10 identities for each operator; windows advance by one observation. Signed lowest μ1 is deliberately not labeled classical connectivity.",
           "Every horizontal axis is the UTC window-end date. Vertical axes show the named dimensionless eigenvalue; panel scales differ and are labeled.",
           f"Positive λ₂ median {valid.positive_lambda2.median():.3f}; negative graph disconnected in {(valid.negative_support_components.gt(1).mean()):.1%} of usable windows.",
           "Operator definitions change the question; eigenvalue magnitude across different operators is not a common risk scale. Adjacent windows overlap 59 returns.")

    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    comparisons=[("mean_pearson","Mean Pearson correlation (dimensionless)"),("positive_min_degree","Minimum positive weighted degree (dimensionless)"),
                 ("negative_fraction","Negative-correlation pair fraction (dimensionless)"),("basket_realized_vol60","Equal-weight basket 60-day realized volatility\n(annualized fraction; √365)")]
    associations={}
    for axis,(field,label) in zip(axes.flat,comparisons):
        pair=valid[[field,"positive_lambda2"]].dropna(); corr=float(pair.corr().iloc[0,1]); associations[field]=corr
        axis.scatter(pair[field],pair.positive_lambda2,s=8,alpha=.3,color=COLORS["positive"],rasterized=True)
        axis.set(xlabel=label,ylabel="Positive-graph λ₂ (dimensionless)",title=f"Descriptive Pearson association = {corr:.3f}")
    finish(fig,"R03_simple_indicators","Is λ₂ describing more than simpler indicators?","vol_60=sd(r_1,...,r_60;ddof=1)√365",
           "Each point uses the same valid 60-return cohort window. Basket is a daily equally weighted return measurement, not a costed portfolio strategy.",
           "Horizontal axes identify a simpler indicator and its units; every vertical axis is dimensionless positive λ₂. Points are overlapping historical windows.",
           f"Association with mean correlation {associations['mean_pearson']:.3f}; with minimum weighted degree {associations['positive_min_degree']:.3f}.",
           "Descriptive association neither proves incremental prediction nor supplies an IID significance test.")

    fig,axes=plt.subplots(2,1,figsize=(12,7.5))
    for name in OPERATORS:
        label=name+(" lowest mode" if name=="signed" else " Fiedler")
        axes[0].plot(dates,frame[name+"_ipr"],lw=.85,color=COLORS[name],alpha=.8,label=label)
        axes[1].plot(dates,frame[name+"_turnover"],lw=.85,color=COLORS[name],alpha=.8,label=label)
    axes[0].set(xlabel="Window end (UTC bar-end date)",ylabel="Inverse participation ratio\n(dimensionless)",ylim=(.09,1.01)); axes[0].axhline(.1,color="gray",ls=":",lw=.8,label="1/n = .1 diffuse lower bound")
    axes[1].set(xlabel="Window end (UTC bar-end date)",ylabel="Sign-invariant mode turnover\n(dimensionless)",ylim=(-.01,1.01))
    for axis in axes: axis.legend(ncol=3,fontsize=8,loc="upper left")
    finish(fig,"R04_mode_stability","Localization and conditional mode turnover","IPR=Σv_i^4; turnover=1-|v_tᵀv_(t-1)|",
           "Unsigned Fiedler vectors require a connected graph and separated second eigenvalue; signed uses its separated lowest mode on connected absolute support. Turnover additionally requires adjacent eligible windows with identical ordered identities.",
           "Horizontal axes are UTC endpoint dates. IPR near .1 is spread over 10 nodes, near 1 localized. Turnover ranges from 0 (same direction up to sign) to 1 (orthogonal). Undefined modes remain gaps.",
           f"Positive-mode IPR median {valid.positive_ipr.median():.3f}; eligible positive modes {valid.positive_mode_unique.sum():,}/{len(valid):,}.",
           "Numerical separation is not statistical stability. In 1,691/1,694 windows all correlations are nonnegative, making signed/positive operators equal and shifted/positive nonconstant eigenvectors identical. Adjacent windows also overlap.")

    selected=valid.iloc[[0,len(valid)//2,len(valid)-1]].date.tolist()
    lookup={(x["date"],x["operator"]):x["eigenvalues"] for x in spectra}
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    palette=["#197d78","#cb8140","#17324D"]
    for axis,name in zip(axes.flat,OPERATORS):
        for date,color in zip(selected,palette): axis.plot(range(1,11),lookup[(date,name)],marker="o",ms=3,lw=1,color=color,label=date[:10])
        axis.set(title=name,xlabel="Ascending eigenvalue rank (ordinal 1–10)",ylabel="Laplacian eigenvalue (dimensionless)",xticks=range(1,11)); axis.legend(fontsize=8)
        axis.set_ylim(bottom=0.)
        if all(np.max(np.abs(lookup[(date,name)]))<1e-10 for date in selected): axis.set_ylim(0.,.05)
    finish(fig,"R05_full_spectra","Full spectra at declared first, middle and last usable windows","L v_k=λ_k v_k, λ_1≤...≤λ_n",
           "Three endpoints are chosen by position in usable-window chronology, never eigenvalue extrema. All 10 eigenvalues are retained for every operator.",
           "Horizontal axes are ascending eigenvalue ranks, not dates. Vertical axes are raw dimensionless eigenvalues; legends identify the exact UTC window-end dates.",
           "Four operators × three predeclared chronological positions show all ten modes rather than only one scalar.",
           "Signed lowest eigenvalues need not be zero. Selected examples do not quantify uncertainty or summarize every historical regime.")

    usable=frame.loc[frame.status.eq("ok"),"positive_lambda2"].to_numpy(); a,b=distribution["cut_positions"]
    train=usable[:a]; test=np.sort(usable[b:]); train_pos=train[train>0]; test_pos=test[test>0]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].hist(train_pos,bins=30,density=True,color="#cce0d8",edgecolor="white",label="Training positive λ₂ histogram")
    grid=np.linspace(max(1e-5,float(usable[usable>0].min())*.6),min(10.,float(usable.max())*1.15),500)
    for name,dist,color in [("gamma",stats.gamma,"#007F86"),("lognormal",stats.lognorm,"#D57B27")]:
        pars=distribution["models"][name]["parameters"]; args=(pars["shape"],0.,pars["scale"])
        axes[0].plot(grid,dist.pdf(grid,*args),color=color,lw=1.6,label=name+" · training fit")
        axes[1].plot(grid,dist.cdf(grid,*args),color=color,lw=1.6,label=name+" · training-fit CDF")
    axes[1].step(test_pos,np.arange(1,len(test_pos)+1)/len(test_pos),where="post",color="#17324D",lw=1.3,label="Untouched-test positive ECDF")
    axes[0].set(xlabel="Positive-graph λ₂ (dimensionless)",ylabel="Conditional positive density (per unit λ₂)")
    axes[1].set(xlabel="Positive-graph λ₂ (dimensionless)",ylabel="Cumulative probability conditional on λ₂>0",ylim=(0,1))
    for axis in axes: axis.legend(fontsize=8)
    scores={name:result["evaluation"]["test"]["mean_log_density_positive"] for name,result in distribution["models"].items()}
    finish(fig,"R06_distribution","Training distribution fits face a chronological test","p(λ₂)=p₀δ₀+(1-p₀)f(λ₂|λ₂>0); Gamma/lognormal loc fixed0",
           "Only the first 60% fit parameters; the next 20% choose a family; the final 20% evaluate it. Zero probability mass is separate. Histogram and overlaid curves share density units.",
           "Left: dimensionless λ₂ versus probability density per λ₂ unit. Right: λ₂ versus conditional cumulative probability; dark steps are final-test observations, curves were fit earlier.",
           f"Validation selected {distribution['selected_by_validation']}; test log density: Gamma {scores['gamma']:.3f}, lognormal {scores['lognormal']:.3f}. Gamma test CDF discrepancy {distribution['models']['gamma']['evaluation']['test']['cdf_max_discrepancy_descriptive']:.3f} shows marked drift.",
           "These approximate marginal fits fail to establish a stable law for overlapping graph windows. No IID goodness-of-fit p-value is reported. Both families assign some mass above the theoretical upper bound 10.")

    prediction=pd.DataFrame(forecast["predictions"]); test_prediction=prediction.loc[prediction.stage.eq("test")]; test_date=pd.to_datetime(test_prediction.date,utc=True)
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    for stage,color in [("train","#dcebe5"),("validation","#e7e4f0"),("test","#fff0df")]:
        split=forecast["splits"][stage]
        axes[0].axvspan(pd.Timestamp(split["first_asof"]),pd.Timestamp(split["last_asof"]),color=color,alpha=.8,label=stage)
    axes[0].plot(dates,frame.future_correlation_rms20,lw=1,color="#17324D")
    axes[0].axhline(forecast["event"]["threshold"],color="#D57B27",ls="--",label="Training 90th-percentile event threshold")
    axes[0].set(xlabel="Historical forecast as-of (UTC bar-end date)",ylabel="Correlation-change RMS\n(past 60 vs future 20; dimensionless)"); axes[0].legend(ncol=2,fontsize=8)
    for name,color in MODEL_COLORS.items(): axes[1].plot(test_date,test_prediction[name],color=color,lw=1.2,label=name)
    axes[1].scatter(test_date,test_prediction.event,s=7,color="#17324D",alpha=.25,label="Historical event outcome 0/1")
    axes[1].set(xlabel="Untouched-test as-of (UTC bar-end date)",ylabel="Forecast probability / event outcome\n(fraction)",ylim=(-.03,1.03)); axes[1].legend(ncol=2,fontsize=8)
    finish(fig,"R07_chronological_forecast","A provisional future-pattern-change test","Event=1{RMS(C_future20-C_past60)>train90%}; p=logistic(β₀+Xβ)",
           "Training labels finish before validation starts; validation labels finish before test starts. Twenty endpoints are purged at each boundary. Models and preprocessing are train-only; C is selected by validation Brier.",
           "Top: historical as-of dates against correlation-change magnitude, with stage shading and a training-only threshold. Bottom: final-test dates against forecast fractions and realized zero/one labels.",
           f"Final test has {forecast['splits']['test']['n']} eligible dates and {forecast['splits']['test']['events']} event labels ({forecast['splits']['test']['event_rate']:.2%}); spectral model mean probability {forecast['models']['indicators_spectral']['test']['mean_probability']:.2%}, a marked overprediction.",
           "The event line is known only after the next 20 observations and is displayed for historical evaluation, never as a predictor. Publication latency and historical selection remain unresolved.")

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    nonempty=[item for value in forecast["models"].values() for item in value["calibration_test"]]
    limit=max(.2,float(np.ceil(10*max(max(item["mean_prediction"],item["observed_frequency"]) for item in nonempty))/10))
    axes[0].plot([0,limit],[0,limit],color="gray",ls=":",label="Perfect calibration reference")
    for name,color in MODEL_COLORS.items():
        bins=forecast["models"][name]["calibration_test"]
        label=name.replace("indicators_spectral","indicators + spectral")+" (bin n="+"/".join(str(item["n"]) for item in bins)+")"
        axes[0].plot([x["mean_prediction"] for x in bins],[x["observed_frequency"] for x in bins],marker="o",ms=5,color=color,label=label)
    axes[0].set(xlabel="Mean predicted event probability in fixed bin\n(fraction; shown range includes all nonempty bins)",ylabel="Observed event frequency in same test bin\n(fraction)",xlim=(0,limit),ylim=(0,limit)); axes[0].legend(fontsize=8)
    pairs=[("indicators","constant"),("indicators_spectral","constant"),("indicators_spectral","indicators")]
    for block,color,offset in [(20,"#007F86",-.09),(60,"#D57B27",.09)]:
        for i,(first,second) in enumerate(pairs):
            item=next(x for x in forecast["uncertainty"]["intervals"] if x["first"]==first and x["second"]==second and x["block_length"]==block)
            mean=item["mean_difference"]; lo,hi=item["percentile_95"]
            axes[1].errorbar(i+offset,mean,yerr=[[max(0.,mean-lo)],[max(0.,hi-mean)]],fmt="o",color=color,capsize=4,label=f"Circular block {block}" if i==0 else None)
    axes[1].axhline(0,color="gray",ls=":"); axes[1].set_xticks(range(3),["Indicators\n− constant","Plus spectral\n− constant","Plus spectral\n− indicators"])
    axes[1].set(xlabel="First model minus second model (categorical)",ylabel="Test mean Brier-loss difference\n(dimensionless)"); axes[1].legend(fontsize=8)
    best=min(forecast["models"],key=lambda name:forecast["models"][name]["test"]["brier"])
    finish(fig,"R08_calibration_uncertainty","Probability calibration and dependence-aware loss differences","Brier=mean((p-y)^2); paired difference=Brier_first-Brier_second",
           "Fixed calibration bins; legend counts show nonempty bin sizes from left to right. Paired circular-block resampling of fixed-model test loss differences uses 2,000 replicates, a declared seed and 20/60-observation blocks.",
           "Left: predicted fraction against realized frequency, zoomed to include all nonempty bins; the diagonal is a reference. Right: categorical model comparisons against mean Brier difference; negative favors the first model. Intervals are approximate 95% block percentiles.",
           f"Lowest observed test Brier: {best} ({forecast['models'][best]['test']['brier']:.4f}); test log loss {forecast['models'][best]['test']['log_loss']:.4f}.",
           "Conditional intervals do not include training/tuning uncertainty, guarantee stationarity or establish a general forecasting advantage. Selecting a winner after seeing test scores does not create a deployable validated strategy.")
    return figures,associations


def validate_research(frame,returns,spectra,forecast,masks,distribution):
    """Check saved mathematics, chronology and public-file boundaries offline."""
    import platform
    import scipy
    self_test()
    valid=frame.loc[frame.status.eq("ok")].set_index("date")
    assert len(spectra)==len(valid)*len(OPERATORS)
    assert frame.date.is_unique and pd.to_datetime(frame.date,utc=True).is_monotonic_increasing
    trace_error=0.
    for record in spectra:
        eig=np.asarray(record["eigenvalues"])
        assert eig.shape==(len(SYMBOLS),) and np.isfinite(eig).all() and np.all(np.diff(eig)>=-1e-9) and eig.min()>=-1e-9
        row=valid.loc[record["date"]]; mean=row.mean_pearson; absolute=row.mean_abs_c
        trace={"positive":(absolute+mean)/2,"negative":(absolute-mean)/2,"signed":absolute,"shifted":(mean+1)/2}[record["operator"]]*len(SYMBOLS)*(len(SYMBOLS)-1)
        trace_error=max(trace_error,abs(float(eig.sum())-trace))
    assert trace_error<1e-9
    all_nonnegative=valid.negative_fraction.eq(0.)
    shift_error=float((valid.loc[all_nonnegative,"shifted_lambda2"]-(len(SYMBOLS)/2+valid.loc[all_nonnegative,"positive_lambda2"]/2)).abs().max())
    assert shift_error<1e-9
    for name in OPERATORS:
        ipr=valid[name+"_ipr"].dropna()
        assert ipr.between(1/len(SYMBOLS)-1e-9,1+1e-9).all()
        assert valid.loc[~valid[name+"_mode_unique"].astype(bool),name+"_ipr"].isna().all()
    threshold=float(np.quantile(frame.loc[masks["train"],"future_correlation_rms20"],.9))
    assert threshold==forecast["event"]["threshold"]
    assert pd.Timestamp(forecast["splits"]["train"]["last_label_available"])<pd.Timestamp(forecast["splits"]["validation"]["first_asof"])
    assert pd.Timestamp(forecast["splits"]["validation"]["last_label_available"])<pd.Timestamp(forecast["splits"]["test"]["first_asof"])
    preprocessing_error=0.
    for name in ("indicators","indicators_spectral"):
        model=forecast["models"][name]; fit=model["fit"]
        assert not any("future" in field for field in model["fields"])
        x=frame.loc[masks["train"],model["fields"]].to_numpy(dtype=float)
        medians=np.array([np.nanmedian(x[:,j]) if np.isfinite(x[:,j]).any() else 0. for j in range(x.shape[1])])
        np.testing.assert_allclose(fit["medians"],medians,atol=0,rtol=0)
        missing=np.isnan(x); expanded=np.column_stack([np.where(missing,medians,x),missing[:,fit["indicator_columns"]].astype(float)])
        check_mean=expanded.mean(axis=0); check_scale=np.where(expanded.std(axis=0)>1e-12,expanded.std(axis=0),1.)
        preprocessing_error=max(preprocessing_error,float(np.max(np.abs(fit["mean"]-check_mean))),float(np.max(np.abs(fit["scale"]-check_scale))))
        np.testing.assert_allclose(fit["mean"],check_mean,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(fit["scale"],check_scale,atol=1e-12,rtol=1e-12)
        chosen=min(model["validation_grid"],key=lambda item:(item["brier"],item["C"]))
        assert model["selected_C"]==chosen["C"]
    assert distribution["train_only_fitted"] and distribution["zero_mass"]["training_empirical_probability"]==0.
    prohibited={"prices","price_dates","returns","raw_prices","raw_closes"}
    def inspect(value):
        if isinstance(value,dict):
            assert not prohibited.intersection(value.keys()),"Raw per-coin feed in hosted research"
            for nested in value.values(): inspect(nested)
        elif isinstance(value,list):
            for nested in value: inspect(nested)
        elif isinstance(value,str):
            assert "C:\\Users\\" not in value and "C:/Users/" not in value
    files=[]
    for path in sorted(SITE.rglob("*.json")):
        inspect(json.loads(path.read_text(encoding="utf-8")))
        files.append({"path":path.relative_to(ROOT/"crypto_mvp/dist").as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    event_dates=[pd.Timestamp(x["date"]) for x in forecast["predictions"] if x["stage"]=="test" and x["event"]]
    clusters=[]
    for date in event_dates:
        if not clusters or date-pd.Timestamp(clusters[-1]["last"])>pd.Timedelta(days=1):
            clusters.append({"first":date.isoformat(),"last":date.isoformat(),"labels":1})
        else: clusters[-1].update(last=date.isoformat(),labels=clusters[-1]["labels"]+1)
    receipt={"status":"pass","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "checks":["Four operators: all saved eigenvalues finite, ascending and nonnegative within tolerance; traces equal degrees derived from the same common correlation sample.",
            "Mode IPR is within [1/n,1], and ineligible modes retain missing IPR; signed mu1 is not ordinary lambda2/q.",
            "Training forward labels end before validation; validation forward labels end before test; threshold recomputes from training labels only.",
            "Selected logistic C minimizes the stored validation Brier grid; learned medians/scalers match training-only data within floating-point tolerance; no future target fields are predictors.",
            "Analytic logistic gradient matches finite differences; intercept is unpenalized; preprocessing state is unchanged by predictions.",
            "Nonfinite current or previous prices invalidate both adjacent returns; no source-price filling is used.",
            "On all-nonnegative correlation windows, the shifted/positive eigenvalue identity holds within floating-point tolerance; these are not independent measurements.",
            "Paired block resampling preserves constant loss differences; no IID p-value is substituted.",
            "Hosted research JSON contains no raw per-coin prices/returns, local machine paths or nonstandard JSON NaN; the full numerical fixture stays local."],
        "maximum_spectrum_trace_error":trace_error,"maximum_preprocessing_recompute_error":preprocessing_error,"all_nonnegative_pair_windows":int(all_nonnegative.sum()),"maximum_shifted_positive_eigenvalue_identity_error":shift_error,"saved_eigenvalues":len(spectra)*len(SYMBOLS),"test_event_clusters":clusters,
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
        "hosted_json_files":files,"limits":"Implementation checks are not independent observation verification or proof of a forecasting advantage. Chronology uses nominal bar ends; provider publication latency is not measured."}
    write_json(OUT/"checks.json",receipt)
    return receipt


def write_findings_html(index,receipt):
    """Portable, beginner-readable local report; all figures are embedded."""
    escape=html.escape; forecast=index["forecast"]; distribution=index["distribution"]
    rows="".join(f"<tr><th>{escape({'constant':'Training-frequency baseline','indicators':'Simple indicators','indicators_spectral':'Indicators + spectral'}[name])}</th><td>{item['test']['mean_probability']:.2%}</td><td>{item['test']['brier']:.6f}</td><td>{item['test']['log_loss']:.6f}</td></tr>" for name,item in forecast["models"].items())
    splitrows="".join(f"<tr><th>{escape(stage.title())}</th><td>{item['n']:,}</td><td>{item['events']}</td><td>{item['event_rate']:.2%}</td><td>{item['first_asof'][:10]} to {item['last_asof'][:10]}</td><td>{item['last_label_available'][:10]}</td></tr>" for stage,item in forecast["splits"].items())
    fitrows="".join(f"<tr><th>{escape(name.title())}</th><td>{item['parameters']['shape']:.6f}</td><td>{item['parameters']['scale']:.6f}</td><td>{item['evaluation']['validation']['mean_log_density_positive']:.4f}</td><td>{item['evaluation']['test']['mean_log_density_positive']:.4f}</td><td>{item['evaluation']['test']['cdf_max_discrepancy_descriptive']:.3f}</td></tr>" for name,item in distribution["models"].items())
    figures=[]
    for number,item in enumerate(index["figures"],1):
        picture=OUT/"figures"/Path(item["path"]).name
        encoded=base64.b64encode(picture.read_bytes()).decode("ascii")
        figures.append(f"<figure id='{escape(item['id'])}'><h3>Graph {number}. {escape(item['title'])}</h3><img src='data:image/png;base64,{encoded}' alt='{escape(item['axis_explanation'],quote=True)}'><figcaption><p class='formula'>{escape(item['equation'])}</p><p><b>How it was made:</b> {escape(item['method'])}</p><p><b>Read the axes:</b> {escape(item['axis_explanation'])}</p><p><b>Finding:</b> {escape(item['finding'])}</p><p class='limits'><b>Limit:</b> {escape(item['limitations'])}</p></figcaption></figure>")
    references="".join(f"<li><a href='{escape(item['url'],quote=True)}'>{escape(item['title'])}</a> — {escape(item['use'])}</li>" for item in index["references"])
    clusters="; ".join(f"{item['first'][:10]}–{item['last'][:10]} ({item['labels']} labels)" for item in receipt["test_event_clusters"])
    findings="".join(f"<li>{escape(line)}</li>" for line in index["summary"]["findings"])
    checkitems="".join(f"<li>{escape(line)}</li>" for line in receipt["checks"])
    hashvalue=index["provenance"].get("prices_sha256","unknown")
    document=f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Crypto research findings — fixed cohort</title><style>
    :root{{--navy:#17324d;--teal:#007f86;--orange:#d57b27;--muted:#526372}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f8;color:#233441;font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1120px;margin:auto;padding:42px 30px 80px}}h1,h2,h3{{color:var(--navy);line-height:1.2}}h1{{font-size:2.35rem;max-width:850px}}h2{{margin-top:48px;font-size:1.65rem}}h3{{font-size:1.2rem}}p{{max-width:980px}}a{{color:var(--teal)}}.eyebrow{{text-transform:uppercase;letter-spacing:.1em;color:var(--teal);font-weight:700;font-size:.82rem}}.lead{{font-size:1.12rem}}.callout{{border-left:5px solid var(--orange);background:#fff7ee;padding:16px 22px;border-radius:5px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}}.card{{background:white;border-top:3px solid var(--teal);padding:17px;border-radius:6px}}.card strong{{font-size:1.65rem;display:block;color:var(--navy)}}.small,.limits{{font-size:.9rem;color:var(--muted)}}.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}}.step{{padding:16px;background:#e6f0ef;border-radius:7px;border:1px solid #bcd4d2}}.step b{{display:block;color:var(--navy)}}table{{border-collapse:collapse;width:100%;background:white;font-size:.93rem;margin:20px 0}}th,td{{padding:10px 12px;border-bottom:1px solid #dfe7ec;text-align:left;vertical-align:top}}thead th{{background:var(--navy);color:white}}.scroll{{overflow-x:auto}}code,.formula{{font-family:ui-monospace,Consolas,monospace}}code{{overflow-wrap:anywhere}}.formula{{background:#edf2f6;border-radius:5px;padding:12px;font-size:.91rem;white-space:normal}}figure{{margin:32px 0;padding:22px;background:white;border:1px solid #dfe7ec;border-radius:9px;break-inside:avoid}}img{{display:block;width:100%;height:auto}}figcaption p{{margin:10px 0}}li{{margin:10px 0}}nav{{display:flex;flex-wrap:wrap;gap:18px;margin:24px 0}}footer{{border-top:1px solid #ccd9df;margin-top:45px;padding-top:20px}}@media(max-width:720px){{main{{padding:24px 16px}}.cards,.flow{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:1.9rem}}figure{{padding:10px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}nav{{display:none}}h2{{break-after:avoid}}figure{{break-inside:avoid}}a{{color:inherit}}}}
    </style></head><body><main><div class='eyebrow'>Frozen historical study · daily data · incremental spend $0</div><h1>What the crypto graphs show, and what the forecast has not established</h1>
    <p class='lead'>This study turns the daily returns of ten fixed cryptocurrencies into correlation graphs. The main finding is that positive-graph λ₂ closely follows simpler measures of co-movement. A small forecast improvement appears in one historical test, but the models overpredict events and the test contains only six event labels, in two clusters.</p>
    <div class='callout'><b>Two important negative results:</b> the fitted Gamma distribution has a large final-test CDF discrepancy of 0.609; it does not establish a stable probability law. The model with spectral features predicts an average event probability of 6.02% while the final-test event frequency is 1.49%. Its small conditional score gain does not establish a general forecasting advantage.</div>
    <nav><a href='#pathway'>Data pathway</a><a href='#results'>Findings</a><a href='#distribution'>Distribution fit</a><a href='#forecast'>Forecast test</a><a href='#graphs'>Eight graphs</a><a href='#evidence'>Evidence and sources</a></nav>
    <div class='cards'><div class='card'><strong>10</strong>fixed coin identities</div><div class='card'><strong>2,099</strong>daily study observations</div><div class='card'><strong>1,694 / 2,040</strong>usable 60-return windows</div><div class='card'><strong>67,760</strong>saved eigenvalues</div></div>
    <h2 id='pathway'>From source observations to a graph</h2>
    <p>The cohort is BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LTC, LINK and DOT on Binance spot, quoted in USDT. USDT is the quote currency; these are not independently observed US-dollar prices. The requested daily interval-start dates are January 1, 2021 through September 30, 2026. The stored closes have UTC <b>bar-end</b> labels from January 2, 2021 through October 1, 2026, plus a January 1 seed close used only to calculate the first return.</p>
    <div class='flow' aria-label='Research workflow, positions are not measured'><div class='step'><b>1 · Preserve</b>Archive and checksum → immutable source snapshot</div><div class='step'><b>2 · Prepare</b>Verified identities and daily intervals → arithmetic returns</div><div class='step'><b>3 · Select windows</b>60 returns → all ten complete and nonconstant</div><div class='step'><b>4 · Measure</b>Pearson correlation → four graph operators → full spectra</div><div class='step'><b>5 · Challenge</b>Simpler indicators, chronological distribution and forecast tests</div><div class='step'><b>6 · Explain</b>Graphs, numerical checks, findings and explicit limitations</div></div><p class='small'>This workflow is a non-quantitative diagram: placement shows the calculation order, not measured distance or time.</p>
    <p>A daily close is retained only when all 24 expected hourly child bars exist and have valid closes. Missing data stay missing: no zero fill, forward fill or interpolation. There are <b>79 missing price cells on eight incomplete daily rows</b>. A missing price can invalidate two returns and many overlapping windows; consequently 346 of the 2,040 scheduled windows are excluded. The first usable fixed-cohort endpoint is June 26, 2021. The original source archives and earlier project panels are preserved.</p>
    <p>Each return is <code>rₜ = Pₜ / Pₜ₋₁ − 1</code>. A price move from 100 to 110 gives 0.10, or 10%. A return is missing if either price is missing, nonfinite or nonpositive, or if the timestamps do not differ by exactly one day. Every research window requires all ten series to have 60 finite returns and sample standard deviation greater than 10⁻¹². This deliberately fixed cohort differs from the interactive explorer, which may drop incomplete variables within a selected window.</p>
    <p>UTC bar end is the time the measurement interval finishes. It is the earliest nominal close availability; provider delivery and publication latency were not measured. The historical forecast treats completed bar-end observations as available, so it remains a provisional test rather than a demonstrated live system. The cohort was selected using current identities and availability: it is not a point-in-time historical top-ten or top-100 universe.</p>
    <h2 id='results'>What is measured, and what was found</h2><p>For each usable window, Pearson correlation C compares the same 60 dates for all ten assets. The diagonal is excluded from graph edges. Positive weights keep positive correlations; negative weights keep the magnitude of negative correlations; shifted weights map an off-diagonal correlation from [−1, 1] into [0, 1]. For these three nonnegative graphs, <code>L = D − W</code>, where D contains each node’s total edge weight. The second-smallest eigenvalue λ₂ measures a form of graph connectivity for that chosen graph.</p>
    <p>The signed graph keeps each correlation’s sign and uses <code>Lₛ = diag(Σⱼ |Sᵢⱼ|) − S</code>, with S equal to C off the diagonal. Its lowest eigenvalue μ₁ addresses a different signed balance question; it is not ordinary connectivity or a Fiedler eigenvalue. A signed or unsigned eigenvalue magnitude cannot be compared as if every operator were the same risk scale.</p>
    <p><b>Four operators do not imply four independent signals.</b> In 1,691 of the 1,694 usable windows, every off-diagonal correlation is nonnegative. There the signed operator equals the positive operator. Also <code>L_shifted = ½L_positive + ½(nI − J)</code>, where J is the all-ones matrix. On every nonconstant mode, shifted eigenvalues are <code>n/2 + λ_positive/2</code> and the eigenvectors are identical. For ten assets this gives <code>λ₂_shifted = 5 + λ₂_positive/2</code>; the saved results satisfy that relation within 1.95 × 10⁻¹⁴. Much of the apparent agreement in the mode plots is an algebraic consequence of the definitions.</p>
    <ul>{findings}</ul><p>The normalized nonnegative Laplacian is calculated only with positive degrees and a connected graph. The scale diagnostic is <code>q = λ₂ / (n × mean off-diagonal weight)</code> when the denominator is positive. Fiedler localization requires a connected graph and a numerically separated second eigenvalue; signed localization instead uses a separated lowest mode. For a unit-length vector, <code>IPR = Σᵢ vᵢ⁴</code>; 0.1 means fully spread across ten nodes and values nearer 1 mean stronger concentration. Turnover <code>1 − |vₜᵀvₜ₋₁|</code> is shown only for adjacent eligible windows with identical ordered identities.</p>
    <h2 id='distribution'>Does λ₂ have an identifiable probability distribution?</h2>
    <p>Gamma and lognormal are two candidate descriptions of the positive-graph λ₂ values, not the distribution of coin returns. The first 1,016 usable windows fit the parameters with location fixed at zero; the next 339 choose the family by average log density; the final 339 evaluate it. Empirical zero mass is treated separately and is zero in these samples. The distribution split is separate from the forecast split below.</p>
    <div class='scroll'><table><thead><tr><th>Candidate</th><th>Shape</th><th>Scale</th><th>Validation log density</th><th>Final-test log density</th><th>Test CDF discrepancy</th></tr></thead><tbody>{fitrows}</tbody></table></div>
    <p>Gamma scores better on validation, but its training-fit cumulative curve differs from the final-test empirical curve by as much as <b>0.609</b> on a probability scale from 0 to 1. That is substantial chronological drift. Neither candidate establishes a stable law. These families also put some mass above the fixed ten-node graph’s upper bound λ₂ ≤ 10: approximately 0.31% for Gamma and 0.77% for lognormal. The overlapping windows are dependent, so no IID Kolmogorov–Smirnov p-value is reported.</p>
    <h2 id='forecast'>A provisional test of a future change in correlation pattern</h2>
    <p>The prediction question is specific: will the correlation pattern over the next 20 observations differ strongly from the preceding 60-return pattern? For the 45 distinct pairs, calculate the root mean square of <code>C_future20 − C_past60</code>. An event occurs when this exceeds <b>{forecast['event']['threshold']:.6f}</b>, the 90th percentile of fully realized training labels. The future matrix is a target used only for historical scoring, never an input feature.</p>
    <p>The first 60%, next 20% and final 20% of scheduled label-eligible dates form training, validation and test periods. Twenty endpoints are purged before each later period, ensuring all training labels finish before validation begins and all validation labels finish before test begins. Incomplete fixed-cohort windows and labels are removed after the chronological boundaries are declared.</p>
    <div class='scroll'><table><thead><tr><th>Period</th><th>Eligible dates</th><th>Event labels</th><th>Event frequency</th><th>As-of dates, UTC</th><th>Last label interval ends</th></tr></thead><tbody>{splitrows}</tbody></table></div>
    <p>The constant model always uses the training event frequency. The indicator model uses mean correlation, mean absolute correlation, negative-pair fraction, positive correlation density at 0.3, largest PCA share, minimum positive degree, Bitcoin and basket 60-day volatility, Bitcoin 20-day momentum, and current Bitcoin/basket daily returns. Volatility is the daily sample standard deviation multiplied by √365. The spectral model adds the defined eigenvalue, q, normalized-eigenvalue, localization and turnover features.</p>
    <p>Logistic regression converts a weighted sum of inputs to a fraction between 0 and 1. Its objective is <code>Σ[log(1+exp(η)) − yη] + Σⱼ&gt;₀ βⱼ²/(2C)</code>; the intercept is unpenalized. Training data alone determine feature medians, missingness flags, centering and scaling. This imputation concerns undefined model features, not missing source prices or returns. C is selected from 0.01, 0.1, 1 and 10 using validation Brier loss; both selected models use C = 0.1. Models are not refit on validation or test.</p>
    <div class='scroll'><table><thead><tr><th>Frozen model</th><th>Mean test probability</th><th>Test Brier loss ↓</th><th>Test log loss ↓</th></tr></thead><tbody>{rows}</tbody></table></div>
    <p>Brier loss is the mean squared difference between probability p and the realized 0/1 outcome. Log loss penalizes confident wrong probabilities more heavily. Lower scores are better. The spectral model’s Brier change versus simple indicators is −0.000373, with a conditional 60-observation block-resampling 95% interval of [−0.000698, −0.000083]. The interval describes resampled test loss differences with the fitted models fixed; it does not include training or tuning uncertainty or prove stationarity.</p>
    <div class='callout'><b>The practical limit is visible:</b> only six of 404 test dates are event labels (1.49%), clustered at {escape(clusters)}. These are not six independent market episodes. All models overpredict the test event frequency; even the spectral model averages 6.02%. The observed small gain is conditional evidence in one split, not a reliable general probability forecast or trading result.</div>
    <p>For context, an always-zero forecast has an ex-post Brier loss of {forecast['splits']['test']['event_rate']:.6f}, lower than all three models here, while assigning zero probability to the six realized events gives infinite un-clipped log loss. This is a descriptive benchmark revealed by the test outcomes, not an additional model selected for deployment. Every twentieth test endpoint is also scored for all 20 possible offsets: each phase has only 20–21 cases and at most two event labels. Its forward targets do not overlap, but the backward predictors still do.</p>
    <h2 id='graphs'>Eight graphs with the equations and axes explained</h2>{''.join(figures)}
    <h2 id='evidence'>What was checked, and how to reproduce it</h2><ul>{checkitems}</ul>
    <p>Maximum saved-spectrum trace discrepancy: <code>{receipt['maximum_spectrum_trace_error']:.3g}</code>. Computation uses Python {receipt['environment']['python']}, NumPy {receipt['environment']['numpy']}, pandas {receipt['environment']['pandas']}, SciPy {receipt['environment']['scipy']} and Matplotlib {receipt['environment']['matplotlib']}. Source price SHA-256: <code>{escape(hashvalue)}</code>.</p>
    <p>Run <code>python crypto_mvp_research.py --self-test</code> and <code>python crypto_mvp_research.py</code> from the project folder. Inspect <code>crypto_mvp_data.py</code> for source loading, <code>crypto_mvp_research.py</code> for these calculations, and <code>output/crypto_mvp/research/</code> for feature tables, spectra, frozen-model parameters, predictions, local numerical fixture and checks. The hosted research directory contains analytical derivatives only. This report embeds every figure and requires no external scripts or fonts.</p>
    <p>Checksum preservation, identity mapping, strict missing-data handling and numerical tests do not verify every historical observation, remove current-cohort bias, make overlapping windows independent, or establish causality. The single-venue USDT history, unmeasured publication latency and limited event count remain substantive limits. No new observations or paid tools were acquired for this study.</p>
    <h3>Primary methods and attribution</h3><ol>{references}</ol><footer><p class='small'>Source: Binance Vision. Analytical derivatives attributed under the recorded <a href='https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md'>provider terms</a> and CC BY-NC-SA 4.0 noncommercial derivative conditions. Raw per-coin prices and returns are retained locally and are not embedded in this report. Generated {escape(index['created_at_utc'])} from the frozen snapshot. Incremental spend: $0.</p></footer></main></body></html>"""
    target=ROOT/"output/crypto_mvp/Findings.html"
    target.write_text(document,encoding="utf-8")
    return target


def run_research():
    from crypto_mvp_data import load_crypto_panel
    OUT.mkdir(parents=True,exist_ok=True); SITE.mkdir(parents=True,exist_ok=True)
    prices,metadata=load_crypto_panel("1d",start="2021-01-01",end="2026-09-30",include_preceding=True)
    export_fixture(prices,metadata)
    frame,returns,spectra=compute_features(prices)
    frame.to_csv(OUT/"features.csv",index=False,float_format="%.17g")
    forecast,masks=chronological_forecast(frame); distribution,_=distribution_study(frame)
    for name,obj in [("forecast",forecast),("distribution",distribution),("spectra",spectra),("timeseries",{"rows":frame.to_dict("records"),"source":public_metadata(metadata)})]:
        write_json(OUT/(name+".json"),obj); shutil.copy2(OUT/(name+".json"),SITE/(name+".json"))
    figures,associations=make_figures(prices,frame,spectra,distribution,forecast)
    valid=frame.loc[frame.status.eq("ok")]
    uncertainty=next(x for x in forecast["uncertainty"]["intervals"] if x["first"]=="indicators_spectral" and x["second"]=="indicators" and x["block_length"]==60)
    difference=uncertainty["mean_difference"]
    advantage=uncertainty["percentile_95"][1]<0 and forecast["models"]["indicators_spectral"]["test"]["brier"]<forecast["models"]["constant"]["test"]["brier"]
    spectral_test=forecast["models"]["indicators_spectral"]["test"]
    summary={"findings":[
        f"Fixed ten-coin cohort: {len(prices)-1:,} daily study observations; {int(prices.iloc[1:].isna().sum().sum())} missing price cells. {len(valid):,}/{len(frame):,} scheduled 60-return windows retain all ten identities.",
        f"Positive λ₂ associates with minimum positive degree at {associations['positive_min_degree']:.3f}, versus mean Pearson correlation at {associations['mean_pearson']:.3f}; these are descriptive overlapping-window associations.",
        f"Negative graph is disconnected in {valid.negative_support_components.gt(1).mean():.1%} of usable windows; undefined Fiedler modes remain null.",
        f"All pair correlations are nonnegative in {valid.negative_fraction.eq(0.).sum():,}/{len(valid):,} usable windows: signed equals positive, while shifted has the same nonconstant eigenvectors and λ₂_shifted = 5 + λ₂_positive/2. These largely redundant operators are not independent signals.",
        f"Validation selects {distribution['selected_by_validation']}, but the Gamma training-fit CDF differs from the final-test empirical CDF by as much as {distribution['models']['gamma']['evaluation']['test']['cdf_max_discrepancy_descriptive']:.3f}; no stable universal distribution is established.",
        f"Adding spectral predictors changes test Brier versus simple indicators by {difference:+.6f}; conditional 60-observation block 95% interval [{uncertainty['percentile_95'][0]:+.6f}, {uncertainty['percentile_95'][1]:+.6f}].",
        f"The spectral model overpredicts: mean test probability {spectral_test['mean_probability']:.2%}, realized frequency {spectral_test['event_rate']:.2%}; only {spectral_test['events']} test event labels occur in two three-day clusters.",
        ("A small conditional score improvement is observed in one historical split; six clustered event labels and strong overprediction leave general forecasting advantage unestablished." if advantage else "No robust incremental forecast advantage is demonstrated by this provisional test.")],
        "incremental_advantage_demonstrated":False,"conditional_improvement_observed":bool(advantage),
        "best_observed_test_model":min(forecast["models"],key=lambda name:forecast["models"][name]["test"]["brier"]),
        "claim":"Historical descriptive and provisional holdout research; no live forecast, causality, executable profit or universal distribution established."}
    index={"schema_version":1,"status":"complete","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "title":"Fixed-cohort crypto: graphs, distributions and a provisional forecast test","interval":"1d","window":WINDOW,"step":1,
        "scope":"Fixed ten-coin complete-cohort research differs from interactive per-window dropping of incomplete variables. Current selection is not historical top-100 membership.",
        "universe":metadata["universe"],"symbols":SYMBOLS,"provenance":public_metadata(metadata),
        "counts":{"daily_price_rows_excluding_seed":len(prices)-1,"daily_returns":len(returns),"scheduled_windows":len(frame),"usable_fixed_cohort_windows":len(valid),
                  "excluded_windows":len(frame)-len(valid),"missing_price_cells":int(prices.iloc[1:].isna().sum().sum()),"incomplete_daily_price_rows":int(prices.iloc[1:].isna().any(axis=1).sum()),"operators":4,"saved_eigenvalues":len(spectra)*len(SYMBOLS)},
        "graph_definitions":{"positive":"W=max(C,0), diagonal0; L=D-W","negative":"W=max(-C,0), diagonal0; ordinary L=D-W","signed":"S=C diagonal0; L_s=diag(sum(abs(S)))-S; lowest eigenvalue μ1; no ordinary lambda2/q/Fiedler claim","shifted":"W=(C+1)/2 off diagonal, diagonal0; L=D-W"},
        "mode_policy":"Unsigned connected and simple second eigenvalue: Fiedler/IPR. Signed connected absolute support and simple lowest eigenvalue: lowest-mode/IPR. Turnover only adjacent eligible windows with identical fixed ordered identities; sign invariant1-|dot|.",
        "timeseries_path":"research/timeseries.json","spectra_path":"research/spectra.json","distribution_path":"research/distribution.json","forecast_path":"research/forecast.json","fixture_path":"research/python_fixture.json","fixture_policy":"Hosted fixture contains derived common correlation, graph matrices and spectra only; full raw-price/return numerical fixture is local only.",
        "feature_fields":frame.columns.tolist(),"distribution":distribution,"forecast":{key:value for key,value in forecast.items() if key!="predictions"},
        "figures":figures,"summary":summary,"references":REFERENCES,
        "limitations":["Single Binance spot venue, USDT quote currency, current fixed cohort and unmeasured publication latency.","Source checksum and identity checks do not independently verify every price or historical contract event.","Overlapping windows/forward labels and regime drift limit inference; one chronological test is not repeated external validation.","Source prices are never filled. Model-only undefined predictors use training-only medians and missingness flags, disclosed separately.","No purchase, credentials, order execution or new feed acquisition."],
        "rights":{"attribution":"Binance Vision","license":"CC BY-NC-SA 4.0 for noncommercial derivatives under recorded dataset terms","publication_policy":"Analytical derivatives only; no raw per-coin price or return feed hosted.","url":"https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md"},"spend_usd":0,
        "reproduce":"python crypto_mvp_research.py; python crypto_mvp_research.py --self-test"}
    write_json(OUT/"index.json",index); write_json(SITE/"index.json",index)
    receipt=validate_research(frame,returns,spectra,forecast,masks,distribution)
    write_findings_html(index,receipt)
    print(json.dumps(serial({"counts":index["counts"],"summary":summary,"test":{name:item["test"] for name,item in forecast["models"].items()}}),indent=2))
    return index


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test()
    else: run_research()
