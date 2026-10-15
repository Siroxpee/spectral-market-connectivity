"""Exploratory fixed-cohort connectivity-trajectory forecasts; frozen protocol.

All transformations, graphs, ridge fits, calibration, grading and figures are
Python. No source acquisition, filling, trading or fresh-holdout claim.
"""
from __future__ import annotations
import os
for _thread in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread] = "1"
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crypto_core import graph_measurements, prepare, common_complete

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/crypto_trajectory"
SITE=ROOT/"crypto_mvp/dist/trajectory"
SYMBOLS=[x+"USDT" for x in ("BTC","ETH","BNB","XRP","ADA","SOL","DOGE","LTC","LINK","DOT")]
WINDOWS=(20,40,60,90,120)
H=20
START=max(WINDOWS)-1+59
ALPHAS=(.1,1.,10.,100.)
VARIANTS=("current","trajectory20","trajectory60")
MODEL_NAMES=("persistence",)+VARIANTS
SEED=20261015
PROTOCOL_HASH="94db998ced2f69b1c064838bfbaaa26050f735d568e38acfb697f1322c4fbc1c"
LABELS={"persistence":"Zero-change baseline","current":"Current + known window parts","trajectory20":"Add 20-score trajectory summaries","trajectory60":"Add 60-score trajectory summaries"}
COLORS={"persistence":"#6c7b83","current":"#17324D","trajectory20":"#007F86","trajectory60":"#D57B27"}


def serial(value):
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [serial(x) for x in value]
    if isinstance(value,np.generic): return serial(value.item())
    if isinstance(value,(pd.Timestamp,datetime)): return value.isoformat()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(serial(value),ensure_ascii=False,allow_nan=False,separators=(",",":")),encoding="utf-8")
    temporary.replace(path)


def verify_protocol():
    path=OUT/"protocol.json"
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=PROTOCOL_HASH: raise ValueError("Frozen protocol bytes changed; stop rather than silently changing the study")
    return json.loads(path.read_text(encoding="utf-8"))


def graph_summary(array):
    array=np.asarray(array,dtype=float)
    if not common_complete(array): return None
    c=np.corrcoef(array,rowvar=False)
    graph=graph_measurements(c,"positive")
    basket=array.mean(axis=1)
    return {"lambda2":float(graph["lambda2"]),"min_degree":float(np.min(graph["degrees"])),
            "mean_c":float(c[np.triu_indices(len(c),1)].mean()),
            "btc_vol":float(array[:,0].std(ddof=1)*np.sqrt(365)),
            "basket_vol":float(basket.std(ddof=1)*np.sqrt(365))}


def build_features(prices):
    """Current/past-only features and separately labeled future targets."""
    if list(prices.columns)!=SYMBOLS or prices.index.tz is None: raise ValueError("Fixed ten ordered identities and timezone-aware UTC labels required")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing: raise ValueError("Duplicate/unordered dates")
    prepared=prepare(prices.to_numpy(),[x.isoformat() for x in prices.index],86400)
    changes=np.asarray(prepared["values"],dtype=float)
    dates=pd.DatetimeIndex(prepared["dates"])
    assert len(changes)==len(prices)-1==len(dates)
    result={}
    for width in WINDOWS:
        rows=[]
        for position in range(len(changes)):
            row={"position":position,"date":dates[position].isoformat(),"window":width,"current_lambda2":np.nan}
            if position>=width-1:
                window=changes[position-width+1:position+1]
                summary=graph_summary(window)
                if summary is not None:
                    row.update({"current_"+k:v for k,v in summary.items()})
                    for label,part in (("retained",window[H:]),("outgoing",window[:H])):
                        component=graph_summary(part) if len(part)>=2 else None
                        for key in ("lambda2","mean_c","basket_vol"):
                            row[label+"_"+key]=component[key] if component else np.nan
            rows.append(row)
        frame=pd.DataFrame(rows)
        values=frame.current_lambda2.to_numpy()
        for count in (20,60):
            prefix="history"+str(count)+"_"
            for field in ("change","mean_minus_current","sd","slope"): frame[prefix+field]=np.nan
            centered=np.arange(count)-(count-1)/2
            for position in range(count-1,len(frame)):
                history=values[position-count+1:position+1]
                if not np.isfinite(history).all(): continue
                frame.loc[position,prefix+"change"]=float(history[-1]-history[0])
                frame.loc[position,prefix+"mean_minus_current"]=float(history.mean()-history[-1])
                frame.loc[position,prefix+"sd"]=float(history.std(ddof=1))
                frame.loc[position,prefix+"slope"]=float(centered@history/(centered@centered))
        frame["target_delta"]=frame.current_lambda2.shift(-H)-frame.current_lambda2
        frame["target_date"]=pd.Series(dates).shift(-H).map(lambda x:x.isoformat() if not pd.isna(x) else None)
        frame["eligible_predictors"]=np.isfinite(frame.current_lambda2) & np.isfinite(frame.history60_change)
        frame["eligible_label"]=frame.eligible_predictors & np.isfinite(frame.target_delta)
        result[width]=frame
    return result,changes,dates


def feature_fields(variant):
    current=["current_lambda2","current_min_degree","current_mean_c","current_btc_vol","current_basket_vol"]
    components=[part+"_"+key for part in ("retained","outgoing") for key in ("lambda2","mean_c","basket_vol")]
    if variant=="current": return current+components
    history="history"+("20" if variant=="trajectory20" else "60")+"_"
    return current+components+[history+key for key in ("change","mean_minus_current","sd","slope")]


def fit_ridge(x,y,alpha):
    """Training-only transforms and an unpenalized intercept."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if len(y)<30 or not np.isfinite(y).all(): raise ValueError("Insufficient/invalid training targets")
    all_missing=np.isnan(x).all(axis=0)
    medians=np.array([np.nanmedian(x[:,j]) if not all_missing[j] else 0. for j in range(x.shape[1])])
    missing=np.isnan(x)
    flags=np.flatnonzero(missing.any(axis=0)&~all_missing)
    expanded=np.column_stack([np.where(missing,medians,x),missing[:,flags].astype(float)])
    mean=expanded.mean(axis=0); scale=expanded.std(axis=0)
    active=scale>1e-12
    scale=np.where(active,scale,1.)
    z=(expanded[:,active]-mean[active])/scale[active]
    intercept=float(y.mean())
    beta=linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-intercept),assume_a="pos") if z.shape[1] else np.array([])
    return {"alpha":float(alpha),"medians":medians,"missing_flags":flags,"mean":mean,"scale":scale,"active":active,
            "intercept":intercept,"beta":beta,"training_n":len(y),"all_missing_columns":np.flatnonzero(all_missing)}


def predict_ridge(model,x):
    x=np.asarray(x,dtype=float); missing=np.isnan(x)
    expanded=np.column_stack([np.where(missing,model["medians"],x),missing[:,model["missing_flags"]].astype(float)])
    z=(expanded[:,model["active"]]-model["mean"][model["active"]])/model["scale"][model["active"]]
    return model["intercept"]+z@model["beta"]


def partitions(origin,common_labels):
    """All boundary choices depend on scheduled positions, never outcomes."""
    positions=np.arange(len(common_labels))
    available=origin-H-START
    tune=START+int(.6*available); calibration=START+int(.8*available)
    masks={"fit":common_labels&(positions<tune-H),
           "tune":common_labels&(positions>=tune)&(positions<calibration-H),
           "calibration":common_labels&(positions>=calibration)&(positions+H<origin),
           "matured":common_labels&(positions+H<origin)}
    if min(int(masks[x].sum()) for x in ("fit","tune","calibration"))<30: raise ValueError("Too few common complete cases for fit/tuning/calibration")
    assert (np.flatnonzero(masks["fit"])+H<tune).all()
    assert (np.flatnonzero(masks["tune"])+H<calibration).all()
    assert (np.flatnonzero(masks["calibration"])+H<origin).all()
    return masks,{"origin":origin,"tuning_start":tune,"calibration_start":calibration,
                  "counts":{name:int(mask.sum()) for name,mask in masks.items()},
                  "max_matured_label_position":int(np.flatnonzero(masks["matured"])[-1]+H)}


def tune_models(frames,origin,common_labels):
    masks,boundary=partitions(origin,common_labels)
    selected={}; scores=[]
    for width,frame in frames.items():
        target=frame.target_delta.to_numpy()
        persistence=float(np.abs(target[masks["tune"]]).mean())
        for variant in VARIANTS:
            fields=feature_fields(variant); x=frame[fields].to_numpy()
            candidates=[]
            for alpha in ALPHAS:
                model=fit_ridge(x[masks["fit"]],target[masks["fit"]],alpha)
                prediction=predict_ridge(model,x[masks["tune"]])
                mae=float(np.abs(prediction-target[masks["tune"]]).mean())
                candidates.append((mae,-alpha,alpha))
                scores.append({"window":width,"variant":variant,"alpha":alpha,"validation_mae":mae,
                               "persistence_validation_mae":persistence,"relative_mae":mae/persistence if persistence>0 else None})
            mae,_,alpha=min(candidates)
            selected[(width,variant)]={"alpha":alpha,"mae":mae,"persistence_mae":persistence,"relative_mae":mae/persistence if persistence>0 else float("inf")}
    winner=min(selected,key=lambda key:(selected[key]["relative_mae"],0 if key[0]==60 else 1,-key[0],VARIANTS.index(key[1]),-selected[key]["alpha"]))
    return selected,winner,scores,masks,boundary


def fit_and_calibrate(frame,variant,alpha,origin,common_labels,calibration_start):
    positions=np.arange(len(frame)); y=frame.target_delta.to_numpy()
    fields=feature_fields(variant) if variant!="persistence" else []
    x=frame[fields].to_numpy() if fields else None
    errors=[]; calibration_rows=[]; audits=[]
    for block in range(calibration_start,origin-H,H):
        training=common_labels&(positions+H<block)
        evaluation=common_labels&(positions>=block)&(positions<min(block+H,origin-H))
        if not evaluation.any(): continue
        if training.sum()<30: continue
        model=fit_ridge(x[training],y[training],alpha) if fields else None
        prediction=predict_ridge(model,x[evaluation]) if fields else np.zeros(evaluation.sum())
        error=y[evaluation]-prediction
        errors.extend(error.tolist())
        calibration_rows.extend(np.flatnonzero(evaluation).tolist())
        audits.append({"origin_position":block,"last_training_label_position":int(np.flatnonzero(training)[-1]+H),"n":int(evaluation.sum())})
    if len(errors)<30: raise ValueError("Too few honest rolling calibration errors")
    training=common_labels&(positions+H<origin)
    model=fit_ridge(x[training],y[training],alpha) if fields else None
    result={"model":model,"fields":fields,"errors":np.asarray(errors),"calibration_positions":calibration_rows,
            "calibration_audit":audits,"last_training_label_position":int(np.flatnonzero(training)[-1]+H),
            "training_n":int(training.sum()),"origin_position":origin}
    assert result["last_training_label_position"]<origin
    assert all(item["last_training_label_position"]<item["origin_position"] for item in audits)
    return result


def direction_labels(delta,band):
    delta=np.asarray(delta)
    return np.where(delta<-band,0,np.where(delta>band,2,1))


def probabilistic_projection(prediction,errors,band):
    """Location-shift empirical earlier-error distribution; no assumed law."""
    prediction=np.asarray(prediction,dtype=float).reshape(-1)
    errors=np.asarray(errors,dtype=float)
    labels=direction_labels(prediction[:,None]+errors[None,:],band)
    probabilities=np.column_stack([(labels==category).sum(axis=1)+1 for category in range(3)])/(len(errors)+3.)
    quantiles=np.quantile(errors,[.05,.95])
    return {"probabilities":probabilities,"lower":prediction+quantiles[0],"upper":prediction+quantiles[1],"residual_quantiles":quantiles}


def bootstrap_mean(values,block=60,replicates=2000,seed=SEED):
    values=np.asarray(values,dtype=float); n=len(values)
    if not n: return None
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(replicates,int(np.ceil(n/block))))
    indices=(starts[:,:,None]+np.arange(block)[None,None,:])%n
    samples=values[indices.reshape(replicates,-1)[:,:n]].mean(axis=1)
    return {"mean":float(values.mean()),"percentile_95":np.quantile(samples,[.025,.975]),"block_length":block,"replicates":replicates,"seed":seed}


def residual_block_quantiles(errors):
    """Sensitivity of residual 5th/95th percentiles to dependent resampling."""
    errors=np.asarray(errors); rng=np.random.default_rng(SEED)
    starts=rng.integers(0,len(errors),size=(1000,int(np.ceil(len(errors)/60))))
    indices=(starts[:,:,None]+np.arange(60)[None,None,:])%len(errors)
    sample=errors[indices.reshape(1000,-1)[:,:len(errors)]]
    quantiles=np.quantile(sample,[.05,.95],axis=1)
    return {"block":60,"replicates":1000,"seed":SEED,"residual_q05_percentile95":np.quantile(quantiles[0],[.025,.975]),
            "residual_q95_percentile95":np.quantile(quantiles[1],[.025,.975]),"note":"Sensitivity of training-time residual quantiles, not a confidence guarantee for a future outcome."}


def score_records(records,include_uncertainty=True):
    if not records: return None
    target=np.array([x["actual_delta"] for x in records]); predicted=np.array([x["predicted_delta"] for x in records])
    truth=np.array([x["direction_label"] for x in records]); probabilities=np.array([x["probabilities"] for x in records])
    onehot=np.eye(3)[truth]
    bins=[]
    for category,label in enumerate(("down","flat","up")):
        for lo,hi in zip(np.arange(0,1,.2),np.arange(.2,1.01,.2)):
            keep=(probabilities[:,category]>=lo)&(probabilities[:,category]<hi if hi<.999 else probabilities[:,category]<=1)
            if keep.any(): bins.append({"class":label,"lower":float(lo),"upper":float(hi),"n":int(keep.sum()),
                "mean_prediction":float(probabilities[keep,category].mean()),"observed_frequency":float((truth[keep]==category).mean())})
    covered=np.array([x["interval_lower"]<=x["actual_delta"]<=x["interval_upper"] for x in records])
    return {"n":len(records),"mae":float(np.abs(predicted-target).mean()),"rmse":float(np.sqrt(np.mean((predicted-target)**2))),
            "bias":float((predicted-target).mean()),"three_class_brier":float(np.sum((probabilities-onehot)**2,axis=1).mean()),
            "nominal90_interval_coverage":float(covered.mean()),"interval_mean_width":float(np.mean([x["interval_upper"]-x["interval_lower"] for x in records])),
            "direction_counts":{label:int((truth==i).sum()) for i,label in enumerate(("down","flat","up"))},
            "mean_probabilities":dict(zip(("down","flat","up"),probabilities.mean(axis=0).tolist())),"calibration":bins,
            "mae_block60":bootstrap_mean(np.abs(predicted-target)) if include_uncertainty else None}


def run_walkforward(frames,dates):
    n=len(dates); position=np.arange(n)
    common_predictors=np.logical_and.reduce([f.eligible_predictors.to_numpy() for f in frames.values()])
    common_labels=np.logical_and.reduce([f.eligible_label.to_numpy() for f in frames.values()])&(position>=START)
    cutoff=np.flatnonzero(dates<=pd.Timestamp("2026-10-01T00:00:00Z"))[-1]
    stop=cutoff-H+1
    first_test=START+int(.8*(stop-START))
    initial_masks,_=partitions(first_test,common_labels)
    bands={width:.5*float(frame.loc[initial_masks["fit"],"target_delta"].std(ddof=1)) for width,frame in frames.items()}
    records=[]; search=[]; boundaries=[]
    for origin in range(first_test,stop,H):
        chosen,winner,grid,masks,boundary=tune_models(frames,origin,common_labels)
        boundary["origin_date"]=dates[origin].isoformat(); boundary["selected_window"]=winner[0]; boundary["selected_variant"]=winner[1]
        boundary["inner_band_fit_n"]=int(initial_masks["fit"].sum())
        search.extend({"origin":dates[origin].isoformat(),**row} for row in grid)
        evaluations=common_labels&(position>=origin)&(position<min(origin+H,stop))
        if not evaluations.any(): boundaries.append(boundary); continue
        tasks=[(60,name) for name in MODEL_NAMES]
        if winner not in tasks: tasks.append(winner)
        if (winner[0],"persistence") not in tasks:tasks.append((winner[0],"persistence"))
        bundles={}
        for width,name in tasks:
            alpha=chosen[(width,name)]["alpha"] if name!="persistence" else None
            bundle=fit_and_calibrate(frames[width],name,alpha,origin,common_labels,boundary["calibration_start"])
            bundles[(width,name)]=bundle
            x=frames[width].loc[evaluations,bundle["fields"]].to_numpy() if bundle["fields"] else None
            point=predict_ridge(bundle["model"],x) if name!="persistence" else np.zeros(evaluations.sum())
            projection=probabilistic_projection(point,bundle["errors"],bands[width])
            for j,i in enumerate(np.flatnonzero(evaluations)):
                target=float(frames[width].iloc[i].target_delta)
                records.append({"date":dates[i].isoformat(),"target_date":dates[i+H].isoformat(),"position":int(i),"origin":dates[origin].isoformat(),
                    "window":width,"model":name,"pipeline":"primary_W60" if width==60 else "secondary_selected_W",
                    "selected_by_inner_validation":bool((width,name)==winner),"actual_delta":target,"predicted_delta":float(point[j]),
                    "interval_lower":float(projection["lower"][j]),"interval_upper":float(projection["upper"][j]),
                    "direction_label":int(direction_labels(target,bands[width])),"material_band":bands[width],
                    "probabilities":projection["probabilities"][j],"alpha":alpha,"calibration_n":len(bundle["errors"]),
                    "last_training_label_date":dates[bundle["last_training_label_position"]].isoformat()})
        boundary["models"]={str(width)+"_"+name:{"window":width,"alpha":chosen[(width,name)]["alpha"] if name!="persistence" else None,
            "training_n":b["training_n"],"calibration_n":len(b["errors"]),"calibration_audit":b["calibration_audit"]} for (width,name),b in bundles.items()}
        boundaries.append(boundary)
        print(f"walk-forward origin {dates[origin].date()} evaluated {int(evaluations.sum())} dates",flush=True)
    primary={name:[x for x in records if x["window"]==60 and x["model"]==name] for name in MODEL_NAMES}
    common_dates=[x["date"] for x in primary["persistence"]]
    assert all([x["date"] for x in rows]==common_dates for rows in primary.values())
    metrics={name:score_records(rows) for name,rows in primary.items()}
    secondary=[x for x in records if x["selected_by_inner_validation"]]
    secondary_baseline=[next(y for y in records if y["position"]==x["position"] and y["window"]==x["window"] and y["model"]=="persistence") for x in secondary]
    secondary_result={"selected_model":score_records(secondary),"same_window_persistence":score_records(secondary_baseline),
        "relative_mae_vs_same_window_persistence":float(np.mean([abs(x["predicted_delta"]-x["actual_delta"]) for x in secondary])/np.mean([abs(x["actual_delta"]) for x in secondary])),
        "paired_loss_difference":bootstrap_mean(np.array([abs(x["predicted_delta"]-x["actual_delta"]) for x in secondary])-np.array([abs(x["actual_delta"]) for x in secondary])),
        "window_counts":{width:sum(x["window"]==width for x in secondary) for width in WINDOWS},
        "limitation":"Selection uses only earlier validation, but W changes target and class bands. Do not compare its raw MAE or probabilities directly with fixed-W60 results; this remains exploratory."}
    paired=[]
    for name in VARIANTS:
        a=np.array([abs(x["predicted_delta"]-x["actual_delta"]) for x in primary[name]])
        b=np.array([abs(x["actual_delta"]) for x in primary["persistence"]])
        paired.append({"first":name,"second":"persistence",**bootstrap_mean(a-b)})
    for first,second in (("trajectory20","current"),("trajectory60","current")):
        a=np.array([abs(x["predicted_delta"]-x["actual_delta"]) for x in primary[first]])
        b=np.array([abs(x["predicted_delta"]-x["actual_delta"]) for x in primary[second]])
        paired.append({"first":first,"second":second,**bootstrap_mean(a-b)})
    sensitivity=[]
    for offset in range(H):
        selected={name:[x for x in rows if (x["position"]-first_test)%H==offset] for name,rows in primary.items()}
        sensitivity.append({"scheduled_offset":offset,"scores":{name:score_records(rows,False) for name,rows in selected.items()},
            "uncertainty_note":"No60-observation block interval is computed on the much smaller stride20 subset; all phase counts/scores are descriptive."})
    context={"common_predictors":common_predictors,"common_labels":common_labels,"bands":bands,"first_test_position":first_test,
             "retrospective_stop":stop,"retrospective_cutoff_position":int(cutoff)}
    result={"status":"EXPLORATORY retrospective expanding walk-forward; not fresh holdout",
        "primary_window":60,"horizon":H,"metrics":metrics,"paired_loss_differences":paired,"secondary_window_selection":secondary_result,
        "nonoverlap_sensitivity":sensitivity,"boundaries":boundaries,"search":search,
        "material_bands":bands,"first_asof":common_dates[0],"last_asof":common_dates[-1],
        "outer_scheduled_dates":stop-first_test,"outer_common_eligible_dates":len(common_dates),
        "earliest_evaluation_boundary":dates[first_test].isoformat(),"latest_label_date":dates[cutoff].isoformat(),
        "target_overlap":"With W60/H20 the target graph retains40 current returns; the next20 innovations are new. Even stride20 graph endpoints remain overlapping measurements.",
        "probability_method":"Earlier rolling calibration residual distribution shifted by each point prediction; Laplace one-count smoothing per class, fixed training-defined bands. No claimed conditional coverage guarantee."}
    return result,records,context


def latest_projection(frames,dates,context):
    common=context["common_labels"]; predictors=context["common_predictors"]
    origin=int(np.flatnonzero(predictors)[-1])
    chosen,winner,grid,masks,boundary=tune_models(frames,origin,common)
    rows=[]
    tasks=[(60,name) for name in MODEL_NAMES]
    if winner not in tasks: tasks.append(winner)
    for width,name in tasks:
        alpha=chosen[(width,name)]["alpha"] if name!="persistence" else None
        bundle=fit_and_calibrate(frames[width],name,alpha,origin,common,boundary["calibration_start"])
        frame=frames[width]
        point=float(predict_ridge(bundle["model"],frame.loc[[origin],bundle["fields"]].to_numpy())[0]) if bundle["model"] is not None else 0.
        projection=probabilistic_projection([point],bundle["errors"],context["bands"][width])
        score=float(frame.iloc[origin].current_lambda2)
        rows.append({"window":width,"model":name,"label":LABELS[name],"predicted_delta":point,"current_lambda2":score,
            "projected_lambda2":score+point,"interval_lower":projection["lower"][0],"interval_upper":projection["upper"][0],
            "projected_lambda2_interval":[score+projection["lower"][0],score+projection["upper"][0]],
            "physical_delta_bounds":[-score,len(SYMBOLS)-score],"interval_policy":"Unconstrained linear residual band; not clipped to physical graph support.",
            "probabilities":dict(zip(("down","flat","up"),projection["probabilities"][0])),"material_band":context["bands"][width],
            "selected_by_inner_validation":bool((width,name)==winner),"alpha":alpha,"fields":bundle["fields"],
            "training_n":bundle["training_n"],"calibration_n":len(bundle["errors"]),"last_training_label_date":dates[bundle["last_training_label_position"]].isoformat(),
            "residual_quantiles_05_95":projection["residual_quantiles"],"residual_quantile_block60_sensitivity":residual_block_quantiles(bundle["errors"]),
            "fit":bundle["model"],"calibration_audit":bundle["calibration_audit"]})
    return {"status":"PENDING FUTURE TARGET; NOT SCORED","asof":dates[origin].isoformat(),"observed_bar_end":dates[origin].isoformat(),
        "issued_at_utc":datetime.now(timezone.utc).isoformat(),"information_timing":"Observed bar end is not issuance time. Latest projection is generated now from the frozen data; retrospective tests assume nominal bar-end observations and unmeasured publication latency.",
        "nominal_target_interval_end":(dates[origin]+pd.Timedelta(days=H)).isoformat(),"horizon":H,
        "actual_future_delta":None,"actual_future_direction":None,"primary_window":60,
        "probability_reliability":"Not established. Retrospective nominal90% bands undercovered; direction probabilities are illustrative research estimates, not dependable warnings.",
        "selected_window":winner[0],"selected_variant":winner[1],"models":rows,"boundary":boundary,
        "note":"These are connectivity-change projections, not coin-price forecasts. Only labels matured strictly before as-of enter fitting/tuning/calibration. Future endpoint may be ineligible if source observations are missing."}


def freeze_projection(latest,source):
    """Keep the issued forecast immutable for a later, genuinely future score."""
    identity={"protocol_sha256":PROTOCOL_HASH,"source_prices_sha256":source.get("prices_sha256"),
              "observed_bar_end":latest["observed_bar_end"],"target":latest["nominal_target_interval_end"],"models":latest["models"]}
    digest=hashlib.sha256(json.dumps(serial(identity),sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()
    target=OUT/"issued"/digest/"forecast.json"
    if target.exists():
        prior=json.loads(target.read_text(encoding="utf-8"))
        latest["issued_at_utc"]=prior["issued_at_utc"]
    latest["frozen_forecast_key"]=digest
    latest["forecast_freeze_policy"]="Immutable local issued forecast; later target must be scored against these stored parameters/probabilities, not retrospectively replaced with a newer model."
    if not target.exists():
        write_json(target,{**latest,"source_prices_sha256":source.get("prices_sha256"),"protocol_sha256":PROTOCOL_HASH,
                           "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return digest


def make_figures(result,records,latest):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.18,"savefig.facecolor":"white"})
    figures=[]
    primary={name:[x for x in records if x["window"]==60 and x["model"]==name] for name in MODEL_NAMES}
    def save(fig,name,title,equation,method,axis,finding,limits):
        fig.suptitle(title,x=.04,ha="left",fontsize=14); fig.tight_layout(rect=(0,0,1,.95))
        path=OUT/"figures"/(name+".png"); path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=155,bbox_inches="tight");plt.close(fig)
        target=SITE/"figures"/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        figures.append({"path":"trajectory/figures/"+path.name,"title":title,"equation":equation,"method":method,"axis_explanation":axis,"finding":finding,"limitations":limits})
    fig,ax=plt.subplots(figsize=(12,5))
    chosen=primary["trajectory60"]; dates=pd.to_datetime([x["date"] for x in chosen],utc=True)
    ax.plot(dates,[x["actual_delta"] for x in chosen],color="#17324D",lw=1,label="Observed 20-day connectivity change")
    ax.plot(dates,[x["predicted_delta"] for x in chosen],color=COLORS["trajectory60"],lw=1.2,label="Ridge with 60-score trajectory summaries")
    ax.axhline(0,color=COLORS["persistence"],ls="--",label="Persistence: zero change")
    ax.set(xlabel="Historical forecast as-of (UTC daily interval end)",ylabel="20-observation change in positive λ₂\n(dimensionless)");ax.legend(fontsize=8,ncol=1)
    save(fig,"T01_trajectory","Can the earlier graph trajectory forecast its next change?","Δλ₂(t)=λ₂(t+20)−λ₂(t); W=60",
         "Expanding 20-date forecast blocks; each fit excludes labels not matured before its origin. Trajectory60 adds four summaries (endpoint change, mean relative to current, SD and slope), not60 individual coefficients. It is shown by predeclared choice, not chosen from test scores.",
         "Horizontal: historical as-of dates. Vertical: change in the graph connection score over the next 20 observations; above zero means higher future graph connectivity, not higher coin prices.",
         f"Trajectory60 MAE {result['metrics']['trajectory60']['mae']:.3f}; zero-change MAE {result['metrics']['persistence']['mae']:.3f}.",
         "All dates were already inspected in earlier research: exploratory, not a fresh holdout. Each future W60 graph shares40 known returns with the current graph.")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for i,name in enumerate(MODEL_NAMES):
        item=result["metrics"][name];lo,hi=item["mae_block60"]["percentile_95"]
        axes[0].bar(i,item["mae"],color=COLORS[name],width=.6)
        axes[0].errorbar(i,item["mae"],yerr=[[max(0,item["mae"]-lo)],[max(0,hi-item["mae"])]],fmt="none",color="#17324D",capsize=4)
    axes[0].set(xticks=range(4),xticklabels=["Zero\nchange","Current +\nknown parts","Prior 20\nscores","Prior 60\nscores"],xlabel="Forecast model (categorical)",ylabel="Mean absolute forecast error\n(λ₂-change units, dimensionless)")
    comparisons=[x for x in result["paired_loss_differences"] if x["second"]=="persistence"]
    for i,item in enumerate(comparisons):
        lo,hi=item["percentile_95"];axes[1].errorbar(i,item["mean"],yerr=[[max(0,item["mean"]-lo)],[max(0,hi-item["mean"])]],fmt="o",capsize=5,color=COLORS[item["first"]])
    axes[1].axhline(0,color="gray",ls=":")
    axes[1].set(xticks=range(3),xticklabels=["Current +\nknown parts","Prior 20\nscores","Prior 60\nscores"],xlabel="Model minus zero-change baseline (categorical)",ylabel="Paired mean absolute-error difference\n(λ₂-change units, dimensionless)")
    save(fig,"T02_errors","Compare error with a simple zero-change forecast","MAE=mean(|predicted Δ−actual Δ|)",
         "Identical complete W60 dates for every model. Error bars use 2,000 circular moving-block resamples of 60 observations, conditional on saved forecasts.",
         "Left: model categories versus absolute error; smaller is better. Right: model categories versus paired error minus persistence; negative favors the fitted model. Bars are approximate95% conditional intervals.",
         f"Common evaluated dates: {result['outer_common_eligible_dates']}. All four models use exactly the same targets.",
         "Resampling does not remove protocol-development bias, dependence or distribution drift; it is not fresh validation.")
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for axis,category in zip(axes.flat,("down","flat","up")):
        axis.plot([0,1],[0,1],ls=":",color="gray")
        for name in MODEL_NAMES:
            bins=[x for x in result["metrics"][name]["calibration"] if x["class"]==category]
            axis.plot([x["mean_prediction"] for x in bins],[x["observed_frequency"] for x in bins],marker="o",ms=4,label=LABELS[name],color=COLORS[name])
        axis.set(title=category.title()+" connectivity-change category",xlabel="Mean predicted category probability\n(fraction)",ylabel="Observed category frequency\n(fraction)",xlim=(0,1),ylim=(0,1))
    axes[0,0].legend(fontsize=7,loc="upper left")
    axis=axes[1,1]
    axis.bar(range(4),[result["metrics"][name]["nominal90_interval_coverage"] for name in MODEL_NAMES],color=[COLORS[n] for n in MODEL_NAMES])
    axis.axhline(.9,ls="--",color="gray",label="Nominal 90% reference");axis.legend(fontsize=8)
    axis.set(xticks=range(4),xticklabels=["Zero\nchange","Current +\nknown parts","Prior 20\nscores","Prior 60\nscores"],xlabel="Forecast model (categorical)",ylabel="Observed prediction-interval coverage\n(fraction)",ylim=(0,1))
    save(fig,"T03_calibration","Do the estimated direction chances match later outcomes?","down:Δ<−b; flat:|Δ|≤b; up:Δ>b; b=0.5×initial-training SD(Δ)",
         "Direction chances and nominal 90% bands use only earlier rolling calibration errors. Classwise fixed probability bins; no Gamma or other imposed continuous error law.",
         "First three panels: predicted fractions versus actual category frequencies; the diagonal indicates agreement. Fourth: model categories versus observed interval coverage; dashed line is 90%, not a guarantee.",
         f"Trajectory60 three-class Brier {result['metrics']['trajectory60']['three_class_brier']:.3f}; persistence {result['metrics']['persistence']['three_class_brier']:.3f}. Trajectory60 interval coverage {result['metrics']['trajectory60']['nominal90_interval_coverage']:.1%}.",
         "Bins can be sparse; see saved counts. Estimated graph changes include finite-sample correlation noise. Later probabilities may drift.")
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    rows=[x for x in latest["models"] if x["window"]==60]
    for i,row in enumerate(rows):
        p=row["predicted_delta"];lo=row["interval_lower"];hi=row["interval_upper"]
        axes[0].vlines(i,lo,hi,color=COLORS[row["model"]],lw=2)
        axes[0].plot(i,p,"o",color=COLORS[row["model"]])
        axes[0].plot([i-.08,i+.08],[lo,lo],color=COLORS[row["model"]]);axes[0].plot([i-.08,i+.08],[hi,hi],color=COLORS[row["model"]])
    axes[0].axhline(0,color="gray",ls=":")
    axes[0].set(xticks=range(4),xticklabels=["Zero\nchange","Current +\nknown parts","Prior 20\nscores","Prior 60\nscores"],xlabel="Newest pending forecast model (categorical)",ylabel="Projected 20-observation change in λ₂\n(dimensionless; nominal 90% residual band)")
    chosen=next(x for x in rows if x["model"]=="trajectory60")
    axes[1].bar(("Down","Flat","Up"),[chosen["probabilities"][k] for k in ("down","flat","up")],color=["#436dae","#6c7b83","#D57B27"])
    axes[1].set(xlabel="Connectivity-change category (categorical)",ylabel="Illustrative estimated probability\n(fraction; reliability not established)",ylim=(0,1))
    save(fig,"T04_latest","Newest projection: its future outcome is still unknown","projected λ₂(t+20)=current λ₂(t)+predicted Δ",
         "All model labels matured strictly before the as-of date; point forecasts are ridge/persistence, bands and class chances use earlier rolling residuals.",
         "Left: model categories versus projected change; zero means unchanged graph connectivity. Lines are nominal 90% residual bands. Right: trajectory60 category probabilities, not coin-price probabilities.",
         f"As-of {latest['asof'][:10]}, nominal target {latest['nominal_target_interval_end'][:10]}; trajectory60 Δ {chosen['predicted_delta']:+.3f}, band [{chosen['interval_lower']:+.3f}, {chosen['interval_upper']:+.3f}]. Outcome pending.",
         "The band is an empirical approximation, not promised coverage; unconstrained linear intervals can cross physical graph bounds. No future observation is invented or scored.")
    return figures


def self_test():
    verify_protocol()
    rng=np.random.default_rng(SEED)
    prices=pd.DataFrame(np.exp(rng.normal(0,.01,size=(300,10)).cumsum(axis=0)),columns=SYMBOLS,index=pd.date_range("2020-01-01",periods=300,tz="UTC"))
    frames,changes,dates=build_features(prices)
    t=220;summary=graph_summary(changes[t-59:t+1]);future=graph_summary(changes[t+H-59:t+H+1])
    np.testing.assert_allclose(frames[60].iloc[t].target_delta,future["lambda2"]-summary["lambda2"],atol=1e-12)
    np.testing.assert_allclose(frames[60].iloc[t].retained_lambda2,graph_summary(changes[t-39:t+1])["lambda2"],atol=1e-12)
    assert np.array_equal(changes[t-39:t+1],changes[t+H-59:t+1]),"W60/H20 known overlap must be40 returns"
    changed=prices.copy();changed.iloc[t+2:]*=np.exp(rng.normal(0,.03,size=(len(changed)-t-2,10)).cumsum(axis=0))
    altered,_,_=build_features(changed)
    np.testing.assert_allclose(frames[60].loc[t,feature_fields("trajectory60")].to_numpy(dtype=float),altered[60].loc[t,feature_fields("trajectory60")].to_numpy(dtype=float),atol=0,rtol=0,equal_nan=True)
    assert abs(frames[60].iloc[t].target_delta-altered[60].iloc[t].target_delta)>1e-6
    x=rng.normal(size=(100,4));x[:10,0]=np.nan;y=rng.normal(size=100)
    model=fit_ridge(x,y,10);frozen=model["mean"].copy();predict_ridge(model,np.full((4,4),999));assert np.array_equal(model["mean"],frozen)
    projection=probabilistic_projection([0.,2.],rng.normal(size=100),.5)
    np.testing.assert_allclose(projection["probabilities"].sum(axis=1),1.)
    assert (projection["probabilities"]>0).all()
    np.testing.assert_allclose(bootstrap_mean(np.full(100,.2),replicates=100)["percentile_95"],[.2,.2],atol=1e-15)
    masks,bound=partitions(1000,np.ones(1200,dtype=bool)&(np.arange(1200)>=START))
    assert np.flatnonzero(masks["matured"])[-1]+H<1000
    assert np.flatnonzero(masks["fit"])[-1]+H<bound["tuning_start"]
    print("PASS frozen protocol; future target alignment/known40 overlap; future perturbation cannot alter past features; train-only ridge; empirical class probabilities;60-block constant loss; horizon purges",flush=True)


def public_source(metadata):
    keys=("status","snapshot_key","prices_sha256","base_snapshot","acquired_at_utc","clock_at_request_utc","latest_complete_interval_start_day","latest_complete_interval_end","coverage","no_fill","source_url","timestamp_policy","notes")
    value={key:metadata[key] for key in keys if key in metadata}
    def redact(item):
        if isinstance(item,dict): return {k:redact(v) for k,v in item.items() if not any(part in k.lower() for part in ("path","credential","secret"))}
        if isinstance(item,list): return [redact(x) for x in item]
        if isinstance(item,str) and ("C:\\Users\\" in item or "C:/Users/" in item): return "local source retained outside hosted assets"
        return item
    return redact(value)


def finalize_index_copy(index):
    """Beginner text from already-computed results; no fitting or tuning."""
    n=index["evaluation_span"]["n"]
    index["plain_summary"]=(f"The models using summaries of the previous 20 or 60 graph windows did not beat a no-change forecast for 20-day-ahead connectivity of the 60-day graph on {n} matched historical cases. "
        "The estimated probabilities and prediction intervals are not yet dependable. These results are exploratory, not a fresh holdout test.")
    secondary=index["secondary_window_selection"]
    a=secondary["selected_model"]["mae"];b=secondary["same_window_persistence"]["mae"]
    lo,hi=secondary["paired_loss_difference"]["percentile_95"]
    counts=", ".join(f"{width}-return graph: {count} dates" for width,count in secondary["window_counts"].items() if count)
    direction="lower" if a<b else "higher"
    uncertainty="The conditional 60-observation block interval includes no improvement." if lo<=0<=hi else "This conditional interval does not include training/development uncertainty or establish fresh validation."
    index["plain_secondary_summary"]=(f"The window-selection rule chose {counts}. Its observed mean absolute error was {a:.3f}, versus {b:.3f} for the matched no-change forecast for each selected graph "
        f"({abs(1-a/b):.1%} {direction}; paired difference interval [{lo:+.3f}, {hi:+.3f}]). {uncertainty} "
        "Choosing a different graph window changes the target itself, so these errors must not be compared directly with the fixed 60-return graph's errors.")
    conclusion="The fixed 60-return graph's trajectory models did not beat the no-change forecast on observed mean absolute error; dependable forecast probabilities remain unestablished."
    if not index["findings"] or index["findings"][0]!=conclusion:index["findings"].insert(0,conclusion)
    return index


def run():
    protocol=verify_protocol()
    from crypto_trajectory_data import load_trajectory_panel
    prices,metadata=load_trajectory_panel()
    OUT.mkdir(parents=True,exist_ok=True);SITE.mkdir(parents=True,exist_ok=True)
    if not (OUT/"computation_start.json").exists():
        write_json(OUT/"computation_start.json",{"started_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":PROTOCOL_HASH,"source":public_source(metadata)})
    print("Computing fixed-cohort graph features from validated local data",flush=True)
    frames,changes,dates=build_features(prices)
    pd.concat(frames.values(),ignore_index=True).to_csv(OUT/"features.csv.gz",index=False,float_format="%.17g",compression="gzip")
    result,records,context=run_walkforward(frames,dates)
    latest=latest_projection(frames,dates,context)
    latest["source_acquired_at_utc"]=metadata.get("acquired_at_utc")
    freeze_projection(latest,metadata)
    figures=make_figures(result,records,latest)
    findings=[f"EXPLORATORY retrospective comparison: {result['outer_common_eligible_dates']} common complete W60 forecast dates from {result['first_asof'][:10]} to {result['last_asof'][:10]}."]
    for name in VARIANTS:
        item=result["metrics"][name];paired=next(x for x in result["paired_loss_differences"] if x["first"]==name and x["second"]=="persistence")
        findings.append(f"{LABELS[name]} MAE {item['mae']:.3f}; zero-change {result['metrics']['persistence']['mae']:.3f}; paired difference {paired['mean']:+.3f}, conditional60-block95% interval [{paired['percentile_95'][0]:+.3f}, {paired['percentile_95'][1]:+.3f}].")
    for name in ("trajectory20","trajectory60"):
        paired=next(x for x in result["paired_loss_differences"] if x["first"]==name and x["second"]=="current")
        findings.append(f"Adding prior{20 if name=='trajectory20' else 60} graph scores versus current + known window parts changes paired MAE by {paired['mean']:+.3f}; conditional60-block95% interval [{paired['percentile_95'][0]:+.3f}, {paired['percentile_95'][1]:+.3f}].")
    findings.append(f"Prior60 trajectory class Brier {result['metrics']['trajectory60']['three_class_brier']:.3f} versus calibrated persistence {result['metrics']['persistence']['three_class_brier']:.3f}; nominal90% interval covered {result['metrics']['trajectory60']['nominal90_interval_coverage']:.1%}.")
    findings.append("The fixed-W60 trajectory models do not beat zero-change on observed MAE; paired intervals cross zero, and the nominal90% bands undercovered. A dependable trajectory forecast or calibrated future probability is not established.")
    findings.append(f"Newest as-of {latest['asof'][:10]} target {latest['nominal_target_interval_end'][:10]} is pending; no future outcome or accuracy is assigned.")
    index={"schema_version":1,"status":"complete","research_status":result["status"],"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "title":"Forecasting the trajectory of graph connectivity","protocol_sha256":PROTOCOL_HASH,"protocol_path":"trajectory/protocol.json",
        "plain_summary":f"The models using summaries of the previous 20 or 60 graph windows did not beat a no-change forecast for 20-day-ahead connectivity of the 60-day graph on {result['outer_common_eligible_dates']} matched historical cases. The estimated probabilities and prediction intervals are not yet dependable. These results are exploratory, not a fresh holdout test.",
        "evaluation_span":{"first_asof":result["first_asof"],"last_asof":result["last_asof"],"n":result["outer_common_eligible_dates"]},
        "source":public_source(metadata),"universe":SYMBOLS,"window":60,"horizon":H,
        "feature_fields":{name:feature_fields(name) for name in VARIANTS},"primary_models":result["metrics"],"secondary_window_selection":result["secondary_window_selection"],
        "method":{"target":"delta=lambda2_W(t+20)-lambda2_W(t)","trajectory_inputs":"Four summaries of last20 or60 graph endpoints, including current: endpoint change, mean minus current, sample SD and linear slope. Not a raw lag-vector model.",
                  "shared_baseline":"Current graph score, minimum positive degree, mean correlation, BTC/basket volatility, plus already-known retained/outgoing subwindow summaries shared by every ridge variant.",
                  "ridge":"Training-only medians/flags/scaling; unpenalized intercept; alpha .1/1/10/100 selected using earlier inner validation MAE; refit only matured labels.",
                  "probabilities":"Empirical earlier rolling calibration residuals, shifted by point prediction; one pseudo-count per down/flat/up class; material band is fixed from initial fitting data.",
                  "information_timing":"Bar-end timestamps are observation labels, not measured provider publication times or forecast issuance. Latest explicitly separates observed_bar_end, source_acquired_at_utc and issued_at_utc."},
        "split_boundaries":{"first_outer_asof":result["first_asof"],"last_outer_asof":result["last_asof"],"latest_retrospective_label":result["latest_label_date"],"outer_block_observations":20,"purge_horizon":20,"details_path":"trajectory/evaluation.json"},
        "latest":latest,"latest_path":"trajectory/latest.json","evaluation_path":"trajectory/evaluation.json","forecasts_path":"trajectory/outer_forecasts.json",
        "figures":figures,"findings":findings,"counts":{"price_rows_including_seed":len(prices),"daily_return_rows":len(changes),"common_predictor_rows":int(context['common_predictors'].sum()),"scheduled_outer_dates":result['outer_scheduled_dates'],"eligible_outer_dates":result['outer_common_eligible_dates']},
        "interpretation":"Positive delta means the future graph is more connected under the same ten identities/rule, not that coin prices rise. Down/flat/up refer only to training-defined graph-change categories.",
        "limitations":["Earlier2021–Sep2026 data already inspected; all retrospective development is exploratory, not fresh holdout validation.","Current fixed cohort, one Binance spot venue, USDT numeraire, unmeasured publication latency and finite-sample correlation noise.","W60 future graph retains40 known returns; high persistence can arise mechanically. Changing W changes the target, so secondary W search is separate.","Residual intervals and probability location-shift calibration assume earlier errors remain informative; dependence and drift may violate this.","No SEC/news acquisition; any later economic inputs require official publication as-of timestamps, never later revised information.","No price forecasts, order execution, new purchases or financial-edge claim."],
        "spend_usd":0,"reproduce":"python crypto_trajectory.py --self-test; python crypto_trajectory.py",
        "rights":{"attribution":"Binance Vision","publication":"Calculated analytical derivatives only; no raw per-coin prices or returns published","license":"CC BY-NC-SA4.0 noncommercial derivative conditions under recorded provider terms"}}
    finalize_index_copy(index)
    for name,obj in (("index",index),("latest",latest),("evaluation",result),("outer_forecasts",records)):
        write_json(OUT/(name+".json"),obj);write_json(SITE/(name+".json"),obj)
    shutil.copy2(OUT/"protocol.json",SITE/"protocol.json")
    checks=validate_outputs(frames,dates,result,records,latest,context)
    write_json(OUT/"checks.json",checks)
    print(json.dumps(serial({"findings":findings,"latest":{k:latest[k] for k in ('asof','nominal_target_interval_end','status')},"checks":checks}),indent=2),flush=True)
    return index


def validate_outputs(frames,dates,result,records,latest,context):
    verify_protocol()
    assert len(result["metrics"])==4
    seen={name:[] for name in MODEL_NAMES}
    for row in records:
        assert pd.Timestamp(row["last_training_label_date"])<pd.Timestamp(row["origin"])<=pd.Timestamp(row["date"])<pd.Timestamp(row["target_date"])
        assert (pd.Timestamp(row["target_date"])-pd.Timestamp(row["date"])).total_seconds()==H*86400
        np.testing.assert_allclose(np.sum(row["probabilities"]),1.,atol=1e-12)
        assert np.min(row["probabilities"])>0
        if row["window"]==60:seen[row["model"]].append(row["date"])
    assert all(x==seen["persistence"] for x in seen.values())
    for row in latest["models"]:
        assert pd.Timestamp(row["last_training_label_date"])<pd.Timestamp(latest["asof"])
        assert all(x["last_training_label_position"]<x["origin_position"] for x in row["calibration_audit"])
    assert latest["actual_future_delta"] is None and latest["actual_future_direction"] is None
    assert pd.Timestamp(latest["nominal_target_interval_end"])>dates[-1]
    forbidden={"prices","price_dates","returns","raw_closes"}
    def inspect(x):
        if isinstance(x,dict):
            assert not forbidden.intersection(x)
            for v in x.values():inspect(v)
        elif isinstance(x,list):
            for v in x:inspect(v)
        elif isinstance(x,str):assert "C:\\Users\\" not in x and "C:/Users/" not in x
    for path in SITE.rglob("*.json"):inspect(json.loads(path.read_text(encoding="utf-8")))
    return {"status":"pass","protocol_sha256":PROTOCOL_HASH,"same_primary_dates_all_models":len(seen["persistence"]),
        "chronology":"Every stored prediction trains only strictly matured labels before block origin; calibration errors have their own causal origins.",
        "latest_target_pending":True,"frozen_material_bands":context["bands"],"hosted_raw_price_or_return_arrays":False,
        "limits":"These checks verify implementation constraints, not an uninspected holdout or reliable probability model."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test()
    else:run()
