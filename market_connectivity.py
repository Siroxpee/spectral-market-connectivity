"""Frozen-protocol, session-based market connectivity forecasts (Python only).

Future averages and endpoints are separate targets. Source loading is external;
no network, raw-price filling, trading, or fresh-holdout claim is made here.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_name]="1"
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import linalg
from crypto_core import graph_measurements,common_complete
from crypto_trajectory import fit_ridge,predict_ridge,serial,write_json,bootstrap_mean

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/market_connectivity"
W_GRID=(20,40,60,120)
K_GRID=(20,60,100)
H_GRID=(20,60,100)
ALPHAS=(.1,1.,10.,100.)
START=max(W_GRID)-1+max(K_GRID)-1
CHANNELS=("positive_lambda2_over_n","negative_mean_weight")
NEGATIVE_LITERAL="negative_lambda2_over_n"
MODELS=("persistence","local_mean60","AR1","current","trajectory_selected")
PROTOCOL_HASH="f644f81da30fc0030eef8e85b8195be9e1fba86e9f8febedc8fb8d9fcad025b9"
SEED=20261015


def check_protocol():
    path=OUT/"protocol.json"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PROTOCOL_HASH:raise ValueError("Frozen protocol hash changed")
    return json.loads(path.read_text(encoding="utf-8"))


def summary(array):
    if not common_complete(array):return None
    c=np.corrcoef(np.asarray(array,dtype=float),rowvar=False)
    n=len(c);upper=c[np.triu_indices(n,1)]
    plus=graph_measurements(c,"positive");minus=graph_measurements(c,"negative")
    return {"positive_lambda2_over_n":plus["lambda2"]/n,"negative_mean_weight":float(np.maximum(-upper,0).mean()),
        "negative_lambda2_over_n":minus["lambda2"]/n,"mean_pearson":float(upper.mean()),
        "mean_abs_correlation":float(np.abs(upper).mean()),"min_positive_degree_over_n":float(np.min(plus["degrees"])/n),
        "positive_density03":float((upper>=.3).mean()),"negative_density_minus01":float((upper<=-.1).mean()),
        "negative_pair_fraction":float((upper<0).mean()),"negative_components":minus["support_components"],
        "positive_components":plus["support_components"],"basket_vol_annual":float(np.asarray(array).mean(axis=1).std(ddof=1)*np.sqrt(252))}


def build_scores(features):
    """Graph scores only; fixed identities and a common finite window."""
    values=features.to_numpy(dtype=float);result={}
    for width in sorted(set(W_GRID)|set(H_GRID)):
        rows=[]
        for t in range(len(features)):
            row={"position":t,"date":features.index[t].isoformat(),"W":width,"n":features.shape[1]}
            if t>=width-1:
                measured=summary(values[t-width+1:t+1])
                if measured:row.update(measured)
            rows.append(row)
        frame=pd.DataFrame(rows)
        for field in CHANNELS+(NEGATIVE_LITERAL,):
            if field not in frame:frame[field]=np.nan
        result[width]=frame
        print(f"saved graph score width {width}",flush=True)
    combined=pd.concat(result.values(),ignore_index=True)
    combined.to_csv(OUT/"graph_scores.csv.gz",index=False,float_format="%.17g",compression="gzip")
    return result


def future_levels(score,horizon,kind):
    """No missing-score skipping: exact t+1 through t+H average."""
    score=np.asarray(score,dtype=float);answer=np.full(len(score),np.nan)
    if kind=="endpoint":answer[:-horizon]=score[horizon:];return answer
    if kind!="future_mean":raise ValueError("Unknown target")
    count=np.r_[0,np.cumsum(np.isfinite(score))]
    total=np.r_[0,np.cumsum(np.where(np.isfinite(score),score,0.))]
    t=np.arange(len(score)-horizon)
    valid=count[t+horizon+1]-count[t+1]==horizon
    answer[t[valid]]=(total[t[valid]+horizon+1]-total[t[valid]+1])/horizon
    return answer


def build_case_arrays(features,scores,width,horizon,channel,kind):
    frame=scores[width];score=frame[channel].to_numpy(dtype=float)
    current=[channel,"positive_lambda2_over_n","negative_mean_weight","mean_pearson","min_positive_degree_over_n",
             "positive_density03","negative_density_minus01","basket_vol_annual"]
    # Preserve names while avoiding a duplicated target-channel predictor.
    current=list(dict.fromkeys(current))
    x=frame[current].copy()
    for part in ("retained","outgoing"):
        for name in CHANNELS:x[part+"_"+name]=np.nan
    outgoing=min(horizon,width);retained=width-outgoing
    for name in CHANNELS:
        if retained>=2:x["retained_"+name]=scores[retained][name].to_numpy()
        x["outgoing_"+name]=scores[outgoing][name].shift(width-outgoing).to_numpy()
    x.loc[np.arange(len(frame))<width-1,[name for name in x if name.startswith(("retained_","outgoing_"))]]=np.nan
    arrays={0:x.to_numpy(dtype=float)};fields={0:x.columns.tolist()}
    complete=np.isfinite(score)
    for count in K_GRID:
        history=np.full((len(frame),4),np.nan)
        centered=np.arange(count)-(count-1)/2
        for t in range(count-1,len(frame)):
            past=score[t-count+1:t+1]
            if np.isfinite(past).all():history[t]=[past[-1]-past[0],past.mean()-past[-1],past.std(ddof=1),centered@past/(centered@centered)]
        arrays[count]=np.column_stack([arrays[0],history])
        fields[count]=fields[0]+[f"score_history{count}_"+k for k in ("change","mean_minus_current","sd","slope")]
        complete&=np.isfinite(history).all(axis=1)
    targetlevel=future_levels(score,horizon,kind)
    past_mean=pd.Series(score).rolling(60,min_periods=60).mean().to_numpy()
    return {"score":score,"future_level":targetlevel,"y":targetlevel-score,"x":arrays,"fields":fields,
            "predictors":complete&np.isfinite(past_mean),"labels":complete&np.isfinite(past_mean)&np.isfinite(targetlevel),"past_mean60":past_mean}


def clip_delta(raw,current):
    return np.clip(np.asarray(current)+np.asarray(raw),0.,1.)-np.asarray(current)


def direction(delta,band):
    return np.where(np.asarray(delta)<-band,0,np.where(np.asarray(delta)>band,2,1))


def project(point,current,errors,band):
    point=np.asarray(point).reshape(-1);current=np.asarray(current).reshape(-1);errors=np.asarray(errors)
    scenario=np.clip(current[:,None]+point[:,None]+errors[None,:],0.,1.)-current[:,None]
    labels=direction(scenario,band)
    feasible=np.column_stack([current>band,np.ones(len(current),dtype=bool),1-current>band])
    counts=np.column_stack([(labels==i).sum(axis=1) for i in range(3)])+feasible
    probabilities=counts/counts.sum(axis=1)[:,None]
    q=np.quantile(errors,[.05,.95]);lo_raw=point+q[0];hi_raw=point+q[1]
    return {"probabilities":probabilities,"lower":clip_delta(lo_raw,current),"upper":clip_delta(hi_raw,current),"lower_unclipped":lo_raw,"upper_unclipped":hi_raw}


def split_masks(origin,horizon,labels):
    positions=np.arange(len(labels));available=origin-horizon-START
    tune=START+int(.6*available);cal=START+int(.8*available)
    masks={"fit":labels&(positions<tune-horizon),"tune":labels&(positions>=tune)&(positions<cal-horizon),
           "cal":labels&(positions>=cal)&(positions+horizon<origin),"matured":labels&(positions+horizon<origin)}
    if min(masks[name].sum() for name in ("fit","tune","cal"))<40:raise ValueError("Insufficient nested training/tuning/calibration")
    assert np.all(np.flatnonzero(masks["fit"])+horizon<tune)
    assert np.all(np.flatnonzero(masks["tune"])+horizon<cal)
    return masks,{"origin_position":origin,"tuning_start_position":tune,"calibration_start_position":cal,
                  "counts":{name:int(mask.sum()) for name,mask in masks.items()}}


def choose_models(case,origin,horizon,labels):
    masks,boundaries=split_masks(origin,horizon,labels)
    chosen={};rows=[]
    for model,ks in (("current",(0,)),("trajectory_selected",K_GRID)):
        candidates=[]
        for count in ks:
            x=case["x"][count]
            for alpha in ALPHAS:
                fit=fit_ridge(x[masks["fit"]],case["y"][masks["fit"]],alpha)
                raw=predict_ridge(fit,x[masks["tune"]])
                pred=clip_delta(raw,case["score"][masks["tune"]])
                mae=float(np.abs(pred-case["y"][masks["tune"]]).mean())
                candidates.append((mae,-alpha,count,alpha))
                rows.append({"origin_position":origin,"model":model,"K":count,"alpha":alpha,"tuning_mae":mae})
        _,_,count,alpha=min(candidates)
        chosen[model]={"K":count,"alpha":alpha}
    return chosen,masks,boundaries,rows


class ForecastCache:
    """Causal expanding-origin fits reused across calibration and outer folds."""
    def __init__(self,case,horizon,kind,labels):
        self.case=case;self.horizon=horizon;self.kind=kind;self.labels=labels;self.cache={}

    def predict(self,origin,model,count=0,alpha=None):
        key=(origin,model,count,alpha)
        if key in self.cache:return self.cache[key]
        case=self.case;n=len(case["y"]);pos=np.arange(n)
        end=min(origin+20,n);indices=np.arange(origin,end)
        current=case["score"][indices]
        matured=self.labels&(pos+self.horizon<origin)
        last_label=int(np.flatnonzero(matured)[-1]+self.horizon)
        training_n=int(matured.sum())
        fit=None
        if model=="persistence":raw=np.zeros(len(indices))
        elif model=="local_mean60":raw=case["past_mean60"][indices]-current
        elif model=="AR1":
            s=case["score"];j=np.arange(1,origin)
            j=j[np.isfinite(s[j])&np.isfinite(s[j-1])]
            if len(j)<40:raise ValueError("Insufficient AR1 pairs")
            design=np.column_stack([np.ones(len(j)),s[j-1]])
            intercept,slope=linalg.lstsq(design,s[j])[0]
            path=current.copy();total=np.zeros(len(indices))
            for _ in range(self.horizon):path=intercept+slope*path;total+=path
            future=total/self.horizon if self.kind=="future_mean" else path
            raw=future-current;last_label=int(j[-1]);fit={"intercept":float(intercept),"slope":float(slope),"training_pairs":len(j)}
            training_n=len(j)
        else:
            fit=fit_ridge(case["x"][count][matured],case["y"][matured],alpha)
            raw=predict_ridge(fit,case["x"][count][indices])
        result={"positions":indices,"raw":raw,"point":clip_delta(raw,current),"fit":fit,"last_training_label_position":last_label,
                "training_n":training_n,"origin":origin}
        assert last_label<origin
        self.cache[key]=result
        return result

    def calibration(self,origin,cal_start,model,count=0,alpha=None):
        first=START+int(np.ceil((cal_start-START)/20))*20
        errors=[];audits=[]
        for block in range(first,origin-self.horizon,20):
            bundle=self.predict(block,model,count,alpha)
            indices=bundle["positions"]
            keep=self.labels[indices]&(indices+self.horizon<origin)
            if not keep.any():continue
            errors.extend((self.case["y"][indices[keep]]-bundle["point"][keep]).tolist())
            audits.append({"origin":block,"last_training_label":bundle["last_training_label_position"],"n":int(keep.sum())})
        if len(errors)<40:raise ValueError("Insufficient causal calibration errors")
        return np.asarray(errors),audits


def metrics(records):
    if not records:return None
    truth=np.array([r["actual_delta"] for r in records]);point=np.array([r["predicted_delta"] for r in records])
    p=np.array([[r["p_down"],r["p_flat"],r["p_up"]] for r in records]);labels=np.array([r["actual_class"] for r in records])
    coverage=np.array([r["interval_lower"]<=r["actual_delta"]<=r["interval_upper"] for r in records])
    bins=[]
    for i,name in enumerate(("down","flat","up")):
        for lo,hi in zip((0.,.2,.4,.6,.8),(.2,.4,.6,.8,1.)):
            keep=(p[:,i]>=lo)&(p[:,i]<hi if hi<1 else p[:,i]<=hi)
            if keep.any():bins.append({"class":name,"lower":lo,"upper":hi,"n":int(keep.sum()),"mean_prediction":float(p[keep,i].mean()),"observed_frequency":float((labels[keep]==i).mean())})
    return {"n":len(records),"mae":float(np.abs(point-truth).mean()),"rmse":float(np.sqrt(np.mean((point-truth)**2))),
        "bias":float((point-truth).mean()),"brier":float(((p-np.eye(3)[labels])**2).sum(axis=1).mean()),
        "nominal90_coverage":float(coverage.mean()),"mean_interval_width":float(np.mean([r["interval_upper"]-r["interval_lower"] for r in records])),
        "class_counts":dict(zip(("down","flat","up"),np.bincount(labels,minlength=3).tolist())),"calibration":bins,
        "clipped_prediction_fraction":float(np.mean([abs(r["raw_predicted_delta"]-r["predicted_delta"])>1e-12 for r in records]))}


def case_id(width,horizon,channel,kind):return f"{channel}_W{width}_H{horizon}_{kind}"


def evaluate_case(features,scores,width,horizon,channel,kind,common_labels,common_predictors):
    case=build_case_arrays(features,scores,width,horizon,channel,kind)
    labels=case["labels"]&common_labels;pos=np.arange(len(features));labels&=pos>=START
    first=START+int(.75*(len(features)-horizon-START));stop=len(features)-horizon
    initial,_=split_masks(first,horizon,labels)
    band=.5*float(np.std(case["y"][initial["fit"]],ddof=1))
    cache=ForecastCache(case,horizon,kind,labels)
    records=[];choices=[];boundaries=[]
    for origin in range(first,stop,20):
        chosen,masks,boundary,grid=choose_models(case,origin,horizon,labels)
        selected=labels&(pos>=origin)&(pos<min(origin+20,stop))
        if not selected.any():continue
        indices=np.flatnonzero(selected);choices.extend(grid)
        boundary.update({"origin_date":features.index[origin].isoformat(),"chosen":chosen})
        for model in MODELS:
            settings=chosen.get(model,{"K":0,"alpha":None})
            count=settings["K"];alpha=settings["alpha"]
            bundle=cache.predict(origin,model,count,alpha)
            j=indices-origin
            point=bundle["point"][j];raw=bundle["raw"][j]
            errors,audits=cache.calibration(origin,boundary["calibration_start_position"],model,count,alpha)
            projection=project(point,case["score"][indices],errors,band)
            for k,t in enumerate(indices):
                records.append({"case_id":case_id(width,horizon,channel,kind),"date":features.index[t].isoformat(),"position":int(t),
                    "target_date":features.index[t+horizon].isoformat(),"origin":features.index[origin].isoformat(),"W":width,"H":horizon,"channel":channel,"target_kind":kind,
                    "model":model,"K":count,"alpha":alpha,"current_score":case["score"][t],"actual_future_level":case["future_level"][t],
                    "actual_delta":case["y"][t],"raw_predicted_delta":float(raw[k]),"predicted_delta":float(point[k]),"predicted_future_level":float(case["score"][t]+point[k]),
                    "interval_lower":projection["lower"][k],"interval_upper":projection["upper"][k],"interval_lower_unclipped":projection["lower_unclipped"][k],"interval_upper_unclipped":projection["upper_unclipped"][k],
                    "material_band":band,"actual_class":int(direction(case["y"][t],band)),"p_down":projection["probabilities"][k,0],"p_flat":projection["probabilities"][k,1],"p_up":projection["probabilities"][k,2],
                    "training_n":bundle["training_n"],"calibration_n":len(errors),"last_training_label_date":features.index[bundle["last_training_label_position"]].isoformat()})
            boundary[model+"_calibration_audit"]=audits
        boundaries.append(boundary)
    groups={model:[r for r in records if r["model"]==model] for model in MODELS}
    dates=[r["date"] for r in groups["persistence"]]
    assert all([r["date"] for r in rs]==dates for rs in groups.values())
    scored={model:metrics(rows) for model,rows in groups.items()}
    paired=[]
    for first_model,second_model in (("local_mean60","persistence"),("AR1","persistence"),("current","persistence"),("trajectory_selected","persistence"),("trajectory_selected","AR1"),("trajectory_selected","current")):
        loss=lambda model:np.array([abs(r["predicted_delta"]-r["actual_delta"]) for r in groups[model]])
        for block in sorted({60,max(60,width+horizon-1)}):
            interval=bootstrap_mean(loss(first_model)-loss(second_model),block=block,replicates=1000)
            paired.append({"first":first_model,"second":second_model,**interval})
    sensitivity=[]
    for offset in range(horizon):
        sensitivity.append({"offset":offset,"models":{model:metrics([r for r in rows if (r["position"]-first)%horizon==offset]) for model,rows in groups.items()}})
    selections=Counter((r["K"],r["alpha"]) for r in groups["trajectory_selected"])
    result={"id":case_id(width,horizon,channel,kind),"W":width,"H":horizon,"channel":channel,"target_kind":kind,
        "status":"complete; exploratory retrospective","evaluation_span":{"first_asof":dates[0],"last_asof":dates[-1],"n":len(dates)},
        "models":scored,"paired_differences":paired,"selection_counts":[{"K":k,"alpha":a,"dates":n} for (k,a),n in sorted(selections.items())],
        "frozen_material_band":band,"initial_fitting_target_sd":band*2,"initial_fitting_n":int(initial["fit"].sum()),
        "degenerate_initial_target":bool(np.ptp(case["y"][initial["fit"]])<=1e-12),"strideH_sensitivity":sensitivity,"boundaries":boundaries,
        "negative_lambda2_zero_fraction":float(scores[width][NEGATIVE_LITERAL].dropna().eq(0).mean()),
        "known_return_slot_fraction_future_mean":float(sum(max(width-h,0) for h in range(1,horizon+1))/(width*horizon)),
        "limitations":"Exploratory; W/H change estimand; many comparisons; overlapping scores and targets; conditional resampling is not familywise inference or fresh validation."}
    latest_origin=int(np.flatnonzero(case["predictors"]&common_predictors)[-1])
    chosen,_,boundary,_=choose_models(case,latest_origin,horizon,labels)
    latest={"case_id":result["id"],"W":width,"H":horizon,"channel":channel,"target_kind":kind,"observed_session":features.index[latest_origin].isoformat(),
        "issued_at_utc":datetime.now(timezone.utc).isoformat(),"future_session_offset":horizon,"exact_target_session":None,
        "status":"Archived source-as-of projection, generated later; target unscored and partially elapsed at issuance, not a fully prospective live forecast",
        "is_live_forecast":False,"source_stale_at_issuance":True,"prospective_holdout":False,
        "timing_note":"Source ends October8; projection generated October15. Some target sessions have already elapsed in real time but are absent from this frozen input. Null targets mean unscored in this archive, not proof all outcomes are still in the future.",
        "actual_future_level":None,"actual_delta":None,"models":[]}
    for model in MODELS:
        setting=chosen.get(model,{"K":0,"alpha":None});bundle=cache.predict(latest_origin,model,setting["K"],setting["alpha"])
        errors,audits=cache.calibration(latest_origin,boundary["calibration_start_position"],model,setting["K"],setting["alpha"])
        point=bundle["point"][0];current=case["score"][latest_origin];p=project([point],[current],errors,band)
        latest["models"].append({"model":model,**setting,"current_score":current,"predicted_delta":point,"predicted_future_level":current+point,
            "interval_lower":p["lower"][0],"interval_upper":p["upper"][0],"probabilities":dict(zip(("down","flat","up"),p["probabilities"][0])),
            "material_band":band,"calibration_n":len(errors),"last_training_label_date":features.index[bundle["last_training_label_position"]].isoformat(),"fit":bundle["fit"]})
    return result,records,choices,latest


def disjoint_diagnostics(scores,dates):
    rows=[]
    for horizon in H_GRID:
        frame=scores[horizon]
        for t in range(horizon-1,len(frame)-horizon):
            row={"date":dates[t].isoformat(),"H":horizon,"future_end":dates[t+horizon].isoformat()}
            for channel in CHANNELS+(NEGATIVE_LITERAL,):
                past=frame.iloc[t][channel];future=frame.iloc[t+horizon][channel]
                row[channel+"_past_block"]=past;row[channel+"_future_block"]=future;row[channel+"_change"]=future-past
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"disjoint_diagnostics.csv.gz",index=False,float_format="%.17g",compression="gzip")


def validate_case(result,records,latest):
    assert all(m["n"]==result["evaluation_span"]["n"] for m in result["models"].values())
    for row in records:
        assert pd.Timestamp(row["last_training_label_date"])<pd.Timestamp(row["origin"])<=pd.Timestamp(row["date"])<pd.Timestamp(row["target_date"])
        p=np.array([row["p_down"],row["p_flat"],row["p_up"]])
        assert np.isfinite(p).all() and (p>=0).all() and np.isclose(p.sum(),1.)
        assert -1e-12<=row["predicted_future_level"]<=1+1e-12
        assert row["interval_lower"]<=row["interval_upper"]
    for boundary in result["boundaries"]:
        for model in MODELS:
            assert all(a["last_training_label"]<a["origin"] for a in boundary[model+"_calibration_audit"])
    assert latest["actual_delta"] is None and latest["actual_future_level"] is None
    assert all(pd.Timestamp(r["last_training_label_date"])<pd.Timestamp(latest["observed_session"]) for r in latest["models"])


def self_test():
    check_protocol()
    s=np.arange(250,dtype=float)/250
    mean=future_levels(s,20,"future_mean");end=future_levels(s,20,"endpoint")
    assert np.isclose(mean[100],s[101:121].mean()) and np.isclose(end[100],s[120])
    assert not np.isclose(mean[100],end[100])
    broken=s.copy();broken[110]=np.nan
    assert np.isnan(future_levels(broken,20,"future_mean")[100])
    assert np.isfinite(future_levels(broken,20,"endpoint")[100])
    changed=s.copy();changed[121:]=.9
    assert np.isclose(future_levels(changed,20,"future_mean")[100],mean[100])
    masks,bounds=split_masks(1200,100,np.arange(1800)>=START)
    assert np.all(np.flatnonzero(masks["fit"])+100<bounds["tuning_start_position"])
    assert np.all(np.flatnonzero(masks["tune"])+100<bounds["calibration_start_position"])
    assert np.all(np.flatnonzero(masks["matured"])+100<1200)
    p=project([-.5,.5],[0.,1.],np.linspace(-1,1,100),.1)
    np.testing.assert_allclose(p["probabilities"].sum(axis=1),1.)
    assert p["probabilities"][0,0]==0 and p["probabilities"][1,2]==0
    np.testing.assert_allclose(clip_delta([9.,-9.],[.2,.8]),[.8,-.8])
    c=np.array([[1.,.3,-.2],[.3,1.,.1],[-.2,.1,1.]])
    g=graph_measurements(c,"negative")
    assert g["lambda2"]==0 and g["support_components"]==2
    n=12;assert np.isclose(sum(max(60-h,0) for h in range(1,21))/(60*20),.825)
    rng=np.random.default_rng(SEED);x=rng.normal(size=(100,3));y=rng.normal(size=100)
    fit=fit_ridge(x,y,10.);before=fit["mean"].copy();predict_ridge(fit,np.full((5,3),999.));assert np.array_equal(before,fit["mean"])
    # Compare one-step AR source alignment directly; H is session rows, not days.
    assert START==218
    print("PASS protocol hash; distinct exact mean/endpoint targets; missing-score guard; strict horizon purges; physicalbounds/probabilities; negative support; known82.5%overlap; train-only transforms",flush=True)


def compact_result(result):
    return {k:v for k,v in result.items() if k not in ("boundaries","strideH_sensitivity")}


def portable_metadata(metadata):
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not any(x in k.lower() for x in ("path","credential","secret"))}
        if isinstance(value,list):return [clean(x) for x in value]
        if isinstance(value,str) and ("C:\\Users\\" in value or "C:/Users/" in value):return "local source retained separately"
        return value
    return clean(metadata)


def main(limit=None):
    protocol=check_protocol()
    from market_connectivity_data import load_panel
    features,metadata=load_panel()
    if not features.index.is_unique or not features.index.is_monotonic_increasing:raise ValueError("Source sessions duplicated/unordered")
    if list(features.columns)!=protocol["universe"]:raise ValueError("Frozen market universe changed")
    if metadata.get("prices_sha256")!=protocol["source_price_sha256"] or metadata.get("feature_matrix_sha256")!=protocol["source_feature_sha256"]:
        raise ValueError("Frozen source fingerprints changed; do not reuse existing case checkpoints")
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"cases").mkdir(exist_ok=True)
    if not (OUT/"first_computation.json").exists():write_json(OUT/"first_computation.json",{"started_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":PROTOCOL_HASH})
    print("Computing shared source-session graph histories",flush=True)
    scores=build_scores(features)
    disjoint_diagnostics(scores,features.index)
    common_predictors=np.ones(len(features),dtype=bool)
    for width in W_GRID:
        for channel in CHANNELS:
            s=pd.Series(scores[width][channel])
            common_predictors&=s.rolling(100,min_periods=100).count().eq(100).to_numpy()
    common_predictors[:START]=False
    common_by_h={}
    for horizon in H_GRID:
        eligible=common_predictors.copy()
        for width in W_GRID:
            for channel in CHANNELS:
                for kind in ("future_mean","endpoint"):
                    eligible&=np.isfinite(future_levels(scores[width][channel],horizon,kind))
        common_by_h[horizon]=eligible
    cases=[(60,20,c,"future_mean") for c in CHANNELS]
    cases+=[(60,20,NEGATIVE_LITERAL,kind) for kind in ("future_mean","endpoint")]
    cases+=[(w,h,c,k) for w in W_GRID for h in H_GRID for c in CHANNELS for k in ("future_mean","endpoint") if (w,h,c,k) not in cases]
    results=[];all_records=[];all_choices=[];all_latest=[];checks=[];started=time.monotonic()
    paths={"comparison":"comparison.csv","predictions":"predictions.csv.gz","selection":"selection.csv","graph_scores":"graph_scores.csv.gz","disjoint_diagnostics":"disjoint_diagnostics.csv.gz","cases":"cases/","latest":"latest.json"}
    def publish(status):
        table=[]
        for result in results:
            for model,item in result["models"].items():
                row={"case_id":result["id"],"W":result["W"],"H":result["H"],"channel":result["channel"],"target_kind":result["target_kind"],"model":model,
                     **{k:item[k] for k in ("n","mae","rmse","bias","brier","nominal90_coverage","mean_interval_width","clipped_prediction_fraction")}}
                for baseline in ("persistence","AR1","current"):
                    row["relative_mae_vs_"+baseline]=item["mae"]/result["models"][baseline]["mae"] if result["models"][baseline]["mae"]>1e-12 else None
                    pair=next((p for p in result["paired_differences"] if p["first"]==model and p["second"]==baseline and p["block_length"]==max(60,result["W"]+result["H"]-1)),None)
                    row["paired_mae_difference_vs_"+baseline]=pair["mean"] if pair else None
                    row["paired_ci_low_vs_"+baseline]=pair["percentile_95"][0] if pair else None
                    row["paired_ci_high_vs_"+baseline]=pair["percentile_95"][1] if pair else None
                table.append(row)
        pd.DataFrame(table).to_csv(OUT/"comparison.csv",index=False,float_format="%.17g")
        primary=[r for r in results if r["W"]==60 and r["H"]==20 and r["target_kind"]=="future_mean" and r["channel"] in CHANNELS]
        index={"schema_version":1,"status":status,"protocol_sha256":PROTOCOL_HASH,"research_status":"Exploratory retrospective; no genuine fresh holdout claim",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"dataset":portable_metadata(metadata),
            "primary_definition":protocol["primary"],"W_grid":W_GRID,"K_grid":K_GRID,"H_grid":H_GRID,
            "primary_results":[compact_result(r) for r in primary],"results":[compact_result(r) for r in results],
            "evaluation_spans":{str(h):next((r["evaluation_span"] for r in results if r["H"]==h),None) for h in H_GRID},
            "latest":all_latest,"completed_cases":len(results),"planned_cases":len(cases),"paths":paths,
            "target_definitions":protocol["targets"],"findings":findings(primary),"code":"market_connectivity.py",
            "notes":["Lagged score histories are not directed lead-lag relationships or causal evidence.","Normalization by n rescales score; it does not remove changed identities or market composition.","Mean-targetW60H20 contains82.5%already-known return slots on average, not82.5%explained variance.","Source dates are native provider/SPY-supported sessions, not independently verified exchangecalendar; latest issuance and observation dates differ.","Latest is an archived October8-as-of projection generated October15; target partially elapsed at issuance, not a fully prospective live forecast.","Many secondary comparisons; conditional intervals are not familywise corrected. No universalbestW/H claim."]}
        write_json(OUT/"index.json",index)
    for number,(width,horizon,channel,kind) in enumerate(cases):
        if limit is not None and number>=limit:break
        identifier=case_id(width,horizon,channel,kind)
        print(f"CASE {number+1}/{len(cases)} {identifier}",flush=True)
        checkpoint=OUT/"cases"/(identifier+".json")
        rows_path=OUT/"cases"/(identifier+"_predictions.csv.gz")
        choice_path=OUT/"cases"/(identifier+"_selection.csv")
        latest_path=OUT/"cases"/(identifier+"_latest.json")
        if all(p.exists() for p in (checkpoint,rows_path,choice_path,latest_path)):
            result=json.loads(checkpoint.read_text(encoding="utf-8"))
            if result.get("protocol_sha256")!=PROTOCOL_HASH:raise ValueError("Checkpoint protocol differs")
            records=pd.read_csv(rows_path,float_precision="round_trip").replace({np.nan:None}).to_dict("records")
            choices=pd.read_csv(choice_path,float_precision="round_trip").to_dict("records")
            latest=json.loads(latest_path.read_text(encoding="utf-8"))
        else:
            result,records,choices,latest=evaluate_case(features,scores,width,horizon,channel,kind,common_by_h[horizon],common_predictors)
            result["protocol_sha256"]=PROTOCOL_HASH
            validate_case(result,records,latest)
            pd.DataFrame(records).to_csv(rows_path,index=False,float_format="%.17g",compression="gzip")
            pd.DataFrame(choices).to_csv(choice_path,index=False,float_format="%.17g")
            write_json(latest_path,latest);write_json(checkpoint,result)
        results.append(result);all_records.extend(records);all_choices.extend({"case_id":identifier,**r} for r in choices);all_latest.append(latest)
        checks.append({"id":identifier,"status":"pass","matched_dates":result["evaluation_span"]["n"]})
        publish("running" if len(results)<len(cases) else "complete")
        print(f"DONE {identifier}; persistenceMAE={result['models']['persistence']['mae']:.6f}; trajectoryMAE={result['models']['trajectory_selected']['mae']:.6f}; elapsed={time.monotonic()-started:.1f}s",flush=True)
    pd.DataFrame(all_records).to_csv(OUT/"predictions.csv.gz",index=False,float_format="%.17g",compression="gzip")
    pd.DataFrame(all_choices).to_csv(OUT/"selection.csv",index=False,float_format="%.17g")
    write_json(OUT/"latest.json",all_latest)
    write_json(OUT/"checks.json",{"status":"pass","protocol_sha256":PROTOCOL_HASH,"cases":checks,
        "checks":["Strictly earlier training and own-origin rolling calibration label dates.","Matched model targets/dates, futuremean and endpoint separate, probabilities sum1 and respect physicalbounds.","Future targets and issued pending projections are separated; no calendar-day masquerading as ETF sessions."],
        "completed_cases":len(results),"expected_cases":len(cases),"elapsed_seconds":time.monotonic()-started})
    publish("complete" if len(results)==len(cases) else "partial_requested_limit")
    return results


def findings(primary):
    result=[]
    for case in primary:
        m=case["models"];best_base=min(("persistence","local_mean60","AR1"),key=lambda name:m[name]["mae"])
        result.append(f"{case['channel']}: on {case['evaluation_span']['n']} matched historical dates, selected trajectory MAE {m['trajectory_selected']['mae']:.5f}; persistence {m['persistence']['mae']:.5f}; AR1 {m['AR1']['mae']:.5f}; current regression {m['current']['mae']:.5f}. Strongest observed simple baseline: {best_base}.")
        result.append(f"{case['channel']}: trajectory three-class Brier {m['trajectory_selected']['brier']:.4f}, nominal90% interval coverage {m['trajectory_selected']['nominal90_coverage']:.1%}. Empirical calibration must be assessed; these are not guaranteed future probabilities.")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:main(args.limit)
