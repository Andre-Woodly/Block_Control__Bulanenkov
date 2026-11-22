"""Experiment 59: pre-forecast alarm for NN residual forecast model.

Synthetic passive two-state thermal system only; no actuators, control, feedback.
Only measurements at t<=20 are used to build warning indicators. Future observations
are used solely after the alarm for calculating coverage and labels.

Run from a checkout containing sibling experiments/passive_thermal_58/experiment.py:
    python experiments/passive_thermal_59/experiment.py --outputs-dir results/experiment_59
"""
import argparse, importlib.util, json, hashlib, datetime, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
import torch

PREV=Path(__file__).resolve().parent.parent/'passive_thermal_58'/'experiment.py'
if not PREV.is_file():
    raise FileNotFoundError(f'Experiment 58 source required at {PREV}')
spec=importlib.util.spec_from_file_location('thermal_58', PREV)
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

TRAIN_N=1200; CAL_N=400; TEST_N=1200; SEEDS=[4,17,29,41,53]
SCENARIOS=['matched','in_support','heldout_combo','outside_support','higher_noise']
SCORE_NAMES=['history_fit','last_innovation','mahalanobis','correction_size','nn_disagreement','combined']

def quantile(s,prob=.90):
    s=np.asarray(s)
    rank=int(np.ceil((len(s)+1)*prob))-1
    return float(np.sort(s)[min(rank,len(s)-1)])

def fit_history_nominal(ds):
    # Physical residual of 5 observed history values to fixed nominal 2-state manifold.
    Y=ds['yobs']; states=np.linalg.lstsq(p.MO,Y.T,rcond=None)[0]
    return np.sqrt(np.mean((Y-states.T@p.MO.T)**2,axis=1))

def last_innovation(ds):
    # Only t=0,5,10,15 used to predict available current t=20.
    M=p.MO[:4,:]
    prior_precision=np.diag(1/p.PRIOR_VAR)
    precision=M.T@M/(p.TRAIN_NOISE**2)+prior_precision
    gain=np.linalg.solve(precision,M.T)/(p.TRAIN_NOISE**2)
    mean=np.linalg.solve(precision,p.PRIOR_M/p.PRIOR_VAR)
    est=ds['yobs'][:,:4]@gain.T+mean
    pred=est@p.MO[4,:]
    return np.abs(ds['yobs'][:,4]-pred)

def fit_mahalanobis(train):
    mu=train['yobs'].mean(axis=0)
    cov=np.cov(train['yobs'],rowvar=False)
    # 0.01 C² ridge to avoid numerical instability; fixed from training data only.
    inv=np.linalg.inv(cov+0.01*np.eye(cov.shape[0]))
    def predict(ds):
        center=ds['yobs']-mu
        return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',center,inv,center)))
    return predict

def summary_predict(ds,predictors):
    all_preds=np.stack([f(ds) for f in predictors])
    return all_preds.mean(axis=0),all_preds.std(axis=0,ddof=1).mean(axis=1)

def raw_scores(ds,predictors,mahal):
    avg,spread=summary_predict(ds,predictors)
    baseline=p.physics(ds,False)
    scores={'history_fit':fit_history_nominal(ds),
            'last_innovation':last_innovation(ds),
            'mahalanobis':mahal(ds),
            'correction_size':np.sqrt(np.mean((avg-baseline)**2,axis=1)),
            'nn_disagreement':spread}
    return scores,avg

def auc_binary(y,score):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    npos=y.sum();nneg=(~y).sum()
    if not npos or not nneg:return float('nan')
    ranks=rankdata(score,method='average')
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))

def wilson(n,k,z=1.96):
    if n==0:return (float('nan'),float('nan'))
    pp=k/n;dd=1+z*z/n
    center=(pp+z*z/(2*n))/dd
    radius=z*np.sqrt(pp*(1-pp)/n+z*z/(4*n*n))/dd
    return float(center-radius),float(center+radius)

def bootstrap_auc(y,score,seed=901,reps=400):
    n=len(y);rng=np.random.default_rng(seed);arr=[]
    for _ in range(reps):
        ix=rng.integers(n,size=n); a=auc_binary(y[ix],score[ix])
        if np.isfinite(a):arr.append(a)
    return np.quantile(arr,[.025,.975]) if arr else np.array([np.nan,np.nan])

def main(dest):
    dest.mkdir(parents=True,exist_ok=True)
    start=time.time()
    tr_nom=p.generate(TRAIN_N,1101,'train_nominal')
    tr_broad=p.generate(TRAIN_N,1101,'train_broad')
    cal_forecast=p.generate(CAL_N,2202,'matched')
    cal_alert=p.generate(CAL_N,2203,'matched')
    scenes={x:p.generate(TEST_N,3303,x) for x in SCENARIOS}
    networks=[p.fit_nn(tr_broad,s,True,steps=500) for s in SEEDS]
    predictors=[f for f,l in networks]
    losses=[float(l) for f,l in networks]
    print('TRAINED LOSSES',losses,flush=True)
    mahal=fit_mahalanobis(tr_nom)
    # Forecast interval calibration, completely separate from alert thresholds.
    pred_cal,_=summary_predict(cal_forecast,predictors)
    q=quantile(np.max(np.abs(pred_cal-cal_forecast['yfut']),axis=1))
    scores_ref,_=raw_scores(tr_nom,predictors,mahal)
    # Scores normalized by 95th-percentile on training observations.
    scales={k:max(quantile(v,.95),1e-9) for k,v in scores_ref.items()}
    def computed(ds):
        sc,pred=raw_scores(ds,predictors,mahal)
        sc['combined']=np.max(np.column_stack([sc[k]/scales[k] for k in scales]),axis=1)
        return sc,pred
    scores_al,_=computed(cal_alert)
    thresholds={k:quantile(scores_al[k],.90) for k in SCORE_NAMES}
    row=[]; sample=[]; alarm_rates=[]
    for scene,ds in scenes.items():
        scores,pred=computed(ds)
        err=pred-ds['yfut']
        future_failure=np.max(np.abs(err),axis=1)>q
        rmse=float(np.sqrt(np.mean(err**2)))
        fail_rate=float(future_failure.mean())
        for method in SCORE_NAMES:
            s=scores[method]; alarm=s>thresholds[method]
            n_alarm=int(alarm.sum());n_silent=len(alarm)-n_alarm
            wc=wilson(len(alarm),n_alarm)
            auc=auc_binary(future_failure,s)
            if method=='combined': ci_auc=bootstrap_auc(future_failure,s)
            else:ci_auc=(float('nan'),float('nan'))
            row.append(dict(scenario=scene,indicator=method,n_test=len(alarm),n_failures=int(future_failure.sum()),
                forecast_rmse=rmse,forecast_failure_rate=fail_rate,
                alarm_rate=n_alarm/len(alarm),alarm_wilson_low=wc[0],alarm_wilson_high=wc[1],
                threshold=thresholds[method],roc_auc_for_future_failure=auc,
                roc_auc_bootstrap_lo=ci_auc[0],roc_auc_bootstrap_hi=ci_auc[1],
                future_failure_given_alarm=float(future_failure[alarm].mean()) if n_alarm else np.nan,
                future_failure_given_silence=float(future_failure[~alarm].mean()) if n_silent else np.nan,
                future_failures_caught=int(np.sum(alarm&future_failure)),
                future_failures_missed=int(np.sum((~alarm)&future_failure)),
                hit_rate_given_failure=float(np.mean(alarm[future_failure])) if future_failure.any() else np.nan))
            if method=='combined':
                for i in range(len(alarm)):
                    sample.append(dict(scenario=scene,index=i,alarm_combined=bool(alarm[i]),future_failure=bool(future_failure[i]),
                        error_maxabs=float(np.max(np.abs(err[i]))),
                        pred_interval_q=q,score_combined=float(s[i])))
        print(scene, 'RMSE',round(rmse,4),'fail',round(fail_rate,4),'combined alert',round(row[-1]['alarm_rate'],4),flush=True)
    full=pd.DataFrame(row);full.to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(sample).to_csv(dest/'individual_outcomes.csv',index=False)
    calibrations=pd.DataFrame([dict(indicator=k,alert_threshold=v,score_scale_95=scales.get(k,np.nan),
                training_reference_n=TRAIN_N,alert_calibration_n=CAL_N) for k,v in thresholds.items()])
    calibrations.to_csv(dest/'calibration.csv',index=False)
    meta={'utc_date':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
          'torch':torch.__version__,'scipy':scipy.__version__,
          'experiment_58_source_sha256':hashlib.sha256(PREV.read_bytes()).hexdigest(),
          'experiment_59_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'training_seed':1101,'interval_calibration_seed':2202,'alarm_calibration_seed':2203,'test_seed':3303,
          'network_initializations':SEEDS,'training_losses':losses,
          'n_train_broad':TRAIN_N,'n_train_nominal_for_detector':TRAIN_N,
          'n_calibrate_interval':CAL_N,'n_calibrate_alarm':CAL_N,'n_test_per_scenario':TEST_N,
          'warning_nominal_threshold_quantile':0.9,
          'forecast_nominal_simultaneous_interval_coverage_target':0.9,
          'forecast_interval_halfwidth':q,'indicator_thresholds':thresholds,'indicator_normalizers':scales,
          'forecast_method':'ensemble mean of five broad-parameter trained hybrid NN correctors',
          'elapsed_seconds':time.time()-start,
          'scope':'Synthetic passive thermal forecast with pre-forecast observed-history alarms. No physical actuator or closed-loop control.'}
    (dest/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nCOMPLETE RESULTS\n',full.to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
    print('\nCALIBRATION',calibrations.to_string(index=False),flush=True)
    print('METADATA',meta['elapsed_seconds'])
    import matplotlib.pyplot as plt
    df=full.pivot(index='indicator',columns='scenario',values='alarm_rate').reindex(SCORE_NAMES)
    df[['matched','in_support','heldout_combo','outside_support','higher_noise']].plot.bar(figsize=(13,5),rot=30)
    plt.ylabel('Warning frequency before future observations');plt.ylim(0,1.05)
    plt.title('Pre-forecast warning indicators: matched 10% nominal alarm budget')
    plt.grid(axis='y',alpha=.2);plt.tight_layout();plt.savefig(dest/'alarms.png',dpi=160)
    df2=full.pivot(index='indicator',columns='scenario',values='roc_auc_for_future_failure').reindex(SCORE_NAMES)
    df2[['matched','in_support','heldout_combo','outside_support','higher_noise']].plot.bar(figsize=(13,5),rot=30)
    plt.ylabel('ROC-AUC for future nominal-interval failure');plt.ylim(0,1)
    plt.title('Early warnings do not directly observe the future failure')
    plt.grid(axis='y',alpha=.2);plt.tight_layout();plt.savefig(dest/'auc.png',dpi=160)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--outputs-dir',default=str(Path(__file__).resolve().parent))
    args=ap.parse_args()
    main(Path(args.outputs_dir))