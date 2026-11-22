"""Exp. 60. Observable, test-leakage-free prediction intervals for passive thermal NN forecasts.

Purpose: compare empirical coverage and width of pre-deployment calibration strategies
on a synthetic passive two-state temperature forecasting task. Does not study
actuators, feedback control, or safety-critical equipment.

Dependency: sibling passive_thermal_58/experiment.py, which supplies the exact
2-state passive thermal simulator, fixed physics prior and NN architecture.

Run from checkout:
 python experiments/passive_thermal_60/experiment.py --outputs-dir results/experiment_60
"""
import argparse, datetime, hashlib, importlib.util, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch

PATH58 = Path(__file__).resolve().parent.parent / 'passive_thermal_58' / 'experiment.py'
if not PATH58.is_file():
    # Supports local archived names /mnt/data/passive_thermal_58
    alt = Path(__file__).resolve().parent.parent / 'passive_thermal_58' / 'experiment.py'
    raise FileNotFoundError(f'Experiment 58 code required at {alt}')
spec=importlib.util.spec_from_file_location('thermal_58',PATH58)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

TARGET=.90
SEEDS=[4,17,29,41,53]
KNOWN=('matched','in_support','higher_noise')
SCENARIOS=('matched','in_support','heldout_combo','outside_support','higher_noise')


def quantile(scores,prob=TARGET):
    v=np.sort(np.asarray(scores,dtype=float))
    idx=min(len(v)-1,int(np.ceil((len(v)+1)*prob))-1)
    return float(v[idx])


def wilson(n,k,z=1.9599639845):
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-rad,center+rad


def fit_mahalanobis(train):
    obs=train['yobs']; m=obs.mean(0)
    inv=np.linalg.inv(np.cov(obs,rowvar=False)+.02*np.eye(obs.shape[1]))
    return lambda ds:np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',ds['yobs']-m,inv,ds['yobs']-m)))


def predict_ensemble(ds,models):
    pred=np.stack([f(ds) for f in models],axis=0)
    return pred.mean(axis=0),pred.std(axis=0,ddof=1).mean(axis=1)


def extract_features(ds,models,mahal,pred=None,spread=None):
    if pred is None:pred,spread=predict_ensemble(ds,models)
    yy=ds['yobs'];m=base.MO
    # Fit nominal 2-state dynamics to 5 available observations, no future data.
    state=np.linalg.lstsq(m,yy.T,rcond=None)[0]
    fit=np.sqrt(((yy-state.T@m.T)**2).mean(axis=1))
    # Last-observation innovation: fit states using only first 4 records.
    ma=m[:4,:];noise=base.TRAIN_NOISE
    precision=(ma.T@ma)/(noise**2)+np.diag(1/base.PRIOR_VAR)
    gain=np.linalg.solve(precision,ma.T)/(noise**2)
    intercept=np.linalg.solve(precision,base.PRIOR_M/base.PRIOR_VAR)
    state4=yy[:,:4]@gain.T+intercept
    innov=np.abs(yy[:,4]-state4@m[4,:])
    prior=base.physics(ds,False)
    corr=np.sqrt(np.mean((pred-prior)**2,axis=1))
    return np.column_stack([fit,innov,mahal(ds),corr,spread])


class ScaleModel:
    # Predicted |forecast error|max as a function of *past* observations.
    # Learned on labeled scale-fitting set, independently from all calibration sets.
    def __init__(self):pass
    def fit(self,features,maxerr):
        x=np.log1p(np.maximum(features,0))
        self.mu=x.mean(0);self.st=np.maximum(x.std(0),.04)
        z=np.column_stack([np.ones(len(x)),(x-self.mu)/self.st])
        yy=np.log(maxerr+.08)
        penalty=np.diag([0,5,5,5,5,5])
        self.coef=np.linalg.solve(z.T@z+penalty,z.T@yy)
        self.scale_floor=.15;self.scale_ceiling=4.0
        return self
    def __call__(self,features):
        x=np.log1p(np.maximum(features,0));z=np.column_stack([np.ones(len(x)),(x-self.mu)/self.st])
        pred=np.exp(np.clip(z@self.coef,-8,4))
        return np.clip(pred,self.scale_floor,self.scale_ceiling)


def collect(dsets,models,mahal):
    chunks=[]
    for name,ds in dsets.items():
        pred,spread=predict_ensemble(ds,models)
        err=np.max(np.abs(pred-ds['yfut']),axis=1)
        features=extract_features(ds,models,mahal,pred,spread)
        chunks.append((name,err,features,pred,ds))
    return chunks


def main(out):
    out.mkdir(parents=True,exist_ok=True);started=time.time()
    train=base.generate(1200,1101,'train_broad')
    nominal_train=base.generate(1200,1101,'train_nominal')
    models_and_losses=[base.fit_nn(train,s,True,steps=500) for s in SEEDS]
    models=[f for f,_ in models_and_losses];losses=[float(l) for _,l in models_and_losses]
    mahal=fit_mahalanobis(nominal_train)

    # Independent sample partitions. Different data seeds per phase.
    cal_nom=base.generate(400,2202,'matched')
    cal_seeds={name:2203+100*i for i,name in enumerate(KNOWN)}
    cal_known={x:base.generate(400,cal_seeds[x],x) for x in KNOWN}
    scale_seeds={name:2204+100*i for i,name in enumerate(KNOWN)}
    scale_known={x:base.generate(400,scale_seeds[x],x) for x in KNOWN}
    test={x:base.generate(1200,3303,x) for x in SCENARIOS}
    cal_nom_obj=collect({'matched':cal_nom},models,mahal)[0]
    known_chunks=collect(cal_known,models,mahal)
    scale_chunks=collect(scale_known,models,mahal)
    test_chunks=collect(test,models,mahal)

    scaler=ScaleModel().fit(np.vstack([x[2] for x in scale_chunks]),np.concatenate([x[1] for x in scale_chunks]))
    q_nom=quantile(cal_nom_obj[1])
    q_pool=quantile(np.concatenate([x[1] for x in known_chunks]))
    q_worst=max(quantile(x[1]) for x in known_chunks)
    ratios=[(name,errors/scaler(feats)) for name,errors,feats,_,_ in known_chunks]
    q_adaptive_pooled=quantile(np.concatenate([r for _,r in ratios]))
    q_adaptive_worst=max(quantile(r) for _,r in ratios)
    rows=[];diagnostics=[]
    for scene,err,features,pred,ds in test_chunks:
        width_adaptive=scaler(features)
        methods={
           'fixed_nominal':np.full(len(err),q_nom),
           'fixed_pooled_known':np.full(len(err),q_pool),
           'fixed_worst_known':np.full(len(err),q_worst),
           'adaptive_pooled_known':q_adaptive_pooled*width_adaptive,
           'adaptive_worst_known':q_adaptive_worst*width_adaptive,
        }
        for method,radius in methods.items():
            ok=err<=radius
            lo,hi=wilson(len(ok),int(ok.sum()))
            rows.append(dict(scenario=scene,method=method,n=len(ok),coverage=ok.mean(),coverage_wilson_lo=lo,coverage_wilson_hi=hi,
                mean_full_width=float(np.mean(radius)*2),median_full_width=float(np.median(radius)*2),
                p95_full_width=float(np.quantile(radius*2,.95)),forecast_rmse=float(np.sqrt(np.mean((pred-ds['yfut'])**2))),
                failures=int((~ok).sum())))
        diagnostics.append(dict(scenario=scene,mean_observable_scale=float(width_adaptive.mean()),
            p95_observable_scale=float(np.quantile(width_adaptive,.95)),
            mean_maxabs_future_error=float(np.mean(err)),
            corr_scale_future_err=float(np.corrcoef(width_adaptive,err)[0,1])))
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    diag=pd.DataFrame(diagnostics);diag.to_csv(out/'diagnostics.csv',index=False)
    qframe=pd.DataFrame([
      dict(method='fixed_nominal',calibration_groups='matched only',conformal_factor=q_nom,n_calibration=400),
      dict(method='fixed_pooled_known',calibration_groups='matched,in_support,higher_noise pooled',conformal_factor=q_pool,n_calibration=1200),
      dict(method='fixed_worst_known',calibration_groups='max groupwise among matched,in_support,higher_noise',conformal_factor=q_worst,n_calibration=400),
      dict(method='adaptive_pooled_known',calibration_groups='normalized pooled matched,in_support,higher_noise',conformal_factor=q_adaptive_pooled,n_calibration=1200),
      dict(method='adaptive_worst_known',calibration_groups='max normalized groupwise among matched,in_support,higher_noise',conformal_factor=q_adaptive_worst,n_calibration=400),
    ])
    qframe.to_csv(out/'calibration.csv',index=False)
    md={
      'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
      'dependency_sha256':hashlib.sha256(PATH58.read_bytes()).hexdigest(),
      'experiment_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'train_n':1200,'network_seed_values':SEEDS,'optimization_steps_each':500,
      'train_seed':1101,'cal_nom_seed':2202,'cal_known_seeds':cal_seeds,'scale_fit_seeds':scale_seeds,'test_seed':3303,
      'n_cal_nom':400,'n_cal_per_known_group':400,'n_scale_fit_per_known_group':400,'n_test_each_scenario':1200,
      'network_losses':losses,'scale_coef':scaler.coef.tolist(),'scale_mu':scaler.mu.tolist(),'scale_sd':scaler.st.tolist(),
      'method':'five broad-trained M0+NN forecasts; mean prediction; simultaneous four-future-observation intervals',
      'coverage_target':TARGET,'calibration_factors':qframe.set_index('method')['conformal_factor'].to_dict(),
      'known_calibration_scenarios':list(KNOWN),
      'unseen_scenarios':['heldout_combo','outside_support'],
      'critical_limit':'Nothing is calibrated on heldout_combo or outside_support; no guaranteed conditional/groupwise coverage there.',
      'elapsed_seconds':time.time()-started,
      'scope':'Only synthetic passive heating/cooling signal forecasting; no actuators, control or feedback.',
    }
    (out/'run_metadata.json').write_text(json.dumps(md,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAIN LOSSES',losses)
    print('CALIBRATION\n',qframe.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('SUMMARY\n',summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('SCALE DIAGNOSTICS\n',diag.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('ELAPSED',md['elapsed_seconds'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=list(SCENARIOS)
    piv=summary.pivot(index='scenario',columns='method',values='coverage').reindex(order)
    cols=list(qframe.method)
    ax=piv[cols].plot.bar(figsize=(13,5),rot=15)
    ax.axhline(.9,color='black',ls='--',lw=1,label='90% target')
    ax.set_ylabel('Simultaneous coverage across four future measurements')
    ax.set_ylim(0,1.05);ax.set_title('Experiment 60: 90% forecast intervals under shifts')
    ax.grid(axis='y',alpha=.25)
    plt.tight_layout();plt.savefig(out/'coverage.png',dpi=150);plt.close()
    piv=summary.pivot(index='scenario',columns='method',values='mean_full_width').reindex(order)
    ax=piv[cols].plot.bar(figsize=(13,5),rot=15)
    ax.set_ylabel('Mean full interval width (°C)');ax.set_title('Experiment 60: the price of wider forecast intervals')
    ax.grid(axis='y',alpha=.25)
    plt.tight_layout();plt.savefig(out/'width.png',dpi=150);plt.close()

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--outputs-dir',type=Path,default=Path(__file__).resolve().parent)
    args=pa.parse_args();main(args.outputs_dir)