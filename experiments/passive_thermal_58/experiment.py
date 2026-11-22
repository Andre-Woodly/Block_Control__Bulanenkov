"""Experiment 58. Synthetic passive two-state thermal forecasting, no control.
Audit of whether varied-parameter training improves extrapolation for an NN
residual correction to a fixed imperfect physics prior.

Reproduction: python experiment.py --outputs-dir /mnt/data/passive_thermal_58
Models use only observed noisy room temperatures at t=0,5,10,15,20 minutes;
forecast future noisy observations at t=25,30,40,60.
"""
import argparse, datetime, json, platform, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import torch

OBS_T=np.array([0.,5.,10.,15.,20.]); FUT_T=np.array([25.,30.,40.,60.]); H=FUT_T-20
NOM=np.array([.035,.085,.012]); LO=np.array([.028,.066,.009]); HI=np.array([.044,.105,.022])
TRAIN_N=1200; CAL_N=400; TEST_N=1200; SEEDS=[4,17,29,41,53]
TRAIN_NOISE=.20

# Closed form matrix exponential for linear coupled passive thermal ODE:
# dA/dt=-(ka+kc)*A+kc*W; dW/dt=kw*(A-W).
def room_evolution(x0, rates, times):
    ka,kc,kw=rates.T
    a=-ka-kc;d=-kw;htrace=(a+d)/2
    delta=np.sqrt(((a-d)/2)**2+kc*kw)
    z=delta[:,None]*times[None,:]
    amp=np.exp(htrace[:,None]*times[None,:])
    aa=x0[:,0,None]; ww=x0[:,1,None]
    return amp*(np.cosh(z)*aa + (np.sinh(z)/delta[:,None])*(((a-d)/2)[:,None]*aa+kc[:,None]*ww))


def obs_mat(rates,times):
    # exact linear sensitivity to the two initial temperatures under a constant rate triple
    basis=np.array([[1.,0.],[0.,1.]])
    rr=np.repeat(np.array(rates,float)[None,:],2,axis=0)
    e=room_evolution(basis,rr,times)
    return e.T

MO=obs_mat(NOM,OBS_T); MF=obs_mat(NOM,FUT_T)
PRIOR_M=np.array([12.5,15.]); PRIOR_VAR=np.array([15**2/12,30**2/12])

def physics(ds,use_history=False):
    # fixed nominal law, fixed observation-noise assumption, Gaussian state prior
    M=MO if use_history else MO[-1:]
    yy=ds['yobs'] if use_history else ds['yobs'][:,-1:]
    prec=(M.T@M)/(TRAIN_NOISE**2)+np.diag(1/PRIOR_VAR)
    gain=np.linalg.solve(prec,M.T)/(TRAIN_NOISE**2)
    const=np.linalg.solve(prec,PRIOR_M/PRIOR_VAR)
    est=yy@gain.T+const
    return est@MF.T


def generate(n,seed,scenario):
    rng=np.random.default_rng(seed)
    air=rng.uniform(5,20,n)
    wall=rng.uniform(0,30,n)
    if scenario in ('train_broad','in_support'):
        params=rng.uniform(LO,HI,size=(n,3))
    elif scenario=='heldout_combo':
        params=np.tile([.030,.101,.020],(n,1))
    elif scenario=='outside_support':
        params=rng.uniform([.050,.115,.024],[.064,.142,.037],size=(n,3))
    elif scenario in ('matched','higher_noise','train_nominal'):
        params=np.tile(NOM,(n,1))
    else:
        raise ValueError(scenario)
    x0=np.c_[air,wall]
    truth_obs=room_evolution(x0,params,OBS_T)
    truth_fut=room_evolution(x0,params,FUT_T)
    noise=.65 if scenario=='higher_noise' else TRAIN_NOISE
    return {'yobs':truth_obs+rng.normal(0,noise,truth_obs.shape),
            'yfut':truth_fut+rng.normal(0,noise,truth_fut.shape),
            'truth':truth_fut,'rates':params,'scenario':scenario}


def make_x(ds):
    yy=ds['yobs']; n=len(yy); h=H/40
    x=np.broadcast_to(yy[:,None,:],(n,len(H),len(OBS_T)))
    t=np.broadcast_to(h[None,:,None],(n,len(H),1))
    return np.concatenate([x/25,t],axis=-1).reshape(-1,6)


def fit_nn(train,seed,hybrid,steps=500):
    torch.manual_seed(seed); torch.set_num_threads(1)
    x=torch.from_numpy(make_x(train).astype('float32'))
    y=train['yfut']-(physics(train) if hybrid else 0.)
    yt=torch.from_numpy((y/25).reshape(-1,1).astype('float32'))
    net=torch.nn.Sequential(torch.nn.Linear(6,24),torch.nn.Tanh(),torch.nn.Linear(24,24),torch.nn.Tanh(),torch.nn.Linear(24,1))
    opt=torch.optim.Adam(net.parameters(),lr=.008)
    for _ in range(steps):
        opt.zero_grad(); loss=((net(x)-yt)**2).mean();loss.backward();opt.step()
    def predict(ds):
        with torch.no_grad():
            p=net(torch.from_numpy(make_x(ds).astype('float32'))).numpy().reshape(-1,len(H))*25
        return p+physics(ds) if hybrid else p
    return predict,float(loss.detach().item())


def fit_ridge(tr,alpha=1.0):
    # Linear multi-output baseline. Standardized obs; unpenalized intercept.
    x=tr['yobs']/25
    X=np.column_stack([np.ones(len(x)),x]); y=tr['yfut']
    reg=np.diag([0.,*([alpha]*5)])
    beta=np.linalg.solve(X.T@X+reg,X.T@y)
    return lambda ds: np.column_stack([np.ones(len(ds['yobs'])),ds['yobs']/25])@beta


def quantile(scores,alpha=.1):
    n=len(scores);rank=min(n-1,int(np.ceil((n+1)*(1-alpha)))-1)
    return float(np.sort(scores)[rank])


def comparison_boot(a,b,seed=4582,bs=700):
    # Arrays (seeds, trajectories, horizons). Trajectories resampled as clusters,
    # initializations fixed. Identical test trajectories across compared models.
    rng=np.random.default_rng(seed)
    na=np.mean(a*a,axis=-1); nb=np.mean(b*b,axis=-1)
    n=a.shape[1]; steps=200
    values=[]
    for st in range(0,bs,steps):
        ix=rng.integers(n,size=(min(bs-st,steps),n))
        xa=np.sqrt(na[:,ix].mean(axis=2)).mean(axis=0)
        xb=np.sqrt(nb[:,ix].mean(axis=2)).mean(axis=0)
        values.extend((xa-xb).tolist())
    return np.quantile(values,[.025,.975])


def main(dest):
    dest.mkdir(parents=True,exist_ok=True);t0=time.time()
    tr_nom=generate(TRAIN_N,1101,'train_nominal');tr_broad=generate(TRAIN_N,1101,'train_broad')
    cal=generate(CAL_N,2202,'matched')
    scenes={s:generate(TEST_N,3303,s) for s in ['matched','in_support','heldout_combo','outside_support','higher_noise']}
    models={
       'Physics_current':[lambda ds:physics(ds,False)],
       'Physics_history':[lambda ds:physics(ds,True)],
       'Ridge_nominal':[fit_ridge(tr_nom)],
       'Ridge_broad':[fit_ridge(tr_broad)]}
    losses={}
    for training,tr in [('nominal',tr_nom),('broad',tr_broad)]:
        for architecture,hybrid in [('Full_NN',False),('Hybrid_M0_plus_NN',True)]:
            name=f'{architecture}_{training}'
            vals=[fit_nn(tr,s,hybrid) for s in SEEDS]
            models[name]=[p for p,l in vals]
            losses[name]=[l for p,l in vals]
            print('trained',name,'losses',losses[name],flush=True)
    rows=[]; per_h=[]; preds={}
    for name,predictors in models.items():
        for seed,p in enumerate(predictors):
            qc=quantile(np.max(np.abs(p(cal)-cal['yfut']),axis=1))
            for scene,ds in scenes.items():
                yy=p(ds)
                e=yy-ds['yfut']; preds[(name,seed,scene)]=e
                rows.append(dict(method=name,seed=seed,scenario=scene,
                    rmse_observed=float(np.sqrt(np.mean(e*e))),
                    rmse_truth=float(np.sqrt(np.mean((yy-ds['truth'])**2))),
                    coverage=float(np.mean(np.max(abs(e),axis=1)<=qc)),
                    calibration_halfwidth=qc))
                for j,h in enumerate(H):
                    per_h.append(dict(method=name,seed=seed,scenario=scene,horizon_min=int(h),
                                      rmse_observed=float(np.sqrt(np.mean(e[:,j]**2)))))
    runs=pd.DataFrame(rows); runs.to_csv(dest/'runs.csv',index=False)
    summary=runs.groupby(['scenario','method'],as_index=False).agg(
       rmse_obs=('rmse_observed','mean'),rmse_true=('rmse_truth','mean'),
       coverage=('coverage','mean'),q=('calibration_halfwidth','mean'),n_seeds=('seed','nunique'))
    summary.to_csv(dest/'summary.csv',index=False)
    pd.DataFrame(per_h).groupby(['scenario','method','horizon_min'],as_index=False).rmse_observed.mean().to_csv(dest/'horizons.csv',index=False)
    bootstrap=[]
    comparisons=[('Hybrid_M0_plus_NN_nominal','Hybrid_M0_plus_NN_broad'),
                 ('Full_NN_nominal','Full_NN_broad'),
                 ('Full_NN_broad','Hybrid_M0_plus_NN_broad')]
    for scene in scenes:
        for first,second in comparisons:
            a=np.stack([preds[(first,s,scene)] for s in range(len(SEEDS))]);b=np.stack([preds[(second,s,scene)] for s in range(len(SEEDS))])
            lo,hi=comparison_boot(a,b)
            md=np.mean([np.sqrt((e*e).mean()) for e in a])-np.mean([np.sqrt((e*e).mean()) for e in b])
            bootstrap.append(dict(scenario=scene,comparison=f'{first} - {second}',delta_rmse=md,ci95_lo=lo,ci95_hi=hi))
    pd.DataFrame(bootstrap).to_csv(dest/'bootstrap.csv',index=False)
    meta={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
        'scipy':scipy.__version__,'pandas':pd.__version__, 'train_n_each_domain':TRAIN_N,
        'calibration_n':CAL_N,'test_n_each_scenario':TEST_N,'seed_values':SEEDS,
        'training_steps_each_nn':500,'training_noise_sigma':TRAIN_NOISE,
        'rate_order':['ka','kc','kw'],'nominal_rates':NOM.tolist(),
        'broad_support_low':LO.tolist(),'broad_support_high':HI.tolist(),
        'training_losses':losses,'elapsed_seconds':time.time()-t0,
        'scope':'Synthetic passive thermal forecasting only. No actuators, control, or feedback.'}
    (dest/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print('SUMMARY\n',summary.to_string(index=False,float_format=lambda v:f'{v:.5f}'),flush=True)
    print('BOOTSTRAP\n',pd.DataFrame(bootstrap).to_string(index=False,float_format=lambda v:f'{v:.5f}'),flush=True)
    import matplotlib.pyplot as plt
    order=['matched','in_support','heldout_combo','outside_support','higher_noise']
    p=summary.pivot(index='scenario',columns='method',values='rmse_obs').reindex(order)
    p[['Physics_current','Physics_history','Ridge_broad','Full_NN_broad',
       'Hybrid_M0_plus_NN_nominal','Hybrid_M0_plus_NN_broad']].plot.bar(figsize=(13,5),rot=15)
    plt.ylabel('RMSE versus observed future temperature (°C)');plt.grid(axis='y',alpha=.25)
    plt.title('Passive thermal dynamics: nominal versus mixed-parameter training')
    plt.tight_layout();plt.savefig(dest/'comparison.png',dpi=160)
    print('METADATA',meta['elapsed_seconds'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outputs-dir',default=str(Path(__file__).resolve().parent))
    args=parser.parse_args();main(Path(args.outputs_dir))