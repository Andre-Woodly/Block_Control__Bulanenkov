"""Experiment 57: passive 2-state thermal process with a hidden wall temperature.
This is ONLY a synthetic passive forecasting task. No actuators, feedback or control.

Inputs: five noisy measurements of room temperature excess at t=0,5,10,15,20 min.
Targets: future independently noisy observations at t=25,30,40,60 min.
Compare one-point vs historical NN, a NN correction to a fixed physics prior,
plus a physics-based Bayesian state reconstruction under both information sets.
"""
from pathlib import Path
import platform, json, datetime
import numpy as np
import pandas as pd
from scipy.linalg import expm
import scipy
import torch

BASE=Path(__file__).resolve().parent
OBS_T=np.array([0.,5.,10.,15.,20.]);H=np.array([5.,10.,20.,40.])
NTRAIN=1000; NCAL=300; NTEST=1000; SEEDS=[4,17,29,41,53]
TRAIN_NOISE=0.2
NOM={'ka':0.035,'kc':0.085,'kw':0.012}

def mat(ka,kc,kw):
    return np.array([[-ka-kc,kc],[kw,-kw]],float)

def matrices(params):
    A=mat(**params)
    obs=np.array([expm(A*t)[0] for t in OBS_T])
    future=np.array([expm(A*(20+h))[0] for h in H])
    return obs,future

def generate(n,seed,scenario='matched'):
    rg=np.random.default_rng(seed)
    params=NOM if scenario!='rates_shift' else {'ka':0.043,'kc':0.100,'kw':0.023}
    obs, fut=matrices(params)
    a0=rg.uniform(5,20,size=n)
    w0=rg.uniform(0,30,size=n) if scenario!='warm_walls' else rg.uniform(20,30,size=n)
    ini=np.stack([a0,w0],axis=1)
    trueobs=ini@obs.T
    truefuture=ini@fut.T
    sigma=0.65 if scenario=='higher_noise' else TRAIN_NOISE
    yobs=trueobs+rg.normal(0,sigma,size=trueobs.shape)
    yfut=truefuture+rg.normal(0,sigma,size=truefuture.shape)
    return {'yobs':yobs,'yfut':yfut,'truth':truefuture,'sigma':sigma}

# Gaussian prior approximates the independent uniform training distribution.
PRIOR_M=np.array([12.5,15.0])
PRIOR_VAR=np.array([15**2/12,30**2/12])
O_M,F_M=matrices(NOM)

def physical(ds,use_history):
    # Posterior mean of initial state given one or five noisy observations,
    # nominal model, nominal measurement variance, and fixed Gaussian prior.
    if use_history:
        M=O_M; yy=ds['yobs']
    else:
        M=O_M[-1:];yy=ds['yobs'][:,-1:]
    prec=M.T@M/(TRAIN_NOISE**2)+np.diag(1/PRIOR_VAR)
    gain=np.linalg.solve(prec,M.T)/(TRAIN_NOISE**2)
    const=np.linalg.solve(prec,PRIOR_M/PRIOR_VAR)
    state=yy@gain.T+const
    return state@F_M.T

def input_matrix(ds,history):
    y=ds['yobs']
    x=y if history else y[:,-1:]
    grid=np.broadcast_to(x[:,None,:],(len(y),len(H),x.shape[1]))
    h=np.broadcast_to((H[None,:,None]/40),(len(y),len(H),1))
    return np.concatenate([grid/25,h],axis=-1).reshape(-1,x.shape[1]+1)

def train_nn(tr,seed,history,hybrid):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x=torch.tensor(input_matrix(tr,history),dtype=torch.float32)
    baseline=physical(tr,False) if hybrid else np.zeros_like(tr['yfut'])
    y=torch.tensor(((tr['yfut']-baseline)/25).reshape(-1,1),dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],24),torch.nn.Tanh(),
                           torch.nn.Linear(24,24),torch.nn.Tanh(),torch.nn.Linear(24,1))
    optim=torch.optim.Adam(net.parameters(),lr=0.008)
    for t in range(1500):
        optim.zero_grad();loss=torch.mean((net(x)-y)**2);loss.backward();optim.step()
    def pred(ds):
        with torch.no_grad():
            out=net(torch.tensor(input_matrix(ds,history),dtype=torch.float32)).numpy().reshape(-1,len(H))*25
        return out+physical(ds,False) if hybrid else out
    return pred,float(loss.detach())

def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))

def covered(pred,ds,q):
    return np.mean(np.max(np.abs(pred-ds['yfut']),axis=1)<=q)

def quantile(scores,alpha=.1):
    n=len(scores);rank=min(n-1,int(np.ceil((n+1)*(1-alpha)))-1)
    return float(np.sort(scores)[rank])

def boot_gap(e_current,e_history,rng,bs=700):
    # Cluster bootstrap by independent test trajectory; neural initializations fixed.
    s1=np.mean(e_current**2,axis=2)  # (5 seeds, n trajectories)
    s2=np.mean(e_history**2,axis=2)
    n=s1.shape[1]
    ix=rng.integers(0,n,size=(bs,n))
    rmse1=np.sqrt(np.mean(s1[:,ix],axis=2)).mean(axis=0)
    rmse2=np.sqrt(np.mean(s2[:,ix],axis=2)).mean(axis=0)
    return np.quantile(rmse1-rmse2,[.025,.975])

def main():
    tr=generate(NTRAIN,1101)
    ca=generate(NCAL,2202)
    tests={name:generate(NTEST,3303,name) for name in
           ['matched','higher_noise','warm_walls','rates_shift']}
    learned={}
    losses={}
    for name,his,hyb in [('NN_current',False,False),('NN_history',True,False),('Hybrid_history',True,True)]:
        learned[name]=[];losses[name]=[]
        for seed in SEEDS:
            p,l=train_nn(tr,seed,his,hyb);learned[name].append(p);losses[name].append(l)
    rows=[]; predictions={}; horizon_rows=[]
    for method in ['Physics_current','Physics_history','NN_current','NN_history','Hybrid_history']:
        predictors=[(lambda ds:physical(ds,method=='Physics_history'))] if method.startswith('Physics') else learned[method]
        for seed,predfun in enumerate(predictors):
            cal_pred=predfun(ca)
            q=quantile(np.max(np.abs(cal_pred-ca['yfut']),axis=1))
            for scene,ds in tests.items():
                pred=predfun(ds)
                predictions[(method,seed,scene)]=pred
                rows.append(dict(method=method,seed=seed,scene=scene,
                    rmse_observed=rmse(pred,ds['yfut']),rmse_truth=rmse(pred,ds['truth']),
                    coverage_all_horizons=float(covered(pred,ds,q)),calibration_halfwidth=q))
                for j,h in enumerate(H):
                    horizon_rows.append(dict(method=method,seed=seed,scene=scene,horizon_min=int(h),
                        rmse_observed=rmse(pred[:,j],ds['yfut'][:,j])))
    runs=pd.DataFrame(rows); runs.to_csv(BASE/'runs.csv',index=False)
    summary=runs.groupby(['scene','method'],as_index=False).agg(
        rmse_obs=('rmse_observed','mean'),rmse_true=('rmse_truth','mean'),
        coverage=('coverage_all_horizons','mean'),halfwidth=('calibration_halfwidth','mean'),
        seeds=('seed','nunique'))
    summary.to_csv(BASE/'summary.csv',index=False)
    pd.DataFrame(horizon_rows).groupby(['scene','method','horizon_min'],as_index=False).rmse_observed.mean().to_csv(BASE/'horizons.csv',index=False)
    boot=[]
    for scene,ds in tests.items():
        for method1,method2 in [('NN_current','NN_history'),('NN_history','Hybrid_history')]:
            e1=np.stack([predictions[(method1,i,scene)]-ds['yfut'] for i in range(5)])
            e2=np.stack([predictions[(method2,i,scene)]-ds['yfut'] for i in range(5)])
            ci=boot_gap(e1,e2,np.random.default_rng(5534))
            boot.append(dict(scene=scene,contrast=f'{method1}_minus_{method2}',
                mean_delta=float(np.sqrt(np.mean(e1**2,axis=(1,2))).mean()-np.sqrt(np.mean(e2**2,axis=(1,2))).mean()),
                ci95_low=float(ci[0]),ci95_high=float(ci[1])))
    pd.DataFrame(boot).to_csv(BASE/'bootstrap.csv',index=False)
    meta={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':SEEDS,
        'n_train':NTRAIN,'n_calibrate':NCAL,'n_test_per_scene':NTEST,
        'obs_t_minutes':OBS_T.tolist(),'forecast_h_minutes':H.tolist(),
        'nominal_rates':NOM,'train_noise_sigma':TRAIN_NOISE,'NN_optim_steps':1500,
        'training_losses':losses,'python':platform.python_version(),'numpy':np.__version__,
        'torch':torch.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
        'description':'Synthetic passive thermal model. No actuator, no closed loop.'}
    (BASE/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print('\nSUMMARY\n',summary.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nBOOTSTRAP\n',pd.DataFrame(boot).to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nLOSSES\n',losses)
    import matplotlib.pyplot as plt
    for scene in ['matched','higher_noise','warm_walls','rates_shift']:
        tmp=summary[summary.scene==scene]
        plt.figure(figsize=(9,4.5)); plt.bar(tmp.method,tmp.rmse_obs)
        plt.xticks(rotation=19,ha='right');plt.ylabel('RMSE vs future measurements, °C')
        plt.title('Passive hidden-state thermal system: '+scene)
        plt.grid(axis='y',alpha=.3);plt.tight_layout()
        plt.savefig(BASE/('plot_'+scene+'.png'),dpi=160);plt.close()
if __name__=='__main__':main()