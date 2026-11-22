"""Passive cooling reliability audit; synthetic forecasts only, no control.
Follows experiments 51-55. Predictions from a single noisy initial temperature.
Calibration uses observed future measurements, never simulator-only clean temperatures.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
T=np.array([10.,20.,30.,40.,50.,60.]); TRAIN_A=.02; B=.0008


def law(d,a,b):
    e=np.exp(-a*T[None,:]); return d[:,None]*e/(1+(b/a)*d[:,None]*(1-e))


def dataset(n,seed,a=TRAIN_A,b=B,noise=.1):
    rg=np.random.default_rng(seed)
    init=rg.uniform(5,25,n)
    initial_obs=init+rg.normal(0,noise,n)
    truth=law(init,a,b)
    future_obs=truth+rg.normal(0,noise,truth.shape)
    X=np.stack(np.broadcast_arrays(initial_obs[:,None]/25,T[None,:]/60),axis=-1)
    prior=law(initial_obs,TRAIN_A,0)
    physical_train=law(initial_obs,TRAIN_A,B)
    reference_current=law(initial_obs,a,b)
    return dict(x=X,initial_obs=initial_obs,truth=truth,y=future_obs,
        prior=prior,physical_train=physical_train,reference_current=reference_current)


def fit(tr,seed,hybrid):
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.Linear(2,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    x=torch.tensor(tr['x'].reshape(-1,2),dtype=torch.float32)
    target=(tr['y']-tr['prior'] if hybrid else tr['y'])/25
    y=torch.tensor(target.reshape(-1,1),dtype=torch.float32)
    opt=torch.optim.Adam(net.parameters(),lr=.01)
    for i in range(1000):
        opt.zero_grad(); loss=((net(x)-y)**2).mean(); loss.backward(); opt.step()
    def predict(ds):
        with torch.no_grad():
            p=net(torch.tensor(ds['x'].reshape(-1,2),dtype=torch.float32)).numpy().reshape(ds['y'].shape)*25
        return p+ds['prior'] if hybrid else p
    return predict,float(loss.detach())


def quantile(scores,alpha=.1):
    n=len(scores); idx=int(np.ceil((n+1)*(1-alpha)))-1
    return np.sort(scores)[min(idx,n-1)]


def bootstrap_difference(a,b,seed=54321,reps=1200):
    # two predictions of shape (5 initializations, 1200 trajectories, 6 horizons)
    # each trajectory is one resampling cluster; NN initializations held fixed.
    n=a.shape[1]; rg=np.random.default_rng(seed)
    boot=[]
    for j in range(reps):
        ix=rg.integers(n,size=n)
        # Each RMSE averages over network seeds and forecast horizons.
        boot.append(np.sqrt(np.mean(a[:,ix,:]**2,axis=(1,2))).mean()-np.sqrt(np.mean(b[:,ix,:]**2,axis=(1,2))).mean())
    return np.quantile(boot,[.025,.975])


def main():
    tr=dataset(120,10101)
    ca=dataset(300,20202)
    scenes={'matched':dataset(1200,30303),
        'higher_noise':dataset(1200,30303,noise=.5),
        'faster_cooling':dataset(1200,30303,a=.03)}
    nets={name:[fit(tr,seed,hybrid=(name=='Hybrid')) for seed in range(5)]
        for name in ('Full_NN','Hybrid')}
    print('NN training losses:', {k:[round(m[1],7) for m in v] for k,v in nets.items()})
    rows=[]; per_h=[]; residuals={}
    methods=['Linear_prior','Known_train_law','Full_NN','Hybrid','Current_law_reference']
    for method in methods:
        reps=5 if method in nets else 1
        for seed in range(reps):
            def predict(ds):
                if method=='Linear_prior': return ds['prior']
                if method=='Known_train_law': return ds['physical_train']
                if method=='Current_law_reference': return ds['reference_current']
                return nets[method][seed][0](ds)
            calibrate_error=predict(ca)-ca['y']
            q=quantile(np.max(np.abs(calibrate_error),axis=1))
            for scene,ds in scenes.items():
                pred=predict(ds)
                e=pred-ds['y']  # honest observed future, not latent truth
                rmse=np.sqrt(np.mean(e**2))
                coverage=np.mean(np.max(np.abs(e),axis=1)<=q)
                rows.append(dict(method=method,seed=seed,scene=scene,rmse=rmse,
                    coverage=coverage,halfwidth=q,mean_width=2*q,mae=np.mean(np.abs(e))))
                for j,t in enumerate(T):
                    per_h.append(dict(method=method,seed=seed,scene=scene,
                        horizon_min=int(t),rmse=float(np.sqrt(np.mean(e[:,j]**2)))))
                residuals[(method,seed,scene)]=e
    run=pd.DataFrame(rows); run.to_csv(OUT/'runs.csv',index=False)
    summary=run.groupby(['method','scene'],as_index=False).agg(rmse=('rmse','mean'),
        coverage=('coverage','mean'),halfwidth=('halfwidth','mean'),
        seeds=('seed','nunique'),mae=('mae','mean'))
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(per_h).groupby(['method','scene','horizon_min'],as_index=False).rmse.mean().to_csv(OUT/'horizons.csv',index=False)
    # test example: NN vs hybrid, evaluate paired test trajectories
    out=[]
    for scene in scenes:
        full=np.stack([residuals[('Full_NN',s,scene)] for s in range(5)])
        hybr=np.stack([residuals[('Hybrid',s,scene)] for s in range(5)])
        dlo,dhi=bootstrap_difference(full,hybr,seed=7444)
        out.append(dict(scene=scene,mean_rmse_full=np.mean([np.sqrt(np.mean(x*x)) for x in full]),
            mean_rmse_hybrid=np.mean([np.sqrt(np.mean(x*x)) for x in hybr]),
            ci95_paired_test_difference_full_minus_hybrid_low=dlo,
            ci95_paired_test_difference_full_minus_hybrid_high=dhi))
    pd.DataFrame(out).to_csv(OUT/'bootstrap_comparison.csv',index=False)
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.5f}'))
    print(pd.DataFrame(out).to_string(index=False,float_format=lambda v:f'{v:.5f}'))
    import matplotlib.pyplot as plt
    p=summary.pivot(index='scene',columns='method',values='rmse').reindex(['matched','higher_noise','faster_cooling'])
    p.plot.bar(figsize=(12,5),rot=0)
    plt.title('Passive cooling: observed-future RMSE; training law fixed under shifts')
    plt.ylabel('RMSE (°C)');plt.grid(axis='y',alpha=.3);plt.tight_layout();plt.savefig(OUT/'comparison.png',dpi=160)

if __name__=='__main__': main()