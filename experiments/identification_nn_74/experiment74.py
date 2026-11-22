"""Checkpoint 14, experiment 74: estimation-error separation vs NN residual.
BENIGN toy two-temperature model only, not a physical actuator.
Three causal 18-transition estimation methods compared under identical feedback study.
DO NOT interpret results as safety/performance guarantees for real systems.
"""
from __future__ import annotations
import json,time,platform,hashlib,datetime,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
import source72 as m

GROUPS=(101,303,509)
INIT=(7,19)
WINDOWS=(2,18)
ESTIMATORS=('raw','filtered3','moment_adjusted')
SCENARIOS=('nominal','compound_shift','higher_noise')
KINDS=('constant','piecewise')
METHODS=('noNN','NN_L2','NN_L18')
NTRAIN=120;NTEST=70; TRAIN_STEPS=65;EVAL_STEPS=68;WARM=30
STEPS=180; WIDTH=24
DT=m.DT

def estimator(ys,us,sigma,method):
    """Reduced-order 18-transition regressions. None recovers all physical parameters."""
    n=ys[-1].size;t=min(len(us),m.PARAM_WINDOW)
    if t<4:return np.full(n,m.A0),np.full(n,m.G0)
    if method=='filtered3':
        # causal 3-point moving average applied to BOTH y and u, preserves causality
        if len(ys)>=t+3 and len(us)>=t+2:
            yr=np.stack(ys[-t-3:]);ur=np.stack(us[-t-2:])
            ysmooth=(yr[:-2]+yr[1:-1]+yr[2:])/3.
            ya=ysmooth[:-1];yb=ysmooth[1:]
            uu=(ur[:-2]+ur[1:-1]+ur[2:])/3.
        else:
            yf=[np.mean(np.stack(ys[max(i-2,0):i+1]),axis=0) for i in range(max(0,len(ys)-t-3),len(ys))]
            uf=[np.mean(np.stack(us[max(i-2,0):i+1]),axis=0) for i in range(max(0,len(us)-t-2),len(us))]
            ya=np.stack(yf[-t-1:-1]);yb=np.stack(yf[-t:]);uu=np.stack(uf[-t:])
    else:
        ya=np.stack(ys[-t-1:-1]);yb=np.stack(ys[-t:]);uu=np.stack(us[-t:])
    x0=-ya;x1=uu;z=(yb-ya)/DT
    s00=np.sum(x0*x0,axis=0)+m.RIDGE_A
    s11=np.sum(x1*x1,axis=0)+m.RIDGE_G
    s01=np.sum(x0*x1,axis=0)
    b0=np.sum(x0*z,axis=0)+m.RIDGE_A*m.A0
    b1=np.sum(x1*z,axis=0)+m.RIDGE_G*m.G0
    if method=='moment_adjusted':
        # Correct two analytically known noise moments. Not an unbiased closed-loop estimator:
        # when u depends on noisy y, correlation in u*z remains; omitted states remain.
        s00-=t*sigma*sigma
        b0-=t*sigma*sigma/DT
    det=s00*s11-s01*s01
    ah=(b0*s11-b1*s01)/det
    gh=(b1*s00-b0*s01)/det
    return np.clip(ah,.05,.6),np.clip(gh,.3,1.3)

def train_set(seed):
    rng=np.random.default_rng(seed)
    domains=[(m.NOM,.018),({**m.NOM,'a':.17,'g':.70},.018),({**m.NOM,'b':.15,'c':.115},.018)]
    data={k:[[],[]] for k in ESTIMATORS}
    for p,sigma in domains:
        n=NTRAIN//len(domains);A=rng.uniform(-.6,.6,n);W=rng.uniform(-.8,.8,n)
        eta=rng.normal(0,sigma,(WARM+TRAIN_STEPS+1,n))
        dist=rng.normal(0,.035,(WARM+TRAIN_STEPS,n))
        ys=[A+eta[0]];us=[];ref=rng.uniform(-.7,.7,(n,10))
        for t in range(WARM+TRAIN_STEPS):
            target=ref[:,min(t//10,9)]
            pred=m.baseline(ys[-1][:,None],m.ACTIONS[None,:],m.A0,m.G0)
            u=m.ACTIONS[np.argmin((pred-target[:,None])**2+.009*m.ACTIONS[None,:]**2,axis=1)]
            vals={e:estimator(ys,us,sigma,e) for e in ESTIMATORS}
            A,W=m.system_step(A,W,u,p,dist[t]);newobs=A+eta[t+1]
            if t>=WARM:
                for e,(a,g) in vals.items():
                    data[e][0].append(m.make_features(ys,us,a,g,u))
                    data[e][1].append(newobs-m.baseline(ys[-1],u,a,g))
            ys.append(newobs);us.append(u)
    return {e:(np.concatenate(v[0]),np.concatenate(v[1])) for e,v in data.items()}

def choose_prediction(ys,us,a,g,kind,p,nets):
    n=len(ys[-1]);nc=len(m.ACTIONS)
    U=np.broadcast_to(m.ACTIONS[None,:],(n,nc))
    pred=m.baseline(np.repeat(ys[-1],nc),U.reshape(-1),np.repeat(a,nc),np.repeat(g,nc))
    if kind!='noNN':
        L=2 if kind=='NN_L2' else 18
        feat=m.make_features(ys,us,a,g,np.zeros(n))
        ff=np.repeat(feat,nc,axis=0);ff[:,-3]=U.reshape(-1)
        pred+=nets[L](m.mask_history(ff,L))
    return U,pred.reshape(n,nc)

def simulate(p,sigma,scenario,est,method,nets,kind,seed,refval=.45):
    rng=np.random.default_rng(seed)
    n=NTEST;A=rng.uniform(-.35,.35,n);W=rng.uniform(-.45,.45,n)
    steps=WARM+EVAL_STEPS
    eta=rng.normal(0,sigma,(steps+1,n));dist=np.clip(rng.normal(0,.035,(steps,n)),-.06,.06)
    ys=[A+eta[0]];us=[]
    for k in range(WARM):
        u=np.clip(-.85*ys[-1],-1,1)
        A,W=m.system_step(A,W,u,p,dist[k]);ys.append(A+eta[k+1]);us.append(u)
    errors=[];urows=[];esterr_a=[];esterr_g=[];clamps=[];trace=None
    for k in range(EVAL_STEPS):
        if est=='oracle':a=np.full(n,p['a']+p['c']);g=np.full(n,p['g'])
        else:a,g=estimator(ys,us,sigma,est)
        U,pr=choose_prediction(ys,us,a,g,method,p,nets)
        ref=refval if kind=='constant' else (0.,.45,-.35,.62,-.2)[min(k//18,4)]
        u=U[np.arange(n),np.argmin((pr-ref)**2+.009*U**2,axis=1)]
        A,W=m.system_step(A,W,u,p,dist[WARM+k]);ys.append(A+eta[WARM+k+1]);us.append(u)
        errors.append(A-ref);urows.append(u)
        esterr_a.append(a-(p['a']+p['c']));esterr_g.append(g-p['g'])
        clamps.append((a>=.59999)|(a<=.05001)|(g>=1.29999)|(g<=.30001))
    E=np.stack(errors,axis=1);U=np.stack(urows,axis=1)
    AE=np.stack(esterr_a);GE=np.stack(esterr_g);CL=np.stack(clamps)
    traj_rmse=np.sqrt(np.mean(E*E,axis=1))
    out=dict(rmse=float(np.sqrt(np.mean(E*E))),rmse_last40=float(np.sqrt(np.mean(E[:,-40:]**2))),
     mae=float(np.mean(abs(E))),iae=float(DT*np.sum(np.mean(abs(E),axis=1))),
     itae=float(DT**2*np.mean(np.sum(abs(E)*np.arange(1,E.shape[1]+1)[None,:],axis=1))),
     p95_abs=float(np.quantile(abs(E),.95)),action_change=float(np.mean(abs(np.diff(U,axis=1)))),
     mean_u2=float(np.mean(U*U)),saturation=float(np.mean(abs(U)>=.999)),
     mean_est_a=float(np.mean(AE)),mae_est_a=float(np.mean(abs(AE))),
     mean_est_g=float(np.mean(GE)),mae_est_g=float(np.mean(abs(GE))),
     clip_frac=float(np.mean(CL)),max_physical=float(max(np.max(abs(A)),np.max(abs(W)))))
    return out,traj_rmse,(E[0].copy(),U[0].copy())

def stationary_noise(seed=747):
    rng=np.random.default_rng(seed);n=40000;t=18;rows=[]
    for sigma in (0.,.018,.085):
        y=[s.copy() for s in rng.normal(0,sigma,(t+1,n))]
        u=[np.zeros(n) for _ in range(t)]
        for est in ESTIMATORS:
            a,g=estimator(y,u,sigma,est)
            rows.append(dict(noise=sigma,estimator=est,mean_a=float(np.mean(a)),std_a=float(np.std(a)),
                             p95_a=float(np.quantile(a,.95)),clipped_a_fraction=float(np.mean(a>=.59999)),
                             g_not_identifiable=True,mean_g=float(np.mean(g))))
    return pd.DataFrame(rows)

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    stat=stationary_noise();stat.to_csv(out/'noise_stress.csv',index=False)
    models={};training=[]
    for group,seed in enumerate(GROUPS):
        dataset=train_set(seed)
        for est in ESTIMATORS:
            X,y=dataset[est]
            for init in INIT:
                nets={}
                for L in WINDOWS:
                    model,loss=m.fit_nn(X,y,L,seed=init+group*101)
                    nets[L]=model
                    training.append(dict(group=group,seed=seed,init=init,estimator=est,window=L,n_samples=len(y),train_mse=loss))
                models[(group,init,est)]=nets
        print('Trained',group,'elapsed',round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(training).to_csv(out/'training.csv',index=False)
    rows=[];per=[];traces={}
    # One input realization per scenario & reference shape; it is shared across all compared models.
    for scenario in SCENARIOS:
        p,sigma=m.SCENARIOS[scenario];si=SCENARIOS.index(scenario)
        for kind in KINDS:
            seed=74000+si*1001+(0 if kind=='constant' else 971)
            for group in range(len(GROUPS)):
                for init in INIT:
                    for est in ESTIMATORS:
                        for method in METHODS:
                            metrics,individual,trace=simulate(p,sigma,scenario,est,method,models[(group,init,est)],kind,seed)
                            rows.append(dict(scenario=scenario,kind=kind,group=group,init=init,estimator=est,method=method,**metrics))
                            for i,z in enumerate(individual):per.append(dict(scenario=scenario,kind=kind,group=group,init=init,estimator=est,method=method,trajectory=i,traj_rmse=float(z)))
                            if group==0 and init==7:traces[(scenario,kind,est,method)]=trace
                    # Oracle is diagnostic ONLY, not available to practical estimator
                    metrics,individual,trace=simulate(p,sigma,scenario,'oracle','noNN',None,kind,seed)
                    rows.append(dict(scenario=scenario,kind=kind,group=group,init=init,estimator='oracle',method='noNN',**metrics))
                    for i,z in enumerate(individual):per.append(dict(scenario=scenario,kind=kind,group=group,init=init,estimator='oracle',method='noNN',trajectory=i,traj_rmse=float(z)))
        print('Tested',scenario,'elapsed',round(time.monotonic()-start,1),flush=True)
    runs=pd.DataFrame(rows);runs.to_csv(out/'runs.csv',index=False)
    summ=runs.groupby(['scenario','kind','estimator','method'],as_index=False)[['rmse','rmse_last40','mae','iae','itae','p95_abs','action_change','mean_u2','saturation','mean_est_a','mae_est_a','mean_est_g','mae_est_g','clip_frac']].mean()
    summ.to_csv(out/'summary.csv',index=False)
    per=pd.DataFrame(per);per.to_csv(out/'per_trajectory.csv',index=False)
    # Postprocess previously saved raw runs: paired grouped bootstrap, charts and metadata.
    import subprocess,sys
    subprocess.run([sys.executable,str(Path(__file__).with_name('finish.py')),str(out)],check=True)
    print('FINISHED all stages, seconds',round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('/mnt/data/exp74'))
    opt=ap.parse_args();main(opt.out)