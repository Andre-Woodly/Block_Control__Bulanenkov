"""Reproducible EDUCATIONAL experiment 61-64 + CL-01.

Two toy thermal chambers with bounded heating/cooling, NOT an electrohydraulic or
flight-control model. Tests direct forecasting, generalization, independent
training samples and supervised feedback of a setpoint. Not a safety validation.
Usage: python experiment.py --out DIR
"""
import argparse, time, datetime, json, platform, hashlib, csv
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch

DT=.2
G0=.8
A0=.19
STEPS=100
TRAIN_GROUPS=[111,222,333]
INIT_SEEDS=[7,19,31,43]
HORIZONS=[1,3,6,10]
FAMS={
 'linear_two_state':{'a':.16,'c':.24,'w':.055,'q':0.,'g':.85},
 'nonlinear_two_state':{'a':.22,'c':.28,'w':.05,'q':.12,'g':.75}
}
REGIMES=['matched','coupling_shift','higher_noise','nonlinear_shift','compound_shift']
NTRAIN=180; TTRAIN=55; NTEST=120

def params(family,regime):
    p=dict(FAMS[family]); noise=.045
    if regime=='coupling_shift':p['c']*=1.60;p['w']*=.70
    if regime=='higher_noise':noise=.16
    if regime=='nonlinear_shift':p['q']+=.24
    if regime=='compound_shift':p['a']*=1.40;p['c']*=1.45;p['w']*=.65;p['g']*=.75;p['q']+=.17
    return p,noise

def transition(a,w,u,p):
    an=a+DT*(-p['a']*a+p['c']*(w-a)-p['q']*a**3+p['g']*u)
    wn=w+DT*p['w']*(a-w)
    return an,wn

def m0(y,u):
    return y+DT*(-A0*y+G0*u)

def feat(y,ui):
    # y = [y_t,y_(t-1),y_(t-2)], ui=[u_t,u_(t-1),u_(t-2)]
    return np.concatenate([y,ui],axis=-1)

def create_trajectories(family, regime, n, length, seed):
    rng=np.random.default_rng(seed)
    p,noise=params(family,regime)
    a=rng.uniform(-1.0,1.0,n); w=rng.uniform(-1.1,1.1,n)
    true=np.empty((n,length+1));obs=np.empty((n,length+1));uc=np.empty((n,length))
    true[:,0]=a;obs[:,0]=a+rng.normal(0,noise,n)
    prev=np.zeros(n)
    for t in range(length):
        prev=np.clip(.52*prev+.48*rng.normal(0,.9,n),-1,1)
        uc[:,t]=prev
        a,w=transition(a,w,prev,p)
        true[:,t+1]=a;obs[:,t+1]=a+rng.normal(0,noise,n)
    return {'observed':obs,'true':true,'u':uc,'initial_a':true[:,0]}

def make_train_data(data):
    y=data['observed'];u=data['u'];X=[];Z=[]
    for t in range(2,u.shape[1]):
        Y=np.column_stack((y[:,t],y[:,t-1],y[:,t-2]))
        U=np.column_stack((u[:,t],u[:,t-1],u[:,t-2]))
        X.append(feat(Y,U));Z.append(y[:,t+1])
    return np.concatenate(X),np.concatenate(Z)

def poly_map(X):
    # predeclared, small interpretable basis, SAME inputs for all methods
    return np.column_stack((X,X[:,0]**2,X[:,0]**3,X[:,1]**2,X[:,1]**3,
                            X[:,0]*X[:,1],X[:,0]*X[:,2],X[:,0]*X[:,3]))

class LinearPred:
    def __init__(self,mat,beta):self.mat=mat;self.beta=beta
    def __call__(self,X):return self.mat(X)@self.beta

class NumpyMLP:
    def __init__(self,net,hybrid):
        self.ws=[];self.bs=[];self.hybrid=hybrid
        for layer in net:
            if isinstance(layer,torch.nn.Linear):
                self.ws.append(layer.weight.detach().numpy().copy().T)
                self.bs.append(layer.bias.detach().numpy().copy())
    def __call__(self,X):
        out=X.astype(np.float64,copy=False)
        for i,(w,b) in enumerate(zip(self.ws,self.bs)):
            out=out@w+b
            if i<len(self.ws)-1:out=np.tanh(out)
        res=out[:,0]
        return res+m0(X[:,0],X[:,3]) if self.hybrid else res

def fit_network(X,y,seed,hybrid,iterations=230):
    torch.manual_seed(seed);torch.set_num_threads(1)
    inputs=torch.tensor(X,dtype=torch.float32)
    target=(y-m0(X[:,0],X[:,3])) if hybrid else y
    labels=torch.tensor(target[:,None],dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(6,24),torch.nn.Tanh(),
        torch.nn.Linear(24,24),torch.nn.Tanh(),torch.nn.Linear(24,1))
    opt=torch.optim.Adam(net.parameters(),lr=.008)
    for k in range(iterations):
        opt.zero_grad()
        loss=((net(inputs)-labels)**2).mean()
        loss.backward();opt.step()
    return NumpyMLP(net,hybrid),float(loss.detach())

def fit_models(X,y,group,init):
    ones=np.ones((len(X),1))
    methods={}
    for key,map_x in [('ARX_linear',lambda x:x),('ARX_poly',poly_map)]:
        XX=np.c_[ones,map_x(X)]
        reg=np.eye(XX.shape[1])*.025;reg[0,0]=0
        beta=np.linalg.solve(XX.T@XX+reg,XX.T@y)
        methods[key]=LinearPred(lambda xx,fn=map_x:np.c_[np.ones(len(xx)),fn(xx)],beta)
    for typ,hybrid in [('Full_NN',False),('Hybrid_M0_plus_NN',True)]:
        net,loss=fit_network(X,y,seed=group*100+init,hybrid=hybrid)
        methods[typ]=net
    methods['M0']=lambda x:m0(x[:,0],x[:,3])
    return methods

def one_step_error(model,d):
    X,y=make_train_data(d)
    return np.sqrt(np.mean((model(X)-y)**2))

def rollout_errors(model,d,hs):
    yy=d['observed']; us=d['u'];n,T=us.shape
    maxh=max(hs);starts=[10,20,30]
    errs={h:[] for h in hs}
    for start in starts:
        past=[yy[:,start].copy(), yy[:,start-1].copy(),yy[:,start-2].copy()]
        uprev=[us[:,start-1].copy(),us[:,start-2].copy()]
        for j in range(1,maxh+1):
            u=us[:,start+j-1]
            X=feat(np.column_stack(past),np.column_stack((u,*uprev)))
            p=model(X)
            if j in hs:errs[j].append((p-yy[:,start+j])**2)
            past=[p,past[0],past[1]]
            uprev=[u,uprev[0]]
    return {h:float(np.sqrt(np.mean(np.concatenate(errs[h])))) for h in hs}

def control_metrics(model, family, regime, seed, n=NTEST, ns=STEPS):
    # Each method receives the same independently seeded disturbances, initial
    # states, and reference. Causality: input uses only observations through time t.
    rng=np.random.default_rng(seed)
    p, noise=params(family,regime)
    a=rng.uniform(-.9,.9,n);w=rng.uniform(-1,1,n)
    hist_y=[a+rng.normal(0,noise,n)]*3
    hist_u=[np.zeros(n),np.zeros(n)]
    yy=[];uu=[];refs=[];aa=[];sat=[]
    targets=[.0,.55,-.45,.68,-.25]
    for t in range(ns):
        r=np.full(n,targets[min(4,t//20)])
        feature=feat(np.column_stack(hist_y),np.column_stack((np.zeros(n),*hist_u)))
        pr=model(feature)
        u=np.clip(.37*(r-pr)/(DT*G0),-1,1)
        a,w=transition(a,w,u,p)
        measurement=a+rng.normal(0,noise,n)
        yy.append(measurement);aa.append(a.copy());uu.append(u.copy());refs.append(r);sat.append(abs(u)>=.999)
        hist_y=[measurement,hist_y[0],hist_y[1]]
        hist_u=[u,hist_u[0]]
    YY=np.array(yy).T;AA=np.array(aa).T;UU=np.array(uu).T;RR=np.array(refs).T
    E=AA-RR
    metrics={'tracking_rmse':float(np.sqrt(np.mean(E**2))),
             'tracking_mae':float(np.mean(abs(E))),
             'p95_abs_tracking':float(np.quantile(abs(E),.95)),
             'energy_mean_u2':float(np.mean(UU**2)),
             'saturation_fraction':float(np.mean(abs(UU)>=.999)),
             'input_variation':float(np.mean(abs(np.diff(UU,axis=1)))),
             'max_abs_state':float(np.max(abs(AA)))}
    metric_by_trajectory=np.sqrt(np.mean(E**2,axis=1))
    return metrics,metric_by_trajectory,{'true_air':AA[:3], 'u':UU[:3],'ref':RR[:3]}

def boot_paired_cl(a,b,seed=554,boots=500):
    # Two training-group sets of trajectories: group-level resampling + test
    # trajectories resampling; network inits handled as fixed within-group repeats.
    rng=np.random.default_rng(seed)
    nsets,ns,n=a.shape # 3 train groups, 4 inits, 120 trajectories
    vals=[]
    for i in range(boots):
        groups=rng.integers(nsets,size=nsets)
        idx=rng.integers(n,size=(nsets,n))
        delta=[]
        for z,g in enumerate(groups):
            # Same trajectory index for every fit / comparison: paired test.
            delta.extend((a[g,:,idx[z]]-b[g,:,idx[z]]).mean(axis=0).tolist())
        vals.append(float(np.mean(delta)))
    return list(map(float,np.quantile(vals,[.025,.975])))

def mean_rmse_from_traj(es):
    return float(np.sqrt(np.mean(es**2)))

def write_csv(path,records):
    if not records:return
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main(out,selected_family=None):
    out.mkdir(parents=True,exist_ok=True)
    st=time.monotonic(); trainings=[];closed=[];forecasts=[];paired=[];trajectories={}; traces={}
    for fam in ([selected_family] if selected_family is not None else FAMS):
        for g,train_seed in enumerate(TRAIN_GROUPS):
            data=create_trajectories(fam,'matched',NTRAIN,TTRAIN,train_seed)
            X,y=make_train_data(data)
            trtest={reg:create_trajectories(fam,reg,NTEST,TTRAIN,7000+55*j)
                    for j,reg in enumerate(REGIMES)}
            for si,seed in enumerate(INIT_SEEDS):
                models=fit_models(X,y,g,seed)
                for meth,model in models.items():
                    # Deterministic methods share same trained fit across init seeds;
                    # metrics duplicated but not counted as independent evidence.
                    for j,reg in enumerate(REGIMES):
                        d=trtest[reg]
                        errors=rollout_errors(model,d,HORIZONS)
                        forecasts.extend(dict(family=fam,training_group=g,init=seed,
                            method=meth,scenario=reg,horizon=h,rmse_observed=val)
                            for h,val in errors.items())
                        c,tr,trace=control_metrics(model,fam,reg,9400+37*j)
                        closed.append(dict(family=fam,training_group=g,init=seed,method=meth,
                                           scenario=reg,**c))
                        trajectories[(fam,g,si,meth,reg)]=tr
                        if g==0 and si==0 and reg in ('matched','compound_shift'):
                            traces[(fam,meth,reg)]=trace
                print('FITTED',fam,'group',g,'initialization',seed,'seconds',round(time.monotonic()-st,1),flush=True)
        for reg in REGIMES:
            # Hierarchical paired bootstrap over independently generated training
            # datasets and held-out initial states; 4 weight seeds per group.
            H=np.array([[trajectories[(fam,g,k,'Hybrid_M0_plus_NN',reg)] for k in range(len(INIT_SEEDS))] for g in range(len(TRAIN_GROUPS))])
            M=np.array([[trajectories[(fam,g,k,'M0',reg)] for k in range(len(INIT_SEEDS))] for g in range(len(TRAIN_GROUPS))])
            A=np.array([[trajectories[(fam,g,k,'ARX_poly',reg)] for k in range(len(INIT_SEEDS))] for g in range(len(TRAIN_GROUPS))])
            F=np.array([[trajectories[(fam,g,k,'Full_NN',reg)] for k in range(len(INIT_SEEDS))] for g in range(len(TRAIN_GROUPS))])
            for name,B in [('M0',M),('ARX_poly',A),('Full_NN',F)]:
                lo,hi=boot_paired_cl(B,H)
                paired.append(dict(family=fam,scenario=reg,
                  comparison=f'{name}_minus_Hybrid_cl_trajectory_rmse',
                  delta=float(np.mean(B-H)),ci95_low=lo,ci95_high=hi,
                  note='hierarchical bootstrap over 3 independent training data sets and 120 paired test trajectories; 4 NN initializations within each set'))
    write_csv(out/'closed_loop_runs.csv',closed)
    write_csv(out/'forecast_horizons.csv',forecasts)
    write_csv(out/'paired_bootstrap.csv',paired)
    import pandas as pd
    df=pd.DataFrame(closed)
    summ=df.groupby(['family','scenario','method'],as_index=False).agg(
       tracking_rmse=('tracking_rmse','mean'),tracking_mae=('tracking_mae','mean'),
       p95_abs_tracking=('p95_abs_tracking','mean'),
       energy_mean_u2=('energy_mean_u2','mean'),
       saturation_fraction=('saturation_fraction','mean'),
       input_variation=('input_variation','mean'),
       max_abs_state=('max_abs_state','max'),
       n_training_groups=('training_group','nunique'),n_initializations=('init','nunique'))
    summ.to_csv(out/'closed_loop_summary.csv',index=False)
    # 61: explicitly separable 1,3,6,10 horizons with the same held-out trajectories.
    pd.DataFrame(forecasts).groupby(['family','scenario','method','horizon'],as_index=False).rmse_observed.mean().to_csv(out/'forecast_summary.csv',index=False)
    figdir=out
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fam=selected_family or 'nonlinear_two_state';sc='matched'
    plt.figure(figsize=(11,5)); methods=['M0','ARX_poly','Full_NN','Hybrid_M0_plus_NN']
    for meth in methods:
        dat=next(c for c in closed if c['family']==fam and c['scenario']==sc and c['method']==meth)
        tr=traces[(fam,meth,sc)]
        plt.plot(tr['true_air'][0],label=meth,lw=1.2)
    plt.plot(traces[(fam,'M0',sc)]['ref'][0],color='black',ls='--',label='reference',lw=1.5)
    plt.xlabel('Discrete steps');plt.ylabel('Synthetic chamber temperature deviation')
    plt.title('Closed loop in toy thermal chamber (one representative trajectory)')
    plt.legend(ncol=2,fontsize=8);plt.tight_layout();plt.savefig(out/'closed_loop_trace.png',dpi=145);plt.close()
    # Check direct horizon errors for nonlinear family at matched scenario.
    fdf=pd.read_csv(out/'forecast_summary.csv')
    cut=fdf[(fdf.family==fam)&(fdf.scenario=='matched')]
    plt.figure(figsize=(8,5));
    for m in methods:
        a=cut[cut.method==m]
        plt.plot(a.horizon,a.rmse_observed,marker='o',label=m)
    plt.xlabel('Future steps');plt.ylabel('Observed RMSE')
    plt.title('61: recursive forecast by horizon; independent test trajectories')
    plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/'horizons.png',dpi=145);plt.close()
    meta=dict(date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         models=FAMS,training_groups=TRAIN_GROUPS,initializations=INIT_SEEDS,
         n_training_trajectories_per_group=NTRAIN,n_test_trajectories=NTEST,
         train_steps=TTRAIN,closed_loop_steps=STEPS,
         training_iters_per_NN=230,noise_nominal=.045, dt=DT,
         method='offline learning of one-step residual, fixed model-predictive academic thermostat with bounded actuator; no online training',
         python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pandas=pd.__version__,
         scope='Synthetic heating/cooling only, not hydraulic actuation or vehicle control',
         elapsed_seconds=round(time.monotonic()-st,3))
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    hs={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
    (out/'SHA256.json').write_text(json.dumps(hs,indent=2),encoding='utf-8')
    print('DONE',meta['elapsed_seconds'])
    print(summ.loc[summ.scenario.isin(['matched','compound_shift']),['family','scenario','method','tracking_rmse','energy_mean_u2','saturation_fraction']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--family',choices=list(FAMS),default=None);args=p.parse_args()
    main(args.out,args.family)