"""Checkpoint 07, study R-01: model reconfiguration vs residual NN in a benign toy thermostat.
No electrohydraulic, aerospace, actuator or industrial implementation.
Output results are synthetic and cannot establish safety of physical control systems.
Run: python closed_loop_reconfiguration_67.py
"""
import os, json, time, platform, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd, torch

OUT=Path('/mnt/data/closed_loop_reconfiguration_67');OUT.mkdir(exist_ok=True)
DT=.2; STEPS=105; WARM=30; WINDOW=18; NTEST=180; NTRAIN=180; TRAIN_SEEDS=[101,303]; NET_SEEDS=[7,19,31]
A0=.22;G0=.8;NSTEP_TRAIN=245
NOM={'a':.13,'c':.09,'w':.045,'b':.055,'g':.8}
SCENARIOS={
 'nominal':(NOM,.018),
 'a_g_shift':({**NOM,'a':.19,'g':.65},.018),
 'nonlinear_shift':({**NOM,'b':.25},.018),
 'compound_shift':({**NOM,'a':.20,'c':.145,'w':.028,'b':.23,'g':.61},.018),
 'higher_noise':(NOM,.085),
}
MODELS=['M0_fixed','M0_reconfigured','M0_true_coeff_oracle','M0_fixed_plus_NN','M0_reconfigured_plus_NN','M0_reconfigured_plus_poly']

def system_step(a,w,u,p,d):
    na=a+DT*(-p['a']*a + p['c']*(w-a)-p['b']*a**3 + p['g']*u + d)
    nw=w+DT*p['w']*(a-w)
    return na,nw

def reference(t, n, shape=None):
    # The reference is identical across the five predictors for paired comparison
    if shape is not None:return shape[:,min(t//16,shape.shape[1]-1)]
    return np.full(n,[0.,.45,-.35,.62,-.2,.3,-.15][min(t//15,6)])

def param_estimate(ys,us):
    n=ys[-1].size;m=min(len(us),WINDOW)
    if m<4:return np.full(n,A0),np.full(n,G0)
    y0=np.stack(ys[-m-1:-1]);y1=np.stack(ys[-m:]);uu=np.stack(us[-m:]);
    x0=-y0;x1=uu;z=(y1-y0)/DT
    # ridge around physically interpretable nominal coefficients; causally uses only past observations
    lam_a,lam_g=1.5,1.5
    s00=np.sum(x0*x0,axis=0)+lam_a;s11=np.sum(x1*x1,axis=0)+lam_g;s01=np.sum(x0*x1,axis=0)
    b0=np.sum(x0*z,axis=0)+lam_a*A0;b1=np.sum(x1*z,axis=0)+lam_g*G0
    det=s00*s11-s01*s01
    a=(b0*s11-b1*s01)/det;g=(b1*s00-b0*s01)/det
    return np.clip(a,.05,.6),np.clip(g,.3,1.3)

def baseline(y,u,a,g):return y+DT*(-a*y+g*u)

def feature(y,yprev,u,uprev,ahat,ghat):
    return np.column_stack([y,yprev,u,uprev,ahat,ghat])

def poly_features(x):
    return np.column_stack([np.ones(len(x)),x,x[:,0]**2,x[:,0]**3,x[:,0]*x[:,2],x[:,1]**2])

class Predictor:
    def __init__(self,ws,bs):self.ws=ws;self.bs=bs
    def __call__(self,X):
        z=X.astype(float)
        for j,(w,b) in enumerate(zip(self.ws,self.bs)):
            z=z@w+b
            if j<len(self.ws)-1:z=np.tanh(z)
        return z[:,0]

def fit_nn(X,res,seed):
    torch.manual_seed(seed);torch.set_num_threads(1)
    # preserve physical scales: features and residuals are dimensionless toy temperatures
    x=torch.tensor(X,dtype=torch.float32);y=torch.tensor(res[:,None],dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(6,18),torch.nn.Tanh(),torch.nn.Linear(18,18),torch.nn.Tanh(),torch.nn.Linear(18,1))
    opt=torch.optim.Adam(net.parameters(),lr=.009)
    for _ in range(NSTEP_TRAIN):
        opt.zero_grad();loss=((net(x)-y)**2).mean();loss.backward();opt.step()
    ws=[];bs=[]
    for lay in net:
        if isinstance(lay,torch.nn.Linear):ws.append(lay.weight.detach().numpy().T.copy());bs.append(lay.bias.detach().numpy().copy())
    return Predictor(ws,bs),float(loss.detach())

def run_warmup(n,p,noise,rng,train=False):
    a=rng.uniform(-.6,.6,n);w=rng.uniform(-.8,.8,n)
    # same drawn disturbance and sensor noise per compared method
    dn=rng.normal(0,.035,(STEPS,n));eta=rng.normal(0,noise,(STEPS+1,n))
    initobs=a+eta[0];ys=[initobs];us=[]
    targets=rng.uniform(-.7,.7,(n,7)) if train else None
    for t in range(WARM):
        r=reference(t,n,targets) if train else reference(t,n)
        u=np.clip(.85*(r-ys[-1]),-1,1)
        a,w=system_step(a,w,u,p,dn[t]);ys.append(a+eta[t+1]);us.append(u)
    return a,w,ys,us,dn,eta,targets

def training_set(seed):
    rng=np.random.default_rng(seed)
    feat_rows=[];tf=[];tr=[]
    # multi-domain training; all data generated independently from test seeds
    portions=[(NOM,.018),({**NOM,'a':.17,'g':.70},.018),({**NOM,'b':.15,'c':.115},.018)]
    for p,noise in portions:
        n=NTRAIN//len(portions)
        a,w,ys,us,dn,eta,targets=run_warmup(n,p,noise,rng,train=True)
        for t in range(WARM,STEPS):
            ah,gh=param_estimate(ys,us)
            r=reference(t,n,targets)
            u=np.clip(.85*(r-ys[-1]),-1,1)
            X=feature(ys[-1],ys[-2],u,us[-1],ah,gh)
            a,w=system_step(a,w,u,p,dn[t]);y1=a+eta[t+1]
            feat_rows.append(X)
            tf.append(y1-baseline(ys[-1],u,A0,G0))
            tr.append(y1-baseline(ys[-1],u,ah,gh))
            ys.append(y1);us.append(u)
    return np.vstack(feat_rows),np.concatenate(tf),np.concatenate(tr)

def learn(group,init):
    X,res_f,res_r=training_set(TRAIN_SEEDS[group])
    nn_f,lf=fit_nn(X,res_f,group*100+NET_SEEDS[init])
    nn_r,lr=fit_nn(X,res_r,group*100+NET_SEEDS[init]+12345)
    Px=poly_features(X)
    reg=np.eye(Px.shape[1])*.5;reg[0,0]=.001
    beta=np.linalg.solve(Px.T@Px+reg,Px.T@res_r)
    return (nn_f,nn_r,beta),dict(n_samples=len(X),loss_fixed=lf,loss_reconf=lr)

def candidates_pred(method,ys,us,ah,gh,trained,p):
    n=ys[-1].size; cand=np.array([-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.])
    # grid search on bounded toy HVAC input; same decision rule for every model
    U=np.broadcast_to(cand[None,:],(n,len(cand)))
    yy=np.repeat(ys[-1],len(cand));yold=np.repeat(ys[-2],len(cand))
    uu=U.reshape(-1);uold=np.repeat(us[-1],len(cand))
    aa=np.repeat(ah,len(cand));gg=np.repeat(gh,len(cand))
    X=feature(yy,yold,uu,uold,aa,gg)
    if method.startswith('M0_reconfigured'): pred=baseline(yy,uu,aa,gg)
    elif method=='M0_true_coeff_oracle':pred=baseline(yy,uu,p['a']+p['c'],p['g'])
    else:pred=baseline(yy,uu,A0,G0)
    if method=='M0_fixed_plus_NN':pred+=trained[0](X)
    elif method=='M0_reconfigured_plus_NN':pred+=trained[1](X)
    elif method=='M0_reconfigured_plus_poly':pred+=poly_features(X)@trained[2]
    return U,pred.reshape(n,len(cand))

def eval_method(group,init,method,scenario,seed=12001,n=NTEST):
    p,noise=SCENARIOS[scenario];rng=np.random.default_rng(seed)
    a,w,ys,us,dn,eta,targets=run_warmup(n,p,noise,rng)
    model=TRAINED[(group,init)]
    e_list=[];u_list=[];slope_err=[];gain_err=[]
    for t in range(WARM,STEPS):
        ah,gh=param_estimate(ys,us)
        U,preds=candidates_pred(method,ys,us,ah,gh,model,p)
        r=reference(t,n)
        cost=(preds-r[:,None])**2 +.009*(U**2)
        u=U[np.arange(n),np.argmin(cost,axis=1)]
        a,w=system_step(a,w,u,p,dn[t]);nextobs=a+eta[t+1]
        e_list.append(a-r);u_list.append(u)
        slope_err.append(ah-(p['a']+p['c']))
        gain_err.append(gh-p['g'])
        ys.append(nextobs);us.append(u)
    E=np.stack(e_list,axis=1);U=np.stack(u_list,axis=1)
    # each trajectory is independent cluster in test; seeds share matching random disturbances
    per=np.sqrt(np.mean(E**2,axis=1));
    MET=dict(tracking_rmse=float(np.sqrt(np.mean(E**2))),
       tracking_mae=float(np.mean(np.abs(E))),
       iae=float(DT*np.mean(np.sum(np.abs(E),axis=1))),
       itae=float(DT*DT*np.mean(np.sum((np.arange(E.shape[1])+1)[None,:]*np.abs(E),axis=1))),
       p95_abs_err=float(np.quantile(np.abs(E),.95)),
       u2=float(np.mean(U**2)),
       variation=float(np.mean(np.abs(np.diff(U,axis=1)))),
       saturation=float(np.mean(np.abs(U)>=.999)),
       estimated_a_bias=float(np.mean(slope_err)),estimated_g_bias=float(np.mean(gain_err)))
    return MET,per,(E[0],U[0])

def interval(a,b,seed=77,nboot=500):
    # hierarchical bootstrap resamples training groups, then paired test trajectories; NN seed variance inside groups
    rng=np.random.default_rng(seed);gs,ns,nt=a.shape
    obs=float(np.mean(a-b));boot=[]
    for _ in range(nboot):
        chosen=rng.integers(gs,size=gs);idx=rng.integers(nt,size=(gs,nt));
        each=[]
        for i,g in enumerate(chosen):each.append(np.mean((a[g,:,idx[i]]-b[g,:,idx[i]])))
        boot.append(np.mean(each))
    return obs,*np.quantile(boot,[.025,.975])

def main():
    st=time.time();rnglog=[];global TRAINED;TRAINED={}
    for group in range(len(TRAIN_SEEDS)):
      for init in range(len(NET_SEEDS)):
        trained,loss=learn(group,init);TRAINED[(group,init)]=trained
        rnglog.append({'train_group':group,'net_seed':NET_SEEDS[init],**loss})
        print('trained',group,init,loss,flush=True)
    runs=[];comparisons=[];trace={};per_map={}
    for sc,scene in enumerate(SCENARIOS):
      for grp in range(len(TRAIN_SEEDS)):
       for init in range(len(NET_SEEDS)):
        for meth in MODELS:
          metrics,per,tr=eval_method(grp,init,meth,scene,seed=9000+73*sc)
          runs.append({'scenario':scene,'train_group':grp,'init':NET_SEEDS[init],'method':meth,**metrics})
          per_map[(scene,grp,init,meth)]=per
          if grp==0 and init==0:trace[(scene,meth)]=tr
      for meth in ['M0_reconfigured','M0_true_coeff_oracle','M0_fixed_plus_NN','M0_reconfigured_plus_NN','M0_reconfigured_plus_poly']:
        ar=np.array([[per_map[(scene,g,j,meth)] for j in range(len(NET_SEEDS))] for g in range(len(TRAIN_SEEDS))])
        bas=np.array([[per_map[(scene,g,j,'M0_fixed')] for j in range(len(NET_SEEDS))] for g in range(len(TRAIN_SEEDS))])
        comparisons.append({'scenario':scene,'comparison':f'M0_fixed_minus_{meth}',**dict(zip(['delta_traj_rmse','ci95_lo','ci95_hi'],interval(bas,ar)))})
      for method in ['M0_reconfigured_plus_NN','M0_reconfigured_plus_poly']:
        rec=np.array([[per_map[(scene,g,j,'M0_reconfigured')] for j in range(len(NET_SEEDS))] for g in range(len(TRAIN_SEEDS))])
        alt=np.array([[per_map[(scene,g,j,method)] for j in range(len(NET_SEEDS))] for g in range(len(TRAIN_SEEDS))])
        comparisons.append({'scenario':scene,'comparison':f'M0_reconfigured_minus_{method}',**dict(zip(['delta_traj_rmse','ci95_lo','ci95_hi'],interval(rec,alt)))})
    runs=pd.DataFrame(runs);runs.to_csv(OUT/'runs.csv',index=False)
    summary=runs.groupby(['scenario','method'],as_index=False).agg({k:'mean' for k in ['tracking_rmse','tracking_mae','iae','itae','p95_abs_err','u2','variation','saturation','estimated_a_bias','estimated_g_bias']});summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'bootstrap.csv',index=False)
    pd.DataFrame(rnglog).to_csv(OUT/'training.csv',index=False)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    piv=summary.pivot(index='scenario',columns='method',values='tracking_rmse')
    piv=piv.reindex(list(SCENARIOS));piv.plot.bar(figsize=(13,5.5),rot=15)
    plt.title('Toy thermostat: nonlinear plant, mathematical coefficient update, NN residual correction')
    plt.ylabel('Closed-loop tracking RMSE (synthetic units)');plt.grid(axis='y',alpha=.25);plt.tight_layout();plt.savefig(OUT/'comparison.png',dpi=145)
    # trace demonstrates all methods compared on same environmental realization and setpoint
    plt.figure(figsize=(11,5));scene='compound_shift'
    for meth in MODELS:
      plt.plot(trace[(scene,meth)][0],lw=1,label=meth)
    plt.plot([reference(t,np.ones(1).size)[0] for t in range(WARM,STEPS)],'k--',label='target',lw=1.7)
    plt.xlabel('Closed-loop time step');plt.ylabel('Toy temperature deviation');plt.legend(ncol=2,fontsize=8);plt.tight_layout();plt.savefig(OUT/'trace_compound.png',dpi=145)
    meta={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':platform.python_version(),
          'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,
          'train_dataset_seeds':TRAIN_SEEDS,'nn_weight_seeds':NET_SEEDS,'evaluation_seed_base':9000,
          'nominal':NOM,'scenarios':SCENARIOS,'training_trajectories_each':NTRAIN,
          'test_trajectories_each_scenario':NTEST,'evaluation_time_steps':STEPS-WARM,
          'warmup_time_steps':WARM,'all_methods':MODELS,'nn_train_steps':NSTEP_TRAIN,
          'physical_scope':'Benign synthetic two-temperature thermostat, no physical drive or aviation system',
          'notes':['Same plant and disturbances across methods; reconfiguration uses causal regularized regressions on observed history',
                   'M0_reconfigured is a reduced-order estimated model, not ground-truth parameter oracle',
                   'M0_true_coeff_oracle is a diagnostic reference with access to hidden true effective coefficients a+c and g, not a deployable method',
                   'Corrector NN weights are fixed after offline training, respond to online parameter estimates as inputs',
                   'Small training-group count limits inferential strength; no formal stability theorem',
                   'The one-step grid search is a toy feedback mechanism; results should not be transferred to physical actuation']}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    hs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    (OUT/'SHA256.json').write_text(json.dumps(hs,indent=2),encoding='utf-8')
    print(summary.pivot(index='scenario',columns='method',values='tracking_rmse').round(4).to_string())
    print('STATS significant?\n',pd.DataFrame(comparisons).to_string(index=False))
    print('DONE',time.time()-st,flush=True)
if __name__=='__main__':main()