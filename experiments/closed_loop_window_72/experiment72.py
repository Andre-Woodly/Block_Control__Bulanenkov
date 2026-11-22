"""Experiment 72. Sliding histories for a residual NN in a BENIGN synthetic two-temperature thermostat.
No electrohydraulic actuator, aviation control, or hardware control instructions.
Reconfiguration uses a fixed 18-transition sliding window in every comparison.
NN sees the last L measurement-action samples (L=1,2,5,10,18) and current candidate action.
"""
from __future__ import annotations
import os, sys, time, platform, argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd, torch, scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DT=.2; WARM=30; TRAIN_STEPS=80; EVAL_STEPS=85
PARAM_WINDOW=18; HISTMAX=18; WINDOWS=(1,2,5,10,18)
NTRAIN=150; NTEST=100; GROUP_SEEDS=(101,303,509); INIT_SEEDS=(7,19)
TRAIN_ITERS=270; NN_WIDTH=24; ACTIONS=np.arange(-1,1.0001,.25)
A0=.22; G0=.8; RIDGE_A=1.5; RIDGE_G=1.5
NOM={'a':.13,'c':.09,'w':.045,'b':.055,'g':.8}
SCENARIOS={'nominal':(NOM,.018), 'compound_shift':({**NOM,'a':.20,'c':.145,'w':.028,'b':.23,'g':.61},.018), 'higher_noise':(NOM,.085), 'nonlinear_shift':({**NOM,'b':.25},.018)}
METHODS=('M0_reconfigured','M0_plus_Ridge18')+tuple(f'M0_plus_NN_L{L}' for L in WINDOWS)


def system_step(A,W,U,p,disturbance):
    return A + DT*(-p['a']*A+p['c']*(W-A)-p['b']*A**3+p['g']*U+disturbance),W+DT*p['w']*(A-W)

def param_estimate(ys,us):
    # common, causal 18-transition estimator for EVERY comparison
    n=len(ys[-1]);m=min(PARAM_WINDOW,len(us))
    if m<4:return np.full(n,A0),np.full(n,G0)
    ya=np.stack(ys[-m-1:-1]);yb=np.stack(ys[-m:]);ux=np.stack(us[-m:]);rate=(yb-ya)/DT
    z0=-ya;z1=ux
    s00=np.sum(z0*z0,axis=0)+RIDGE_A;s11=np.sum(z1*z1,axis=0)+RIDGE_G;s01=np.sum(z0*z1,axis=0)
    b0=np.sum(z0*rate,axis=0)+RIDGE_A*A0;b1=np.sum(z1*rate,axis=0)+RIDGE_G*G0
    det=s00*s11-s01*s01
    return np.clip((b0*s11-b1*s01)/det,.05,.6),np.clip((b1*s00-b0*s01)/det,.3,1.3)

def baseline(y,u,ahat,ghat):return y+DT*(-ahat*y+ghat*u)

def make_features(ys,us,ahat,ghat,candidate):
    # same dimensionality 39 for ALL windows, history entries are newest-first;
    # for L<18, older samples are zero-padded (not shown to NN).
    n=len(candidate);X=np.zeros((n,2*HISTMAX+3),dtype=np.float32)
    for j in range(HISTMAX):
        if j<len(ys):X[:,j]=ys[-j-1]
        if j<len(us):X[:,HISTMAX+j]=us[-j-1]
    X[:,-3]=candidate;X[:,-2]=ahat;X[:,-1]=ghat
    return X

def mask_history(X,L):
    if L==HISTMAX:return X
    z=X.copy();z[:,L:HISTMAX]=0;z[:,HISTMAX+L:2*HISTMAX]=0
    return z

def choose_reference(t,kind,target):
    return target if kind=='constant' else (0.,.45,-.35,.62,-.2)[min(t//22,4)]

def training_data(seed):
    rng=np.random.default_rng(seed)
    scen=[(NOM,.018),({**NOM,'a':.17,'g':.70},.018),({**NOM,'b':.15,'c':.115},.018)]
    xs,labels=[],[]
    for p,sigma in scen:
        n=NTRAIN//len(scen); A=rng.uniform(-.6,.6,n);W=rng.uniform(-.8,.8,n)
        eta=rng.normal(0,sigma,(WARM+TRAIN_STEPS+1,n));dist=rng.normal(0,.035,(WARM+TRAIN_STEPS,n))
        ys=[A+eta[0]];us=[];targets=rng.uniform(-.7,.7,(n,11))
        for t in range(WARM+TRAIN_STEPS):
            r=targets[:,min(t//11,10)]
            # Behavior policy has the same bounded discrete candidate actions as evaluation
            pred=baseline(ys[-1][:,None],ACTIONS[None,:],A0,G0)
            idx=np.argmin((pred-r[:,None])**2+.009*ACTIONS[None,:]**2,axis=1)
            u=ACTIONS[idx]
            ah,gh=param_estimate(ys,us)
            x=make_features(ys,us,ah,gh,u)
            A,W=system_step(A,W,u,p,dist[t]);yn=A+eta[t+1]
            if t>=WARM:
                xs.append(x);labels.append(yn-baseline(ys[-1],u,ah,gh))
            ys.append(yn);us.append(u)
    return np.concatenate(xs,axis=0),np.concatenate(labels,axis=0)

class FastNN:
    def __init__(self,ws,bs):self.ws,self.bs=ws,bs
    def __call__(self,x):
        z=x.astype(np.float64,copy=False)
        for i,(w,b) in enumerate(zip(self.ws,self.bs)):
            z=z@w+b
            if i<len(self.ws)-1:z=np.tanh(z)
        return z[:,0]

def fit_nn(x,y,L,seed):
    torch.manual_seed(seed);torch.set_num_threads(1)
    inp=torch.from_numpy(mask_history(x,L));target=torch.as_tensor(y[:,None].astype('float32'))
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],NN_WIDTH),torch.nn.Tanh(),torch.nn.Linear(NN_WIDTH,NN_WIDTH),torch.nn.Tanh(),torch.nn.Linear(NN_WIDTH,1))
    opt=torch.optim.Adam(net.parameters(),lr=.008)
    for it in range(TRAIN_ITERS):
        opt.zero_grad();loss=((net(inp)-target)**2).mean();loss.backward();opt.step()
    ws,bs=[],[]
    for layer in net:
        if isinstance(layer,torch.nn.Linear):ws.append(layer.weight.detach().numpy().T.astype(float).copy());bs.append(layer.bias.detach().numpy().astype(float).copy())
    return FastNN(ws,bs),float(loss.item())

def ridge_feats(x,L=18):
    # strong cheap comparator; Ridge on history plus nonlinear current state terms
    x=mask_history(x,L)
    z=x[:,0];cand=x[:,-3]
    return np.column_stack([np.ones(len(x)),x,z*z,z*z*z,cand*z,z*cand*cand])

def fit_ridge(X,y):
    Z=ridge_feats(X);reg=.5*np.eye(Z.shape[1]);reg[0,0]=.001
    return np.linalg.solve(Z.T@Z+reg,Z.T@y)

def pred_all(ys,us,ahat,ghat,method,nets,ridge):
    n=len(ys[-1]);k=len(ACTIONS)
    U=np.broadcast_to(ACTIONS[None,:],(n,k))
    Y=np.repeat(ys[-1],k);ah=np.repeat(ahat,k);gh=np.repeat(ghat,k)
    pred=baseline(Y,U.reshape(-1),ah,gh)
    if method!='M0_reconfigured':
        hist=make_features(ys,us,ahat,ghat,np.zeros(n))
        feats=np.repeat(hist,k,axis=0);feats[:,-3]=U.reshape(-1)
        if method=='M0_plus_Ridge18':pred+=ridge_feats(feats)@ridge
        else:
            L=int(method.rsplit('L',1)[1]);pred+=nets[L](mask_history(feats,L))
    return U,pred.reshape(n,k)

def simulate(method,models,scenario,kind,ref,seed):
    p,sigma=SCENARIOS[scenario];rng=np.random.default_rng(seed)
    n=NTEST;A=rng.uniform(-.35,.35,n);W=rng.uniform(-.45,.45,n)
    sz=WARM+EVAL_STEPS
    eta=rng.normal(0,sigma,(sz+1,n));dist=np.clip(rng.normal(0,.035,(sz,n)),-.06,.06)
    ys=[A+eta[0]];us=[]
    # shared warm-up to make all methods start with identical states
    for t in range(WARM):
        u=np.clip(-.85*ys[-1],-1,1)
        A,W=system_step(A,W,u,p,dist[t]);ys.append(A+eta[t+1]);us.append(u)
    errors=[];actions=[];trace=[]
    for t in range(EVAL_STEPS):
        ah,gh=param_estimate(ys,us)
        U,pred=pred_all(ys,us,ah,gh,method,*models)
        r=choose_reference(t,kind,ref)
        ix=np.argmin((pred-r)**2+.009*U**2,axis=1)
        u=U[np.arange(n),ix]
        A,W=system_step(A,W,u,p,dist[WARM+t]);ys.append(A+eta[WARM+t+1]);us.append(u)
        errors.append(A-r);actions.append(u)
        if t==0:trace.append((A[0],r))
    E=np.stack(errors,axis=1);U=np.stack(actions,axis=1)
    rmse_by_traj=np.sqrt(np.mean(E**2,axis=1))
    mets={'tracking_rmse':float(np.sqrt(np.mean(E**2))),
          'rmse_last40':float(np.sqrt(np.mean(E[:,-40:]**2))),
          'iae':float(DT*np.mean(np.sum(np.abs(E),axis=1))),
          'itae':float(DT*DT*np.mean(np.sum(np.abs(E)*(np.arange(E.shape[1])+1)[None,:],axis=1))),
          'p95_abs_error':float(np.quantile(abs(E),.95)),
          'max_abs_error':float(abs(E).max()),
          'action_var':float(np.mean(abs(np.diff(U,axis=1)))),
          'mean_u2':float(np.mean(U*U)),
          'saturation_fraction':float(np.mean(abs(U)>=.999))}
    return mets,rmse_by_traj,(E[0],U[0])

def forecast_validation(X,y,nets,ridge):
    return {f'M0_plus_NN_L{L}':float(np.sqrt(np.mean((nets[L](mask_history(X,L))-y)**2))) for L in WINDOWS} | {'M0_reconfigured':float(np.sqrt(np.mean(y*y))), 'M0_plus_Ridge18':float(np.sqrt(np.mean((ridge_feats(X)@ridge-y)**2)))}

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();models_by_group={};trainrows=[]
    for g,seed in enumerate(GROUP_SEEDS):
        X,y=training_data(seed);ridge=fit_ridge(X,y);nets_by_init={}
        for j,init in enumerate(INIT_SEEDS):
            nets={}
            for L in WINDOWS:
                model,loss=fit_nn(X,y,L,init+100*g)
                nets[L]=model
                trainrows.append({'group':g,'data_seed':seed,'init_seed':init,'window':L,'train_MSE':loss,'n_examples':len(y)})
            nets_by_init[j]=nets
        models_by_group[g]=(nets_by_init,ridge)
        print('trained group',g,'seconds',round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(trainrows).to_csv(out/'training.csv',index=False)
    valrows=[]
    for scenario in SCENARIOS:
        # generate policy-induced unseen data with same construction, but different physical domain
        # independent of training data and of feedback-evaluation realizations
        # evaluate one-step residual prediction on an unseen episode using exact real observation as target
        p,sigma=SCENARIOS[scenario];n=90;rng=np.random.default_rng(88000+list(SCENARIOS).index(scenario)*701)
        A=rng.uniform(-.55,.55,n);W=rng.uniform(-.75,.75,n);ys=[A+rng.normal(0,sigma,n)];us=[]
        features=[];resids=[]
        targets=rng.uniform(-.7,.7,(n,12))
        for t in range(WARM+80):
            r=targets[:,min(t//10,11)];pred=baseline(ys[-1][:,None],ACTIONS[None,:],A0,G0)
            U=ACTIONS[np.argmin((pred-r[:,None])**2+.009*ACTIONS[None,:]**2,axis=1)]
            ah,gh=param_estimate(ys,us);feat=make_features(ys,us,ah,gh,U)
            A,W=system_step(A,W,U,p,rng.normal(0,.035,n));yn=A+rng.normal(0,sigma,n)
            if t>=WARM:features.append(feat);resids.append(yn-baseline(ys[-1],U,ah,gh))
            ys.append(yn);us.append(U)
        FX=np.concatenate(features);Y=np.concatenate(resids)
        for g in models_by_group:
            for j in range(len(INIT_SEEDS)):
                nets,ridge=models_by_group[g][0][j],models_by_group[g][1]
                for meth,v in forecast_validation(FX,Y,nets,ridge).items():
                    valrows.append({'scenario':scenario,'group':g,'init':INIT_SEEDS[j],'method':meth,'one_step_rmse_observed':v})
    val=pd.DataFrame(valrows);val.to_csv(out/'forecast_runs.csv',index=False)
    val.groupby(['scenario','method'],as_index=False)['one_step_rmse_observed'].mean().to_csv(out/'forecast_summary.csv',index=False)
    runs=[];per={};traces={}; experiments=[('nominal','constant',.45),('compound_shift','constant',.45),('higher_noise','constant',.45),('nonlinear_shift','constant',.45),('nominal','piecewise',None),('compound_shift','piecewise',None)]
    for si,(sc,kind,ref) in enumerate(experiments):
        seed=721000+si*977
        for g in models_by_group:
            for j,init in enumerate(INIT_SEEDS):
                model=(models_by_group[g][0][j],models_by_group[g][1]);
                for method in METHODS:
                    stats,traj,trace=simulate(method,model,sc,kind,ref,seed)
                    runs.append({'scenario':sc,'kind':kind,'target':ref if kind=='constant' else 'piecewise', 'group':g,'seed':init,'method':method,**stats})
                    per[(si,g,j,method)]=traj
                    if g==0 and j==0:traces[(si,method)]=trace
        print('evaluated',sc,kind,'seconds',round(time.monotonic()-start,1),flush=True)
    runs=pd.DataFrame(runs);runs.to_csv(out/'closed_loop_runs.csv',index=False)
    metrics=['tracking_rmse','rmse_last40','iae','itae','p95_abs_error','max_abs_error','action_var','mean_u2','saturation_fraction']
    summ=runs.groupby(['scenario','kind','target','method'],as_index=False)[metrics].mean();summ.to_csv(out/'closed_loop_summary.csv',index=False)
    # grouped data: each group is separate training dataset; summarize means across independent groups
    group=runs.groupby(['scenario','kind','target','group','method'],as_index=False)[metrics].mean()
    group.to_csv(out/'group_summary.csv',index=False)
    # difference in per-trajectory RMSE, same disturbances for all methods; CIs conditional on trained groups
    rng=np.random.default_rng(7261); comparisons=[]
    for si,(sc,kind,ref) in enumerate(experiments):
        for met in ('M0_plus_NN_L1','M0_plus_NN_L2','M0_plus_NN_L5','M0_plus_NN_L10','M0_plus_NN_L18','M0_plus_Ridge18'):
            a=np.array([per[(si,g,j,'M0_reconfigured')]-per[(si,g,j,met)] for g in models_by_group for j in range(len(INIT_SEEDS))])
            b=np.array([per[(si,g,j,'M0_plus_NN_L2')]-per[(si,g,j,met)] for g in models_by_group for j in range(len(INIT_SEEDS))])
            for label,ary in [('M0minusMethod',a),('L2minusMethod',b)]:
                obs=float(ary.mean());boot=[]
                for _ in range(350):
                    group_idx=rng.integers(len(GROUP_SEEDS),size=len(GROUP_SEEDS));sample_idx=rng.integers(NTEST,size=(len(GROUP_SEEDS),NTEST))
                    vals=[]
                    for k,g in enumerate(group_idx):vals.append(float(ary[g*len(INIT_SEEDS):(g+1)*len(INIT_SEEDS),sample_idx[k]].mean()))
                    boot.append(float(np.mean(vals)))
                comparisons.append({'scenario':sc,'kind':kind,'comparison':met,'base':label,'delta_per_trajectory_RMSE':obs,'ci95_lo':float(np.quantile(boot,.025)),'ci95_hi':float(np.quantile(boot,.975))})
    pd.DataFrame(comparisons).to_csv(out/'bootstrap.csv',index=False)
    # Chart: steady state RMSE by window for each scenario, distinct figures not subplots
    plot=summ[(summ.kind=='constant')&(summ.method.str.startswith('M0_plus_NN'))].copy()
    plot['window']=plot.method.str.rsplit('L',n=1).str[-1].astype(int)
    fig,ax=plt.subplots(figsize=(9,5))
    for sc in SCENARIOS:
        ss=plot[plot.scenario==sc].sort_values('window');ax.plot(ss.window,ss.rmse_last40,marker='o',label=sc)
    ax.set_xlabel('Window length (past measurement-action samples)');ax.set_ylabel('Last-40-step tracking RMSE');ax.set_title('Experiment 72: NN history depth, identical 18-step M0 estimation');ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'window_rmse.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    si=1
    for method in ['M0_reconfigured','M0_plus_NN_L1','M0_plus_NN_L2','M0_plus_NN_L5','M0_plus_NN_L10','M0_plus_NN_L18','M0_plus_Ridge18']:
        ax.plot(traces[(si,method)][0],label=method,linewidth=1)
    ax.set_xlabel('Closed-loop step');ax.set_ylabel('True tracking error A−r');ax.set_title('Compound shift, constant target, same random realization');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'example_trace.png',dpi=155);plt.close(fig)
    meta={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
          'NTRAIN_trajectories_per_group':NTRAIN,'NTEST_per_evaluation':NTEST,'training_data_seeds':list(GROUP_SEEDS),'network_initialization_seeds':list(INIT_SEEDS), 'WINDOWS':list(WINDOWS),'M0_parameter_estimation_window':PARAM_WINDOW,
          'nn_architecture':'39-24-24-1 tanh; identical architecture for all windows with unused older history entries set to 0', 'nn_train_iterations':TRAIN_ITERS,
          'features':'newest-first last L y measurements, last L past actions, prospective candidate u, ahat and ghat; zero padding to 18', 'notes':['M0 regression always uses last 18 past transitions; independent of NN input history', 'NN weights fixed after training; coefficients ahat,ghat recalculated each simulated step','Model is a synthetic two-temperature thermostat, not an electrohydraulic actuator', 'All methods use same 9-action discrete educational policy and same disturbance/noise realizations per scenario', 'Reported bootstrap intervals use only three independent training groups; inferential scope is limited', 'Closed-loop results do not establish Lyapunov stability or physical safety']}
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*') if f.is_file() and f.name!='SHA256.json'}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('CONSTANT STEADY RMSE',summ[(summ.kind=='constant')].pivot(index='scenario',columns='method',values='rmse_last40').round(5).to_string(),flush=True)
    print('PIECEWISE ALL RMSE',summ[(summ.kind=='piecewise')].pivot(index='scenario',columns='method',values='tracking_rmse').round(5).to_string(),flush=True)
    print('finished',round(time.monotonic()-start,1), 'seconds',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='/mnt/data/exp72');arg=ap.parse_args();run(Path(arg.out))