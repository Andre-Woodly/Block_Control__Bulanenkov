import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import binomtest, spearmanr
from scipy.interpolate import make_smoothing_spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
N=128; SIGMA=.03; RMS=.1; STEPS=1000; LR=.01
SEEDS=list(range(10)); WIDTHS=[8,16,32]
# Each row is (sin coefficients, cos coefficients). These holdout shapes
# are fixed independently of the results; all are smooth periodic functions.
SHAPES={
 'sine': ([1.],[0.]),
 'H1': ([.8,.6],[.3,0]), 'H2': ([.2,.7,.4],[.5,0,.2]),
 'H3': ([0,.5],[1.,-.4]), 'H4': ([.7,-.3,.3],[.4,.5,0]),
 'H5': ([.9,0,.3],[.1,-.4,.2]), 'H6': ([.2,.5,.2,.1],[.8,.2,0,.2]),
 'H7': ([.4,.6,-.2],[.5,-.1,.4]), 'H8': ([.3,.2,.4],[.8,.4,-.1])}

def residual(x,shape,k):
    a,b=map(np.asarray,SHAPES[shape]); j=np.arange(1,len(a)+1)
    scale=RMS/np.sqrt(np.sum(a*a+b*b)/2)
    t=2*np.pi*k*x[...,None]*j
    return scale*np.sum(a*np.sin(t)+b*np.cos(t),axis=-1)

def curvature(shape,k):
    a,b=map(np.asarray,SHAPES[shape]); j=np.arange(1,len(a)+1)
    return RMS*np.sqrt(np.sum((a*a+b*b)*(2*np.pi*k*j)**4)/np.sum(a*a+b*b))

def main():
    cases=[('main','sine',k) for k in [1,2,4,8]]+ [('holdout',s,k) for s in SHAPES if s!='sine' for k in [1,4]]
    grid=np.arange(1024)/1024
    rows=[]
    for stage,shape,k in cases:
        true=residual(grid,shape,k)
        for seed in SEEDS:
            rng=np.random.default_rng(seed+18000+(100 if stage=='holdout' else 0))
            x=np.sort(rng.uniform(0,1,N)); noise=rng.normal(0,SIGMA,N)
            r=residual(x,shape,k); y=r+noise
            # fixed truth f(x), prior constructed as f(x)-r(x); only residual labels are supplied
            # to the learner. This is a static regression task, not a control system.
            sp=make_smoothing_spline(x,y) # GCV, no test data; natural extrapolation near boundary
            pred=sp(grid)
            rows.append(dict(stage=stage,shape=shape,k=k,seed=seed,method='spline',width=0,params=N,curvature=curvature(shape,k),residual_rms=RMS,rmse=np.sqrt(np.mean((pred-true)**2)),train_rmse=np.sqrt(np.mean((sp(x)-y)**2))))
    # Batch independent fits to avoid Python optimizer overhead. Per-network gradients
    # use sum of individual MSEs; no weights are shared.
    for h in WIDTHS:
        batch=[(st,sh,k,seed) for st,sh,k in cases if st=='main' or h in [16,32] for seed in SEEDS]
        xx=[]; yy=[]; ww=[]; bb=[]; vv=[]; cc=[]; truth=[]
        for st,sh,k,seed in batch:
            rng=np.random.default_rng(seed+18000+(100 if st=='holdout' else 0))
            x=np.sort(rng.uniform(0,1,N)); noise=rng.normal(0,SIGMA,N)
            xx.append(x); yy.append(residual(x,sh,k)+noise); truth.append(residual(grid,sh,k))
            init=np.random.default_rng(seed+30000+h*100)
            ww.append(init.uniform(-1,1,(1,h))); bb.append(init.uniform(-1,1,(1,h)))
            vv.append(init.uniform(-1/np.sqrt(h),1/np.sqrt(h),(h,1))); cc.append(init.uniform(-1/np.sqrt(h),1/np.sqrt(h),(1,1)))
        def tensor(x): return torch.tensor(np.array(x),dtype=torch.float32)
        x=tensor(xx)[:,:,None]*2-1; y=tensor(yy)[:,:,None]
        W,B,V,C=[torch.nn.Parameter(tensor(a)) for a in [ww,bb,vv,cc]]
        opt=torch.optim.Adam([W,B,V,C],lr=LR)
        def forward(z): return torch.bmm(torch.tanh(torch.bmm(z,W)+B),V)+C
        for _ in range(STEPS):
            opt.zero_grad(); p=forward(x); loss=((p-y)**2).mean((1,2)).sum(); loss.backward(); opt.step()
        with torch.no_grad():
            gp=tensor(grid)[None,:,None].expand(len(batch),-1,-1)*2-1
            pred=forward(gp).squeeze(-1).numpy(); train=((forward(x)-y)**2).mean((1,2)).sqrt().numpy()
        for i,(st,sh,k,seed) in enumerate(batch):
            rows.append(dict(stage=st,shape=sh,k=k,seed=seed,method='NN',width=h,params=3*h+1,curvature=curvature(sh,k),residual_rms=RMS,rmse=np.sqrt(np.mean((pred[i]-truth[i])**2)),train_rmse=float(train[i])))
        print('done width',h,len(batch),flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'runs.csv',index=False)
    a=df[df.stage=='main'].groupby(['method','width','k','curvature'],as_index=False).agg(mean_rmse=('rmse','mean'),sd=('rmse','std'),mean_train=('train_rmse','mean'))
    a.to_csv(OUT/'main_summary.csv',index=False)
    ho=df[df.stage=='holdout'].groupby(['shape','method','width','k'],as_index=False).rmse.mean()
    ho.to_csv(OUT/'holdout_function_means.csv',index=False)
    print(a.to_string(index=False))
    for (m,h),g in ho.groupby(['method','width']):
        p=g.pivot(index='shape',columns='k',values='rmse'); d=p[4]-p[1]
        print('HOLDOUT',m,h,'means',p.mean().to_dict(),'worse',int((d>0).sum()),'p',binomtest(int((d>0).sum()),len(d),.5).pvalue)
    for h in WIDTHS:
        g=a[(a.method=='NN')&(a.width==h)]; print('main rho',h,spearmanr(g.curvature,g.mean_rmse).statistic)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for (m,h),g in a.groupby(['method','width']):
        ax.errorbar(g.k,g.mean_rmse,yerr=g.sd,marker='o',capsize=3,label=f'NN, {h} neurons' if m=='NN' else 'Smoothing spline (GCV)')
    ax.set(xlabel='Frequency (cycles on [0, 1])',ylabel='RMSE (mean ± SD across 10 runs)',xticks=[1,2,4,8]); ax.grid(alpha=.3); ax.legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150)
if __name__=='__main__': main()
