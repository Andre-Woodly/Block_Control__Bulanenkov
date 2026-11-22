"""Experiment 71: NONZERO reference tracking on a benign synthetic two-temperature system.

Scope: mathematical residual NN versus online coefficient recalculation under sampled feedback.
NOT an electrohydraulic actuator, aviation system, or control law for a physical device.
Independent results pertain solely to the toy model in experiment67_source.py.

Run: python experiment71.py --out /mnt/data/exp71
"""
from __future__ import annotations
import sys, json, platform, datetime, hashlib, time, argparse, shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np, pandas as pd, scipy, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experiment67_source as m

P=np.diag([1.0,2.0]);D=np.diag([1.0,1/np.sqrt(2)]); R=3.; RREF=.7
QMAX=R*R+R*RREF+RREF*RREF
GRID=np.arange(-1,1.0001,.25)
SCENARIOS=('nominal','compound_shift','higher_noise')
TARGETS=(0.,.45,.62,-.35)
METHODS=('M0_fixed','M0_reconfigured','M0_true_coeff_oracle','M0_fixed_plus_NN','M0_reconfigured_plus_NN','M0_reconfigured_plus_poly')
TRAIN_GROUPS=(0,1); INIT_GROUPS=(0,1)
N=120; WARM=30; LENGTH=165

def theoretical_certificate():
    vals=[]; rhos=[]
    for name,(p,noise) in m.SCENARIOS.items():
      for q in (F(0),F(1159,100)):
        a,c,w,b=[F(str(p[k])) for k in ('a','c','w','b')];h=F(str(m.DT))
        t11=1-h*(a+c+b*q); t12=h*c; t21=h*w; t22=1-h*w
        # M=P-T'PT; exact rational Sylvester test
        M11=1-t11*t11-2*t21*t21
        M12=-t11*t12-2*t21*t22
        M22=2-t12*t12-2*t22*t22
        det=M11*M22-M12*M12
        assert M11>0 and det>0,(name,q,M11,det)
        T=np.array([[float(t11),float(t12)],[float(t21),float(t22)]])
        rho=float(np.linalg.eigvalsh(D@T.T@P@T@D).max())
        vals.append({'scenario':name,'q':float(q),'P_minus_TPT_11':float(M11),'P_minus_TPT_det':float(det),
                    'positive_definite_exact':True,'lambda2_Pnorm':rho,
                    'principal_minor_11_exact':str(M11),'det_exact':str(det)})
        rhos.append(rho)
    rho=max(rhos)
    ideal=[]
    for label,(p,noise) in m.SCENARIOS.items():
      for r in TARGETS:
        u_star=(p['a']*r+p['b']*r**3)/p['g']
        dist=float(np.min(abs(GRID-u_star)))
        ideal.append({'scenario':label,'target':r,'ideal_equilibrium_u_star':u_star,
                     'distance_to_action_grid':dist,
                     'smallest_one_step_air_deviation_from_exact_target':m.DT*p['g']*dist,
                     'target_is_exact_equilibrium_with_grid':bool(dist<1e-12)})
    return vals,ideal,{'P_diag':[1,2],'model_box_R':R,'reference_bound':RREF,
         'q_max':QMAX,'rho':rho,'lambda':rho**.5,'one_step_disturbance_h':m.DT,
         'u_grid':GRID.tolist(),
         'tracking_bound':'||z[k+1]||P <= lambda||z[k]||P + h(gmax|u[k]-u_star(r[k],p[k])|+|d[k]|)+sqrt(3)|r[k+1]-r[k]|',
         'regional_domain':'|A|,|W|<=3; |reference|<=0.7; physical p in convex hull of 5 scenarios; |u|<=1, |d|<=0.06',
         'theorem_type':'Uniform incremental input-to-state inequality for tracking physical state relative to [r,r], NOT a closed-loop NN tracking stability theorem',
         'proof':'T(p,q) affine convex combination of ten vertices (5 p, q in {0,11.59}); each P-T.T P T positive definite by exact rational principal minor and determinant; convexity of quadratic norm yields global lambda<1 on K and r range',
         'nonzero_equilibrium':'For a constant target r and d=0, exact equilibrium A=W=r implies u=(a*r+b*r^3)/g; if this is not in the discrete action grid no exact equilibrium at that reference is possible irrespective of NN.'}

def simulate(p, noise, method, trained, group, init, scenario, kind, rconst, seed):
    rng=np.random.default_rng(seed)
    a=rng.uniform(-.35,.35,N);w=rng.uniform(-.45,.45,N)
    disturbances=np.clip(rng.normal(0,.035,(WARM+LENGTH,N)),-.06,.06)
    eta=rng.normal(0,noise,(WARM+LENGTH+1,N))
    ys=[a+eta[0]];us=[]
    for t in range(WARM):
        u=np.clip(-.85*ys[-1],-1,1)
        a,w=m.system_step(a,w,u,p,disturbances[t]);ys.append(a+eta[t+1]);us.append(u)
    a0=a.copy();w0=w.copy()
    e_all=[]; u_all=[]; v_all=[]; awmax=max(float(np.abs(a).max()),float(np.abs(w).max()))
    for k in range(LENGTH):
        r=(rconst if kind=='constant' else (0.,.45,-.35,.62,-.2)[min(k//33,4)])
        ah,gh=m.param_estimate(ys,us)
        U,preds=m.candidates_pred(method,ys,us,ah,gh,trained,p)
        ix=np.argmin((preds-r)**2+.009*U**2,axis=1)
        u=U[np.arange(N),ix]
        a,w=m.system_step(a,w,u,p,disturbances[WARM+k]);ys.append(a+eta[WARM+k+1]);us.append(u)
        e_all.append(a-r);u_all.append(u)
        awmax=max(awmax,float(max(np.abs(a).max(),np.abs(w).max())))
    E=np.stack(e_all,axis=1);U=np.stack(u_all,axis=1)
    ef=E[:,-55:]; uf=U[:,-55:]
    metrics={'kind':kind,'scenario':scenario,'target':rconst if kind=='constant' else 'piecewise',
      'train_group':group,'nn_init':init,'method':method,'sample_n':N,
      'rmse_all':float(np.sqrt(np.mean(E**2))),
      'rmse_last55':float(np.sqrt(np.mean(ef**2))),
      'mae_last55':float(np.mean(abs(ef))),
      'p95_abs_last55':float(np.quantile(abs(ef),.95)),
      'mean_offset_last55':float(np.mean(ef)),
      'fraction_abs_error_gt_0p05_last55':float(np.mean(abs(ef)>.05)),
      'mean_abs_action_change':float(np.mean(abs(np.diff(U,axis=1)))),
      'fraction_action_change':float(np.mean(np.diff(U,axis=1)!=0)),
      'mean_abs_u_last55':float(np.mean(abs(uf))),
      'max_abs_physical_state':awmax,
      'box_violations':int(awmax>3+1e-10),
      'fraction_saturation':float(np.mean(abs(U)>=1)),
      'iae_all':float(m.DT*np.mean(np.sum(abs(E),axis=1))) }
    # For controlled example trace, export a single realization
    tr={'error':E[0].copy(),'u':U[0].copy()}
    return metrics,tr

def main(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    vertex,equilibrium,theory=theoretical_certificate()
    pd.DataFrame(vertex).to_csv(out/'lyapunov_vertices.csv',index=False)
    pd.DataFrame(equilibrium).to_csv(out/'equilibrium_feasibility.csv',index=False)
    (out/'theory_certificate.json').write_text(json.dumps(theory,ensure_ascii=False,indent=2),encoding='utf-8')
    trained={};train_rows=[]
    for grp in TRAIN_GROUPS:
      for init in INIT_GROUPS:
        trained[(grp,init)],loss=m.learn(grp,init)
        train_rows.append({'train_group':grp,'init':init,**loss})
        print('trained',grp,init,'elapsed',round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(train_rows).to_csv(out/'training.csv',index=False)
    records=[];sample_trace=[]
    scenario_index={x:i for i,x in enumerate(SCENARIOS)}
    for scenario in SCENARIOS:
      p,noise=m.SCENARIOS[scenario]
      for kind,rset in [('constant',TARGETS),('piecewise',(None,))]:
        for r in rset:
          # exact environmental realization shared among models/weights
          seed=71000+777*scenario_index[scenario]+(int((r+1)*100) if r is not None else 999)
          for grp in TRAIN_GROUPS:
           for init in INIT_GROUPS:
            for method in METHODS:
             metr,tr=simulate(p,noise,method,trained[(grp,init)],grp,init,scenario,kind,r,seed)
             records.append(metr)
             if grp==0 and init==0 and scenario=='compound_shift' and kind=='constant' and r==.45:
                sample_trace.append({'method':method,'e':tr['error'],'u':tr['u']})
          print('scenario',scenario,'kind',kind,'r',r,'elapsed',round(time.monotonic()-start,1),flush=True)
    runs=pd.DataFrame(records);runs.to_csv(out/'runs.csv',index=False)
    metrics=('rmse_all','rmse_last55','mae_last55','p95_abs_last55','mean_offset_last55',
             'fraction_abs_error_gt_0p05_last55','fraction_action_change','mean_abs_action_change',
             'max_abs_physical_state','fraction_saturation','iae_all')
    summary=runs.groupby(['kind','scenario','target','method'],as_index=False,dropna=False)[list(metrics)].mean()
    summary.to_csv(out/'summary.csv',index=False)
    # Determine whether NN improves against mathematical reconf in paired configurations, no p-values from shared seeds.
    piv=runs.pivot(index=['kind','scenario','target','train_group','nn_init'],columns='method',values='rmse_last55').reset_index()
    piv['delta_reconf_minus_hybrid']=piv['M0_reconfigured']-piv['M0_reconfigured_plus_NN']
    piv['delta_reconf_minus_poly']=piv['M0_reconfigured']-piv['M0_reconfigured_plus_poly']
    piv['delta_fixed_minus_hybrid']=piv['M0_fixed']-piv['M0_reconfigured_plus_NN']
    piv.to_csv(out/'paired_deltas.csv',index=False)
    # plot steady error at target .45 for three models
    sl=summary[(summary.kind=='constant') & (summary.target==.45)].copy()
    labels={'M0_fixed':'M0', 'M0_reconfigured':'M0 + reassessment', 'M0_reconfigured_plus_NN':'M0+reassessment+NN', 'M0_reconfigured_plus_poly':'M0+reassessment+poly'}
    col=list(labels);x=np.arange(len(SCENARIOS));width=.18
    fig,ax=plt.subplots(figsize=(10,5))
    for i,method in enumerate(col):
       vals=[float(sl[(sl.scenario==sc)&(sl.method==method)]['rmse_last55'].iloc[0]) for sc in SCENARIOS]
       ax.bar(x+(i-1.5)*width,vals,width,label=labels[method])
    ax.set_xticks(x,list(SCENARIOS));ax.set_ylabel('Tracking RMSE, last 55 steps (toy temperature units)');ax.set_title('Experiment 71: nonzero reference + feedback, r=0.45')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'tracking_constant.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    for row in sample_trace:ax.plot(row['e'],label=row['method'])
    ax.set_xlabel('Step after warmup');ax.set_ylabel('Tracking error A-r');ax.set_title('Compound parameter shift, target 0.45, same environment');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'tracking_trace.png',dpi=150);plt.close(fig)
    metadata={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__, 'pandas':pd.__version__,'scipy':scipy.__version__,
      'source67_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
      'code71_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'training_group_seeds':[m.TRAIN_SEEDS[i] for i in TRAIN_GROUPS],
      'nn_weight_seeds':[m.NET_SEEDS[i] for i in INIT_GROUPS],
      'n_test_per_configuration':N,'steps_after_warmup':LENGTH,'warmup':WARM,
      'scenarios':SCENARIOS,'targets':TARGETS,'methods':METHODS,
      'reference_piecewise':'(0,.45,-.35,.62,-.2) each 33 steps',
      'disturbance_clipping':0.06,
      'limitations':'fixed offline NN; 2 train groups x 2 initializations; synthetic thermal system; no physical drive; grid controller never guarantees exact nonzero tracking; no formal stability proof for complete NN controller'}
    (out/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('rho',theory['rho'],'lambda',theory['lambda'],'runs',len(runs),'seconds',round(time.monotonic()-start,1),flush=True)
    print('target .45, steady RMSE\n',sl.pivot(index='scenario',columns='method',values='rmse_last55').to_string(float_format=lambda z:f'{z:.5f}'))
    print('box all?',not runs.box_violations.any())

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
  args=parser.parse_args();main(args.out)