"""Finish experiment 74 from already completed independent training & evaluation outputs."""
import json,datetime,time,hashlib,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import source72 as m
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
SCENARIOS=('nominal','compound_shift','higher_noise');KINDS=('constant','piecewise');ESTIMATORS=('raw','filtered3','moment_adjusted');INIT=(7,19);GROUPS=(101,303,509);NTEST=70
start=time.monotonic()
per=pd.read_csv(out/'per_trajectory.csv');runs=pd.read_csv(out/'runs.csv');summ=pd.read_csv(out/'summary.csv');stat=pd.read_csv(out/'noise_stress.csv')
print('loaded',len(per),len(runs),flush=True)
comp=[];rng=np.random.default_rng(74551)
pivot=per.pivot(index=['scenario','kind','group','init','trajectory'],columns=['estimator','method'],values='traj_rmse')
for scenario in SCENARIOS:
  for kind in KINDS:
    pidx=pivot.loc[(scenario,kind)]
    pairs=[]
    for est in ESTIMATORS:
      pairs += [((est,'noNN'),(est,'NN_L2')),((est,'noNN'),(est,'NN_L18'))]
    for alt in ('filtered3','moment_adjusted'):
      pairs.append((('raw','noNN'),(alt,'noNN')))
      pairs.append((('raw','NN_L18'),(alt,'NN_L18')))
    pairs.append((('raw','noNN'),('oracle','noNN')))
    for base,other in pairs:
      dif=pidx[base]-pidx[other]
      A=np.stack([np.stack([dif.loc[(g,s)].to_numpy() for s in INIT]) for g in range(len(GROUPS))])
      B=500
      sampled=rng.integers(len(GROUPS),size=(B,len(GROUPS)))
      tix=rng.integers(NTEST,size=(B,len(GROUPS),NTEST))
      boots=np.empty(B)
      for j in range(B):
        boots[j]=np.mean([A[g,:,tix[j,k]].mean() for k,g in enumerate(sampled[j])])
      comp.append(dict(scenario=scenario,kind=kind,base='|'.join(base),alternative='|'.join(other),
       delta_rmse=float(A.mean()),ci95_low=float(np.quantile(boots,.025)),ci95_high=float(np.quantile(boots,.975))))
    print('bootstrapped',scenario,kind,'elapsed',round(time.monotonic()-start,2),flush=True)
pd.DataFrame(comp).to_csv(out/'paired.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sub=summ[summ.kind=='constant'];labels=['raw','filtered3','moment_adjusted'];x=np.arange(3)
fig,ax=plt.subplots(figsize=(11,5.5))
for j,label in enumerate(labels):
 vals=[float(sub[(sub.scenario==sc)&(sub.estimator==label)&(sub.method=='noNN')]['rmse_last40'].iloc[0]) for sc in SCENARIOS]
 ax.bar(x+(j-1)*.22,vals,.22,label=label)
ax.set_xticks(x,SCENARIOS);ax.set_ylabel('Tracking RMSE, last 40 steps (toy units)');ax.set_title('Estimator contribution, 18-step history')
ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'estimator_comparison.png',dpi=145);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
for label in labels:
 sl=stat[stat.estimator==label];ax.plot(sl.noise,sl.mean_a,'o-',label=label)
ax.axhline(.22,ls='--',label='nominal prior');ax.set(xlabel='Sensor noise SD',ylabel='Estimated slope',title='Stationary physical state: apparent coefficient drift')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'noise_stress.png',dpi=145);plt.close(fig)
meta={'scope':'Synthetic benign two-temperature model; not a real physical actuator','training_dataset_seeds':GROUPS,
'weight_seeds':INIT,'scenarios':SCENARIOS,'estimators':ESTIMATORS,'NN_windows':(2,18),'history_estimator':18,
'num_training_trajectories_each':120,'test_trajectories_per_config':NTEST,'eval_steps':68,
'stationary_noise_windows_per_setting':40000,
'caution':'Moment correction assumes known sigma and corrects only two moments; due to feedback noise correlations and omitted dynamics it is not a statistically consistent parameter estimator. Filtered3 is not an optimal observer. Oracle has privileged true coefficients for diagnostic comparison.',
'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
'source72_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
print('STATIONARY',stat.to_string(index=False),flush=True)
print('CONSTANT RMSE_LAST40',sub.pivot(index=['scenario','estimator'],columns='method',values='rmse_last40').to_string(float_format=lambda v:f'{v:.5f}'),flush=True)
print('TOTAL',round(time.monotonic()-start,2),flush=True)