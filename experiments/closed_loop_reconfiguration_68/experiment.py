"""Educational checkpoint 08. Step-response metrics and Lyapunov diagnostic.

Uses only synthetic two-temperature thermostat from experiment 67.
A proof for an ideal passive/linear feedback law DOES NOT certify the
neural-network, quantized, noise-affected or time-varying reference loop.
"""
import importlib.util, pathlib, datetime, json, platform, hashlib, time
import numpy as np, pandas as pd, torch
ROOT=pathlib.Path(__file__).resolve().parent; ROOT.mkdir(exist_ok=True)
SRC=ROOT.parent/'closed_loop_reconfiguration_67'/'experiment.py'
spec=importlib.util.spec_from_file_location('thermostat67',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
TRAIN_GROUPS=[0,1]; INIT_IDS=[0,1]
METHODS=['M0_fixed','M0_reconfigured','M0_reconfigured_plus_NN','M0_reconfigured_plus_poly','M0_fixed_plus_NN','M0_true_coeff_oracle']
SCENARIOS=list(m.SCENARIOS)
STEP_N=140; STEP_STEPS=220; LYAP_N=120; LYAP_STEPS=140; BOOT_RNG=123

def cmd(method,ys,us,p,trained,r):
    ah,gh=m.param_estimate(ys,us)
    U,P=m.candidates_pred(method,ys,us,ah,gh,trained,p)
    objective=(P-r[:,None])**2+.009*U**2
    return U[np.arange(len(r)),np.argmin(objective,axis=1)]

def eval_step(method,p,noise,trained,seed,n=STEP_N,steps=STEP_STEPS):
    rng=np.random.default_rng(seed)
    # Initial heat state and previous measurements are generated independently of method.
    a=rng.normal(0,.06,n);w=rng.normal(0,.06,n)
    eta=rng.normal(0,noise,(m.WARM+steps+1,n));d=rng.normal(0,.025,(m.WARM+steps,n))
    ys=[a+eta[0]];us=[]
    for t in range(m.WARM):
        # Fixed, mild, bounded excitation shared across all candidates, BEFORE step.
        u=np.full(n,[.25,-.2,.2,-.25,0.][min(t//6,4)])
        a,w=m.system_step(a,w,u,p,d[t]);ys.append(a+eta[t+1]);us.append(u)
    initial=a.copy();target=initial+.35
    hist=[]; uh=[]
    for j in range(steps):
        u=cmd(method,ys,us,p,trained,target)
        a,w=m.system_step(a,w,u,p,d[m.WARM+j]);yh=a+eta[m.WARM+j+1]
        hist.append(a.copy());uh.append(u.copy());ys.append(yh);us.append(u)
    X=np.stack(hist,axis=1);U=np.stack(uh,axis=1)
    E=X-target[:,None]
    # Measured dynamics per independent test trajectory, metrics computed on hidden true temperature.
    tau=np.arange(1,steps+1)*m.DT
    rise10=np.full(n,np.nan);rise90=np.full(n,np.nan);settle=np.full(n,np.nan)
    for i in range(n):
        yrel=X[i]-initial[i]
        ix=np.flatnonzero(yrel>=.035)
        if len(ix):rise10[i]=tau[ix[0]]
        ix=np.flatnonzero(yrel>=.315)
        if len(ix):rise90[i]=tau[ix[0]]
        outside=np.flatnonzero(np.abs(E[i])>.05*.35)
        if len(outside)==0: settle[i]=0
        elif outside[-1]<steps-1: settle[i]=tau[outside[-1]+1]
    p_over=np.maximum(0,(X.max(axis=1)-target)/.35)*100
    per=dict(
        rmse=np.sqrt(np.mean(E**2,axis=1)),
        iae=m.DT*np.sum(np.abs(E),axis=1),
        itae=m.DT*np.sum(np.abs(E)*tau[None,:],axis=1),
        peak_error=np.max(np.abs(E),axis=1),
        overshoot_pct=p_over,
        steady_abs_err=np.mean(np.abs(E[:,-30:]),axis=1),
        t_rise=rise90-rise10,
        t_settle=settle,
        u2=np.mean(U**2,axis=1),
        delta_u=np.mean(np.abs(np.diff(U,axis=1)),axis=1),
        saturation=np.mean(np.abs(U)>=.999,axis=1),
    )
    return per

def energy(a,w,p):
    return .5*(a*a+(p['c']/p['w'])*w*w)

def vdot(a,w,u,p,d=0.):
    return -p['a']*a*a-p['c']*(a-w)**2-p['b']*a**4+a*(p['g']*u+d)

def eval_lyapunov(method,p,noise,trained,seed,n=LYAP_N,steps=LYAP_STEPS,with_disturbance=False):
    rng=np.random.default_rng(seed)
    a=rng.uniform(-.7,.7,n);w=rng.uniform(-.7,.7,n)
    eta=rng.normal(0,noise,(m.WARM+steps+1,n))
    disturbance=rng.normal(0,.035,(m.WARM+steps,n)) if with_disturbance else np.zeros((m.WARM+steps,n))
    ys=[a+eta[0]];us=[];rr=np.zeros(n)
    # Same simple steady reference for the entire trial; avoid mistaking tracking of moving target for Lyapunov decay.
    for t in range(m.WARM):
        u=np.clip(-.5*ys[-1],-1,1)
        a,w=m.system_step(a,w,u,p,disturbance[t]);ys.append(a+eta[t+1]);us.append(u)
    positive_v_steps=0; positive_vdot_steps=0;tot=0
    dv_max=-np.inf;positive_trajectories=np.zeros(n,dtype=bool)
    first=energy(a,w,p);v0=first.copy();vgrowth=np.zeros(n)
    for t in range(steps):
        u=cmd(method,ys,us,p,trained,rr)
        dv_dt=vdot(a,w,u,p,disturbance[m.WARM+t])
        old=energy(a,w,p)
        a,w=m.system_step(a,w,u,p,disturbance[m.WARM+t]);nxt=energy(a,w,p)
        dv=nxt-old
        # Tolerance excludes floating point roundoff.
        positive_v_steps += np.count_nonzero(dv>1e-9)
        positive_vdot_steps += np.count_nonzero(dv_dt>1e-9)
        positive_trajectories |= dv>1e-9
        dv_max=max(dv_max,float(dv.max()));tot+=n
        vgrowth=np.maximum(vgrowth,nxt-v0)
        ys.append(a+eta[m.WARM+t+1]);us.append(u)
    final=energy(a,w,p)
    return dict(positive_discrete_fraction=positive_v_steps/tot,
                positive_vdot_fraction=positive_vdot_steps/tot,
                trajectories_with_any_growth=float(np.mean(positive_trajectories)),
                max_single_dV=dv_max,
                fraction_final_V_below_initial=float(np.mean(final<=first)),
                median_ratio_Vend_Vinitial=float(np.median(final/(first+1e-12))),
                p95_max_growth_from_initial=float(np.quantile(vgrowth,.95)))

def ideal_theorem_check():
    rng=np.random.default_rng(56047)
    ret=[]
    for name,(p,noise) in m.SCENARIOS.items():
        a=rng.uniform(-1.5,1.5,100000);w=rng.uniform(-1.5,1.5,100000)
        for k in [0,.6]:
            u=-k*a
            d=vdot(a,w,u,p)
            theoretical=- (p['a']+p['g']*k)*a*a-p['c']*(a-w)**2-p['b']*a**4
            P=np.diag([1,p['c']/p['w']]);M=np.array([[p['a']+p['g']*k+p['c'],-p['c']],[-p['c'],p['c']]])
            eig=np.linalg.eigvalsh(np.diag(1/np.sqrt(np.diag(P))) @M @np.diag(1/np.sqrt(np.diag(P))))
            ret.append(dict(scenario=name,ideal_gain=k,max_vdot=float(d.max()),positive_samples=int(np.count_nonzero(d>1e-12)),max_formula_diff=float(np.max(np.abs(d-theoretical))),
                continuous_exp_rate_bound=float(2*eig.min()),min_eigenvalue_P=float(np.linalg.eigvalsh(P).min())))
    pd.DataFrame(ret).to_csv(ROOT/'theory_checks.csv',index=False)
    # Common quadratic Lyapunov function for a FINITE FAMILY of ideal continuous-time dynamics.
    # Vc(a,w)=.5*a**2+(.5*kappa)*w**2, kappa independent of plant parameters.
    # Under u=-k*a and no disturbances: Vdot=-[a,w]^T M[a,w]-b*a**4,
    # M=[[pa+pc+pg*k,-(pc+kappa*pw)/2],[-(pc+kappa*pw)/2,kappa*pw]].
    common=[];kappa=2.0;P=np.diag([1.,kappa]);Ri=np.diag([1.,1/np.sqrt(kappa)])
    for scen,(p,_) in m.SCENARIOS.items():
        for gain in [0.0,0.6]:
            Q=np.array([[p['a']+p['c']+p['g']*gain,-.5*(p['c']+kappa*p['w'])],[-.5*(p['c']+kappa*p['w']),kappa*p['w']]])
            ev=np.linalg.eigvalsh(Ri@Q@Ri)
            a=rng.uniform(-1.5,1.5,100000);w=rng.uniform(-1.5,1.5,100000)
            true_vdot=-p['a']*a*a-p['c']*(a-w)**2-p['b']*a**4+p['g']*a*(-gain*a)+(kappa-p['c']/p['w'])*p['w']*w*(a-w)
            matrix_vdot=-(Q[0,0]*a*a+2*Q[0,1]*a*w+Q[1,1]*w*w+p['b']*a**4)
            common.append(dict(scenario=scen,ideal_gain=gain,common_kappa=kappa,min_matrix_eigenvalue=float(ev.min()),exp_rate_bound=float(2*ev.min()),max_sample_vdot=float(matrix_vdot.max()),sample_violations=int(np.count_nonzero(matrix_vdot>1e-10)),max_formula_difference=float(np.max(np.abs(true_vdot-matrix_vdot)))))
    pd.DataFrame(common).to_csv(ROOT/'common_lyapunov.csv',index=False)
    return ret

def main():
    start=time.time()
    trained={};training=[]
    for g in TRAIN_GROUPS:
        for j in INIT_IDS:
            a,loss=m.learn(g,j);trained[(g,j)]=a;training.append(dict(group=g,init=j,**loss))
    pd.DataFrame(training).to_csv(ROOT/'training.csv',index=False)
    theory=ideal_theorem_check()
    rows=[]; step_individual=[]
    for s,scenario in enumerate(SCENARIOS):
        p,noise=m.SCENARIOS[scenario]
        for group in TRAIN_GROUPS:
            for j in INIT_IDS:
                for method in METHODS:
                    per=eval_step(method,p,noise,trained[(group,j)],seed=78000+s*23)
                    vals={k:float(np.nanmean(v)) if np.any(np.isfinite(v)) else None for k,v in per.items()}
                    vals.update(t_rise_observed_pct=float(np.mean(np.isfinite(per['t_rise']))),t_settle_observed_pct=float(np.mean(np.isfinite(per['t_settle']))))
                    rows.append(dict(scenario=scenario,method=method,group=group,nnseed=m.NET_SEEDS[j],**vals))
                    for idx in range(STEP_N):
                        step_individual.append(dict(scenario=scenario,method=method,group=group,nnseed=m.NET_SEEDS[j],trajectory=idx,**{k:float(v[idx]) for k,v in per.items()}))
        print('step done',scenario,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'step_runs.csv',index=False)
    df=pd.DataFrame(rows)
    df.groupby(['scenario','method'],as_index=False).mean(numeric_only=True).to_csv(ROOT/'step_summary.csv',index=False)
    pd.DataFrame(step_individual).to_csv(ROOT/'step_trajectories.csv',index=False)
    lyaps=[]
    for scenario in SCENARIOS:
        p,noise=m.SCENARIOS[scenario]
        for case,eps in [('quiet',0.),('sensor_noise',noise),('sensor_noise_and_disturbance',noise)]:
            for method in METHODS:
                # Fixed training group for exploratory diagnostic; full step evaluation tests multiple groups.
                x=eval_lyapunov(method,p,eps,trained[(0,0)],seed=10000+SCENARIOS.index(scenario)*42,with_disturbance=case.endswith('disturbance'))
                lyaps.append(dict(scenario=scenario,case=case,method=method,**x))
        print('lyapunov numeric',scenario,flush=True)
    pd.DataFrame(lyaps).to_csv(ROOT/'lyapunov_numeric.csv',index=False)
    # Repro metadata and hash manifest
    params=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),origin_script=str(SRC),script_source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pandas=pd.__version__,train_groups=TRAIN_GROUPS,init_ids=INIT_IDS,step_n=STEP_N,step_steps=STEP_STEPS,step_amplitude=.35,dt=m.DT,lyap_n=LYAP_N,lyap_steps=LYAP_STEPS,seed_step_base=78000,seed_lyapunov_base=10000,note='Toy thermostat only. Ideal-law Lyapunov proof not applicable to actual NN-based quantized controller. No control hardware/aviation.')
    (ROOT/'run_metadata.json').write_text(json.dumps(params,indent=2,ensure_ascii=False))
    files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in ROOT.iterdir() if x.is_file()}
    (ROOT/'SHA256.json').write_text(json.dumps(files,indent=2))
    d=pd.read_csv(ROOT/'step_summary.csv'); print('SUMMARY RMSE:');print(d.pivot(index='scenario',columns='method',values='rmse').round(5))
    print('THEORY CHECKS:',max(x['positive_samples'] for x in theory),'max',max(x['max_formula_diff'] for x in theory))
    print('elapsed',time.time()-start)
if __name__=='__main__':main()