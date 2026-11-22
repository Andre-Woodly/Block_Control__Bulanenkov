"""Experiment 69: rigorous invariant box + regional discrete Lyapunov for
BENIGN synthetic two-temperature system; numerical diagnostics of complete
history-dependent model-reconfiguration + frozen NN forecast feedback.

No real actuators, aircraft, hydraulic plants, or physical deployment claims.

Usage: python experiment69.py --output /mnt/data/exp69_results
Reuses original archived experiment 67 as `closed_loop_reconfiguration_67/experiment.py`.
"""
from __future__ import annotations
import argparse, json, datetime, platform, sys, time, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, scipy

HERE=Path(__file__).resolve().parent
SOURCE67=HERE/'closed_loop_reconfiguration_67'/'experiment.py'
if not SOURCE67.exists():
    SOURCE67=HERE.parent/'closed_loop_reconfiguration_67'/'experiment.py'
if not SOURCE67.exists():
    raise FileNotFoundError('Source 67 experiment.py not found next to, or one directory above, experiment69.py')
sys.path.insert(0,str(SOURCE67.parent))
import experiment as m

P=np.diag([1.,2.]); rootP=np.diag([1.,np.sqrt(2.)]); rootP_inv=np.diag([1.,1/np.sqrt(2.)]);
R=3.0; D_BOUND=0.06
METHODS=['M0_fixed','M0_reconfigured','M0_reconfigured_plus_NN','M0_reconfigured_plus_poly']
STRESS=['quiet','bounded_noise_disturbance']
TEST_N=120; STEPS=160
SCENARIOS=list(m.SCENARIOS)


def cubic_candidates(p,R):
    h=m.DT; c1=1-h*(p['a']+p['c']); b=p['b']
    vals=[-R,R]
    if c1>0 and b>0:
        stationary=np.sqrt(c1/(3*h*b))
        if stationary<R:vals.extend([-stationary,stationary])
    return vals


def one_step_box_max(p,R=R,d_bound=D_BOUND):
    # worst A+ over all A,W in [-R,R], |u| <=1, |d|<=d_bound
    # maximization is scalar cubic over A; extremes of W,u,d are positive endpoints
    a_cand=cubic_candidates(p,R)
    f=lambda a:(1-m.DT*(p['a']+p['c']))*a -m.DT*p['b']*a**3 +m.DT*p['c']*R +m.DT*(p['g']+d_bound)
    return float(max(f(x) for x in a_cand))


def drift_matrix(p,a_squared=0):
    return np.array([[-p['a']-p['c']-p['b']*a_squared,p['c']],[p['w'],-p['w']]])


def weighted_gain2(p,R=R):
    # Exact discrete-time map with u=d=0:
    # z_next = [I + h (F_lin - b A^2 diag(1,0))] z
    # T is affine in r = A^2 in [0,R^2]; x^T T^TPT x is matrix convex in T.
    ends=[]
    for r in (0.,R*R):
        T=np.eye(2)+m.DT*drift_matrix(p,r)
        eig=np.linalg.eigvalsh(rootP_inv@T.T@P@T@rootP_inv)
        ends.append((float(eig.max()),float(eig.min())))
    return max(x[0] for x in ends),ends


def certificate():
    rows=[]
    for name,(p,noise) in m.SCENARIOS.items():
        mx=one_step_box_max(p); rho,endpoints=weighted_gain2(p)
        # The wall update is a convex combination for h*pw in [0,1].
        wall_ok=(0<=m.DT*p['w']<=1)
        rows.append(dict(scenario=name,R=R,disturbance_bound=D_BOUND,
            min_margin_box=R-mx,max_next_air=mx,wall_update_convex=wall_ok,
            max_weighted_gain2=rho,lyapunov_decay_bound=1-rho,
            unforced_discrete_regional_certificate=bool(rho<1),
            box_certificate=bool(mx<R and wall_ok),
            endpoint_eigenvalues=json.dumps(endpoints)))
    table=pd.DataFrame(rows)
    assert table.box_certificate.all(),table
    assert table.unforced_discrete_regional_certificate.all(),table
    # For any parameters in convex hull of 5 scenarios, the discrete transition
    # matrix for fixed r is affine in parameters, and for all r in [0,R²]
    # is a convex combination of 10 endpoint matrices. Weighted squared norm
    # is matrix convex, hence holds for every point in that convex hull.
    return table


def eval_reconfiguration_loop(trained,method,scenario,case,seed=19478,n=TEST_N,steps=STEPS):
    p,noise=m.SCENARIOS[scenario];rng=np.random.default_rng(seed)
    air=rng.uniform(-.85,.85,n);wall=rng.uniform(-.85,.85,n)
    # Clipped disturbance enforces the deterministic bound in our certificate.
    d=np.clip(rng.normal(0,.03,(m.WARM+steps,n)),-D_BOUND,D_BOUND)
    if case=='quiet':
        d*=0;etas=np.zeros((m.WARM+steps+1,n))
    else:
        etas=rng.normal(0,noise,(m.WARM+steps+1,n))
    ys=[air+etas[0]];us=[]
    # Warmup zero setpoint with shared mild feedback. Exact code inherited from 67.
    for t in range(m.WARM):
        u=np.clip(-.5*ys[-1],-1,1)
        air,wall=m.system_step(air,wall,u,p,d[t])
        ys.append(air+etas[t+1]);us.append(u)
    z0=.5*(air*air+2*wall*wall)
    dV_pos=np.zeros(n,dtype=int);npos_dot=np.zeros(n,dtype=int);npos_bad=np.zeros(n,dtype=int)
    maxabs=np.maximum(np.abs(air),np.abs(wall));maxV=z0.copy()
    V_prev=z0.copy();tr_state=np.empty((steps,n)); controls=np.empty((steps,n));
    for t in range(steps):
        ah,gh=m.param_estimate(ys,us)
        U,Pred=m.candidates_pred(method,ys,us,ah,gh,trained,p)
        cost=Pred**2+ .009*U**2 # reference setpoint = 0
        u=U[np.arange(n),np.argmin(cost,axis=1)]
        assert np.max(np.abs(u))<=1.000001
        pg=p['g']*u+d[m.WARM+t]
        # Evaluate continuous-time derivative for diagnostics, not proof for Euler loop
        vdot=(-(p['a']+p['c'])*air**2+(p['c']+2*p['w'])*air*wall
                -2*p['w']*wall**2-p['b']*air**4+air*pg)
        npos_dot+=(vdot>1e-10)
        npos_bad+=(air*u>1e-10)
        air,wall=m.system_step(air,wall,u,p,d[m.WARM+t])
        V=.5*(air**2+2*wall**2)
        dV_pos+=(V-V_prev>1e-10)
        V_prev=V;maxV=np.maximum(maxV,V)
        maxabs=np.maximum(maxabs,np.maximum(np.abs(air),np.abs(wall)))
        tr_state[t]=air;controls[t]=u
        ys.append(air+etas[m.WARM+t+1]);us.append(u)
    initial_ratio=np.median(V_prev/(z0+1e-12))
    out=dict(scenario=scenario,case=case,method=method,train_group=trained.group,init=trained.init,
        n=n,steps=steps,mean_abs_error=float(np.mean(np.abs(tr_state))),
        root_mean_sq_state=float(np.sqrt(np.mean(tr_state**2))),
        mean_u2=float(np.mean(controls**2)),
        share_saturation=float(np.mean(abs(controls)>=.999)),
        share_steps_positive_deltaV=float(np.mean(dV_pos)/steps),
        share_steps_positive_vdot=float(np.mean(npos_dot)/steps),
        share_steps_A_times_u_positive=float(np.mean(npos_bad)/steps),
        max_state_over_all=float(np.max(maxabs)),
        trajectories_outside_box=int(np.sum(maxabs>R+1e-12)),
        median_final_to_initial_V= float(initial_ratio),
        max_energy=float(maxV.max()))
    return out


class Bundle:
    def __init__(self,predictors,group,init):self.predictors=predictors;self.group=group;self.init=init
    def __getitem__(self,key):return self.predictors[key]


def main(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    proof=certificate();proof.to_csv(out/'certificate.csv',index=False)
    print('Rigorous box margin min',proof.min_margin_box.min(),flush=True)
    print('Discrete contraction rho max',proof.max_weighted_gain2.max(),flush=True)
    # train exactly as existing model 67, no change in structure.
    models=[];training=[]
    for group in [0,1]:
        for init in [0,1]:
            predictors,loss=m.learn(group,init)
            models.append(Bundle(predictors,group,init));training.append(dict(group=group,init=init,**loss))
    pd.DataFrame(training).to_csv(out/'training.csv',index=False)
    rows=[]
    for i,sc in enumerate(SCENARIOS):
        for j,case in enumerate(STRESS):
            for bundle in models:
                for method in METHODS:
                    rows.append(eval_reconfiguration_loop(bundle,method,sc,case,seed=23451+i*100+j*23))
            print('calculated:',sc,case,flush=True)
    r=pd.DataFrame(rows);r.to_csv(out/'runs.csv',index=False)
    ag=r.groupby(['scenario','case','method'],as_index=False).agg({k:'mean' for k in [
        'mean_abs_error','root_mean_sq_state','mean_u2','share_saturation',
        'share_steps_positive_deltaV','share_steps_positive_vdot',
        'share_steps_A_times_u_positive','median_final_to_initial_V',
        'max_energy']})
    maxstate=r.groupby(['scenario','case','method'],as_index=False).max(numeric_only=True)[['scenario','case','method','max_state_over_all','trajectories_outside_box']]
    ag=ag.merge(maxstate,on=['scenario','case','method']);ag.to_csv(out/'summary.csv',index=False)
    assert (r.trajectories_outside_box==0).all()
    meta={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'torch':torch.__version__,'pandas':pd.__version__,
        'experiment':69,'model':'Synthetic discrete Euler two-temperature state with arbitrary clamped feedback',
        'source_67_sha256':hashlib.sha256(SOURCE67.read_bytes()).hexdigest(),
        'study_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sample_dt':m.DT,'box_radius':R,'P_diagonal':[1,2],
        'noise_disturbance_bound':D_BOUND,
        'scenarios':SCENARIOS,'methods':METHODS,'stress_cases':STRESS,
        'training_groups':[0,1],'net_init_indices':[0,1],
        'initial_test_state_seed_pattern':'23451+i*100+j*23',
        'training_trajectories_per_group':m.NTRAIN,
        'test_trajectories_per_group_method_scenario':TEST_N,
        'evaluation_steps':STEPS,'warmup_steps':m.WARM,
        'no_training_data_from_eval':True,
        'scope':'BENIGN abstract thermal system; region invariance for physical state under bounded input, NOT full NN-controller asymptotic stability.'}
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(ag.pivot(index=['scenario','case'],columns='method',values='share_steps_positive_deltaV').round(3).to_string())
    print('FINISHED elapsed seconds',time.monotonic()-begin,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/mnt/data/exp69_results')
    args=parser.parse_args();main(Path(args.output))