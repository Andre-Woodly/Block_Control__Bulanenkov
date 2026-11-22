"""Stage 70: robust physical-state ISS bound and switching diagnostic.
BENIGN abstract two-temperature dynamical system; no real equipment, actuator design or deployment.
Imports exact stored experiment-67 source from stage-69 archive. Does NOT prove closed-loop NN stability.
"""
from __future__ import annotations
import sys, json, datetime, platform, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd, scipy, torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import experiment67_source as m
P=np.diag([1.,2.]); pinvhalf=np.diag([1.,1./np.sqrt(2.)]); R=3.0; D=0.06; H=m.DT

def weighted_gain2(p,R=R):
    vals=[]
    for r in (0.,R*R):
        T=np.array([[1-H*(p['a']+p['c']+p['b']*r),H*p['c']],[H*p['w'],1-H*p['w']]])
        vals.append(float(np.linalg.eigvalsh(pinvhalf@T.T@P@T@pinvhalf).max()))
    return max(vals)

def main():
    now=time.monotonic();out=HERE
    pvals=[p for p,_ in m.SCENARIOS.values()]
    rho=max(weighted_gain2(p) for p in pvals)
    lam=np.sqrt(rho);gmax=max(p['g'] for p in pvals)
    denom=1-lam
    safe_lipschitz_total=denom/(H*gmax)
    energy_box=(R*R*P[0,0]+R*R*P[1,1])**.5
    bound=H*(gmax*1+D)/denom
    theorem={'P_diag':[1.,2.], 'R':R,'h':H,'dmax':D,'u_max':1,
        'rho':rho,'sqrt_rho':lam,'g_max':gmax,
        'min_relative_decay':denom,
        'u_gain_in_weighted_norm':H*gmax,
        'disturbance_gain_in_weighted_norm':H,
        'ISS_ultimate_bound_with_u1_d006':bound,
        'max_weighted_norm_in_invariant_box':energy_box,
        'small_gain_threshold_L0_plus_Lhistory':safe_lipschitz_total,
        'small_gain_form':'sqrt(rho)+h*gmax*(L0+Lh)<1',
        'proved_for':'physical state in K, arbitrary saturated controller, time-varying parameters in convex hull; conditional full-loop zero-origin only if origin-compatible Lipschitz controller and small-gain inequality',
        'does_not_prove':'closed-loop NN stability, parameter estimator convergence, tracking with nonzero reference, monotonic Lyapunov decline under noise'}
    records=[]; actions=[]
    xs=np.linspace(-.4,.4,801)
    for group in [0,1]:
      for init in [0,1]:
        trained,loss=m.learn(group,init)
        for method in ['M0_fixed','M0_reconfigured','M0_reconfigured_plus_NN','M0_reconfigured_plus_poly']:
            # static history is a diagnostic input, not a physically reachable evolution.
            ys=[xs.copy() for _ in range(m.WINDOW+1)];us=[np.zeros_like(xs) for _ in range(m.WINDOW)]
            ah,gh=m.param_estimate(ys,us)
            U,pred=m.candidates_pred(method,ys,us,ah,gh,trained,m.NOM)
            cost=pred**2+.009*U**2
            chosen=U[np.arange(len(xs)),np.argmin(cost,axis=1)]
            switch_ids=np.where(np.diff(chosen)!=0)[0]
            origin_idx=int(np.where(np.isclose(xs,0))[0][0])
            # controller zero at origin? necessary (not sufficient) for equilibrium.
            records.append({'group':group,'init':init,'method':method,
                 'control_at_zero':float(chosen[origin_idx]),
                 'zero_action_cost_gap_at_origin':float(np.sort(cost[origin_idx])[1]-np.sort(cost[origin_idx])[0]),
                 'zero_action_region_radius_on_scan':float(min([abs(xs[i]) for i in range(len(xs)) if abs(chosen[i])>1e-12], default=float('nan'))),
                 'switches_over_A_-0p4_to_0p4':int(len(switch_ids)),
                 'smallest_scan_spacing_for_jump':float(np.min(np.diff(xs))) if len(switch_ids) else np.nan,
                 'max_action_jump':float(np.max(np.abs(np.diff(chosen)))) if len(switch_ids) else 0.,
                 'nn_loss_fixed':loss['loss_fixed'],'nn_loss_reconf':loss['loss_reconf']})
            for s in switch_ids:
                 actions.append({'group':group,'init':init,'method':method,
                   'A_left':float(xs[s]),'A_right':float(xs[s+1]),
                   'u_left':float(chosen[s]),'u_right':float(chosen[s+1]),
                   'action_jump':float(chosen[s+1]-chosen[s]),
                   'cost_gap_at_left':float(np.sort(cost[s])[1]-np.sort(cost[s])[0])})
        print('trained',group,init,'elapsed',round(time.monotonic()-now,1),flush=True)
    pd.DataFrame(records).to_csv(out/'controller_diagnostic.csv',index=False)
    pd.DataFrame(actions).to_csv(out/'switching_points.csv',index=False)
    (out/'theory_check.json').write_text(json.dumps(theorem,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
      'pandas':pd.__version__,'source67_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
      'code70_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'seeds':{'training_groups':[0,1],'NN_init_index':[0,1]},
      'selected_five_scenarios':list(m.SCENARIOS.keys()),'scan':'A from -0.4 to 0.4 with history A_i=A and old controls=0 (off-trajectory diagnostic)',
      'elapsed_seconds':time.monotonic()-now,
      'limitations':'calculation of conditional small-gain threshold; no certification that actual NN controller satisfies it'}
    (out/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT theorem',theorem,flush=True)
    print(pd.DataFrame(records)[['group','init','method','control_at_zero','switches_over_A_-0p4_to_0p4','max_action_jump']].to_string(index=False),flush=True)
if __name__=='__main__':main()