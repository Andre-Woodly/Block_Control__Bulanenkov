"""Experiment 73: identifiability and sensor-noise conditioning in a BENIGN
synthetic two-temperature model. Does not implement physical actuator control.

Verifies observability rank, reconstruction conditioning, and sensor-noise
bias of the legacy 18-transition ridge estimate (experiment 72).
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, platform
from pathlib import Path
import numpy as np, pandas as pd

H=0.2
NOM={"a":0.13,"c":0.09,"w":0.045,"b":0.055,"g":0.8}
COMPOUND={**NOM,"a":0.20,"c":0.145,"w":0.028,"b":0.23,"g":0.61}
N_MC=160000
SEED=7319
PARAM_WINDOW=18
REGULARIZATION_A=1.5
PRIOR_A=0.22


def step(A,W,u,p):
    return (A+H*(-p["a"]*A+p["c"]*(W-A)-p["b"]*A**3+p["g"]*u),
            W+H*p["w"]*(A-W))


def obs_matrix(A,p):
    f11=1-H*(p["a"]+p["c"]+3*p["b"]*A*A)
    return np.array([[1.,0.],[f11,H*p["c"]]],dtype=float)


def reconstruct_W(y0,y1,u,p):
    return ((y1-y0)/H+(p["a"]+p["c"])*y0+p["b"]*y0**3-p["g"]*u)/p["c"]


def estimator_a(y):
    # Exact simplified form for identically u=0 used by the legacy 18-step
    # estimator: z=(y1-y0)/h, regressor x=-y0; g is unexcited.
    y0=y[:-1];y1=y[1:]
    n=y0.shape[0]
    s00=(y0*y0).sum(axis=0)+REGULARIZATION_A
    b0=((-y0)*(y1-y0)/H).sum(axis=0)+REGULARIZATION_A*PRIOR_A
    raw=b0/s00
    return np.clip(raw,.05,.6),raw


def run(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    matrices=[]
    for name,p in [("nominal",NOM),("compound_shift",COMPOUND)]:
      for A in (0.,.45,.62):
        O=obs_matrix(A,p)
        matrices.append({"scenario":name,"A":A,"h_times_c":H*p["c"],
          "det_observability":np.linalg.det(O),"rank":int(np.linalg.matrix_rank(O)),
          "condition2":float(np.linalg.cond(O)),
          "F11":O[1,0]})
    pd.DataFrame(matrices).to_csv(out/"observability.csv",index=False)

    # One-step hidden state reconstruction: known TRUE p and u, but noisy y0,y1.
    A=rng.uniform(-.35,.35,N_MC)
    W=rng.uniform(-.45,.45,N_MC)
    U=rng.uniform(-1.,1.,N_MC)
    rows=[]
    for name,p in [("nominal",NOM),("compound_shift",COMPOUND)]:
      A1,_=step(A,W,U,p)
      for sigma in (0.,.018,.085):
        y0=A+rng.normal(0,sigma,N_MC)
        y1=A1+rng.normal(0,sigma,N_MC)
        wh=reconstruct_W(y0,y1,U,p)
        err=wh-W
        rows.append({"scenario":name,"sensor_sigma":sigma,"n":N_MC,
          "rmse_W_reconstruction":float(np.sqrt(np.mean(err*err))),
          "median_absolute_error_W":float(np.median(np.abs(err))),
          "p95_abs_error_W":float(np.quantile(np.abs(err),.95)),
          "naive_W_zero_rmse":float(np.sqrt(np.mean(W*W))),
          "noise_amplification_sqrt2_over_hc":np.sqrt(2.)/(H*p["c"]),
          "note":"True p and applied u known; no process disturbance."})
    pd.DataFrame(rows).to_csv(out/"reconstruction_noise.csv",index=False)

    # No input and state identically zero => true parameters NEVER enter output.
    # Yet the 18-step regression returns different a_hat as sensor noise changes.
    bias_rows=[]
    for sigma in (0.,.018,.085):
      y=rng.normal(0,sigma,(PARAM_WINDOW+1,N_MC))
      ah,raw=estimator_a(y)
      # Ratio of expectations is NOT exactly the expected estimator; diagnostic only.
      diagnostic=(PARAM_WINDOW*sigma**2/H+REGULARIZATION_A*PRIOR_A)/(PARAM_WINDOW*sigma**2+REGULARIZATION_A)
      bias_rows.append({"sigma":sigma,"n":N_MC,"u":"0 for all steps","A":"0 for all steps",
        "W":"0 for all steps","mean_estimated_slope":float(ah.mean()),
        "std_estimated_slope":float(ah.std()),
        "p05_estimated_slope":float(np.quantile(ah,.05)),
        "p95_estimated_slope":float(np.quantile(ah,.95)),
        "fraction_clipped_upper":float(np.mean(raw>.6)),
        "ratio_of_expected_numerator_denominator":diagnostic,
        "data_identifies_a_or_g":False,
        "note":"No excitation: any physical parameter set produces A=W=0, u=0; fitted slope comes from prior and noise."})
    pd.DataFrame(bias_rows).to_csv(out/"identifiability_failure.csv",index=False)

    # A rigorous counterexample, using different parameters and identical noiseless history.
    yy=[]
    for p_name,p in [("nominal",NOM),("compound_shift",COMPOUND)]:
      a=np.zeros(1);w=np.zeros(1);y=[]
      for i in range(26):
        y.append(float(a[0])); a,w=step(a,w,np.zeros(1),p)
      yy.append((p_name,np.array(y)))
    assert np.array_equal(yy[0][1],yy[1][1])

    report={"date_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "scope":"Synthetic passive two-state heat-exchange equations; no physical drive or industrial control.",
       "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
       "seed":SEED,"n_mc":N_MC,"h":H,"parameter_sets":{"nominal":NOM,"compound_shift":COMPOUND},
       "legacy_estimator":{"window":PARAM_WINDOW,"lambda_a":REGULARIZATION_A,"prior_a":PRIOR_A,
          "clip_a":[.05,.6]},
       "inference_limits":["Known-parameter rank 2 does not prove unknown-parameter identifiability.",
          "One-step reconstruction is not statistically optimal with noise.",
          "A zero-input zero-state experiment cannot identify physical parameters, regardless of sample size.",
          "Monte Carlo standard errors and interval estimates are not surrogate guarantees about broader scenarios."]}
    (out/"run_metadata.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    hs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!="SHA256.json"}
    (out/"SHA256.json").write_text(json.dumps(hs,indent=2),encoding="utf8")
    print("OBSERVABILITY\n",pd.read_csv(out/"observability.csv").to_string(index=False))
    print("NOISE\n",pd.read_csv(out/"reconstruction_noise.csv").to_string(index=False))
    print("IDENTIFIABILITY\n",pd.read_csv(out/"identifiability_failure.csv").to_string(index=False))
    return

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("/mnt/data/exp73"));args=ap.parse_args()
    run(args.out)