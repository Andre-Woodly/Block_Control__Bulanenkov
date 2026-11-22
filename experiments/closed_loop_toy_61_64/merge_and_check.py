"""Merge independently reproduced family runs from experiment.py --family ..."""
from pathlib import Path
import pandas as pd, json, hashlib
ROOT=Path(__file__).resolve().parent
FILES=['closed_loop_runs','forecast_horizons','paired_bootstrap','closed_loop_summary','forecast_summary']
for f in FILES:
    frames=[]
    for folder in ['linear','nonlinear']:
        p=ROOT/folder/(f+'.csv')
        assert p.exists(),p
        frames.append(pd.read_csv(p))
    t=pd.concat(frames,ignore_index=True)
    assert t['family'].nunique()==2
    t.to_csv(ROOT/(f+'.csv'),index=False)
    print(f,len(t))
A=pd.read_csv(ROOT/'closed_loop_summary.csv');assert len(A)==50
B=pd.read_csv(ROOT/'forecast_summary.csv');assert len(B)==200
C=pd.read_csv(ROOT/'paired_bootstrap.csv');assert len(C)==30
# Each model has 3 training sets * 4 seeds * 5 scenarios * 2 families
assert len(pd.read_csv(ROOT/'closed_loop_runs.csv'))==600
meta={'date_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
 'reproduction':['python experiment.py --family linear_two_state --out linear','python experiment.py --family nonlinear_two_state --out nonlinear','python merge_and_check.py'],
 'notes':'2 synthetic thermal families; 3 independent training sets each; 4 neural initializations per set; 120 held-out trajectories per scenario; 100 feedback steps; all models use the same fixed regulator, differing in one-step predictor. No EGSP/aviation model or safety certification.',
 'training_files':{'linear':json.loads((ROOT/'linear'/'run_metadata.json').read_text()),'nonlinear':json.loads((ROOT/'nonlinear'/'run_metadata.json').read_text())}}
(ROOT/'run_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
for family in A.family.unique():
    d=A[A.family==family].pivot(index='scenario',columns='method',values='tracking_rmse')
    d=d.reindex(['matched','coupling_shift','higher_noise','nonlinear_shift','compound_shift'])
    d.plot.bar(figsize=(10,5),rot=10)
    plt.ylabel('Closed-loop tracking RMSE (normalized temperature)')
    plt.title(f'{family}: toy chamber feedback; mean of 3 training groups × 4 seeds')
    plt.tight_layout();plt.savefig(ROOT/f'closed_loop_{family}.png',dpi=150);plt.close()
sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*') if p.is_file()}
(ROOT/'SHA256.json').write_text(json.dumps(sha,indent=2,ensure_ascii=False))
print('QA PASS; files=',len(sha))