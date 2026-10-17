"""Recreate every held-out input and check retained hashes and key outcomes.

Uses frozen learned banks and regenerates (but does not solve) training costs to
advance each declared random stream. Checks independent SciPy optima and exact
fusion bounds. Wall times are intentionally not replayed.
"""
import sys,json,gzip,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from experiment import instance
from latch import repair,fuse,feasible

data=gzip.decompress((root/'results/raw.jsonl.gz').read_bytes()).decode()
rows=[json.loads(x) for x in data.splitlines()]
records={}
for row in rows:
 key=(row['seed'],row['n'],row['family'],row['shifted'],row['case'])
 records.setdefault(key,{})[row['policy']]=row
models=np.load(root/'results/prototypes.npz')
count=0
for seed in range(3):
 for n in (16,32,64):
  for family,code in [('block',1),('spatial',2),('iid',3)]:
   rng=np.random.default_rng(1720406+seed*100000+n*100+code)
   for _ in range(240):instance(rng,n,family)
   bank=models[f'{seed}-{n}-{family}']
   a=np.r_[np.ones(n,dtype=int),-np.ones(n,dtype=int)]
   for shifted in (False,True):
    for case in range(80):
     c=instance(rng,n,family,shifted)
     r=records[(seed,n,family,shifted,case)]
     digest=hashlib.sha256(c.tobytes()).hexdigest()
     assert len(r)==11 and all(x['cost_sha256']==digest for x in r.values())
     i,j=linear_sum_assignment(c);opt=int(c[i,j].sum())
     assert all(x['optimum']==opt for x in r.values())
     z=np.array([repair(c,u) for u in bank])
     assert int(max(z@a))==r['best']['lower']
     last=None
     for k in (2,4,8):
      ans=fuse(z[:k],a)
      assert feasible(c,ans.vector) and ans.value==r[f'latch{k}']['lower']
      assert last is None or last<=ans.value
      last=ans.value
     assert last>=r['best']['lower']
     count+=1
summary=dict(recreated_inputs=count,checked_run_records=len(rows),checked_fusion_bounds=count*3)
(root/'results/replay_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
