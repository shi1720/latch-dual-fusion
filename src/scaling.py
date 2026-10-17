"""Timing of fusion alone, independent of assignment matrix size."""
import time,json
from pathlib import Path
import numpy as np
from latch import fuse

root=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(1720407);rows=[]
for k in (2,4,8):
 for d in (128,512,2048,8192,32768):
  z=rng.integers(-100,101,size=(k,d));a=rng.choice([-1,1],size=d)
  fuse(z,a)
  times=[]
  for repeat in range(11):
   start=time.perf_counter();ans=fuse(z,a);times.append(time.perf_counter()-start)
  rows.append(dict(k=k,d=d,nodes=len(ans.masks),edges=ans.implication_count,
                   times=times,median_seconds=float(np.median(times))))
(root/'results/scaling.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='times'} for r in rows],indent=2))
