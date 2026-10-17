"""Fixed seeded synthetic experiment. No test labels enter prototype learning."""
import argparse,json,time,platform,sys,hashlib
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from latch import fuse,repair,feasible,hungarian


def instance(rng,n,family,shift=False):
    if family=='block':
        # Persistent identities and heterogeneous regional price regimes.
        group=np.arange(n)%4
        base=20+50*(group[:,None]!=group[None,:])
        row=rng.normal(0,12 if not shift else 30,size=4)[group]
        col=rng.normal(0,12,size=4)[group]
        c=base+row[:,None]+col[None,:]+rng.exponential(14 if not shift else 35,size=(n,n))
    elif family=='spatial':
        theta=np.arange(n)*2*np.pi/n
        sites=np.c_[np.cos(theta),np.sin(theta)]
        demand=sites+rng.normal(0,.3 if not shift else .8,size=(n,2))
        c=100*np.linalg.norm(sites[:,None,:]-demand[None,:,:],axis=2)
        c+=rng.uniform(0,20,size=(n,n))
    elif family=='iid':
        c=rng.exponential(100 if not shift else 250,size=(n,n))
    else:raise ValueError(family)
    return np.rint(c).astype(np.int64)


def learn(rng,n,family,train,kmax):
    duals=[]
    for _ in range(train):
        c=instance(rng,n,family)
        _,z,_=hungarian(c)
        # Fix each training seed to row median zero before clustering.
        u=z[:n]-int(np.median(z[:n]));duals.append(u)
    x=np.array(duals)
    km=KMeans(n_clusters=kmax,n_init=10,random_state=1720406).fit(x)
    prototypes=np.array([np.rint(np.median(x[km.labels_==h],axis=0)).astype(int)
                         for h in range(kmax)])
    return prototypes,x


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=240)
    p.add_argument('--test',type=int,default=80);p.add_argument('--sizes',type=int,nargs='+',default=[16,32,64])
    p.add_argument('--seeds',type=int,default=3);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];(root/'results').mkdir(exist_ok=True)
    rows=[];models={}
    for seed in range(args.seeds):
      for n in args.sizes:
       for family in ('block','spatial','iid'):
        rng=np.random.default_rng(1720406+seed*100000+n*100+{'block':1,'spatial':2,'iid':3}[family])
        t=time.perf_counter();prototypes,x=learn(rng,n,family,args.train,8)
        train_seconds=time.perf_counter()-t
        models[f'{seed}-{n}-{family}']=prototypes
        for shifted in (False,True):
         for case in range(args.test):
          c=instance(rng,n,family,shifted)
          weights=np.r_[np.ones(n,dtype=int),-np.ones(n,dtype=int)]
          t=time.perf_counter();zs=np.array([repair(c,u) for u in prototypes]);repair_seconds=time.perf_counter()-t
          # A compiled exact solver independently establishes the reference optimum.
          t=time.perf_counter();r,cols=linear_sum_assignment(c);scipy_seconds=time.perf_counter()-t
          optimum=int(c[r,cols].sum())
          allmax=zs.max(axis=0);allmin=zs.min(axis=0)
          best=int(np.argmax(zs@weights));selected=zs[best]
          # Even-cardinality median average is rounded safely toward lower dual value.
          med=np.r_[np.floor(np.median(zs[:,:n],axis=0)),
                    np.ceil(np.median(zs[:,n:],axis=0))].astype(int)
          # Lower/upper order medians are lattice expressions; mixed rounding stays safe.
          assert feasible(c,med)
          convex=np.r_[np.floor(zs[:,:n].mean(axis=0)),np.ceil(zs[:,n:].mean(axis=0))].astype(int)
          policies={'cold':None,'best':selected,'mean':convex,'median':med,
                    'join':allmax,'meet':allmin}
          meanu=np.rint(zs[:,:n].mean(axis=0)).astype(int)
          policies['mean_complete']=repair(c,meanu)
          fusion_info={}
          for k in (2,4,8):
            t=time.perf_counter();ans=fuse(zs[:k],weights);elapsed=time.perf_counter()-t
            policies[f'latch{k}']=ans.vector
            fusion_info[f'latch{k}']=(elapsed,len(ans.masks),ans.implication_count)
          # A cost-aware postprocessing baseline exposes the price of matrix-free fusion.
          policies['latch8_complete']=repair(c,policies['latch8'][:n])
          # Keep every policy, including failures to accelerate.
          for policy,z in policies.items():
            t=time.perf_counter();perm,dual,stats=hungarian(c,z);solve=time.perf_counter()-t
            value=int(c[np.arange(n),perm].sum());assert value==optimum
            lower=int(np.min(c,axis=1).sum()) if z is None else int(z@weights)
            assert lower<=optimum
            fused_time,nodes,edges=fusion_info.get(policy,(0.,0,0))
            rows.append(dict(seed=seed,n=n,family=family,shifted=shifted,case=case,
              policy=policy,optimum=optimum,lower=lower,gap=optimum-lower,
              solve_seconds=solve,fusion_seconds=fused_time,repair_seconds=repair_seconds,
              scipy_seconds=scipy_seconds,training_seconds=train_seconds,
              nodes=nodes,edges=edges,cost_sha256=hashlib.sha256(c.tobytes()).hexdigest(),**stats))
        print(f'finished seed={seed} n={n} family={family}',flush=True)
        # Checkpoint whole completed configurations so interruption loses no results.
        with (root/'results/raw.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        np.savez_compressed(root/'results/prototypes.npz',**models)
    env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
             sklearn=sklearn.__version__,platform=platform.platform(),arguments=vars(args))
    (root/'results/environment.json').write_text(json.dumps(env,indent=2)+'\n')


if __name__=='__main__':main()
