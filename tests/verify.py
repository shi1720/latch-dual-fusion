"""Independent bounded exhaustive verification; run from repository root."""
import sys,itertools,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import linear_sum_assignment
from latch import fuse,expression,repair,feasible,hungarian


def closure(vectors):
    # Does not use threshold masks, Boolean functions, flow or the theorem.
    seen=set(map(tuple,vectors));todo=list(seen)
    while todo:
        x=todo.pop()
        for y in list(seen):
            for z in (tuple(map(min,x,y)),tuple(map(max,x,y))):
                if z not in seen:seen.add(z);todo.append(z)
    return seen


def main():
    rng=np.random.default_rng(1720406);counts={}
    cases=0
    # All binary K=3,d=3 inputs and all signed +/-1 objectives.
    for bits in itertools.product((0,1),repeat=9):
        z=np.array(bits).reshape(3,3);cl=closure(z)
        for a in itertools.product((-1,1),repeat=3):
            ans=fuse(z,np.array(a))
            optimum=max(sum(x*y for x,y in zip(a,t)) for t in cl)
            assert ans.value==optimum
            assert tuple(ans.vector) in cl
            assert np.array_equal(expression(z,ans.selected_masks),ans.vector)
            cases+=1
    counts['exhaustive_binary_fusions']=cases
    for k in (1,2,3,4):
        for rep in range(100):
            z=rng.integers(-8,9,size=(k,6));a=rng.integers(-3,4,size=6)
            ans=fuse(z,a);cl=closure(z)
            assert ans.value==max(sum(int(x)*int(y) for x,y in zip(a,t)) for t in cl)
            assert tuple(ans.vector) in cl
            assert np.array_equal(expression(z,ans.selected_masks),ans.vector)
    counts['integer_fusions']=400
    for n in (2,3,4,8,16):
        for rep in range(100):
            c=rng.integers(-20,50,size=(n,n));a=np.r_[np.ones(n,dtype=int),-np.ones(n,dtype=int)]
            zs=np.array([repair(c,rng.integers(-20,20,size=n)) for _ in range(4)])
            ans=fuse(zs,a)
            assert feasible(c,ans.vector)
            assert ans.value>=max(zs@a)
            r,col=linear_sum_assignment(c);ref=int(c[r,col].sum())
            for z in (None,*zs,ans.vector):
                p,dual,stats=hungarian(c,z)
                assert int(c[np.arange(n),p].sum())==ref
    counts['assignment_instances']=500
    counts['assignment_solver_runs']=3000
    # Constructive two-candidate separation from every convex combination.
    c=np.array([[1,2],[2,1]])
    zs=np.array([[1,-2,0,-1],[-2,1,-1,0]])
    ans=fuse(zs,np.array([1,1,-1,-1]))
    assert all(feasible(c,z) for z in zs)
    assert np.all(zs@np.array([1,1,-1,-1])==0)
    assert ans.value==2 and feasible(c,ans.vector)
    counts['separation_example']=True
    for rep in range(100):
        z=rng.integers(-10,11,size=(4,8));a=rng.integers(-3,4,size=8)
        ans=fuse(z,a)
        assert fuse(z[::-1],a).value==ans.value
        assert fuse(np.vstack([z,z[0]]),a).value==ans.value
        assert fuse(z[:2],a).value<=fuse(z[:3],a).value<=ans.value
        shift=rng.integers(-2,3,size=z.shape)
        assert abs(fuse(z+shift,a).value-ans.value)<=2*int(np.abs(a).sum())
    counts['metamorphic_cases']=100
    # The middle antichain used in the two-objective hardness reduction
    # realizes every one of 64 binary assignments (plus a fixed one coordinate).
    masks=[3,5,6,9,10,12]
    z=np.array([[int(m>>h&1) for m in masks]+[1] for h in range(4)])
    assert closure(z)=={x+(1,) for x in itertools.product((0,1),repeat=6)}
    counts['antichain_vectors']=64
    # A feasible coordinate splice need not belong to the generated lattice.
    z=np.array([[1,0,0,1],[0,1,1,0]])
    c=np.ones((2,2),dtype=int);a=np.array([1,1,-1,-1])
    assert all(feasible(c,x) for x in z)
    assert fuse(z,a).value==0
    assert feasible(c,np.array([1,1,0,0]))
    assert (1,1,0,0) not in closure(z)
    counts['menu_limitation']=True
    # Explicit runtime nonmonotonicity for the supplied exact solver.
    c=np.array([[6,0,5,3],[8,6,3,2],[7,5,3,9],[7,4,8,8]])
    z=np.array([[0,-3,5,-5,-2,0,2,-3],[4,-1,-3,4,-2,4,-1,1],
                [2,6,-5,-2,-2,2,3,4]])
    a=np.array([1,1,1,1,-1,-1,-1,-1]);ans=fuse(z,a)
    _,_,s=hungarian(c,z[1]);_,_,t=hungarian(c,ans.vector)
    assert int(z[1]@a)==2 and ans.value==10
    assert s['loops']==6 and t['loops']==8
    counts['runtime_nonmonotonicity']=True
    # Reject unsafe backend capacities and floating input rather than rounding.
    try:fuse(np.array([[0,0],[2**31,0]]),np.array([1,1]))
    except OverflowError:pass
    else:raise AssertionError('capacity guard')
    try:fuse(np.array([[0.,1.]]),np.array([1,1]))
    except TypeError:pass
    else:raise AssertionError('integer guard')
    Path('results').mkdir(exist_ok=True)
    Path('results/verification.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
