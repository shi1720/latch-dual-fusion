"""Rebuild all manuscript empirical tables and plots from retained raw output."""
import json,collections,csv,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

root=Path(__file__).resolve().parents[1];results=root/'results';figures=root/'paper/figures'
figures.mkdir(parents=True,exist_ok=True)
raw=results/'raw.jsonl'
data=raw.read_text() if raw.exists() else gzip.decompress((results/'raw.jsonl.gz').read_bytes()).decode()
rows=[json.loads(x) for x in data.splitlines()]
cases=collections.defaultdict(dict)
for x in rows:cases[(x['seed'],x['n'],x['family'],x['shifted'],x['case'])][x['policy']]=x
assert len(rows)==47520 and len(cases)==4320
assert all(len(v)==11 for v in cases.values())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
colors=['#186B8D','#D27B23','#548644']
summary=[];cells=[]
for fam in ('block','spatial','iid'):
 for shift in (False,True):
  vs=[v for k,v in cases.items() if k[2]==fam and k[3]==shift]
  bg=np.array([v['best']['gap'] for v in vs]);fg=np.array([v['latch8']['gap'] for v in vs])
  total=lambda v,p:v[p]['solve_seconds']+v[p]['repair_seconds']+v[p]['fusion_seconds']
  bank_reduction=[]
  for bank in range(3):
   vv=[v for k,v in cases.items() if k[2]==fam and k[3]==shift and k[0]==bank]
   bank_reduction.append(100*(1-np.mean([v['latch8']['gap'] for v in vv])/np.mean([v['best']['gap'] for v in vv])))
  item=dict(family=fam,shifted=shift,n=len(vs),best_gap=float(bg.mean()),latch_gap=float(fg.mean()),
    reduction_percent=float(100*(1-fg.mean()/bg.mean())),strict=int(np.sum(fg<bg)),ties=int(np.sum(fg==bg)),
    lower_loops=sum(v['latch8']['loops']<v['best']['loops'] for v in vs),
    equal_loops=sum(v['latch8']['loops']==v['best']['loops'] for v in vs),
    higher_loops=sum(v['latch8']['loops']>v['best']['loops'] for v in vs),
    loop_ratio=float(np.mean([v['latch8']['loops']/v['best']['loops'] for v in vs])),
    time_ratio_median=float(np.median([total(v,'latch8')/total(v,'best') for v in vs])),
    bank_reduction_range=[min(bank_reduction),max(bank_reduction)])
  summary.append(item)
  for n in (16,32,64):
   for seed in range(3):
    vv=[v for key,v in cases.items() if key[1]==n and key[2]==fam and key[3]==shift and key[0]==seed]
    b=np.mean([v['best']['gap'] for v in vv]);f=np.mean([v['latch8']['gap'] for v in vv])
    cells.append(dict(family=fam,shifted=shift,n=n,seed=seed,reduction=100*(1-f/b)))
(results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (results/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (results/'cell_reductions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
table=[]
for x in summary:
 table.append(f"{x['family'].capitalize() if x['family']!='iid' else 'IID'} & {'Shift' if x['shifted'] else 'Matched'} & {x['best_gap']:.2f} & {x['latch_gap']:.2f} & {x['reduction_percent']:.2f} & {x['strict']}/720 & {x['time_ratio_median']:.2f}\\\\")
(root/'paper/results_table.tex').write_text('\n'.join(table)+'\n\\bottomrule\n')

fig,axes=plt.subplots(1,2,figsize=(6.8,2.65),sharey=True)
for ax,shift in zip(axes,(False,True)):
 for fam,color in zip(('block','spatial','iid'),colors):
  med=[];lo=[];hi=[]
  for n in (16,32,64):
   r=[x['reduction'] for x in cells if x['family']==fam and x['shifted']==shift and x['n']==n]
   med.append(np.mean(r));lo.append(min(r));hi.append(max(r))
  ax.plot([16,32,64],med,'o-',color=color,label=fam.capitalize(),lw=1.7,ms=4)
  ax.fill_between([16,32,64],lo,hi,color=color,alpha=.13,lw=0)
 ax.set(title='Distribution shift' if shift else 'Matched distribution',xlabel='Assignment size n',xticks=[16,32,64],ylim=(0,30))
 ax.grid(axis='y',alpha=.18)
axes[0].set_ylabel('Reduction in mean dual gap (%)');axes[1].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(figures/'gap.pdf');fig.savefig(figures/'gap.png',dpi=220);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(6.8,2.65))
labels=[];series=[]
for fam in ('block','spatial','iid'):
 for shift in (False,True):
  vv=[v for key,v in cases.items() if key[2]==fam and key[3]==shift]
  series.append([(v['latch8']['solve_seconds']+v['latch8']['repair_seconds']+v['latch8']['fusion_seconds'])/(v['best']['solve_seconds']+v['best']['repair_seconds']) for v in vv])
  labels.append(f'{fam[:3].upper()}\n'+('shift' if shift else 'match'))
bp=axes[0].boxplot(series,showfliers=False,patch_artist=True,widths=.6)
for patch,color in zip(bp['boxes'],np.repeat(colors,2)):patch.set(facecolor=color,alpha=.6)
axes[0].axhline(1,color='#666',ls='--',lw=1);axes[0].set_xticks(range(1,7),labels,fontsize=7)
axes[0].set(ylabel='LATCH-8 / best-seed pipeline time',title='Measured overhead');axes[0].set_ylim(bottom=0)
for fam,color in zip(('block','spatial','iid'),colors):
 r=[x for x in summary if x['family']==fam]
 counts=np.array([[x['lower_loops'],x['equal_loops'],x['higher_loops']] for x in r]).sum(axis=0)
 axes[1].plot(['Lower','Equal','Higher'],counts/1440*100,'o-',label=fam.capitalize(),color=color)
axes[1].set(ylabel='Share of held-out instances (%)',title='Inner loops relative to best seed',ylim=(0,100))
axes[1].legend(frameon=False,fontsize=8);fig.tight_layout()
fig.savefig(figures/'runtime.pdf');fig.savefig(figures/'runtime.png',dpi=220);plt.close(fig)

scaling=json.loads((results/'scaling.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(6.8,2.5))
for k,color in zip((2,4,8),colors):
 r=[x for x in scaling if x['k']==k]
 axes[0].loglog([x['d'] for x in r],[x['median_seconds']*1000 for x in r],'o-',label=f'K={k}',color=color,ms=4)
 axes[1].semilogx([x['d'] for x in r],[x['nodes'] for x in r],'o-',label=f'K={k}',color=color,ms=4)
axes[0].set(xlabel='Vector dimension d',ylabel='Fusion time (ms)',title='Fusion without a cost matrix')
axes[1].set(xlabel='Vector dimension d',ylabel='Observed threshold patterns',title='Compressed cut graph')
axes[0].legend(frameon=False,fontsize=8);fig.tight_layout()
fig.savefig(figures/'scaling.pdf');fig.savefig(figures/'scaling.png',dpi=220);plt.close(fig)

fig,ax=plt.subplots(figsize=(6.8,1.6));ax.set(xlim=(0,10),ylim=(0,2));ax.axis('off')
texts=['K feasible hints\n(u, -v)','Coordinate levels\nand predictor masks','Maximum-weight\nclosure / min cut','Feasible fused dual\n+ lattice expression']
for j,text in enumerate(texts):
 x=.1+2.5*j
 box=FancyBboxPatch((x,.55),2.1,.95,boxstyle='round,pad=.10',fc='#EBF2F5',ec='#186B8D',lw=1)
 ax.add_patch(box);ax.text(x+1.05,1.03,text,ha='center',va='center',fontsize=7.8)
 if j<3:ax.add_patch(FancyArrowPatch((x+2.23,1.03),(x+2.47,1.03),arrowstyle='->',mutation_scale=11,color='#186B8D'))
ax.text(5,.13,'Fusion reads the hint vectors and objective coefficients; feasibility comes from lattice closure.',ha='center',fontsize=8,color='#444')
fig.savefig(figures/'workflow.pdf');fig.savefig(figures/'workflow.png',dpi=220);plt.close(fig)

policies=[]
for p in ('cold','best','mean','median','join','meet','mean_complete','latch2','latch4','latch8','latch8_complete'):
 policies.append(dict(policy=p,mean_gap=float(np.mean([v[p]['gap'] for v in cases.values()])),
   mean_loops=float(np.mean([v[p]['loops'] for v in cases.values()])),
   mean_updates=float(np.mean([v[p]['updates'] for v in cases.values()]))))
(results/'policies.json').write_text(json.dumps(policies,indent=2)+'\n')
meta=dict(instances=len(cases),solver_runs=len(rows),strict=sum(x['strict'] for x in summary),
 ties=sum(x['ties'] for x in summary),median_fusion_ms=float(np.median([v['latch8']['fusion_seconds']*1000 for v in cases.values()])),
 lower_loops=sum(x['lower_loops'] for x in summary),equal_loops=sum(x['equal_loops'] for x in summary),higher_loops=sum(x['higher_loops'] for x in summary),
 median_pipeline_ratio=float(np.median([x for s in series for x in s])),
 median_scipy_seconds=float(np.median([v['best']['scipy_seconds'] for v in cases.values()])))
(results/'headline.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2));print(json.dumps(policies,indent=2))
