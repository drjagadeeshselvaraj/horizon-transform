"""Per-seed paired comparisons of memory capacity and scale-free recall, and sensitivity to observation noise."""
import numpy as np, json, time
from scipy import stats
from lib import *
t0=time.time(); R={}
cfg={'HT default':lambda s:bank_ht(64),'S4D-Lin':lambda s:bank_s4d_lin(64),'LRU (ring)':lambda s:bank_lru(64,seed=s),'HiPPO-LegT':lambda s:bank_legt(64,4096),'STFT (W=256)':lambda s:bank_stft(64,256),'Morlet bank':lambda s:bank_cwt(64),'Delay line':lambda s:bank_delay(64)}
per={k:{'mc':[],'sfr':[]} for k in cfg}
for s in range(5):
    for k,mk in cfg.items():
        b=mk(s); per[k]['mc'].append(float(memory_capacity(b,s))); per[k]['sfr'].append(float(scale_free_recall(b,s)))
    print('seed',s,round(time.time()-t0),flush=True)
R['per_seed']=per
comp={}
for metric in ('mc','sfr'):
    rows=[]
    for k in cfg:
        if k=='HT default': continue
        d=np.array(per['HT default'][metric])-np.array(per[k][metric]); n=len(d); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n); tcr=stats.t.ppf(0.975,n-1); t=m/se if se>0 else np.inf; p=2*(1-stats.t.cdf(abs(t),n-1))
        rows.append({'vs':k,'mean':float(m),'lo':float(m-tcr*se),'hi':float(m+tcr*se),'p':float(p)})
    order=np.argsort([r['p'] for r in rows]); m_=len(rows); adj=[0]*m_; prev=0.0
    for rank,i in enumerate(order): prev=max(prev,min(1.0,(m_-rank)*rows[i]['p'])); adj[i]=prev
    for r,a in zip(rows,adj): r['p_holm']=float(a)
    comp[metric]=rows
R['paired']=comp
# ---- sensitivity to observation noise (seeds 0-2)
noise=[1e-3,1e-2,1e-1]; sens={k:{'mc':[],'sfr':[]} for k in cfg}
for nz in noise:
    for k,mk in cfg.items():
        mcs=[];sfs=[]
        for s in range(3): b=mk(s); mcs.append(memory_capacity(b,s,noise=nz)); sfs.append(scale_free_recall(b,s,noise=nz))
        sens[k]['mc'].append(float(np.mean(mcs))); sens[k]['sfr'].append(float(np.mean(sfs)))
    print('noise',nz,round(time.time()-t0),flush=True)
R['noise']={'levels':noise,'banks':sens}
json.dump(R,open('out/e11a.json','w'),default=float)
for metric in ('mc','sfr'):
    print(metric,[(r['vs'],round(r['mean'],3),round(r['p_holm'],3)) for r in comp[metric]])
print('noise MC',{k:[round(x,1) for x in v['mc']] for k,v in sens.items()})
