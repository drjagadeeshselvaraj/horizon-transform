"""Feature-count sweep for the main banks (seeds 0-2)."""
import numpy as np, json, time
from lib import *
t0=time.time(); NF=[16,32,64,128]; cfg={'HT default':lambda n,s:bank_ht(n),'S4D-Lin':lambda n,s:bank_s4d_lin(n),'LRU (ring)':lambda n,s:bank_lru(n,seed=s),'STFT (W=256)':lambda n,s:bank_stft(n,256),'Delay line':lambda n,s:bank_delay(n)}
out={k:{'mc':[],'sfr':[]} for k in cfg}
for n in NF:
    for k,mk in cfg.items():
        mcs=[];sfs=[]
        for s in range(3): b=mk(n,s); mcs.append(memory_capacity(b,s)); sfs.append(scale_free_recall(b,s))
        out[k]['mc'].append(float(np.mean(mcs))); out[k]['sfr'].append(float(np.mean(sfs)))
    print('nf',n,round(time.time()-t0),flush=True)
json.dump({'nf':NF,'banks':out},open('out/e11b.json','w'))
print({k:[round(x,2) for x in v['sfr']] for k,v in out.items()})
