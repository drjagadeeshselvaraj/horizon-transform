"""Pilot 2: number of scales against frequencies per scale (effective spacing ratio) for the default and the orthogonalized bank."""
import numpy as np, json, time
from lib import *
t0=time.time(); Fs=[1,2,3,4,6,8,16]; res={'F':Fs,'r_eff':[],'M':[],'default':{'mc':[],'sfr':[]},'orthogonalized':{'mc':[],'sfr':[]}}
def whiten(b,Lk=32768,floor=1e-12):
    Km=b.kernel(Lk); G=Km.T@Km; w,V=np.linalg.eigh(G); w=np.maximum(w,floor*w.max()); return (V/np.sqrt(w))@V.T
for F in Fs:
    b=bank_ht(64,F=F); M=int(np.ceil(32/F)); res['M'].append(M); res['r_eff'].append(float(4096.0**(1/(M-1))) if M>1 else None); Wm=whiten(b)
    for name,st in (('default',None),('orthogonalized',lambda u,b=b,Wm=Wm:b.stream(u)@Wm)):
        mcs=[];sfs=[]
        for s in range(3): mcs.append(memory_capacity(b,s,stream=st)); sfs.append(scale_free_recall(b,s,stream=st))
        res[name]['mc'].append(float(np.mean(mcs))); res[name]['sfr'].append(float(np.mean(sfs)))
    print('F',F,'M',M,'r',round(res['r_eff'][-1],2),{k:(round(res[k]['mc'][-1],1),round(res[k]['sfr'][-1],2)) for k in ('default','orthogonalized')},round(time.time()-t0),flush=True)
json.dump(res,open('out/e16b.json','w'))
