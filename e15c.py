"""True float32 recursion for the default and the orthogonalized HT bank (memory probes, 3 seeds)."""
import numpy as np, json, time
from lib import *
t0=time.time(); b0=bank_ht(64); lam=b0.lam; b=b0.b; Km=b0.kernel(32768); G=Km.T@Km; w,V=np.linalg.eigh(G); w=np.maximum(w,1e-12*w.max()); Wm=(V/np.sqrt(w))@V.T
lam32=lam.astype(np.complex64); b32=b.astype(np.float32)
def stream32(u):
    L=len(u); out=np.empty((L,64),np.float32); s=np.zeros(len(lam),np.complex64); u32=u.astype(np.float32)
    for t in range(L):
        s=lam32*s+b32*u32[t]; out[t,0::2]=s.real; out[t,1::2]=s.imag
    return out
cache={}
def get32(u):
    key=(len(u),float(u[:5].sum()))
    if key not in cache: cache[key]=stream32(u)
    return cache[key]
V_={'float64 recursion, float64 mixing':lambda u:b0.stream(u)@Wm,'float32 recursion, float64 mixing':lambda u:get32(u).astype(np.float64)@Wm,'float32 recursion, float32 mixing':lambda u:(get32(u)@Wm.astype(np.float32)).astype(np.float64),'default bank, float32 recursion':lambda u:get32(u).astype(np.float64)}
res={}
for nz in (1e-2,1e-6):
    for name,fn in V_.items():
        mcs=[];sfs=[]
        for s_ in range(3): mcs.append(memory_capacity(b0,s_,noise=nz,stream=fn)); sfs.append(scale_free_recall(b0,s_,noise=nz,stream=fn))
        res[f'{name}|{nz}']={'mc':float(np.mean(mcs)),'sfr':float(np.mean(sfs))}; print(name,nz,round(res[f'{name}|{nz}']['mc'],1),round(res[f'{name}|{nz}']['sfr'],2),flush=True)
json.dump(res,open('out/e15c.json','w'))
