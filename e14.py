"""Mitigation experiments for the weak exact recall of HT: orthogonalized bank and HT + exact-recall hybrids (64 real features, 5 seeds)."""
import numpy as np, json, time
from lib import *
t0=time.time(); R={}
class Hybrid:
    def __init__(s,parts,nfeat): s.parts=parts; s.nfeat=nfeat
    def stream(s,u): return np.concatenate([p.stream(u) for p in s.parts],1)
def evaluate(mkbank,noise=1e-2,seeds=range(5),stream_fn=None):
    mcs=[];sfs=[]
    for sd in seeds:
        b=mkbank(sd); st=stream_fn(b) if stream_fn else None
        mcs.append(memory_capacity(b,sd,noise=noise,stream=st)); sfs.append(scale_free_recall(b,sd,noise=noise,stream=st))
    return {'mc':[float(np.mean(mcs)),float(np.std(mcs))],'sfr':[float(np.mean(sfs)),float(np.std(sfs))]}
b0=bank_ht(64); Km=b0.kernel(32768); G=Km.T@Km; w,V=np.linalg.eigh(G); w=np.maximum(w,1e-12*w.max()); Wm=(V/np.sqrt(w))@V.T
res={}
for nz in (1e-2,1e-1):
    k=str(nz)
    res.setdefault('HT default',{})[k]=evaluate(lambda s:bank_ht(64),noise=nz)
    res.setdefault('HT, orthogonalized',{})[k]=evaluate(lambda s:bank_ht(64),noise=nz,stream_fn=lambda b:(lambda u:b.stream(u)@Wm))
    res.setdefault('HT 32 + delay line 32',{})[k]=evaluate(lambda s:Hybrid([bank_ht(32),bank_delay(32)],64),noise=nz)
    res.setdefault('HT 48 + delay line 16',{})[k]=evaluate(lambda s:Hybrid([bank_ht(48),bank_delay(16)],64),noise=nz)
    res.setdefault('HT 32 + Legendre window 32',{})[k]=evaluate(lambda s:Hybrid([bank_ht(32),bank_legt(32,512)],64),noise=nz)
    res.setdefault('HT 32 + STFT 32',{})[k]=evaluate(lambda s:Hybrid([bank_ht(32),bank_stft(32,256)],64),noise=nz)
    res.setdefault('S4D-Lin',{})[k]=evaluate(lambda s:bank_s4d_lin(64),noise=nz)
    res.setdefault('Delay line',{})[k]=evaluate(lambda s:bank_delay(64),noise=nz)
    print('noise',nz,round(time.time()-t0),flush=True)
R['res']=res; R['cond_gram']=float(np.linalg.cond(G))
json.dump(R,open('out/e14.json','w'))
for k,v in res.items(): print(f'{k:28s}',{nz:(round(x['mc'][0],1),round(x['sfr'][0],2)) for nz,x in v.items()})
print('gram cond',R['cond_gram'])
