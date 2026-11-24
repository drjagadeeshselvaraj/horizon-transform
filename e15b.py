"""Robustness of the memory probes to the input distribution (3 seeds, 64 features, 1% noise)."""
import numpy as np, json, time, scipy.signal as ss
from lib import *
t0=time.time()
def gen(kind,rng,L):
    if kind=='Gaussian white': return rng.standard_normal(L)
    if kind=='uniform white': return rng.uniform(-1,1,L)*np.sqrt(3)
    if kind=='Student-t (3 d.f.)': x=rng.standard_t(3,L); return x/np.sqrt(3.0)
    if kind=='AR(1), φ = 0.9': x=ss.lfilter([1],[1,-0.9],rng.standard_normal(L+500))[500:]; return x/x.std()
def mc_u(bank,u,seed,noise=1e-2,L=40000,burn=10000,lags=LAGS):
    rng=np.random.default_rng(1000+seed); F=bank.stream(u)[burn:]; Z=(F-F.mean(0))/(F.std(0)+1e-12); Z=Z+noise*rng.standard_normal(Z.shape)
    Y=np.stack([u[burn-d:L-d] for d in lags],1); n=len(Z); r2=np.clip(ridge_curves(Z,Y,int(.4*n),int(.2*n)),0,1); return float(np.trapezoid(np.interp(np.arange(1,lags.max()+1),lags,r2),dx=1.0))
def sfr_u(bank,u,seed,noise=1e-2,L=40000,burn=10000,ds=SF_D):
    rng=np.random.default_rng(2000+seed); F=bank.stream(u)[burn:]; Z=(F-F.mean(0))/(F.std(0)+1e-12); Z=Z+noise*rng.standard_normal(Z.shape)
    C=np.concatenate([[0],np.cumsum(u)]); t=np.arange(burn,L); cols=[]
    for d in ds: s=max(1,d//4); cols.append(C[t-d+1]-C[t-d-s+1])
    Y=np.stack(cols,1); n=len(Z); r2=np.clip(ridge_curves(Z,Y,int(.4*n),int(.2*n)),0,1); return float(r2.mean())
banks={'HT default':lambda s:bank_ht(64),'S4D-Lin':lambda s:bank_s4d_lin(64),'LRU (ring)':lambda s:bank_lru(64,seed=s),'STFT (W=256)':lambda s:bank_stft(64,256),'Morlet bank':lambda s:bank_cwt(64),'Delay line':lambda s:bank_delay(64)}
kinds=['Gaussian white','uniform white','Student-t (3 d.f.)','AR(1), φ = 0.9']; res={k:{d:{'mc':[],'sfr':[]} for d in kinds} for k in banks}
for d in kinds:
    for s in range(3):
        u=gen(d,np.random.default_rng(555+s),40000)
        for k,mk in banks.items():
            b=mk(s); res[k][d]['mc'].append(mc_u(b,u,s)); res[k][d]['sfr'].append(sfr_u(b,u,s))
    print(d,round(time.time()-t0),flush=True)
out={k:{d:{m:[float(np.mean(v)),float(np.std(v))] for m,v in dd.items()} for d,dd in dk.items()} for k,dk in res.items()}
json.dump({'kinds':kinds,'res':out},open('out/e15b.json','w'))
for k,dk in out.items(): print(f'{k:14s}','MC',[round(dk[d]['mc'][0],1) for d in kinds],'SFR',[round(dk[d]['sfr'][0],2) for d in kinds])
