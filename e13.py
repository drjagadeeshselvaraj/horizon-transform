"""Timing repeated with five repetitions at every length (d = 64 channels, N = 16 modes), single CPU thread."""
import numpy as np, time, tracemalloc, json, scipy.signal as ss
from lib import *
d=64; N=16; h,w=ht_default(N,F=1,hmax=4096.0); lam=ht_poles(h,w); b=ht_gain(lam); rng=np.random.default_rng(0)
C=(rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(N)
def ht_conv(x):
    T=len(x); k=np.real((C*b)[None]*np.exp(np.arange(T)[:,None]*np.log(lam)[None])).sum(1); n=1<<int(np.ceil(np.log2(2*T)))
    return np.fft.irfft(np.fft.rfft(x,n,axis=0)*np.fft.rfft(k,n)[:,None],n,axis=0)[:T]
def ht_rec(x):
    y=np.zeros(x.shape)
    for n_ in range(N): y+=np.real(C[n_]*ss.lfilter([b[n_]],[1,-lam[n_]],x.astype(complex),axis=0))
    return y
def hta_scan(x):
    T=len(x); D=np.log1p(np.exp(x))[:,:,None]; a=np.exp(-D/h[None,None]+1j*w[None,None]); bb=(b*np.ones(N))[None,None]*x[:,:,None]*(1+0j); s=1
    while s<T:
        bb2=bb.copy(); a2=a.copy(); bb2[s:]=a[s:]*bb[:-s]+bb[s:]; a2[s:]=a[s:]*a[:-s]; a,bb=a2,bb2; s*=2
    return np.real(bb@C)
def attn(x):
    q=x.astype(np.float32); S=q@q.T; S*=np.float32(1/np.sqrt(d)); S-=S.max(1,keepdims=True); np.exp(S,out=S); S/=S.sum(1,keepdims=True); return S@q
fns={'HT (FFT conv)':ht_conv,'HT (recurrence)':ht_rec,'HT-A (parallel scan)':hta_scan,'Self-attention':attn}; Ts=[256,512,1024,2048,4096,8192]; out={k:{} for k in fns}
for T in Ts:
    x=rng.standard_normal((T,d))
    for k,f in fns.items():
        f(x); ts=[]
        for _ in range(5): t=time.perf_counter(); f(x); ts.append(time.perf_counter()-t)
        tracemalloc.start(); f(x); pk=tracemalloc.get_traced_memory()[1]/2**20; tracemalloc.stop(); out[k][str(T)]={'t':ts,'mem':float(pk)}
    print('T',T,{k:round(np.median(v[str(T)]['t'])*1e3,1) for k,v in out.items()},flush=True)
json.dump({'Ts':Ts,'out':out},open('out/e13.json','w'))
