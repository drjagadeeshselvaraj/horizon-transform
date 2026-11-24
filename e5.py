import numpy as np, time, tracemalloc, json, scipy.signal as ss
from lib import *
d=64; N=16
h,w=ht_default(N,F=1,hmax=4096.0); lam=ht_poles(h,w); b=ht_gain(lam); rng=np.random.default_rng(0)
C=(rng.standard_normal(N)+1j*rng.standard_normal(N))/np.sqrt(N)
def ht_conv(x):                       # HT layer as global convolution (kernel = sum_n C_n b_n lam_n^tau), FFT
    T=len(x); k=np.real((C*b)[None]*np.exp(np.arange(T)[:,None]*np.log(lam)[None])).sum(1); n=1<<int(np.ceil(np.log2(2*T)))
    return np.fft.irfft(np.fft.rfft(x,n,axis=0)*np.fft.rfft(k,n)[:,None],n,axis=0)[:T]
def ht_rec(x):                        # streaming recurrence per mode (C loop via lfilter), O(T d N)
    y=np.zeros(x.shape)
    for n_ in range(N): y+=np.real(C[n_]*ss.lfilter([b[n_]],[1,-lam[n_]],x.astype(complex),axis=0))
    return y
def hta_scan(x):                      # input-dependent poles; parallel (Hillis-Steele) associative scan, O(T d N log T)
    T=len(x); D=np.log1p(np.exp(x))[:,:,None]; a=np.exp(-D/h[None,None]+1j*w[None,None]); bb=(b*np.ones(N))[None,None]*x[:,:,None]*(1+0j)
    s=1
    while s<T:
        bb2=bb.copy(); a2=a.copy(); bb2[s:]=a[s:]*bb[:-s]+bb[s:]; a2[s:]=a[s:]*a[:-s]; a,bb=a2,bb2; s*=2
    return np.real(bb@C)
def attn(x):
    q=x.astype(np.float32); S=q@q.T; S*=np.float32(1/np.sqrt(d)); S-=S.max(1,keepdims=True); np.exp(S,out=S); S/=S.sum(1,keepdims=True); return S@q
def bench(f,x,reps):
    f(x); ts=[]
    for _ in range(reps): t=time.perf_counter(); f(x); ts.append(time.perf_counter()-t)
    tracemalloc.start(); f(x); pk=tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return float(np.median(ts)),pk/2**20
Ts=[128,256,512,1024,2048,4096,8192,16384]; R={'T':Ts,'d':d,'N':N}
fns={'HT (FFT conv)':ht_conv,'HT (recurrence, lfilter)':ht_rec,'HT-A (parallel scan)':hta_scan,'Self-attention (1 head, fp32)':attn}
lim={'HT (FFT conv)':16384,'HT (recurrence, lfilter)':16384,'HT-A (parallel scan)':8192,'Self-attention (1 head, fp32)':8192}
for k,f in fns.items():
    R[k]={'t':[],'mem':[]}
    for T in Ts:
        if T>lim[k]: R[k]['t'].append(None); R[k]['mem'].append(None); continue
        x=rng.standard_normal((T,d)); t,m=bench(f,x,5 if T<=2048 else 2); R[k]['t'].append(t); R[k]['mem'].append(m); print(k,T,f"{t*1e3:.1f} ms {m:.0f} MiB",flush=True)
R['flops']={'attn':[4*T*T*d+5*T*T for T in Ts],'ht_rec':[14*T*d*N for T in Ts],'ht_fft':[10*(2*T)*np.log2(2*T)*d+8*N*T for T in Ts]}
json.dump(R,open('out/e5.json','w'),default=float)
