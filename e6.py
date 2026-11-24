"""Decoder comparison for noisy coefficients: pooled (Sec. 4 of the proof chapter), weighted pooled, unweighted least squares, and weighted (GLS) least squares.
Noise: per-channel model of Chapter 8 (noise power in each channel = channel power / SNR, 20 dB)."""
import numpy as np, json, time
import e1
from e1 import N, sig, KINDS, A_ht, A_stft, A_udwt, ls_dec, Q_ht
from lib import ht_poles, ht_gain
t0=time.time(); o1=json.load(open('out/e1.json')); RHOS=[2,4,8,16]; SNR=100.0
def chan_pow(A_rows_y,ch):
    ids=np.unique(ch); return {c:np.mean(A_rows_y[ch==c]**2) for c in ids}
def snr_db(x,err): return 10*np.log10(np.sum(x**2)/err)
def evaluate(A,ch,xs,extra=None):
    """returns dict decoder -> mean output SNR over signals"""
    P=ls_dec(A); colsq=(P**2).sum(0); out={'LS (unweighted)':[], 'GLS (weighted LS)':[]}
    if extra: 
        for k in extra['dec']: out[k]=[]
    for x in xs:
        y=A@x; pw=chan_pow(y,ch); sig2=np.array([pw[c] for c in ch])/SNR+1e-30
        out['LS (unweighted)'].append(snr_db(x,np.sum(sig2*colsq)))
        W=1/sig2; G=(A*W[:,None]).T@A; out['GLS (weighted LS)'].append(snr_db(x,np.trace(np.linalg.inv(G+1e-12*np.trace(G)/N*np.eye(N)))))
        if extra:
            for k,fn in extra['dec'].items(): out[k].append(snr_db(x,np.sum(sig2*(fn(sig2)**2).sum(0))))
    return {k:float(np.mean(v)) for k,v in out.items()}
res={}
testx={k:[sig(k,np.random.default_rng(5000+i)) for i in range(10)] for k in KINDS}
for rho in RHOS:
    K=rho//2; cfgs={}
    from lib import ht_default
    h,w=ht_default(K,F=min(2,K),hmax=N); cfgs['HT default']=(h,w)
    th=o1['ht_tuned'][str(rho)]; cfgs['HT tuned']=(np.array(th[0]),np.array(th[1]))
    for nm,(h,w) in cfgs.items():
        A,ch,lam,b=A_ht(h,w); Qp=Q_ht(lam,b,K)
        def weighted_pooled(sig2,lam=lam,b=b,K=K):
            sk=np.array([sig2[2*k*N] for k in range(K)])              # per-channel noise variance (Re/Im rows share it)
            wk=1/(sk*(1+np.abs(lam)**2)); sc=wk*b/np.sum(wk*b**2); Q=np.zeros((N,2*K*N))
            for k in range(K):
                re0=2*k*N; im0=(2*k+1)*N
                for t in range(N):
                    Q[t,re0+t]+=sc[k]
                    if t>0: Q[t,re0+t-1]+=-sc[k]*lam[k].real; Q[t,im0+t-1]+=sc[k]*lam[k].imag
            return Q
        for kind in KINDS:
            r=evaluate(A,ch,testx[kind],extra={'dec':{'Pooled (Chapter 4 inverse)':lambda s,Qp=Qp:Qp,'Pooled, noise-weighted':weighted_pooled}})
            res.setdefault(nm,{}).setdefault(str(rho),{})[kind]=r
    others=[('STFT (W=128)',)+tuple(A_stft(rho))]+([('Undecimated wavelet (db2)',)+tuple(A_udwt(rho-1))] if rho<=8 else [])
    for nm,A,ch in others:
        for kind in KINDS: res.setdefault(nm,{}).setdefault(str(rho),{})[kind]=evaluate(A,ch,testx[kind])
    print('rho',rho,round(time.time()-t0),flush=True)
json.dump(res,open('out/e6.json','w'),default=float)
K4=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']
for nm in res:
    for rho in ('4','8'):
        if rho in res[nm]:
            d=res[nm][rho]; decs=list(d[K4[0]].keys())
            print(nm,rho,{dc:[round(d[k][dc],1) for k in K4] for dc in decs})
