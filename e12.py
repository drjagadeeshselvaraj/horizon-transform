"""Classification accuracy against task difficulty (burst amplitude), 3 seeds, five banks."""
import numpy as np, json, time, scipy.signal as ss
from lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t0=time.time(); fs=128; Lc=1024; FREQS=np.geomspace(2,40,32)
def make_ds(n,rng,a):
    X=np.empty((n,Lc)); yv=rng.integers(0,3,n); centers=np.array([6.,10.,20.])
    for i in range(n):
        f=centers[yv[i]]*rng.uniform(.85,1.15); t=np.arange(Lc)/fs; am=1+.5*np.sin(2*np.pi*rng.uniform(.1,.5)*t+rng.uniform(0,6))
        burst=np.cos(2*np.pi*f*t+rng.uniform(0,6.28))*am*a; pink=ss.lfilter([1],[1,-0.97],rng.standard_normal(Lc))*0.25
        X[i]=burst+pink+0.35*rng.standard_normal(Lc)
    return X,yv
def cwt_freq(freqs,w0=6.0):
    Lk=int(6*w0/(2*np.pi*freqs.min())*fs)+8; tau=np.arange(Lk); K=len(freqs); out=np.zeros((Lk,2*K))
    for i,f in enumerate(freqs):
        s=w0/(2*np.pi*f)*fs; m=tau<=6*s; t=tau[m]-3*s; g=np.exp(-t**2/(2*s**2))*np.exp(1j*2*np.pi*f/fs*t); g/=np.sqrt(np.sum(np.abs(g)**2)); out[m,2*i]=g.real; out[m,2*i+1]=g.imag
    return Bank('cwt',2*K,ker=out)
def ht_freq(freqs,cycles=2.0):
    w=2*np.pi*freqs/fs; h=cycles*fs/(freqs*2); return ht_bank(h,w,'htq')
def stft_lin(W=128,fmax=40.0):
    nb=32; bins=np.unique(np.round(np.linspace(2/fs*W,fmax/fs*W,nb)).astype(int)); nb=len(bins); tau=np.arange(W); w=np.hanning(W+2)[1:-1]; ker=np.empty((W,2*nb))
    for i,j in enumerate(bins): ker[:,2*i]=w*np.cos(2*np.pi*j*tau/W); ker[:,2*i+1]=-w*np.sin(2*np.pi*j*tau/W)
    return Bank('stft',2*nb,ker=ker)
cb={'HT, constant-Q':lambda s:ht_freq(FREQS),'HT default':lambda s:bank_ht(64,hmax=1024),'Morlet bank':lambda s:cwt_freq(FREQS),'STFT (W = 128)':lambda s:stft_lin(128),'LRU':lambda s:bank_lru(64,seed=s,rmin=.9,rmax=.99,phimax=np.pi/2)}
def feats(b,X):
    out=[]
    for x in X: F=b.stream(x)[300:]; P=F[:,0::2]**2+F[:,1::2]**2; out.append(np.log(P.mean(0)+1e-9))
    return np.array(out)
AMPS=[0.03,0.05,0.07,0.10,0.15]; res={k:{str(a):[] for a in AMPS} for k in cb}
for a in AMPS:
    for s in range(3):
        rng=np.random.default_rng(300+s); Xtr,ytr=make_ds(450,rng,a); Xte,yte=make_ds(450,rng,a)
        for k,mk in cb.items():
            b=mk(s); Ftr,Fte=feats(b,Xtr),feats(b,Xte); sc=StandardScaler().fit(Ftr); clf=LogisticRegression(C=1.0,max_iter=2000).fit(sc.transform(Ftr),ytr); res[k][str(a)].append(float(clf.score(sc.transform(Fte),yte)))
    print('amp',a,round(time.time()-t0),flush=True)
json.dump({'amps':AMPS,'acc':res},open('out/e12.json','w'))
for k,v in res.items(): print(k,[round(100*np.mean(v[str(a)]),1) for a in AMPS])
