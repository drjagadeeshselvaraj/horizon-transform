"""Pilot 3: cascades of order 1-4 (gammatone-style) at equal total mean delay, constant-Q classification (3 seeds)."""
import numpy as np, json, time, scipy.signal as ss
from lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t0=time.time(); fs=128; Lc=1024; FREQS=np.geomspace(2,40,32); wq=2*np.pi*FREQS/fs; hq=2.0*fs/(FREQS*2)
def make_ds(n,rng,a):
    X=np.empty((n,Lc)); yv=rng.integers(0,3,n); centers=np.array([6.,10.,20.])
    for i in range(n):
        f=centers[yv[i]]*rng.uniform(.85,1.15); t=np.arange(Lc)/fs; am=1+.5*np.sin(2*np.pi*rng.uniform(.1,.5)*t+rng.uniform(0,6))
        X[i]=np.cos(2*np.pi*f*t+rng.uniform(0,6.28))*am*a+ss.lfilter([1],[1,-0.97],rng.standard_normal(Lc))*0.25+0.35*rng.standard_normal(Lc)
    return X,yv
def feats(X,order):
    hs=hq/order; lam=ht_poles(hs,wq); out=[]
    for x in X:
        y=x.astype(complex); P=np.empty(len(lam))
        for k,l in enumerate(lam):
            z=y
            for _ in range(order): z=ss.lfilter([1-abs(l)],[1,-l],z)
            P[k]=np.log(np.mean(np.abs(z[300:])**2)+1e-12)
        out.append(P)
    return np.array(out)
res={}
for amp in (0.07,0.10):
    for order in (1,2,3,4):
        accs=[]
        for s in range(3):
            rng=np.random.default_rng(300+s); Xtr,ytr=make_ds(450,rng,amp); Xte,yte=make_ds(450,rng,amp)
            Ftr,Fte=feats(Xtr,order),feats(Xte,order); sc=StandardScaler().fit(Ftr); clf=LogisticRegression(C=1.0,max_iter=3000).fit(sc.transform(Ftr),ytr); accs.append(clf.score(sc.transform(Fte),yte))
        res[f'{order}|{amp}']=[float(np.mean(accs)),float(np.std(accs))]
    print('amp',amp,{k:round(v[0]*100,1) for k,v in res.items() if k.endswith(str(amp))},round(time.time()-t0),flush=True)
# selectivity: relative bandwidth (-3 dB) of the cascade response for a channel at 10 Hz
f0=10.0; k=int(np.argmin(abs(FREQS-f0))); sel={}
for order in (1,2,3,4):
    l=ht_poles(hq[k]/order,wq[k]); om=np.linspace(0,np.pi,20000); H=((1-abs(l))/(1-l*np.exp(-1j*om)))**order; mag=np.abs(H)**2; pk=mag.max(); sel[order]=float((om[mag>=pk/2].max()-om[mag>=pk/2].min())/(wq[k]))
res['relative_bandwidth']=sel; res['group_delay_samples']={str(o):float(o*hq[k]/o) for o in (1,2,3,4)}
json.dump(res,open('out/e16c.json','w')); print('relbw',sel)
