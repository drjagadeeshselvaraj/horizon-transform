"""Pilot 4: irregular sampling. Constant-Q HT features from randomly retained samples: naive (index-time), zero-filled (regular grid) and time-aware (elapsed-time decay)."""
import numpy as np, json, time, scipy.signal as ss
from lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t0=time.time(); fs=128; Lc=1024; FREQS=np.geomspace(2,40,32); wq=2*np.pi*FREQS/fs; hq=2.0*fs/(FREQS*2); lam=ht_poles(hq,wq); bg=ht_gain(lam)
def make_ds(n,rng,a=0.15):
    X=np.empty((n,Lc)); yv=rng.integers(0,3,n); centers=np.array([6.,10.,20.])
    for i in range(n):
        f=centers[yv[i]]*rng.uniform(.85,1.15); t=np.arange(Lc)/fs; am=1+.5*np.sin(2*np.pi*rng.uniform(.1,.5)*t+rng.uniform(0,6))
        X[i]=np.cos(2*np.pi*f*t+rng.uniform(0,6.28))*am*a+ss.lfilter([1],[1,-0.97],rng.standard_normal(Lc))*0.25+0.35*rng.standard_normal(Lc)
    return X,yv
def f_zero(x,keep):                  # missing samples set to zero on the regular grid
    y=np.where(keep,x,0.0).astype(complex); P=[]
    for l,b in zip(lam,bg): z=ss.lfilter([b],[1,-l],y); P.append(np.log(np.mean(np.abs(z[300:])**2)+1e-12))
    return np.array(P)
def f_steps(x,keep,aware):           # recursion over retained samples only; aware uses the elapsed time as the exponent
    idx=np.flatnonzero(keep); s=np.zeros(len(lam),complex); P=np.zeros(len(lam)); cnt=0; last=-1; ll=np.log(lam)
    for i in idx:
        dt=(i-last) if aware else 1; s=np.exp(ll*dt)*s+bg*x[i]; last=i
        if i>=300: P+=np.abs(s)**2; cnt+=1
    return np.log(P/max(cnt,1)+1e-12)
res={}
def run(ptrain,ptest,name,seeds=2):
    accs=[]
    for s in range(seeds):
        rng=np.random.default_rng(300+s); Xtr,ytr=make_ds(300,rng); Xte,yte=make_ds(300,rng)
        def ft(X,p,seed):
            mrng=np.random.default_rng(seed); out=[]
            for x in X:
                keep=mrng.random(len(x))<p
                if p<1: keep[:300]=True
                out.append(f_zero(x,keep) if name=='zero-filled' else f_steps(x,keep,name=='time-aware'))
            return np.array(out)
        Ftr=ft(Xtr,ptrain,900+s); Fte=ft(Xte,ptest,1900+s); sc=StandardScaler().fit(Ftr); clf=LogisticRegression(C=1.0,max_iter=3000).fit(sc.transform(Ftr),ytr); accs.append(clf.score(sc.transform(Fte),yte))
    return [float(np.mean(accs)),float(np.std(accs))]
for cond in ('matched','trained on regular samples'):
    for p in (1.0,0.7,0.5,0.3):
        for name in ('naive (index time)','zero-filled','time-aware'):
            res[f'{cond}|{name}|{p}']=run(p if cond=='matched' else 1.0,p,name)
        print(cond,p,{k.split('|')[1]:round(v[0]*100,1) for k,v in res.items() if k.startswith(cond) and k.endswith('|'+str(p))},round(time.time()-t0),flush=True)
json.dump(res,open('out/e16d.json','w'))
