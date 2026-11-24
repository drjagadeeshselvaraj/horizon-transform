import numpy as np, json, time
from lib import *
t0=time.time()
def softplus(x): return np.log1p(np.exp(np.clip(x,-30,30)))
def sigm(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def data(n,T,rng):
    v=rng.standard_normal((n,T)); f=np.zeros((n,T)); pm=np.zeros(n,int)
    for i in range(n):
        ps=rng.choice(T,3,replace=False); f[i,ps]=1; pm[i]=ps.max()
    return v,f,v[np.arange(n),pm]          # target: value at the LAST of 3 flagged steps (needs flush-and-hold)
def run(model,theta,v,f):
    ab,cb,ad,cd=theta; n,T=v.shape
    if model=='HT-A (doc: omega ungated)': h,w=ht_default(32,F=2,hmax=4096); lamc=-1/h+0j; om=w; K=32; mode='doc'
    elif model=='HT-A (omega scaled by Delta)': h,w=ht_default(32,F=2,hmax=4096); K=32; mode='zoh'
    elif model=='Selective real-pole SSM (Mamba-style)': K=64; mode='real'
    else: K=32; h,w=ht_default(32,F=2,hmax=4096); mode='doc'
    X=np.zeros((n,K),complex if mode!='real' else float)
    for t in range(T):
        D=softplus(ad*f[:,t]+cd)[:,None]; g=(sigm(ab*f[:,t]+cb)*v[:,t])[:,None]
        if mode=='doc': lam=np.exp(-D/h[None]+1j*w[None])
        elif mode=='zoh': lam=np.exp(D*(-1/h[None]+1j*w[None]))
        else: lam=np.exp(-D*(np.arange(K)[None]+1.0)*0.02)       # S4D-real A_n=-(n+1), time unit 0.02
        X=lam*X+g
    F=np.concatenate([X.real,X.imag],1) if mode!='real' else X
    return F
def r2_ridge(Ftr,ytr,Fva,yva,Fte=None,yte=None):
    mu,sd=Ftr.mean(0),Ftr.std(0)+1e-12; A=(Ftr-mu)/sd; best=(-9,None)
    w,V=np.linalg.eigh(A.T@A); B=V.T@(A.T@(ytr-ytr.mean()))
    for a in (1e-6,1e-4,1e-2,1,1e2):
        W=V@(B/(w+a*len(A))); pv=((Fva-mu)/sd)@W+ytr.mean(); r=1-np.mean((yva-pv)**2)/np.var(yva)
        if r>best[0]: best=(r,W)
    if Fte is None: return best[0]
    pt=((Fte-mu)/sd)@best[1]+ytr.mean(); return 1-np.mean((yte-pt)**2)/np.var(yte)
MODELS=['HT (time-invariant, no gating)','HT-A (doc: omega ungated)','HT-A (omega scaled by Delta)','Selective real-pole SSM (Mamba-style)']
def search(model,seed,Tt=128,n=300,rand=80,climb=80):
    rng=np.random.default_rng(seed); dtr=data(n,Tt,rng); dva=data(n,Tt,rng)
    def obj(th): return r2_ridge(run(model,th,dtr[0],dtr[1]),dtr[2],run(model,th,dva[0],dva[1]),dva[2])
    if model.startswith('HT (time'): return (0,0,0,0),obj((0,0,0,0))
    best=(None,-9)
    for _ in range(rand):
        th=(rng.normal(0,6),rng.normal(0,3),rng.normal(0,6),rng.normal(-3,4)); v_=obj(th)
        if v_>best[1]: best=(th,v_)
    th,bv=best
    for it in range(climb):
        c=tuple(np.array(th)+rng.normal(0,1.5*(1-it/climb)+.1,4)*(rng.random(4)<.6)); v_=obj(c)
        if v_>bv: th,bv=c,v_
    return th,bv
R={m:{} for m in MODELS}; TS=[128,512,1024]
for seed in range(5):
    for m in MODELS:
        th,bv=search(m,seed)
        for T in TS:
            rng=np.random.default_rng(900+seed); d1=data(400,T,rng); d2=data(200,T,rng); d3=data(400,T,rng)
            r=r2_ridge(run(m,th,d1[0],d1[1]),d1[2],run(m,th,d2[0],d2[1]),d2[2],run(m,th,d3[0],d3[1]),d3[2])
            R[m].setdefault(str(T),[]).append(float(r))
        R[m].setdefault('theta',[]).append([float(x) for x in th]); R[m].setdefault('val',[]).append(float(bv))
    print('seed',seed,{m:[round(R[m][str(T)][-1],3) for T in TS] for m in MODELS},round(time.time()-t0),flush=True)
json.dump(R,open('out/e4.json','w')); print('total',time.time()-t0)
