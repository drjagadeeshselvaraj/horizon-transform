"""Pilot 1: does the orthogonalized HT bank help beyond the memory probes? Reconstruction, forecasting and classification."""
import numpy as np, json, time, scipy.signal as ss
from lib import *
import e1
from e1 import sig, KINDS, A_ht, A_stft, ls_dec, analytic_snr, N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t0=time.time(); R={}
def whiten_matrix(lam,b,Lk=32768,floor=1e-12):
    tau=np.arange(Lk)[:,None]; kc=b[None]*np.exp(tau*np.log(lam)[None]); Km=np.empty((Lk,2*len(lam))); Km[:,0::2]=kc.real; Km[:,1::2]=kc.imag
    G=Km.T@Km; w,V=np.linalg.eigh(G); w=np.maximum(w,floor*w.max()); return (V/np.sqrt(w))@V.T
# ---- reconstruction (analytic, per-feature 20 dB noise)
testx={k:[sig(k,np.random.default_rng(5000+i)) for i in range(20)] for k in KINDS}
rec={}
for rho in (8,16):
    K=rho//2; h,w=ht_default(K,F=min(2,K),hmax=N); A,ch,lam,b=A_ht(h,w); out={}
    def score(A_,ch_):
        P=ls_dec(A_); return float(np.mean([analytic_snr(A_,ch_,P,testx[k],20,'chan') for k in KINDS]))
    out['HT default']=score(A,ch)
    Wm=whiten_matrix(lam,b); A3=A.reshape(2*K,N,N); Aw=np.einsum('fg,fnm->gnm',Wm,A3).reshape(2*K*N,N); chw=np.repeat(np.arange(2*K),N)
    out['HT default, orthogonalized']=score(Aw,chw); out['cond default']=float(np.linalg.cond(A)); out['cond orthogonalized']=float(np.linalg.cond(Aw))
    As,chs=A_stft(rho); out['STFT (W=128)']=score(As,chs); rec[str(rho)]=out; print('rho',rho,{k:round(v,2) for k,v in out.items()},round(time.time()-t0),flush=True)
R['reconstruction']=rec
# ---- forecasting with observation noise on the features
HZ=[24,168,720]
def series(seed,L=70000):
    rng=np.random.default_rng(seed); t=np.arange(L)
    y=np.cos(2*np.pi*t/24+rng.uniform(0,6.28))+.7*np.cos(2*np.pi*t/168+rng.uniform(0,6.28))+.5*np.cos(2*np.pi*t/2000+rng.uniform(0,6.28))
    return y+ss.lfilter([1],[1,-0.98],0.15*rng.standard_normal(L))+0.3*rng.standard_normal(L)
b0=bank_ht(64); Wm64=whiten_matrix(b0.lam,b0.b)
fc={}
for nz in (1e-2,1e-1):
    for name in ('HT default','HT default, orthogonalized'):
        m=[]
        for s in range(3):
            y=series(s); F=b0.stream(y)
            if 'orth' in name: F=F@Wm64
            burn=9000; mx=max(HZ); L=len(y); Z=F[burn:L-mx]; Z=(Z-Z.mean(0))/(Z.std(0)+1e-12); Z=Z+nz*np.random.default_rng(50+s).standard_normal(Z.shape)
            Y=np.stack([y[burn+H:L-mx+H] for H in HZ],1); n=len(Z); r2=ridge_curves(Z,Y,int(.5*n),int(.15*n),alphas=(1e-8,1e-6,1e-4,1e-2,1)); m.append(1-r2)
        fc[f'{name}|{nz}']=np.mean(m,0).tolist()
    print('forecast noise',nz,{k:np.round(v,3).tolist() for k,v in fc.items() if k.endswith(str(nz))},round(time.time()-t0),flush=True)
R['forecast']={'hz':HZ,'nmse':fc}
# ---- classification: per-feature log energy, default bank vs orthogonalized
fs=128; Lc=1024
def make_ds(n,rng,a=0.10):
    X=np.empty((n,Lc)); yv=rng.integers(0,3,n); centers=np.array([6.,10.,20.])
    for i in range(n):
        f=centers[yv[i]]*rng.uniform(.85,1.15); t=np.arange(Lc)/fs; am=1+.5*np.sin(2*np.pi*rng.uniform(.1,.5)*t+rng.uniform(0,6))
        X[i]=np.cos(2*np.pi*f*t+rng.uniform(0,6.28))*am*a+ss.lfilter([1],[1,-0.97],rng.standard_normal(Lc))*0.25+0.35*rng.standard_normal(Lc)
    return X,yv
FREQS=np.geomspace(2,40,32); wq=2*np.pi*FREQS/fs; hq=2.0*fs/(FREQS*2); bq=ht_bank(hq,wq,'htq'); Wq=whiten_matrix(bq.lam,bq.b)
def fe(F,mode): 
    if mode=='pair': P=F[:,0::2]**2+F[:,1::2]**2
    else: P=F**2
    return np.log(P.mean(0)+1e-9)
cl={}
for amp in (0.10,0.15):
    for name,bank,Wm_,mode in [('default bank, pair power',bank_ht(64,hmax=1024),None,'pair'),('constant-Q bank, pair power',bq,None,'pair'),('constant-Q bank, per-feature energy',bq,None,'feat'),('constant-Q bank, orthogonalized',bq,Wq,'feat')]:
        accs=[]
        for s in range(3):
            rng=np.random.default_rng(300+s); Xtr,ytr=make_ds(450,rng,amp); Xte,yte=make_ds(450,rng,amp)
            def ft(X):
                out=[]
                for x in X:
                    F=bank.stream(x)[300:]
                    if Wm_ is not None: F=F@Wm_
                    out.append(fe(F,mode))
                return np.array(out)
            Ftr,Fte=ft(Xtr),ft(Xte); sc=StandardScaler().fit(Ftr); clf=LogisticRegression(C=1.0,max_iter=3000).fit(sc.transform(Ftr),ytr); accs.append(clf.score(sc.transform(Fte),yte))
        cl[f'{name}|{amp}']=[float(np.mean(accs)),float(np.std(accs))]
    print('classification amp',amp,{k:round(v[0]*100,1) for k,v in cl.items() if k.endswith(str(amp))},round(time.time()-t0),flush=True)
R['classification']=cl; json.dump(R,open('out/e16a.json','w'))
