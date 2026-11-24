import numpy as np, json, scipy.signal as ss
from lib import *
rng=np.random.default_rng(0); R={}
# 1 recurrence == unrolled conv == FFT conv
T=4096; x=rng.standard_normal(T); h,w=ht_default(16,F=2,hmax=4096); lam=ht_poles(h,w); b=ht_gain(lam)
X=np.zeros((T,16),complex); s=np.zeros(16,complex)
for t in range(T): s=lam*s+b*x[t]; X[t]=s
k=b[None]*lam[None]**np.arange(T)[:,None]
Xc=np.stack([np.convolve(x,k[:,i])[:T] for i in range(16)],1)
n=8192; Xf=np.fft.ifft(np.fft.fft(x,n)[:,None]*np.fft.fft(k,n,axis=0),axis=0)[:T]
R['rec_vs_conv']=float(np.max(np.abs(X-Xc))/np.max(np.abs(X))); R['rec_vs_fft']=float(np.max(np.abs(X-Xf))/np.max(np.abs(X)))
# 2 exact inverse (noise free), K=1 and K=16, incl. long horizon
def inv(X,lam,b):
    Xp=np.vstack([np.zeros((1,X.shape[1])),X[:-1]]); r=X-lam*Xp
    return np.real((np.conj(b)*r).sum(1)/np.sum(np.abs(b)**2))
R['inv_err_K16']=float(np.max(np.abs(inv(X,lam,b)-x)))
l1=ht_poles([4096.],[0.3]); b1=ht_gain(l1); X1=ss.lfilter([b1[0]],[1,-l1[0]],x.astype(complex))[:,None]
R['inv_err_K1_h4096']=float(np.max(np.abs(inv(X1,l1,b1)-x)))
# 3 pooled LS inverse under noise: bias & variance vs closed form
sig=0.1; N=20000; xs=rng.standard_normal(N); h3,w3=ht_default(8,F=2,hmax=256); l3=ht_poles(h3,w3); b3=ht_gain(l3)
X3=np.stack([ss.lfilter([b3[i]],[1,-l3[i]],xs.astype(complex)) for i in range(8)],1)
nz=sig/np.sqrt(2)*(rng.standard_normal(X3.shape)+1j*rng.standard_normal(X3.shape))
e=inv(X3+nz,l3,b3)-xs
th=0.5*sig**2*np.sum(np.abs(b3)**2*(1+np.abs(l3)**2))/np.sum(np.abs(b3)**2)**2
R['noise_bias']=float(e.mean()); R['noise_var_emp']=float(e.var()); R['noise_var_theory']=float(th)
# 4 stability bound: sup |X| for |x|<=1 vs |b|/(1-|lam|)
rows=[]
for hh,ww in [(50,0.0),(50,0.5),(50,3.0)]:
    l=ht_poles([hh],[ww])[0]; bb=1.0; tau=np.arange(5000); kern=bb*l**tau
    best=max(np.sum(np.abs(np.real(np.exp(-1j*th_)*kern))) for th_ in np.linspace(0,np.pi,721))
    rows.append((hh,ww,float(best),float(bb/(1-abs(l)))))
R['stability']=rows
# 5 horizon coverage
R['scales_needed_4096']=int(np.log2(4096))+1; R['maxh_K12']=2**11
R['residual_weight_at_T_K12']=float(np.exp(-4096/2048)); R['residual_weight_at_T_K13']=float(np.exp(-4096/4096))
# 6 HT-A parallel scan == sequential
def scan(a,bb):                     # inclusive scan of s_t=a_t s_{t-1}+bb_t (Hillis-Steele)
    a=a.copy(); bb=bb.copy(); T=len(a); d=1
    while d<T:
        bb2=bb.copy(); a2=a.copy()
        bb2[d:]=a[d:]*bb[:-d]+bb[d:]; a2[d:]=a[d:]*a[:-d]; a,bb=a2,bb2; d*=2
    return bb
T=2048; xa=rng.standard_normal(T); Wd=0.7; Dl=np.log1p(np.exp(Wd*xa-1)); hk,wk=ht_default(8,F=2,hmax=512)
Ls=np.exp(-Dl[:,None]/hk[None]+1j*wk[None]); Bs=(1/(1+np.exp(-xa))[:,None])*np.ones((1,8))*xa[:,None]
S=np.zeros((T,8),complex); s=np.zeros(8,complex)
for t in range(T): s=Ls[t]*s+Bs[t]; S[t]=s
R['scan_err']=float(np.max(np.abs(scan(Ls,Bs.astype(complex))-S)))
# 7 HT-A invertibility probe: b const (x-indep), Delta=softplus(w x); solve X_t - lam(x) X_{t-1} = b x  for x
def roots_count(K,trials=400,wd=1.0,c=-1.0):
    hk,wk=ht_default(K,F=1 if K<2 else 2,hmax=64); bk=0.3*np.ones(K); xs=np.linspace(-6,6,24001); counts=[];errs=[]
    for _ in range(trials):
        xt=rng.standard_normal(); Xp=rng.standard_normal(K)+1j*rng.standard_normal(K)
        lam_f=lambda x: np.exp(-np.log1p(np.exp(wd*x+c))[...,None]/hk+1j*wk)
        Xt=lam_f(xt)*Xp+bk*xt
        res=np.sum(np.abs(Xt[None]-lam_f(xs)*Xp[None]-bk[None]*xs[:,None])**2,1)
        loc=np.where((res[1:-1]<res[:-2])&(res[1:-1]<res[2:])&(res[1:-1]<1e-6))[0]
        counts.append(len(loc)); xh=xs[loc+1][np.argmin(res[loc+1])] if len(loc) else np.nan; errs.append(abs(xh-xt))
    return float(np.mean(np.array(counts)>1)), float(np.nanmax(errs))
R['hta_nonunique_frac_K1'],R['hta_maxerr_K1']=roots_count(1)
R['hta_nonunique_frac_K4'],R['hta_maxerr_K4']=roots_count(4)
json.dump(R,open('out/e0.json','w'),indent=1); print(json.dumps(R,indent=1))
