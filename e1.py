import numpy as np, json, time
from lib import *
N=256; RHOS=[2,4,8,16]; rng0=np.random.default_rng(7)
# ---------------- signals
def sig(kind,rng):
    t=np.arange(N)
    if kind=='AR1':
        e=rng.standard_normal(N+200); x=np.zeros(N+200)
        for i in range(1,N+200): x[i]=0.95*x[i-1]+e[i]
        x=x[200:]
    elif kind=='Tones+chirp+impulses':
        x=sum(np.cos(2*np.pi*rng.uniform(.02,.4)*t+rng.uniform(0,6.28)) for _ in range(3))
        f0,f1=rng.uniform(.02,.1),rng.uniform(.2,.4); x=x+np.cos(2*np.pi*(f0*t+(f1-f0)*t**2/(2*N))); 
        for p in rng.integers(0,N,2): x[p]+=6*rng.choice([-1,1])
    elif kind=='Piecewise-constant':
        x=np.zeros(N); pos=np.sort(rng.integers(5,N-5,6)); lev=rng.standard_normal(7); idx=np.searchsorted(pos,t); x=lev[idx]
    else: x=rng.standard_normal(N)
    return x/np.sqrt(np.mean(x**2))
KINDS=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']
# ---------------- transforms: return A (rows x N), channel id per row
def toep(k): # lower-triangular Toeplitz N x N from causal kernel k[:N]
    idx=np.arange(N)[:,None]-np.arange(N)[None]; M=np.where(idx>=0,k[np.clip(idx,0,len(k)-1)],0.0); return M
def A_ht(h,w):
    lam=ht_poles(h,w); b=ht_gain(lam); rows=[];ch=[]
    for i,(l,bb) in enumerate(zip(lam,b)):
        k=bb*l**np.arange(N); rows+= [toep(k.real),toep(k.imag)]; ch+=[i]*(2*N)
    return np.vstack(rows),np.array(ch),lam,b
def A_stft(rho,W=128):
    H=W//rho; M=N//H; w=np.sin(np.pi*(np.arange(W)+.5)/W); rows=[];ch=[]
    for m in range(M):
        pos=(m*H+np.arange(W))%N
        for j in range(W//2+1):
            c=np.zeros(N); s=np.zeros(N); np.add.at(c,pos,w*np.cos(2*np.pi*j*np.arange(W)/W)); np.add.at(s,pos,-w*np.sin(2*np.pi*j*np.arange(W)/W))
            rows.append(c); ch.append(j)
            if 0<j<W//2: rows.append(s); ch.append(j)
    return np.array(rows),np.array(ch)
def A_cwt(K,w0=6.0):
    sig_=np.geomspace(2,N/8,K) if K>1 else np.array([6.0]); rows=[];ch=[]
    for i,s in enumerate(sig_):
        t=np.arange(N); t=np.minimum(t,N-t); g=np.exp(-t**2/(2*s**2))*np.exp(1j*w0*t/s*np.sign(np.arange(N)-N/2+.5)*-1)
        g=np.exp(-t**2/(2*s**2)); sgn=np.where(np.arange(N)<N/2,1,-1); g=g*np.exp(1j*w0*sgn*t/s); g=g/np.sqrt(np.sum(np.abs(g)**2))
        C=np.array([np.roll(g,m) for m in range(N)])
        rows+=[C.real,C.imag]; ch+=[i]*(2*N)
    t=np.arange(N); t=np.minimum(t,N-t); g=np.exp(-t**2/(2*(1.5*sig_.max())**2)); g/=np.sqrt(np.sum(g**2))
    rows.append(np.array([np.roll(g,m) for m in range(N)])); ch+=[K]*N      # extra real scaling-function channel (rho+1)
    return np.vstack(rows),np.array(ch)
def A_udwt(J):  # stationary db2 transform, circular
    h=np.array([0.4829629131445341,0.8365163037378079,0.2241438680420134,-0.1294095225512229])/np.sqrt(2); g=h[::-1]*np.array([1,-1,1,-1])
    a=np.eye(N); rows=[];ch=[]
    def circ(M,f,step): return sum(f[i]*np.roll(M,i*step,axis=1) for i in range(4))  # filter along input index
    # operate on operator matrices: rows are coefficient functionals
    for j in range(J):
        d=circ(a,g,2**j); rows.append(d); ch+=[j]*N; a=circ(a,h,2**j)
    rows.append(a); ch+=[J]*N; return np.vstack(rows)*np.sqrt(2),np.array(ch)
def ls_dec(A):
    G=A.T@A; w,V=np.linalg.eigh(G); wi=1/np.maximum(w,1e-12*w.max()); return (V*wi)@V.T@A.T
# ---------------- HT paper inverse (pooled local LS) as a linear operator on stacked [Re,Im] rows
def Q_ht(lam,b,K):
    Q=np.zeros((N,2*K*N)); 
    for k in range(K):
        re0=2*k*N; im0=(2*k+1)*N; sc=b[k]/np.sum(b**2)
        for t in range(N):
            Q[t,re0+t]+=sc; 
            if t>0: Q[t,re0+t-1]+=-sc*lam[k].real; Q[t,im0+t-1]+=sc*lam[k].imag
    return Q
# ---------------- evaluation
def analytic_snr(A,ch,P,xs,snr_db,model):
    snr=10**(snr_db/10); colsq=(P**2).sum(0); out=[]
    ids=np.unique(ch)
    for x in xs:
        y=A@x
        if model=='global': err=np.mean(y**2)/snr*colsq.sum()
        else: err=sum(np.mean(y[ch==c]**2)/snr*colsq[ch==c].sum() for c in ids)
        out.append(10*np.log10(np.sum(x**2)/err))
    return float(np.mean(out))
def quant_snr(A,ch,P,xs,bits,rng=None):
    out=[]; ids=np.unique(ch); lv=2**bits
    for x in xs:
        y=A@x; yq=np.empty_like(y)
        for c in ids:
            m=ch==c; s=np.sqrt(np.mean(y[m]**2))+1e-12; R=4*s; step=2*R/lv
            yq[m]=np.clip((np.floor(np.clip(y[m],-R,R-1e-9)/step)+.5)*step,-R,R)
        out.append(10*np.log10(np.sum(x**2)/np.sum((P@yq-x)**2)))
    return float(np.mean(out))
# ---------------- HT tuning (analytic, normal-equation trick)
def ht_obj(theta,K,xs,snr_db=20):
    h=np.exp(theta[:K]); w=theta[K:]; lam=ht_poles(h,w); b=ht_gain(lam); G=np.zeros((N,N)); Gc=[]
    for l,bb in zip(lam,b):
        k=bb*l**np.arange(N); T1=toep(k.real); T2=toep(k.imag); g=T1.T@T1+T2.T@T2; Gc.append((g,T1,T2)); G+=g
    wv,V=np.linalg.eigh(G); wi=1/np.maximum(wv,1e-12*wv.max()); H=(V*wi**2)@V.T
    q=np.array([np.sum(g*H) for g,_,_ in Gc]); tot=[]
    for x in xs:
        Pc=np.array([(np.sum((T1@x)**2)+np.sum((T2@x)**2))/(2*N) for _,T1,T2 in Gc])
        tot.append(10*np.log10(np.sum(x**2)*10**(snr_db/10)/np.sum(Pc*q)))
    return np.mean(tot)
def tune_ht(K,xs,iters=140,seed=0):
    rng=np.random.default_rng(seed); h,w=ht_default(K,F=min(2,K),hmax=N)
    th=np.concatenate([np.log(h),w]); best=ht_obj(th,K,xs)
    for it in range(iters):
        sc=0.5*(1-it/iters)+0.05; c=th+sc*rng.standard_normal(2*K)*(rng.random(2*K)<0.5)
        c[:K]=np.clip(c[:K],np.log(0.5),np.log(4*N)); c[K:]=np.clip(c[K:],0,np.pi)
        v=ht_obj(c,K,xs)
        if v>best: best,th=v,c
    return th,best
if __name__=='__main__':
    t0=time.time(); out={'N':N,'rhos':RHOS}
    trainx=[sig(k,np.random.default_rng(100+i)) for i in range(6) for k in KINDS[:3]]
    testx={k:[sig(k,np.random.default_rng(5000+i)) for i in range(20)] for k in KINDS}
    cfg={}
    for rho in RHOS:
        K=rho//2
        h,w=ht_default(K,F=min(2,K),hmax=N); th,best=tune_ht(K,trainx,seed=rho); print('tuned rho',rho,best,time.time()-t0,flush=True)
        cfg[('HT (default)',rho)]=ht_default(K,F=min(2,K),hmax=N); cfg[('HT (tuned)',rho)]=(np.exp(th[:K]),th[K:])
    out['ht_tuned']={str(r):[list(map(float,cfg[('HT (tuned)',r)][0])),list(map(float,cfg[('HT (tuned)',r)][1]))] for r in RHOS}
    res={}   # res[name][rho]= dict
    for rho in RHOS:
        K=rho//2; T={}
        for nm in ('HT (default)','HT (tuned)'):
            h,w=cfg[(nm,rho)]; A,ch,lam,b=A_ht(h,w); T[nm+' - LS inverse']=(A,ch,ls_dec(A)); T[nm+' - paper inverse']=(A,ch,Q_ht(lam,b,K))
        A,ch=A_stft(rho); T['STFT (W=128)']=(A,ch,ls_dec(A))
        A,ch=A_cwt(K); T['CWT (Morlet)']=(A,ch,ls_dec(A))
        if rho<=8: A,ch=A_udwt(rho-1); T['Undecimated wavelet (db2)']=(A,ch,ls_dec(A))
        for nm,(A,ch,P) in T.items():
            d={}
            for kind in KINDS:
                xs=testx[kind]; d[kind]={'global20':analytic_snr(A,ch,P,xs,20,'global'),'chan20':analytic_snr(A,ch,P,xs,20,'chan'),'chan10':analytic_snr(A,ch,P,xs,10,'chan'),
                   'q':{b_:quant_snr(A,ch,P,xs[:8],b_) for b_ in (1,2,4,8)}}
            d['cond']=float(np.linalg.cond(A)) if A.shape[0]*A.shape[1]<4e6 else None
            res.setdefault(nm,{})[rho]=d
        print('rho',rho,'done',time.time()-t0,flush=True)
    # critically sampled references (rho=1)
    from scipy.fft import dct
    Dm=dct(np.eye(N),norm='ortho',axis=0); ref={}
    for nm,(A,ch) in {'DCT (orthonormal, rho=1)':(Dm,np.arange(N)//16),'PCM (identity, rho=1)':(np.eye(N),np.zeros(N,int))}.items():
        P=A.T; ref[nm]={kind:{'q':{b_:quant_snr(A,ch,P,testx[kind][:8],b_) for b_ in (1,2,4,8,12)}} for kind in KINDS}
    out['res']={nm:{str(r):v for r,v in d.items()} for nm,d in res.items()}; out['ref']=ref
    json.dump(out,open('out/e1.json','w'),indent=1,default=float); print('total',time.time()-t0)
