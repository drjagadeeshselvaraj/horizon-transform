"""(1) numerical precision of the exact inverse, (2) erasure robustness of least-squares decoding, (3) effective channel count and LS symbol (analytic)."""
import numpy as np, json, time
import e1
from e1 import N, sig, KINDS, A_ht, A_stft, A_udwt
from lib import ht_poles, ht_gain, ht_default
t0=time.time(); R={}
# ---- (1) precision of the exact inverse (single channel, default frequency, unit-power gain)
rng=np.random.default_rng(0); T=6000; hs=np.geomspace(1,4096,13); pr={'float32':[], 'float64':[], 'bound':[]}
for h in hs:
    w=min(np.pi,np.pi/(2*h)); lam=ht_poles([h],[w])[0]; b=float(ht_gain(lam)); x=rng.standard_normal(T)
    for name,cd,rd in (('float32',np.complex64,np.float32),('float64',np.complex128,np.float64)):
        l=cd(lam); bb=rd(b); xx=x.astype(rd); X=np.zeros(T,cd); s=cd(0)
        for t in range(T): s=l*s+bb*xx[t]; X[t]=s
        Xp=np.concatenate([[cd(0)],X[:-1]]); xh=np.real(np.conj(bb)*(X-l*Xp))/(bb*bb); pr[name].append(float(np.max(np.abs(xh-xx))))
    pr['bound'].append(float(np.finfo(np.float32).eps*np.sqrt(h/2)))
R['precision']={'h':hs.tolist(),**pr}; print('precision',round(time.time()-t0),flush=True)
# ---- (2) erasure robustness, rho=8, per-channel noise at 30 dB, GLS on remaining rows
o1=json.load(open('out/e1.json')); rho=8; K=rho//2; SNR=1000.0; PS=[0,0.1,0.2,0.3,0.4,0.5]
mats={}
h,w=ht_default(K,F=2,hmax=N); A,ch,_,_=A_ht(h,w); mats['HT default']=(A,ch)
th=o1['ht_tuned'][str(rho)]; A,ch,_,_=A_ht(np.array(th[0]),np.array(th[1])); mats['HT tuned']=(A,ch)
mats['STFT (W=128)']=tuple(A_stft(rho)); mats['Undecimated wavelet (db2)']=tuple(A_udwt(rho-1))
res={k:{str(p):[] for p in PS} for k in mats}; rng=np.random.default_rng(5)
xs=[sig(k,np.random.default_rng(7000+i)) for i in range(5) for k in KINDS]
for nm,(A,ch) in mats.items():
    for x in xs:
        y=A@x; ids=np.unique(ch); pw={c:np.mean(y[ch==c]**2) for c in ids}; sig2=np.array([pw[c] for c in ch])/SNR+1e-30
        for p in PS:
            for rep in range(2):
                keep=rng.random(len(y))>=p; As=A[keep]; Wv=1/sig2[keep]; G=(As*Wv[:,None]).T@As
                try: err=np.trace(np.linalg.inv(G+1e-10*np.trace(G)/N*np.eye(N)))
                except np.linalg.LinAlgError: err=np.inf
                res[nm][str(p)].append(10*np.log10(np.sum(x**2)/err))
    print('erasure',nm,round(time.time()-t0),flush=True)
R['erasure']={nm:{p:[float(np.mean(v)),float(np.std(v))] for p,v in d.items()} for nm,d in res.items()}; R['erasure_ps']=PS
# ---- (3) analytic: effective channel count and noise gain of the pooled inverse; LS symbol
def pooled_gain(h,w):
    lam=ht_poles(h,w); b=ht_gain(lam); return 0.5*np.sum(b**2*(1+np.abs(lam)**2))/np.sum(b**2)**2, float(np.sum(b**2))
cnt={}
for F in (1,2,4):
    ms=np.arange(1,17); g=[];sb=[]
    for M in ms:
        hh=np.geomspace(1,4096,M) if M>1 else np.array([1.0]); hh2=np.repeat(hh,F); ww=np.minimum(np.pi,np.pi*(np.tile(np.arange(F),M)+1)/(2*hh2)); gg,ss=pooled_gain(hh2,ww); g.append(float(gg)); sb.append(ss)
    cnt[str(F)]={'M':ms.tolist(),'gain':g,'sumb2':sb}
R['pooled']=cnt
th_=np.linspace(0,np.pi,2000)
def symbol(h,w):
    lam=ht_poles(h,w); b=ht_gain(lam); s=np.zeros_like(th_)
    for l,bb in zip(lam,b): s+=0.5*(np.abs(bb/(1-l*np.exp(-1j*th_)))**2+np.abs(bb/(1-l*np.exp(1j*th_)))**2)
    return s
sym={}; 
for nm,(h,w) in {'HT default, ρ = 8':ht_default(4,F=2,hmax=N),'HT tuned, ρ = 8':(np.array(o1['ht_tuned']['8'][0]),np.array(o1['ht_tuned']['8'][1])),'HT default, ρ = 32':ht_default(16,F=2,hmax=N)}.items():
    s=symbol(h,w); sym[nm]={'theta':th_.tolist(),'s':s.tolist(),'gain_symbol':float(np.mean(1/s)),'sumb2':float(np.sum(ht_gain(ht_poles(h,w))**2))}
    K_=len(h); A,ch,_,_=A_ht(h,w); G=A.T@A; sym[nm]['gain_exact']=float(np.trace(np.linalg.inv(G))/N)
R['symbol']=sym; json.dump(R,open('out/e7.json','w'),default=float)
print({k:(round(v['gain_symbol'],3),round(v['gain_exact'],3)) for k,v in sym.items()})
for nm,d in R['erasure'].items(): print(nm,[round(d[str(p)][0],1) for p in PS])
print('prec float32',[f'{v:.1e}' for v in pr['float32']][::3],'float64',[f'{v:.1e}' for v in pr['float64']][::3])
