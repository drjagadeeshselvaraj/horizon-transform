import numpy as np, json, time, scipy.signal as ss
from lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t0=time.time(); R={}
# ============ T3: multi-scale forecasting (synthetic) ============
HZ=[1,24,168,720]; NF=64
def series(seed,L=70000):
    rng=np.random.default_rng(seed); t=np.arange(L)
    y=np.cos(2*np.pi*t/24+rng.uniform(0,6.28))+.7*np.cos(2*np.pi*t/168+rng.uniform(0,6.28))+.5*np.cos(2*np.pi*t/2000+rng.uniform(0,6.28))
    ar=ss.lfilter([1],[1,-0.98],0.15*rng.standard_normal(L)); y=y+ar+0.3*rng.standard_normal(L); return y
phi,si,sw=0.98,0.15,0.3
ceil={H:float(((si**2)*(1-phi**(2*H))/(1-phi**2)+sw**2)) for H in HZ}   # irreducible variance (innovation + white noise); h=1 adds obs-noise
def fc_eval(mk,seed):
    y=series(seed); b=mk(seed); F=b.stream(y); burn=9000; mx=max(HZ); L=len(y)
    Z=F[burn:L-mx]; Z=(Z-Z.mean(0))/(Z.std(0)+1e-12); Y=np.stack([y[burn+H:L-mx+H] for H in HZ],1); n=len(Z)
    r2=ridge_curves(Z,Y,int(.5*n),int(.15*n),alphas=(1e-8,1e-6,1e-4,1e-2,1)); var=Y[int(.65*n):].var(0)
    return (1-r2)   # normalised MSE
fb={'HT (geometric, default)':lambda s:bank_ht(NF),'HT real poles':lambda s:bank_ht_real(NF),'S4D-Lin':lambda s:bank_s4d_lin(NF),'LRU':lambda s:bank_lru(NF,seed=s),
 'HiPPO-LegT (W=4096)':lambda s:bank_legt(NF,4096),'STFT (W=256)':lambda s:bank_stft(NF,256),'STFT (W=1024)':lambda s:bank_stft(NF,1024),'STFT (W=256, low band)':lambda s:bank_stft(NF,256,fmax_frac=0.25,name='STFT (W=256, low band)'),'STFT (W=1024, low band)':lambda s:bank_stft(NF,1024,fmax_frac=0.0625,name='STFT (W=1024, low band)'),'CWT (causal Morlet)':lambda s:bank_cwt(NF),'Delay line (64 taps)':lambda s:bank_delay(NF)}
FC={}
for k,mk in fb.items():
    m=np.array([fc_eval(mk,s) for s in range(5)]); FC[k]={'mean':m.mean(0).tolist(),'std':m.std(0).tolist()}; print('FC',k,np.round(m.mean(0),3),round(time.time()-t0),flush=True)
var=[float(series(0).var())]; FC['_ceiling_nmse']=[ceil[H]/var[0] for H in HZ]; FC['_hz']=HZ; R['forecast']=FC
# ============ T4: spectral-signature classification (EEG/audio-like, synthetic) ============
fs=128; Lc=1024; FREQS=np.geomspace(2,40,32)
def make_ds(n,rng):
    X=np.empty((n,Lc)); yv=rng.integers(0,3,n); centers=np.array([6.,10.,20.])
    for i in range(n):
        f=centers[yv[i]]*rng.uniform(.85,1.15); t=np.arange(Lc)/fs; am=1+.5*np.sin(2*np.pi*rng.uniform(.1,.5)*t+rng.uniform(0,6))
        burst=np.cos(2*np.pi*f*t+rng.uniform(0,6.28))*am*0.10
        w=rng.standard_normal(Lc); pink=ss.lfilter([1],[1,-0.97],w)*0.25          # 1/f-ish background
        X[i]=burst+pink+0.35*rng.standard_normal(Lc)
    return X,yv
def cwt_freq(freqs,w0=6.0,name='CWT (Morlet, log freqs)'):
    ker=[];Lk=int(6*w0/(2*np.pi*freqs.min())*fs)+8; tau=np.arange(Lk); K=len(freqs); out=np.zeros((Lk,2*K))
    for i,f in enumerate(freqs):
        s=w0/(2*np.pi*f)*fs; m=tau<=6*s; t=tau[m]-3*s; g=np.exp(-t**2/(2*s**2))*np.exp(1j*2*np.pi*f/fs*t); g/=np.sqrt(np.sum(np.abs(g)**2)); out[m,2*i]=g.real; out[m,2*i+1]=g.imag
    return Bank(name,2*K,ker=out)
def ht_freq(freqs,cycles=2.0,name='HT (freq-tuned, constant-Q)'):
    w=2*np.pi*freqs/fs; h=cycles/(w/(2*np.pi)*1.0)*1.0; h=cycles*fs/(2*np.pi*freqs)*2*np.pi/ (2*np.pi) ; h=cycles/(freqs/fs)/ (2*np.pi) *np.pi  # ~ cycles/(2 f) seconds*fs
    h=cycles*fs/(freqs*2); return ht_bank(h,w,name)
def stft_lin(W=128,fmax=40.0):
    nb=32; bins=np.unique(np.round(np.linspace(2/fs*W,fmax/fs*W,nb)).astype(int)); nb=len(bins); tau=np.arange(W); w=np.hanning(W+2)[1:-1]; ker=np.empty((W,2*nb))
    for i,j in enumerate(bins): ker[:,2*i]=w*np.cos(2*np.pi*j*tau/W); ker[:,2*i+1]=-w*np.sin(2*np.pi*j*tau/W)
    return Bank(f'STFT (W={W}, linear bins)',2*nb,ker=ker)
cb={'HT (freq-tuned, constant-Q)':lambda s:ht_freq(FREQS),'HT (geometric, default)':lambda s:bank_ht(64,hmax=1024),'CWT (Morlet, log freqs)':lambda s:cwt_freq(FREQS),
    'STFT (W=128, linear bins)':lambda s:stft_lin(128),'STFT (W=256, linear bins)':lambda s:stft_lin(256),'LRU':lambda s:bank_lru(64,seed=s,rmin=.9,rmax=.99,phimax=np.pi/2)}
def feats(b,X):
    out=[]
    for x in X:
        F=b.stream(x)[300:]; P=F[:,0::2]**2+F[:,1::2]**2; out.append(np.log(P.mean(0)+1e-9))
    return np.array(out)
CL={}
for k,mk in cb.items():
    acc=[]
    for s in range(5):
        rng=np.random.default_rng(300+s); Xtr,ytr=make_ds(450,rng); Xte,yte=make_ds(450,rng); b=mk(s)
        Ftr,Fte=feats(b,Xtr),feats(b,Xte); sc=StandardScaler().fit(Ftr)
        clf=LogisticRegression(C=1.0,max_iter=2000).fit(sc.transform(Ftr),ytr); acc.append(clf.score(sc.transform(Fte),yte))
    CL[k]={'mean':float(np.mean(acc)),'std':float(np.std(acc))}; print('CL',k,CL[k],round(time.time()-t0),flush=True)
R['classify']=CL; json.dump(R,open('out/e3.json','w'),default=float); print('total',time.time()-t0)
