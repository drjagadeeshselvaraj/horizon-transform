"""Relations of HT to other transforms: DFT limit, recursive STFT equivalence, quality factors, gammatone cascade, S4D-Lin mapping, mixing matrices."""
import numpy as np, json, scipy.signal as ss
from scipy.special import comb
from lib import *
rng=np.random.default_rng(0); R={}
# ---- (1) DFT limit
T=512; x=rng.standard_normal(T); ks=np.arange(1,60); w=2*np.pi*ks/T; dft=np.fft.fft(x)[ks]; hs=np.geomspace(10,1e8,18); err=[]
for h in hs:
    lam=np.exp(-1/h+1j*w); X=np.array([ss.lfilter([1.0],[1,-l],x.astype(complex))[-1] for l in lam]); pred=np.exp(1j*w*(T-1))*dft
    err.append(float(np.linalg.norm(X-pred)/np.linalg.norm(pred)))
R['dft']={'h':hs.tolist(),'err':err,'T':T}
# ---- (2) recursive STFT equivalence (exponential window, modulation)
Tn=3000; x=rng.standard_normal(Tn); h=30.0; wd=0.7; lam=np.exp(-1/h+1j*wd); b=float(ht_gain(lam)); X=ss.lfilter([b],[1,-lam],x.astype(complex))
tau=np.arange(0,int(60*h)); kern=b*np.exp(-tau/h)*np.exp(1j*wd*tau); Xd=np.convolve(x,kern)[:Tn]
tt=np.arange(Tn); demod=np.exp(-1j*wd*tt)*x; low=np.convolve(demod,b*np.exp(-tau/h))[:Tn]*np.exp(1j*wd*tt)
R['stft_equiv']={'direct_vs_ht':float(np.max(np.abs(X-Xd))/np.max(np.abs(X))),'mod_demod_vs_ht':float(np.max(np.abs(X-low))/np.max(np.abs(X)))}
# ---- (3) spectrogram comparison
fs=128; N=1024; t=np.arange(N)/fs; xs=np.sin(2*np.pi*(3*t+ (30-3)/(2*8)*t**2))
xs=xs+1.5*np.sin(2*np.pi*12*t)*np.where((t>=2.5)&(t<4),0.5*(1-np.cos(2*np.pi*(t-2.5)/1.5)),0); xs[int(1.5*fs)]+=6; xs[int(6.2*fs)]-=6; xs=xs+0.1*rng.standard_normal(N)
freqs=np.geomspace(2,45,48); om=2*np.pi*freqs/fs
Hm=np.zeros((N,48))
for i,w_ in enumerate(om):
    h_=2*np.pi/w_; lam=np.exp(-1/h_+1j*w_); Hm[:,i]=np.abs(ss.lfilter([np.sqrt(1-abs(lam)**2)],[1,-lam],xs.astype(complex)))
def cwtk(f,w0=6.0):
    s=w0/(2*np.pi*f)*fs; L=int(6*s)+2; tau=np.arange(L); k=np.zeros(L,complex); m=tau<=6*s; tt_=tau[m]-3*s; k[m]=np.exp(-tt_**2/(2*s**2))*np.exp(1j*2*np.pi*f/fs*tt_); return k/np.sqrt(np.sum(np.abs(k)**2))
Cm=np.zeros((N,48))
for i,f in enumerate(freqs): Cm[:,i]=np.abs(np.convolve(xs,cwtk(f))[:N])
W=128; win=np.hanning(W+2)[1:-1]; fb=np.arange(2,46); Sm=np.zeros((N,len(fb)))
for j,f in enumerate(fb):
    k=win*np.exp(-2j*np.pi*f/fs*np.arange(W))[::-1]*np.sqrt(2/W); Sm[:,j]=np.abs(np.convolve(xs,k[::-1])[:N])
R['spec']={'fs':fs,'t':t[::4].tolist(),'x':xs[::4].tolist(),'freqs':freqs.tolist(),'ht':Hm[::4].tolist(),'cwt':Cm[::4].tolist(),'stft_f':fb.tolist(),'stft':Sm[::4].tolist()}
# ---- (4) quality factors
def hw(lam): return -1/np.log(np.abs(lam)),np.abs(np.angle(lam))
Q={}
h1,w1=ht_default(32,F=2,hmax=4096.0); Q['HT default']=[w1.tolist(),(w1*h1/2).tolist()]
f=np.geomspace(2,40,32); wq=2*np.pi*f/128; hq=2*np.pi/wq; Q['HT constant-Q']=[wq.tolist(),(wq*hq/2).tolist()]
lamS=bank_s4d_lin(64).lam; hS,wS=hw(lamS); m=wS>0; Q['S4D-Lin']=[wS[m].tolist(),(wS[m]*hS[m]/2).tolist()]
lamL=bank_lru(64,seed=0).lam; hL,wL=hw(lamL); Q['LRU ring']=[wL.tolist(),(wL*hL/2).tolist()]
R['Q']=Q; R['Q_stft_slope']=float(128/(2*np.pi*1.44))
# ---- (5) cascade -> gammatone
h=12.0; wd=0.5; lam=np.exp(-1/h+1j*wd); b=float(ht_gain(lam)); imp=np.zeros(300); imp[0]=1; casc={}
for n in (1,2,3,4):
    y=imp.astype(complex)
    for _ in range(n): y=ss.lfilter([b],[1,-lam],y)
    tau=np.arange(300); th=b**n*comb(tau+n-1,n-1)*lam**tau; casc[n]={'err':float(np.max(np.abs(y-th))/np.max(np.abs(th))),'re':np.real(y).tolist()}
R['cascade']=casc
# ---- (6) S4D-Lin mapping and LRU horizon density
dts=np.geomspace(1e-3,1e-1,4); S4={'dt':dts.tolist(),'h':[],'err_h':0.0,'err_w':0.0}
for dt in dts:
    A=-0.5+1j*np.pi*np.arange(8); lam=np.exp(dt*A); h_,w_=hw(lam); S4['h'].append(float(h_[0])); S4['err_h']=max(S4['err_h'],float(np.max(np.abs(h_-2/dt)/(2/dt)))); S4['err_w']=max(S4['err_w'],float(np.max(np.abs(w_-np.pi*np.arange(8)*dt))))
R['s4d']=S4
rr=np.sqrt(np.random.default_rng(1).uniform(0.9**2,0.999**2,100000)); hL=-1/np.log(rr)
R['lru_density']={'lt30':float(np.mean(hL<30)),'30_300':float(np.mean((hL>=30)&(hL<300))),'gt300':float(np.mean(hL>=300))}
hg=np.geomspace(10,1000,100000); R['geo_density_10_1000']={'lt30':float(np.mean(hg<30)),'30_300':float(np.mean((hg>=30)&(hg<300))),'gt300':float(np.mean(hg>=300))}
hh=np.geomspace(1,4096,32); R['ring_h_range']=[float(hL.min()),float(hL.max())]
# ---- (7) mixing matrices
Tm=96; ti=np.arange(Tm)[:,None]; si=np.arange(Tm)[None,:]; mask=(ti>=si)
A1=np.where(mask,np.exp(-(ti-si)/16.0),0.0)
A2=np.zeros((Tm,Tm))
for k in range(1,7): A2+=np.where(mask,np.sqrt(2/2.0**k)*np.exp(-(ti-si)/2.0**k),0.0)/6
D=(np.random.default_rng(2).random(Tm)<0.12).astype(float); S=np.cumsum(D); A3=np.where(mask,D[None,:]*np.exp(-(S[:,None]-S[None,:])/16.0),0.0)
q=np.random.default_rng(3).standard_normal((Tm,8)); kk=np.random.default_rng(4).standard_normal((Tm,8)); sc=q@kk.T/np.sqrt(8); sc=np.where(mask,sc,-1e9); sc-=sc.max(1,keepdims=True); A4=np.exp(sc); A4/=A4.sum(1,keepdims=True)
R['mix']={'ht':A1.tolist(),'multi':A2.tolist(),'hta':A3.tolist(),'attn':A4.tolist()}
json.dump(R,open('out/e9.json','w'),default=float)
print('dft err',[f'{e:.1e}' for e in err][::4]); print('stft equiv',R['stft_equiv']); print('cascade err',{n:f"{v['err']:.1e}" for n,v in casc.items()}); print('s4d',S4['err_h'],S4['err_w'],[round(h,1) for h in S4['h']]); print('lru density',R['lru_density'],'geo',R['geo_density_10_1000'])
