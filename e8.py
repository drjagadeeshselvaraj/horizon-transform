"""Experiments for Chapter 5 (Properties): cost scaling, group delay, stability, frame bounds, gradients, dilation equivariance."""
import numpy as np, json, time, scipy.signal as ss
from lib import *
R={}; t0=time.time(); rng=np.random.default_rng(0)
def med(f,reps=5):
    f(); ts=[]
    for _ in range(reps): t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))
# ---- (a) cost scaling of the recurrence and of FFT convolution
def mk(K): 
    h,w=ht_default(K,F=2,hmax=4096.0); lam=ht_poles(h,w); return lam,ht_gain(lam)
def rec(x,lam,b):
    return [ss.lfilter([bb],[1,-l],x.astype(complex)) for l,bb in zip(lam,b)]
def fftc(x,lam,b):
    T=len(x); n=1<<int(np.ceil(np.log2(2*T))); Fx=np.fft.fft(x,n); tau=np.arange(T)
    return [np.fft.ifft(Fx*np.fft.fft(bb*np.exp(tau*np.log(l)),n))[:T] for l,bb in zip(lam,b)]
T=16384; x=rng.standard_normal(T); Ks=[2,4,8,16,32,64,128,256]; cK={'rec':[], 'fft':[]}
for K in Ks:
    lam,b=mk(K); cK['rec'].append(med(lambda:rec(x,lam,b),3)); cK['fft'].append(med(lambda:fftc(x,lam,b),3))
Ts=[2**p for p in range(10,18)]; cT={'rec':[], 'fft':[]}; lam,b=mk(32)
for T_ in Ts:
    xx=rng.standard_normal(T_); cT['rec'].append(med(lambda:rec(xx,lam,b),3)); cT['fft'].append(med(lambda:fftc(xx,lam,b),3))
R['cost']={'T':T,'K':Ks,'vsK':cK,'Ts':Ts,'vsT':cT,'state_bytes_per_channel':16}
def slope(xs,ys): return float(np.polyfit(np.log(xs),np.log(ys),1)[0])
R['cost']['slopes']={'rec_K':slope(Ks[2:],cK['rec'][2:]),'fft_K':slope(Ks[2:],cK['fft'][2:]),'rec_T':slope(Ts,cT['rec']),'fft_T':slope(Ts,cT['fft'])}; print('cost',R['cost']['slopes'],round(time.time()-t0),flush=True)
# ---- (b) group delay at the channel frequency, and burst latency
gd=[]
for h in [2,4,8,16,32,64,128,256,512,1024]:
    w=1.0; lam=np.exp(-1/h+1j*w); th=np.linspace(w-0.2,w+0.2,400001); H=1/(1-lam*np.exp(-1j*th)); ph=np.unwrap(np.angle(H)); g=-np.gradient(ph,th)[len(th)//2]
    gd.append((h,float(g),float(abs(lam)/(1-abs(lam)))))
R['groupdelay']=gd
tt=np.arange(0,2400); P=16; burst=np.where((tt>=200)&(tt<1800),np.sin(2*np.pi*tt/P),0.0); lat={}
for h in (8,32,128):
    lam=np.exp(-1/h+1j*2*np.pi/P); b=np.sqrt(1-abs(lam)**2); X=ss.lfilter([b],[1,-lam],burst.astype(complex)); m=np.abs(X); peak=m[:1800].max(); t63=int(np.argmax(m>=(1-1/np.e)*peak))-200
    lat[h]={'t63':t63,'curve':(m/peak).reshape(-1,16).max(1).tolist()}
R['burst']={'t':(tt.reshape(-1,16)[:,8]).tolist(),'x':burst[::16].tolist(),'lat':lat}
# ---- (c) real-input L-infinity gain relative to the bound
ws=np.linspace(0,np.pi,61); ratio={}
for r in (0.9,0.99,0.999):
    Tm=int(30/(1-r)); tau=np.arange(Tm); rr=r**tau; th=np.linspace(0,np.pi,60); out=[]
    for w in ws: out.append(float(np.max((rr[None]*np.abs(np.cos(w*tau[None]-th[:,None]))).sum(1))*(1-r)))
    ratio[str(r)]=out
R['bibo']={'w':ws.tolist(),'ratio':ratio}
# ---- (d) HT-A: coupled vs uncoupled gate
Tn=40000; flag=np.zeros(Tn); flag[::100]=1; D=flag.copy(); h=300.0; xx=np.ones(Tn)
def sim(bg):
    X=np.zeros(Tn); s=0.0
    for t in range(Tn): s=np.exp(-D[t]/h)*s+bg[t]*xx[t]; X[t]=s
    return X
Xc=sim(D*1.0); Xu=sim(np.ones(Tn)); bound=1.0*h*np.exp(D.max()/h)
trials=[]
for _ in range(60):
    Dd=rng.choice([0,0,0,0.05,0.5,1.5],size=3000); beta=rng.uniform(0.2,2); hh=rng.choice([10,50,300]); xr=rng.uniform(-1,1,3000); X=0; mx=0
    for t in range(3000): X=np.exp(-Dd[t]/hh)*X+beta*Dd[t]*xr[t]; mx=max(mx,abs(X))
    trials.append(mx/(beta*hh*np.exp(Dd.max()/hh)))
R['coupled']={'t':list(range(Tn))[::5],'Xc':Xc[::5].tolist(),'Xu':Xu[::5].tolist(),'bound':float(bound),'max_ratio_random':float(max(trials)),'final_u':float(Xu[-1]),'final_c':float(Xc[-1])}
# ---- (e) frame bounds against channel count (symbol of the least-squares frame operator, signal length 256)
th=np.linspace(0,np.pi,4000)
def symbol(h,w):
    lam=ht_poles(h,w); b=ht_gain(lam); s=np.zeros_like(th)
    for l,bb in zip(lam,b): s+=0.5*(np.abs(bb/(1-l*np.exp(-1j*th)))**2+np.abs(bb/(1-l*np.exp(1j*th)))**2)
    return s
fb=[]
for K in [1,2,3,4,6,8,12,16,24,32,48,64]:
    h,w=ht_default(K,F=min(2,K),hmax=256.0); s=symbol(h,w); fb.append((K,float(s.min()),float(s.max()),float(s.max()/s.min()),float(np.mean(1/s)*K)))
o1=json.load(open('out/e1.json')); tb=[]
for rho in (2,4,8,16):
    k=rho//2; hw=o1['ht_tuned'][str(rho)]; s=symbol(np.array(hw[0]),np.array(hw[1])); tb.append((k,float(s.min()),float(s.max()),float(s.max()/s.min()),float(np.mean(1/s)*k)))
R['frame']={'default':fb,'tuned':tb}
# ---- (f) gradient check and gradient decay
gc=[]
for h in [4,16,64,256]:
    w=0.7; xg=rng.standard_normal(600)
    def XT(hh,b0):
        lam=np.exp(-1/hh+1j*w); return ss.lfilter([b0],[1,-lam],xg.astype(complex))[-1]
    b0=float(ht_gain(np.exp(-1/h+1j*w))); lam=np.exp(-1/h+1j*w); X=ss.lfilter([b0],[1,-lam],xg.astype(complex)); Xp=np.concatenate([[0],X[:-1]]); Dg=ss.lfilter([1],[1,-lam],Xp.astype(complex))
    ana=Dg[-1]*lam/h**2; eps=1e-5*h; fd=(XT(h+eps,b0)-XT(h-eps,b0))/(2*eps); gc.append((h,float(abs(ana-fd)/abs(fd)),float(abs(fd))))
R['grad']=gc
# ---- (g) dilation equivariance of the default bank (r = 2)
M=12; F=2; hh=np.repeat(2.0**np.arange(M),F); jj=np.tile(np.arange(F),M); ww=np.minimum(np.pi,np.pi*(jj+1)/(2*hh)); lam=ht_poles(hh,ww); b=ht_gain(lam)
def burst_sig(P,D,T=40000,t0=8000):
    t=np.arange(T); win=np.where((t>=t0)&(t<t0+D),0.5*(1-np.cos(2*np.pi*(t-t0)/D)),0.0); return np.sin(2*np.pi*t/P)*win
def peaks(x): return np.array([np.abs(ss.lfilter([bb],[1,-l],x.astype(complex))).max() for l,bb in zip(lam,b)])
m1=peaks(burst_sig(8,200)); m2=peaks(burst_sig(16,400)); m4=peaks(burst_sig(32,800))
R['dilation']={'m1':m1.tolist(),'m2':m2.tolist(),'m4':m4.tolist(),'F':F}
k1=m1.reshape(M,F); k2=m2.reshape(M,F); ok=np.arange(1,M-1); rat=(k2[ok+1]/k1[ok]).ravel(); R['dilation']['ratio_median']=float(np.median(rat[np.isfinite(rat)&(k1[ok].ravel()>0.05*k1.max())])); 
sel=k1[ok].ravel()>0.2*k1.max(); R['dilation']['ratio_sel']=[float(np.median((k2[ok+1]/k1[ok]).ravel()[sel])),float(np.min((k2[ok+1]/k1[ok]).ravel()[sel])),float(np.max((k2[ok+1]/k1[ok]).ravel()[sel]))]
json.dump(R,open('out/e8.json','w'),default=float)
print('groupdelay',[(h,round(g,2),round(f,2)) for h,g,f in gd][::3]); print('burst t63',{h:v['t63'] for h,v in lat.items()})
print('bibo ratio at w=0, mid, pi (r=.99)',ratio['0.99'][0],ratio['0.99'][30],ratio['0.99'][-1])
print('coupled final',Xc[-1],'bound',bound,'uncoupled final',Xu[-1],'random max ratio',max(trials))
print('frame default',[(k,round(mn,3),round(mx,1),round(c,0),round(g,2)) for k,mn,mx,c,g in fb][::3]); print('frame tuned',[(k,round(mn,3),round(mx,1),round(c,0),round(g,2)) for k,mn,mx,c,g in tb])
print('grad rel err',[(h,f'{e:.1e}') for h,e,_ in gc]); print('dilation ratio',R['dilation']['ratio_sel'],'time',round(time.time()-t0))
