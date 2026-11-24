import numpy as np, json, time
from lib import *
t0=time.time(); SEEDS=range(5)
def ev(mk, seeds=SEEDS):
    mcs=[];sf=[];cur=[];cm=[]
    for s in seeds:
        b=mk(s); m,c=memory_capacity(b,s,return_curve=True); f,cf=scale_free_recall(b,s,return_curve=True); mcs.append(m); sf.append(f); cur.append(cf); cm.append(c)
    return dict(mc=[float(np.mean(mcs)),float(np.std(mcs))],sfr=[float(np.mean(sf)),float(np.std(sf))],sfr_curve=np.mean(cur,0).tolist(),mc_curve=np.mean(cm,0).tolist())
# ---- free-pole tuning (hill-climb) on proxy objective
def proxy(h,w,seed,real=False):
    b=ht_bank(h,w,real=real); return scale_free_recall(b,seed,L=24000,burn=6000,ds=SF_D[SF_D<=2048])
def tune(h0,w0,iters=120,seed=0,fix_w=False):
    rng=np.random.default_rng(seed); K=len(h0); th=np.concatenate([np.log(h0),w0]); 
    best=np.mean([proxy(h0,w0,s) for s in (50,51)])
    for it in range(iters):
        sc=0.4*(1-it/iters)+0.03; c=th+sc*rng.standard_normal(2*K)*(rng.random(2*K)<0.3)
        c[:K]=np.clip(c[:K],np.log(0.7),np.log(8192)); c[K:]=np.clip(c[K:],0,np.pi)
        v=np.mean([proxy(np.exp(c[:K]),c[K:],s) for s in (50,51)])
        if v>best: best,th=v,c
    return np.exp(th[:K]),th[K:],best
R={}
NF=64
cfgs={
 'HT (geometric, default)':lambda s:bank_ht(NF),
 'HT real poles only':lambda s:bank_ht_real(NF),
 'S4D-Lin (HiPPO-style)':lambda s:bank_s4d_lin(NF),
 'LRU (ring init)':lambda s:bank_lru(NF,seed=s),
 'HiPPO-LegT (W=4096)':lambda s:bank_legt(NF,4096),
 'STFT (W=256)':lambda s:bank_stft(NF,256),
 'STFT (W=1024)':lambda s:bank_stft(NF,1024),
 'CWT (causal Morlet)':lambda s:bank_cwt(NF),
 'Delay line (64 taps)':lambda s:bank_delay(NF)}
for k,mk in cfgs.items(): R[k]=ev(mk); print(k,R[k]['mc'],R[k]['sfr'],round(time.time()-t0),flush=True)
# tuned (free poles) from HT default and from LRU init
h0,w0=ht_default(NF//2,F=2,hmax=4096); h1,w1,b1=tune(h0,w0,seed=1); print('tuned HT proxy',b1,round(time.time()-t0),flush=True)
lb=bank_lru(NF,seed=0); h2,w2,b2=tune(-1/np.log(np.abs(lb.lam)),np.angle(lb.lam),seed=2); print('tuned LRU proxy',b2,round(time.time()-t0),flush=True)
R['HT tuned (free h, w; init=HT)']=ev(lambda s:ht_bank(h1,w1)); R['LRU tuned (free poles; init=LRU)']=ev(lambda s:ht_bank(h2,w2))
R['_tuned']={'ht':[h1.tolist(),w1.tolist()],'lru':[h2.tolist(),w2.tolist()]}
for k in ('HT tuned (free h, w; init=HT)','LRU tuned (free poles; init=LRU)'): print(k,R[k]['mc'],R[k]['sfr'])
# ---- ablations (HT geometric family)
A={}
for nf in (16,32,64,128): A[f'K: nfeat={nf}']=ev(lambda s,nf=nf:bank_ht(nf))
for F in (1,2,4,8): A[f'F: {F} freq/scale']=ev(lambda s,F=F:bank_ht(NF,F=F))
for hm in (256,1024,4096,16384): A[f'hmax={hm}']=ev(lambda s,hm=hm:bank_ht(NF,hmax=hm))
for r in (1.25,1.5,2.0,3.0): A[f'ratio r={r}']=ev(lambda s,r=r:bank_ht(NF,F=2,r=r))
A['real poles (64 real)']=R['HT real poles only']; A['complex poles (32 complex)']=R['HT (geometric, default)']
A['fixed (default)']=R['HT (geometric, default)']; A['learned/tuned']=R['HT tuned (free h, w; init=HT)']
R['_ablation']=A
for k,v in A.items(): print(k,v['mc'],v['sfr'],flush=True)
json.dump(R,open('out/e2.json','w'),default=float); print('total',time.time()-t0)
