"""Limits: (1) nonlinear targets with a linear and a second-order readout; (2) storage precision of the orthogonalized bank."""
import numpy as np, json, time
from lib import *
t0=time.time(); R={}
def quad_fit(Z,Y,ntr,nva,alphas=(1e-4,1e-2,1,1e2)):
    iu=np.triu_indices(Z.shape[1]); Q=(Z[:,:,None]*Z[:,None,:])[:,iu[0],iu[1]]; Q=np.concatenate([Z,Q],1); Q=(Q-Q[:ntr].mean(0))/(Q[:ntr].std(0)+1e-12)
    A=Q[:ntr]; w,V=np.linalg.eigh(A.T@A); B=V.T@(A.T@(Y[:ntr]-Y[:ntr].mean(0))); best=None
    for al in alphas:
        W=V@(B/(w[:,None]+al*ntr)); pv=Q[ntr:ntr+nva]@W+Y[:ntr].mean(0); r=1-((Y[ntr:ntr+nva]-pv)**2).mean(0)/Y[ntr:ntr+nva].var(0)
        if best is None or r.mean()>best[0]: best=(r.mean(),W)
    pt=Q[ntr+nva:]@best[1]+Y[:ntr].mean(0); return 1-((Y[ntr+nva:]-pt)**2).mean(0)/Y[ntr+nva:].var(0)
banks={'HT default':lambda s:bank_ht(64),'S4D-Lin':lambda s:bank_s4d_lin(64),'STFT (W=256)':lambda s:bank_stft(64,256),'Delay line':lambda s:bank_delay(64)}
tg={'u[t−3]·u[t−10]':(3,10,'prod'),'u[t−5]²':(5,5,'sq'),'u[t−30]·u[t−100]':(30,100,'prod')}
res={k:{t:{'lin':[],'nl':[]} for t in tg} for k in banks}
L=24000; burn=10000
for s in range(3):
    rng=np.random.default_rng(1000+s); u=rng.standard_normal(L)
    Y=np.stack([ (u[burn-a:L-a]*u[burn-b:L-b]) if kind=='prod' else (u[burn-a:L-a]**2-1) for (a,b,kind) in tg.values()],1)
    for k,mk in banks.items():
        b=mk(s); F=b.stream(u)[burn:]; Z=(F-F.mean(0))/(F.std(0)+1e-12); Z=Z+1e-2*rng.standard_normal(Z.shape); n=len(Z); ntr=int(.4*n); nva=int(.2*n)
        rl=ridge_curves(Z,Y,ntr,nva); rn=quad_fit(Z,Y,ntr,nva)
        for j,t in enumerate(tg): res[k][t]['lin'].append(float(rl[j])); res[k][t]['nl'].append(float(rn[j]))
    print('seed',s,round(time.time()-t0),flush=True)
R['nonlinear']={k:{t:{r:[float(np.mean(v)),float(np.std(v))] for r,v in d.items()} for t,d in dd.items()} for k,dd in res.items()}
b0=bank_ht(64); Km=b0.kernel(32768); G=Km.T@Km; w,V=np.linalg.eigh(G); w=np.maximum(w,1e-12*w.max()); Wm=(V/np.sqrt(w))@V.T
variants={'float64 throughout':lambda u:b0.stream(u)@Wm,'features stored in float32':lambda u:(b0.stream(u).astype(np.float32).astype(np.float64))@Wm,'features and mixing in float32':lambda u:(b0.stream(u).astype(np.float32)@Wm.astype(np.float32)).astype(np.float64),'default bank, float32 features':lambda u:b0.stream(u).astype(np.float32).astype(np.float64)}
prec={}
for name,fn in variants.items():
    mcs=[];sfs=[]
    for s in range(3): mcs.append(memory_capacity(b0,s,stream=fn)); sfs.append(scale_free_recall(b0,s,stream=fn))
    prec[name]={'mc':[float(np.mean(mcs)),float(np.std(mcs))],'sfr':[float(np.mean(sfs)),float(np.std(sfs))]}
lown={}
for nz in (1e-4,1e-6):
    for name,fn in {'float64':variants['float64 throughout'],'float32 mixing':variants['features and mixing in float32']}.items():
        mcs=[];sfs=[]
        for s_ in range(3): mcs.append(memory_capacity(b0,s_,noise=nz,stream=fn)); sfs.append(scale_free_recall(b0,s_,noise=nz,stream=fn))
        lown[f'{name}|{nz}']={'mc':float(np.mean(mcs)),'sfr':float(np.mean(sfs))}
R['precision_lownoise']=lown; R['precision']=prec; R['max_mix_entry']=float(np.abs(Wm).max())
json.dump(R,open('out/e15a.json','w'))
for k,dd in R['nonlinear'].items(): print(k,{t:(round(v['lin'][0],2),round(v['nl'][0],2)) for t,v in dd.items()})
