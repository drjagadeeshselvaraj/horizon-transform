"""Chapter 7: statistical planning (minimal detectable effects), retrospective application of the decision rules to H1 and H2, and run counts."""
import numpy as np, json
from scipy import stats
import e1
from e1 import N, sig, KINDS, A_ht, A_stft, A_udwt, ls_dec
from lib import ht_default
R={}; rng=np.random.default_rng(0)
# ---- (a) minimal detectable effect (paired / one-sample t test, power 0.8) in units of the SD of the paired difference
def mde(n,alpha,power=0.8):
    lo,hi=0.0,10.0
    for _ in range(60):
        d=(lo+hi)/2; df=n-1; tcrit=stats.t.ppf(1-alpha/2,df); p=1-stats.nct.cdf(tcrit,df,d*np.sqrt(n))+stats.nct.cdf(-tcrit,df,d*np.sqrt(n))
        if p<power: lo=d
        else: hi=d
    return (lo+hi)/2
ns=list(range(3,31)); R['mde']={'n':ns,'a05':[mde(n,0.05) for n in ns],'a0167':[mde(n,0.05/3) for n in ns],'a0083':[mde(n,0.05/6) for n in ns]}
# ---- (b) per-signal H2 comparison at rho = 8 (least-squares decoder, per-channel noise 20 dB)
o1=json.load(open('out/e1.json')); SNR=100.0; rho=8; K=4
h,w=ht_default(K,F=2,hmax=N); mats={}; A,ch,_,_=A_ht(h,w); mats['HT default']=(A,ch)
th=o1['ht_tuned'][str(rho)]; A,ch,_,_=A_ht(np.array(th[0]),np.array(th[1])); mats['HT tuned']=(A,ch)
mats['STFT']=tuple(A_stft(rho)); mats['Wavelet']=tuple(A_udwt(rho-1))
def snr_signal(A,ch,x):
    P=ls_dec(A); colsq=(P**2).sum(0); y=A@x; ids=np.unique(ch); pw={c:np.mean(y[ch==c]**2) for c in ids}; s2=np.array([pw[c] for c in ch])/SNR+1e-30
    return 10*np.log10(np.sum(x**2)/np.sum(s2*colsq))
per={nm:{k:[] for k in KINDS} for nm in mats}
for k in KINDS:
    xs=[sig(k,np.random.default_rng(5000+i)) for i in range(20)]
    for nm,(A,ch) in mats.items(): per[nm][k]=[float(snr_signal(A,ch,x)) for x in xs]
def boot_ci(d,B=10000):
    d=np.array(d); m=[np.mean(rng.choice(d,len(d))) for _ in range(B)]; return float(d.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
h2={}
for a,b in [('HT default','STFT'),('HT tuned','STFT'),('Wavelet','STFT'),('HT tuned','Wavelet')]:
    h2[f'{a} - {b}']={k:boot_ci(np.array(per[a][k])-np.array(per[b][k])) for k in KINDS}
R['h2']=h2; R['h2_sd']={nm:{k:float(np.std(v)) for k,v in d.items()} for nm,d in per.items()}
# ---- (c) H1 proxy: selective recall, paired by seed (n = 5), equivalence margin 0.05 in R^2
e4=json.load(open('out/e4.json')); h1={}
for T in ('128','512','1024'):
    for a,b in [('HT-A (omega scaled by Delta)','Selective real-pole SSM (Mamba-style)'),('HT-A (doc: omega ungated)','Selective real-pole SSM (Mamba-style)')]:
        d=np.array(e4[a][T])-np.array(e4[b][T]); n=len(d); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n); tcr=stats.t.ppf(0.975,n-1); h1[f'{a}|{T}']=[float(m),float(m-tcr*se),float(m+tcr*se)]
R['h1']=h1; R['h1_sd']={m:{T:float(np.std(v,ddof=1)) for T,v in d.items() if T in ('128','512','1024')} for m,d in e4.items()}
# ---- (d) planned run counts (arithmetic, no compute assumptions)
models=6; seeds=5; trials=20
plan={'LRA classification':{'final':6*models*seeds,'search':6*models*trials},'Forecasting benchmarks':{'final':7*4*models*seeds,'search':7*models*trials},'Audio and EEG':{'final':2*models*seeds,'search':2*models*trials},'Ablations (HT, HT-A)':{'final':4*6*2*seeds,'search':0},'Cost and latency':{'final':8*models*3,'search':0}}
R['plan']=plan
json.dump(R,open('out/e10.json','w'),default=float)
print('mde n=5,10,20 (a=.05):',[round(R['mde']['a05'][ns.index(n)],2) for n in (5,10,20)],' (a=.0167):',[round(R['mde']['a0167'][ns.index(n)],2) for n in (5,10,20)])
for k,v in h2.items(): print(k,{kk:[round(x,1) for x in vv] for kk,vv in v.items()})
for k,v in h1.items(): print(k,[round(x,3) for x in v])
print('plan',{k:(v['final'],v['search']) for k,v in plan.items()})
