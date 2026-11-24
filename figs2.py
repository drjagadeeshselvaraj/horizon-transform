import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25})
e1=json.load(open('out/e1.json')); e2=json.load(open('out/e2.json')); e3=json.load(open('out/e3.json')); e4=json.load(open('out/e4.json'))
KK=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']
# ---- quantisation equal-rate bars
def avgq(nm,rho,b): d=e1['res'][nm][rho]; return np.mean([d[k]['q'][b] for k in KK])
def avgref(nm,b): return np.mean([e1['ref'][nm][k]['q'][b] for k in KK])
groups=[('ρ=1\n8 bit/coef',None),('ρ=2\n4 bit/coef','2','4'),('ρ=4\n2 bit/coef','4','2'),('ρ=8\n1 bit/coef','8','1')]
meths=[('HT default, LS','HT (default) - LS inverse','#1F3864'),('HT default, Ch. 4 inv.','HT (default) - paper inverse','#8FAADC'),('HT tuned, LS','HT (tuned) - LS inverse','#C55A11'),('STFT','STFT (W=128)','#548235'),('Undecimated wavelet','Undecimated wavelet (db2)','#7F7F7F')]
fig,ax=plt.subplots(figsize=(6.3,2.9)); wd=.14
ax.bar(-0.5*wd-0.0,avgref('PCM (identity, rho=1)','8'),wd,color='#BF9000',label='PCM'); ax.bar(0.5*wd,avgref('DCT (orthonormal, rho=1)','8'),wd,color='#7030A0',label='DCT')
for gi,g in enumerate(groups[1:],1):
    for mi,(lab,key,c) in enumerate(meths):
        v=avgq(key,g[1],g[2]); ax.bar(gi+(mi-2)*wd,v,wd,color=c,label=lab if gi==1 else None)
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylabel('output SNR (dB)'); ax.set_title('Every group spends 8 bits per sample',fontsize=8.5)
ax.axhline(0,color='k',lw=.6); fig.legend(loc='lower center',ncol=4,frameon=False,fontsize=7); fig.tight_layout(rect=(0,.15,1,1)); fig.savefig('out/fig_quant.png',dpi=200); plt.close()
# ---- MC vs SFR scatter
pts={'HT default':'HT (geometric, default)','HT tuned':'HT tuned (free h, w; init=HT)','S4D-Lin':'S4D-Lin (HiPPO-style)','LRU':'LRU (ring init)','LRU tuned':'LRU tuned (free poles; init=LRU)','LegT':'HiPPO-LegT (W=4096)','Morlet':'CWT (causal Morlet)','STFT 256':'STFT (W=256)','STFT 1024':'STFT (W=1024)','Delay line':'Delay line (64 taps)','HT real':'HT real poles only'}
off={'HT default':(4,6),'HT tuned':(5,-10),'S4D-Lin':(5,4),'LRU':(-6,-10),'LRU tuned':(5,-10),'LegT':(-8,-11),'Morlet':(5,4),'STFT 256':(-10,8),'STFT 1024':(-14,-11),'Delay line':(-12,-12),'HT real':(5,4)}
fig,ax=plt.subplots(figsize=(6.0,3.4))
for lab,k in pts.items():
    mc,sd=e2[k]['mc']; sf,ssd=e2[k]['sfr']; c='#1F3864' if lab.startswith('HT') and lab!='HT real' else '#C00000' if lab=='HT real' else '#7F7F7F'
    ax.errorbar(mc,sf,xerr=sd,yerr=ssd,fmt='o',color=c,ms=5,lw=.8,capsize=1.5); ax.annotate(lab,(mc,sf),xytext=off[lab],textcoords='offset points',fontsize=7)
ax.set_xlabel('memory capacity, exact recall (max 64)'); ax.set_ylabel('scale-free recall (R²)'); ax.set_xlim(0,70); ax.set_ylim(0,.95)
fig.tight_layout(); fig.savefig('out/fig_tradeoff.png',dpi=200); plt.close()
# ---- forecasting + classification
fc=e3['forecast']; hz=[1,24,168,720]
names=[('HT (geometric, default)','HT default','#1F3864','-','o'),('S4D-Lin','S4D-Lin','#C55A11','-','s'),('LRU','LRU','#BF9000','-','^'),('HiPPO-LegT (W=4096)','HiPPO-LegT','#7030A0','-','v'),('CWT (causal Morlet)','CWT','#00B0F0','-','D'),('Delay line (64 taps)','Delay line','#7F7F7F','--','x'),('STFT (W=256, low band)','STFT low band','#548235','--','+')]
fig,ax=plt.subplots(1,2,figsize=(6.3,2.9),gridspec_kw={'width_ratios':[1.15,1]})
for k,l,c,ls,m in names: ax[0].plot(range(4),fc[k]['mean'],color=c,ls=ls,marker=m,ms=4,lw=1.2,label=l)
ax[0].plot(range(4),fc['_ceiling_nmse'],color='k',ls=':',lw=1.2,label='irreducible level'); ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f'H={h}' for h in hz]); ax[0].set_ylabel('normalized MSE (lower is better)'); ax[0].set_title('Forecasting (synthetic multi-scale series)',fontsize=8.5)
ax[0].set_ylim(0.05,0.88); ax[0].legend(fontsize=6,frameon=False,loc='upper left',ncol=2)
cl=e3['classify']; order=[('CWT (Morlet, log freqs)','CWT, log freqs'),('HT (freq-tuned, constant-Q)','HT, log freqs'),('LRU','LRU'),('STFT (W=128, linear bins)','STFT W=128'),('HT (geometric, default)','HT default'),('STFT (W=256, linear bins)','STFT W=256')][::-1]
cc=['#1F3864' if 'HT' in l else '#7F7F7F' for _,l in order]
ax[1].barh([l for _,l in order],[100*cl[k]['mean'] for k,_ in order],xerr=[100*cl[k]['std'] for k,_ in order],color=cc,capsize=2); ax[1].axvline(100/3,color='k',ls=':',lw=1); ax[1].set_ylim(-.6,5.9); ax[1].text(100/3+.5,5.55,'chance',fontsize=6.5); ax[1].set_xlim(30,70); ax[1].set_xlabel('accuracy (%)'); ax[1].set_title('Spectral-signature classification',fontsize=8.5); ax[1].grid(axis='y',alpha=0)
fig.tight_layout(); fig.savefig('out/fig_fc_cls.png',dpi=200); plt.close()
# ---- selective recall
ms=[('HT (time-invariant, no gating)','HT\n(no gates)'),('HT-A (doc: omega ungated)','HT-A\n(Sec. 2.2)'),('HT-A (omega scaled by Delta)','HT-A\n(ω·Δ_t)'),('Selective real-pole SSM (Mamba-style)','Selective\nSSM')]
fig,ax=plt.subplots(1,3,figsize=(6.3,2.7),sharey=True); rng=np.random.default_rng(0)
for a,T in zip(ax,('128','512','1024')):
    for i,(k,l) in enumerate(ms):
        v=np.array(e4[k][T]); vc=np.clip(v,-.22,1.05); a.scatter(i+rng.uniform(-.12,.12,len(v)),vc,s=14,color='#1F3864' if i else '#7F7F7F',zorder=3)
        a.hlines(np.median(v),i-.28,i+.28,color='#C55A11',lw=2)
        for x,y in zip(v,vc):
            if x<-1: a.annotate(f'{x:.1f}\n(off scale)',(i,y),xytext=(8,-2),textcoords='offset points',fontsize=6)
    a.set_xticks(range(4)); a.set_xticklabels([l for _,l in ms],fontsize=6.5); a.set_title(f'T = {T}'+(' (tuned)' if T=='128' else ''),fontsize=8.5); a.set_ylim(-.25,1.1)
ax[0].set_ylabel('held-out R² of recalled value')
fig.text(.5,.005,'dots: 5 seeds; orange bar: median; one value (−34.6) is drawn at the bottom edge',ha='center',fontsize=6.5,color='#444'); fig.tight_layout(rect=(0,.04,1,1)); fig.savefig('out/fig_selective.png',dpi=200); plt.close()
# ---- ablations
A=e2['_ablation']
sets=[('Features (K)',[('K: nfeat=16',16),('K: nfeat=32',32),('K: nfeat=64',64),('K: nfeat=128',128)],True),('Frequencies per scale',[('F: 1 freq/scale',1),('F: 2 freq/scale',2),('F: 4 freq/scale',4),('F: 8 freq/scale',8)],True),('Longest horizon',[('hmax=256',256),('hmax=1024',1024),('hmax=4096',4096),('hmax=16384',16384)],True),('Spacing ratio r (16 scales)',[('ratio r=1.25',1.25),('ratio r=1.5',1.5),('ratio r=2.0',2.0),('ratio r=3.0',3.0)],False)]
fig,ax=plt.subplots(2,2,figsize=(6.3,4.2))
for a,(t,items,lg) in zip(ax.ravel(),sets):
    xs=[x for _,x in items]; sf=[A[k]['sfr'][0] for k,_ in items]; se=[A[k]['sfr'][1] for k,_ in items]; mc=[A[k]['mc'][0] for k,_ in items]; me=[A[k]['mc'][1] for k,_ in items]
    a.errorbar(range(4),sf,yerr=se,color='#1F3864',marker='o',ms=4,lw=1.3,capsize=2,label='scale-free recall'); a.set_ylim(0,1); a.set_xticks(range(4)); a.set_xticklabels([str(x) for x in xs]); a.set_title(t,fontsize=8.5)
    b=a.twinx(); b.errorbar(range(4),mc,yerr=me,color='#C55A11',marker='s',ms=4,lw=1.3,ls='--',capsize=2,label='memory capacity'); b.set_ylim(0,50); b.spines['right'].set_visible(True); b.grid(False)
    a.set_ylabel('scale-free recall',color='#1F3864',fontsize=7.5); b.set_ylabel('memory capacity',color='#C55A11',fontsize=7.5)
fig.tight_layout(); fig.savefig('out/fig_ablation.png',dpi=200); plt.close(); print('ok')
