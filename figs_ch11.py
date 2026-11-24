import numpy as np, json, textwrap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR,RED='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0','#C00000'
a=json.load(open('out/e16a.json')); b=json.load(open('out/e16b.json')); c=json.load(open('out/e16c.json')); d=json.load(open('out/e16d.json'))
# ---- Fig 11.1 roadmap
fig,ax=plt.subplots(figsize=(6.5,3.9)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off'); ax.grid(False)
def box(x,y,w,h,title,lines,col,wrap=30):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc='white',ec=col,lw=1.6)); ax.text(x+w/2,y+h-0.28,title,ha='center',va='center',fontsize=7.0,fontweight='bold',color=col)
    ax.text(x+w/2,y+h/2-0.2,'\n'.join(textwrap.fill(l,wrap) for l in lines),ha='center',va='center',fontsize=6.2,linespacing=1.25)
box(0.1,2.6,2.9,3.1,'Stage 1: feature banks',['Orthogonalization across tasks','Scales against frequencies','Gammatone cascades','Irregular sampling','Real, non-white inputs (open)'],GRN)
box(3.55,2.6,2.9,3.1,'Stage 2: small models',['Initialization swap','Frequency placement','Exact-recall hybrid','Step-scaled HT-A as default'],ORG)
box(7.0,2.6,2.9,3.1,'Stage 3: full protocol',['Long Range Arena, forecasting, audio, EEG','Trained S4, LRU, Mamba, Transformer','Ten seeds, rules of Table 7.5','Verdicts on H1 to H3'],NAVY)
box(0.1,0.15,4.7,1.9,'Track A: theory',['HT-A invertibility and frame bounds','Conditioning and capacity of exponential banks','Optimal bank under noise'],PUR,wrap=52)
box(5.2,0.15,4.7,1.9,'Track B: engineering',['Fused scan kernels; GPU benchmark against FlashAttention','Precision: float16 and bfloat16','Streaming latency against a key-value cache'],GLD,wrap=52)
for (x0,x1) in ((3.0,3.55),(6.45,7.0)): ax.add_patch(FancyArrowPatch((x0,4.15),(x1,4.15),arrowstyle='-|>',mutation_scale=12,lw=1.6,color=GRY))
ax.text(3.27,4.45,'gate',ha='center',fontsize=6.2,color=GRY); ax.text(6.72,4.45,'gate',ha='center',fontsize=6.2,color=GRY)
ax.add_patch(FancyArrowPatch((2.45,2.05),(2.45,2.6),arrowstyle='<|-|>',mutation_scale=9,lw=1,color=PUR,ls='--')); ax.add_patch(FancyArrowPatch((7.6,2.05),(7.6,2.6),arrowstyle='<|-|>',mutation_scale=9,lw=1,color=GLD,ls='--'))
fig.savefig('out/fig_ch11_roadmap.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 11.2 pilot 1
fig,ax=plt.subplots(1,3,figsize=(6.6,2.9))
rec=a['reconstruction']; w=0.26; x=np.arange(2)
for j,(k,cc) in enumerate([('HT default',NAVY),('HT default, orthogonalized',GRN),('STFT (W=128)',GRY)]): ax[0].bar(x+(j-1)*w,[rec[r][k] for r in ('8','16')],w,color=cc,label=k.replace('HT default, orthogonalized','HT orthogonalized').replace('HT default','HT default'))
ax[0].set_xticks(x); ax[0].set_xticklabels(['ρ = 8','ρ = 16']); ax[0].set_ylim(20,34); ax[0].set_ylabel('output SNR (dB)'); ax[0].set_title('Reconstruction',fontsize=8); ax[0].legend(frameon=False,fontsize=5.6,loc='upper left')
fc=a['forecast']; hz=fc['hz']; x=np.arange(3)
for j,(k,cc) in enumerate([('HT default',NAVY),('HT default, orthogonalized',GRN)]): ax[1].bar(x+(j-0.5)*0.35,fc['nmse'][f'{k}|0.01'],0.35,color=cc)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'H = {h}' for h in hz],fontsize=7); ax[1].set_ylim(0.25,0.6); ax[1].set_ylabel('normalized MSE'); ax[1].set_title('Forecasting (1% noise)',fontsize=8)
cl=a['classification']; labs=['default,\npair','const-Q,\npair','const-Q,\nfeature','const-Q,\northog.']; keys=['default bank, pair power','constant-Q bank, pair power','constant-Q bank, per-feature energy','constant-Q bank, orthogonalized']; cs=[NAVY,GRY,ORG,GRN]
for j,amp in enumerate(('0.1','0.15')):
    pass
x=np.arange(4); ax[2].bar(x-0.19,[100*cl[f'{k}|0.1'][0] for k in keys],0.38,color=cs,alpha=.55); ax[2].bar(x+0.19,[100*cl[f'{k}|0.15'][0] for k in keys],0.38,color=cs)
ax[2].set_xticks(x); ax[2].set_xticklabels(labs,fontsize=5.6); ax[2].set_ylim(40,95); ax[2].set_ylabel('accuracy (%)'); ax[2].set_title('Classification',fontsize=8); ax[2].text(1.5,91,'light: 0.10   dark: 0.15',ha='center',fontsize=5.8,color='#444')
fig.tight_layout(); fig.savefig('out/fig_ch11_pilot1.png',dpi=200); plt.close()
# ---- Fig 11.3 pilot 2
fig,ax=plt.subplots(1,2,figsize=(6.4,2.8)); r=b['r_eff']
for a_,m,t in zip(ax,['mc','sfr'],['Memory capacity','Scale-free recall']):
    a_.semilogx(r,b['default'][m],'o-',color=NAVY,ms=4,lw=1.4,label='default'); a_.semilogx(r,b['orthogonalized'][m],'s-',color=GRN,ms=4,lw=1.4,label='orthogonalized'); a_.set_xlabel('effective spacing ratio r (32 complex channels)'); a_.set_title(t,fontsize=8.3)
ax[0].legend(frameon=False,fontsize=6.6,loc='upper right')
fig.tight_layout(); fig.savefig('out/fig_ch11_pilot2.png',dpi=200); plt.close()
# ---- Fig 11.4 pilot 3
fig,ax=plt.subplots(1,2,figsize=(6.4,2.7)); orders=[1,2,3,4]
for amp,cc in (('0.07',GRY),('0.1',NAVY)):
    m=[100*c[f'{o}|{amp}'][0] for o in orders]; s=[100*c[f'{o}|{amp}'][1] for o in orders]; ax[0].errorbar(orders,m,yerr=s,marker='o',ms=4,lw=1.4,capsize=2,color=cc,label=f'amplitude {float(amp):.2f}')
ax[0].set_xticks(orders); ax[0].set_xlabel('cascade order (equal total delay)'); ax[0].set_ylabel('accuracy (%)'); ax[0].legend(frameon=False,fontsize=6.6); ax[0].set_title('Classification',fontsize=8.3)
ax[1].plot(orders,[c['relative_bandwidth'][str(o)] for o in orders],'s-',color=ORG,ms=4,lw=1.4); ax[1].set_xticks(orders); ax[1].set_xlabel('cascade order (equal total delay)'); ax[1].set_ylabel('−3 dB bandwidth / channel frequency'); ax[1].set_title('Width of the 10 Hz channel',fontsize=8.3); ax[1].set_ylim(0,0.7)
fig.tight_layout(); fig.savefig('out/fig_ch11_pilot3.png',dpi=200); plt.close()
# ---- Fig 11.5 pilot 4
fig,ax=plt.subplots(1,2,figsize=(6.4,2.8),sharey=True); ps=[1.0,0.7,0.5,0.3]; names=['naive (index time)','zero-filled','time-aware']; cs=[RED,GRY,NAVY]
for a_,cond,t in zip(ax,['matched','trained on regular samples'],['Trained and tested at the same density','Trained on regular samples']):
    for n,cc in zip(names,cs): a_.errorbar(ps,[100*d[f'{cond}|{n}|{p}'][0] for p in ps],yerr=[100*d[f'{cond}|{n}|{p}'][1] for p in ps],marker='o',ms=4,lw=1.4,capsize=2,color=cc,label=n)
    a_.axhline(100/3,color='k',ls=':',lw=1); a_.set_xlabel('fraction of samples retained'); a_.set_title(t,fontsize=8); a_.invert_xaxis()
ax[0].set_ylabel('accuracy (%)'); ax[0].legend(frameon=False,fontsize=6.3,loc='lower left')
fig.tight_layout(); fig.savefig('out/fig_ch11_pilot4.png',dpi=200); plt.close()
# ---- Fig 11.6 planned size of the studies
st=[('Full protocol (final runs)',1464,NAVY),('Full protocol (search runs)',1800,NAVY),('Initialization swap',270,ORG),('Frequency placement',80,ORG),('Exact-recall hybrid',75,ORG),('GPU benchmark (timings)',600,GLD),('HT-A counterexample search (draws)',10000,PUR)]
fig,ax=plt.subplots(figsize=(6.2,2.6)); y=np.arange(len(st))[::-1]
for yi,(n,v,cc) in zip(y,st): ax.barh(yi,v,color=cc,height=0.65); ax.text(v*1.12,yi,f'{v:,}',va='center',fontsize=7)
ax.set_xscale('log'); ax.set_xlim(30,40000); ax.set_yticks(y); ax.set_yticklabels([s[0] for s in st],fontsize=7.4); ax.set_xlabel('planned runs, timings or draws (logarithmic axis)')
fig.tight_layout(); fig.savefig('out/fig_ch11_sizes.png',dpi=200); plt.close(); print('ok')
