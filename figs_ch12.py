import numpy as np, json, textwrap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR,RED='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0','#C00000'
e2=json.load(open('out/e2.json')); e3=json.load(open('out/e3.json')); e4=json.load(open('out/e4.json')); e6=json.load(open('out/e6.json')); e10=json.load(open('out/e10.json')); e13=json.load(open('out/e13.json')); e14=json.load(open('out/e14.json'))
K4=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']
# ---- Fig 12.1 the book at a glance
boxes=[('Ch. 3','Definition','13 scales cover horizons 1 to 4,096; each channel is a one-pole complex filter'),('Ch. 4','Inverse','Exact inverse to about 1e−15; weighted decoders gain 0.6 to 2.8 dB'),('Ch. 5','Properties','Bounded gain; HT-A bounded when the gain is tied to the step'),
('Ch. 6','Relations','Sliding Fourier bin, gammatone and S4D-Lin are HT banks'),('Ch. 7','Protocol','Rules fixed first; five seeds detect 1.7 SD, ten detect 1.0'),('Ch. 8','Results','H2 not supported; H3 supported; H1 not established'),
('Ch. 9','Strengths and weaknesses','Orthogonalization: capacity 29 to 43, scale-free recall 0.80 to 0.88'),('Ch. 10','Limits','Weak exact recall is specific to white inputs; the lead is fragile'),('Ch. 11','Future directions','Pilots: repairs do not generalize; three-stage roadmap')]
cols=[NAVY,NAVY,NAVY,NAVY,ORG,ORG,GRN,GRN,GRN]
fig,ax=plt.subplots(figsize=(6.5,4.1)); ax.set_xlim(0,3); ax.set_ylim(0,3); ax.axis('off'); ax.grid(False)
for i,(ch,t,txt) in enumerate(boxes):
    r,cx=divmod(i,3); x=cx; y=2-r
    ax.add_patch(FancyBboxPatch((x+0.04,y+0.06),0.86,0.84,boxstyle='round,pad=0.01,rounding_size=0.04',fc='white',ec=cols[i],lw=1.6))
    ax.text(x+0.47,y+0.77,textwrap.fill(f'{ch} {t}',26),ha='center',va='center',fontsize=6.3,fontweight='bold',color=cols[i]); ax.text(x+0.47,y+0.37,textwrap.fill(txt,30),ha='center',va='center',fontsize=5.7,linespacing=1.25)
    if cx<2: ax.add_patch(FancyArrowPatch((x+0.91,y+0.48),(x+1.04,y+0.48),arrowstyle='-|>',mutation_scale=8,lw=1.2,color=GRY))
fig.savefig('out/fig_ch12_glance.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 12.2 verdicts
fig,ax=plt.subplots(1,3,figsize=(6.6,3.2))
h2=e10['h2']; fams=K4; lab=['AR(1)','tones','piecewise','white']
for j,(k,cc,off) in enumerate([('HT default - STFT',NAVY,-0.12),('HT tuned - STFT',ORG,0.12)]):
    for i,f in enumerate(fams): m,lo,hi=h2[k][f]; ax[0].plot([lo,hi],[i+off]*2,color=cc,lw=2); ax[0].plot(m,i+off,'o',color=cc,ms=4,label=k.replace(' - ',' − ') if i==0 else None)
ax[0].axvline(0,color='#333',lw=1); ax[0].set_yticks(range(4)); ax[0].set_yticklabels(lab,fontsize=7); ax[0].set_xlabel('SNR difference to STFT (dB)',fontsize=7); ax[0].set_title('H2: reconstruction',fontsize=8); ax[0].legend(frameon=False,fontsize=5.8,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=1); ax[0].invert_yaxis()
Ts=['128','512','1024']; x=np.arange(3); mods=[('HT-A (doc: omega ungated)','as written',ORG),('HT-A (omega scaled by Delta)','step-scaled',GRN),('Selective real-pole SSM (Mamba-style)','Mamba-style',GRY)]
for j,(m,l,cc) in enumerate(mods): ax[1].bar(x+(j-1)*0.26,[np.median(e4[m][T]) for T in Ts],0.26,color=cc,label=l)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'T={t}' for t in Ts],fontsize=7); ax[1].set_ylim(-0.1,1.2); ax[1].set_ylabel('median R²',fontsize=7); ax[1].set_title('H1: selective recall (toy task)',fontsize=8); ax[1].legend(frameon=False,fontsize=5.8,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3,columnspacing=0.8,handlelength=1.0); ax[1].text(1,1.12,'no trained model run',ha='center',fontsize=6,color=RED)
Tl=e13['Ts']; od=e13['out']
for k,l,cc in [('HT (FFT conv)','FFT form',NAVY),('HT (recurrence)','recurrence',GRN),('HT-A (parallel scan)','HT-A scan',ORG)]:
    ax[2].semilogx(Tl,[np.median(od['Self-attention'][str(T)]['t'])/np.median(od[k][str(T)]['t']) for T in Tl],'o-',color=cc,ms=3.5,lw=1.3,label=l)
ax[2].set_yscale('log'); ax[2].axhline(1.1,color=GRY,ls='--',lw=1); ax[2].axhline(1,color='k',lw=.8); ax[2].set_xscale('log',base=2); ax[2].set_xlabel('sequence length T',fontsize=7); ax[2].set_ylabel('attention time / HT time',fontsize=7); ax[2].set_title('H3: cost',fontsize=8); ax[2].legend(frameon=False,fontsize=5.8,loc='upper left')
fig.tight_layout(); fig.savefig('out/fig_ch12_verdicts.png',dpi=200); plt.close()
# ---- Fig 12.3 evidence ladder
levels=[('Proved',NAVY,['Exact inverse of HT','Stability bound b/(1−|λ|)']),('Derived in this book',NAVY,['HT-A bound when b ≤ βΔ','Noise formulas, weighted decoders','Frame symbol']),('Checked numerically',GRN,['Inverse to 1e−15; variance','Relations: DFT limit, gammatone, S4D-Lin','Scale equivariance (one family)']),('Supported by reduced-scale tests',GLD,['H3 for time-invariant HT; scan not','Scale-free lead (fragile); weak exact recall','Constant-Q helps classification; HT-A needs scaling']),('Pilot level',ORG,['Orthogonalization (memory probes only)','Cascades: no gain; irregular sampling: open']),('Untested',RED,['Trained-model accuracy (H1)','Real data; GPU kernels','HT-A inverse uniqueness'])]
fig,ax=plt.subplots(figsize=(6.5,3.6)); ax.set_xlim(0,11.2); ax.set_ylim(0,len(levels)); ax.axis('off'); ax.grid(False)
for i,(t,cc,cl) in enumerate(levels):
    y=len(levels)-1-i; wdt=10-0.55*i*1.0
    ax.add_patch(Rectangle((0,y+0.08),2.6,0.84,fc=cc,ec='none')); ax.text(1.3,y+0.5,textwrap.fill(t,17),ha='center',va='center',fontsize=6.6,color='white',fontweight='bold')
    ax.add_patch(Rectangle((2.6,y+0.08),8.1,0.84,fc=cc,ec='none',alpha=0.12)); ax.text(2.75,y+0.5,'\n'.join(textwrap.fill(c_,62) for c_ in ['  ·  '.join(cl)]),ha='left',va='center',fontsize=5.7,linespacing=1.3)
ax.annotate('',xy=(10.95,0.1),xytext=(10.95,len(levels)-0.1),arrowprops=dict(arrowstyle='<|-',color=GRY,lw=1.2)); ax.text(11.15,len(levels)/2,'weaker evidence',rotation=90,va='center',fontsize=6.3,color=GRY)
fig.savefig('out/fig_ch12_ladder.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 12.4 scorecard
def avg(nm,dec='LS (unweighted)'): return float(np.mean([e6[nm]['8'][k][dec] for k in K4]))
rec_best=max(avg('STFT (W=128)'),avg('Undecimated wavelet (db2)')); nonHT=['S4D-Lin (HiPPO-style)','LRU (ring init)','HiPPO-LegT (W=4096)','STFT (W=256)','STFT (W=1024)','CWT (causal Morlet)','Delay line (64 taps)']
mcb=max(e2[k]['mc'][0] for k in nonHT); sfb=max(e2[k]['sfr'][0] for k in nonHT); cl=e3['classify']; clb=max(cl[k]['mean'] for k in ['CWT (Morlet, log freqs)','STFT (W=128, linear bins)','LRU'])
T='8192'; att=np.median(e13['out']['Self-attention'][T]['t']); ratio=lambda k:att/np.median(e13['out'][k][T]['t']); mamba=np.median(e4['Selective real-pole SSM (Mamba-style)']['1024'])
rows=['HT default','HT tuned or constant-Q','HT orthogonalized','HT-A as written','HT-A step-scaled']
cols=['Reconstruction\n(ρ = 8)','Exact\nrecall','Scale-free\nrecall','Classification','Selective\nrecall (T=1024)','Latency\n(T=8,192)']
nan=np.nan
M=np.array([[avg('HT default')/rec_best,e2['HT (geometric, default)']['mc'][0]/mcb,e2['HT (geometric, default)']['sfr'][0]/sfb,cl['HT (geometric, default)']['mean']/clb,nan,ratio('HT (FFT conv)')],
[avg('HT tuned')/rec_best,e2['HT tuned (free h, w; init=HT)']['mc'][0]/mcb,e2['HT tuned (free h, w; init=HT)']['sfr'][0]/sfb,cl['HT (freq-tuned, constant-Q)']['mean']/clb,nan,ratio('HT (FFT conv)')],
[nan,e14['res']['HT, orthogonalized']['0.01']['mc'][0]/mcb,e14['res']['HT, orthogonalized']['0.01']['sfr'][0]/sfb,nan,nan,ratio('HT (FFT conv)')],
[nan,nan,nan,nan,np.median(e4['HT-A (doc: omega ungated)']['1024'])/mamba,ratio('HT-A (parallel scan)')],
[nan,nan,nan,nan,np.median(e4['HT-A (omega scaled by Delta)']['1024'])/mamba,ratio('HT-A (parallel scan)')]])
fig,ax=plt.subplots(figsize=(6.4,2.9)); L=np.log2(M); im=ax.imshow(np.ma.masked_invalid(np.clip(L,-2,2)),cmap='RdBu',vmin=-2,vmax=2,aspect='auto'); ax.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]; ax.text(j,i,'—' if np.isnan(v) else (f'{v:.2f}' if v<10 else f'{v:.0f}'),ha='center',va='center',fontsize=8,color='white' if (not np.isnan(v) and abs(np.log2(v))>1.2) else '#222')
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,fontsize=6.6); ax.xaxis.tick_top(); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=7.4); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.tight_layout(); fig.savefig('out/fig_ch12_scorecard.png',dpi=200); plt.close()
# ---- Fig 12.5 decision guide
rowsd=[('Is cost at long context the main concern, and does the task fit a fixed filter bank?','Use the time-invariant HT: FFT convolution for training, recurrence for streaming. Faster than attention beyond about 1,000 samples (CPU, numpy).',GRN),
('Does the task need exact recall of single recent samples under noise?','Orthogonalize the bank, or pair it with an exact-recall component (delay line, short window). Consider another layer if exact recall dominates.',ORG),
('Is the signal band known and narrow?','Place channels on that band with constant-Q spacing (Q of about π or more). Otherwise use geometric spacing with four frequencies per scale.',GRN),
('Does the task need input-dependent selection?','Use HT-A with the rotation scaled by the step and the gain tied to the step. Evidence is a toy task; compare with a selective SSM. The scan showed no cost advantage.',GLD),
('Is reconstruction accuracy at equal coefficient count the goal?','Prefer an STFT or a wavelet (the default HT is 2.4 to 3.9 dB lower). If HT is needed, tune the placement and use a weighted decoder.',RED)]
fig,ax=plt.subplots(figsize=(6.5,4.1)); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis('off'); ax.grid(False)
for i,(q,a_,cc) in enumerate(rowsd):
    y=4-i
    ax.add_patch(FancyBboxPatch((0.05,y+0.1),4.1,0.8,boxstyle='round,pad=0.01,rounding_size=0.05',fc='white',ec=NAVY,lw=1.4)); ax.text(2.1,y+0.5,textwrap.fill(q,46),ha='center',va='center',fontsize=6.4)
    ax.add_patch(FancyArrowPatch((4.2,y+0.5),(4.9,y+0.5),arrowstyle='-|>',mutation_scale=9,lw=1.3,color=GRY)); ax.text(4.55,y+0.64,'yes',ha='center',fontsize=5.8,color=GRY)
    ax.add_patch(FancyBboxPatch((4.95,y+0.1),5.0,0.8,boxstyle='round,pad=0.01,rounding_size=0.05',fc=cc,ec='none',alpha=0.14)); ax.add_patch(FancyBboxPatch((4.95,y+0.1),5.0,0.8,boxstyle='round,pad=0.01,rounding_size=0.05',fc='none',ec=cc,lw=1.4)); ax.text(7.45,y+0.5,textwrap.fill(a_,58),ha='center',va='center',fontsize=6.0)
    if i<len(rowsd)-1: ax.add_patch(FancyArrowPatch((2.1,y+0.1),(2.1,y-0.1),arrowstyle='-|>',mutation_scale=8,lw=1,color=GRY)); ax.text(2.2,y-0.0,'no / also',fontsize=5.4,color=GRY,va='center')
fig.savefig('out/fig_ch12_guide.png',dpi=200,bbox_inches='tight'); plt.close(); print('ok')
