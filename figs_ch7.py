import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0'
e10=json.load(open('out/e10.json'))
# ---- Fig 7.1 protocol flow
fig,ax=plt.subplots(figsize=(6.3,3.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.grid(False)
steps=[('1  Hypotheses','H1 adaptive memory\nH2 reconstruction\nH3 cost',NAVY),('2  Operationalize','tasks, metrics,\nmargins, decision rules',NAVY),('3  Design','baselines, equal budgets,\nseeds, ablations',GRN),('4  Execute','trained models (planned)\nor feature-bank proxies',ORG),('5  Analyze','paired differences,\nconfidence intervals,\nHolm correction',PUR),('6  Decide','supported, not supported\nor inconclusive',GLD)]
xs=[0.01,0.345,0.68]; ys=[0.60,0.12]
pos=[(xs[0],ys[0]),(xs[1],ys[0]),(xs[2],ys[0]),(xs[2],ys[1]),(xs[1],ys[1]),(xs[0],ys[1])]
for (t,d,c),(x,y) in zip(steps,pos):
    ax.add_patch(FancyBboxPatch((x,y),0.31,0.30,boxstyle='round,pad=0.005,rounding_size=0.015',fc='white',ec=c,lw=1.5)); ax.text(x+0.155,y+0.24,t,ha='center',va='center',fontsize=8,fontweight='bold',color=c); ax.text(x+0.155,y+0.11,d,ha='center',va='center',fontsize=7)
for (x0,y0),(x1,y1) in [((0.32,0.75),(0.345,0.75)),((0.655,0.75),(0.68,0.75)),((0.835,0.60),(0.835,0.42)),((0.68,0.27),(0.655,0.27)),((0.345,0.27),(0.32,0.27))]:
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=11,lw=1.4,color='#444'))
ax.text(0.5,0.50,'rules for step 6 are fixed at step 2, before any result is seen',ha='center',fontsize=7.3,color=ORG,style='italic')
fig.savefig('out/fig_ch7_flow.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 7.2 minimal detectable effect
m=e10['mde']; fig,ax=plt.subplots(figsize=(5.8,2.9)); n=np.array(m['n'])
ax.plot(n,m['a05'],color=NAVY,lw=1.5,label='α = 0.05'); ax.plot(n,m['a0167'],color=ORG,lw=1.5,label='α = 0.05 / 3 (three hypotheses)'); ax.plot(n,m['a0083'],color=GRN,lw=1.5,label='α = 0.05 / 6')
for nn in (5,10,20): ax.plot(nn,m['a05'][m['n'].index(nn)],'o',color=NAVY,ms=4); ax.annotate(f"{m['a05'][m['n'].index(nn)]:.2f}",(nn,m['a05'][m['n'].index(nn)]),xytext=(4,5),textcoords='offset points',fontsize=7,color=NAVY)
ax.set_xlabel('number of seeds (or paired replicates) n'); ax.set_ylabel('minimal detectable effect (SD of the difference)'); ax.legend(frameon=False,fontsize=7); ax.set_title('Power 0.8, two-sided paired t test',fontsize=8.5); fig.tight_layout(); fig.savefig('out/fig_ch7_mde.png',dpi=200); plt.close()
# ---- Fig 7.3 forest plot
fig,ax=plt.subplots(1,2,figsize=(6.4,3.8),gridspec_kw={'width_ratios':[1.05,1]})
fam=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']; fl=['AR(1)','tones','piecewise','white']; rows=[];cols=[]
for cmp_,c in [('HT default - STFT',ORG),('HT tuned - STFT',NAVY)]:
    for k,l in zip(fam,fl): rows.append((f'{cmp_.replace(" - "," − ")}: {l}',e10['h2'][cmp_][k],c))
y=np.arange(len(rows))[::-1]
for yi,(lab,(m_,lo,hi),c) in zip(y,rows): ax[0].plot([lo,hi],[yi,yi],color=c,lw=2); ax[0].plot(m_,yi,'o',color=c,ms=4.5)
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows],fontsize=6.6); ax[0].axvline(0,color='#333',lw=1); ax[0].axvline(1,color=GRY,ls='--',lw=1); ax[0].text(1.1,7.35,'+1 dB',fontsize=6.5,color=GRY); ax[0].set_xlabel('SNR difference (dB), 95% CI'); ax[0].set_title('H2: ρ = 8, 20 signals per family',fontsize=8)
rows=[]
for T in ('128','512','1024'):
    rows.append((f'HT-A scaled − Mamba, T = {T}',e10['h1'][f'HT-A (omega scaled by Delta)|{T}'],NAVY))
for T in ('128','512','1024'):
    rows.append((f'HT-A as written − Mamba, T = {T}',e10['h1'][f'HT-A (doc: omega ungated)|{T}'],ORG))
y=np.arange(len(rows))[::-1]; ax[1].axvspan(-0.05,0.05,color='#E2F0D9',alpha=.9,lw=0)
for yi,(lab,(m_,lo,hi),c) in zip(y,rows): ax[1].plot([max(lo,-0.4),min(hi,0.4)],[yi,yi],color=c,lw=2); ax[1].plot(max(min(m_,0.4),-0.4),yi,'o',color=c,ms=4.5)
ax[1].set_yticks(y); ax[1].set_yticklabels([r[0] for r in rows],fontsize=6.4); ax[1].axvline(0,color='#333',lw=1); ax[1].set_xlim(-0.4,0.4); ax[1].set_xlabel('R² difference, 95% CI (n = 5)'); ax[1].set_title('H1 proxy: selective recall',fontsize=8); ax[1].text(0.0,5.4,'shaded: ±0.05',fontsize=6.3,color=GRN,ha='center')
fig.tight_layout(); fig.savefig('out/fig_ch7_forest.png',dpi=200); plt.close()
# ---- Fig 7.4 coverage matrix
tasks=['LRA classification','Forecasting, standard datasets','Audio and EEG, real data','Reconstruction under noise','Memory probes (synthetic)','Multi-scale forecasting (synthetic)','Spectral classification (synthetic)','Selective recall (synthetic)','Cost: latency and memory']
meths=['STFT /\nwavelet','S4D /\nLRU','Mamba-\nstyle','Transformer','HT','HT-A']
S=[['N','N','N','N','N','N'],['N','N','N','N','N','N'],['N','N','N','N','N','N'],['R','-','-','-','R','-'],['R','R','N','N','R','N'],['R','R','N','N','R','N'],['R','R','N','N','R','N'],['-','N','R','N','R','R'],['N','N','N','R','R','R']]
cm={'R':'#A9D18E','N':'#D9D9D9','-':'white'}; lab={'R':'run','N':'not run','-':'n/a'}
fig,ax=plt.subplots(figsize=(6.3,3.6)); ax.set_xlim(0,len(meths)); ax.set_ylim(0,len(tasks)); ax.invert_yaxis(); ax.grid(False); ax.set_frame_on(False)
for i,r in enumerate(S):
    for j,v in enumerate(r): ax.add_patch(Rectangle((j,i),1,1,fc=cm[v],ec='white',lw=2)); ax.text(j+.5,i+.5,lab[v],ha='center',va='center',fontsize=7,color='#222' if v!='-' else '#999')
ax.set_xticks(np.arange(len(meths))+.5); ax.set_xticklabels(meths,fontsize=7.2); ax.xaxis.tick_top(); ax.set_yticks(np.arange(len(tasks))+.5); ax.set_yticklabels(tasks,fontsize=7.2); ax.tick_params(length=0)
fig.tight_layout(); fig.savefig('out/fig_ch7_coverage.png',dpi=200); plt.close()
# ---- Fig 7.5 run counts
p=e10['plan']; names=list(p); fin=[p[k]['final'] for k in names]; sea=[p[k]['search'] for k in names]; fig,ax=plt.subplots(figsize=(6.0,2.7)); y=np.arange(len(names))[::-1]
ax.barh(y,fin,color=NAVY,label='final runs'); ax.barh(y,sea,left=fin,color='#9DB4D8',label='hyperparameter search runs')
for yi,f_,s_ in zip(y,fin,sea): ax.text(f_+s_+15,yi,f'{f_+s_:,}',va='center',fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=7.3); ax.set_xlabel('number of training or measurement runs'); ax.legend(frameon=False,fontsize=7,loc='lower right'); ax.set_xlim(0,1900); fig.tight_layout(); fig.savefig('out/fig_ch7_runs.png',dpi=200); plt.close(); print('ok')
