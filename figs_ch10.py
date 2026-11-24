import numpy as np, json, textwrap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR,RED='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0','#C00000'
a=json.load(open('out/e15a.json')); b=json.load(open('out/e15b.json')); c=json.load(open('out/e15c.json'))
# ---- Fig 10.1 envelope of tested conditions
rows=[('Sequence length T (samples)',128,131072,'selective 128–1,024; reconstruction 256; memory stream 40,000; cost scaling to 131,072',NAVY),('Longest horizon (samples)',256,16384,'ablation of Section 8.8',NAVY),('Real features per layer',16,128,'64 in most experiments',GRN),('Observation noise (fraction of SD)',1e-6,1e-1,'1e-6 only in the precision test',ORG),('Seeds per comparison',3,5,'one search seed for tuned variants',PUR),('Input channels',1,1,'scalar input only',RED)]
lims={'Seq':(0.7,2e6),'Lon':(0.7,2e6),'Rea':(8,512),'Obs':(1e-7,1),'See':(1,100),'Inp':(0.7,1000)}
fig,ax=plt.subplots(len(rows),1,figsize=(6.3,3.7))
for a_,(lab,lo,hi,note,col) in zip(ax,rows):
    a_.set_xscale('log'); a_.set_xlim(*lims[lab[:3]]); a_.set_ylim(-0.6,1.0)
    if hi>lo: a_.hlines(0,lo,hi,lw=11,color=col,alpha=.9,zorder=3)
    else: a_.plot([lo],[0],'s',color=col,ms=9,zorder=3)
    a_.set_yticks([]); a_.set_ylabel(lab,rotation=0,ha='right',va='center',fontsize=7.2); a_.grid(False); a_.spines['left'].set_visible(False)
    a_.text(a_.get_xlim()[1]*0.9,0.55,note,ha='right',va='center',fontsize=6,color='#444'); a_.tick_params(labelsize=6.3)
fig.tight_layout(); fig.savefig('out/fig_ch10_envelope.png',dpi=200); plt.close()
# ---- Fig 10.2 nonlinear targets
nl=a['nonlinear']; tg=list(next(iter(nl.values())).keys()); banks=list(nl); fig,ax=plt.subplots(1,2,figsize=(6.4,2.8),sharey=True); wd=0.2; cs=[NAVY,ORG,GRN,GRY]
for a_,ro,tt in zip(ax,['lin','nl'],['Linear readout','Second-order readout']):
    for j,(k,cc) in enumerate(zip(banks,cs)): a_.bar(np.arange(len(tg))+(j-1.5)*wd,[max(nl[k][t][ro][0],0) for t in tg],wd,color=cc,label=k)
    a_.set_xticks(range(len(tg))); a_.set_xticklabels(tg,fontsize=6.6); a_.set_title(tt,fontsize=8.3); a_.set_ylim(0,1.12)
ax[0].set_ylabel('held-out R² of the nonlinear target'); ax[0].text(1,0.5,'R² ≈ 0 for every bank',ha='center',fontsize=8,color='#444'); ax[1].legend(frameon=False,fontsize=6.4,loc='upper right')
fig.tight_layout(); fig.savefig('out/fig_ch10_nonlinear.png',dpi=200); plt.close()
# ---- Fig 10.3 input distribution
kinds=b['kinds']; res=b['res']; banks=list(res); fig,ax=plt.subplots(1,2,figsize=(6.4,2.9)); wd=0.13; cs=[NAVY,ORG,GLD,GRN,'#00B0F0',GRY]
for j,(k,cc) in enumerate(zip(banks,cs)):
    ax[0].bar(np.arange(4)+(j-2.5)*wd,[res[k][d]['mc'][0] for d in kinds],wd,color=cc,label=k); ax[1].bar(np.arange(4)+(j-2.5)*wd,[res[k][d]['sfr'][0] for d in kinds],wd,color=cc)
for a_,t in zip(ax,['Memory capacity (log scale)','Scale-free recall']): a_.set_xticks(range(4)); a_.set_xticklabels(['Gaussian','uniform','Student-t','AR(1) 0.9'],fontsize=7); a_.set_title(t,fontsize=8.3)
ax[0].set_yscale('log'); ax[0].set_ylim(15,800); ax[0].axhline(64,color='k',ls=':',lw=1); ax[0].text(-0.45,70,'bound for white input: 64',fontsize=6,color='#444'); ax[0].legend(frameon=False,fontsize=5.8,ncol=2,loc='upper left')
fig.tight_layout(); fig.savefig('out/fig_ch10_inputs.png',dpi=200); plt.close()
# ---- Fig 10.4 precision
vs=['float64 recursion, float64 mixing','float32 recursion, float64 mixing','float32 recursion, float32 mixing','default bank, float32 recursion']; lab=['orth.\nfloat64','orth.\nf32 recursion','orth.\nf32 all','default\nf32']
fig,ax=plt.subplots(1,2,figsize=(6.4,2.8)); w_=0.36
for a_,m,t in zip(ax,['mc','sfr'],['Memory capacity','Scale-free recall']):
    for j,(nz,cc) in enumerate([('0.01',NAVY),('1e-06',ORG)]): a_.bar(np.arange(4)+(j-0.5)*w_,[c[f'{v}|{float(nz)}'][m] for v in vs],w_,color=cc,label=f'noise {float(nz):g}')
    a_.set_xticks(range(4)); a_.set_xticklabels(lab,fontsize=6.6); a_.set_title(t,fontsize=8.3)
ax[0].legend(frameon=False,fontsize=6.4,loc='lower right'); ax[0].set_ylim(0,52); ax[1].set_ylim(0,1.05)
fig.tight_layout(); fig.savefig('out/fig_ch10_precision.png',dpi=200); plt.close()
# ---- Fig 10.5 open questions by what is needed
cols=[('Answerable with\nfeature banks',GRN,['Does orthogonalization help beyond white-noise memory probes?','Which pole placement maximizes capacity at a given noise level?','Do higher-order (gammatone) cascades improve selectivity?','Can HT-A use Δ for irregular sampling?','Do the conclusions hold for real, non-white inputs?']),('Needs trained models\nor a GPU',ORG,['Does HT-A match selective SSMs on Long Range Arena?','How should frequencies be placed for EEG and audio?','Does the cost advantage survive fused GPU kernels?','How does HT behave in deep stacks with nonlinearities?','Can the bank be learned with the exact inverse retained?']),('Needs theory',NAVY,['Is the HT-A inverse unique for realistic gates?','What is the optimal bank under noise (Laguerre, Kautz, or other)?','Which operators can HT with a nonlinear readout approximate?','Tight bounds for HT-A stability and frame constants'])]
fig,ax=plt.subplots(figsize=(6.4,3.9)); ax.set_xlim(0,3); ax.set_ylim(0,1); ax.axis('off'); ax.grid(False)
for j,(t,cc,qs) in enumerate(cols):
    ax.add_patch(FancyBboxPatch((j+0.03,0.02),0.94,0.96,boxstyle='round,pad=0.005,rounding_size=0.02',fc='white',ec=cc,lw=1.6)); ax.text(j+0.5,0.915,t,ha='center',va='center',fontsize=8,fontweight='bold',color=cc)
    for i,q in enumerate(qs): ax.add_patch(FancyBboxPatch((j+0.06,0.74-i*0.165),0.88,0.13,boxstyle='round,pad=0.003,rounding_size=0.012',fc=cc,ec='none',alpha=0.10)); ax.text(j+0.5,0.805-i*0.165,textwrap.fill(q,31),ha='center',va='center',fontsize=6.0,linespacing=1.15)
fig.savefig('out/fig_ch10_questions.png',dpi=200,bbox_inches='tight'); plt.close(); print('ok')
