import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR,RED='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0','#C00000'
e2=json.load(open('out/e2.json')); e3=json.load(open('out/e3.json')); e4=json.load(open('out/e4.json')); e6=json.load(open('out/e6.json')); e13=json.load(open('out/e13.json')); e14=json.load(open('out/e14.json')); e5=json.load(open('out/e5.json'))
K4=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']
# ---- Fig 9.1 classification grid
items=[('S1 low cost at long context',['impl','struct']),('S2 exact inverse, stable gain',['struct']),('S3 every timescale from few channels',['design','struct']),('S4 phase-aware complex poles',['struct','design']),('S5 content-adaptive memory',['design','evid']),('S6 simple and interpretable',['struct']),
('W1 no demonstrated accuracy advantage',['evid']),('W2 weak exact recall, low capacity',['struct','design']),('W3 reconstruction behind STFT, wavelet',['design']),('W4 pooled inverse suboptimal',['design']),('W5 redundancy costs bits',['struct']),('W6 HT-A unreliable as written, unproven',['design','evid']),('W7 HT-A cost advantage not shown',['impl']),('W8 sensitive, under-specified design',['design']),('W9 thin evidence base',['evid'])]
cols=['structural','design-dependent','implementation-\ndependent','evidence-limited']; key={'struct':0,'design':1,'impl':2,'evid':3}
fig,ax=plt.subplots(figsize=(6.3,4.6)); ax.set_xlim(0,4); ax.set_ylim(0,len(items)+0.0); ax.invert_yaxis(); ax.grid(False); ax.set_frame_on(False)
for i,(lab,cs) in enumerate(items):
    ax.add_patch(Rectangle((0,i),4,1,fc='#EAF3E3' if lab[0]=='S' else '#FBEADF',ec='white',lw=1.5,zorder=0))
    for k,c in enumerate(cs):
        x=key[c]+0.5; ax.plot(x,i+0.5,'o',ms=9,mfc=NAVY if k==0 else 'white',mec=NAVY,mew=1.5,zorder=3)
ax.set_yticks(np.arange(len(items))+.5); ax.set_yticklabels([t for t,_ in items],fontsize=7.2); ax.set_xticks(np.arange(4)+.5); ax.set_xticklabels(cols,fontsize=7.4); ax.xaxis.tick_top(); ax.tick_params(length=0)
for x in range(1,4): ax.axvline(x,color='white',lw=2,zorder=1)
ax.plot([],[],'o',mfc=NAVY,mec=NAVY,ms=7,label='primary cause'); ax.plot([],[],'o',mfc='white',mec=NAVY,ms=7,label='secondary'); ax.legend(frameon=False,fontsize=7,loc='lower center',bbox_to_anchor=(0.5,-0.08),ncol=2)
fig.tight_layout(); fig.savefig('out/fig_ch9_grid.png',dpi=200); plt.close()
# ---- Fig 9.2 profile against the best non-HT method
def avg(nm,dec='LS (unweighted)'): return np.mean([e6[nm]['8'][k][dec] for k in K4])
rec_best=max(avg('STFT (W=128)'),avg('Undecimated wavelet (db2)')); mc=lambda k:e2[k]['mc'][0]; sf=lambda k:e2[k]['sfr'][0]
mc_best=max(mc(k) for k in ['S4D-Lin (HiPPO-style)','LRU (ring init)','HiPPO-LegT (W=4096)','STFT (W=256)','STFT (W=1024)','CWT (causal Morlet)','Delay line (64 taps)']); sf_best=max(sf(k) for k in ['S4D-Lin (HiPPO-style)','LRU (ring init)','HiPPO-LegT (W=4096)','STFT (W=256)','STFT (W=1024)','CWT (causal Morlet)','Delay line (64 taps)'])
fc=e3['forecast']; fbest=min(fc[k]['mean'][1] for k in ['S4D-Lin','LRU','HiPPO-LegT (W=4096)','STFT (W=256, low band)','CWT (causal Morlet)','Delay line (64 taps)']); cl=e3['classify']; cbest=max(cl[k]['mean'] for k in ['CWT (Morlet, log freqs)','STFT (W=128, linear bins)','LRU'])
T='8192'; att_t=np.median(e13['out']['Self-attention'][T]['t']); att_m=e13['out']['Self-attention'][T]['mem']
rows=[('Reconstruction SNR, ρ = 8','HT default',avg('HT default')/rec_best,NAVY),('Reconstruction SNR, ρ = 8','HT tuned',avg('HT tuned')/rec_best,ORG),
('Memory capacity','HT default',mc('HT (geometric, default)')/mc_best,NAVY),('Memory capacity','HT orthogonalized',e14['res']['HT, orthogonalized']['0.01']['mc'][0]/mc_best,GRN),
('Scale-free recall','HT default',sf('HT (geometric, default)')/sf_best,NAVY),('Scale-free recall','HT orthogonalized',e14['res']['HT, orthogonalized']['0.01']['sfr'][0]/sf_best,GRN),
('Forecast error, H = 24','HT default',fbest/fc['HT (geometric, default)']['mean'][1],NAVY),('Classification accuracy','HT default',cl['HT (geometric, default)']['mean']/cbest,NAVY),('Classification accuracy','HT constant-Q',cl['HT (freq-tuned, constant-Q)']['mean']/cbest,ORG),
('Latency, T = 8,192','HT, FFT form',att_t/np.median(e13['out']['HT (FFT conv)'][T]['t']),NAVY),('Latency, T = 8,192','HT, recurrence',att_t/np.median(e13['out']['HT (recurrence)'][T]['t']),ORG),('Latency, T = 8,192','HT-A, scan',att_t/np.median(e13['out']['HT-A (parallel scan)'][T]['t']),RED),('Peak memory, T = 8,192','HT, FFT form',att_m/e13['out']['HT (FFT conv)'][T]['mem'],NAVY)]
fig,ax=plt.subplots(figsize=(6.3,3.9)); y=np.arange(len(rows))[::-1]
for yi,(c,v,r,col) in zip(y,rows): ax.barh(yi,np.log2(r),color=col,height=0.7); ax.text(np.log2(r)+(0.12 if r>=1 else -0.12),yi,f'{r:.2f}' if r<10 else f'{r:.0f}',va='center',ha='left' if r>=1 else 'right',fontsize=7)
ax.axvline(0,color='#333',lw=1); ax.set_yticks(y); ax.set_yticklabels([f'{c}: {v}' for c,v,_,_ in rows],fontsize=6.9); ax.set_xticks([-2,-1,0,1,2,3,4,5]); ax.set_xticklabels(['¼','½','1','2','4','8','16','32']); ax.set_xlim(-2.4,6); ax.set_xlabel('ratio to the best non-HT method (log scale; above 1 means HT is better)')
fig.tight_layout(); fig.savefig('out/fig_ch9_profile.png',dpi=200); plt.close()
# ---- Fig 9.3 sensitivity of scale-free recall to design choices
ab=e2['_ablation']; base=e2['HT (geometric, default)']['sfr'][0]
fac=[('Features K',[('K: nfeat=16','16'),('K: nfeat=32','32'),('K: nfeat=128','128')]),('Frequencies per scale F',[('F: 1 freq/scale','1'),('F: 4 freq/scale','4'),('F: 8 freq/scale','8')]),('Longest horizon',[('hmax=256','256'),('hmax=1024','1024'),('hmax=16384','16384')]),('Spacing ratio r',[('ratio r=1.25','1.25'),('ratio r=1.5','1.5'),('ratio r=3.0','3')]),('Pole type',[('real poles (64 real)','real only')]),('Poles learned',[('learned/tuned','tuned')])]
fig,ax=plt.subplots(figsize=(6.3,3.2)); y=np.arange(len(fac))[::-1]
for yi,(name,lv) in zip(y,fac):
    ds=[ab[k]['sfr'][0]-base for k,_ in lv]; ax.plot([min(ds+[0]),max(ds+[0])],[yi,yi],color='#bbb',lw=3,zorder=1)
    for d_,(k,l) in zip(ds,lv): ax.plot(d_,yi,'o',color=NAVY if d_>=0 else ORG,ms=6,zorder=3); ax.text(d_,yi+0.28,l,ha='center',fontsize=6.4,color='#333')
ax.axvline(0,color='#333',lw=1); ax.set_yticks(y); ax.set_yticklabels([f[0] for f in fac],fontsize=7.5); ax.set_xlabel(f'change in scale-free recall from the default ({base:.2f})'); ax.set_ylim(-0.6,len(fac)-0.3)
fig.tight_layout(); fig.savefig('out/fig_ch9_sens.png',dpi=200); plt.close()
# ---- Fig 9.4 mitigation scatter
fig,ax=plt.subplots(figsize=(6.0,3.7)); pts={'HT default':(mc('HT (geometric, default)'),sf('HT (geometric, default)'),NAVY,'o'),'HT tuned (free poles)':(mc('HT tuned (free h, w; init=HT)'),sf('HT tuned (free h, w; init=HT)'),ORG,'s'),'S4D-Lin':(mc('S4D-Lin (HiPPO-style)'),sf('S4D-Lin (HiPPO-style)'),GLD,'^'),'LRU':(mc('LRU (ring init)'),sf('LRU (ring init)'),GRY,'v'),'STFT (W=256)':(mc('STFT (W=256)'),sf('STFT (W=256)'),GRN,'d'),'Delay line':(mc('Delay line (64 taps)'),sf('Delay line (64 taps)'),PUR,'x')}
r=e14['res']
for k,(m,s,c,mk) in pts.items(): ax.scatter(m,s,s=34,color=c,marker=mk,zorder=3); ax.annotate(k,(m,s),xytext=((-52,6) if k=='HT default' else (-4,12) if k.startswith('HT tuned') else (5,4)),textcoords='offset points',fontsize=6.8,color=c)
mit={'HT, orthogonalized':('orthogonalized',RED,'*'),'HT 32 + delay line 32':('HT 32 + delay 32',RED,'P'),'HT 48 + delay line 16':('HT 48 + delay 16',RED,'P'),'HT 32 + Legendre window 32':('HT 32 + Legendre 32',RED,'P'),'HT 32 + STFT 32':('HT 32 + STFT 32',RED,'P')}
OFF={'HT, orthogonalized':(8,6),'HT 32 + delay line 32':(8,-2),'HT 48 + delay line 16':(-62,8),'HT 32 + Legendre window 32':(-30,-14),'HT 32 + STFT 32':(6,-12)}
for k,(lab,c,mk) in mit.items():
    m,s=r[k]['0.01']['mc'][0],r[k]['0.01']['sfr'][0]; ax.scatter(m,s,s=46 if mk!='*' else 90,color=c,marker=mk,zorder=4); ax.annotate(lab,(m,s),xytext=OFF[k],textcoords='offset points',fontsize=6.6,color=c)
    ax.add_patch(FancyArrowPatch((pts['HT default'][0],pts['HT default'][1]),(m,s),arrowstyle='-|>',mutation_scale=7,lw=.8,color=c,alpha=.6,zorder=2))
ax.set_xlabel('memory capacity (exact recall, 64 features)'); ax.set_ylabel('scale-free recall'); ax.set_xlim(20,66); ax.set_ylim(0,1.0); fig.tight_layout(); fig.savefig('out/fig_ch9_mitigation.png',dpi=200); plt.close()
# ---- Fig 9.5 closing the gaps
fig,ax=plt.subplots(1,4,figsize=(6.5,3.2))
def bars(a,labs,vals,cols,ref=None,title='',ylab='',ylim=None):
    a.bar(range(len(vals)),vals,color=cols,width=0.65)
    for i,v in enumerate(vals): a.text(i,v,(f'{v:.1f}' if v>=5 else f'{v:.2f}'),ha='center',va='bottom',fontsize=6.3)
    for (rv,rl,rc) in (ref or []): a.axhline(rv,color=rc,ls='--',lw=1); a.text(-0.45,rv,rl,fontsize=6,color=rc,va='bottom',ha='left')
    a.set_xticks(range(len(vals))); a.set_xticklabels(labs,fontsize=6.2,rotation=40,ha='right'); a.set_title(title,fontsize=7.6); a.set_ylabel(ylab,fontsize=7)
    if ylim: a.set_ylim(*ylim)
bars(ax[0],['default','learned','orthog.','hybrid'],[mc('HT (geometric, default)'),mc('HT tuned (free h, w; init=HT)'),r['HT, orthogonalized']['0.01']['mc'][0],r['HT 32 + delay line 32']['0.01']['mc'][0]],[NAVY,ORG,GRN,GRN],ref=[(mc('STFT (W=256)'),'STFT',GRY)],title='Exact recall (MC)',ylab='memory capacity',ylim=(0,70))
rd=lambda nm,dec: avg(nm,dec)
bars(ax[1],['def. LS','def. wtd','tuned LS','tuned wtd'],[rd('HT default','LS (unweighted)'),rd('HT default','GLS (weighted LS)'),rd('HT tuned','LS (unweighted)'),rd('HT tuned','GLS (weighted LS)')],[NAVY,GRN,ORG,GRN],ref=[(rd('STFT (W=128)','GLS (weighted LS)'),'STFT',GRY)],title='Reconstruction, ρ = 8',ylab='output SNR (dB)',ylim=(20,34))
med=lambda m,T_: float(np.median(e4[m][T_]))
bars(ax[2],['as written','scaled'],[med('HT-A (doc: omega ungated)','1024'),med('HT-A (omega scaled by Delta)','1024')],[ORG,GRN],ref=[(med('Selective real-pole SSM (Mamba-style)','1024'),'Mamba-style',GRY)],title='HT-A, T = 1024',ylab='median R²',ylim=(0,1.25))
bars(ax[3],['default','constant-Q'],[100*cl['HT (geometric, default)']['mean'],100*cl['HT (freq-tuned, constant-Q)']['mean']],[NAVY,GRN],ref=[(100*cl['CWT (Morlet, log freqs)']['mean'],'Morlet',GRY)],title='Classification',ylab='accuracy (%)',ylim=(30,72))
fig.tight_layout(); fig.savefig('out/fig_ch9_gaps.png',dpi=200); plt.close(); print('ok')
