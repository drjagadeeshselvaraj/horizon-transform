import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR,RED='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0','#C00000'
a=json.load(open('out/e11a.json')); b=json.load(open('out/e11b.json')); c=json.load(open('out/e12.json')); t=json.load(open('out/e13.json')); e2=json.load(open('out/e2.json')); e3=json.load(open('out/e3.json'))
# ---- Fig 8.9 paired memory comparisons
fig,ax=plt.subplots(1,2,figsize=(6.4,2.9),sharey=True)
for a_,metric,xl in [(ax[0],'mc','memory capacity: HT − baseline'),(ax[1],'sfr','scale-free recall: HT − baseline')]:
    rows=a['paired'][metric]; y=np.arange(len(rows))[::-1]
    for yi,r in zip(y,rows):
        col=NAVY if r['p_holm']<0.05 else GRY; a_.plot([r['lo'],r['hi']],[yi,yi],color=col,lw=2); a_.plot(r['mean'],yi,'o',color=col,ms=4.5)
        a_.text(0.995,(yi+0.0)/(len(rows)-1)*0.88+0.04,('p < 0.001' if r['p_holm']<0.001 else f"p = {r['p_holm']:.3f}"),transform=a_.transAxes,ha='right',va='center',fontsize=6,color=col)
    a_.axvline(0,color='#333',lw=1); a_.set_yticks(y); a_.set_yticklabels([r['vs'] for r in rows],fontsize=7); a_.set_xlabel(xl,fontsize=7.5)
ax[0].set_xlim(-37,16); ax[1].set_xlim(-0.05,0.9); ax[0].set_title('5 seeds, 95% t interval, Holm-adjusted p',fontsize=7.8,loc='left')
fig.tight_layout(); fig.savefig('out/fig_ch8x_paired.png',dpi=200); plt.close()
# ---- Fig 8.10 noise sensitivity
nz=a['noise']['levels']; fig,ax=plt.subplots(1,2,figsize=(6.4,2.9)); cs=[NAVY,ORG,GLD,PUR,GRN,'#00B0F0',GRY]
for (k,v),cc in zip(a['noise']['banks'].items(),cs):
    ax[0].semilogx(nz,v['mc'],'o-',color=cc,ms=3.5,lw=1.3,label=k); ax[1].semilogx(nz,v['sfr'],'o-',color=cc,ms=3.5,lw=1.3)
ax[0].set_xlabel('observation noise (fraction of feature SD)'); ax[1].set_xlabel('observation noise (fraction of feature SD)'); ax[0].set_ylabel('memory capacity'); ax[1].set_ylabel('scale-free recall'); ax[0].legend(frameon=False,fontsize=6,loc='center',ncol=2,bbox_to_anchor=(0.5,0.48)); ax[0].set_title('Exact recall',fontsize=8.3); ax[1].set_title('Scale-free recall',fontsize=8.3)
fig.tight_layout(); fig.savefig('out/fig_ch8x_noise.png',dpi=200); plt.close()
# ---- Fig 8.11 feature count
fig,ax=plt.subplots(1,2,figsize=(6.4,2.9)); cs2={'HT default':NAVY,'S4D-Lin':ORG,'LRU (ring)':GLD,'STFT (W=256)':GRN,'Delay line':GRY}
for k,cc in cs2.items(): ax[0].plot(b['nf'],b['banks'][k]['mc'],'o-',color=cc,ms=3.5,lw=1.3,label=k); ax[1].plot(b['nf'],b['banks'][k]['sfr'],'o-',color=cc,ms=3.5,lw=1.3)
ax[0].plot(b['nf'],b['nf'],'--',color='#bbb',lw=1); ax[0].text(70,112,'MC = N',fontsize=6.5,color='#999'); ax[0].set_yscale('log',base=2)
from matplotlib.ticker import NullFormatter
for a_ in ax: a_.set_xlabel('number of real features N'); a_.set_xscale('log',base=2); a_.set_xticks(b['nf']); a_.set_xticklabels([str(n) for n in b['nf']]); a_.xaxis.set_minor_formatter(NullFormatter())
ax[0].set_ylabel('memory capacity'); ax[1].set_ylabel('scale-free recall'); ax[0].legend(frameon=False,fontsize=6.3); ax[0].set_title('Exact recall',fontsize=8.3); ax[1].set_title('Scale-free recall',fontsize=8.3)
fig.tight_layout(); fig.savefig('out/fig_ch8x_nf.png',dpi=200); plt.close()
# ---- Fig 8.12 classification difficulty
fig,ax=plt.subplots(figsize=(5.8,3.0)); amps=c['amps']; sty={'HT, constant-Q':(NAVY,'o'),'HT default':(ORG,'s'),'Morlet bank':(GRN,'^'),'STFT (W = 128)':(PUR,'v'),'LRU':(GLD,'d')}
for k,(cc,mk) in sty.items():
    m=[100*np.mean(c['acc'][k][str(x)]) for x in amps]; s=[100*np.std(c['acc'][k][str(x)]) for x in amps]; ax.errorbar(amps,m,yerr=s,color=cc,marker=mk,ms=4,lw=1.3,capsize=2,label=k)
ax.axhline(100/3,color='k',ls=':',lw=1); ax.text(0.1,34.2,'chance (33%)',fontsize=6.5); ax.set_xlabel('burst amplitude (signal-to-noise increases to the right)'); ax.set_ylabel('accuracy (%)'); ax.legend(frameon=False,fontsize=6.8,loc='upper left'); ax.set_title('3 seeds; amplitude 0.10 is the setting of Section 8.5',fontsize=8)
fig.tight_layout(); fig.savefig('out/fig_ch8x_difficulty.png',dpi=200); plt.close()
# ---- Fig 8.13 timing with five repetitions
Ts=t['Ts']; fig,ax=plt.subplots(1,2,figsize=(6.4,2.9)); cs3={'HT (FFT conv)':NAVY,'HT (recurrence)':GRN,'HT-A (parallel scan)':ORG,'Self-attention':RED}
for k,cc in cs3.items():
    med=np.array([np.median(t['out'][k][str(T)]['t']) for T in Ts])*1e3; lo=np.array([min(t['out'][k][str(T)]['t']) for T in Ts])*1e3; hi=np.array([max(t['out'][k][str(T)]['t']) for T in Ts])*1e3
    ax[0].errorbar(Ts,med,yerr=[med-lo,hi-med],color=cc,marker='o',ms=3.5,lw=1.3,capsize=2,label=k)
ax[0].set_xscale('log',base=2); ax[0].set_yscale('log'); ax[0].set_xlabel('sequence length T'); ax[0].set_ylabel('latency (ms), median and range of 5'); ax[0].legend(frameon=False,fontsize=6.5)
for k,cc in cs3.items():
    if k=='Self-attention': continue
    r=[np.median(t['out']['Self-attention'][str(T)]['t'])/np.median(t['out'][k][str(T)]['t']) for T in Ts]; ax[1].semilogx(Ts,r,'o-',color=cc,ms=3.5,lw=1.3,label=k)
ax[1].set_yscale('log'); ax[1].axhline(1,color='k',lw=1); ax[1].axhline(1.1,color=GRY,ls='--',lw=1); ax[1].set_xscale('log',base=2); ax[1].set_xlabel('sequence length T'); ax[1].set_ylabel('attention time / HT time'); ax[1].text(260,1.2,'10% margin',fontsize=6.5,color=GRY); ax[1].legend(frameon=False,fontsize=6.3,loc='upper left')
fig.tight_layout(); fig.savefig('out/fig_ch8x_timing.png',dpi=200); plt.close()
# ---- Fig 8.14 rank heatmap (min-max normalised score per column; best = dark)
fc=e3['forecast']; cl=e3['classify']
rows=[('HT default','HT (geometric, default)','HT (geometric, default)','HT (geometric, default)','HT (geometric, default)'),('S4D-Lin','S4D-Lin (HiPPO-style)','S4D-Lin',None,None),('LRU (ring)','LRU (ring init)','LRU','LRU','LRU'),('HiPPO-LegT','HiPPO-LegT (W=4096)','HiPPO-LegT (W=4096)',None,None),('STFT','STFT (W=256)','STFT (W=256, low band)',"STFT (W=128, linear bins)",None),('Morlet bank','CWT (causal Morlet)','CWT (causal Morlet)','CWT (Morlet, log freqs)',None),('Delay line','Delay line (64 taps)','Delay line (64 taps)',None,None)]
cols=['Memory capacity','Scale-free recall','Forecast NMSE\n(H = 24)','Classification\naccuracy']
M=np.full((len(rows),4),np.nan); V=[['']*4 for _ in rows]
for i,(lab,k2,kf,kc,_) in enumerate(rows):
    M[i,0]=e2[k2]['mc'][0]; V[i][0]=f"{M[i,0]:.0f}"; M[i,1]=e2[k2]['sfr'][0]; V[i][1]=f"{M[i,1]:.2f}"; M[i,2]=fc[kf]['mean'][1]; V[i][2]=f"{M[i,2]:.2f}"
    if kc: M[i,3]=cl[kc]['mean']*100; V[i][3]=f"{M[i,3]:.0f}"
    for j in range(4):
        if np.isnan(M[i,j]): V[i][j]='—'
S=np.full_like(M,np.nan)
for j in range(4):
    col=M[:,j]; ok=~np.isnan(col); lo,hi=np.nanmin(col),np.nanmax(col); s=(col-lo)/(hi-lo+1e-12); S[:,j]=(1-s) if j==2 else s
fig,ax=plt.subplots(figsize=(6.0,3.1)); im=ax.imshow(np.ma.masked_invalid(S),cmap='Blues',vmin=0,vmax=1.15,aspect='auto'); ax.grid(False)
for i in range(len(rows)):
    for j in range(4): ax.text(j,i,V[i][j],ha='center',va='center',fontsize=7.5,color='white' if (not np.isnan(S[i,j]) and S[i,j]>0.6) else '#222')
ax.set_xticks(range(4)); ax.set_xticklabels(cols,fontsize=7.2); ax.xaxis.tick_top(); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows],fontsize=7.5); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.tight_layout(); fig.savefig('out/fig_ch8x_rank.png',dpi=200); plt.close(); print('ok')
