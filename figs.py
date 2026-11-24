import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lib import LAGS, SF_D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25})
e1=json.load(open('out/e1.json')); e2=json.load(open('out/e2.json')); e5=json.load(open('out/e5.json'))
K=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']; T=['AR(1) lowpass','Tones + chirp + impulses','Piecewise-constant','White noise']
# Fig 1 reconstruction vs rho
sty={'HT (default) - LS inverse':('#1F3864','-','o'),'HT (default) - paper inverse':('#1F3864','--','o'),'HT (tuned) - LS inverse':('#C55A11','-','s'),'HT (tuned) - paper inverse':('#C55A11','--','s'),'STFT (W=128)':('#548235','-','^'),'Undecimated wavelet (db2)':('#7F7F7F','-','v')}
lab={'HT (default) - LS inverse':'HT default, LS inverse','HT (default) - paper inverse':'HT default, paper inverse (Ch. 4)','HT (tuned) - LS inverse':'HT tuned, LS inverse','HT (tuned) - paper inverse':'HT tuned, paper inverse','STFT (W=128)':'STFT (W=128)','Undecimated wavelet (db2)':'Undecimated wavelet (db2)'}
fig,ax=plt.subplots(2,2,figsize=(6.3,4.8),sharex=True)
for a,k,t in zip(ax.ravel(),K,T):
    for nm,(c,ls,m) in sty.items():
        d=e1['res'][nm]; rs=[r for r in ('2','4','8','16') if r in d]; a.plot([int(r) for r in rs],[d[r][k]['chan20'] for r in rs],color=c,ls=ls,marker=m,ms=4,lw=1.3,label=lab[nm])
    a.set_title(t,fontsize=8.5); a.set_xscale('log',base=2); a.set_xticks([2,4,8,16]); a.set_xticklabels(['2','4','8','16'])
for a in ax[1]: a.set_xlabel('real coefficients per sample (ρ)')
for a in ax[:,0]: a.set_ylabel('output SNR (dB)')
h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=7); fig.tight_layout(rect=(0,.09,1,1)); fig.savefig('out/fig1.png',dpi=200); plt.close()
# Fig 2 memory
show=['HT (geometric, default)','S4D-Lin (HiPPO-style)','LRU (ring init)','HiPPO-LegT (W=4096)','STFT (W=256)','CWT (causal Morlet)','Delay line (64 taps)','HT real poles only']
cols=['#1F3864','#C55A11','#BF9000','#7030A0','#548235','#00B0F0','#7F7F7F','#C00000']
fig,ax=plt.subplots(1,2,figsize=(6.3,2.7))
for nm,c in zip(show,cols):
    ls='-' if nm.startswith('HT (') else '--'; ax[0].plot(LAGS,e2[nm]['mc_curve'],color=c,ls=ls,lw=1.3,label=nm.replace(' (geometric, default)','').replace(' (HiPPO-style)',''))
    ax[1].plot(SF_D,e2[nm]['sfr_curve'],color=c,ls=ls,lw=1.3)
for a in ax: a.set_xscale('log'); a.set_ylim(-.02,1.03); a.set_xlabel('lag d (samples)')
ax[0].set_ylabel('R² (held-out)'); ax[0].set_title('Exact recall of u[t−d]',fontsize=8.5); ax[1].set_title('Scale-free recall (window ≈ d/4 at lag d)',fontsize=8.5)
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=6.5); fig.tight_layout(rect=(0,.17,1,1)); fig.savefig('out/fig2.png',dpi=200); plt.close()
# Fig 3 complexity
Ts=e5['T']; fig,ax=plt.subplots(1,2,figsize=(6.3,2.7)); cs={'HT (FFT conv)':'#1F3864','HT (recurrence, lfilter)':'#548235','HT-A (parallel scan)':'#C55A11','Self-attention (1 head, fp32)':'#C00000'}
for k,c in cs.items():
    for j,key in enumerate(('t','mem')):
        xs=[t for t,v in zip(Ts,e5[k][key]) if v is not None]; ys=[(v*1e3 if key=='t' else max(v,0.05)) for v in e5[k][key] if v is not None]; ax[j].plot(xs,ys,color=c,marker='o',ms=3,lw=1.3,label=k)
for a,yl in zip(ax,('latency (ms)','peak memory (MiB)')): a.set_xscale('log',base=2); a.set_yscale('log'); a.set_xlabel('sequence length T'); a.set_ylabel(yl)
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=7); fig.tight_layout(rect=(0,.15,1,1)); fig.savefig('out/fig3.png',dpi=200); plt.close()
print('ok')
