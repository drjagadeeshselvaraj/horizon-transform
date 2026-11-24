import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0'
e9=json.load(open('out/e9.json'))
# ---- Fig 6.1 relation map
fig,ax=plt.subplots(figsize=(6.4,4.4)); ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.axis('off'); ax.grid(False)
ax.add_patch(FancyBboxPatch((0.39,0.42),0.22,0.16,boxstyle='round,pad=0.005,rounding_size=0.02',fc='#EAF0FA',ec=NAVY,lw=1.8)); ax.text(0.5,0.50,'HT channel\n$\\lambda=e^{-1/h+i\\omega}$',ha='center',va='center',fontsize=8.5,fontweight='bold',color=NAVY)
nodes=[(0.01,0.80,'Fourier / DFT','$h\\to\\infty$ at the last sample',NAVY,'Eq. 6.1'),(0.37,0.88,'Sliding DFT, SWIFT','identical up to demodulation',GRN,'Eq. 6.2'),(0.73,0.80,'Constant-Q, wavelets','same Q, one-sided window',PUR,'Eq. 6.3'),
       (0.01,0.42,'Gammatone, gamma','cascade of $n$ channels',GLD,'Eq. 6.4'),(0.73,0.42,'S4D-Lin','harmonic $\\omega$ per horizon',ORG,'Eq. 6.5'),
       (0.01,0.04,'Linear attention','kernel $\\lambda^{t-s}$, fixed',GRY,'Eq. 6.6'),(0.37,0.04,'Mamba-type SSM','HT-A with $\\omega=0$',ORG,'Sec. 6.6'),(0.73,0.04,'Softmax attention,\ndilated convolution','no equivalence',"#C00000",'Sec. 6.7, 6.8')]
for x,y,t,d,c,e in nodes:
    ax.add_patch(FancyBboxPatch((x,y),0.26,0.13,boxstyle='round,pad=0.004,rounding_size=0.012',fc='white',ec=c,lw=1.3)); ax.text(x+0.13,y+0.095,t,ha='center',va='center',fontsize=7.3,fontweight='bold',color=c); ax.text(x+0.13,y+0.04,d+f'  ({e})',ha='center',va='center',fontsize=6.2)
    cx=x+0.13; cy=y+(0.0 if y>0.5 else 0.13) if y!=0.42 else y+0.065; tx=0.39 if x<0.3 else (0.61 if x>0.6 else 0.5); ty=0.58 if y>0.5 else (0.42 if y<0.3 else 0.50)
    if abs(x-0.01)<1e-6 and y==0.42: cx=0.27; cy=0.485
    if abs(x-0.73)<1e-6 and y==0.42: cx=0.73; cy=0.485
    ls='--' if c=="#C00000" else '-'; ax.add_patch(FancyArrowPatch((cx,cy),(tx,ty),arrowstyle='-|>',mutation_scale=8,lw=1,color=c,ls=ls,alpha=.85))
fig.savefig('out/fig_ch6_map.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 6.2 DFT limit
d=e9['dft']; h=np.array(d['h']); fig,ax=plt.subplots(figsize=(5.6,2.8)); ax.loglog(h,d['err'],'o-',color=NAVY,ms=4,lw=1.3,label='HT at the last sample vs DFT, Eq. (6.1)'); ax.loglog(h,np.minimum(1,d['T']/(2*h)),'--',color=ORG,lw=1,label='T / (2h)')
ax.set_xlabel('horizon h (samples)'); ax.set_ylabel('relative difference'); ax.legend(frameon=False,fontsize=7); ax.set_title(f"T = {d['T']}, frequencies 2πk/T for k = 1 … 59",fontsize=8.5); fig.tight_layout(); fig.savefig('out/fig_ch6_dft.png',dpi=200); plt.close()
# ---- Fig 6.3 spectrograms
sp=e9['spec']; t=np.array(sp['t']); fr=np.array(sp['freqs']); sf=np.array(sp['stft_f']); fig,ax=plt.subplots(4,1,figsize=(6.2,6.4),sharex=True,gridspec_kw={'height_ratios':[0.7,1.3,1.3,1.3]})
ax[0].plot(t,sp['x'],color='#333',lw=.7); ax[0].set_ylabel('input'); ax[0].text(0.3,4.5,'chirp 3 → 30 Hz',fontsize=7,color=GRY); ax[0].text(2.7,4.5,'12 Hz burst',fontsize=7,color=GRY); ax[0].annotate('impulse',(1.5,6),xytext=(1.6,6.5),fontsize=7,color=GRY); ax[0].set_ylim(-8,8)
for a,M,fq,name in [(ax[1],np.array(sp['ht']),fr,'HT bank (constant-Q rule, causal, 48 channels)'),(ax[2],np.array(sp['cwt']),fr,'Morlet bank (causal, delayed)'),(ax[3],np.array(sp['stft']),sf.astype(float),'STFT, Hann window 128 (causal, 1 Hz bins)')]:
    Z=np.sqrt(M.T/M.max()); a.pcolormesh(t,fq,Z,shading='nearest',cmap='magma',rasterized=True); a.set_yscale('log'); a.set_ylim(2,45); a.set_ylabel('Hz'); a.set_title(name,fontsize=7.8,loc='left'); a.grid(False)
ax[3].set_xlabel('time (s)'); ax[3].set_yticks([2,5,10,20,40]); ax[3].set_yticklabels(['2','5','10','20','40']); fig.tight_layout(); fig.savefig('out/fig_ch6_spectro.png',dpi=200); plt.close()
# ---- Fig 6.4 Q
Q=e9['Q']; fig,ax=plt.subplots(figsize=(6.0,3.1))
for (nm,(w,q)),c,mk in zip(Q.items(),[NAVY,GRN,ORG,GLD],['o','^','s','x']): ax.scatter(w,q,s=16,color=c,marker=mk,label=nm,alpha=.85)
ww=np.geomspace(1e-3,3.1,50); ax.plot(ww,ww*e9['Q_stft_slope'],color=GRY,ls='--',lw=1.2,label='STFT, Hann W = 128'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.1,300); ax.set_xlabel('channel frequency ω (rad/sample)'); ax.set_ylabel('quality factor Q = ω h / 2'); ax.legend(frameon=False,fontsize=6.8,loc='upper left'); fig.tight_layout(); fig.savefig('out/fig_ch6_q.png',dpi=200); plt.close()
# ---- Fig 6.5 cascade
c=e9['cascade']; fig,ax=plt.subplots(1,2,figsize=(6.3,2.6),gridspec_kw={'width_ratios':[1.5,1]})
for n,cc in zip(['1','2','3','4'],plt.cm.viridis(np.linspace(0,.85,4))):
    y=np.array(c[n]['re']); ax[0].plot(y/np.abs(y).max(),color=cc,lw=1.2,label=f'n = {n}')
ax[0].set_xlabel('lag (samples)'); ax[0].set_ylabel('real part, scaled to unit peak'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title('Impulse response of n cascaded channels',fontsize=8.3)
ax[1].semilogy([1,2,3,4],[max(c[n]['err'],1e-17) for n in ['1','2','3','4']],'o-',color=NAVY,ms=4); ax[1].axhline(2.2e-16,color=GRY,ls='--',lw=1); ax[1].text(1.05,3e-16,'machine epsilon',fontsize=7,color=GRY); ax[1].set_ylim(1e-17,1e-13); ax[1].set_xticks([1,2,3,4]); ax[1].set_xlabel('number of cascaded channels n'); ax[1].set_ylabel('relative error vs Eq. (6.4)'); ax[1].set_title('Closed form is exact',fontsize=8.3)
fig.tight_layout(); fig.savefig('out/fig_ch6_cascade.png',dpi=200); plt.close()
# ---- Fig 6.6 mixing matrices
mx=e9['mix']; fig,ax=plt.subplots(1,4,figsize=(6.4,1.95))
for a,k,tt in zip(ax,['ht','multi','hta','attn'],['HT channel (h = 16)','HT, six scales','HT-A, gated step','softmax attention']):
    a.imshow(np.array(mx[k]),cmap='viridis',aspect='equal',origin='upper'); a.set_title(tt,fontsize=7.3); a.set_xticks([]); a.set_yticks([]); a.grid(False)
ax[0].set_ylabel('output t',fontsize=7); ax[0].set_xlabel('input s',fontsize=7)
fig.tight_layout(); fig.savefig('out/fig_ch6_mix.png',dpi=200); plt.close(); print('ok')
