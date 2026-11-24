import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0'
e7=json.load(open('out/e7.json')); e6=json.load(open('out/e6.json'))
# ---- Fig 4.1 inversion pipeline
fig,ax=plt.subplots(figsize=(6.3,3.3)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.grid(False)
def box(x,y,w,h,t,fc='#EAF0FA',ec=NAVY,fs=7.5): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',fc=fc,ec=ec,lw=1.1)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(p,q,c='#333',ls='-'): ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8,lw=1,color=c,ls=ls))
ys=[0.72,0.46,0.14]; labs=[r'channel 0',r'channel 1',r'channel $K-1$']
box(0.01,0.40,0.13,0.2,'noisy states\n$Y[t,k]$\n$Y[t-1,k]$',fc='#FFF2CC',ec=GLD,fs=7.3)
for y,l in zip(ys,labs):
    arr((0.14,0.5),(0.17,y+0.07)); box(0.17,y,0.30,0.14,r'$r_k=Y[t,k]-\lambda_k\,Y[t-1,k]$',fs=6.8); arr((0.47,y+0.07),(0.51,y+0.07)); box(0.51,y,0.13,0.14,r'divide by $b_k$',fs=6.8); arr((0.64,y+0.07),(0.70,0.5))
    ax.text(0.32,y+0.17,l,ha='center',fontsize=6.8,color='#555')
ax.text(0.32,0.34,r'$\vdots$',ha='center',fontsize=12); ax.text(0.575,0.34,r'$\vdots$',ha='center',fontsize=12)
box(0.70,0.40,0.15,0.2,'pool with\nweights $w_k$\n(take real part)',fc='#E2F0D9',ec=GRN,fs=7); arr((0.85,0.5),(0.90,0.5)); box(0.90,0.42,0.09,0.16,r'$\hat{x}[t]$',fc='#FFF2CC',ec=GLD,fs=9)
ax.text(0.5,0.96,'Noise-free: every channel gives the same exact $x[t]$.   Noisy: pooling averages the channel estimates.',ha='center',fontsize=7.8,color=NAVY)
ax.text(0.5,0.03,'Only $Y[t]$ and $Y[t-1]$ are needed, so this inverse is causal and streaming.',ha='center',fontsize=7.6,color='#444')
fig.savefig('out/fig_ch4_pipeline.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 4.2 precision
p=e7['precision']; h=np.array(p['h']); fig,ax=plt.subplots(figsize=(5.8,2.9))
ax.loglog(h,p['float32'],'o-',color=ORG,ms=4,lw=1.3,label='float32, measured'); ax.loglog(h,p['float64'],'s-',color=NAVY,ms=4,lw=1.3,label='float64, measured')
ax.loglog(h,np.finfo(np.float32).eps*np.sqrt(h/2),color=ORG,ls='--',lw=1,label='ε₃₂·√(h/2)'); ax.loglog(h,np.finfo(np.float64).eps*np.sqrt(h/2),color=NAVY,ls='--',lw=1,label='ε₆₄·√(h/2)')
ax.set_xlabel('horizon h (samples)'); ax.set_ylabel('largest inversion error over 6000 samples'); ax.legend(frameon=False,fontsize=7); fig.tight_layout(); fig.savefig('out/fig_ch4_precision.png',dpi=200); plt.close()
# ---- Fig 4.3 effective channel count (computed here)
from lib import ht_poles, ht_gain
def bank(order,F,M=13):
    ks=list(range(M)); ks=ks if order=='short first' else ks[::-1]; hh=[];ww=[]
    for k in ks:
        h=2.0**k
        for j in range(F): hh.append(h); ww.append(min(np.pi,np.pi*(j+1)/(2*h)))
    return np.array(hh),np.array(ww)
def stats(h,w):
    lam=ht_poles(h,w); b=ht_gain(lam); return float(np.sum(b**2)),float(0.5*np.sum(b**2*(1+np.abs(lam)**2))/np.sum(b**2)**2)
fig,ax=plt.subplots(1,2,figsize=(6.3,2.8)); M=13; cols={1:NAVY,2:GRN,4:ORG}
for F,c in cols.items():
    sb=[];gn=[]
    for m in range(1,M+1):
        h,w=bank('short first',F,M); h,w=h[:m*F],w[:m*F]; a,b_=stats(h,w); sb.append(a); gn.append(b_)
    ax[0].plot(range(1,M+1),sb,'o-',color=c,ms=3.5,lw=1.3,label=f'F = {F}'); ax[1].semilogy(range(1,M+1),gn,'o-',color=c,ms=3.5,lw=1.3,label=f'F = {F}, shortest first')
h,w=bank('long first',2,M); gl=[stats(h[:m*2],w[:m*2])[1] for m in range(1,M+1)]; ax[1].semilogy(range(1,M+1),gl,'s--',color=GRY,ms=3.5,lw=1.2,label='F = 2, longest first')
ax[0].set_xlabel('number of scales added (r = 2, horizons 1, 2, 4, …)'); ax[0].set_ylabel('Σ b² over all channels'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title('Long horizons add little',fontsize=8.5)
ax[1].set_xlabel('number of scales added'); ax[1].set_ylabel('noise gain of the pooled inverse'); ax[1].legend(frameon=False,fontsize=6.5); ax[1].set_title('Noise gain falls only with short horizons',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch4_effective.png',dpi=200); plt.close()
# ---- Fig 4.4 least-squares symbol
sym=e7['symbol']; fig,ax=plt.subplots(1,2,figsize=(6.3,2.9),gridspec_kw={'width_ratios':[1.3,1]}); cs=[NAVY,ORG,GRN]
for (nm,d),c in zip(sym.items(),cs): ax[0].semilogy(d['theta'],d['s'],color=c,lw=1.3,label=nm)
ax[0].set_xlabel('frequency θ (rad/sample)'); ax[0].set_ylabel('s(θ), Eq. (4.9)'); ax[0].legend(frameon=False,fontsize=6.8); ax[0].set_title('Where the bank sees the signal',fontsize=8.5)
names=list(sym); K=[4,4,16]; ideal=[10*np.log10(1/k) for k in K]; act=[10*np.log10(sym[n]['gain_exact']) for n in names]; x=np.arange(3)
ax[1].bar(x-0.19,ideal,0.36,color='#BBBBBB',label='even coverage, 1/K'); ax[1].bar(x+0.19,act,0.36,color=cs,label='this bank'); ax[1].set_xticks(x); ax[1].set_xticklabels(['default\nρ = 8','tuned\nρ = 8','default\nρ = 32'],fontsize=7); ax[1].set_ylabel('white-noise gain of LS (dB)'); ax[1].legend(frameon=False,fontsize=6.8,loc='lower left'); ax[1].set_title('Cost of uneven coverage',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch4_symbol.png',dpi=200); plt.close()
# ---- Fig 4.5 decoders
K4=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']; decs=['Pooled (Chapter 4 inverse)','Pooled, noise-weighted','LS (unweighted)','GLS (weighted LS)']; dc=['#9DB4D8',NAVY,'#F4B183',ORG]
trs=['HT default','HT tuned','STFT (W=128)','Undecimated wavelet (db2)']; fig,ax=plt.subplots(figsize=(6.3,2.9)); wd=0.2
for j,(d,c) in enumerate(zip(decs,dc)):
    vals=[np.mean([e6[t]['8'][k][d] for k in K4]) if d in e6[t]['8'][K4[0]] else np.nan for t in trs]; ax.bar(np.arange(4)+(j-1.5)*wd,vals,wd,color=c,label=d.replace('Pooled (Chapter 4 inverse)','pooled (Eq. 4.1)').replace('Pooled, noise-weighted','pooled, noise-weighted').replace('LS (unweighted)','least squares').replace('GLS (weighted LS)','weighted least squares'))
ax.set_xticks(range(4)); ax.set_xticklabels(['HT default','HT tuned','STFT','wavelet (db2)']); ax.set_ylabel('output SNR (dB), mean of 4 families'); ax.set_ylim(20,34); ax.legend(frameon=False,fontsize=6.8,ncol=2,loc='upper left'); ax.set_title('ρ = 8, 20 dB per-channel coefficient SNR',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch4_decoders.png',dpi=200); plt.close()
# ---- Fig 4.6 erasures
fig,ax=plt.subplots(figsize=(5.8,2.9)); ps=e7['erasure_ps']
for (nm,d),c,mk in zip(e7['erasure'].items(),[NAVY,ORG,GRN,GRY],['o','s','^','v']): ax.plot([100*p for p in ps],[d[str(p)][0] for p in ps],marker=mk,color=c,ms=4,lw=1.3,label=nm)
ax.set_xlabel('share of coefficients erased at random (%)'); ax.set_ylabel('output SNR (dB)'); ax.legend(frameon=False,fontsize=7); ax.set_title('ρ = 8, 30 dB per-channel coefficient SNR, weighted LS on the remaining coefficients',fontsize=8); fig.tight_layout(); fig.savefig('out/fig_ch4_erasure.png',dpi=200); plt.close(); print('ok')
