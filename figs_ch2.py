import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.special import eval_laguerre, eval_legendre
from lib import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0'
# ---------------- Fig 2.1 timeline
lanes=[('Time–frequency\nanalysis',NAVY,[(1946,'Gabor atoms',1),(1952,'frames',-1),(1977,'STFT',1),(1984,'wavelets',-1),(1991,'constant-Q',1),(1999,'damped-sinusoid\natoms',-1),(2003,'sliding DFT',1),(2017,'SWIFT',-3)]),
 ('Causal filterbanks,\nexponential memory',GRN,[(1991,'Laguerre models',3),(1992,'gamma memory',-1),(1993,'gammatone bank',1),(2018,'SincNet',-1),(2021,'LEAF',1)]),
 ('Recurrent memory',ORG,[(1994,'vanishing gradients',1),(1997,'LSTM',-1),(2002,'echo-state memory',3),(2014,'Clockwork RNN, GRU',-1),(2018,'chrono init.',1)]),
 ('Attention','#C00000',[(2017,'Transformer',1),(2020,'efficient variants',-1),(2022,'FlashAttention',3)]),
 ('State-space models',PUR,[(2020,'HiPPO',1),(2022,'S4, S4D',-1),(2023,'LRU, Mamba',3),(2024,'Mamba-2',-3),(2025,'LinOSS',2)])]
fig,ax=plt.subplots(figsize=(6.4,5.6)); ax.set_xlim(1938,2031); ax.set_ylim(-0.55,len(lanes)-0.4); ax.grid(False); ax.spines['left'].set_visible(False); ax.set_yticks([])
for i,(name,c,items) in enumerate(lanes):
    y=len(lanes)-1-i; ax.axhline(y,color=c,lw=1.2,alpha=.5); ax.text(1936.5,y,name,ha='right',va='center',fontsize=7,color=c,fontweight='bold')
    for yr,lab,L in items:
        off=(8+11*(abs(L)-1))*(1 if L>0 else -1)
        ax.plot(yr,y,'o',color=c,ms=4.5,zorder=3)
        ax.annotate(lab,(yr,y),xytext=(0,off),textcoords='offset points',ha='center',va='bottom' if L>0 else 'top',fontsize=6.4,color='#222',arrowprops=dict(arrowstyle='-',color=c,lw=.6,shrinkA=0,shrinkB=3))
ax.set_xticks([1950,1970,1990,2010,2025]); ax.set_xlabel('year of the cited work'); fig.subplots_adjust(left=0.22,right=0.985,top=0.985,bottom=0.08); fig.savefig('out/fig_ch2_timeline.png',dpi=200); plt.close()
# ---------------- Fig 2.2 time-frequency tilings (schematic)
fig,ax=plt.subplots(1,3,figsize=(6.4,2.6),sharey=True)
for a in ax: a.set_xlim(0,8); a.set_ylim(0,4); a.set_xticks([]); a.set_yticks([0.5,1.5,2.5,3.5]); a.set_yticklabels(['low','','','high']); a.grid(False); a.set_xlabel('time  →')
ax[0].set_ylabel('frequency (log scale)')
for r in range(4):
    for c in range(8): ax[0].add_patch(Rectangle((c,r),1,1,fc='#EAF0FA',ec=NAVY,lw=.6))
widths=[4,2,1,0.5]
for r,w in enumerate(widths):
    c=0
    while c<8-1e-9: ax[1].add_patch(Rectangle((c,r),w,1,fc='#FBE5D6',ec=ORG,lw=.6)); c+=w
ts=np.linspace(0,8,300)
for r,h in enumerate([8,4,2,1]):
    img=np.exp(-(8-ts)/h)[None,:]; ax[2].imshow(img,extent=(0,8,r+0.1,r+0.9),aspect='auto',cmap='Greens',vmin=0,vmax=1.15,origin='lower',zorder=2)
    ax[2].add_patch(Rectangle((0,r+0.1),8,0.8,fc='none',ec=GRN,lw=.6,zorder=3))
ax[2].axvline(8,color='#333',lw=1.2,zorder=4); ax[2].text(7.9,3.95,'now',ha='right',va='top',fontsize=7)
for a,t in zip(ax,['(a) STFT: fixed windows','(b) wavelets: dyadic windows','(c) HT: causal exponential windows']): a.set_title(t,fontsize=7.6)
ax[2].set_xlim(0,8.2); fig.tight_layout(); fig.savefig('out/fig_ch2_tilings.png',dpi=200); plt.close()
# ---------------- Fig 2.3 causal kernel shapes
t=np.arange(0,90); fig,ax=plt.subplots(2,3,figsize=(6.4,3.6),sharex=False)
lam=np.exp(-1/20+1j*2*np.pi/16); k1=np.real(lam**t)
k2=t**3*np.exp(-t/6.0)*np.cos(2*np.pi*t/16); k3=t**2*np.exp(-t/10.0); p=1/20; k4=np.exp(-p*t)*eval_laguerre(3,2*p*t)
s=8.0; m=t<=6*s; k5=np.where(m,np.exp(-(t-3*s)**2/(2*s**2))*np.cos(6*(t-3*s)/s),0.0); k6=np.where(t<64,eval_legendre(3,1-2*(t+0.5)/64),0.0)
for a,k,tt,c in zip(ax.ravel(),[k1,k2,k3,k4,k5,k6],['HT channel\n(damped complex exponential)','gammatone, order 4','gamma memory, order 3','Laguerre function, order 3','causal Morlet (delayed)','Legendre window, order 3'],[NAVY,GRN,GRN,GLD,ORG,PUR]):
    a.plot(t,k/np.abs(k).max(),color=c,lw=1.3); a.axhline(0,color='#999',lw=.6); a.set_title(tt,fontsize=7.4); a.set_ylim(-1.25,1.25); a.set_yticks([-1,0,1])
for a in ax[1]: a.set_xlabel('lag (samples)')
fig.tight_layout(); fig.savefig('out/fig_ch2_kernels.png',dpi=200); plt.close()
# ---------------- Fig 2.4 where the poles sit
h,w=ht_default(32,F=2,hmax=4096); lamH=ht_poles(h,w); lamL=bank_lru(64,seed=0).lam; lamS=bank_s4d_lin(64).lam
def hw(lam): return -1/np.log(np.abs(lam)),np.abs(np.angle(lam))
fig,ax=plt.subplots(1,2,figsize=(6.4,2.8),gridspec_kw={'width_ratios':[1.25,1]})
for lm,lab,c,mk in [(lamH,'HT default (geometric)',NAVY,'o'),(lamL,'LRU ring init',GLD,'^'),(lamS,'S4D-Lin, 4 step sizes',ORG,'s')]:
    hh,ww=hw(lm); ax[0].scatter(np.log10(hh),ww,s=14,color=c,marker=mk,alpha=.85,label=lab)
ax[0].set_xlabel('log₁₀ horizon (samples)'); ax[0].set_ylabel('pole angle (rad/sample)'); ax[0].set_yscale('symlog',linthresh=0.02); ax[0].legend(frameon=False,fontsize=6.8,loc='upper right')
bins=np.linspace(-0.5,4.5,11)
for lm,c in [(lamH,NAVY),(lamL,GLD),(lamS,ORG)]: ax[1].hist(np.log10(hw(lm)[0]),bins=bins,histtype='step',lw=1.6,color=c)
ax[1].set_xlabel('log₁₀ horizon (samples)'); ax[1].set_ylabel('number of poles'); ax[0].set_title('Pole placement',fontsize=8.5); ax[1].set_title('Horizon coverage',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch2_poles.png',dpi=200); plt.close()
# ---------------- Fig 2.5 lineage map
fig,ax=plt.subplots(figsize=(6.4,4.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.grid(False)
prec=['Time–frequency analysis:\nSTFT, constant-Q, wavelets','Damped-sinusoid atoms,\nrecursive filter banks','Causal filterbanks:\ngammatone, gamma, Laguerre','Reservoir and polynomial memory:\nESN, HiPPO, LMU','Diagonal and selective SSMs:\nS4D, LRU, Mamba, LinOSS','Parallel scans and\nFFT convolution']
ingr=['multi-scale channels','one-sided damped exponentials','causal exponential kernels','fixed-size state','complex diagonal recurrence, gating','efficient parallel computation']
cols=[NAVY,GRN,GRN,GLD,PUR,GRY]; ys=np.linspace(0.88,0.10,6)
for y,p,i,c in zip(ys,prec,ingr,cols):
    ax.add_patch(FancyBboxPatch((0.01,y-0.055),0.33,0.11,boxstyle='round,pad=0.003,rounding_size=0.012',fc='white',ec=c,lw=1.1)); ax.text(0.175,y,p,ha='center',va='center',fontsize=6.6)
    ax.add_patch(FancyBboxPatch((0.42,y-0.04),0.33,0.08,boxstyle='round,pad=0.003,rounding_size=0.012',fc='#F3F3F3',ec=c,lw=1.1)); ax.text(0.585,y,i,ha='center',va='center',fontsize=6.4)
    ax.add_patch(FancyArrowPatch((0.34,y),(0.42,y),arrowstyle='-|>',mutation_scale=8,lw=1,color=c)); ax.add_patch(FancyArrowPatch((0.75,y),(0.84,0.49),arrowstyle='-|>',mutation_scale=8,lw=1,color=c))
ax.add_patch(FancyBboxPatch((0.84,0.30),0.15,0.38,boxstyle='round,pad=0.004,rounding_size=0.015',fc='#EAF0FA',ec=NAVY,lw=1.6)); ax.text(0.915,0.49,'Horizon\nTransform\n(HT, HT-A)',ha='center',va='center',fontsize=7.8,fontweight='bold',color=NAVY)
ax.text(0.175,0.985,'Precedent',ha='center',fontsize=8,color='#444'); ax.text(0.585,0.985,'Ingredient of HT',ha='center',fontsize=8,color='#444'); ax.text(0.915,0.985,'Combination',ha='center',fontsize=8,color='#444')
fig.savefig('out/fig_ch2_lineage.png',dpi=200,bbox_inches='tight'); plt.close(); print('ok')
