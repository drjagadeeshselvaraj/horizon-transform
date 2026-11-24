import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, scipy.signal as ss
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY='#1F3864','#C55A11','#548235','#7F7F7F'
# ---------- Fig 1.1: operation counts
d,N=64,16; T=np.unique(np.geomspace(16,2**17,60).astype(int)); att=4*T**2*d+5*T**2; rec=14*T*d*N; fft=10*(2*T)*np.log2(2*T)*d+8*N*T
fig,ax=plt.subplots(figsize=(6.0,3.0)); ax.loglog(T,att,color='#C00000',lw=1.6,label='exact attention  (4T²d + 5T²)'); ax.loglog(T,rec,color=NAVY,lw=1.6,label='HT streaming recurrence  (14TdN)'); ax.loglog(T,fft,color=GRN,lw=1.6,ls='--',label='HT as FFT convolution')
ts=14*d*N/(4*d+5); ax.axvline(ts,color=GRY,ls=':',lw=1); ax.annotate(f'equal at T ≈ {ts:.0f}',(ts,3e5),xytext=(75,4e9),fontsize=7.5,color='#444',arrowprops=dict(arrowstyle='->',color=GRY))
ax.set_xlabel('sequence length T'); ax.set_ylabel('arithmetic operations per layer'); ax.legend(frameon=False,fontsize=7.5,loc='upper left'); fig.tight_layout(); fig.savefig('out/fig_ch1_cost.png',dpi=200); plt.close()
# ---------- Fig 1.2: schematic of the HT bank
fig,ax=plt.subplots(figsize=(6.3,3.6)); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.axis('off'); ax.grid(False)
def box(x,y,w,h,t,fc='#EAF0FA',ec=NAVY,fs=8,ls='-'): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.015',fc=fc,ec=ec,lw=1.1,ls=ls)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(p,q,c='#333',ls='-',cs='arc3'): ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=9,lw=1,color=c,ls=ls,connectionstyle=cs))
ax.text(0.5,0.985,r'every channel:  $X[t,k]=\lambda_k\,X[t-1,k]+b_k\,x[t]$,   $\lambda_k=e^{-1/h_k+i\omega_k}$',ha='center',fontsize=8.5,color=NAVY)
ax.text(0.5,0.925,r'geometric horizons:  $h_k=h_{\min}\,r^{k}$   (for example 1, 2, 4, 8, …)',ha='center',fontsize=8,color='#444')
box(0.01,0.42,0.12,0.16,'input\n$x[t]$',fc='#FFF2CC',ec='#BF9000')
ys=[0.70,0.54,0.38,0.18]; labs=[r'channel $k=0$:   $h_0,\ \omega_0$',r'channel $k=1$:   $h_1,\ \omega_1$',None,r'channel $k=K-1$:   $h_{K-1},\ \omega_{K-1}$']
for y,l in zip(ys,labs):
    if l is None: ax.text(0.45,y+0.05,r'$\vdots$',ha='center',va='center',fontsize=12); continue
    box(0.27,y,0.38,0.1,l); arr((0.13,0.50),(0.27,y+0.05)); arr((0.65,y+0.05),(0.74,0.52))
box(0.74,0.42,0.25,0.16,'coefficients\n$X[t,k]$,  $k=0,\\ldots,K-1$',fc='#E2F0D9',ec=GRN,fs=7.5)
ax.plot([0.865,0.865,0.07,0.07],[0.42,0.045,0.045,0.41],color=ORG,lw=1.1); ax.add_patch(FancyArrowPatch((0.07,0.30),(0.07,0.415),arrowstyle='-|>',mutation_scale=9,lw=1,color=ORG))
ax.text(0.47,0.085,r'exact inverse:  $x[t]=\mathrm{Re}\,[\,b_k^{*}(X[t,k]-\lambda_k X[t-1,k])\,]\,/\,|b_k|^{2}$',ha='center',fontsize=8,color=ORG)
box(0.01,0.70,0.19,0.12,'HT-A: input sets\n$\\Delta_t$ and gain',fc='white',ec=GRY,fs=7,ls='--'); arr((0.20,0.76),(0.27,0.755),c=GRY,ls='--')
fig.savefig('out/fig_ch1_bank.png',dpi=200,bbox_inches='tight'); plt.close()
# ---------- Fig 1.3: coverage windows and impulse responses
fig,ax=plt.subplots(1,2,figsize=(6.3,2.7)); tau=np.geomspace(1,5000,400)
for h,c in zip([1,4,16,64,256,1024],plt.cm.viridis(np.linspace(0,.85,6))): ax[0].semilogx(tau,1-np.exp(-tau/h),color=c,lw=1.5,label=f'h = {h}')
ax[0].set_xlabel('lag τ (samples)'); ax[0].set_ylabel('share of channel weight within lag τ'); ax[0].set_title('Windows spaced evenly on a log axis',fontsize=8.5)
t=np.arange(0,260)
for h,w,c,lab in [(64,2*np.pi/16,NAVY,'h = 64, period 16'),(16,2*np.pi/8,ORG,'h = 16, period 8')]:
    lam=np.exp(-1/h+1j*w); b=np.sqrt(1-abs(lam)**2); ax[1].plot(t,np.real(b*lam**t),color=c,lw=1.2,label=lab)
ax[1].set_xlabel('lag τ (samples)'); ax[1].set_ylabel('real part of impulse response'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('Damped oscillations',fontsize=8.5)
h0,l0=ax[0].get_legend_handles_labels(); fig.legend(h0,l0,loc='lower left',ncol=6,frameon=False,fontsize=6.5,bbox_to_anchor=(0.02,0.0)); fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig('out/fig_ch1_kernels.png',dpi=200); plt.close()
# ---------- Fig 1.4: worked example
rng=np.random.default_rng(3); n=1024; t=np.arange(n)
slow=0.8*np.sin(2*np.pi*t/512+0.5); win=np.where((t>=300)&(t<420),0.5*(1-np.cos(2*np.pi*(t-300)/120)),0); burst=1.1*np.sin(2*np.pi*t/16)*win
spk=np.zeros(n); spk[[120,640,800]]=[3.0,-3.0,3.0]; x=slow+burst+spk+0.25*rng.standard_normal(n)
def chan(h,w):
    lam=np.exp(-1/h+1j*w); b=np.sqrt(1-abs(lam)**2); X=ss.lfilter([b],[1,-lam],x.astype(complex)); return X,lam,b
fig,ax=plt.subplots(4,1,figsize=(6.2,5.4),sharex=True)
ax[0].plot(t,x,color='#333',lw=.8); ax[0].set_ylabel('input x[t]')
for a,(h,w,lab,c) in zip(ax[1:],[(4,np.pi/2,'A: h = 4, period 4',ORG),(32,2*np.pi/16,'B: h = 32, period 16',GRN),(256,2*np.pi/512,'C: h = 256, period 512',NAVY)]):
    X,lam,b=chan(h,w); a.plot(t,np.abs(X),color=c,lw=1.1); a.set_ylabel(f'|X|, channel {lab[0]}'); a.text(0.995,0.86,lab,transform=a.transAxes,ha='right',fontsize=7.5,color=c)
for tt in (120,640,800): ax[0].annotate('spike',(tt,3.1*np.sign(spk[tt])),xytext=(tt+12,3.1*np.sign(spk[tt])),fontsize=7,color='#666',va='center')
ax[0].annotate('burst, period 16',(360,1.7),xytext=(430,2.3),fontsize=7,color='#666',arrowprops=dict(arrowstyle='->',color=GRY)); ax[0].text(20,-1.9,'slow drift, period 512',fontsize=7,color='#666')
ax[-1].set_xlabel('time t (samples)'); fig.tight_layout(); fig.savefig('out/fig_ch1_example.png',dpi=200); plt.close()
X,lam,b=chan(32,2*np.pi/16); xr=np.real(np.conj(b)*(X-lam*np.concatenate([[0],X[:-1]]))/abs(b)**2); print('inverse error channel B',np.abs(xr-x).max())
# ---------- Fig 1.5: roadmap
fig,ax=plt.subplots(figsize=(6.3,3.5)); ax.set_xlim(0,1); ax.set_ylim(-0.06,1); ax.axis('off'); ax.grid(False)
bands=[(0.01,0.34,'Foundations','#EAF0FA',NAVY,['1  Introduction','2  Literature Review','3  Definition','4  Inverse','5  Properties','6  Relation to\nExisting Transforms']),
       (0.375,0.27,'Evidence','#FBE5D6',ORG,['7  Experimental\nProtocol','8  Experimental\nResults']),
       (0.68,0.31,'Assessment','#E2F0D9',GRN,['9  Strengths and\nWeaknesses','10  Limitations','11  Future Directions','12  Conclusion'])]
pos={}
for x0,w,t_,fc,ec,items in bands:
    ax.add_patch(Rectangle((x0,0.04),w,0.88,fc=fc,ec=ec,lw=1.2,alpha=.9)); ax.text(x0+w/2,0.885,t_,ha='center',fontsize=9,fontweight='bold',color=ec)
    m=len(items); hh=0.8/ (6 if m>4 else m*1.9) ; 
    for i,it in enumerate(items):
        y=0.78-i*(0.115 if m==6 else 0.16 if m==4 else 0.2)-(0 if m!=2 else 0.05)
        ax.add_patch(FancyBboxPatch((x0+0.015,y-0.05),w-0.03,0.1,boxstyle='round,pad=0.003,rounding_size=0.012',fc='white',ec=ec,lw=.9)); ax.text(x0+w/2,y,it,ha='center',va='center',fontsize=7.2); pos[it.split()[0]]=(x0,w,y)
ax.add_patch(FancyArrowPatch((0.35,0.48),(0.375,0.48),arrowstyle='-|>',mutation_scale=12,lw=1.5,color='#333')); ax.add_patch(FancyArrowPatch((0.645,0.48),(0.68,0.48),arrowstyle='-|>',mutation_scale=12,lw=1.5,color='#333'))
x7=pos['7']; x11=pos['11']
ax.add_patch(FancyArrowPatch((x11[0]+0.015,x11[2]-0.045),(x7[0]+x7[1]-0.015,x7[2]-0.045),arrowstyle='-|>',mutation_scale=9,lw=1,color=ORG,ls='--',connectionstyle='arc3,rad=0.35'))
ax.text(0.5,-0.04,'dashed arrow: the studies proposed in Chapter 11 feed back into the protocol of Chapter 7',ha='center',fontsize=6.8,color=ORG)
fig.savefig('out/fig_ch1_roadmap.png',dpi=200,bbox_inches='tight'); plt.close(); print('ok')
