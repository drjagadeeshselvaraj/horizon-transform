import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from lib import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0'
# ---- Fig 3.1 poles in the complex plane
h,w=ht_default(14,F=2,hmax=64); lam=ht_poles(h,w); fig,ax=plt.subplots(1,2,figsize=(6.3,3.1),gridspec_kw={'width_ratios':[1,1]})
for a,(xl,yl) in zip(ax,[((-1.15,1.15),(-1.15,1.15)),((0.80,1.02),(-0.14,0.14))]):
    th=np.linspace(0,2*np.pi,400); a.plot(np.cos(th),np.sin(th),color=GRY,lw=1); a.axhline(0,color='#bbb',lw=.6); a.axvline(0,color='#bbb',lw=.6)
    sc=a.scatter(lam.real,lam.imag,c=np.log10(h),cmap='viridis',s=26,zorder=3,edgecolor='k',linewidth=.3); a.scatter(lam.real,-lam.imag,c=np.log10(h),cmap='viridis',s=14,alpha=.35,zorder=2)
    a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect('equal',adjustable='box'); a.grid(False)
ax[0].set_title('Poles of a 14-channel bank',fontsize=8.5); ax[1].set_title('Zoom near z = 1',fontsize=8.5)
ax[0].annotate('unit circle',(0.7,0.72),xytext=(0.25,1.02),fontsize=7,color=GRY,arrowprops=dict(arrowstyle='->',color=GRY)); ax[0].text(-1.1,-1.05,'radius = exp(−1/h),  angle = ω',fontsize=7,color='#333')
cb=fig.colorbar(sc,ax=ax,shrink=.8,pad=.02); cb.set_label('log₁₀ horizon',fontsize=7)
for a in ax: a.set_xlabel('real part')
ax[0].set_ylabel('imaginary part')
fig.savefig('out/fig_ch3_poles.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 3.2 frequency response, peak gain and bandwidth
th=np.linspace(0.4,1.6,1500); w0=1.0; fig,ax=plt.subplots(1,2,figsize=(6.3,2.8))
for hh,c in zip([8,32,128,512],plt.cm.viridis(np.linspace(0,.85,4))):
    lam=np.exp(-1/hh+1j*w0); b=np.sqrt(1-abs(lam)**2); ax[0].plot(th,np.abs(b/(1-lam*np.exp(-1j*th))),color=c,lw=1.4,label=f'h = {hh}')
ax[0].set_xlabel('frequency θ (rad/sample)'); ax[0].set_ylabel('|H(θ)|'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title('Narrower and taller as h grows',fontsize=8.5)
hs=np.geomspace(2,4096,25); pk=[];hw=[]
for hh in hs:
    lam=np.exp(-1/hh+1j*w0); b=np.sqrt(1-abs(lam)**2); tt=np.linspace(w0-np.pi,w0+np.pi,400001); H=np.abs(b/(1-lam*np.exp(-1j*tt))); p=H.max(); pk.append(p); m=H>=p/np.sqrt(2); hw.append((tt[m].max()-tt[m].min())/2)
ax[1].loglog(hs,pk,'o',color=NAVY,ms=3.5,label='peak gain, measured'); ax[1].loglog(hs,np.sqrt(2*hs),color=NAVY,lw=1,label='√(2h)'); ax[1].loglog(hs,hw,'s',color=ORG,ms=3.5,label='half-power half-width, measured'); ax[1].loglog(hs,1/hs,color=ORG,lw=1,label='1/h')
ax[1].set_xlabel('horizon h (samples)'); ax[1].legend(frameon=False,fontsize=6.5,loc='center right',bbox_to_anchor=(1.0,0.5)); ax[1].set_title('Measured against Eqs. (3.9) and (3.10)',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch3_response.png',dpi=200); plt.close()
# ---- Fig 3.3 geometric spacing
fig,ax=plt.subplots(1,2,figsize=(6.3,2.8)); T=np.geomspace(8,65536,60)
for r,c in zip([1.5,2,3,4],[NAVY,GRN,ORG,PUR]): ax[0].semilogx(T,np.floor(np.log(T)/np.log(r))+1,color=c,lw=1.4,label=f'r = {r}')
ax[0].set_xlabel('longest horizon (sequence length T)'); ax[0].set_ylabel('number of horizon scales M'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title('Scales grow with log T',fontsize=8.5)
tau=np.geomspace(1,8000,400)
for hh in 2.0**np.arange(0,11): ax[1].semilogx(tau,1-np.exp(-tau/hh),color=NAVY,lw=.9,alpha=.75)
for hh in 8.0**np.arange(0,4): ax[1].semilogx(tau,1-np.exp(-tau/hh),color=ORG,lw=1.6,ls='--')
ax[1].plot([],[],color=NAVY,lw=1,label='r = 2'); ax[1].plot([],[],color=ORG,lw=1.6,ls='--',label='r = 8'); ax[1].legend(frameon=False,fontsize=7,loc='lower right'); ax[1].set_xlabel('lag τ (samples)'); ax[1].set_ylabel('share of weight within lag τ'); ax[1].set_title('Overlap and gaps',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch3_spacing.png',dpi=200); plt.close()
# ---- Fig 3.4 frequency placement rules
fig,ax=plt.subplots(figsize=(6.0,2.9)); h1,w1=ht_default(32,F=2,hmax=4096); h2,w2=ht_default(32,F=4,hmax=4096); fs=128; f=np.geomspace(2,40,32); w3=2*np.pi*f/fs; h3=2*np.pi/w3
ax.scatter(np.log10(h1),w1,s=18,color=NAVY,marker='o',label='default rule, F = 2 (Eq. 3.11)'); ax.scatter(np.log10(h2),w2,s=18,color=GRN,marker='s',label='default rule, F = 4'); ax.scatter(np.log10(h3),w3,s=20,color=ORG,marker='^',label='constant-Q, Q = 2π, 2–40 Hz at 128 Hz (Eq. 3.12)')
ax.set_yscale('log'); ax.set_xlabel('log₁₀ horizon (samples)'); ax.set_ylabel('channel frequency ω (rad/sample)'); ax.legend(frameon=False,fontsize=6.8,loc='upper right'); fig.tight_layout(); fig.savefig('out/fig_ch3_placement.png',dpi=200); plt.close()
# ---- Fig 3.5 gating: ungated rotation vs step-scaled rotation
n=400; t=np.arange(n); rng=np.random.default_rng(1); v=0.3*rng.standard_normal(n); flag=np.zeros(n); flag[50]=1; v[50]=1.0
D=np.where(flag>0,1.0,0.002); g=np.where(flag>0,1.0,0.01); hh,ww=400.0,0.3
def run(scaled):
    X=np.zeros(n,complex); s=0j
    for i in range(n):
        lam=np.exp(D[i]*(-1/hh+1j*ww)) if scaled else np.exp(-D[i]/hh+1j*ww); s=lam*s+g[i]*v[i]; X[i]=s
    return X
fig,ax=plt.subplots(3,1,figsize=(6.2,4.6),sharex=True)
ax[0].plot(t,v,color='#333',lw=.7); ax[0].axvline(50,color=ORG,ls=':',lw=1); ax[0].text(56,0.8,'flagged step: value 1',fontsize=7,color=ORG); ax[0].set_ylabel('input'); ax[0].set_title('A channel is asked to hold the flagged value',fontsize=8.5)
for a,sc,lab,c in [(ax[1],False,'rotation not scaled by the step (Eq. 3.6 as written)',ORG),(ax[2],True,'rotation scaled by the step (recommended form)',NAVY)]:
    X=run(sc); a.plot(t,X.real,color=c,lw=1.2); a.axhline(1,color=GRY,ls='--',lw=.8); a.set_ylabel('Re X'); a.set_title(lab,fontsize=7.6,color=c,loc='left'); a.set_ylim(-1.25,1.25)
ax[2].set_xlabel('time t (samples)'); fig.tight_layout(); fig.savefig('out/fig_ch3_gating.png',dpi=200); plt.close()
# ---- Fig 3.6 design recipe
fig,ax=plt.subplots(figsize=(6.3,4.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.grid(False)
steps=[('1  Horizon range','h$_{min}$ = 1;  h$_{max}$ ≈ longest dependency (often T)','Sec. 3.1'),('2  Spacing ratio','r from M; r = 2 gives 13 scales for 1 to 4096','Sec. 3.1'),('3  Frequencies','F per scale, or a constant-Q grid tied to the signal band','Sec. 3.4'),('4  Gains','b = √(1 − |λ|²) unless learned','Sec. 3.3'),('5  Computation','recurrence (streaming), FFT (fixed HT), scan (HT-A)','Sec. 3.7'),('6  Learning (optional)','h = exp(θ); keep ω in [0, π]; recompute b from λ','Sec. 3.7'),('7  Gating (optional)','input sets Δ$_t$ and gain; scale the rotation with Δ$_t$','Sec. 3.2')]
ys=np.linspace(0.92,0.07,7); cols=[NAVY,NAVY,GRN,GRN,GLD,ORG,ORG]
for (a,b,c),y,col in zip(steps,ys,cols):
    ax.add_patch(FancyBboxPatch((0.02,y-0.045),0.27,0.09,boxstyle='round,pad=0.004,rounding_size=0.012',fc='white',ec=col,lw=1.3)); ax.text(0.155,y,a,ha='center',va='center',fontsize=7.2,fontweight='bold',color=col)
    ax.text(0.31,y,b,ha='left',va='center',fontsize=6.6,color='#222'); ax.text(0.995,y,c,ha='right',va='center',fontsize=6.6,color=GRY)
for y0,y1 in zip(ys[:-1],ys[1:]): ax.add_patch(FancyArrowPatch((0.155,y0-0.047),(0.155,y1+0.047),arrowstyle='-|>',mutation_scale=8,lw=1,color='#555'))
fig.savefig('out/fig_ch3_recipe.png',dpi=200,bbox_inches='tight'); plt.close(); print('ok')
