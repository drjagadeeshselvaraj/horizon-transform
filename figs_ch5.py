import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from lib import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'mathtext.fontset':'dejavusans'})
NAVY,ORG,GRN,GRY,GLD,PUR='#1F3864','#C55A11','#548235','#7F7F7F','#BF9000','#7030A0'
e8=json.load(open('out/e8.json'))
# ---- Fig 5.1 property map
fig,ax=plt.subplots(figsize=(6.3,3.7)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.grid(False)
ax.add_patch(FancyBboxPatch((0.38,0.40),0.24,0.20,boxstyle='round,pad=0.005,rounding_size=0.02',fc='#EAF0FA',ec=NAVY,lw=1.8)); ax.text(0.5,0.50,'HT channel\n$X[t]=\\lambda X[t-1]+b\\,x[t]$',ha='center',va='center',fontsize=8.5,fontweight='bold',color=NAVY)
items=[(0.02,0.74,'Cost','O(K) state, O(TK) time;\nFFT and scan forms','5.1',NAVY),(0.37,0.80,'Causality','no look-ahead;\ndelay ≈ horizon','5.2',GRN),(0.72,0.74,'Stability','gain ≤ b/(1−|λ|);\nHT-A needs b ∝ Δ','5.3',ORG),
       (0.02,0.10,'Resolution','time h, frequency 1/h;\nconstant area','5.4',PUR),(0.37,0.04,'Invertibility','frame bounds A, B;\ncondition number','5.5',GLD),(0.72,0.10,'Gradients','closed-form derivatives;\ngradient decays as e^(−τ/h)','5.6',GRY)]
for x,y,t,d,s,c in items:
    ax.add_patch(FancyBboxPatch((x,y),0.26,0.17,boxstyle='round,pad=0.004,rounding_size=0.012',fc='white',ec=c,lw=1.3)); ax.text(x+0.13,y+0.125,t+f' (Sec. {s})',ha='center',va='center',fontsize=7,fontweight='bold',color=c); ax.text(x+0.13,y+0.055,d,ha='center',va='center',fontsize=6.7)
    cx=x+0.13; cy=y+(0.0 if y>0.5 else 0.17); ax.add_patch(FancyArrowPatch((cx,cy),(0.5,0.60 if y>0.5 else 0.40),arrowstyle='-',lw=0.9,color=c,alpha=.7))
ax.text(0.5,0.31,'plus scale equivariance (Sec. 5.7)',ha='center',fontsize=7.5,color='#444')
fig.savefig('out/fig_ch5_map.png',dpi=200,bbox_inches='tight'); plt.close()
# ---- Fig 5.2 cost scaling
c=e8['cost']; fig,ax=plt.subplots(1,2,figsize=(6.3,2.8))
ax[0].loglog(c['K'],np.array(c['vsK']['rec'])*1e3,'o-',color=NAVY,ms=3.5,lw=1.3,label=f"recurrence (slope {c['slopes']['rec_K']:.2f})"); ax[0].loglog(c['K'],np.array(c['vsK']['fft'])*1e3,'s-',color=GRN,ms=3.5,lw=1.3,label=f"FFT convolution (slope {c['slopes']['fft_K']:.2f})")
ax[0].set_xlabel('number of complex channels K'); ax[0].set_ylabel('time (ms), T = 16384'); ax[0].legend(frameon=False,fontsize=6.8); ax[0].set_title('Linear in K',fontsize=8.5)
ax[1].loglog(c['Ts'],np.array(c['vsT']['rec'])*1e3,'o-',color=NAVY,ms=3.5,lw=1.3,label=f"recurrence (slope {c['slopes']['rec_T']:.2f})"); ax[1].loglog(c['Ts'],np.array(c['vsT']['fft'])*1e3,'s-',color=GRN,ms=3.5,lw=1.3,label=f"FFT convolution (slope {c['slopes']['fft_T']:.2f})")
ax[1].set_xlabel('sequence length T'); ax[1].set_ylabel('time (ms), K = 32'); ax[1].legend(frameon=False,fontsize=6.8); ax[1].set_title('Linear to T log T in T',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch5_cost.png',dpi=200); plt.close()
# ---- Fig 5.3 causality and delay
fig,ax=plt.subplots(1,2,figsize=(6.3,2.8)); b=e8['burst']; t=np.array(b['t'])
ax[0].plot(t,np.array(b['x'])*0.0+0.0,alpha=0); cs=plt.cm.viridis(np.linspace(0,.8,3))
for (h,v),cc in zip(b['lat'].items(),cs): ax[0].plot(t,v['curve'],color=cc,lw=1.3,label=f"h = {h}: 63% at +{v['t63']}")
ax[0].axvline(200,color=GRY,ls=':',lw=1); ax[0].text(210,0.05,'burst starts',fontsize=7,color=GRY); ax[0].set_xlim(100,1000); ax[0].set_xlabel('time t (samples)'); ax[0].set_ylabel('|X| / peak'); ax[0].legend(frameon=False,fontsize=6.8,loc='lower right'); ax[0].set_title('Response to a sinusoidal burst',fontsize=8.5)
gd=np.array(e8['groupdelay']); ax[1].loglog(gd[:,0],gd[:,1],'o',color=NAVY,ms=4,label='measured'); ax[1].loglog(gd[:,0],gd[:,2],'-',color=ORG,lw=1.2,label='|λ|/(1−|λ|), Eq. (5.1)'); ax[1].set_xlabel('horizon h (samples)'); ax[1].set_ylabel('group delay at ω (samples)'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('Delay grows with the horizon',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch5_delay.png',dpi=200); plt.close()
# ---- Fig 5.4 stability
fig,ax=plt.subplots(1,2,figsize=(6.3,2.8)); w=np.array(e8['bibo']['w'])
for r,cc in zip(['0.9','0.99','0.999'],[NAVY,GRN,ORG]): ax[0].plot(w,e8['bibo']['ratio'][r],color=cc,lw=1.3,label=f'|λ| = {r}')
ax[0].axhline(2/np.pi,color=GRY,ls='--',lw=1); ax[0].text(2.2,2/np.pi-0.035,'2/π',fontsize=7,color=GRY); ax[0].set_xlabel('channel frequency ω (rad/sample)'); ax[0].set_ylabel('worst-case |X| / bound of Eq. (5.2)'); ax[0].set_ylim(0.55,1.03); ax[0].legend(frameon=False,fontsize=7,loc='upper center'); ax[0].set_title('How tight is the bound?',fontsize=8.5)
cp=e8['coupled']; ax[1].semilogy(cp['t'],np.maximum(cp['Xu'],1e-3),color=ORG,lw=1.3,label='gain not tied to step (b = 1)'); ax[1].semilogy(cp['t'],np.maximum(cp['Xc'],1e-3),color=NAVY,lw=1.3,label='gain tied to step (b = Δ)'); ax[1].axhline(cp['bound'],color=GRY,ls='--',lw=1); ax[1].text(2000,cp['bound']*1.3,'bound, Eq. (5.4)',fontsize=7,color=GRY)
ax[1].set_xlabel('time t (samples)'); ax[1].set_ylabel('|X| for input x ≡ 1'); ax[1].legend(frameon=False,fontsize=6.8,loc='lower right'); ax[1].set_title('HT-A: coupling gain to step',fontsize=8.5); ax[1].set_ylim(1,1e5)
fig.tight_layout(); fig.savefig('out/fig_ch5_stability.png',dpi=200); plt.close()
# ---- Fig 5.5 time-frequency tiles
fig,ax=plt.subplots(1,2,figsize=(6.3,3.0)); h,wd=ht_default(32,F=2,hmax=4096.0)
for hh,ww,cc in zip(h,wd,plt.cm.viridis(np.log10(h)/3.7)):
    lo=max(ww-1/hh,1e-4); hi=ww+1/hh; ax[0].add_patch(Rectangle((1,lo),hh-1,hi-lo,fc=cc,ec=cc,alpha=.35,lw=.6))
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlim(1,8000); ax[0].set_ylim(1e-4,4); ax[0].set_xlabel('time extent: lags up to h (samples)'); ax[0].set_ylabel('frequency band ω ± 1/h (rad/sample)'); ax[0].set_title('Default bank: 32 tiles',fontsize=8.5)
hh=np.geomspace(1,4096,30); ax[1].loglog(hh,hh,color=NAVY,lw=1.3,label='time extent h'); ax[1].loglog(hh,2/hh,color=ORG,lw=1.3,label='frequency width 2/h'); ax[1].loglog(hh,hh*2/hh,color=GRY,lw=1.3,ls='--',label='product = 2'); ax[1].set_xlabel('horizon h (samples)'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('Constant time–bandwidth product',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch5_tiles.png',dpi=200); plt.close()
# ---- Fig 5.6 frame bounds
fb=np.array(e8['frame']['default']); tb=np.array(e8['frame']['tuned']); fig,ax=plt.subplots(1,2,figsize=(6.3,2.8))
ax[0].loglog(fb[:,0],fb[:,1],'o-',color=NAVY,ms=3.5,lw=1.3,label='A = min s(θ)'); ax[0].loglog(fb[:,0],fb[:,2],'s-',color=ORG,ms=3.5,lw=1.3,label='B = max s(θ)'); ax[0].loglog(fb[:,0],fb[:,0],'--',color=GRY,lw=1,label='mean of s = K'); ax[0].set_xlabel('number of complex channels K (default bank)'); ax[0].set_ylabel('frame bounds'); ax[0].legend(frameon=False,fontsize=7); ax[0].set_title('Frame bounds, default rule',fontsize=8.5)
ax[1].loglog(fb[:,0],fb[:,3],'o-',color=NAVY,ms=3.5,lw=1.3,label='default rule'); ax[1].loglog(tb[:,0],tb[:,3],'s-',color=GRN,ms=3.5,lw=1.3,label='tuned (Chapter 8)'); ax[1].set_xlabel('number of complex channels K'); ax[1].set_ylabel('B / A (condition number)'); ax[1].legend(frameon=False,fontsize=7); ax[1].set_title('Condition number of the frame',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch5_frame.png',dpi=200); plt.close()
# ---- Fig 5.7 gradients
g=np.array(e8['grad']); fig,ax=plt.subplots(1,2,figsize=(6.3,2.7)); ax[0].loglog(g[:,0],g[:,1],'o-',color=NAVY,ms=4,lw=1.3); ax[0].set_xlabel('horizon h (samples)'); ax[0].set_ylabel('relative error, analytic vs finite difference'); ax[0].set_title('Eq. (5.9) against finite differences',fontsize=8.5); ax[0].set_ylim(1e-13,1e-6)
tau=np.arange(0,3000)
for hh_,cc in zip([8,32,128,512,2048],plt.cm.viridis(np.linspace(0,.85,5))):
    lam=np.exp(-1/hh_); b=np.sqrt(1-lam**2); ax[1].semilogy(tau,b*lam**tau,color=cc,lw=1.3,label=f'h = {hh_}')
ax[1].set_ylim(1e-6,2); ax[1].set_xlabel('lag τ (samples)'); ax[1].set_ylabel('|∂X[t]/∂x[t−τ]|'); ax[1].legend(frameon=False,fontsize=6.6,ncol=2); ax[1].set_title('Gradient to the past decays as e^(−τ/h)',fontsize=8.5)
fig.tight_layout(); fig.savefig('out/fig_ch5_grad.png',dpi=200); plt.close()
# ---- Fig 5.8 dilation equivariance
d=e8['dilation']; M=12; F=d['F']; m1=np.array(d['m1']).reshape(M,F); m2=np.array(d['m2']).reshape(M,F); m4=np.array(d['m4']).reshape(M,F); k=np.arange(M)
fig,ax=plt.subplots(1,2,figsize=(6.3,2.8))
for j,(mk,ls) in enumerate([('o','-'),('s','--')]):
    ax[0].plot(k,m1[:,j],color=NAVY,marker=mk,ms=3.5,lw=1.2,ls=ls,label=f'original, j = {j}'); ax[0].plot(k,m2[:,j],color=ORG,marker=mk,ms=3.5,lw=1.2,ls=ls,label=f'dilated ×2, j = {j}')
ax[0].set_xlabel('horizon scale k  (h = 2^k)'); ax[0].set_ylabel('peak |X| over time'); ax[0].legend(frameon=False,fontsize=6.3); ax[0].set_title('Dilation shifts the profile by one scale',fontsize=8.2)
for j,(mk,ls) in enumerate([('o','-'),('s','--')]):
    ax[1].plot(k[1:-1],m1[1:-1,j],color=NAVY,marker=mk,ms=3.5,lw=1.2,ls=ls); ax[1].plot(k[1:-1],m2[2:,j]/np.sqrt(2),color=ORG,marker=mk,ms=3.5,lw=1.2,ls=ls)
ax[1].set_xlabel('horizon scale k of the original'); ax[1].set_ylabel('original (blue), shifted and ÷√2 (orange)'); ax[1].set_title('They coincide',fontsize=8.2)
fig.tight_layout(); fig.savefig('out/fig_ch5_dilation.png',dpi=200); plt.close(); print('ok')
