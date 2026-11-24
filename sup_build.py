"""Build the Supplementary Materials document (Letter, same look as the book) from the result files and the book itself."""
import re, os, sys, json, shutil, subprocess, platform
import numpy as np
sys.path.insert(0,'/home/claude/build'); sys.path.insert(0,'/home/claude/ht')
import bookkit as bk
from bookkit import *
OUT='/home/claude/ht/out/'; SP='/home/claude/sup/pkg/'; bk.PKG=SP
BOOK='/mnt/user-data/outputs/Horizon_Transform.docx'; OUTDOCX='/home/claude/sup/Horizon_Transform_Supplementary_Materials.docx'
shutil.rmtree(SP,ignore_errors=True); shutil.copytree('/home/claude/kdp/pkg',SP)
# ---- strip the book's body and media from the template package
for f in os.listdir(SP+'word/media'): os.remove(SP+'word/media/'+f)
rl=open(SP+'word/_rels/document.xml.rels',encoding='utf8').read(); rl=re.sub(r'<Relationship [^>]*relationships/image"[^>]*/>','',rl); open(SP+'word/_rels/document.xml.rels','w',encoding='utf8').write(rl)
st=open(SP+'word/settings.xml',encoding='utf8').read().replace('<w:mirrorMargins/>',''); open(SP+'word/settings.xml','w',encoding='utf8').write(st)
J=lambda n:json.load(open(OUT+n+'.json'))
e0,e2,e3,e4,e6,e7,e10,e11a,e11b,e12,e13,e14,e15a,e15b,e15c,e16a,e16b,e16c,e16d=[J(n) for n in ['e0','e2','e3','e4','e6','e7','e10','e11a','e11b','e12','e13','e14','e15a','e15b','e15c','e16a','e16b','e16c','e16d']]
# ---- helpers
def H1(t): return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:pageBreakBefore/><w:spacing w:before="0" w:after="200"/></w:pPr>{run(t,b=True,sz=34,color="1F3864")}</w:p>'
def mono(t): return f'<w:p><w:pPr><w:shd w:fill="F2F4F7" w:val="clear"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>'
def code(lines): return ''.join(mono(l) for l in lines)+'<w:p><w:pPr><w:spacing w:after="80"/></w:pPr></w:p>'
def ms(v,dec=2): return f'{v[0]:.{dec}f} ± {v[1]:.{dec}f}'
def tbl(cap,header,rows,widths,left=(),fills=None):
    LEFTCOLS.clear(); LEFTCOLS.update(left); x=TCAP(cap)+TBL(header,rows,widths,fills=fills); LEFTCOLS.clear(); return x
def scale(ws): 
    s=sum(ws); w=[round(x*9026/s) for x in ws]; w[-1]+=9026-sum(w); return w
X=[]
# ============ title page
def centered(t,sz,b=False,i=False,color=None,after=120): return f'<w:p><w:pPr><w:spacing w:before="0" w:after="{after}"/><w:jc w:val="center"/></w:pPr>{run(t,b=b,i=i,sz=sz,color=color)}</w:p>'
X.append('<w:p><w:pPr><w:spacing w:before="2400" w:after="0"/></w:pPr></w:p>')
X.append(centered('Supplementary Materials',24,color='7F7F7F',after=240))
X.append(centered('The Horizon Transform',52,b=True,color='1F3864',after=160))
X.append(centered('An Exactly Invertible, Linear-Time Multi-Scale Layer for Long Sequences',26,i=True,color='404040',after=480))
X.append(centered('Companion to the monograph of the same title',22,after=80))
X.append(centered('Prof. Dr. Jagadeesh Selvaraj',22,after=80))
X.append(centered('NextGen Technologies Private Limited, Chennai, India',20,after=40))
X.append(centered('dr.jagadeeshselvaraj@nextgentpl.com',20,after=80))
X.append(centered('Reproducibility guide, task definitions, bank parameters, full numerical results, notation, indexes and change log',20,i=True,color='595959',after=80))
# ============ contents placeholder
X.append('<w:p><w:pPr><w:pageBreakBefore/><w:spacing w:before="0" w:after="200"/></w:pPr>'+run('Contents',b=True,sz=34,color='1F3864')+'</w:p>')
X.append('<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\z \\u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>x</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')
# ============ S1
X.append(H1('S1 About this supplement'))
X.append(P('This supplement accompanies the monograph on the Horizon Transform (HT). It collects what a reader needs to check, reproduce or extend the results of the book but that does not belong in the running text: the environment and commands used, exact definitions of the synthetic tasks, the parameters of the feature banks, the full numerical results behind every table and figure of Chapters 8 to 11, the notation, an index of the figures and tables, and a log of changes and corrections.'))
X.append(P('All experiments are reduced-scale. They were run with numpy and scipy on one CPU core, on synthetic data, and no trained model was used. The numbers in this supplement are those written by the scripts to the result files; they are reported without rounding beyond the number of digits shown.'))
rows=[['S2','Reproducibility','Environment, seeds, commands, run times, script catalogue','All'],['S3','Task definitions','Exact generators for the synthetic signals and probes','7, 8'],['S4','Bank definitions','Parameters of the default HT bank and descriptions of the baselines','3, 6, 8'],['S5','Full numerical results','Tables for every experiment, with standard deviations or ranges','8 to 11'],['S6','Notation','Symbols and abbreviations','All'],['S7','Index of figures and tables','Captions of every figure and table of the book','All'],['S8','Change log and open items','Corrections made, items for the author to verify','All'],['S9','Files','What each delivered file contains','—']]
X.append(tbl('Table S1.1. Contents of the supplement',['Part','Title','Contents','Chapters of the book'],rows,[800,2200,4600,1426],left=(1,2)))
# ============ S2
X.append(H1('S2 Reproducibility'))
X.append(H2('S2.1 Environment'))
import numpy,scipy,sklearn,matplotlib
rows=[['Python',platform.python_version()],['numpy',numpy.__version__],['scipy',scipy.__version__],['scikit-learn',sklearn.__version__],['matplotlib',matplotlib.__version__],['Hardware','One CPU core; no GPU'],['Linear algebra','numpy default (single thread in the test environment)'],['Random numbers','numpy default_rng with fixed seeds (S2.3)']]
X.append(tbl('Table S2.1. Environment used for the reported results',['Item','Value'],rows,[2600,6426],left=(1,)))
X.append(H2('S2.2 Script catalogue'))
X.append(P('Table S2.2 lists the scripts that produce the result files. Each script writes a JSON file to the folder out/ and prints a short summary. The figure scripts read the JSON files and write the PNG figures. The library lib.py holds the banks, the memory probes and the ridge readout.'))
readme=open('/home/claude/ht/README.md',encoding='utf8').read().split('\n'); rows=[]
for l in readme:
    if l.startswith('| `'):
        c=[x.strip().strip('`') for x in l.strip().strip('|').split('|')]
        if len(c)>=4: rows.append(['figs.py, figs2.py, figs_ch1.py to figs_ch12.py, figs_ch8x.py']+c[1:4] if c[0].startswith('figs') else c[:4])
rows.append(['lib.py','Banks (HT, S4D-Lin, LRU, Legendre, STFT, Morlet, delay line), memory probes, ridge readout','All','—'])
X.append(tbl('Table S2.2. Experiment scripts, what they produce, and approximate run time',['Script','Purpose','Chapters or sections','Time'],rows,scale([2000,4200,1900,926]),left=(1,2)))
X.append(H2('S2.3 Seeds'))
rows=[['Memory capacity probe','input and noise: 1000 + seed; seeds 0 to 4 (3 for sweeps)'],['Scale-free recall probe','input and noise: 2000 + seed'],['Reconstruction test signals','5000 + i, 20 signals per family (i = 0 to 19)'],['Classification data','300 + seed for training and test sets; seeds 0 to 2 (0 to 1 for the irregular-sampling pilot)'],['Forecasting series','series seed = run seed; noise seed 50 + run seed'],['Selective recall','search seed = run seed; five seeds'],['Timing','no random effects except the data; five repetitions per point'],['Input distributions (e15b)','555 + seed']]
X.append(tbl('Table S2.3. Seeds',['Experiment','Seed rule'],rows,[3000,6026],left=(1,)))
X.append(H2('S2.4 Commands'))
X.append(P('From the folder that contains lib.py, each experiment is run as python3 e<number>.py, and the figures for a chapter as python3 figs_ch<number>.py. The experiment scripts must be run before the figure scripts. For example:'))
X.append(code(['pip install -r requirements.txt','python3 e2.py            # memory probes, ablations, free-pole tuning','python3 e11a.py          # paired comparisons and noise sweeps','python3 e14.py           # orthogonalized bank and hybrids','python3 figs_ch9.py      # figures of Chapter 9','python3 e13.py           # timing; run on an idle machine']))
X.append(H2('S2.5 Notes on determinism'))
X.append(B('Every experiment except the timing scripts is deterministic given its seeds.'))
X.append(B('The timing scripts (e5, e13) depend on the machine. Medians varied by up to about 10% between two runs on the same machine, and the spread across five repetitions reached about a third of the median at the longest lengths.'))
X.append(B('Results that depend on the library versions (for example the ridge solver and the eigendecomposition) can differ in the last digits.'))
X.append(B('The tuned HT and tuned LRU of Chapter 8 come from a random search followed by hill climbing; the search is seeded but its outcome depends on the order of floating-point operations.'))
# ============ S3
X.append(H1('S3 Task definitions'))
X.append(P('This part states the generators exactly as coded, so that the tasks can be reimplemented without reading the scripts.'))
X.append(H2('S3.1 Reconstruction signals'))
X.append(P('Signals have N = 256 samples and are normalized to unit mean square. Four families are used:'))
rows=[['AR(1)','x[n] = 0.95 x[n−1] + e[n], e standard normal, after a burn-in of 200 samples'],['Tones, chirp, impulses','three cosines with frequencies uniform in [0.02, 0.4] cycles per sample and random phases; a linear chirp from f0 uniform in [0.02, 0.1] to f1 uniform in [0.2, 0.4]; two impulses of amplitude ±6 at random positions'],['Piecewise constant','six random breakpoints in [5, N − 5], seven levels drawn from the standard normal'],['White noise','standard normal samples']]
X.append(tbl('Table S3.1. Signal families (e1.py, function sig)',['Family','Definition'],rows,[2200,6826],left=(1,)))
X.append(P('Reconstruction is analytic: the transform is a causal N × N Toeplitz operator per real feature, the decoders are the least-squares inverse, the generalized (noise-weighted) least squares and the pooled inverse of Chapter 4, and the output SNR is computed from the closed-form noise gain for a given noise model, with 20 dB of noise per channel (each channel’s noise relative to its own coefficient power) unless stated. The redundancy ρ is the number of real coefficients per sample; ρ = 8 means K = 4 complex channels.'))
X.append(H2('S3.2 Memory probes'))
X.append(P('The input is standard normal white noise of length 40,000 (the first 10,000 samples are discarded). Features are standardized per feature and perturbed by Gaussian noise equal to 1% of their standard deviation unless stated. A ridge readout is fitted on the first 40% of the retained samples, its regularization is chosen on the next 20%, and R² is measured on the last 40%.'))
X.append(B([('Memory capacity (MC). ','b'),'The targets are u[t − d] for the 24 lags d in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096}. R² is clipped to [0, 1], interpolated linearly over all lags 1 to 4096 and integrated (trapezoid).']))
X.append(B([('Scale-free recall (SFR). ','b'),'For each of 20 delays d, geometrically spaced from 2 to 4096 and rounded, the target is the sum of u over a window of length s = max(1, ⌊d/4⌋) that ends at lag d, so that the resolution is proportional to the lag. SFR is the mean of the clipped R² over the 20 delays.']))
X.append(H2('S3.3 Forecasting'))
X.append(P('The series are sums of three cosines with periods 24, 168 and 2,000 samples (random phases and amplitudes 1, 0.7 and 0.5), plus AR(1) noise with coefficient 0.98 scaled by 0.15 and white noise with standard deviation 0.3; length 70,000. The target is the value H steps ahead for H in {1, 24, 168, 720}. The readout is ridge regression on the standardized features; the error is the mean squared error divided by the variance of the target.'))
X.append(H2('S3.4 Classification'))
X.append(P('Signals have 1,024 samples at a sampling rate of 128 Hz. A class is one of three burst centers, 6, 10 and 20 Hz; the burst frequency is the center times a factor uniform in [0.85, 1.15], with random phase and an amplitude modulation 1 + 0.5 sin(2π f_m t + φ) with f_m uniform in [0.1, 0.5] Hz. The burst is scaled by the amplitude a (0.10 unless stated) and added to colored noise, the output of 1/(1 − 0.97 z⁻¹) driven by standard normal noise and scaled by 0.25, plus white noise with standard deviation 0.35. There are 450 training and 450 test signals. The features are the logarithms of the mean energy of each channel (or of each real feature) after the first 300 samples, standardized, and the classifier is a logistic regression with C = 1.'))
X.append(H2('S3.5 Selective recall'))
X.append(P('Each sequence has T independent standard normal values and three flagged time steps chosen at random. The target is the value at the last of the three flagged steps. Each model receives the values and the flags, computes a gate and a step from the flag by affine maps followed by a sigmoid and a softplus, with four parameters, and the readout is ridge regression on the final state. The four parameters are found by random search (80 draws) followed by hill climbing (80 steps) on sequences of length 128 (300 sequences), and the models are tested at T = 128, 512 and 1,024. The compared models are the time-invariant HT without gating, HT-A with the rotation as written, HT-A with the rotation scaled by the step, and a selective real-pole model with S4D-real poles.'))
X.append(H2('S3.6 Cost benchmark'))
X.append(P('One layer with d = 64 channels and N = 16 modes is applied to a random sequence of length T, with the default HT poles and horizons up to 4,096. The forms timed are the FFT convolution with the summed kernel, the streaming recurrence (one scipy lfilter per mode), the parallel scan of HT-A, and a single-head softmax attention layer (queries, keys and values equal to the input, scaled by 1/√d, float32). The reported latency is the median of five repetitions after one warm-up call, and the memory is the peak allocation measured by tracemalloc.'))
# ============ S4
X.append(H1('S4 Bank definitions'))
X.append(H2('S4.1 The default HT bank'))
from lib import ht_default
h,w=ht_default(32,F=2,hmax=4096.0)
X.append(P('The default bank has 32 complex channels, that is 64 real features: M = 16 horizons geometrically spaced from 1 to 4,096 for the memory probes of Chapter 8, with F = 2 frequencies per horizon, ω = min(π, π(j + 1)/(2h)) for j = 0, 1. Table S4.1 lists the parameters. The pole of channel k is λ = exp(−1/h + iω) and the gain is b = √(1 − |λ|²).'))
rows=[[str(k),f'{h[k]:.4g}',f'{w[k]:.5g}',f'{np.exp(-1/h[k]):.6f}'] for k in range(32)]
X.append(tbl('Table S4.1. Horizons, angular frequencies and pole radii of the default 32-channel bank (F = 2, h_max = 4,096)',['Channel k','Horizon h (samples)','ω (rad/sample)','|λ|'],rows,[1500,2500,2700,2326]))
X.append(H2('S4.2 Baseline banks'))
rows=[['S4D-Lin','Four timescales Δ geometric from 10⁻³ to 10⁻¹; per timescale N = nfeat/8 complex modes with A = −0.5 + iπn, n = 0 to N − 1; λ = exp(ΔA)'],['LRU (ring init)','Radii r with r² uniform in [0.9², 0.999²]; phases uniform in [0, π/10]; gain √(1 − |λ|²); seeded'],['HiPPO-LegT','Finite window W = 4,096 (or as stated) with orthonormal Legendre polynomials as FIR kernels'],['STFT','Causal sliding Hann window of length W (256 or 1,024, or 128 in the classification task), nfeat/2 bins, real and imaginary parts, hop 1'],['Morlet bank (CWT)','One-sided delayed Morlet atoms with w0 = 6; scales geometric from 1 to h_max/6; each atom has unit energy'],['Delay line','Identity FIR kernel: feature j is u[t − j]'],['HT real poles','Real poles only, horizons geometric from 1 to 4,096, one real feature per channel'],['Orthogonalized HT','Features multiplied by G^(−1/2), where G is the Gram matrix of the 32,768-tap kernels; eigenvalues floored at 10⁻¹² of the largest']]
X.append(tbl('Table S4.2. Baseline and variant banks as coded in lib.py',['Bank','Definition'],rows,[2200,6826],left=(1,)))
# ============ S5
X.append(H1('S5 Full numerical results'))
X.append(P('Entries are means over the stated number of seeds, with standard deviations (±) or ranges where available.'))
X.append(H2('S5.1 Property checks (e0)'))
lab={'rec_vs_conv':'Maximum difference, recurrence against convolution form','rec_vs_fft':'Maximum difference, recurrence against FFT form','inv_err_K16':'Inverse error, 16 channels','inv_err_K1_h4096':'Inverse error, 1 channel, h = 4,096','noise_bias':'Bias of the inverse estimator under noise','noise_var_emp':'Empirical noise variance of the estimator','noise_var_theory':'Predicted noise variance','scales_needed_4096':'Scales needed to cover horizons up to 4,096','maxh_K12':'Longest horizon reached with 12 scales','residual_weight_at_T_K12':'Residual weight at the sequence length, 12 scales','residual_weight_at_T_K13':'Residual weight at the sequence length, 13 scales','scan_err':'Maximum difference, HT-A parallel scan against sequential','hta_nonunique_frac_K1':'HT-A inverse: fraction of non-unique cases, 1 channel','hta_maxerr_K1':'HT-A inverse: maximum error, 1 channel','hta_nonunique_frac_K4':'HT-A inverse: fraction of non-unique cases, 4 channels','hta_maxerr_K4':'HT-A inverse: maximum error, 4 channels'}
rows=[[lab.get(k,k),f'{v:.4g}' if isinstance(v,float) else str(v)] for k,v in e0.items() if not isinstance(v,(list,dict))]
X.append(tbl('Table S5.1. Scalar results of the property checks',['Quantity','Value'],rows,[6400,2626],left=(0,)))
X.append(H2('S5.2 Reconstruction (e6)'))
fams=['AR1','Tones+chirp+impulses','Piecewise-constant','White noise']; fl=['AR(1)','Tones','Piecewise','White']
for rho in ('8','16'):
    rows=[]
    for nm in e6:
        if rho not in e6[nm]: continue
        for dec in e6[nm][rho][fams[0]]:
            v=[e6[nm][rho][f][dec] for f in fams]; rows.append([f'{nm}, {dec}']+[f'{x:.2f}' for x in v]+[f'{np.mean(v):.2f}'])
    X.append(tbl(f'Table S5.2{"a" if rho=="8" else "b"}. Output SNR (dB) at ρ = {rho}, 20 dB noise per channel',['Transform and decoder']+fl+['Mean'],rows,[3200,1100,1100,1250,1100,1276],left=(0,)))
X.append(H2('S5.3 Memory probes (e2)'))
rows=[[k,ms(v['mc'],1),ms(v['sfr'],3)] for k,v in e2.items() if not k.startswith('_')]
X.append(tbl('Table S5.3. Memory capacity and scale-free recall, 64 features, 5 seeds, 1% noise',['Bank','Memory capacity','Scale-free recall'],rows,[4200,2400,2426],left=(0,)))
rows=[[k,ms(v['mc'],1),ms(v['sfr'],3)] for k,v in e2['_ablation'].items()]
X.append(tbl('Table S5.4. Ablations of the default HT bank',['Setting','Memory capacity','Scale-free recall'],rows,[4200,2400,2426],left=(0,)))
X.append(H2('S5.4 Forecasting and classification (e3)'))
hz=[1,24,168,720]; rows=[[k]+[f'{m:.3f} ± {s:.3f}' for m,s in zip(v['mean'],v['std'])] for k,v in e3['forecast'].items() if not k.startswith('_')]
X.append(tbl('Table S5.5. Normalized forecasting error by horizon (mean ± SD)',['Bank']+[f'H = {x}' for x in hz],rows,[3200,1456,1456,1456,1458],left=(0,)))
rows=[[k,f'{100*v["mean"]:.1f} ± {100*v["std"]:.1f}'] for k,v in e3['classify'].items()]
X.append(tbl('Table S5.6. Classification accuracy (%), burst amplitude 0.10',['Bank','Accuracy'],rows,[5600,3426],left=(0,)))
X.append(H2('S5.5 Selective recall (e4)'))
rows=[]
for m,v in e4.items():
    rows.append([m]+[f'{np.median(v[T]):.3f} [{min(v[T]):.3f}, {max(v[T]):.3f}]' for T in ('128','512','1024')])
X.append(tbl('Table S5.7. Held-out R² on the selective-recall task: median [minimum, maximum] over 5 seeds',['Model','T = 128','T = 512','T = 1,024'],rows,[3200,1942,1942,1942],left=(0,)))
X.append(H2('S5.6 Timing (e13)'))
rows=[]
for k,v in e13['out'].items():
    for T in e13['Ts']:
        t=np.array(v[str(T)]['t'])*1e3; rows.append([k,f'{T:,}',f'{np.median(t):.2f}',f'{t.min():.2f}–{t.max():.2f}',f'{v[str(T)]["mem"]:.1f}'])
X.append(tbl('Table S5.8. Latency (ms) and peak memory (MiB) with five repetitions',['Method','T','Median ms','Min–max ms','Peak MiB'],rows,[2800,1000,1500,2200,1526],left=(0,)))
X.append(H2('S5.7 Paired statistics and sweeps (e10, e11)'))
rows=[]
for metric,lab_ in (('mc','memory capacity'),('sfr','scale-free recall')):
    for r in e11a['paired'][metric]: rows.append([lab_,r['vs'],f'{r["mean"]:+.3f}',f'[{r["lo"]:+.3f}, {r["hi"]:+.3f}]',f'{r["p"]:.4f}',f'{r["p_holm"]:.4f}'])
X.append(tbl('Table S5.9. Paired differences, default HT minus baseline, 5 seeds (95% t interval; p and Holm-adjusted p)',['Measure','Baseline','Mean','95% interval','p','p (Holm)'],rows,[1500,1800,1100,2100,1200,1326],left=(0,1)))
rows=[]
for k,v in e11a['per_seed'].items(): rows.append([k,'MC']+[f'{x:.1f}' for x in v['mc']]); rows.append([k,'SFR']+[f'{x:.3f}' for x in v['sfr']])
X.append(tbl('Table S5.10. Per-seed values behind Table S5.9',['Bank','Measure','Seed 0','Seed 1','Seed 2','Seed 3','Seed 4'],rows,[2300,900,1166,1166,1166,1166,1162],left=(0,)))
nz=e11a['noise']['levels']; rows=[[k]+[f'{x:.1f}' for x in v['mc']]+[f'{x:.2f}' for x in v['sfr']] for k,v in e11a['noise']['banks'].items()]
X.append(tbl('Table S5.11. Sensitivity to observation noise (3 seeds)',['Bank']+[f'MC {n:g}' for n in nz]+[f'SFR {n:g}' for n in nz],rows,[2200,1000,1000,1000,1275,1275,1276],left=(0,)))
rows=[[k]+[f'{x:.1f}' for x in v['mc']]+[f'{x:.2f}' for x in v['sfr']] for k,v in e11b['banks'].items()]
X.append(tbl('Table S5.12. Sensitivity to the number of features (3 seeds)',['Bank']+[f'MC {n}' for n in e11b['nf']]+[f'SFR {n}' for n in e11b['nf']],rows,[1700]+[917]*8,left=(0,)) if False else tbl('Table S5.12. Sensitivity to the number of features (3 seeds)',['Bank']+[f'MC {n}' for n in e11b['nf']]+[f'SFR {n}' for n in e11b['nf']],rows,scale([1800]+[900]*8),left=(0,)))
rows=[[k]+[f'{100*np.mean(v[str(a)]):.1f} ± {100*np.std(v[str(a)]):.1f}' for a in e12['amps']] for k,v in e12['acc'].items()]
X.append(tbl('Table S5.13. Classification accuracy (%) against burst amplitude (3 seeds)',['Bank']+[f'a = {a}' for a in e12['amps']],rows,scale([2200]+[1300]*5),left=(0,)))
X.append(H2('S5.8 Repairs and limits (e14, e15)'))
rows=[[k,ms(v['0.01']['mc'],1),ms(v['0.01']['sfr'],3),ms(v['0.1']['mc'],1),ms(v['0.1']['sfr'],3)] for k,v in e14['res'].items()]
X.append(tbl('Table S5.14. Repaired banks (64 features, 5 seeds); noise 1% and 10%',['Bank','MC, 1%','SFR, 1%','MC, 10%','SFR, 10%'],rows,[2826,1500,1500,1600,1600],left=(0,)))
rows=[]
for k,dd in e15a['nonlinear'].items():
    for t,v in dd.items(): rows.append([k,t,f'{v["lin"][0]:.3f}',f'{v["nl"][0]:.3f} ± {v["nl"][1]:.3f}'])
X.append(tbl('Table S5.15. Nonlinear targets, held-out R² (3 seeds)',['Bank','Target','Linear readout','Second-order readout'],rows,[2200,2800,1700,2326],left=(0,1)))
kinds=e15b['kinds']; rows=[[k]+[f'{v[d]["mc"][0]:.1f}' for d in kinds]+[f'{v[d]["sfr"][0]:.2f}' for d in kinds] for k,v in e15b['res'].items()]
X.append(tbl('Table S5.16. Memory probes for four input distributions (3 seeds)',['Bank']+[f'MC {d.split(" ")[0]}' for d in kinds]+[f'SFR {d.split(" ")[0]}' for d in kinds],rows,scale([1700]+[900]*8),left=(0,)))
rows=[[k.split('|')[0],k.split('|')[1],f'{v["mc"]:.1f}',f'{v["sfr"]:.3f}'] for k,v in e15c.items()]
X.append(tbl('Table S5.17. Single-precision test of the orthogonalized bank (3 seeds)',['Variant','Noise','MC','SFR'],rows,[4300,1300,1700,1726],left=(0,)))
X.append(H2('S5.9 Pilot studies (e16)'))
rec=e16a['reconstruction']; rows=[[r,f'{v["HT default"]:.2f}',f'{v["HT default, orthogonalized"]:.2f}',f'{v["STFT (W=128)"]:.2f}',f'{v["cond default"]:.1f}',f'{v["cond orthogonalized"]:.1f}'] for r,v in rec.items()]
X.append(tbl('Table S5.18. Orthogonalized bank, reconstruction (output SNR in dB; condition numbers)',['ρ','HT default','HT orthogonalized','STFT','cond. default','cond. orthog.'],rows,[1100,1500,1900,1300,1600,1626]))
fc=e16a['forecast']; rows=[[k.split('|')[0],k.split('|')[1]]+[f'{x:.3f}' for x in v] for k,v in fc['nmse'].items()]
X.append(tbl('Table S5.19. Orthogonalized bank, forecasting error (3 seeds)',['Bank','Noise']+[f'H = {h_}' for h_ in fc['hz']],rows,[3400,1100,1500,1500,1526],left=(0,)))
rows=[[k.split('|')[0],k.split('|')[1],f'{100*v[0]:.1f} ± {100*v[1]:.1f}'] for k,v in e16a['classification'].items()]
X.append(tbl('Table S5.20. Orthogonalized bank, classification accuracy (%, 3 seeds)',['Variant','Amplitude','Accuracy'],rows,[4700,1700,2626],left=(0,)))
rows=[[str(F),str(M),f'{r:.2f}',f'{a:.1f}',f'{b:.2f}',f'{c:.1f}',f'{d_:.2f}'] for F,M,r,a,b,c,d_ in zip(e16b['F'],e16b['M'],e16b['r_eff'],e16b['default']['mc'],e16b['default']['sfr'],e16b['orthogonalized']['mc'],e16b['orthogonalized']['sfr'])]
X.append(tbl('Table S5.21. Scales against frequencies per scale (3 seeds)',['F','Scales M','r (effective)','MC default','SFR default','MC orthog.','SFR orthog.'],rows,[700,1100,1500,1400,1400,1450,1476]))
rows=[[k.split('|')[0],k.split('|')[1],f'{100*v[0]:.1f} ± {100*v[1]:.1f}'] for k,v in e16c.items() if '|' in k]
rows.append(['relative −3 dB bandwidth, orders 1 to 4',', '.join(f'{v:.2f}' for v in e16c['relative_bandwidth'].values()),''])
X.append(tbl('Table S5.22. Cascade order at equal total delay: classification accuracy (%, 3 seeds)',['Order','Amplitude','Accuracy'],rows,[3600,2600,2826],left=(0,1)))
rows=[]
for k,v in e16d.items(): c,n,p=k.split('|'); rows.append([c,n,p,f'{100*v[0]:.1f} ± {100*v[1]:.1f}'])
X.append(tbl('Table S5.23. Irregular sampling: accuracy (%, 2 seeds) against the fraction of samples retained',['Training','Method','Fraction retained','Accuracy'],rows,[2900,2400,1700,2026],left=(0,1)))
# ============ S6
X.append(H1('S6 Notation'))
rows=[['x[t]','Input sequence (scalar, or vector-valued channel-wise)'],['K','Number of complex channels (poles)'],['h_k','Horizon of channel k, in samples'],['ω_k','Angular frequency of channel k, rad/sample'],['λ_k','Pole of channel k, exp(−1/h_k + iω_k)'],['b_k','Gain of channel k; b_k = √(1 − |λ_k|²) in the experiments'],['X[t, k]','Coefficient of channel k at time t'],['Δ','Step (elapsed time) used by HT-A'],['F','Number of frequencies per horizon scale'],['r','Ratio between neighboring horizons'],['ρ','Redundancy: real coefficients per input sample'],['κ = B/A','Condition number of the frame operator (frame bounds A, B)'],['s(θ)','Frame symbol: sum over channels of |H_k(θ)|²'],['Q','Quality factor of a channel, ωh/2'],['T, N','Sequence length; signal length in the reconstruction experiments (256)'],['MC','Memory capacity (S3.2)'],['SFR','Scale-free recall (S3.2)'],['R²','Held-out coefficient of determination'],['NMSE','Mean squared error divided by the variance of the target'],['LS, GLS','Least-squares and generalized (noise-weighted) least-squares decoders'],['HT, HT-A','Horizon Transform; its input-adaptive variant'],['H1, H2, H3','Hypotheses of Chapter 7'],['SSM','State-space model']]
X.append(tbl('Table S6.1. Symbols and abbreviations',['Symbol','Meaning'],rows,[2000,7026],left=(1,)))
# ============ S7 index of figures and tables from the book
bd=zipfile=None
import zipfile
bx=zipfile.ZipFile(BOOK).read('word/document.xml').decode(); caps=[]
for m in para_iter(bx):
    t=ptext(m.group(0)); mm=re.match(r'^(Figure|Table) (\d+)\.(\d+)\. (.*)',t)
    if mm: caps.append((mm.group(1),int(mm.group(2)),int(mm.group(3)),mm.group(4)))
figscript={1:'figs_ch1.py',2:'figs_ch2.py',3:'figs_ch3.py',4:'figs_ch4.py',5:'figs_ch5.py',6:'figs_ch6.py',7:'figs_ch7.py',9:'figs_ch9.py',10:'figs_ch10.py',11:'figs_ch11.py',12:'figs_ch12.py'}
def short(s,n=150):
    s=s.split('. ')[0]; return s if len(s)<=n else s[:n-1].rstrip()+'…'
X.append(H1('S7 Index of figures and tables'))
rows=[]
for kind,ch,n,c in caps:
    if kind!='Figure': continue
    sc=figscript.get(ch) if ch!=8 else ('figs.py, figs2.py' if n<=8 else 'figs_ch8x.py')
    rows.append([f'{ch}.{n}',short(c),sc or '—'])
X.append(tbl('Table S7.1. Figures of the book, with the script that draws them',['Figure','Caption (first sentence)','Script'],rows,[800,6126,2100],left=(1,2)))
rows=[[f'{ch}.{n}',short(c,170)] for kind,ch,n,c in caps if kind=='Table']
X.append(tbl('Table S7.2. Tables of the book',['Table','Caption (first sentence)'],rows,[900,8126],left=(1,)))
# ============ S8
X.append(H1('S8 Change log and open items'))
X.append(H2('S8.1 Corrections and changes'))
rows=[['Section 3.1','K = 12 scales for horizons 1 to 4,096 corrected to K = 13 scales (h = 1, 2, 4, …, 4096)','The longest horizon with 12 scales is 2,048 (S5.1)'],['Eq. (3.6)','HT-A rotation not scaled by the step; step-scaled rotation recommended','Selective recall (Section 8.6)'],['Chapter 7, Section 7.7','Caveat on two timing repetitions above T = 2,048 replaced by a pointer to Section 8.13','Timings repeated with five repetitions'],['Section 9.6','Statement that the orthogonalized bank needs double precision replaced by the result of the single-precision test','Section 10.5'],['Section 9.6','Statement that other tasks were untested replaced by the pilot result','Section 11.7'],['Chapter 10','Section 10.6 (what the book does not claim) removed; Sections 10.7 and 10.8 renumbered 10.6 and 10.7','Author’s request'],['Chapter 12','Closing paragraph kept and moved to Section 12.7; third change (orthogonalization) added','—'],['Print edition','Review comment removed; equations rendered as images in the PDF; chapters start on right-hand pages','Section S9']]
X.append(tbl('Table S8.1. Corrections and changes made during preparation of the book',['Where','Change','Basis'],rows,[1700,5000,2326],left=(0,1,2)))
X.append(H2('S8.2 Items for the author to verify'))
for t in ['Chapter 2 adds 35 references. Six were checked against web sources during preparation (the Slaney report on the gammatone filter, Jaeger’s GMD report 152, Goodwin and Vetterli 1999, Jacobsen and Lyons 2003, the SWIFT paper, and LinOSS). The others were cited from memory and must be checked.','The author list of the SWIFT reference (IEEE Signal Processing Magazine, vol. 34, no. 5, pp. 183–188, 2017, doi 10.1109/MSP.2017.2718039) is missing.','The Word edition was not opened in Word during preparation; the Contents page numbers were computed from a LibreOffice rendering and may shift in Word (update the field).','The Word equations were not rendered during preparation; check them in Word. The print PDF shows them as images.','The figures print at about 210 dpi (167 dpi at the lowest), below the 300 dpi that KDP recommends.','The table of design lessons, the evidence ladder and the decision guide of Chapter 12 are rules of thumb from synthetic tasks, not validated results.']: X.append(B(t))
# ============ S9
X.append(H1('S9 Files'))
rows=[['Horizon_Transform.docx','The monograph (Word), A4, 12 chapters, 118 pages'],['Horizon_Transform_KDP_8.5x11.pdf','Print PDF for Amazon KDP, 8.5 × 11 in, 140 pages, chapters on right-hand pages'],['Horizon_Transform_KDP_8.5x11.docx','The same layout as a Word file'],['ht_experiments.zip','Scripts, result files (JSON) and figures'],['ht_code.zip','Scripts only, with requirements.txt'],['Horizon_Transform_Supplementary_Materials.docx and .pdf','This document']]
X.append(tbl('Table S9.1. Files',['File','Contents'],rows,[3600,5426],left=(0,1)))
def protect(x): return re.sub(r'\b[A-Za-z0-9]+(?:_[A-Za-z0-9.]+)+\.(?:py|json|md|txt|docx|pdf|zip|png)\b|figs_ch<number>|figs_ch\d+',lambda m:m.group(0).replace('_','UNDERSCOREPH'),x)
body=conv_lists(subscript_runs(protect(''.join(X)))).replace('UNDERSCOREPH','_')
d=open(SP+'word/document.xml',encoding='utf8').read(); head=d[:d.index('<w:body>')+len('<w:body>')]
SECT='<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/><w:pgSz w:w="12240" w:h="15840" w:orient="portrait"/><w:pgMar w:top="1440" w:right="1607" w:bottom="1440" w:left="1607" w:header="708" w:footer="708" w:gutter="0"/><w:pgNumType/><w:titlePg/><w:docGrid w:linePitch="360"/></w:sectPr>'
open(SP+'word/document.xml','w',encoding='utf8').write(head+body+SECT+'</w:body></w:document>')
import re as _re
_core=open(SP+'docProps/core.xml',encoding='utf8').read(); open(SP+'docProps/core.xml','w',encoding='utf8').write(_re.sub(r'<dc:title>.*?</dc:title>','<dc:title>Supplementary Materials: The Horizon Transform: An Exactly Invertible, Linear-Time Multi-Scale Layer for Long Sequences</dc:title>',_core))
# ---- contents and pagination
def pack():
    if os.path.exists(OUTDOCX): os.remove(OUTDOCX)
    subprocess.run(f"cd {SP} && zip -Xqr -D {OUTDOCX} '[Content_Types].xml' _rels docProps word",shell=True,check=True)
def render():
    r='/home/claude/sup/render'; shutil.rmtree(r,ignore_errors=True); os.makedirs(r); shutil.copy(OUTDOCX,r)
    subprocess.run(f'cd {r} && python3 /mnt/skills/public/docx/scripts/office/soffice.py --headless --convert-to pdf {os.path.basename(OUTDOCX)} >/dev/null 2>&1',shell=True)
    t=subprocess.run(['pdftotext','-layout',r+'/'+os.path.basename(OUTDOCX).replace('.docx','.pdf'),'-'],capture_output=True,text=True).stdout.split('\f')
    return (t[:-1] if t and not t[-1].strip() else t)
def locate(entries,pages):
    res={}
    for l,e in entries:
        key=e[:28]; res[e]=next(i+1 for i,p in enumerate(pages) if any(ln.strip().startswith(key) and '....' not in ln for ln in p.split('\n')))
    return res
entries=bk.rebuild_toc(None); pack(); pages=render(); pg=locate(entries,pages); entries=bk.rebuild_toc(pg); pack(); pages2=render(); pg2=locate(entries,pages2)
if pg2!=pg: entries=bk.rebuild_toc(pg2); pack(); pages2=render()
print('pages',len(pages2),'entries',len(entries))
