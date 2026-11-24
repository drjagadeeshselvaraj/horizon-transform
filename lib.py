"""Shared library for Horizon Transform (HT) experiments. numpy/scipy only."""
import numpy as np, scipy.signal as ss
from numpy.fft import rfft, irfft

# ---------------------------------------------------------------- HT parametrisation
def ht_poles(h, w):
    return np.exp(-1.0/np.asarray(h, float) + 1j*np.asarray(w, float))

def ht_gain(lam):                       # unit-power normalisation for white input
    return np.sqrt(1.0 - np.abs(lam)**2)

def ht_default(K, F=2, hmin=1.0, hmax=4096.0, r=None):
    """Default HT bank: M=ceil(K/F) horizon scales (geometric), F frequencies per scale.
    w_{k,j} = min(pi, pi*(j+1)/(2 h_k))  (period 4h/(j+1): about one oscillation per horizon)."""
    M = int(np.ceil(K/F))
    h = np.geomspace(hmin, hmax, M) if r is None else hmin*float(r)**np.arange(M)
    hh = np.repeat(h, F)[:K]; jj = np.tile(np.arange(F), M)[:K]
    w = np.minimum(np.pi, np.pi*(jj+1)/(2*hh))
    return hh, w

# ---------------------------------------------------------------- banks
class Bank:
    """Causal linear feature bank with nfeat real outputs per time step."""
    def __init__(self, name, nfeat, lam=None, b=None, ker=None, real=False):
        self.name, self.nfeat, self.lam, self.b, self.ker, self.real = name, nfeat, lam, b, ker, real
    def kernel(self, Lk):                      # (Lk, nfeat) real FIR kernel (truncation of the IIR)
        if self.ker is not None:
            k = np.zeros((Lk, self.nfeat)); n = min(Lk, len(self.ker)); k[:n] = self.ker[:n]; return k
        tau = np.arange(Lk)[:, None]
        kc = self.b[None, :]*np.exp(tau*np.log(self.lam)[None, :])
        if self.real: return kc.real
        out = np.empty((Lk, 2*len(self.lam))); out[:, 0::2] = kc.real; out[:, 1::2] = kc.imag; return out
    def stream(self, u):
        L = len(u)
        if self.ker is None:                   # exact IIR recurrence X[t]=lam X[t-1]+b x[t]
            cols = []
            for lam, b in zip(self.lam, self.b):
                X = ss.lfilter([b], [1, -lam], u.astype(complex) if np.iscomplexobj(lam) else u)
                cols += [X.real] if self.real else [X.real, X.imag]
            return np.stack(cols, 1)
        n = 1 << int(np.ceil(np.log2(L + len(self.ker))))
        return irfft(rfft(u, n)[:, None]*rfft(self.ker, n, axis=0), n, axis=0)[:L]

def ht_bank(h, w, name="HT", real=False):
    lam = ht_poles(h, np.zeros_like(h) if real else w)
    return Bank(name, len(lam) if real else 2*len(lam), lam=lam, b=ht_gain(lam), real=real)

def bank_ht(nfeat, F=2, hmax=4096.0, r=None, name="HT (geometric)"):
    h, w = ht_default(nfeat//2, F=F, hmax=hmax, r=r); return ht_bank(h, w, name)

def bank_ht_real(nfeat, hmax=4096.0, name="HT real poles"):
    h = np.geomspace(1.0, hmax, nfeat); return ht_bank(h, np.zeros(nfeat), name, real=True)

def bank_s4d_lin(nfeat, nd=4, N=None, name="S4D-Lin (HiPPO-style)"):
    N = nfeat//2//nd; lams, bs = [], []
    for dt in np.geomspace(1e-3, 1e-1, nd):
        A = -0.5 + 1j*np.pi*np.arange(N); lam = np.exp(dt*A); lams.append(lam)
        bs.append(np.sqrt(np.maximum(1-np.abs(lam)**2, 1e-12)))
    lam = np.concatenate(lams); b = np.concatenate(bs)
    return Bank(name, 2*len(lam), lam=lam, b=b)

def bank_lru(nfeat, seed=0, rmin=0.9, rmax=0.999, phimax=np.pi/10, name="LRU (ring init)"):
    rng = np.random.default_rng(seed); K = nfeat//2
    r = np.sqrt(rng.uniform(rmin**2, rmax**2, K)); th = rng.uniform(0, phimax, K)
    lam = r*np.exp(1j*th); return Bank(name, 2*K, lam=lam, b=ht_gain(lam))

def bank_legt(nfeat, W=4096, name="HiPPO-LegT window"):
    from numpy.polynomial import legendre as Lg
    tau = np.arange(W); x = 1 - 2*(tau+0.5)/W; ker = np.empty((W, nfeat))
    for n in range(nfeat):
        c = np.zeros(n+1); c[n] = 1; ker[:, n] = Lg.legval(x, c)*np.sqrt((2*n+1)/W)
    return Bank(name, nfeat, ker=ker)

def bank_stft(nfeat, W=256, name=None, fmax_frac=1.0):
    """Causal sliding STFT (Hann window length W, hop 1): nfeat/2 bins, Re & Im."""
    nb = nfeat//2; tau = np.arange(W); w = np.hanning(W+2)[1:-1]
    bins = np.unique(np.round(np.linspace(1, W/2*fmax_frac, nb)).astype(int)); bins = np.resize(bins, nb) if len(bins) < nb else bins
    ker = np.empty((W, 2*nb))
    for i, j in enumerate(bins[:nb]):
        ker[:, 2*i] = w*np.cos(2*np.pi*j*tau/W)*np.sqrt(2/W); ker[:, 2*i+1] = -w*np.sin(2*np.pi*j*tau/W)*np.sqrt(2/W)
    return Bank(name or f"STFT (W={W})", 2*nb, ker=ker)

def bank_cwt(nfeat, smin=1.0, smax=None, w0=6.0, name="CWT (causal Morlet)", hmax=4096.0):
    """One-sided (delayed) Morlet bank; scales geometric so support spans up to ~hmax."""
    K = nfeat//2; smax = smax or hmax/6.0; sig = np.geomspace(smin, smax, K); Lk = int(6*smax)+1
    tau = np.arange(Lk); ker = np.zeros((Lk, 2*K))
    for i, s in enumerate(sig):
        m = tau <= 6*s; t = tau[m] - 3*s
        g = np.exp(-t**2/(2*s**2))*np.exp(1j*w0*t/s); g = g/np.sqrt(np.sum(np.abs(g)**2))
        ker[m, 2*i] = g.real; ker[m, 2*i+1] = g.imag
    return Bank(name, 2*K, ker=ker)

def bank_delay(nfeat, name="Delay line (FIR)"):
    return Bank(name, nfeat, ker=np.eye(nfeat))

# ---------------------------------------------------------------- ridge / memory capacity
LAGS = np.array([1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096])

def ridge_curves(Z, Y, ntr, nva, alphas=(1e-6,1e-4,1e-2,1,1e2)):
    """Fit multi-output ridge on first ntr rows; pick alpha per output on val; return test R^2 per output."""
    A = Z[:ntr]; w, V = np.linalg.eigh(A.T@A); B = V.T@(A.T@Y[:ntr])
    sc = ntr; best = None
    Zv, Yv = Z[ntr:ntr+nva], Y[ntr:ntr+nva]; Zt, Yt = Z[ntr+nva:], Y[ntr+nva:]
    r2v, r2t = [], []
    for a in alphas:
        W = V@(B/(w[:, None] + a*sc))
        for Zs, Ys, store in ((Zv, Yv, r2v), (Zt, Yt, r2t)):
            e = Ys - Zs@W; store.append(1 - e.var(0)/Ys.var(0))
    r2v, r2t = np.array(r2v), np.array(r2t)
    return r2t[np.argmax(r2v, 0), np.arange(Y.shape[1])]

def memory_capacity(bank, seed, L=40000, burn=10000, noise=1e-2, lags=LAGS, return_curve=False, stream=None):
    rng = np.random.default_rng(1000+seed); u = rng.standard_normal(L)
    F = (bank.stream(u) if stream is None else stream(u))[burn:]
    Z = (F-F.mean(0))/(F.std(0)+1e-12); Z = Z + noise*rng.standard_normal(Z.shape)
    Y = np.stack([u[burn-d:L-d] for d in lags], 1); n = len(Z)
    r2 = np.clip(ridge_curves(Z, Y, int(.4*n), int(.2*n)), 0, 1)
    mc = np.trapezoid(np.interp(np.arange(1, lags.max()+1), lags, r2), dx=1.0)
    return (mc, r2) if return_curve else mc

# ---- scale-free recall: target = mean of u over lags [d, d+s-1] with s = max(1, d//4)  (resolution proportional to lag)
SF_D = np.unique(np.round(np.geomspace(2, 4096, 20)).astype(int))
def scale_free_recall(bank, seed, L=40000, burn=10000, noise=1e-2, ds=SF_D, stream=None, return_curve=False):
    rng = np.random.default_rng(2000+seed); u = rng.standard_normal(L)
    F = (bank.stream(u) if stream is None else stream(u))[burn:]
    Z = (F-F.mean(0))/(F.std(0)+1e-12); Z = Z + noise*rng.standard_normal(Z.shape)
    C = np.concatenate([[0], np.cumsum(u)])          # C[i]=sum u[:i]
    t = np.arange(burn, L); cols = []
    for d in ds:
        s = max(1, d//4); cols.append(C[t-d+1] - C[t-d-s+1])   # sum u[t-d-s+1 .. t-d]
    Y = np.stack(cols, 1); n = len(Z)
    r2 = np.clip(ridge_curves(Z, Y, int(.4*n), int(.2*n)), 0, 1)
    return (r2.mean(), r2) if return_curve else r2.mean()
