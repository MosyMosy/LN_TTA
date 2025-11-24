# Plot ONE pre-LN average-channel density curve for B=256, d=512.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(101)

# -------------------------------
# Config
# -------------------------------
B = 256
d = 512
sep = 1.0
X_LIM = 10
bins = np.linspace(-X_LIM, X_LIM, 361)

# -------------------------------
# Data generator (same family as before)
# -------------------------------
DISTROS = ["normal","laplace","uniform","student_t","logistic","exponential"]
SEP_LOC, SEP_SCALE = 1.0, 0.5
T_DF_RANGE = (3,12)
UNIFORM_HW_BASE = 0.8
EXP_LOGRATE_STD = 0.5

def sample_channel(name, B, sep, rng,
                   sep_loc=SEP_LOC, sep_scale=SEP_SCALE,
                   t_df_range=T_DF_RANGE, uniform_hw_base=UNIFORM_HW_BASE,
                   exp_lograte_std=EXP_LOGRATE_STD):
    if name == "normal":
        mu = rng.normal(0.0, sep_loc * 3.0)
        sigma = np.exp(rng.normal(0.0, sep_scale * 0.5)) * sep
        return rng.normal(mu, sigma, size=B)
    if name == "laplace":
        mu = rng.normal(0.0, sep_loc * 3.0)
        b  = np.exp(rng.normal(0.0, sep_scale * 0.3)) * sep
        return rng.laplace(mu, b, size=B)
    if name == "uniform":
        center    = rng.normal(0.0, sep_loc * 3.0)
        halfwidth = UNIFORM_HW_BASE * np.exp(rng.normal(0.0, sep_scale * 0.7)) * sep
        a, b = center - halfwidth, center + halfwidth
        return rng.uniform(a, b, size=B)
    if name == "student_t":
        df = rng.integers(t_df_range[0], t_df_range[1] + 1)
        x  = rng.standard_t(df, size=B)
        mu = rng.normal(0.0, sep_loc * 2.0)
        s  = np.exp(rng.normal(0.0, sep_scale * 0.6)) * sep
        return mu + s * x
    if name == "logistic":
        mu = rng.normal(0.0, sep_loc * 3.0)
        s  = np.exp(rng.normal(0.0, sep_scale * 0.4)) * sep
        return rng.logistic(mu, s, size=B)
    if name == "exponential":
        lam = np.exp(rng.normal(0.0, EXP_LOGRATE_STD * sep_scale))
        x   = rng.exponential(1.0 / lam, size=B)
        mu  = rng.normal(0.0, sep_loc * 3.0)
        s   = np.exp(rng.normal(0.0, sep_scale * 0.7)) * sep
        return mu + s * (x - (1.0 / lam))

def synthesize(B, d, sep, rng):
    chosen = rng.choice(DISTROS, size=d, replace=True)
    X = np.zeros((B, d), dtype=np.float32)
    for j, dj in enumerate(chosen):
        X[:, j] = sample_channel(dj, B, sep, rng).astype(np.float32)
    return X, chosen

def per_channel_histograms(X, bins):
    d = X.shape[1]
    Hs = np.zeros((d, len(bins) - 1), dtype=np.float64)
    for j in range(d):
        H, _ = np.histogram(X[:, j], bins=bins, density=True)
        Hs[j] = H
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)[None, :]
    Z = (Hs * widths).sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    Hs = Hs / Z
    return centers, Hs

def average_channel_pdf(Hs, bins):
    H_avg = Hs.mean(axis=0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    Z = (H_avg * widths).sum()
    if Z > 0:
        H_avg = H_avg / Z
    return centers, H_avg

# -------------------------------
# Compute pre-LN average density and plot
# -------------------------------
X, _ = synthesize(B, d, sep, rng)
centers, Hs_pre = per_channel_histograms(X, bins)
_, Havg_pre = average_channel_pdf(Hs_pre, bins)

plt.figure(figsize=(5.8, 3.6))
plt.plot(centers, Havg_pre, color="#0F62FE", linewidth=1.8)  # IBM blue
plt.xlim(-X_LIM, X_LIM)
plt.xlabel("value")
plt.ylabel("density")
plt.title(f"Pre-LN average channel marginal (B={B}, d={d})")
plt.tight_layout()
outfile = f"preLN_avg_B{B}_d{d}_sep{sep}_xlim{X_LIM}.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"Saved {outfile}")
