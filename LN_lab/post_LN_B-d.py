import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # <-- add this import

rng = np.random.default_rng(101)

# -------------------------------
# Config
# -------------------------------
eps = 0.0
sep = 1

# ---- Reduced x-range ----
X_LIM = 10
bins = np.linspace(-X_LIM, X_LIM, 361)

B_list = [32, 128, 512, 2048, 8192]
d_list = [32, 128, 512, 2048, 8192]
d_fixed = 768
B_fixed = 256

# IBM Design Library palette
palette = [
    "#0F62FE",  # IBM Blue
    "#8A3FFC",  # IBM Purple
    "#EE5396",  # IBM Magenta
    "#FF832B",  # IBM Orange
    "#F1C21B",  # IBM Yellow
]

DISTROS = ["normal", "laplace", "uniform", "student_t", "logistic", "exponential"]

# -------------------------------
# Helpers (unchanged)
# -------------------------------
T_DF_RANGE       = (3, 12)
UNIFORM_HW_BASE  = 0.8
EXP_LOGRATE_STD  = 0.9
SEP_LOC   = 3.0
SEP_SCALE = 1.0
LOG_SCALE_CENTER = np.log(2.0)
SIGMA_FLOOR = 0.0

def sample_channel(name: str, B: int, sep: float, rng: np.random.Generator,
                   sep_loc: float = SEP_LOC, sep_scale: float = SEP_SCALE,
                   t_df_range: tuple[int,int] = T_DF_RANGE,
                   uniform_hw_base: float = UNIFORM_HW_BASE,
                   exp_lograte_std: float = EXP_LOGRATE_STD) -> np.ndarray:
    mu_range3 = sep_loc * 3.0
    mu_range2 = sep_loc * 2.0

    def draw_scale(std_mult: float) -> float:
        s = np.exp(LOG_SCALE_CENTER + rng.normal(0.0, sep_scale * std_mult)) * sep
        return max(s, SIGMA_FLOOR)

    if name == "normal":
        mu = rng.uniform(-mu_range3, mu_range3)
        sigma = draw_scale(0.5)
        return rng.normal(mu, sigma, size=B)

    if name == "laplace":
        mu = rng.uniform(-mu_range3, mu_range3)
        b  = draw_scale(0.3)
        return rng.laplace(mu, b, size=B)

    if name == "uniform":
        center    = rng.uniform(-mu_range3, mu_range3)
        halfwidth = uniform_hw_base * draw_scale(0.7)
        a, b = center - halfwidth, center + halfwidth
        return rng.uniform(a, b, size=B)

    if name == "student_t":
        df_min, df_max = t_df_range
        df = rng.integers(df_min, df_max + 1)
        x  = rng.standard_t(df, size=B)
        mu = rng.uniform(-mu_range2, mu_range2)
        s  = draw_scale(0.6)
        return mu + s * x

    if name == "logistic":
        mu = rng.uniform(-mu_range3, mu_range3)
        s  = draw_scale(0.4)
        return rng.logistic(mu, s, size=B)

    if name == "exponential":
        lam = np.exp(rng.normal(0.0, exp_lograte_std * SEP_SCALE))
        x   = rng.exponential(1.0 / lam, size=B)
        mu  = rng.uniform(-mu_range3, mu_range3)
        s   = draw_scale(0.7)
        return mu + s * (x - (1.0 / lam))

def synthesize(B: int, d: int, sep: float, rng: np.random.Generator):
    chosen = rng.choice(DISTROS, size=d, replace=True)
    X = np.zeros((B, d), dtype=np.float32)
    for j, dj in enumerate(chosen):
        X[:, j] = sample_channel(dj, B, sep, rng).astype(np.float32)
    return X, chosen

def layer_norm_per_sample(X: np.ndarray, eps: float) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    var = ((X - mu)**2).mean(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)

def per_channel_histograms(X: np.ndarray, bins: np.ndarray):
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

def average_channel_pdf(Hs: np.ndarray, bins: np.ndarray):
    H_avg = Hs.mean(axis=0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    Z = (H_avg * widths).sum()
    if Z > 0:
        H_avg = H_avg / Z
    return centers, H_avg

def std_normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# -------------------------------
# Single plate (2 subplots side by side) — PRE (dotted) + POST (solid)
# -------------------------------
centers = 0.5 * (bins[:-1] + bins[1:])
ideal_y = std_normal_pdf(centers)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)

# Left: sweep over B (fixed d)
axL = axes[0]
for i, B in enumerate(B_list):
    color = palette[i % len(palette)]
    X, _ = synthesize(B, d_fixed, sep, rng)

    # --- pre-LN (dashed, lighter) ---
    _, Hs_pre = per_channel_histograms(X, bins)
    _, Hm_pre = average_channel_pdf(Hs_pre, bins)
    axL.plot(centers, Hm_pre, linestyle="--", linewidth=1.0, color=color, alpha=0.35)

    # --- post-LN (solid) ---
    Z = layer_norm_per_sample(X, eps=eps)
    _, Hs_post = per_channel_histograms(Z, bins)
    _, Hm_post = average_channel_pdf(Hs_post, bins)
    axL.plot(centers, Hm_post, linewidth=1.2, color=color, alpha=0.6, label=f"$B$={B}")

# Ideal on top
axL.plot(centers, ideal_y, color="black", linestyle="-",
         linewidth=1.8, alpha=0.6, zorder=10, label="Ideal $\mathcal{N}(0,1)$")

axL.set_xlim(-X_LIM, X_LIM)
axL.set_xlabel(f"Sweep over $B$  ($d$={d_fixed})", fontsize=16)
axL.set_ylabel("density", fontsize=16)

# ---- add proxy dashed line to legend for pre-LN ----
pre_proxy = Line2D([0], [0], linestyle="--", color="gray", linewidth=1.2, alpha=0.8, label="Pre-LN avg")
handles, labels = axL.get_legend_handles_labels()
handles.append(pre_proxy)
labels.append("Pre-LN avg")
# increase the distance between legend columns
axL.legend(handles, labels, frameon=False, fontsize=10, ncol=2, columnspacing=5)

# Right: sweep over d (fixed B)
axR = axes[1]
for i, d in enumerate(d_list):
    color = palette[i % len(palette)]
    X, _ = synthesize(B_fixed, d, sep, rng)

    # --- pre-LN (dashed, lighter) ---
    _, Hs_pre = per_channel_histograms(X, bins)
    _, Hm_pre = average_channel_pdf(Hs_pre, bins)
    axR.plot(centers, Hm_pre, linestyle="--", linewidth=1.0, color=color, alpha=0.35)

    # --- post-LN (solid) ---
    Z = layer_norm_per_sample(X, eps=eps)
    _, Hs_post = per_channel_histograms(Z, bins)
    _, Hm_post = average_channel_pdf(Hs_post, bins)
    axR.plot(centers, Hm_post, linewidth=1.2, color=color, alpha=0.6, label=f"$d$={d}")

# Ideal on top
axR.plot(centers, ideal_y, color="black", linestyle="-",
         linewidth=1.8, alpha=0.6, zorder=10, label="Ideal $\mathcal{N}(0,1)$")

axR.set_xlim(-X_LIM, X_LIM)
axR.set_xlabel(f"Sweep over $d$  ($B$={B_fixed})", fontsize=16)

# ---- add proxy dashed line to legend for pre-LN ----
pre_proxy_R = Line2D([0], [0], linestyle="--", color="gray", linewidth=1.2, alpha=0.8, label="Pre-LN avg")
handles_R, labels_R = axR.get_legend_handles_labels()
handles_R.append(pre_proxy_R)
labels_R.append("Pre-LN avg")
axR.legend(handles_R, labels_R, frameon=False, fontsize=10, ncol=2, columnspacing=5)

plt.tight_layout()
outfile = f"PrePost-LN_averages_side-by-side_xlim{X_LIM}_d{d_fixed}_B{B_fixed}_sep{sep}_eps{eps}.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"Saved {outfile}")
