"""
Smooth, borderless contour map of distance-to-ideal vs (B, d)
— with axes on LOG2 scale and ticks at exact powers of two (e.g., 2, 4, 8, 16, 32, ...).
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Config
# -----------------------------------
rng = np.random.default_rng(101)

# LN / data gen
eps = 0.0
sep = 1.0
X_LIM = 6.0
NBINS = 801
bins = np.linspace(-X_LIM, X_LIM, NBINS)

# Grid in log-space (wider range + more points)
B_MIN, B_MAX, NB_GRID = 8, 1024, 32
D_MIN, D_MAX, ND_GRID = 8, 1024, 32

REPEATS = 4
METRIC = 'js'  # 'w1' | 'tv' | 'js' | 'ks'
UPSAMPLE_NB = 300
UPSAMPLE_ND = 300

# -----------------------------------
# Generator + LN
# -----------------------------------
DISTROS = [
    "normal", "laplace", "uniform", "student_t", "logistic", "exponential"
]
SEP_LOC, SEP_SCALE = 1.0, 0.5
T_DF_RANGE = (3, 12)
UNIFORM_HW_BASE = 0.8
EXP_LOGRATE_STD = 0.5


def sample_channel(name,
                   B,
                   sep,
                   rng,
                   sep_loc=SEP_LOC,
                   sep_scale=SEP_SCALE,
                   t_df_range=T_DF_RANGE,
                   uniform_hw_base=UNIFORM_HW_BASE,
                   exp_lograte_std=EXP_LOGRATE_STD):
    if name == "normal":
        mu = rng.normal(0.0, sep_loc * 3.0)
        sigma = np.exp(rng.normal(0.0, sep_scale * 0.5)) * sep
        return rng.normal(mu, sigma, size=B)
    if name == "laplace":
        mu = rng.normal(0.0, sep_loc * 3.0)
        b = np.exp(rng.normal(0.0, sep_scale * 0.3)) * sep
        return rng.laplace(mu, b, size=B)
    if name == "uniform":
        center = rng.normal(0.0, sep_loc * 3.0)
        halfwidth = UNIFORM_HW_BASE * np.exp(rng.normal(0.0,
                                                        sep_scale * 0.7)) * sep
        a, b = center - halfwidth, center + halfwidth
        return rng.uniform(a, b, size=B)
    if name == "student_t":
        df = rng.integers(t_df_range[0], t_df_range[1] + 1)
        x = rng.standard_t(df, size=B)
        mu = rng.normal(0.0, sep_loc * 2.0)
        s = np.exp(rng.normal(0.0, sep_scale * 0.6)) * sep
        return mu + s * x
    if name == "logistic":
        mu = rng.normal(0.0, sep_loc * 3.0)
        s = np.exp(rng.normal(0.0, sep_scale * 0.4)) * sep
        return rng.logistic(mu, s, size=B)
    if name == "exponential":
        lam = np.exp(rng.normal(0.0, EXP_LOGRATE_STD * sep_scale))
        x = rng.exponential(1.0 / lam, size=B)
        mu = rng.normal(0.0, sep_loc * 3.0)
        s = np.exp(rng.normal(0.0, sep_scale * 0.7)) * sep
        return mu + s * (x - (1.0 / lam))


def synthesize(B, d, sep, rng):
    chosen = rng.choice(DISTROS, size=d, replace=True)
    X = np.zeros((B, d), dtype=np.float32)
    for j, dj in enumerate(chosen):
        X[:, j] = sample_channel(dj, B, sep, rng).astype(np.float32)
    return X, chosen


def layer_norm_per_sample(X, eps):
    mu = X.mean(axis=1, keepdims=True)
    var = ((X - mu)**2).mean(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)


def per_channel_histograms(X, bins):
    d = X.shape[1]
    Hs = np.zeros((d, len(bins) - 1), dtype=np.float64)
    for j in range(d):
        H, _ = np.histogram(X[:, j], bins=bins, density=True)
        Hs[j] = H
    widths = np.diff(bins)[None, :]
    Z = (Hs * widths).sum(axis=1, keepdims=True)
    Z[Z == 0] = 1.0
    Hs = Hs / Z
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, Hs


def average_channel_pdf(Hs, bins):
    H_avg = Hs.mean(axis=0)
    widths = np.diff(bins)
    Z = (H_avg * widths).sum()
    if Z > 0:
        H_avg = H_avg / Z
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, H_avg


def std_normal_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


# -----------------------------------
# Distances on the histogram grid
# -----------------------------------
def densities_to_mass(density, bins):
    widths = np.diff(bins)
    mass = density * widths
    s = mass.sum()
    return (mass / s) if s > 0 else mass, widths


def distance_metrics(post_density, ideal_density, bins, metric='w1'):
    p, widths = densities_to_mass(post_density, bins)
    q, _ = densities_to_mass(ideal_density, bins)

    if metric == 'tv':
        return 0.5 * np.abs(p - q).sum()
    if metric == 'js':
        eps = 1e-12
        p1, q1 = np.clip(p, eps, 1.0), np.clip(q, eps, 1.0)
        m = 0.5 * (p1 + q1)
        return 0.5 * (p1 * (np.log(p1) - np.log(m))).sum() + \
               0.5 * (q1 * (np.log(q1) - np.log(m))).sum()
    if metric == 'ks':
        Fp = np.cumsum(p)
        Fq = np.cumsum(q)
        return np.max(np.abs(Fp - Fq))
    # default: Wasserstein-1 (approximate)
    Fp = np.cumsum(p)
    Fq = np.cumsum(q)
    diffs = Fp - Fq
    mid = 0.5 * (np.concatenate(([diffs[0]], diffs[:-1])) + diffs)
    return np.sum(np.abs(mid) * widths)


# -----------------------------------
# Build the B×d grid (log-spaced)
# -----------------------------------
B_vals = np.unique(
    np.logspace(np.log10(B_MIN), np.log10(B_MAX), NB_GRID).astype(int))
d_vals = np.unique(
    np.logspace(np.log10(D_MIN), np.log10(D_MAX), ND_GRID).astype(int))

centers = 0.5 * (bins[:-1] + bins[1:])
ideal_density = std_normal_pdf(centers)

D = np.zeros((len(B_vals), len(d_vals)), dtype=np.float64)

for i, B in enumerate(B_vals):
    for j, d in enumerate(d_vals):
        dist_accum = 0.0
        for r in range(REPEATS):
            sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            X, _ = synthesize(int(B), int(d), sep, sub_rng)
            Z = layer_norm_per_sample(X, eps=eps)
            _, Hs = per_channel_histograms(Z, bins)
            _, Havg = average_channel_pdf(Hs, bins)
            dist_accum += distance_metrics(Havg,
                                           ideal_density,
                                           bins,
                                           metric=METRIC)
        D[i, j] = dist_accum / REPEATS


# -----------------------------------
# Bilinear upsampling in LOG-space
# -----------------------------------
def bilinear_upsample_rect(Z, x, y, xf, yf):
    Zy = np.empty((len(x), len(yf)), dtype=float)
    for i in range(len(x)):
        Zy[i] = np.interp(yf, y, Z[i])
    Zxy = np.empty((len(xf), len(yf)), dtype=float)
    for j in range(len(yf)):
        Zxy[:, j] = np.interp(xf, x, Zy[:, j])
    return Zxy


Bx = np.log10(B_vals)
Dx = np.log10(d_vals)
BxF = np.linspace(Bx.min(), Bx.max(), UPSAMPLE_NB)
DxF = np.linspace(Dx.min(), Dx.max(), UPSAMPLE_ND)

D_smooth = bilinear_upsample_rect(D, Bx, Dx, BxF, DxF)

# grids for plotting (back to linear space)
B_fine = 10**BxF
d_fine = 10**DxF
B_mesh, d_mesh = np.meshgrid(d_fine, B_fine)  # X columns=d, Y rows=B


# -----------------------------------
# Helpers: log2 ticks at powers of two
# -----------------------------------
def set_log2_pow2_ticks(ax, x_vals, y_vals):
    kx_min = int(np.ceil(np.log2(np.min(x_vals))))
    kx_max = int(np.floor(np.log2(np.max(x_vals))))
    ky_min = int(np.ceil(np.log2(np.min(y_vals))))
    ky_max = int(np.floor(np.log2(np.max(y_vals))))
    xticks = 2**np.arange(kx_min, kx_max + 1)
    yticks = 2**np.arange(ky_min, ky_max + 1)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{int(t)}" for t in xticks])
    ax.set_yticklabels([f"{int(t)}" for t in yticks])


# -----------------------------------
# Contour-based heat map (no smooth gradient)
# -----------------------------------
fig, ax = plt.subplots(figsize=(5, 2.5))

# Choose discrete contour levels (linear or log). Linear works well here.
N_LEVELS = 32
vmin = float(np.nanmin(D_smooth))
vmax = float(np.nanmax(D_smooth))
levels = np.linspace(vmin, vmax, N_LEVELS)

# Filled contours = the "heat map" (discrete color bands)
csf = ax.contourf(
    B_mesh, d_mesh, D_smooth,
    levels=levels,
    cmap="viridis",
    antialiased=True
)

# Optional: draw contour lines on top for crisp boundaries
cs = ax.contour(
    B_mesh, d_mesh, D_smooth,
    levels=levels,
    colors="k",
    linewidths=0.4,
    alpha=0.8
)

# (Optional) label a sparse subset of lines
# ax.clabel(cs, fmt="%.3g", inline=True, fontsize=7)

# Reference star
ax.scatter([384], [32], c="gold", s=100, edgecolors="orange",
           linewidths=0.4, zorder=3, marker="*")

# Colorbar with matching ticks
cbar = fig.colorbar(csf, ax=ax, pad=0.02)
cbar.set_label({
    'w1': 'Wasserstein-1 distance',
    'tv': 'Total variation distance',
    'js': 'Jensen–Shannon (nats)',
    'ks': 'KS statistic'
}[METRIC])
ticks = [float(levels[0]), float(levels[N_LEVELS // 2]), float(levels[-1])]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.2f}" for t in ticks])


# Log2 axes with power-of-two ticks
set_log2_pow2_ticks(ax, d_fine, B_fine)
ax.set_xlabel("feature dimension $d$ ($\log_2$)", fontsize=12)
ax.set_ylabel("batch size $B$ ($\log_2$)", fontsize=12)

plt.tight_layout()
plt.savefig(
    f"distance_contour_HEATMAP_log2_metric-{METRIC}_repeats{REPEATS}_bins{NBINS}_grid{len(B_vals)}x{len(d_vals)}.pdf",
    bbox_inches="tight"
)
print("Saved contour-based heat map with log2 power-of-two ticks.")
