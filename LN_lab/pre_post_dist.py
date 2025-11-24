# Grid version for per-channel PDFs:
# - Builds two figures (pre-LN grid, post-LN grid), each of size N_B x N_d
# - Each subplot shows ALL channel curves (transparent), the dashed Ideal N(0,1),
#   and the solid average channel marginal.
# - All parameters/styling are kept; only titles/figure layout are changed.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(33)

# --- Config (unchanged) ---
eps = 0.0
sep = 1.0
bins = np.linspace(-12.0, 12.0, 361)

# Grids to sweep
B_list = [8, 32, 256, 1024, 4096, 32768]
d_list = [32, 128, 512, 2048, 8192, 32768]

# Paul Tol's "Muted" palette
tol_muted_list = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#6699CC"
]

DISTROS = [
    "normal", "laplace", "uniform", "student_t", "logistic", "exponential"
]

# -------------------------------
# Helpers (unchanged)
# -------------------------------
def sample_channel(name: str, B: int, sep: float,
                   rng: np.random.Generator) -> np.ndarray:
    if name == "normal":
        mu = rng.normal(0.0, 3.0)
        sigma = np.exp(rng.normal(0.0, 0.5)) * sep
        return rng.normal(mu, sigma, size=B)
    if name == "laplace":
        mu = rng.normal(0.0, 3.0)
        b = np.exp(rng.normal(0.0, 0.3)) * sep
        return rng.laplace(mu, b, size=B)
    if name == "uniform":
        center = rng.normal(0.0, 3.0)
        halfwidth = 0.5 * np.exp(rng.normal(0.0, 0.7)) * sep
        a, b = center - halfwidth, center + halfwidth
        return rng.uniform(a, b, size=B)
    if name == "student_t":
        df = rng.integers(3, 12)
        x = rng.standard_t(df, size=B)
        mu = rng.normal(0.0, 2.0)
        s = np.exp(rng.normal(0.0, 0.6)) * sep
        return mu + s * x
    if name == "logistic":
        mu = rng.normal(0.0, 3.0)
        s = np.exp(rng.normal(0.0, 0.4)) * sep
        return rng.logistic(mu, s, size=B)
    if name == "exponential":
        lam = np.exp(rng.normal(0.0, 0.5))  # rate
        x = rng.exponential(1.0 / lam, size=B)  # mean = 1/lam
        mu = rng.normal(0.0, 3.0)
        s = np.exp(rng.normal(0.0, 0.7)) * sep
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
    Z = (Hs * widths).sum(axis=1, keepdims=True)  # approximate integral per channel
    Z[Z == 0] = 1.0
    Hs = Hs / Z  # ensure each curve is a density
    return centers, Hs

def average_channel_pdf(Hs: np.ndarray, bins: np.ndarray):
    H_avg = Hs.mean(axis=0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    Z = (H_avg * widths).sum()
    if Z > 0:
        H_avg = H_avg / Z  # ensure density
    return centers, H_avg

def std_normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Panel plotter that draws on a given Axes
def plot_panel_on_ax(ax, centers, H_all, H_avg, ideal_y):
    # background: all channel curves, colorful and transparent
    for j in range(H_all.shape[0]):
        color = tol_muted_list[j % len(tol_muted_list)]
        ax.plot(centers, H_all[j], color=color, alpha=0.20, linewidth=1.0)

    # ideal N(0,1): dashed black
    ax.plot(centers, ideal_y, color="black", linestyle="--", linewidth=1.5)

    # average channel marginal: solid black
    ax.plot(centers, H_avg, color="black", linewidth=1.2)

    ax.grid(alpha=0.2, linewidth=0.5)

# -------------------------------
# Build grids (pre-LN and post-LN)
# -------------------------------
N_B, N_d = len(B_list), len(d_list)
centers = 0.5 * (bins[:-1] + bins[1:])
ideal_y = std_normal_pdf(centers)

# Pre-LN grid
fig_pre, axes_pre = plt.subplots(
    nrows=N_B, ncols=N_d, figsize=(2.8 * N_d, 2.2 * N_B), sharex=True, sharey=True
)
if N_B == 1 and N_d == 1:
    axes_pre = np.array([[axes_pre]])
elif N_B == 1:
    axes_pre = axes_pre[None, :]
elif N_d == 1:
    axes_pre = axes_pre[:, None]

# Post-LN grid
fig_post, axes_post = plt.subplots(
    nrows=N_B, ncols=N_d, figsize=(2.8 * N_d, 2.2 * N_B), sharex=True, sharey=True
)
if N_B == 1 and N_d == 1:
    axes_post = np.array([[axes_post]])
elif N_B == 1:
    axes_post = axes_post[None, :]
elif N_d == 1:
    axes_post = axes_post[:, None]

for i, B in enumerate(B_list):
    for j, d in enumerate(d_list):
        # simulate once per (B,d)
        X, _ = synthesize(B, d, sep, rng)
        Z = layer_norm_per_sample(X, eps=eps)

        # densities
        centers_pre, H_pre = per_channel_histograms(X, bins)
        _, H_post = per_channel_histograms(Z, bins)
        _, Havg_pre = average_channel_pdf(H_pre, bins)
        _, Havg_post = average_channel_pdf(H_post, bins)

        # draw panels
        ax_pre = axes_pre[i, j]
        plot_panel_on_ax(ax_pre, centers_pre, H_pre, Havg_pre, ideal_y)
        ax_pre.set_title(f"B={B}, d={d}", fontsize=9)

        ax_post = axes_post[i, j]
        plot_panel_on_ax(ax_post, centers_pre, H_post, Havg_post, ideal_y)
        ax_post.set_title(f"B={B}, d={d}", fontsize=9)

# Axis labels
for ax in axes_pre[-1, :]:
    ax.set_xlabel("value")
for ax in axes_pre[:, 0]:
    ax.set_ylabel("density")

for ax in axes_post[-1, :]:
    ax.set_xlabel("value")
for ax in axes_post[:, 0]:
    ax.set_ylabel("density")

# Figure-wide titles
fig_pre.suptitle(f"Per-channel marginal PDFs — pre-LN  (sep={sep}, eps={eps})",
                 y=1.02, fontsize=12)
fig_post.suptitle(f"Per-channel marginal PDFs — post-LN (sep={sep}, eps={eps})",
                  y=1.02, fontsize=12)

# Single legend for each figure
legend_elems = [
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Ideal N(0,1)"),
    Line2D([0], [0], color="black", linestyle="-",  linewidth=1.2, label="Average"),
    Line2D([0], [0], color=tol_muted_list[0], alpha=0.2, linewidth=1.0, label="Channels (bg)")
]
fig_pre.legend(handles=legend_elems, loc="upper right", frameon=False)
fig_post.legend(handles=legend_elems, loc="upper right", frameon=False)

plt.tight_layout()

# Save both grids
fname_pre  = f"Per-channel_marginals_pre-LN_grid_BxD_sep-{sep}_eps-{eps}.png"
fname_post = f"Per-channel_marginals_post-LN_grid_BxD_sep-{sep}_eps-{eps}.png"
plt.savefig  # keep reference usable
fig_pre.savefig(fname_pre,  dpi=300, bbox_inches="tight")
fig_post.savefig(fname_post, dpi=300, bbox_inches="tight")
# plt.show()  # uncomment when running interactively
