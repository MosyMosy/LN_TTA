import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(33)

# --- Config (unchanged except titles/layout) ---
eps = 0.0
sep = 1.0
bins = np.linspace(-12.0, 12.0, 361)

B_list = [32, 128, 512, 2048, 8192]
d_list = [32, 128, 512, 2048, 8192]

# Paul Tol's "Muted" palette
tol_muted_list = [
    "#332288", "#88CCEE", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#6699CC"
]

DISTROS = [
    "normal", "laplace", "uniform", "student_t", "logistic", "exponential"
]

# -------------------------------
# Helpers (unchanged)
# -------------------------------
# ---- Separation knobs (global or pass as args) ----
T_DF_RANGE       = (3, 12)   # used for Student-t degrees of freedom
UNIFORM_HW_BASE  = 0.8       # baseline half-width for uniform channels
EXP_LOGRATE_STD  = 0.9       # std of log(rate) for exponential
SEP_LOC   = 3.0
SEP_SCALE = 1.0
LOG_SCALE_CENTER = np.log(2.0)   # ↑ this lifts the *median* width
SIGMA_FLOOR = 0.0                # set to e.g. 0.8 if you want a hard minimum

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
        lam = np.exp(rng.normal(0.0, exp_lograte_std * sep_scale))
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

def plot_panel_on_ax(ax, centers, H_all, H_avg, ideal_y, color_avg="black"):
    # background: all channel curves, colorful and transparent
    for j in range(H_all.shape[0]):
        color = tol_muted_list[j % len(tol_muted_list)]
        ax.plot(centers, H_all[j], color=color, alpha=0.30, linewidth=1)
    # ideal N(0,1): dashed black
    ax.plot(centers, ideal_y, color="black", linestyle="--", linewidth=1.2, alpha=0.80)
    # average channel marginal: solid black
    ax.plot(centers, H_avg, color=color_avg, linewidth=1.1, alpha=0.80)
    ax.grid(alpha=0.2, linewidth=0.5)

# -------------------------------
# Per-B figures: 2 rows (pre/post), N_d columns
# -------------------------------
centers = 0.5 * (bins[:-1] + bins[1:])
ideal_y = std_normal_pdf(centers)

for B in B_list:
    fig, axes = plt.subplots(
        nrows=2, ncols=len(d_list),
        figsize=(2.8 * len(d_list), 8),  # width scales with N_d
        sharex=True, sharey=True
    )
    if len(d_list) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # force 2x1 array

    for j, d in enumerate(d_list):
        # simulate once per (B,d)
        X, _ = synthesize(B, d, sep, rng)
        Z = layer_norm_per_sample(X, eps=eps)

        # densities
        centers_pre, H_pre = per_channel_histograms(X, bins)
        _, H_post = per_channel_histograms(Z, bins)
        _, Havg_pre = average_channel_pdf(H_pre, bins)
        _, Havg_post = average_channel_pdf(H_post, bins)

        # PRE row (row 0)
        ax_pre = axes[0, j]
        plot_panel_on_ax(ax_pre, centers_pre, H_pre, Havg_pre, ideal_y, color_avg="red")
        ax_pre.set_title(f"d={d}", fontsize=14)

        # POST row (row 1)
        ax_post = axes[1, j]
        plot_panel_on_ax(ax_post, centers_pre, H_post, Havg_post, ideal_y, color_avg="green")
        # ax_post.set_title(f"d={d}", fontsize=9)

    # axis labels
    for ax in axes[1, :]:
        ax.set_xlabel("value")
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("density")

    # per-figure legend
    # legend_elems = [
    #     Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Ideal N(0,1)"),
    #     Line2D([0], [0], color="black", linestyle="-",  linewidth=1.2, label="Average"),
    #     Line2D([0], [0], color=tol_muted_list[0], alpha=0.2, linewidth=1.0, label="Channels (bg)")
    # ]
    # fig.legend(handles=legend_elems, loc="upper right", frameon=False)

    # suptitle and save
    fig.suptitle(f"Per-channel marginal PDFs — B={B}  (sep={sep}, eps={eps})",
                 y=1.02, fontsize=18)
    plt.tight_layout()
    out_name = f"Per-channel_marginals_B-{B}_2x{len(d_list)}_grid_sep-{sep}_eps-{eps}.pdf"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    # plt.close(fig)  # uncomment if running many to free memory
    print(f"Saved {out_name}")
