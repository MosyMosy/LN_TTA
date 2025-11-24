# ln_theory_empirical_suite.py
# Empirical tests for LN theory: shift/gain invariances, surviving components,
# angular behavior, coordinate concentration, epsilon sensitivity, sign flip,
# linear probing, concentration of per-token mean/RMS, ablation heatmap,
# mixture robustness, and confidence bands.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

# ============================================================
# Global Config
# ============================================================
SAVE_DPI = 200
FIGSIZE_SINGLE = (6, 3.2)
FIGSIZE_WIDE = (8.5, 3.0)
FIGSIZE_TALL = (6.0, 5.5)
PALETTE = [
    "#0F62FE",  # blue
    "#8A3FFC",  # purple
    "#EE5396",  # magenta
    "#FF832B",  # orange
    "#F1C21B",  # yellow
    "#24A148",  # green
]

# ============================================================
# Data synthesis helpers (from your setup, kept compatible)
# ============================================================
T_DF_RANGE       = (3, 12)
UNIFORM_HW_BASE  = 0.8
EXP_LOGRATE_STD  = 0.9
SEP_LOC   = 3.0
SEP_SCALE = 1.0
LOG_SCALE_CENTER = np.log(2.0)
SIGMA_FLOOR = 0.0

DISTROS = ["normal", "laplace", "uniform", "student_t", "logistic", "exponential"]

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

    raise ValueError(f"Unknown distro: {name}")

def synthesize(B: int, d: int, sep: float, rng: np.random.Generator):
    chosen = rng.choice(DISTROS, size=d, replace=True)
    X = np.zeros((B, d), dtype=np.float64)
    for j, dj in enumerate(chosen):
        X[:, j] = sample_channel(dj, B, sep, rng).astype(np.float64)
    return X, chosen

def layer_norm_per_sample(X: np.ndarray, eps: float) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    var = ((X - mu)**2).mean(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)

def std_normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

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

# ============================================================
# Utilities for domain block decomposition tests
# ============================================================
def make_basis(d, rng):
    u = np.ones((d,1))/np.sqrt(d)
    # Make U orthonormal and orthogonal to u
    Q = rng.normal(size=(d, d-1))
    Q = Q - u @ (u.T @ Q)
    Q, _ = np.linalg.qr(Q)
    U = Q[:, :d-1]
    return u, U

def T_from_blocks(u, U, a, eta, zeta, H):
    # assemble in basis [u, U]
    B = np.block([[np.array([[a]]), eta[None,:]], [zeta[:,None], H]])
    P = np.concatenate([u, U], axis=1)  # dxd
    return P @ B @ P.T

def synth_domain(B, d, rng, a=1.0, eta=None, zeta=None, H=None, b=0.0):
    u, U = make_basis(d, rng)
    if eta is None:  eta  = np.zeros(d-1)
    if zeta is None: zeta = np.zeros(d-1)
    if H is None:    H    = np.eye(d-1)
    T = T_from_blocks(u, U, a, eta, zeta, H)
    s = rng.normal(size=(B, d))
    X = b + s @ T.T
    return X

def project_theta(X):
    # Return θ = Cx / ||Cx|| for each token (rows)
    mu = X.mean(axis=1, keepdims=True)
    Cx = X - mu
    norms = np.linalg.norm(Cx, axis=1, keepdims=True) + 1e-12
    return Cx / norms

# ============================================================
# Simple logistic regression (no external deps)
# ============================================================
def sigmoid(z): return 1/(1+np.exp(-z))

def fit_logistic_regression(X, y, lr=0.1, epochs=200, reg=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    B, d = X.shape
    w = rng.normal(scale=0.01, size=(d,))  # no bias, fold into X with 1 col if needed
    for _ in range(epochs):
        z = X @ w
        p = sigmoid(z)
        grad = (X.T @ (p - y)) / B + reg * w
        w -= lr * grad
    return w

def eval_logistic(X, y, w):
    p = sigmoid(X @ w)
    yhat = (p >= 0.5).astype(int)
    return (yhat == y).mean()

def standardize_train_test(Xtr, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-12
    return (Xtr - mu)/std, (Xte - mu)/std

# ============================================================
# 1) Shift invariance (b * 1)
# ============================================================
def test_shift_invariance(B=512, d=512, sep=1.0, eps=0.0, b_list=None, out="T1_shift_invariance.pdf"):
    if b_list is None:
        b_list = np.linspace(-10, 10, 9)
    X, _ = synthesize(B, d, sep, rng)
    rels = []
    for b in b_list:
        Z1 = layer_norm_per_sample(X, eps)
        Z2 = layer_norm_per_sample(X + b, eps)
        rel = np.linalg.norm(Z1 - Z2) / (np.linalg.norm(Z1) + 1e-12)
        rels.append(rel)
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(b_list, rels, marker="o")
    plt.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.6)
    plt.xlabel("Added common-mode shift b")
    plt.ylabel("relative Frobenius error")
    plt.title("Shift invariance of LN (pre-affine)")
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T1] Saved {out}  | max rel error={np.max(rels):.3e}")

# ============================================================
# 2) Global gain invariance and epsilon modulation
# ============================================================
def test_scale_invariance(B=512, d=512, sep=1.0, eps_list=None, a_list=None, out="T2_scale_invariance.pdf"):
    if eps_list is None:
        eps_list = [0.0, 1e-8, 1e-5, 1e-3, 1e-1]
    if a_list is None:
        a_list = [0.25, 0.5, 1.0, 2.0, 4.0]
    X, _ = synthesize(B, d, sep, rng)

    mu = X.mean(axis=1, keepdims=True)
    CX = X - mu
    CX_norm = np.linalg.norm(CX, axis=1, keepdims=True)
    ratios = {eps: [] for eps in eps_list}
    preds  = {eps: [] for eps in eps_list}

    for eps in eps_list:
        Zx = layer_norm_per_sample(X, eps)
        base = np.linalg.norm(Zx)
        n_t = np.linalg.norm(Zx, axis=1, keepdims=True)  # per-token norms (eps>0 varies)
        for a in a_list:
            Za = layer_norm_per_sample(a*X, eps)
            r = np.linalg.norm(Za) / (base + 1e-12)
            ratios[eps].append(r)
            # predicted ratio from f(a;x_t)
            f_t = a * np.sqrt(CX_norm**2 + d*eps) / np.sqrt((a**2)*CX_norm**2 + d*eps)
            # ratio of Frobenius norms with per-token scaling:
            pred = np.sqrt(np.sum((f_t**2 * n_t**2)) / (np.sum(n_t**2) + 1e-12))
            preds[eps].append(float(pred))

    plt.figure(figsize=FIGSIZE_SINGLE)
    for i, eps in enumerate(eps_list):
        color = PALETTE[i % len(PALETTE)]
        plt.plot(a_list, ratios[eps], marker="o", color=color, label=f"eps={eps:g} (meas.)", alpha=0.5)
        plt.plot(a_list, preds[eps], linestyle="--", color=color, alpha=0.7, label=f"eps={eps:g} (pred.)")
    plt.xlabel("global scale a")
    plt.ylabel("||LN(aX)||_F / ||LN(X)||_F")
    plt.title("Scale invariance & epsilon modulation")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T2] Saved {out}")

# ============================================================
# 3) What survives LN: block decomposition
# ============================================================
def test_block_decomposition(B=512, d=256, eps=0.0,
                                   out="T3_block_decomposition_FIXED.png",
                                   seed=7):
    local = np.random.default_rng(seed)
    # One shared basis [u, U] and one shared latent s
    u, U = make_basis(d, local)
    s = local.normal(size=(B, d))

    def X_from_blocks(b, a, eta, zeta, H):
        T = T_from_blocks(u, U, a, eta, zeta, H)
        return b + s @ T.T  # same s for all variants

    zeros = np.zeros(d-1)
    I = np.eye(d-1)

    # Baseline
    X0 = X_from_blocks(b=0.0, a=1.0, eta=zeros, zeta=zeros, H=I)
    Z0 = layer_norm_per_sample(X0, eps)

    # Variants: toggle one block at a time (same s, same basis)
    variants = {
        "b":    X_from_blocks(b=5.0, a=1.0, eta=zeros, zeta=zeros, H=I),
        "a":    X_from_blocks(b=0.0, a=2.0, eta=zeros, zeta=zeros, H=I),
        "eta":  X_from_blocks(b=0.0, a=1.0, eta=local.normal(size=d-1)*0.8, zeta=zeros, H=I),
        "zeta": X_from_blocks(b=0.0, a=1.0, eta=zeros, zeta=local.normal(size=d-1)*0.8, H=I),
        "H":    X_from_blocks(b=0.0, a=1.0, eta=zeros, zeta=zeros,
                              H=(lambda A:(A+A.T)/2 + 0.5*np.eye(d-1))(local.normal(size=(d-1,d-1)))),
    }

    rel_dists, cos_sims, labels = [], [], []
    for name, Xv in variants.items():
        Zv = layer_norm_per_sample(Xv, eps)
        rel = np.linalg.norm(Zv - Z0) / (np.linalg.norm(Z0) + 1e-12)
        # token-wise cosine between corresponding rows (since s is shared)
        num = np.sum(Zv * Z0, axis=1)
        den = np.linalg.norm(Zv, axis=1) * np.linalg.norm(Z0, axis=1) + 1e-12
        cos = (num / den).mean()
        rel_dists.append(rel); cos_sims.append(cos); labels.append(name)

    # Plot relative distances
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3.0))
    plt.bar(labels, rel_dists, color=[PALETTE[i%len(PALETTE)] for i in range(len(labels))])
    plt.ylabel("relative Frobenius distance to baseline")
    plt.title("Components affecting post-LN features (fixed s, fixed basis)")
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"[T3-FIXED] Saved {out}")
    for l, r, c in zip(labels, rel_dists, cos_sims):
        print(f"{l:>5s}  rel_dist={r:.4e}   mean cos={c:.4f}")

# ============================================================
# 4) Angular distribution in H: cosine histograms across domains
# ============================================================
def test_angular_distribution(B=800, d=256, eps=0.0, out="T4_angular_distribution.pdf"):
    # Domains differ only in zeta/H (should survive)
    X1 = synth_domain(B, d, rng, a=1.0, eta=np.zeros(d-1), zeta=np.zeros(d-1), H=np.eye(d-1), b=0.0)
    X2 = synth_domain(B, d, rng, a=1.0, eta=np.zeros(d-1), zeta=rng.normal(size=d-1)*0.7,
                      H=(lambda A: (A+A.T)/2 + 0.3*np.eye(d-1))(rng.normal(size=(d-1,d-1))), b=0.0)

    theta1 = project_theta(X1)
    theta2 = project_theta(X2)
    # Pair tokens by index to get cross-domain cosine
    cos12 = np.sum(theta1 * theta2, axis=1)

    # As controls: within-domain cosines (shuffle pairing)
    perm = rng.permutation(B)
    cos11 = np.sum(theta1 * theta1[perm], axis=1)
    cos22 = np.sum(theta2 * theta2[perm], axis=1)

    plt.figure(figsize=FIGSIZE_SINGLE)
    kwargs = dict(bins=50, density=True, alpha=0.5)
    plt.hist(cos11, color=PALETTE[0], label="within domain 1", **kwargs)
    plt.hist(cos22, color=PALETTE[1], label="within domain 2", **kwargs)
    plt.hist(cos12, color=PALETTE[2], label="across domains", **kwargs)
    plt.xlabel("cosine similarity of θ")
    plt.ylabel("density")
    plt.title("Angular distribution in mean-free subspace")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T4] Saved {out}")

# ============================================================
# 5) Coordinate-wise concentration vs (B,d)
# ============================================================
def test_coordinate_concentration(d_list=(64,256,1024), B_list=(32,128,512,2048), sep=1.0, eps=0.0,
                                  out="T5_coordinate_concentration.pdf"):
    j = 0  # fixed coordinate index
    plt.figure(figsize=FIGSIZE_SINGLE)
    for i, d in enumerate(d_list):
        errs_mean = []
        errs_var  = []
        for B in B_list:
            X, _ = synthesize(B, d, sep, rng)
            Z = layer_norm_per_sample(X, eps)
            mean_j = Z[:, j].mean()
            var_j  = Z[:, j].var()
            errs_mean.append(abs(mean_j))
            errs_var.append(abs(var_j - 1.0))
        color = PALETTE[i%len(PALETTE)]
        plt.plot(B_list, errs_mean, marker="o", color=color, label=f"|mean_j|, d={d}")
        plt.plot(B_list, errs_var, marker="s", linestyle="--", color=color, alpha=0.8, label=f"|var_j-1|, d={d}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("B (log scale)")
    plt.ylabel("absolute error (log scale)")
    plt.title("Coordinate-wise concentration at LN output (ε=0)")
    plt.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T5] Saved {out}")

# ============================================================
# 6) Epsilon sensitivity under heavy tails
# ============================================================
def test_epsilon_heavy_tails(B=4000, d=128, eps_list=(0.0, 1e-8, 1e-5, 1e-3, 1e-2, 1e-1),
                             k_list=(3,4,5), out="T6_epsilon_heavy_tails.pdf"):
    # Student-t heavy tails
    def synth_student(B, d):
        X = np.zeros((B,d))
        for j in range(d):
            df = rng.integers(3, 10)
            X[:, j] = rng.standard_t(df, size=B)
        return X

    X = synth_student(B, d)
    probs = {k: [] for k in k_list}
    for eps in eps_list:
        Z = layer_norm_per_sample(X, eps)
        Z_flat = Z.ravel()
        for k in k_list:
            probs[k].append(np.mean(np.abs(Z_flat) > k))
    plt.figure(figsize=FIGSIZE_SINGLE)
    for i, k in enumerate(k_list):
        color = PALETTE[i%len(PALETTE)]
        plt.plot(eps_list, probs[k], marker="o", color=color, label=f"P(|Z|>{k})")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("epsilon (log)")
    plt.ylabel("tail probability (log)")
    plt.title("Heavy tails damped by ε at LN")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T6] Saved {out}")

# ============================================================
# 7) Negative scaling (sign flip)
# ============================================================
def test_negative_scaling(B=1000, d=256, sep=1.0, out="T7_negative_scaling.pdf"):
    X, _ = synthesize(B, d, sep, rng)
    th1 = project_theta(X)
    th2 = project_theta(-X)
    cos = np.sum(th1 * th2, axis=1)
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.hist(cos, bins=50, density=True, color=PALETTE[3], alpha=0.7)
    plt.axvline(-1.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("cos(θ(X), θ(-X))")
    plt.ylabel("density")
    plt.title("Sign flip under negative scaling")
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T7] Saved {out}  | mean cosine = {cos.mean():.3f}")

# ============================================================
# 8) Linear probe: domain identification pre vs post LN
# ============================================================
def test_linear_probe(Btr=2000, Bte=2000, d=256, out="T8_linear_probe.pdf"):
    # (a) Domains differ only by b (should vanish post-LN)
    X1_tr = synth_domain(Btr, d, rng, b=0.0)
    X2_tr = synth_domain(Btr, d, rng, b=5.0)
    X1_te = synth_domain(Bte, d, rng, b=0.0)
    X2_te = synth_domain(Bte, d, rng, b=5.0)

    # (b) Domains differ by zeta/H (should persist post-LN)
    X3_tr = synth_domain(Btr, d, rng, zeta=rng.normal(size=d-1)*0.7,
                         H=(lambda A:(A+A.T)/2 + 0.4*np.eye(d-1))(rng.normal(size=(d-1,d-1))))
    X4_tr = synth_domain(Btr, d, rng, zeta=rng.normal(size=d-1)*0.7,
                         H=(lambda A:(A+A.T)/2 + 0.4*np.eye(d-1))(rng.normal(size=(d-1,d-1))))
    X3_te = synth_domain(Bte, d, rng, zeta=rng.normal(size=d-1)*0.7,
                         H=(lambda A:(A+A.T)/2 + 0.4*np.eye(d-1))(rng.normal(size=(d-1,d-1))))
    X4_te = synth_domain(Bte, d, rng, zeta=rng.normal(size=d-1)*0.7,
                         H=(lambda A:(A+A.T)/2 + 0.4*np.eye(d-1))(rng.normal(size=(d-1,d-1))))

    # Build datasets
    def build(Xa, Xb, post_ln=False):
        if post_ln:
            Xa = layer_norm_per_sample(Xa, 0.0)
            Xb = layer_norm_per_sample(Xb, 0.0)
        X = np.vstack([Xa, Xb])
        y = np.hstack([np.zeros(Xa.shape[0], dtype=int), np.ones(Xb.shape[0], dtype=int)])
        return X, y

    # Train/eval four scenarios
    results = []
    labels  = ["pre-LN (b)", "post-LN (b)", "pre-LN (ζ/H)", "post-LN (ζ/H)"]
    # b-diff
    Xtr, ytr = build(X1_tr, X2_tr, post_ln=False)
    Xte, yte = build(X1_te, X2_te, post_ln=False)
    Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
    w = fit_logistic_regression(Xtr_s, ytr, lr=0.5, epochs=300, reg=1e-4, seed=1)
    acc_pre_b = eval_logistic(Xte_s, yte, w); results.append(acc_pre_b)

    Xtr, ytr = build(X1_tr, X2_tr, post_ln=True)
    Xte, yte = build(X1_te, X2_te, post_ln=True)
    Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
    w = fit_logistic_regression(Xtr_s, ytr, lr=0.5, epochs=300, reg=1e-4, seed=1)
    acc_post_b = eval_logistic(Xte_s, yte, w); results.append(acc_post_b)

    # zeta/H-diff
    Xtr, ytr = build(X3_tr, X4_tr, post_ln=False)
    Xte, yte = build(X3_te, X4_te, post_ln=False)
    Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
    w = fit_logistic_regression(Xtr_s, ytr, lr=0.5, epochs=300, reg=1e-4, seed=2)
    acc_pre_z = eval_logistic(Xte_s, yte, w); results.append(acc_pre_z)

    Xtr, ytr = build(X3_tr, X4_tr, post_ln=True)
    Xte, yte = build(X3_te, X4_te, post_ln=True)
    Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
    w = fit_logistic_regression(Xtr_s, ytr, lr=0.5, epochs=300, reg=1e-4, seed=2)
    acc_post_z = eval_logistic(Xte_s, yte, w); results.append(acc_post_z)

    plt.figure(figsize=FIGSIZE_SINGLE)
    cols = [PALETTE[i%len(PALETTE)] for i in range(4)]
    plt.bar(range(4), results, color=cols)
    plt.xticks(range(4), labels, rotation=20)
    plt.ylim(0.45, 1.0)
    plt.ylabel("accuracy")
    plt.title("Domain probe (linear) pre vs post LN")
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T8] Saved {out}  | accs: " + ", ".join(f"{l}:{v:.3f}" for l,v in zip(labels, results)))

# ============================================================
# 9) Per-token mean and RMS concentration for y = Γh + β
# ============================================================
def test_token_mean_rms_concentration(B=1024, d_list=(32,64,128,256,512,1024),
                                      out="T9_token_mean_rms_concentration.pdf"):
    # Fix Γ and β
    def gen_gamma_beta(d):
        gamma = np.ones(d)  # keep simple; theory still applies
        beta  = np.zeros(d)
        return gamma, beta

    vars_m, vars_s = [], []
    for d in d_list:
        X, _ = synthesize(B, d, sep=1.0, rng=rng)
        h = layer_norm_per_sample(X, eps=0.0)
        gamma, beta = gen_gamma_beta(d)
        y = h * gamma[None,:] + beta[None,:]
        m = y.mean(axis=1)                                  # per-token mean across features
        s = np.sqrt((y**2).mean(axis=1))                    # per-token RMS across features
        vars_m.append(np.var(m))
        vars_s.append(np.var(s))

    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.loglog(d_list, vars_m, marker="o", color=PALETTE[0], label="Var[m(y)]")
    plt.loglog(d_list, vars_s, marker="s", color=PALETTE[1], label="Var[s(y)]")
    plt.xlabel("d (log)")
    plt.ylabel("variance (log)")
    plt.title("Concentration of per-token mean and RMS vs d")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T9] Saved {out}")

# ============================================================
# 10) Removed vs preserved: pairwise distance heatmap
# ============================================================
def test_ablation_heatmap(B=600, d=128, out="T10_ablation_heatmap.pdf"):
    datasets = {}
    datasets["baseline"] = synth_domain(B, d, rng, a=1.0, eta=np.zeros(d-1), zeta=np.zeros(d-1), H=np.eye(d-1), b=0.0)
    datasets["b"]        = synth_domain(B, d, rng, a=1.0, eta=np.zeros(d-1), zeta=np.zeros(d-1), H=np.eye(d-1), b=4.0)
    datasets["a"]        = synth_domain(B, d, rng, a=2.0,  eta=np.zeros(d-1), zeta=np.zeros(d-1), H=np.eye(d-1), b=0.0)
    datasets["eta"]      = synth_domain(B, d, rng, a=1.0, eta=rng.normal(size=d-1)*0.6, zeta=np.zeros(d-1), H=np.eye(d-1), b=0.0)
    datasets["zeta"]     = synth_domain(B, d, rng, a=1.0, eta=np.zeros(d-1), zeta=rng.normal(size=d-1)*0.6, H=np.eye(d-1), b=0.0)
    datasets["H"]        = synth_domain(B, d, rng, a=1.0, eta=np.zeros(d-1),
                             zeta=np.zeros(d-1),
                             H=(lambda A:(A+A.T)/2 + 0.4*np.eye(d-1))(rng.normal(size=(d-1,d-1))), b=0.0)

    keys = list(datasets.keys())
    Zs = {k: layer_norm_per_sample(v, 0.0) for k, v in datasets.items()}
    n = len(keys)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            A, Bm = Zs[keys[i]], Zs[keys[j]]
            D[i, j] = np.linalg.norm(A - Bm) / (np.linalg.norm(A) + 1e-12)

    plt.figure(figsize=FIGSIZE_TALL)
    im = plt.imshow(D, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="relative Frobenius distance")
    plt.xticks(range(n), keys, rotation=20)
    plt.yticks(range(n), keys)
    plt.title("Post-LN pairwise distances across domain-component toggles")
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T10] Saved {out}")

# ============================================================
# 11) Mixture-of-domains robustness (post-LN probe acc vs mixture)
# ============================================================
def test_mixture_robustness(Btr=3000, Bte=3000, d=256, out="T11_mixture_robustness.pdf"):
    # Domains differing only in b and only in a
    dom_b_0 = lambda B: synth_domain(B, d, rng, b=0.0)
    dom_b_1 = lambda B: synth_domain(B, d, rng, b=5.0)
    dom_a_0 = lambda B: synth_domain(B, d, rng, a=1.0)
    dom_a_1 = lambda B: synth_domain(B, d, rng, a=2.0)

    ps = np.linspace(0.0, 1.0, 6)  # proportion of domain 1
    acc_b, acc_a = [], []

    def mix_data(B, p, dom0, dom1):
        B0 = int(B * (1-p)); B1 = B - B0
        Xa = dom0(B0); Xb = dom1(B1)
        X = np.vstack([Xa, Xb])
        y = np.hstack([np.zeros(B0, dtype=int), np.ones(B1, dtype=int)])
        return X, y

    for p in ps:
        # b-only difference
        Xtr, ytr = mix_data(Btr, p, dom_b_0, dom_b_1)
        Xte, yte = mix_data(Bte, p, dom_b_0, dom_b_1)
        Xtr = layer_norm_per_sample(Xtr, 0.0); Xte = layer_norm_per_sample(Xte, 0.0)
        Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
        w = fit_logistic_regression(Xtr_s, ytr, lr=0.5, epochs=300, reg=1e-4, seed=3)
        acc_b.append(eval_logistic(Xte_s, yte, w))

        # a-only difference
        Xtr, ytr = mix_data(Btr, p, dom_a_0, dom_a_1)
        Xte, yte = mix_data(Bte, p, dom_a_0, dom_a_1)
        Xtr = layer_norm_per_sample(Xtr, 0.0); Xte = layer_norm_per_sample(Xte, 0.0)
        Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
        w = fit_logistic_regression(Xtr_s, ytr, lr=0.5, epochs=300, reg=1e-4, seed=4)
        acc_a.append(eval_logistic(Xte_s, yte, w))

    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(ps, acc_b, marker="o", color=PALETTE[0], label="diff only in b")
    plt.plot(ps, acc_a, marker="s", color=PALETTE[1], label="diff only in a")
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.xlabel("mixture proportion p (domain 1)")
    plt.ylabel("post-LN probe accuracy")
    plt.title("Post-LN domain probe under domain mixing")
    plt.ylim(0.45, 0.7)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T11] Saved {out}")

# ============================================================
# 12) Visual tails + confidence bands across seeds
# ============================================================
def test_confidence_bands(B=512, d=512, sep=1.0, eps=0.0, seeds=10, X_LIM=6.0,
                          out="T12_confidence_bands.pdf"):
    bins = np.linspace(-X_LIM, X_LIM, 361)
    centers = 0.5 * (bins[:-1] + bins[1:])
    pre_stack = []
    post_stack = []
    for s in range(seeds):
        local_rng = np.random.default_rng(1000 + s)
        X, _ = synthesize(B, d, sep, local_rng)
        _, Hs_pre = per_channel_histograms(X, bins)
        _, Hm_pre = average_channel_pdf(Hs_pre, bins)
        Z = layer_norm_per_sample(X, eps)
        _, Hs_post = per_channel_histograms(Z, bins)
        _, Hm_post = average_channel_pdf(Hs_post, bins)
        pre_stack.append(Hm_pre)
        post_stack.append(Hm_post)
    pre_stack = np.stack(pre_stack, axis=0)
    post_stack = np.stack(post_stack, axis=0)
    pre_mean, pre_std = pre_stack.mean(axis=0), pre_stack.std(axis=0)
    post_mean, post_std = post_stack.mean(axis=0), post_stack.std(axis=0)
    ideal = std_normal_pdf(centers)

    plt.figure(figsize=FIGSIZE_SINGLE)
    # pre-LN band
    plt.fill_between(centers, pre_mean - pre_std, pre_mean + pre_std, color=PALETTE[2], alpha=0.2, label="pre-LN ±1σ")
    plt.plot(centers, pre_mean, color=PALETTE[2], linewidth=1.2)
    # post-LN band
    plt.fill_between(centers, post_mean - post_std, post_mean + post_std, color=PALETTE[0], alpha=0.2, label="post-LN ±1σ")
    plt.plot(centers, post_mean, color=PALETTE[0], linewidth=1.2)
    # ideal
    plt.plot(centers, ideal, color="black", linestyle="--", linewidth=1.5, label="Ideal N(0,1)")
    plt.xlim(-X_LIM, X_LIM)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.title(f"Average channel PDF: mean ± std across {seeds} seeds")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    print(f"[T12] Saved {out}")

# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    # test_shift_invariance()
    test_scale_invariance()
    # test_block_decomposition()
    # test_angular_distribution()
    # test_coordinate_concentration()
    # test_epsilon_heavy_tails()
    # test_negative_scaling()
    # test_linear_probe()
    # test_token_mean_rms_concentration()
    # test_ablation_heatmap()
    # test_mixture_robustness()
    # test_confidence_bands()
    # print("All figures saved.")
