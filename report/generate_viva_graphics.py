import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 15
})

# Helper to create a heart path
def make_heart(center_x, center_y, scale=1.0):
    verts = [
        (center_x, center_y - 0.7 * scale),
        (center_x - 1.0 * scale, center_y),
        (center_x - 0.7 * scale, center_y + 0.8 * scale),
        (center_x, center_y + 0.4 * scale),
        (center_x + 0.7 * scale, center_y + 0.8 * scale),
        (center_x + 1.0 * scale, center_y),
        (center_x, center_y - 0.7 * scale),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)

# -------------------------------------------------------------
# 1. Disentanglement Concept: Entangled vs. Disentangled
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)

# Left: Entangled
ax1.set_title("Standard VAE: Entangled Latent Space", color="#b91c1c", fontweight="bold", pad=12)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.axis('off')

rect1 = patches.FancyBboxPatch((0.5, 0.5), 9, 7, boxstyle="round,pad=0.3",
                              ec="#fca5a5", fc="#fef2f2", lw=2)
ax1.add_patch(rect1)

ax1.text(5, 6.8, r"Tuning Single Latent Knob $z_1$", ha='center', va='center', fontweight='bold', fontsize=12.5, color="#991b1b")
ax1.text(5, 6.1, "(Coupled Attributes: Everything changes together!)", ha='center', va='center', fontstyle='italic', fontsize=10.5, color="#7f1d1d")

# Shape 1: small rotated red ellipse
e1 = patches.Ellipse((2.0, 4.0), 0.8, 1.6, angle=45, fc="#ef4444", ec="#b91c1c", lw=1.5)
ax1.add_patch(e1)
ax1.text(2.0, 2.3, r"$z_1 = -2.0$" + "\n• Small Scale\n• Tilted $45^\circ$\n• Position Left", ha='center', fontsize=9.5, color="#374151")

ax1.annotate('', xy=(3.8, 4.0), xytext=(2.9, 4.0), arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.5))

# Shape 2: medium upright square
r2 = patches.Rectangle((4.3, 3.3), 1.4, 1.4, fc="#3b82f6", ec="#1d4ed8", lw=1.5)
ax1.add_patch(r2)
ax1.text(5.0, 2.3, r"$z_1 = 0.0$" + "\n• Medium Scale\n• Upright $0^\circ$\n• Center", ha='center', fontsize=9.5, color="#374151")

ax1.annotate('', xy=(7.0, 4.0), xytext=(6.1, 4.0), arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.5))

# Shape 3: giant green circle shifted right
c3 = patches.Circle((8.0, 4.0), 1.1, fc="#10b981", ec="#047857", lw=1.5)
ax1.add_patch(c3)
ax1.text(8.0, 2.3, r"$z_1 = +2.0$" + "\n• Huge Scale\n• Position Right\n• Shape Changed!", ha='center', fontsize=9.5, color="#374151")

ax1.text(5, 1.0, "Problem: You cannot control Position without altering Shape and Scale!",
         ha='center', fontsize=10, fontweight='bold', color="#b91c1c",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#ef4444"))


# Right: Disentangled
ax2.set_title("AdaTC-DIP-VAE: Disentangled Latent Space", color="#15803d", fontweight="bold", pad=12)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')

rect2 = patches.FancyBboxPatch((0.5, 0.5), 9, 7, boxstyle="round,pad=0.3",
                              ec="#86efac", fc="#f0fdf4", lw=2)
ax2.add_patch(rect2)

ax2.text(5, 6.8, "Isolated Independent Latent Knobs", ha='center', va='center', fontweight='bold', fontsize=12.5, color="#166534")
ax2.text(5, 6.1, "(1 Latent Knob = 1 Ground-Truth Factor of Variation)", ha='center', va='center', fontstyle='italic', fontsize=10.5, color="#14532d")

# Three isolated square examples moving only horizontally
sq1 = patches.Rectangle((1.3, 3.3), 1.4, 1.4, fc="#3b82f6", ec="#1d4ed8", lw=1.5)
ax2.add_patch(sq1)
ax2.text(2.0, 2.3, r"$z_{\mathrm{pos}} = -2.0$" + "\n• Only Pos X: Left\n• Shape: Square (Fixed)\n• Scale: 1.0 (Fixed)", ha='center', fontsize=9.5, color="#374151")

ax2.annotate('', xy=(3.8, 4.0), xytext=(2.9, 4.0), arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2.5))

sq2 = patches.Rectangle((4.3, 3.3), 1.4, 1.4, fc="#3b82f6", ec="#1d4ed8", lw=1.5)
ax2.add_patch(sq2)
ax2.text(5.0, 2.3, r"$z_{\mathrm{pos}} = 0.0$" + "\n• Only Pos X: Center\n• Shape: Square (Fixed)\n• Scale: 1.0 (Fixed)", ha='center', fontsize=9.5, color="#374151")

ax2.annotate('', xy=(7.0, 4.0), xytext=(6.1, 4.0), arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2.5))

sq3 = patches.Rectangle((7.3, 3.3), 1.4, 1.4, fc="#3b82f6", ec="#1d4ed8", lw=1.5)
ax2.add_patch(sq3)
ax2.text(8.0, 2.3, r"$z_{\mathrm{pos}} = +2.0$" + "\n• Only Pos X: Right\n• Shape: Square (Fixed)\n• Scale: 1.0 (Fixed)", ha='center', fontsize=9.5, color="#374151")

ax2.text(5, 1.0, "Factorization: Altering Position X moves object cleanly across canvas!",
         ha='center', fontsize=10, fontweight='bold', color="#15803d",
         bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#22c55e"))

plt.tight_layout()
plt.savefig("report/viva_diagrams/disentanglement_concept.png", bbox_inches='tight')
plt.close()
print("Saved disentanglement_concept.png")


# -------------------------------------------------------------
# 2. ELBO KL Surgery: The 3 Information Terms
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.0), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(7.5, 5.5, "The Exact ELBO KL Decomposition (Chen et al. 2018)", ha='center', fontsize=15, fontweight='bold', color="#1e293b")
ax.text(7.5, 4.9, r"$\mathbb{E}_{p(n)} [D_{\mathrm{KL}}(q(z|n) \parallel p(z))] = I_q(x; z) \text{ [Term 1]} + \mathrm{TC}(z) \text{ [Term 2]} + \sum_j D_{\mathrm{KL}}(q(z_j) \parallel p(z_j)) \text{ [Term 3]}$",
        ha='center', fontsize=12.0, color="#334155")

# Term 1: Index-Code MI
t1 = patches.FancyBboxPatch((0.5, 0.6), 4.4, 3.8, boxstyle="round,pad=0.2", ec="#0284c7", fc="#f0f9ff", lw=2.5)
ax.add_patch(t1)
ax.text(2.7, 4.0, "Term 1: Index-Code MI", ha='center', fontweight='bold', fontsize=12.5, color="#0369a1")
ax.text(2.7, 3.4, r"$I_q(x; z) = \mathrm{KL}(q(z, n) \parallel q(z)p(n))$", ha='center', fontsize=10.5, color="#0c4a6e")
ax.text(2.7, 2.4, "Role: Reconstruction Fidelity\n\n• Measures info $z$ retains about $x$.\n• When penalized (e.g. in $\\beta$-VAE),\nimages turn severely blurry!\n• AdaTC Rule: Keep at 1.0!", ha='center', fontsize=9.5, color="#1e293b")
ax.text(2.7, 0.9, r"KEEP AT FULL STRENGTH ($\alpha = 1.0$)", ha='center', fontweight='bold', fontsize=9, color="#0369a1", bbox=dict(boxstyle="round,pad=0.2", fc="#e0f2fe", ec="#0284c7"))

# Term 2: Total Correlation
t2 = patches.FancyBboxPatch((5.3, 0.6), 4.4, 3.8, boxstyle="round,pad=0.2", ec="#dc2626", fc="#fef2f2", lw=2.5)
ax.add_patch(t2)
ax.text(7.5, 4.0, "Term 2: Total Correlation", ha='center', fontweight='bold', fontsize=12.5, color="#b91c1c")
ax.text(7.5, 3.4, r"$\mathrm{TC}(z) = \mathrm{KL}\left(q(z) \parallel \prod_j q(z_j)\right)$", ha='center', fontsize=10.5, color="#991b1b")
ax.text(7.5, 2.4, "Role: Factor Entanglement\n\n• Multi-variable statistical dependence\namong all latent dimensions.\n• TC = 0 means complete independence!\n• AdaTC Rule: Anneal with $\\beta(t)$!", ha='center', fontsize=9.5, color="#1e293b")
ax.text(7.5, 0.9, r"MODULATE CAREFULLY ($\beta(t)$)", ha='center', fontweight='bold', fontsize=9, color="#b91c1c", bbox=dict(boxstyle="round,pad=0.2", fc="#fee2e2", ec="#dc2626"))

# Term 3: Dimension-wise KL
t3 = patches.FancyBboxPatch((10.1, 0.6), 4.4, 3.8, boxstyle="round,pad=0.2", ec="#d97706", fc="#fffbeb", lw=2.5)
ax.add_patch(t3)
ax.text(12.3, 4.0, "Term 3: Dimension-wise KL", ha='center', fontweight='bold', fontsize=12.5, color="#b45309")
ax.text(12.3, 3.4, r"$\sum_j D_{\mathrm{KL}}(q(z_j) \parallel p(z_j))$", ha='center', fontsize=10.5, color="#92400e")
ax.text(12.3, 2.4, "Role: Coordinate Prior Anchor\n\n• Prevents single dimensions $z_j$\nfrom drifting from $\\mathcal{N}(0, 1)$.\n• Regularizes individual complexity.\n• AdaTC Rule: Keep at 1.0!", ha='center', fontsize=9.5, color="#1e293b")
ax.text(12.3, 0.9, r"KEEP PRIOR ANCHOR ($\gamma = 1.0$)", ha='center', fontweight='bold', fontsize=9, color="#b45309", bbox=dict(boxstyle="round,pad=0.2", fc="#fef3c7", ec="#d97706"))

plt.tight_layout()
plt.savefig("report/viva_diagrams/kl_surgery_breakdown.png", bbox_inches='tight')
plt.close()
print("Saved kl_surgery_breakdown.png")


# -------------------------------------------------------------
# 3. Cosine Warmup Schedule vs Static Beta
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

steps = np.linspace(0, 6000, 500)
T_warm = 5000
beta_target = 6.0

# Cosine
beta_cos = np.where(steps < T_warm,
                    1.0 + 0.5 * (beta_target - 1.0) * (1.0 - np.cos(np.pi * steps / T_warm)),
                    beta_target)

# Linear schedule
beta_lin = np.where(steps < T_warm,
                    1.0 + (beta_target - 1.0) * (steps / T_warm),
                    beta_target)

beta_static = np.full_like(steps, 6.0)
beta_vae = np.full_like(steps, 1.0)

ax.plot(steps, beta_static, color="#ef4444", lw=2.8, linestyle="--", label=r"$\beta$-TCVAE (Static $\beta=6.0$ from Step 0: Causes Blur!)")
ax.plot(steps, beta_lin, color="#9ca3af", lw=1.8, linestyle=":", label="Linear Warmup (Non-zero initial jerk)")
ax.plot(steps, beta_cos, color="#10b981", lw=3.2, label=r"AdaTC-DIP-VAE (Smooth Cosine $\beta(t)$: Sharp Reconstruction)")
ax.plot(steps, beta_vae, color="#6b7280", lw=1.8, linestyle="-.", label=r"Standard VAE ($\beta=1.0$)")

ax.annotate(r"$\mathbf{\beta'(0) = 0}$ (Zero Initial Derivative)" + "\nFilters form cleanly with NO optimization shock",
            xy=(0, 1.0), xytext=(500, 2.2),
            arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.8, headwidth=7),
            bbox=dict(boxstyle="round,pad=0.4", fc="#d1fae5", ec="#10b981", lw=1.5),
            fontsize=10, color="#065f46")

ax.annotate("Burgess Stage 1:\nLearn basic position & scale",
            xy=(1500, 2.2), xytext=(1200, 3.8),
            arrowprops=dict(facecolor='#3b82f6', shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec="#3b82f6"),
            fontsize=9.5, color="#1e40af")

ax.annotate("Full Target Regularizer:\nAsymptotic factor separation",
            xy=(5000, 6.0), xytext=(3600, 5.0),
            arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", fc="#d1fae5", ec="#10b981"),
            fontsize=9.5, color="#065f46")

ax.axvspan(0, T_warm, color='#f8fafc', alpha=0.8)

ax.set_title(r"Dynamic Regularization: Cosine Annealing $\beta(t)$ vs. Static $\beta$", fontweight="bold", fontsize=13, pad=10)
ax.set_xlabel("Training Optimization Steps $t$", fontweight="bold")
ax.set_ylabel(r"Effective Total Correlation Weight $\beta(t)$", fontweight="bold")
ax.set_ylim(0.5, 6.8)
ax.set_xlim(0, 6000)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

plt.tight_layout()
plt.savefig("report/viva_diagrams/cosine_warmup_vs_static.png", bbox_inches='tight')
plt.close()
print("Saved cosine_warmup_vs_static.png")


# -------------------------------------------------------------
# 4. The Sampling vs. Mean Disconnect & DIP Fix
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)

# Left: The Cheat
ax1.set_title("Locatello's Dilemma: Sampled z vs. Mean", color="#b91c1c", fontweight="bold", pad=12)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.axis('off')

box1 = patches.FancyBboxPatch((0.5, 0.5), 9, 7, boxstyle="round,pad=0.3", ec="#fca5a5", fc="#fef2f2", lw=2)
ax1.add_patch(box1)

ax1.text(5, 7.0, r"Reparameterization: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$", ha='center', fontsize=11.5, fontweight='bold', color="#1e293b")
ax1.text(5, 6.0, r"How $\beta$-TCVAE can artificially lower TC without true disentangling:", ha='center', fontsize=10, color="#7f1d1d")

b1 = patches.FancyBboxPatch((1.0, 3.4), 8.0, 2.0, boxstyle="round,pad=0.2", ec="#f87171", fc="#ffffff")
ax1.add_patch(b1)
ax1.text(5, 4.8, r"1. Stochastic Sample $z$ (Optimized during training)", ha='center', fontweight='bold', color="#b91c1c", fontsize=10.5)
ax1.text(5, 3.9, r"Encoder inflates variance $\sigma^2 \to \infty$ (drowning codes in noise!)." + "\n" + r"Training $\mathrm{TC}(z)$ drops, but factors remain entangled!", ha='center', fontsize=9, color="#4b5563")

b2 = patches.FancyBboxPatch((1.0, 1.0), 8.0, 1.9, boxstyle="round,pad=0.2", ec="#f87171", fc="#ffffff")
ax1.add_patch(b2)
ax1.text(5, 2.3, r"2. Deterministic Mean $\mu(x)$ (Evaluated downstream)", ha='center', fontweight='bold', color="#b91c1c", fontsize=10.5)
ax1.text(5, 1.5, "MIG, DCI, and FactorVAE evaluate deterministic means $\\mu(x)$!\nBecause $\\mu(x)$ is still correlated, downstream tests FAIL.", ha='center', fontsize=9, color="#4b5563")


# Right: The DIP Solution
ax2.set_title("AdaTC Solution: Anchored Mean Covariance", color="#15803d", fontweight="bold", pad=12)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')

box2 = patches.FancyBboxPatch((0.5, 0.5), 9, 7, boxstyle="round,pad=0.3", ec="#86efac", fc="#f0fdf4", lw=2)
ax2.add_patch(box2)

ax2.text(5, 7.0, r"$\mathcal{P}_{\mathrm{cov}}(\mu) = \lambda_{od} \sum_{i \ne j} \mathrm{Cov}_{ij}^2 + \lambda_d \sum_i (\mathrm{Cov}_{ii} - 1.0)^2$",
         ha='center', fontsize=10.5, fontweight='bold', color="#1e293b")
ax2.text(5, 6.0, "Directly enforces linear independence on the evaluated representations:", ha='center', fontsize=10, color="#14532d")

b3 = patches.FancyBboxPatch((1.0, 3.4), 8.0, 2.0, boxstyle="round,pad=0.2", ec="#4ade80", fc="#ffffff")
ax2.add_patch(b3)
ax2.text(5, 4.8, r"1. Off-Diagonal Decorrelation ($\lambda_{od} = 10.0$)", ha='center', fontweight='bold', color="#15803d", fontsize=10.5)
ax2.text(5, 3.9, "Drives cross-dimensional linear covariance between $\\mu_j$ and $\\mu_k$ to 0.\nCloses the loophole: the encoder CANNOT cheat with noise!", ha='center', fontsize=9, color="#4b5563")

b4 = patches.FancyBboxPatch((1.0, 1.0), 8.0, 1.9, boxstyle="round,pad=0.2", ec="#4ade80", fc="#ffffff")
ax2.add_patch(b4)
ax2.text(5, 2.3, r"2. Diagonal Variance Anchor ($\lambda_d = 1.0$)", ha='center', fontweight='bold', color="#15803d", fontsize=10.5)
ax2.text(5, 1.5, "Anchors coordinate variance to 1.0 (unit variance).\nPrevents the trivial collapse failure mode of shrinking $\\mu \\to 0$!", ha='center', fontsize=9, color="#4b5563")

plt.tight_layout()
plt.savefig("report/viva_diagrams/mean_vs_sampled_cheat.png", bbox_inches='tight')
plt.close()
print("Saved mean_vs_sampled_cheat.png")
