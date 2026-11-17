import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set MPLCONFIGDIR
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')

# 1. INPUT IMAGE
rect_in = patches.FancyBboxPatch((0.5, 3.7), 1.2, 1.6, boxstyle='round,pad=0.1', facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2)
ax.add_patch(rect_in)
ax.text(1.1, 4.5, "Input Image\n$x$\n(64x64x1)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0d47a1')

ax.annotate('', xy=(2.0, 4.5), xytext=(1.7, 4.5), arrowprops=dict(arrowstyle='->', lw=2, color='#37474f'))

# 2. ENCODER BLOCKS
enc_blocks = [
    ("Conv 1\n4x4, s=2\n(32, 32, 32)", 2.0, 3.4, 1.1, 2.2, '#bbdefb'),
    ("Conv 2\n4x4, s=2\n(16, 16, 32)", 3.3, 3.6, 1.1, 1.8, '#90caf9'),
    ("Conv 3\n4x4, s=2\n(8, 8, 64)", 4.6, 3.8, 1.1, 1.4, '#64b5f6'),
    ("Conv 4\n4x4, s=2\n(4, 4, 64)", 5.9, 4.0, 1.1, 1.0, '#42a5f5'),
]
for text, x, y, w, h, col in enc_blocks:
    p = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=col, edgecolor='#1565c0', linewidth=1.5)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, fontweight='bold', color='#0d47a1')

for x in [3.1, 4.4, 5.7]:
    ax.annotate('', xy=(x+0.2, 4.5), xytext=(x, 4.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))

# FC 256
ax.annotate('', xy=(7.2, 4.5), xytext=(7.0, 4.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))
p_fc = patches.FancyBboxPatch((7.2, 3.5), 0.9, 2.0, boxstyle='round,pad=0.08', facecolor='#bbdefb', edgecolor='#1565c0', linewidth=1.5)
ax.add_patch(p_fc)
ax.text(7.65, 4.5, "FC 256\n(ReLU)", ha='center', va='center', fontsize=9, fontweight='bold', color='#0d47a1')

# Mu and Logvar
ax.annotate('', xy=(8.4, 5.4), xytext=(8.1, 4.7), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))
ax.annotate('', xy=(8.4, 3.6), xytext=(8.1, 4.3), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))

p_mu = patches.FancyBboxPatch((8.4, 5.0), 1.2, 0.8, boxstyle='round,pad=0.08', facecolor='#ffecb3', edgecolor='#ff8f00', linewidth=2)
ax.add_patch(p_mu)
ax.text(9.0, 5.4, r"Mean $\mu_\phi(x)$" + "\n($d=10$)", ha='center', va='center', fontsize=9, fontweight='bold', color='#e65100')

p_lv = patches.FancyBboxPatch((8.4, 3.2), 1.2, 0.8, boxstyle='round,pad=0.08', facecolor='#ffe082', edgecolor='#ff8f00', linewidth=1.5)
ax.add_patch(p_lv)
ax.text(9.0, 3.6, r"LogVar $\log\sigma^2$" + "\n($d=10$)", ha='center', va='center', fontsize=9, fontweight='bold', color='#e65100')

# Reparameterization node
ax.annotate('', xy=(10.2, 4.5), xytext=(9.6, 5.2), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))
ax.annotate('', xy=(10.2, 4.5), xytext=(9.6, 3.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))

p_reparam = patches.Circle((10.45, 4.5), 0.35, facecolor='#fff9c4', edgecolor='#f57f17', linewidth=2)
ax.add_patch(p_reparam)
ax.text(10.45, 4.5, r"$z$", ha='center', va='center', fontsize=12, fontweight='bold', color='#b71c1c')
ax.text(10.45, 5.1, r"$z = \mu + \sigma \odot \epsilon$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#b71c1c')

# Noise eps
p_eps = patches.FancyBboxPatch((9.9, 3.1), 1.1, 0.5, boxstyle='round,pad=0.05', facecolor='#ffffff', edgecolor='#9e9e9e', linestyle='--')
ax.add_patch(p_eps)
ax.text(10.45, 3.35, r"$\epsilon \sim \mathcal{N}(0, I)$", ha='center', va='center', fontsize=8, color='#616161')
ax.annotate('', xy=(10.45, 4.15), xytext=(10.45, 3.6), arrowprops=dict(arrowstyle='->', lw=1.2, color='#616161', linestyle='--'))

# DECODER BLOCKS
ax.annotate('', xy=(11.1, 4.5), xytext=(10.8, 4.5), arrowprops=dict(arrowstyle='->', lw=2, color='#37474f'))

dec_blocks = [
    ("Dense 256\n+ 1024", 11.1, 3.5, 0.9, 2.0, '#c8e6c9'),
    ("Deconv 1\n4x4, s=2\n(8, 8, 64)", 12.2, 4.0, 0.9, 1.0, '#a5d6a7'),
    ("Deconv 2\n4x4, s=2\n(16, 16, 32)", 13.3, 3.8, 0.9, 1.4, '#81c784'),
    ("Deconv 3\n4x4, s=2\n(32, 32, 32)", 14.4, 3.6, 0.9, 1.8, '#66bb6a'),
]
for text, x, y, w, h, col in dec_blocks:
    p = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=col, edgecolor='#2e7d32', linewidth=1.5)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8, fontweight='bold', color='#1b5e20')

for x in [12.0, 13.1, 14.2]:
    ax.annotate('', xy=(x+0.2, 4.5), xytext=(x, 4.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='#37474f'))

# Output Image
ax.annotate('', xy=(15.5, 4.5), xytext=(15.3, 4.5), arrowprops=dict(arrowstyle='->', lw=2, color='#37474f'))
rect_out = patches.FancyBboxPatch((15.4, 3.7), 1.2, 1.6, boxstyle='round,pad=0.1', facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
ax.add_patch(rect_out)
ax.text(16.0, 4.5, "Reconstruction\n" + r"$\hat{x}$" + "\n(64x64x1)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1b5e20')

# ================= LOSS STREAMS =================

# 1. Reconstruction Loss (Top curve)
ax.annotate(r"Reconstruction Fidelity: $\mathcal{L}_{\mathrm{recon}}(x, \hat{x})$ (BCE / MSE) [Weight = 1.0]", 
            xy=(16.0, 5.5), xytext=(8.5, 7.9),
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#b71c1c',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebee', edgecolor='#c62828', lw=1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='#c62828', connectionstyle='arc3,rad=-0.2'))

ax.annotate('', xy=(1.1, 5.5), xytext=(5.5, 7.9),
            arrowprops=dict(arrowstyle='<-', lw=2, color='#c62828', connectionstyle='arc3,rad=0.2'))

# 2. Total Correlation Loss with Cosine Schedule (Bottom right)
rect_tc = patches.FancyBboxPatch((8.1, 1.1), 4.6, 1.4, boxstyle='round,pad=0.15', facecolor='#ede7f6', edgecolor='#512da8', linewidth=2)
ax.add_patch(rect_tc)
ax.text(10.4, 2.15, r"Stream 1: Total Correlation on Sampled $z$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#311b92')
ax.text(10.4, 1.55, r"$\beta(t) \cdot \mathrm{KL}(q(z) \,\|\, \prod_j q(z_j))$" + "\n" + r"Cosine Warmup: $\beta(0)=1 \to \beta_{\mathrm{target}}=6.0$", ha='center', va='center', fontsize=8.5, color='#4527a0')
ax.annotate('', xy=(10.4, 2.5), xytext=(10.45, 4.15), arrowprops=dict(arrowstyle='<-', lw=2, color='#512da8', connectionstyle='arc3,rad=0.15'))

# 3. Deterministic Mean Covariance Penalty (Bottom left)
rect_cov = patches.FancyBboxPatch((2.2, 1.1), 5.3, 1.4, boxstyle='round,pad=0.15', facecolor='#fff3e0', edgecolor='#e65100', linewidth=2)
ax.add_patch(rect_cov)
ax.text(4.85, 2.15, r"Stream 2: Anchored Covariance on Means $\mu_\phi(x)$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#bf360c')
ax.text(4.85, 1.55, r"$\lambda_{od} \sum_{i \ne j} [\mathrm{Cov}(\mu)]_{ij}^2 + \lambda_d \sum_i ([\mathrm{Cov}(\mu)]_{ii} - 1.0)^2$" + "\nDirectly Decorrelates Downstream Latent Coordinates", ha='center', va='center', fontsize=8.5, color='#d84315')
ax.annotate('', xy=(4.85, 2.5), xytext=(8.6, 5.0), arrowprops=dict(arrowstyle='<-', lw=2, color='#e65100', connectionstyle='arc3,rad=0.2'))

# Title
ax.text(8.5, 8.7, "AdaTC-DIP-VAE: Complete Network Architecture & Dual-Stream Objective Flow", ha='center', va='center', fontsize=13.5, fontweight='bold', color='#263238')

plt.tight_layout()
out_png = '/Users/vishalsingh/Documents/Academics/semester 7/GenAI/research/report/architecture_diagram.png'
out_pdf = '/Users/vishalsingh/Documents/Academics/semester 7/GenAI/research/report/architecture_diagram.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print("Success: Generated", out_png, "and", out_pdf)
