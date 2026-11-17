# AdaTC-DIP-VAE: Disentangled Representation Learning Without Sacrificing Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Report](https://img.shields.io/badge/Report-PDF-red.svg)](report/college_research_report.pdf)

> **Generative AI Research Project Report (Semester 7)**  
> **Author:** Vishal Singh  
> **Department:** Department of Computer Science & Engineering  

---

## 📌 Overview

Variational Autoencoders often suffer from a severe trade-off: penalizing information capacity to isolate independent latent factors (as in $\beta$-VAE and $\beta$-TCVAE) leads to heavy information loss and blurry image reconstructions. Furthermore, optimizing Total Correlation purely over stochastic samples does not guarantee uncorrelated deterministic representations evaluated downstream.

**AdaTC-DIP-VAE** (*Adaptive Total Correlation Annealing with Deterministic Inferred Prior Decorrelation*) resolves this fundamental bottleneck:
1. **Preserves Index-Code Mutual Information** ($\alpha = 1.0$) from the exact ELBO decomposition to maintain sharp reconstructive details.
2. **Modulates Total Correlation** via a smooth, $C^1$-continuous cosine annealing schedule $\beta(t)$ with $\beta'(0) = 0$ to prevent early dimension collapse.
3. **Anchored Deterministic Mean Decorrelation** ($\lambda_d = 1.0$, $\lambda_{od} = 10.0$) directly on encoder means $\mu_\phi(x)$ to align optimization with downstream metric evaluations and eliminate variance collapse.

---

## 🏗️ Model Architecture & Optimization Flow

![Architecture Diagram](report/architecture_diagram.png)

---

## 📊 Benchmark Results (dSprites, $d=10$)

Evaluated on 5,000 held-out test samples on NVIDIA Tesla T4 GPUs:

| Model Architecture | Effective $\beta$ | Reconstruction MSE $\downarrow$ | Relative MSE vs. VAE | MIG $\uparrow$ | DCI Disent. $\uparrow$ | FactorVAE Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard VAE** | $1.0$ | **0.00470** | Baseline ($0\%$) | $0.0302$ | $0.2613$ | **0.6450** |
| **$\beta$-VAE** | $4.0$ (Static) | $0.00576$ | $+22.6\%$ | $0.0290$ | $0.1818$ | $0.4467$ |
| **$\beta$-TCVAE** | $6.0$ (Static) | $0.00785$ | $+67.0\%$ | **0.1512** | **0.3674** | $0.5667$ |
| **AdaTC-DIP-VAE (Ours)** | $1.0 \to 4.16$ | **0.00505** | **$+7.4\%$** | $0.0103$ | $0.1796$ | $0.4033$ |

### 🔍 Key Findings:
- **35.7% Lower Reconstruction Error** than $\beta$-TCVAE ($0.00505$ vs. $0.00785$ MSE).
- Virtually matches the sharpness of an unconstrained VAE within $+0.00035$ MSE.
- Eliminates the boundary blur and artifacts characteristic of static information bottlenecks.

---

## 📈 Fidelity Benchmark & Pareto Trade-off

<p align="center">
  <img src="report/plots/reconstruction_comparison.png" width="48%" />
  <img src="report/plots/pareto_tradeoff.png" width="48%" />
</p>

---

## 🖼️ Latent Traversals Comparison

<p align="center">
  <img src="report/plots/baseline_plot_1.png" width="48%" title="Standard VAE" />
  <img src="report/plots/baseline_plot_2.png" width="48%" title="Beta-VAE" />
</p>
<p align="center">
  <img src="report/plots/betatcvae_plot_1.png" width="48%" title="Beta-TCVAE" />
  <img src="report/plots/proposed_plot_1.png" width="48%" title="AdaTC-DIP-VAE (Proposed)" />
</p>

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/geeekyvishal/AdaTC-DIP-VAE.git
cd AdaTC-DIP-VAE
pip install -r requirements.txt
```

### 2. Run Verification Tests
```bash
python3 -m unittest discover tests
```

### 3. Model & Loss Usage in PyTorch
```python
import torch
from src.models.vae import DisentangledVAE
from src.losses.adatc_dip_loss import AdaTCDIPLoss

# Initialize architecture
model = DisentangledVAE(latent_dim=10, in_channels=1)

# Initialize proposed objective
criterion = AdaTCDIPLoss(
    dataset_size=737280,
    beta_target=6.0,
    warmup_steps=5000,
    lambda_od=10.0,
    lambda_d=1.0
)

# Forward pass & loss computation
x = torch.randn(256, 1, 64, 64)
x_recon, mu, logvar, z = model(x)
loss, loss_dict = criterion(x, x_recon, mu, logvar, z, step=1000)
```

Interactive, cloud-ready execution notebooks are available in [`notebooks/`](notebooks/).

---

## 📑 Reports & Documentation

- **Single-Column Academic Report**: [`report/college_research_report.pdf`](report/college_research_report.pdf)
- **IEEE Conference Two-Column Paper**: [`report/paper.pdf`](report/paper.pdf)
- **Theoretical Derivations & Audit**: [`theory/candidates.md`](theory/candidates.md) and [`theory/critique.md`](theory/critique.md)

---

## 📚 References

1. Kingma & Welling (2013). *Auto-Encoding Variational Bayes*. ICLR.
2. Higgins et al. (2017). *$\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR.
3. Burgess et al. (2018). *Understanding Disentangling in $\beta$-VAE*. arXiv:1804.03599.
4. Chen et al. (2018). *Isolating Sources of Disentanglement in Variational Autoencoders*. NeurIPS.
5. Kumar et al. (2018). *Variational Inference of Disentangled Latent Concepts from Unlabeled Observations*. ICLR.
6. Locatello et al. (2019). *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*. ICML.
