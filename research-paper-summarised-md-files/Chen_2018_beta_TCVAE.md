# Paper Summary: Isolating Sources of Disentanglement in VAEs (Chen et al., 2018)

- **Paper ID**: `Chen_2018_beta_TCVAE`
- **File**: `1802.04942v5.pdf`
- **Authors**: Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud
- **Venue/Year**: NeurIPS 2018 / arXiv:1802.04942 (v5, 2018)

---

## 1. Core Claim
The paper claims that the average KL divergence term in the VAE ELBO can be decomposed into three distinct, interpretable components: (i) Index-Code Mutual Information, (ii) Total Correlation, and (iii) Dimension-wise KL. The authors argue that Total Correlation (TC) is the primary driver of disentanglement, and that penalizing index-code mutual information needlessly degrades reconstruction fidelity. By penalizing Total Correlation alone while keeping the other weights at 1, $\beta$-TCVAE achieves significantly better disentangled representations without hurting reconstruction. Furthermore, the paper claims TC can be estimated stochastically without adversarial discriminators via Minibatch Weighted Sampling (MWS), and introduces the Mutual Information Gap (MIG) metric (Section 1, Section 3, Section 4).

---

## 2. Objective Function (LaTeX)
From Section 3 ("Sources of Disentanglement in the ELBO"), Equation (2) presents the complete KL divergence decomposition:

$$\mathbb{E}_{p(n)} \left[ KL(q(z|n) \parallel p(z)) \right] = \underbrace{KL(q(z, n) \parallel q(z)p(n))}_{\text{i. Index-Code MI}} + \underbrace{KL\left(q(z) \parallel \prod_j q(z_j)\right)}_{\text{ii. Total Correlation}} + \underbrace{\sum_j KL(q(z_j) \parallel p(z_j))}_{\text{iii. Dimension-wise KL}}$$

From Section 3.1.1 ("Special case: β-TCVAE"), Equation (4):

$$\mathcal{L}_{\beta\text{-TC}} := \mathbb{E}_{q(z|n)p(n)}[\log p(n|z)] - \alpha I_q(z; n) - \beta KL\left(q(z) \parallel \prod_j q(z_j)\right) - \gamma \sum_j KL(q(z_j) \parallel p(z_j))$$

where $\beta$-TCVAE sets $\alpha = 1, \gamma = 1$, and varies $\beta > 1$:

$$\mathcal{L}_{\beta\text{-TC}} = \mathbb{E}_{q(z|n)p(n)}[\log p(n|z)] - I_q(z; n) - \beta KL\left(q(z) \parallel \prod_j q(z_j)\right) - \sum_j KL(q(z_j) \parallel p(z_j))$$

Verbatim quote from Section 3.1.1, Page 4:
> "With minibatch-weighted sampling, it is easy to assign different weights $(\alpha,\beta,\gamma)$ to the terms in (2):
> $\mathcal{L}_{\beta-TC} := \mathbb{E}_{q(z|n)p(n)}[\log p(n|z)] - \alpha I_q(z; n) - \beta KL\left(q(z) \parallel \prod_j q(z_j)\right) - \gamma \sum_j KL(q(z_j) \parallel p(z_j)) \quad (4)$
> While we performed ablation experiments with different values for $\alpha$ and $\gamma$, we ultimately find that tuning $\beta$ leads to the best results. Our proposed $\beta$-TCVAE uses $\alpha = \gamma = 1$ and only modifies the hyperparameter $\beta$. While Kim & Mnih [8] have proposed an equivalent objective, they estimate TC using an auxiliary discriminator network."

---

## 3. Which KL Term It Modifies
- **Modifications**: Directly and selectively penalizes **Total Correlation (TC)**:
  $$TC(z) = KL\left(q(z) \parallel \prod_j q(z_j)\right)$$
  by scaling it with hyperparameter $\beta > 1$.
- **Term Breakdown**:
  - **Index-Code MI**: $\alpha = 1$. Preserves mutual information between data index $n$ and latent $z$, maintaining high reconstruction quality.
  - **Total Correlation (TC)**: $\beta > 1$. Explicitly forces the aggregated posterior to factorize into independent coordinates.
  - **Dimension-wise KL**: $\gamma = 1$. Acts as a standard capacity/complexity regularizer preventing individual latent dimensions from drifting arbitrarily far from prior $p(z_j)$.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: Minibatch Weighted Sampling (MWS).
- **Formulation**:
  From Section 3.1, Equation (3):
  $$\mathbb{E}_{q(z)}[\log q(z)] \approx \frac{1}{M} \sum_{i=1}^M \left[ \log \frac{1}{NM} \sum_{j=1}^M q(z(n_i)|n_j) \right]$$
  where $\{n_1, \dots, n_M\}$ is a minibatch sampled from dataset of size $N$, and $z(n_i) \sim q(z|n_i)$.
  Similarly, for each marginal dimension $j$:
  $$\mathbb{E}_{q(z_j)}[\log q(z_j)] \approx \frac{1}{M} \sum_{i=1}^M \left[ \log \frac{1}{NM} \sum_{k=1}^M q(z_j(n_i)|n_k) \right]$$
  Total Correlation is evaluated without an auxiliary discriminator:
  $$\widehat{TC}(z) = \frac{1}{M} \sum_{i=1}^M \left[ \log \frac{q(z(n_i))}{\prod_j q(z_j(n_i))} \right]$$

Verbatim quote from Section 3.1, Page 4:
> "To this end, we propose using a weighted version for estimating the function $\log q(z)$ during training, inspired by importance sampling. When provided with a minibatch of samples $\{n_1,\dots, n_M\}$, we can use the estimator
> $\mathbb{E}_{q(z)}[\log q(z)] \approx \frac{1}{M} \sum_{i=1}^M \left[ \log \frac{1}{NM} \sum_{j=1}^M q(z(n_i)|n_j) \right] \quad (3)$
> where $z(n_i)$ is a sample from $q(z|n_i)$ (see derivation in Appendix C). This minibatch estimator is biased, since its expectation is a lower bound. However, computing it does not require any additional hyperparameters."

---

## 5. Datasets, Metrics, and Results
- **Datasets**:
  - dSprites: 737,280 binary $64 \times 64$ images with 5 ground-truth factors (shape, scale, rotation, pos X, pos Y) (Section 6).
  - 3D Faces: Basel Face model renderings (Section 6).
  - CelebA: Qualitative latent traversals (Section 1, Figure 1).
- **Metrics**:
  - **Mutual Information Gap (MIG)** (Section 4.1, Equation 6):
    $$\text{MIG} = \frac{1}{K} \sum_{k=1}^K \frac{1}{\mathcal{H}(v_k)} \left( I_n(z_{j^{(k)}}; v_k) - \max_{j \ne j^{(k)}} I_n(z_j; v_k) \right)$$
    where $j^{(k)} = \arg\max_j I_n(z_j; v_k)$, enforcing that each generative factor $v_k$ is captured predominantly by exactly one latent coordinate with a large gap over the second-highest informative coordinate.
  - ELBO and Total Correlation.
- **Results**:
  - $\beta$-TCVAE created more disentangled representations while preserving a much higher ELBO than $\beta$-VAE across all $\beta \in [2, 10]$ on dSprites and 3D Faces (Section 6, Figure 2).
  - Strongly negative empirical correlation between Total Correlation and MIG ($\rho = -0.89$ on dSprites, $\rho = -0.99$ on 3D Faces) (Section 6, Figure 4).
  - Retained robust disentanglement even under factor sampling bias (Figure 5).

---

## 6. Limitations
1. **Biased Estimator**: The MWS estimator is biased (its expectation is a mathematical lower bound by Jensen's inequality: $\mathbb{E}_{p(n)}[\log q(z|n)] \le \log \mathbb{E}_{p(n)}[q(z|n)]$, Section 3.1, Footnote 3).
2. **Minibatch Size Dependency**: The quality of the MWS density estimation depends on batch size $M$ relative to dataset size $N$; smaller batches can underestimate density or exhibit high variance.
3. **Inherent Prior-Posterior Rotation Ambiguity**: While TC penalizes inter-dimensional dependence in $q(z)$, the standard isotropic Gaussian prior $p(z) = \mathcal{N}(0, I)$ is rotationally symmetric. As shown in later work, TC alone does not mathematically prevent orthogonal rotations of the latent space without decoder axis-aligning inductive biases.
