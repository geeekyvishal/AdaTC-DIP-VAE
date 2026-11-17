# Paper Summary: Disentangling Disentanglement in Variational Autoencoders (Mathieu et al., 2019)

- **Paper ID**: `Mathieu_2019_Disentangling_Disentanglement`
- **File**: `1812.02833v3.pdf`
- **Authors**: Emile Mathieu, Tom Rainforth, N. Siddharth, Yee Whye Teh
- **Venue/Year**: ICML 2019 / arXiv:1812.02833 (v3, 2019)

---

## 1. Core Claim
The paper claims that learning disentangled representations in VAEs is a special case of representation decomposition governed by two distinct components: (a) controlling posterior overlap (the amount of information/stochasticity between data and latents, captured by encoder variance and mutual information $I(x; z)$), and (b) matching the aggregate encoding $q_\phi(z)$ to a structured prior $p(z)$. Crucially, the authors prove that:
1. $\beta$-VAE is equivalent (up to a coordinate scaling of $\sqrt{\beta}$) to the standard VAE ELBO plus a maximum-entropy regularizer on encoder covariance ($\frac{\beta-1}{2}\log|S_\phi(x)|$), thus controlling only overlap.
2. Under standard isotropic Gaussian priors $p(z) = \mathcal{N}(0, \sigma^2 I)$, the $\beta$-VAE objective is invariant to arbitrary orthogonal rotations of the latent space (Theorem 2). Consequently, on its own, it cannot favor axis-aligned generative factors over any rotated entangled representation without architectural or prior symmetry breaking (Section 1, Section 4).

---

## 2. Objective Function (LaTeX)
From Section 4.1, Equation (5) proves the maximum-entropy equivalence for Gaussian encoders:

$$\mathcal{L}_{H, \beta}(x) \triangleq \mathcal{L}(x) + \frac{\beta - 1}{2} \log |S_\phi(x)|$$

From Section 4.1, Theorem 2 proves the rotational invariance of $\beta$-VAE:
$$\mathcal{L}_\beta(x; \theta, \phi) = \mathcal{L}_\beta(x; \theta^\dagger(R), \phi^\dagger(R))$$
where $\theta^\dagger(R)$ and $\phi^\dagger(R)$ are rotated networks such that $p_{\theta^\dagger}(x|z) = p_\theta(x|R^T z)$ and $q_{\phi^\dagger}(z|x) = \mathcal{N}(z; R\mu_\phi(x), R S_\phi(x) R^T)$.

From Section 5 ("An Objective for Enforcing Decomposition"), Equation (7) introduces the generalized two-factor objective:

$$\mathcal{L}_{\alpha, \beta}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta KL(q_\phi(z|x) \parallel p(z)) - \alpha \mathcal{D}(q_\phi(z), p(z))$$

Verbatim quote from Section 4.1, Page 5:
> "Theorem 2. If $p(z) = \mathcal{N}(z; 0, \sigma I)$ and $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), S_\phi(x))$, then for all rotation matrices $R$,
> $\mathcal{L}_\beta(x; \theta,\phi) = \mathcal{L}_\beta(x; \theta^\dagger(R),\phi^\dagger(R)) \quad (6)$
> where $\theta^\dagger(R)$ and $\phi^\dagger(R)$ are transformed networks such that
> $p_{\theta^\dagger}(x|z) = p_\theta(x|R^T z), \quad q_{\phi^\dagger}(z|x) = \mathcal{N}(z; R\mu_\phi(x), R S_\phi(x) R^T)$.
> This shows that the $\beta$-VAE objective does not directly encourage latent variables to take on meaningful representations when using the standard choice of an isotropic Gaussian prior. In fact, on its own, it encourages latent representations which match the true generative factors no more than it encourages any arbitrary rotation of these factors, with such rotations capable of exhibiting strong correlations between latents."

Verbatim quote from Section 5, Page 5:
> "$\mathcal{L}_{\alpha,\beta}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta KL(q_\phi(z|x) \parallel p(z)) - \alpha \mathcal{D}(q_\phi(z), p(z)) \quad (7)$"

---

## 3. Which KL Term It Modifies
- **Modifications**: Separates the conditional KL term $KL(q_\phi(z|x) \parallel p(z))$ (scaled by $\beta$, which dictates encoder entropy and overlap) from the marginal divergence $\mathcal{D}(q_\phi(z), p(z))$ (scaled by $\alpha$, which enforces marginal alignment with the prior structure).
- **Term Breakdown**:
  Demonstrates that simply turning up $\beta$ in $\beta$-VAE degrades reconstruction because it over-anneals posterior overlap (collapsing the latent mutual information), rather than selectively enforcing coordinate independence.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: Evaluates and critiques existing estimators.
- **Analysis**:
  In Section 5 and Appendix C, the paper analyzes:
  - Moment-matching covariance regularizers (Kumar et al., DIP-VAE).
  - Adversarial density-ratio estimation (Kim & Mnih, FactorVAE).
  - Minibatch estimators (Chen et al., $\beta$-TCVAE).
  - Shows that direct minibatch estimators of $\mathcal{D}(q(z) \parallel p(z))$ or $TC(z)$ introduce substantial estimation bias when minibatch sizes are small, leading to severe deviation from the true divergence.

---

## 5. Datasets, Metrics, and Results
- **Datasets**:
  - 2D Shapes (dSprites): 737,280 images (Section 6.1).
  - Synthetic 2D Spirals dataset (Section 6.2).
- **Metrics**:
  - Reconstruction loss.
  - Kim & Mnih FactorVAE disentanglement metric (Section 6.1).
  - Hoyer extrinsic sparsity metric: $\text{Hoyer}(y) = \frac{\sqrt{d} - \|y\|_1 / \|y\|_2}{\sqrt{d} - 1} \in [0, 1]$ (Section 6.2).
- **Results**:
  - Showed that varying $\beta$ with a standard isotropic Gaussian prior on 2D Shapes yields huge variability in FactorVAE metric scores with no reliable improvement (Section 6.1, Figure 2 left).
  - Replacing the isotropic prior with a rotation-breaking prior—such as an anisotropic diagonal prior with variances matched to dataset PCA eigenvalues, or a factorized Student-t prior $p_\nu(z) = \prod_d \text{Student-T}(z_d; \nu)$—breaks rotational symmetry and achieves high FactorVAE scores ($\sim 0.8$) at $\beta = 1$ without sacrificing reconstruction accuracy (Figure 2 right).

---

## 6. Limitations
1. **Divergence Estimation Dilemma**: Estimating $\mathcal{D}(q(z), p(z))$ stochastically during training is either computationally unstable (GAN/discriminator, MMD U-statistics) or significantly biased with small batches (MWS).
2. **Prior Design Dependency**: Achieving axis alignment requires either non-Gaussian factorized priors (e.g. Student-t) or pre-computing empirical principal components to design an anisotropic prior, requiring knowledge about the data spectrum.
