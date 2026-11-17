# Paper Summary: Auxiliary Deep Generative Models (Maaløe et al., 2016)

- **Paper ID**: `Maaloe_2016_ADGM`
- **File**: `1602.05473v4.pdf`
- **Authors**: Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, Ole Winther
- **Venue/Year**: ICML 2016 / arXiv:1602.05473 (v4, 2016)
- **Discrepancy Note**: In `research-paper/x.bash`, this file was referenced under the target name `Hoffman_Johnson_2016_ELBO_Surgery.pdf`. However, the downloaded arXiv paper `1602.05473` is actually *Auxiliary Deep Generative Models* by Maaløe et al. This summary analyzes the actual file present in the directory per Rule 1.

---

## 1. Core Claim
The paper claims that standard VAEs are restricted by the assumption of a factorized diagonal Gaussian approximate posterior $q(z|x)$, leading to poor variational bounds on multimodal distributions. By extending deep generative models with continuous auxiliary variables $a$ such that the generative distribution is unchanged under marginalization ($p(x, z, a) = p(a|x, z)p(x, z)$), the marginal approximate posterior $q(z|x) = \int q(z|a, x) q(a|x) da$ becomes a highly flexible, non-Gaussian distribution that dramatically improves variational inference and semi-supervised classification (Section 1, Section 2).

---

## 2. Objective Function (LaTeX)
From Section 2.2 ("Auxiliary variables"), Equation (2):

$$\log p(x) \ge \mathbb{E}_{q_\phi(a, z|x)} \left[ \log \frac{p_\theta(a|z, x) p_\theta(x|z) p(z)}{q_\phi(a|x) q_\phi(z|a, x)} \right] \equiv -\mathcal{U}_{AVAE}(x)$$

For semi-supervised classification with labeled observations $(x_l, y_l)$ and unlabeled observations $x_u$, the objective combines $-\mathcal{U}(x_u)$, $-\mathcal{U}(x_l, y_l)$, and an explicit classification loss $\alpha \mathbb{E}[\log q_\phi(y|a, x)]$ (Section 2.3–2.4).

Verbatim quote from Section 2.2, Page 2:
> "The auxiliary VAE lower bound becomes
> $\log p(x) = \log \int_a \int_z p(x,a,z) dadz \ge \mathbb{E}_{q_\phi(a,z|x)} \left[ \log \frac{p_\theta(a|z,x)p_\theta(x|z)p(z)}{q_\phi(a|x)q_\phi(z|a,x)} \right] \equiv -\mathcal{U}_{AVAE}(x) \quad (2)$
> with $p_\theta(a|z,x)$ and $q_\phi(a|x)$ diagonal Gaussian distributions parameterized by deep neural networks."

---

## 3. Which KL Term It Modifies
- **Modifications**: Does not modify or decompose the KL divergence into index-code MI, Total Correlation, or dimension-wise KL.
- **Analysis**: Instead of penalizing or isolating latent factor independence, it enriches the approximate posterior $q(z|x)$ by conditioning $z$ on stochastic auxiliary latent variable $a$.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: None.

---

## 5. Datasets, Metrics, and Results
- **Datasets**:
  - MNIST ($28 \times 28$ handwritten digits) (Section 4).
  - SVHN (Street View House Numbers) (Section 4).
  - NORB (multiview object images) (Section 4).
- **Metrics**:
  - Semi-supervised classification error rate (%) with limited labels (100, 1000 labels).
  - Test marginal log-likelihood lower bounds.
- **Results**:
  - Achieved state-of-the-art semi-supervised classification error rates: $0.96\%$ on MNIST with 100 labels, $16.61\%$ on SVHN with 1000 labels, and $10.03\%$ on NORB with 1000 labels (Table 1, Table 2).

---

## 6. Limitations
1. **Unrelated to Unsupervised Factor Disentanglement**: The model is designed for posterior flexibility in density estimation and semi-supervised classification, not for learning coordinate-aligned independent factors of variation.
2. **Additional Latent Overhead**: Introducing auxiliary variable $a$ adds additional encoder/decoder networks and sampling steps.
