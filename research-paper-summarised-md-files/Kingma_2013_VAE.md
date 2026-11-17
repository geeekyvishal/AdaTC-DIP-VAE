# Paper Summary: Auto-Encoding Variational Bayes (Kingma & Welling, 2013/2014)

- **Paper ID**: `Kingma_2013_VAE`
- **File**: `1312.6114v11.pdf`
- **Authors**: Diederik P. Kingma, Max Welling
- **Venue/Year**: ICLR 2014 / arXiv:1312.6114 (v11, 2014)

---

## 1. Core Claim
The paper introduces the Auto-Encoding Variational Bayes (AEVB) algorithm and the Stochastic Gradient Variational Bayes (SGVB) estimator. The core claim is that continuous latent variable models with intractable posterior distributions and large datasets can be efficiently trained end-to-end using standard stochastic gradient ascent by reparameterizing the approximate posterior $q_\phi(z|x)$, yielding a low-variance differentiable estimator of the variational lower bound (ELBO) on the marginal likelihood (Section 1, Section 2.1–2.3).

---

## 2. Objective Function (LaTeX)
From Section 2.2 ("The variational bound"), Equation (3):

$$\mathcal{L}(\theta, \phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z)) + \mathbb{E}_{q_\phi(z|x^{(i)})}[\log p_\theta(x^{(i)}|z)]$$

Verbatim quote from Section 2.2, Page 3:
> "Since this KL-divergence is non-negative, the second RHS term $\mathcal{L}(\theta,\phi; x^{(i)})$ is called the (variational) lower bound on the marginal likelihood of datapoint $i$, and can be written as:
> $\log p_\theta(x^{(i)}) \ge \mathcal{L}(\theta,\phi; x^{(i)}) = \mathbb{E}_{q_\phi(z|x)} [-\log q_\phi(z|x) + \log p_\theta(x,z)] \quad (2)$
> which can also be written as:
> $\mathcal{L}(\theta,\phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z)) + \mathbb{E}_{q_\phi(z|x^{(i)})}[\log p_\theta(x^{(i)}|z)] \quad (3)$"

Minibatch SGVB estimator with analytical KL and reparameterization $z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})$ with $\epsilon \sim p(\epsilon)$ from Section 2.3, Equation (7):

$$\tilde{\mathcal{L}}^B(\theta,\phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z)) + \frac{1}{L} \sum_{l=1}^L \log p_\theta(x^{(i)}|z^{(i,l)})$$

Verbatim quote from Section 2.3, Page 4:
> "The first term is (the KL divergence of the approximate posterior from the prior) acts as a regularizer, while the second term is an expected negative reconstruction error. The function $g_\phi(.)$ is chosen such that it maps a datapoint $x^{(i)}$ and a random noise vector $\epsilon^{(l)}$ to a sample from the approximate posterior for that datapoint: $z^{(i,l)} = g_\phi(\epsilon^{(l)}, x^{(i)})$ where $z^{(i,l)} \sim q_\phi(z|x^{(i)})$."

---

## 3. Which KL Term It Modifies
- **Modifications**: None (standard baseline).
- **Analysis**: The objective applies a uniform weight ($\beta = 1$) to the aggregate KL divergence $D_{KL}(q_\phi(z|x^{(i)}) \parallel p(z))$. It does not isolate or decompose the KL term into Index-Code Mutual Information, Total Correlation, or Dimension-wise KL.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: None.
- **Details**: Standard VAE optimizes the unweighted ELBO directly using an analytical Gaussian-Gaussian KL divergence per datapoint (Appendix B), without estimating or penalizing Total Correlation among latent dimensions.

---

## 5. Datasets, Metrics, and Results
- **Datasets**:
  - MNIST digit dataset ($28 \times 28$ binarized images) (Section 5).
  - Frey Face dataset ($20 \times 28$ continuous face images) (Section 5).
- **Metrics**:
  - Variational lower bound ($\mathcal{L}$) per datapoint across training steps (Section 5, Figure 2).
  - Marginal log-likelihood estimated via Monte Carlo expectation-maximization / importance sampling (Section 5, Figure 3).
- **Results**:
  - AEVB converged faster and attained higher variational lower bounds and marginal log-likelihoods compared to the Wake-Sleep algorithm and Monte Carlo EM across latent dimensionalities $N_z \in \{2, 3, 5, 10, 20, 200\}$ (Section 5, Figures 2 & 3).
  - Visualizations in 2D latent space showed structured, smooth manifold transitions (Section 5, Appendix A).

---

## 6. Limitations
1. **Lack of Disentanglement Enforcement**: The standard VAE objective optimizes data log-likelihood lower bounds without penalizing statistical dependencies between latent variables in the aggregated posterior $q(z) = \frac{1}{N}\sum_{i=1}^N q(z|x^{(i)})$. Consequently, generative factors are frequently entangled across latent dimensions.
2. **Prior-Posterior Tension**: Because $D_{KL}(q(z|x) \parallel p(z))$ is enforced per data point rather than across the marginal distribution, the model can over-regularize individual codes towards the prior $\mathcal{N}(0, I)$ or experience posterior collapse without ensuring coordinate alignment with ground-truth factors.
