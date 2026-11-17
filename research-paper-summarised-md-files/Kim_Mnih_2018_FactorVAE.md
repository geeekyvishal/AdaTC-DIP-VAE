# Paper Summary: Disentangling by Factorising (Kim & Mnih, 2018)

- **Paper ID**: `Kim_Mnih_2018_FactorVAE`
- **File**: `1802.05983v3.pdf`
- **Authors**: Hyunjik Kim, Andriy Mnih
- **Venue/Year**: ICML 2018 / arXiv:1802.05983 (v3, 2018)

---

## 1. Core Claim
The paper claims that unsupervised disentanglement requires encouraging the aggregated posterior (marginal code distribution) $q(z) = \mathbb{E}_{p_{data}(x)}[q(z|x)]$ to be factorial. The authors show that the penalty in $\beta$-VAE penalizes both Total Correlation (TC) and the mutual information between data and latents $I(x; z)$, which needlessly impairs reconstruction fidelity. By directly augmenting the standard VAE ELBO with an explicit Total Correlation penalty $-\gamma KL(q(z) \parallel \bar{q}(z))$, FactorVAE achieves superior disentanglement at identical reconstruction quality (Section 1, Section 3).

---

## 2. Objective Function (LaTeX)
From Section 3 ("FactorVAE"), Equation (2):

$$\frac{1}{N} \sum_{i=1}^N \left[ \mathbb{E}_{q(z|x^{(i)})}[\log p(x^{(i)}|z)] - KL(q(z|x^{(i)}) \parallel p(z)) \right] - \gamma KL(q(z) \parallel \bar{q}(z))$$

where $\bar{q}(z) := \prod_{j=1}^d q(z_j)$.

Verbatim quote from Section 3, Page 3:
> "motivate FactorVAE by augmenting the VAE objective with a term that directly encourages independence in the code distribution, arriving at the following objective:
> $\frac{1}{N} \sum_{i=1}^N \left[ \mathbb{E}_{q(z|x^{(i)})}[\log p(x^{(i)}|z)] - KL(q(z|x^{(i)}) \parallel p(z)) \right] - \gamma KL(q(z) \parallel \bar{q}(z)), \quad (2)$
> where $\bar{q}(z) := \prod_{j=1}^d q(z_j)$. Note that this is also a lower bound on the marginal log likelihood $\mathbb{E}_{p_{data}(x)}[\log p(x)]$. $KL(q(z) \parallel \bar{q}(z))$ is known as Total Correlation (TC, Watanabe, 1960), a popular measure of dependence for multiple random variables."

---

## 3. Which KL Term It Modifies
- **Modifications**: Directly penalizes Total Correlation (TC):
  $$TC(z) = KL(q(z) \parallel \prod_{j=1}^d q(z_j))$$
- **Term Breakdown**:
  In the decomposed ELBO, the standard KL term $KL(q(z|x) \parallel p(z))$ already includes $I(x; z) + TC(z) + \sum_j KL(q(z_j) \parallel p(z_j))$. Adding $-\gamma TC(z)$ increases the effective penalty on Total Correlation to $(1 + \gamma)$, while preserving the penalty on the index-code mutual information $I(x; z)$ and dimension-wise KL at weight $1$. This decouples latent factor independence from information bottleneck compression.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: Adversarial Discriminator via the Density-Ratio Trick.
- **Formulation**:
  From Section 3, Equation (3):
  $$TC(z) = KL(q(z) \parallel \bar{q}(z)) = \mathbb{E}_{q(z)}\left[\log \frac{q(z)}{\bar{q}(z)}\right] \approx \mathbb{E}_{q(z)}\left[\log \frac{D(z)}{1 - D(z)}\right]$$
- **Sampling Mechanism**:
  - Samples from $q(z)$ are obtained by sampling $x^{(i)} \sim p_{data}(x)$ and $z \sim q(z|x^{(i)})$.
  - Samples from the product of marginals $\bar{q}(z)$ are obtained by sampling a minibatch from $q(z)$ and randomly permuting the values across the batch independently for each latent dimension (Algorithm 1, *permute_dims*).
  - An auxiliary discriminator (MLP) $D_\psi(z)$ is trained concurrently with cross-entropy to distinguish between true samples from $q(z)$ and permuted samples from $\bar{q}(z)$ (Algorithm 2).

Verbatim quote from Section 3, Page 3:
> "Having access to samples from both distributions allows us to minimise their KL divergence using the density-ratio trick (Nguyen et al., 2010; Sugiyama et al., 2012) which involves training a classifier/discriminator to approximate the density ratio that arises in the KL term. Suppose we have a discriminator $D$ (in our case an MLP) that outputs an estimate of the probability $D(z)$ that its input is a sample from $q(z)$ rather than from $\bar{q}(z)$. Then we have
> $TC(z) = KL(q(z)\parallel\bar{q}(z)) = \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{\bar{q}(z)} \right] \approx \mathbb{E}_{q(z)} \left[ \log \frac{D(z)}{1 - D(z)} \right]. \quad (3)$"

---

## 5. Datasets, Metrics, and Results
- **Datasets**:
  - 2D Shapes (dSprites): 737,280 binary images with 5 ground truth factors (shape, scale, rotation, pos X, pos Y) (Section 6).
  - 3D 3D Shapes / 3D Faces, 3D Chairs, CelebA (Section 6, Appendix A).
- **Metrics**:
  - Reconstruction error (pixel-wise MSE / cross-entropy).
  - Higgins et al. Disentanglement Metric (2016).
  - **Proposed FactorVAE Disentanglement Metric**:
    1. Fix one factor $k$ and generate images with all other factors random.
    2. Obtain representations $z = \mathbb{E}[q(z|x)]$ and normalize each dimension by its empirical standard deviation $s_j$.
    3. Compute empirical variance along each dimension; find the index of the dimension with minimum variance: $d^* = \arg\min_j \mathrm{Var}(z_j / s_j)$.
    4. Train a majority-vote classifier mapping $d^*$ to factor index $k$. Classification accuracy is the FactorVAE metric score.
- **Results**:
  - FactorVAE achieved a disentanglement score of **$0.889$** ($\gamma = 35$) on 2D Shapes, outperforming $\beta$-VAE ($0.814$ at $\beta = 4$) at lower reconstruction error (Section 6, Figure 5, 6).
  - Accurately decoupled Total Correlation minimization from reconstruction fidelity, where true TC decreased steadily during training without collapsing latents (Figure 7).

---

## 6. Limitations
1. **Adversarial Training Overhead & Instability**: Requires maintaining and training an auxiliary discriminator network in an alternating min-max fashion, introducing additional optimization hyperparameters (discriminator learning rate, optimizer, architecture) and potential instability.
2. **Permutation Minibatch Requirement**: Requires drawing a second distinct minibatch per training step to perform dimension permutation, doubling the batch-sampling cost.
3. **Discriminator Bias**: If the discriminator fails to converge, produces high-variance gradients, or suffers mode collapse, the density ratio $\frac{D(z)}{1 - D(z)}$ becomes an inaccurate surrogate for the true Total Correlation.
