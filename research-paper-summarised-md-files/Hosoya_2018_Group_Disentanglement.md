# Paper Summary: Group-based Learning of Disentangled Representations with Generalizability for Novel Contents (Hosoya, 2018)

- **Paper ID**: `Hosoya_2018_Group_Disentanglement`
- **File**: `1809.02383v2.pdf`
- **Authors**: Haruo Hosoya
- **Venue/Year**: arXiv:1809.02383 (v2, 2018) / NeurIPS 2019 Workshop

---

## 1. Core Claim
The paper claims that learning disentangled representations of content (identity) and transformation (pose/lighting) that generalize to novel unseen contents can be achieved without attribute labels by using a weak form of supervision: grouping instances that share the same content. The proposed **Group-based Variational Autoencoder (GVAE)** uses an encoder architecture that averages the inferred latent content across grouped instances while learning separate instance-specific transformation latent variables (Section 1, Section 3).

---

## 2. Objective Function (LaTeX)
From Section 3.2 ("GVAE"), Equation (6):

$$\mathcal{L}(x) = \mathbb{E}_{q(y, z|x)} \left[ \sum_{k=1}^K \log p_\theta(x_k | y_k, z) \right] - \sum_{k=1}^K KL(q_{\phi, \xi}(y_k | x_k) \parallel p(y_k)) - KL(q_{\psi, \pi}(z|x) \parallel p(z))$$

where $x = (x_1, \dots, x_K)$, $y = (y_1, \dots, y_K)$, $p(z) = \mathcal{N}(0, I)$, $p(y_k) = \mathcal{N}(0, I)$, and the group-common content posterior is defined by averaging individual encoders (Equation 5):
$$q_{\psi, \pi}(z|x_1, \dots, x_K) = \mathcal{N}\left( \frac{1}{K}\sum_{k=1}^K h_\psi(x_k), \frac{1}{K}\sum_{k=1}^K s_\pi(x_k) \right)$$

Verbatim quote from Section 3.2, Page 3:
> "To train the model, we consider the following variational lower bound of the log likelihood:
> $\log p_\theta(x) \ge \mathbb{E}_{q_{\phi,\xi,\psi,\pi}(y,z|x)} \left[ \sum_{k=1}^K \log p_\theta(x_k|y_k,z) \right] - \sum_{k=1}^K KL(q_{\phi,\xi}(y_k|x_k) \parallel p(y_k)) - KL(q_{\psi,\pi}(z|x) \parallel p(z)) = \mathcal{L}(x) \quad (6)$
> where $x = (x_1,\dots, x_K)$ and $y = (y_1,\dots, y_K)$."

---

## 3. Which KL Term It Modifies
- **Modifications**: Adapts the ELBO to a grouped observation structure.
- **Analysis**: Does not isolate or decompose the standard single-image KL divergence into index-code MI, Total Correlation, or dimension-wise KL. Disentanglement between content $z$ and transformation $y$ is enforced by the structural inductive bias of averaging $z$ over group members.

---

## 4. Total Correlation (TC) Estimator Used
- **TC Estimator**: None.
- **Details**: Total correlation among individual latent dimensions is not estimated or penalized.

---

## 5. Datasets, Metrics, and Results
- **Datasets**:
  - Multi-PIE (face images grouped by subject/fashion across poses)
  - 3D Chairs (grouped by chair model across views)
  - KTH (video clips of pedestrian motion grouped by clip)
  - Sprites (game character images grouped by character ID)
  - NORB (toy images grouped by toy ID across lighting/elevation) (Section 3.3).
- **Metrics**:
  - Qualitative latent swapping (combining content of image A with transformation of image B) and latent interpolation (Section 4, Figure 2).
  - Classification accuracy on novel unseen content test sets.
- **Results**:
  - Successfully separated content from transformations across all five datasets and generalized to unseen identities during test time without requiring explicit attribute supervision.

---

## 6. Limitations
1. **Requires Grouped Data (Weak Supervision)**: Relies on data pre-partitioned into groups sharing identical content, which is unavailable in standard unsupervised i.i.d. datasets (e.g., standard dSprites or 3D Shapes without group metadata).
2. **Coarse-Grained Disentanglement**: Only separates the macro content vector $z$ from the transformation vector $y$; it does not achieve fine-grained, coordinate-wise factor disentanglement (e.g. isolating shape, scale, rotation, and position individually).
