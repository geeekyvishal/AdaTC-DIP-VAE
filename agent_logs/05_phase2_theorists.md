# Agent Log: Phase 2 — Theory & Objective Formulation

- **Module**: Phase 2 (Theory, Objective Derivation, Mathematical Audit, Synthesis)
- **Timestamp**: 2026-11-17T21:55:00+05:30
- **Roster & Architecture**:
  - **Theorist A**: Adaptive Total Correlation Annealing + Deterministic Mean Decorrelation (`AdaTC-DIP-VAE`)
  - **Theorist B**: Coordinate Sparsity + Automatic Relevance Determination (`SB-ARD-TCVAE`)
  - **Theorist C**: Dual-Stream Total Correlation + Hilbert-Schmidt Independence Criterion (`DS-TC-HSIC VAE`)
  - **Skeptic**: Rigorous mathematical audit, citation check, gradient verification, and failure mode analysis
  - **Synthesizer**: Comparative synthesis, candidate ranking, incorporation of skeptic revisions, and generation of `theory/candidates.md`
- **Scope**: Grounded strictly in `./research-paper/*.pdf`

---

## 1. Candidate Proposals Summary

### Candidate 1: AdaTC-DIP-VAE (Theorist A)
- **Core Concept**: Bridges the gap between the sampled aggregated posterior $q(z)$ and deterministic encoder means $\mu_\phi(x)$ (Locatello et al. 2019, Section 5.1) while introducing a $C^1$-continuous cosine annealing schedule on Total Correlation $\beta(t)$ that respects the hierarchical factor acquisition curriculum (Burgess et al. 2018, Section 4.3).
- **Objective Function**:
  $$\mathcal{L}_{\text{AdaTC-DIP}} = \mathcal{L}_{\text{ELBO}} - (\beta(t) - 1) KL\left(q(z) \parallel \prod_{j=1}^d q(z_j)\right) - \lambda_{od} \sum_{i \ne j} \left[ \text{Cov}_{p(x)}[\mu_\phi(x)] \right]_{ij}^2$$
- **Addressed Gaps**: Gap 3 (Static trade-off), Gap 4 (Sampled vs. deterministic disconnect), Gap 1 (Second-order mean decorrelation without adversarial discriminator).
- **Hyperparameters**: $\beta_{\text{target}} \in [4, 10]$, $T_{\text{warm}} \approx 20,000$ steps, $\lambda_{od} \in [5, 20]$.

### Candidate 2: SB-ARD-TCVAE (Theorist B)
- **Core Concept**: $\ell_1$ coordinate sparsity on encoder means to break rotational symmetry (Mathieu et al. 2019, Theorem 2) + learnable anisotropic prior variances $\sigma_{0,j}^2$ with exponential ARD shrinkage to prune surplus dimensions when $d > K$ (Locatello et al. 2019, Section 5).
- **Addressed Gaps**: Gap 2 (Rotational invariance), Gap 5 (Excess capacity).
- **Critique Finding**: Rejected due to fatal mathematical singularities (unbounded objective as $\sigma_{0,j}^2 \to 0$, infinite gradient, NaN explosion) and conceptual misapplication of $\ell_1$ sparsity to continuous generative factors.

### Candidate 3: DS-TC-HSIC VAE (Theorist C)
- **Core Concept**: Proves that DIP-VAE-I's covariance penalty is exactly linear-kernel HSIC, then generalizes to pairwise Gaussian RBF-kernel Hilbert-Schmidt Independence Criterion on encoder means to capture all non-linear dependencies. Incorporates soft dimension pruning and adaptive scheduling tied to reconstruction loss.
- **Addressed Gaps**: Gap 1 (Non-linear independence without GANs), Gap 2 (Coordinate alignment), Gap 3 (Reconstruction-adaptive schedule), Gap 4 (Dual-stream sampled/deterministic), Gap 5 (Soft dimension gating).
- **Critique Finding**: Theoretically elegant with rigorous proofs, but mechanically fragile for a course project due to initialization collapse ($\mathcal{A} = \emptyset$), limit-cycle feedback oscillations, and 6 interacting hyperparameters.

---

## 2. Skeptic Audit & Verification Summary

The Skeptic reviewed all 3 proposals against the literature summaries in `./research-paper-summarised-md-files/`:
- **Candidate 1 Score: 8.5/10 (Accepted with Minor Revision)**:
  - ELBO decomposition (Chen et al. 2018, Eq. 2) and cosine schedule are mathematically exact.
  - Closed-form analytical gradient $\nabla_{\mu} \mathcal{P}_{\text{cov}} = \frac{4}{M-1}\tilde{U}A$ verified.
  - *Identified Vulnerability*: Setting $\lambda_d = 0$ leaves the scale of $\mu_\phi(x)$ unanchored from below; extreme $\lambda_{od}$ can cause variance collapse $\text{Var}(\mu_j) \to 0, \mathbb{E}[\sigma_j^2] \to 1$.
  - *Resolution*: Incorporate a small diagonal anchor $\lambda_d \in [0.5, 1.0]$.
- **Candidate 2 Score: 3.0/10 (Rejected)**:
  - Singular objective: $\lim_{\sigma_{0,j}^2 \to 0} \mathcal{L} = +\infty$ and gradient explodes to $-\infty$.
  - $\ell_1$ sparsity crushes continuous factors that are simultaneously present in every image.
  - Misattributed citation to Mathieu et al. (2019).
- **Candidate 3 Score: 6.5/10 (Rejected as-is / Proposed as Stretch Hybrid)**:
  - Exact proof of DIP-VAE as linear HSIC; correct analytical gradient for RBF-HSIC.
  - Active set $\mathcal{A} = \emptyset$ at initialization ($t = 0$), causing undefined TC.
  - Closed-loop feedback between reconstruction error and regularization causes limit-cycle oscillations.
  - Recommended simplification: AdaTC-HSIC hybrid (open-loop cosine schedule, global pairwise HSIC).

---

## 3. Final Artifacts Generated

1. `theory/candidates.md` (~63 KB): Complete theoretical synthesis document, containing:
   - Executive summary and comparative ranking
   - Foundational literature citations and verbatim quotes
   - Full mathematical derivations from first principles
   - Primary Recommended Objective: **AdaTC-DIP-VAE (Revised with $\lambda_d$ anchor)**
   - Secondary High-Aspiration Hybrid: **AdaTC-HSIC VAE**
   - Detailed failure mode audit of all 3 candidates
   - Gradient tractability and implementation complexity analysis
   - Concrete PyTorch implementation plan and hyperparameter calibration table
   - Benchmark evaluation protocol for dSprites and 3D Shapes (MIG, DCI, FactorVAE Score)
2. `theory/critique.md` (~25 KB): Skeptic's comprehensive mathematical audit and critique.
3. `theory/candidate_A_adaptive_tc_mean_decorrelation.md` (~30 KB): Theorist A's full raw derivation and analysis.
4. `agent_logs/05_phase2_theorists.md`: This step log.

---

## 4. Phase Status & Next Actions

- **Phase 1**: Completed (Paper summaries + Gap analysis)
- **Phase 2**: Completed (`theory/candidates.md` + `theory/critique.md`)
- **Awaiting User Review & Approval**: In accordance with the project instructions, stopping now to present the recommended objective to the user before proceeding to Phase 3 (PyTorch implementation in `src/` and `tests/`).
