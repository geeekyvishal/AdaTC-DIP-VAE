# Agent Log 01: Scout & Paper Cataloguer

## Overview
- **Agent**: Scout / Cataloguer
- **Target Directory**: `./research-paper/`
- **Action**: Identified, catalogued, and verified all 12 PDF files present in the local repository.
- **Timestamp**: 2026-11-17T21:14:00+05:30

## Catalog of Papers
1. **1312.6114v11.pdf**
   - Title: *Auto-Encoding Variational Bayes*
   - Authors: Diederik P. Kingma, Max Welling (2013 / ICLR 2014)
   - Canonical ID: `Kingma_2013_VAE`
   - Relevance: Foundational VAE derivation, ELBO, reparameterization trick, SGVB estimator.

2. **291_beta_vae_learning_basic_visual.pdf**
   - Title: *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*
   - Authors: Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner (ICLR 2017)
   - Canonical ID: `Higgins_2017_beta_VAE`
   - Relevance: Introduction of $\beta$-VAE ($\beta > 1$) constrained optimization framework for unsupervised disentanglement.

3. **1804.03599v1.pdf**
   - Title: *Understanding disentangling in β-VAE*
   - Authors: Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, Alexander Lerchner (arXiv 2018)
   - Canonical ID: `Burgess_2018_Understanding_beta_VAE`
   - Relevance: Capacity-controlled objective $\mathcal{L}(\theta, \phi; x, z, C) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \gamma |D_{KL}(q_\phi(z|x) \parallel p(z)) - C|$ and progressive information bottleneck.

4. **1802.05983v3.pdf**
   - Title: *Disentangling by Factorising*
   - Authors: Hyunjik Kim, Andriy Mnih (ICML 2018)
   - Canonical ID: `Kim_Mnih_2018_FactorVAE`
   - Relevance: FactorVAE; isolates Total Correlation (TC) from the aggregate posterior $q(z)$ using an auxiliary discriminator and density ratio trick. Introduces FactorVAE metric.

5. **1802.04942v5.pdf**
   - Title: *Isolating Sources of Disentanglement in VAEs*
   - Authors: Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud (NeurIPS 2018)
   - Canonical ID: `Chen_2018_beta_TCVAE`
   - Relevance: $\beta$-TCVAE; exact KL divergence decomposition into Index-Code Mutual Information, Total Correlation, and Dimension-wise KL. Proposes Minibatch Weighted Sampling (MWS) estimator and Mutual Information Gap (MIG).

6. **1711.00848v3.pdf**
   - Title: *Variational Inference of Disentangled Latent Concepts from Unlabeled Observations*
   - Authors: Abhishek Kumar, Prasanna Sattigeri, Avinash Balakrishnan (ICLR 2018)
   - Canonical ID: `Kumar_2018_DIP_VAE`
   - Relevance: DIP-VAE (Disentangled Inferred Prior VAE); matches moments of the aggregated posterior $q(z)$ to a factorized prior $p(z)$ (DIP-VAE-I and DIP-VAE-II).

7. **1602.05473v4.pdf**
   - Title: *Auxiliary Deep Generative Models*
   - Authors: Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, Ole Winther (ICML 2016)
   - Canonical ID: `Maaloe_2016_ADGM`
   - Discrepancy Note: In `x.bash`, the curl script labeled this entry as `Hoffman_Johnson_2016_ELBO_Surgery.pdf`. However, the downloaded arXiv paper `1602.05473` is actually *Auxiliary Deep Generative Models* by Maaløe et al. Verified against PDF text.

8. **1802.05831v1.pdf**
   - Title: *Component Outage Estimation based on Support Vector Machine*
   - Authors: Rozhin Eskandarpour, Amin Khodaei (2018)
   - Canonical ID: `Eskandarpour_2018_SVM_Outage`
   - Discrepancy Note: In `x.bash`, this entry was labeled `Eastwood_Williams_2018_DCI.pdf`. However, arXiv `1802.05831` is a power systems engineering paper on hurricane power grid component outages. Verified against PDF text.

9. **1809.02383v2.pdf**
   - Title: *Group-based Learning of Disentangled Representations with Generalizability for Novel Contents*
   - Authors: Haruo Hosoya (2018)
   - Canonical ID: `Hosoya_2018_Group_Disentanglement`
   - Discrepancy Note: In `x.bash`, this entry was labeled `Esmaeili_2019_Structured_Disentangled.pdf`. However, arXiv `1809.02383` is Hosoya's paper on group-based disentanglement. Verified against PDF text.

10. **1811.12359v4.pdf**
    - Title: *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*
    - Authors: Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem (ICML 2019)
    - Canonical ID: `Locatello_2019_Challenging_Assumptions`
    - Relevance: Fundamental impossibility theorem of unsupervised disentanglement without inductive bias; empirical benchmark across 12,000+ models comparing $\beta$-VAE, FactorVAE, $\beta$-TCVAE, DIP-VAE-I, DIP-VAE-II with metrics (BetaVAE score, FactorVAE score, MIG, DCI, SAP, Modularity).

11. **1812.02230v1.pdf**
    - Title: *Towards a Definition of Disentangled Representations*
    - Authors: Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, Alexander Lerchner (arXiv 2018)
    - Canonical ID: `Higgins_2018_Definition_Disentangled`
    - Relevance: Formal mathematical definition of disentanglement based on group theory, symmetry transformations, and Lie algebras.

12. **1812.02833v3.pdf**
    - Title: *Disentangling Disentanglement in Variational Autoencoders*
    - Authors: Emile Mathieu, Tom Rainforth, N. Siddharth, Yee Whye Teh (ICML 2019)
    - Canonical ID: `Mathieu_2019_Disentangling_Disentanglement`
    - Relevance: Distinguishes between latent decomposition (overlap/mutual information) and axis alignment (rotation invariance of isotropic Gaussian priors); formalizes why standard VAE/$\beta$-VAE struggle with axis alignment without specific architectural/prior biases.
