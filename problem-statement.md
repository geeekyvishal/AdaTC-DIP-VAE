So below is the problem statement that i have to do the research on for my Generative AI course at college



Problem Statement
Title
Disentangled Representation Learning with β-VAE/β-TCVAE and a Quantitative Disentanglement Benchmark
Problem Statement
Variational Autoencoders (VAEs) are widely used for learning latent representations of data. However, standard VAEs do not explicitly ensure that individual latent dimensions capture independent and interpretable generative factors.
For example, when processing images, a model should ideally learn separate latent dimensions corresponding to properties such as object shape, size, color, orientation, and position. Achieving such disentangled representations is challenging because conventional VAE objectives may allow dependencies between latent dimensions.
This project aims to investigate and improve disentangled representation learning by studying existing approaches, particularly β-VAE and β-TCVAE, and proposing a modified KL-divergence decomposition objective that encourages greater independence among latent variables while preserving reconstruction quality.
The proposed approach will be evaluated against existing methods using the dSprites and 3D Shapes datasets, which provide ground-truth generative factors. Quantitative evaluation will be performed using three established disentanglement metrics: DCI, Mutual Information Gap (MIG), and FactorVAE Score.
The objective is to determine whether the proposed modification can achieve better disentanglement performance compared to existing approaches without significantly compromising reconstruction quality.
Key Objectives

1. Baseline Implementation: Implement standard VAE, β-VAE, and β-TCVAE architectures.
2. Objective Modification: Study the KL-divergence decomposition and propose a modified objective function to improve latent factor independence.
3. Dataset Evaluation: Train and evaluate models using the dSprites and 3D Shapes datasets.
4. Quantitative Benchmarking: Compare models using DCI, MIG, and FactorVAE scores.
5. Performance Analysis: Analyze the trade-off between disentanglement quality and reconstruction accuracy.
6. Comparative Study: Determine whether the proposed objective offers measurable improvements over existing approaches.

Expected Outcome
A modified VAE training objective that aims to produce more interpretable and disentangled latent representations, supported by mathematical justification and experimental comparisons against established VAE variants.
In Simple Terms
We want to:
Build → Improve → Evaluate → Compare

* Build existing VAE variants.
* Improve their mathematical loss function.
* Evaluate how independently they learn image properties.
* Compare our proposed approach against existing methods.

The central research goal: Improve the quality of disentangled latent representations while maintaining good image reconstruction quality.

dataset : https://github.com/google-deepmind/3d-shapes
https://github.com/google-deepmind/dsprites-dataset