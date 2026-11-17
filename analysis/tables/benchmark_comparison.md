# Benchmark Comparison across Models on dSprites (N=5000 test set)

| Model Architecture | Effective $\beta$ | Reconstruction MSE $\downarrow$ | Relative MSE vs. VAE | MIG $\uparrow$ | DCI Disentanglement $\uparrow$ | FactorVAE Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard VAE** | $1.0$ | **0.00470** | Baseline ($0\%$) | $0.0302$ | $0.2613$ | $0.6450$ |
| **$\beta$-VAE** | $4.0$ (Static) | $0.00576$ | $+22.6\%$ | $0.0290$ | $0.1818$ | $0.4467$ |
| **$\beta$-TCVAE** | $6.0$ (Static) | $0.00785$ | $+67.0\%$ | $0.1512$ | $0.3674$ | $0.5667$ |
| **AdaTC-DIP-VAE (Ours)** | **$1.0 \to 6.0$** | **0.00679** | **$+44.5\%$** | **0.2385** | **0.5436** | **0.7100** |

## Key Findings:
- **Highest MIG**: AdaTC-DIP-VAE achieves **0.2385**, beating $\beta$-TCVAE ($0.1512$) by **+57.7%**.
- **Highest DCI**: AdaTC-DIP-VAE achieves **0.5436**, beating $\beta$-TCVAE ($0.3674$) by **+48.0%**.
- **Highest FactorVAE**: AdaTC-DIP-VAE achieves **0.7100**, beating all baselines.
- **Lower Reconstruction Blur than $\beta$-TCVAE**: AdaTC-DIP-VAE achieves **0.00679** vs. $0.00785$ (13.5% lower reconstruction error than $\beta$-TCVAE).
