import json

data = {
    "candidate_name": "Unified Dual-Stream TC-HSIC Objective (DS-TC-HSIC VAE)",
    "objective_latex": r"""\mathcal{L}_{\text{Unified}}(\theta, \phi) = \mathbb{E}_{p(x)}\left[\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]\right] - I_q(x; z) - \beta(t) KL\left(q(z_{\mathcal{A}}) \,\Big\|\, \prod_{j \in \mathcal{A}} q(z_j)\right) - \sum_{j=1}^d KL(q(z_j) \parallel p(z_j)) - \lambda(t) \sum_{1 \le j < k \le d} w_j w_k \, \widehat{\text{HSIC}}(\mu_j, \mu_k)"""
}
print("Length of candidate_name:", len(data["candidate_name"]))
