"""
Disentanglement VAE Loss Functions and Regularizers.

Implements:
1. Standard VAE Loss (-ELBO; Kingma & Welling, 2013)
2. Beta-VAE Loss (Higgins et al., 2017)
3. Beta-TCVAE Loss with Minibatch Weighted Sampling (MWS; Chen et al., 2018, Eq. 2-4)
4. Proposed AdaTC-DIP-VAE Loss (Adaptive Cosine TC Annealing + Deterministic Mean Decorrelation)
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    loss_type: str = "bce",
    reduction: str = "sum_per_sample",
) -> torch.Tensor:
    """
    Compute reconstruction loss between reconstructed and target images.

    Args:
        recon_x: Reconstructed images in [0, 1], shape (B, C, H, W).
        x: Target images in [0, 1], shape (B, C, H, W).
        loss_type: 'bce' (Binary Cross Entropy) or 'mse' (Mean Squared Error).
        reduction: 'sum_per_sample' (sum over channels & pixels, averaged over batch; standard
                   in disentanglement literature) or 'mean' (averaged over all elements).

    Returns:
        Scalar reconstruction loss tensor.
    """
    batch_size = x.size(0)
    # Clip recon_x for numerical stability when using bce
    if loss_type == "bce":
        eps = 1e-7
        recon_x_clamped = torch.clamp(recon_x, min=eps, max=1.0 - eps)
        loss = F.binary_cross_entropy(recon_x_clamped, x, reduction="none")
    elif loss_type == "mse":
        loss = F.mse_loss(recon_x, x, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'bce' or 'mse'.")

    if reduction == "sum_per_sample":
        # Sum over spatial and channel dims, then mean over batch
        return loss.view(batch_size, -1).sum(dim=1).mean()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}.")


def compute_gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Standard analytical KL divergence KL(q(z|x) || p(z)) for diagonal Gaussian
    and standard normal prior N(0, I):
    KL = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))

    Args:
        mu: Latent mean of shape (B, d).
        logvar: Latent log-variance of shape (B, d).

    Returns:
        Scalar tensor with batch-averaged KL divergence.
    """
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
    return kl_per_sample.mean()


def compute_mws_kl_decomposition(
    z_sample: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    dataset_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Minibatch Weighted Sampling (MWS) estimator from Chen et al. (2018, Eq. 2-4).

    Decomposes the expected KL divergence E_{p(x)}[KL(q(z|x) || p(z))] into:
    1. Index-Code Mutual Information: I_q(x; z)
    2. Total Correlation: TC(z) = KL(q(z) || prod_j q(z_j))
    3. Dimension-wise KL: sum_j KL(q(z_j) || p(z_j))

    Args:
        z_sample: Reparameterized latent codes of shape (B, d).
        mu: Encoder posterior means of shape (B, d).
        logvar: Encoder posterior log-variances of shape (B, d).
        dataset_size: Total number of observations N in the training dataset.

    Returns:
        Tuple of (mi_loss, tc_loss, dw_kl_loss):
          - mi_loss: Scalar tensor for I_q(x; z).
          - tc_loss: Scalar tensor for Total Correlation TC(z).
          - dw_kl_loss: Scalar tensor for Dimension-wise KL.
    """
    batch_size, latent_dim = z_sample.shape

    # Expand tensors for pairwise log density computation:
    # z: (B, 1, d), mu: (1, B, d), logvar: (1, B, d)
    z_exp = z_sample.unsqueeze(1)
    mu_exp = mu.unsqueeze(0)
    logvar_exp = logvar.unsqueeze(0)

    # Per-dimension log Gaussian density log q(z_k^{(i)} | x^{(j)}): shape (B, B, d)
    # log N(z; mu, sigma^2) = -0.5 * [log(2*pi) + logvar + (z - mu)^2 / exp(logvar)]
    log_q_z_per_dim = -0.5 * (
        math.log(2.0 * math.pi)
        + logvar_exp
        + (z_exp - mu_exp).pow(2) / torch.exp(logvar_exp)
    )

    # Joint log density log q(z^{(i)} | x^{(j)}): sum over latent dims -> shape (B, B)
    log_q_z_joint = log_q_z_per_dim.sum(dim=-1)

    # Minibatch-weighted aggregate log densities using logsumexp over j (dim 1):
    # log q(z^{(i)}) approx logsumexp_j(log q(z^{(i)} | x^{(j)})) - log(B * N)
    log_dataset_factor = math.log(batch_size * dataset_size)
    log_q_z_agg = torch.logsumexp(log_q_z_joint, dim=1) - log_dataset_factor  # (B,)

    # Dimension-wise aggregate log density log q(z_k^{(i)}): shape (B, d)
    log_q_zk_agg = torch.logsumexp(log_q_z_per_dim, dim=1) - log_dataset_factor  # (B, d)
    log_prod_q_zk = log_q_zk_agg.sum(dim=-1)  # (B,)

    # Conditional log density under sample's own conditioning x^{(i)} (diagonal i == j): shape (B,)
    log_q_z_cond = torch.diagonal(log_q_z_joint, dim1=0, dim2=1)

    # Standard normal prior log density log p(z^{(i)}): shape (B,)
    # p(z_k) = N(0, 1) -> log p(z_k) = -0.5 * [log(2*pi) + z_k^2]
    log_p_zk = -0.5 * (math.log(2.0 * math.pi) + z_sample.pow(2))  # (B, d)
    log_p_z = log_p_zk.sum(dim=-1)  # (B,)

    # Decomposition terms:
    # 1. Index-Code Mutual Information: E[log q(z|x) - log q(z)]
    mi_loss = (log_q_z_cond - log_q_z_agg).mean()

    # 2. Total Correlation: E[log q(z) - sum_k log q(z_k)]
    # Clamped to min=0.0 to prevent negative TC due to finite minibatch sample bias
    tc_loss = torch.clamp((log_q_z_agg - log_prod_q_zk).mean(), min=0.0)

    # 3. Dimension-wise KL: sum_k E[log q(z_k) - log p(z_k)]
    dw_kl_loss = (log_q_zk_agg.sum(dim=-1) - log_p_z).mean()

    return mi_loss, tc_loss, dw_kl_loss


def compute_covariance_penalty(
    mu: torch.Tensor, lambda_od: float = 10.0, lambda_d: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Minibatch Deterministic Mean Covariance Regularizer (Kumar et al., 2018; DIP-VAE-I).

    Computes:
      P_cov = lambda_od * sum_{i != j} Cov(mu)_{ij}^2 + lambda_d * sum_i (Cov(mu)_{ii} - 1)^2

    Args:
        mu: Batch of encoder posterior means of shape (B, d).
        lambda_od: Weight for off-diagonal covariance penalty (drives cross-covariances to 0).
        lambda_d: Weight for diagonal covariance penalty (drives variances to 1; default 0).

    Returns:
        Tuple of (cov_penalty, off_diag_penalty, diag_penalty):
          - cov_penalty: Combined weighted penalty scalar tensor.
          - off_diag_penalty: sum_{i != j} Cov_{ij}^2.
          - diag_penalty: sum_i (Cov_{ii} - 1)^2.
    """
    batch_size = mu.size(0)
    if batch_size < 2:
        zero = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        return zero, zero, zero

    # Center encoder means: mu_centered = mu - mean(mu, dim=0)
    mu_centered = mu - torch.mean(mu, dim=0, keepdim=True)

    # Sample covariance matrix: Cov = (1 / (B - 1)) * (mu_centered^T @ mu_centered)
    cov = torch.matmul(mu_centered.t(), mu_centered) / (batch_size - 1.0)

    # Off-diagonal elements (zero out the diagonal)
    diag_cov = torch.diag(cov)
    cov_off_diag = cov - torch.diag_embed(diag_cov)

    off_diag_penalty = torch.sum(cov_off_diag.pow(2))
    diag_penalty = torch.sum((diag_cov - 1.0).pow(2))

    cov_penalty = (lambda_od * off_diag_penalty) + (lambda_d * diag_penalty)
    return cov_penalty, off_diag_penalty, diag_penalty


def get_cosine_annealed_beta(
    step: int, beta_target: float = 6.0, warm_steps: int = 10000
) -> float:
    """
    Cosine annealing schedule for Total Correlation weight beta(t).

    beta(t) = 1 + ((beta_target - 1) / 2) * (1 - cos(pi * t / T_warm)) for t < T_warm,
              beta_target for t >= T_warm.

    Properties:
      - beta(0) = 1.0 (starts at standard ELBO)
      - beta'(0) = 0.0 (smooth initial acceleration, avoids initial shock)
      - beta(T_warm) = beta_target
      - beta'(T_warm) = 0.0 (smooth asymptote)
    """
    if warm_steps <= 0 or step >= warm_steps:
        return float(beta_target)
    ratio = float(step) / float(warm_steps)
    beta_t = 1.0 + ((beta_target - 1.0) / 2.0) * (1.0 - math.cos(math.pi * ratio))
    return float(beta_t)


class StandardVAELoss(nn.Module):
    """
    Standard VAE Loss (-ELBO; Kingma & Welling, 2013).
    L = Recon_loss + KL(q(z|x) || p(z))
    """

    def __init__(
        self,
        loss_type: str = "bce",
        reduction: str = "sum_per_sample",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        recon_loss = compute_reconstruction_loss(
            recon_x, x, loss_type=self.loss_type, reduction=self.reduction
        )
        kl_loss = compute_gaussian_kl(mu, logvar)
        total_loss = recon_loss + kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class BetaVAELoss(nn.Module):
    """
    Beta-VAE Loss (Higgins et al., 2017).
    L = Recon_loss + beta * KL(q(z|x) || p(z))
    """

    def __init__(
        self,
        beta: float = 4.0,
        loss_type: str = "bce",
        reduction: str = "sum_per_sample",
    ):
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        recon_loss = compute_reconstruction_loss(
            recon_x, x, loss_type=self.loss_type, reduction=self.reduction
        )
        kl_loss = compute_gaussian_kl(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "beta": torch.tensor(self.beta, device=recon_x.device),
        }


class BetaTCVAELoss(nn.Module):
    """
    Beta-TCVAE Loss with Minibatch Weighted Sampling (Chen et al., 2018).

    L = Recon_loss + alpha * I(x; z) + beta * TC(z) + gamma * Dimwise_KL
    Standard beta-TCVAE sets alpha = 1.0, gamma = 1.0, and tunes beta > 1.0.
    """

    def __init__(
        self,
        beta: float = 6.0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        dataset_size: int = 737280,
        loss_type: str = "bce",
        reduction: str = "sum_per_sample",
    ):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.dataset_size = dataset_size
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        dataset_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        n_data = dataset_size if dataset_size is not None else self.dataset_size

        recon_loss = compute_reconstruction_loss(
            recon_x, x, loss_type=self.loss_type, reduction=self.reduction
        )
        mi_loss, tc_loss, dw_kl_loss = compute_mws_kl_decomposition(
            z, mu, logvar, dataset_size=n_data
        )

        total_loss = (
            recon_loss
            + self.alpha * mi_loss
            + self.beta * tc_loss
            + self.gamma * dw_kl_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "mi_loss": mi_loss,
            "tc_loss": tc_loss,
            "dw_kl_loss": dw_kl_loss,
            "beta": torch.tensor(self.beta, device=recon_x.device),
        }


class AdaTCDIPVAELoss(nn.Module):
    """
    Proposed AdaTC-DIP-VAE Loss:
    Adaptive Cosine TC Annealing + Deterministic Mean Decorrelation.

    L = Recon_loss + alpha * I(x; z) + beta(t) * TC(z) + gamma * Dimwise_KL + P_cov(mu)
    where:
      - beta(t) follows a smooth cosine ramp from 1.0 to beta_target over warm_steps.
      - P_cov(mu) = lambda_od * sum_{i != j} Cov(mu)_{ij}^2 + lambda_d * sum_i (Cov(mu)_{ii} - 1)^2
        applied on the encoder deterministic posterior means mu_phi(x).
    """

    def __init__(
        self,
        beta_target: float = 6.0,
        warm_steps: int = 10000,
        lambda_od: float = 10.0,
        lambda_d: float = 0.0,
        alpha: float = 1.0,
        gamma: float = 1.0,
        dataset_size: int = 737280,
        loss_type: str = "bce",
        reduction: str = "sum_per_sample",
    ):
        super().__init__()
        self.beta_target = beta_target
        self.warm_steps = warm_steps
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d
        self.alpha = alpha
        self.gamma = gamma
        self.dataset_size = dataset_size
        self.loss_type = loss_type
        self.reduction = reduction

    def get_current_beta(self, current_step: int) -> float:
        return get_cosine_annealed_beta(
            current_step, beta_target=self.beta_target, warm_steps=self.warm_steps
        )

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        current_step: int,
        dataset_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        n_data = dataset_size if dataset_size is not None else self.dataset_size
        beta_t = self.get_current_beta(current_step)

        recon_loss = compute_reconstruction_loss(
            recon_x, x, loss_type=self.loss_type, reduction=self.reduction
        )
        mi_loss, tc_loss, dw_kl_loss = compute_mws_kl_decomposition(
            z, mu, logvar, dataset_size=n_data
        )

        cov_loss, off_diag_cov, diag_cov = compute_covariance_penalty(
            mu, lambda_od=self.lambda_od, lambda_d=self.lambda_d
        )

        total_loss = (
            recon_loss
            + self.alpha * mi_loss
            + beta_t * tc_loss
            + self.gamma * dw_kl_loss
            + cov_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "mi_loss": mi_loss,
            "tc_loss": tc_loss,
            "dw_kl_loss": dw_kl_loss,
            "cov_loss": cov_loss,
            "off_diag_cov": off_diag_cov,
            "diag_cov": diag_cov,
            "beta_t": torch.tensor(beta_t, device=recon_x.device),
        }


# Standard Aliases
VAELoss = StandardVAELoss
AdaTCDIPLoss = AdaTCDIPVAELoss

