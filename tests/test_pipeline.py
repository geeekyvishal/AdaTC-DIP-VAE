"""
Comprehensive Test Suite for Disentanglement VAE Research Pipeline.

Verifies:
1. `src/models/vae.py`: Forward pass, latent shapes (B, 10), reparameterization trick,
   output reconstruction shapes (B, 1, 64, 64) and (B, 3, 64, 64).
2. `src/losses/vae_losses.py`:
   - Standard VAE ELBO computes without NaN.
   - Beta-VAE loss scales KL properly.
   - Beta-TCVAE computes MWS Total Correlation without NaN and preserves non-negative TC.
   - AdaTC-DIP-VAE loss computes cosine schedule correctly at t=0, t=T_warm/2, and t=T_warm.
   - Covariance penalty computes off-diagonal and diagonal Frobenius norms correctly.
   - Autograd test: Gradients backward pass through all 4 loss functions without exploding or NaNs.
3. `src/metrics/disentanglement_metrics.py`:
   - MIG computes correctly on synthetic representations with known ground truth.
   - FactorVAE score computes correctly.
   - DCI Disentanglement computes correctly.
"""

import math
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Model imports
from src.models.vae import DisentangledVAE, ConvEncoder, ConvDecoder, VAE

# Loss imports
from src.losses.vae_losses import (
    compute_reconstruction_loss,
    compute_gaussian_kl,
    compute_mws_kl_decomposition,
    compute_covariance_penalty,
    get_cosine_annealed_beta,
    StandardVAELoss,
    VAELoss,
    BetaVAELoss,
    BetaTCVAELoss,
    AdaTCDIPVAELoss,
    AdaTCDIPLoss,
)

# Metric imports
from src.metrics.disentanglement_metrics import (
    compute_discrete_mutual_info,
    compute_discrete_entropy,
    compute_mig,
    compute_factorvae_score,
    compute_factorvae_score_from_representations,
    compute_dci,
    compute_reconstruction_mse,
)

# Dataset imports
from src.datasets.dataset_loader import DummyDisentanglementDataset


class TestVAEArchitecture(unittest.TestCase):
    """Verifies VAE forward pass, latent shapes, reparameterization, and reconstruction shapes."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 8
        self.latent_dim = 10

    def test_forward_pass_grayscale(self):
        """Test forward pass on single-channel 64x64 images (dSprites shape)."""
        model = VAE(in_channels=1, latent_dim=self.latent_dim)
        x = torch.rand(self.batch_size, 1, 64, 64)

        recon_x, mu, logvar, z = model(x)

        # Output reconstruction shape: (B, 1, 64, 64)
        self.assertEqual(recon_x.shape, (self.batch_size, 1, 64, 64))
        # Latent shapes: (B, 10)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # Check values are in valid image range [0, 1] due to Sigmoid
        self.assertTrue(torch.all(recon_x >= 0.0) and torch.all(recon_x <= 1.0))
        # Ensure no NaNs or Infs
        self.assertFalse(torch.isnan(recon_x).any())
        self.assertFalse(torch.isnan(mu).any())
        self.assertFalse(torch.isnan(logvar).any())
        self.assertFalse(torch.isnan(z).any())

    def test_forward_pass_rgb(self):
        """Test forward pass on 3-channel 64x64 images (3D Shapes shape)."""
        model = DisentangledVAE(in_channels=3, latent_dim=self.latent_dim)
        x = torch.rand(self.batch_size, 3, 64, 64)

        recon_x, mu, logvar, z = model(x)

        # Output reconstruction shape: (B, 3, 64, 64)
        self.assertEqual(recon_x.shape, (self.batch_size, 3, 64, 64))
        # Latent shapes: (B, 10)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # Check values are in valid range [0, 1]
        self.assertTrue(torch.all(recon_x >= 0.0) and torch.all(recon_x <= 1.0))
        self.assertFalse(torch.isnan(recon_x).any())

    def test_reparameterization_trick(self):
        """Verify reparameterization trick behavior in training and evaluation modes."""
        model = VAE(in_channels=1, latent_dim=self.latent_dim)
        mu = torch.full((1000, self.latent_dim), 2.0)
        # log(4.0) gives std = 2.0
        logvar = torch.full((1000, self.latent_dim), math.log(4.0))

        # 1. Training mode: stochasticity check
        model.train()
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)

        # Stochastic samples must differ
        self.assertFalse(torch.allclose(z1, z2))
        # Empirical mean and std should match mu=2.0 and std=2.0
        sample_mean = z1.mean().item()
        sample_std = z1.std().item()
        self.assertAlmostEqual(sample_mean, 2.0, delta=0.2)
        self.assertAlmostEqual(sample_std, 2.0, delta=0.2)

        # 2. Evaluation mode: deterministic check (returns mean)
        model.eval()
        z_eval = model.reparameterize(mu, logvar)
        self.assertTrue(torch.allclose(z_eval, mu))

    def test_encoder_and_decoder_methods(self):
        """Test individual encode and decode methods."""
        model = VAE(in_channels=1, latent_dim=self.latent_dim)
        x = torch.rand(4, 1, 64, 64)

        mu, logvar = model.encode(x)
        self.assertEqual(mu.shape, (4, self.latent_dim))
        self.assertEqual(logvar.shape, (4, self.latent_dim))

        z = torch.randn(4, self.latent_dim)
        recon_x = model.decode(z)
        self.assertEqual(recon_x.shape, (4, 1, 64, 64))

        # Sampling from prior
        sampled_imgs = model.sample(num_samples=5)
        self.assertEqual(sampled_imgs.shape, (5, 1, 64, 64))


class TestVAELosses(unittest.TestCase):
    """Verifies all loss functions, MWS TC estimation, cosine annealing, and autograd."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 16
        self.latent_dim = 10
        self.dataset_size = 10000

        self.x = torch.rand(self.batch_size, 1, 64, 64)
        self.recon_x = torch.clamp(self.x + 0.05 * torch.randn_like(self.x), 0.001, 0.999)
        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.logvar = torch.randn(self.batch_size, self.latent_dim)
        self.z = self.mu + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mu)

    def test_standard_vae_elbo(self):
        """Verify standard VAE ELBO computes without NaN."""
        loss_fn = StandardVAELoss(loss_type="bce", reduction="sum_per_sample")
        out = loss_fn(self.recon_x, self.x, self.mu, self.logvar)

        self.assertIn("loss", out)
        self.assertIn("recon_loss", out)
        self.assertIn("kl_loss", out)

        loss_val = out["loss"].item()
        recon_val = out["recon_loss"].item()
        kl_val = out["kl_loss"].item()

        self.assertFalse(math.isnan(loss_val), "Standard VAE loss is NaN")
        self.assertFalse(math.isinf(loss_val), "Standard VAE loss is Inf")
        self.assertGreater(recon_val, 0.0)
        self.assertGreaterEqual(kl_val, 0.0)
        # ELBO = recon_loss + kl_loss
        self.assertAlmostEqual(loss_val, recon_val + kl_val, places=3)

        # Also verify VAELoss alias
        alias_loss_fn = VAELoss()
        alias_out = alias_loss_fn(self.recon_x, self.x, self.mu, self.logvar)
        self.assertAlmostEqual(alias_out["loss"].item(), loss_val, places=3)

    def test_beta_vae_loss_scaling(self):
        """Verify Beta-VAE loss properly scales KL by beta."""
        loss_fn_1 = BetaVAELoss(beta=1.0)
        loss_fn_4 = BetaVAELoss(beta=4.0)

        out_1 = loss_fn_1(self.recon_x, self.x, self.mu, self.logvar)
        out_4 = loss_fn_4(self.recon_x, self.x, self.mu, self.logvar)

        recon_1 = out_1["recon_loss"].item()
        recon_4 = out_4["recon_loss"].item()
        kl_1 = out_1["kl_loss"].item()
        kl_4 = out_4["kl_loss"].item()

        # Reconstruction and KL are identical before scaling
        self.assertAlmostEqual(recon_1, recon_4, places=3)
        self.assertAlmostEqual(kl_1, kl_4, places=3)

        # Loss 4 must equal recon + 4.0 * kl
        expected_loss_4 = recon_4 + 4.0 * kl_4
        self.assertAlmostEqual(out_4["loss"].item(), expected_loss_4, places=3)

    def test_beta_tcvae_mws_and_non_negative_tc(self):
        """Verify Beta-TCVAE computes MWS Total Correlation without NaN and preserves non-negative TC."""
        loss_fn = BetaTCVAELoss(
            beta=6.0,
            alpha=1.0,
            gamma=1.0,
            dataset_size=self.dataset_size,
        )

        out = loss_fn(
            self.recon_x,
            self.x,
            self.mu,
            self.logvar,
            self.z,
            dataset_size=self.dataset_size,
        )

        self.assertFalse(torch.isnan(out["loss"]).any())
        self.assertFalse(torch.isnan(out["tc_loss"]).any())
        self.assertFalse(torch.isnan(out["mi_loss"]).any())
        self.assertFalse(torch.isnan(out["dw_kl_loss"]).any())

        # TC must be non-negative (preserved against finite-sample bias)
        self.assertGreaterEqual(out["tc_loss"].item(), 0.0)

        # Verify beta scaling on Total Correlation term
        loss_fn_beta10 = BetaTCVAELoss(
            beta=10.0,
            alpha=1.0,
            gamma=1.0,
            dataset_size=self.dataset_size,
        )
        out_beta10 = loss_fn_beta10(
            self.recon_x,
            self.x,
            self.mu,
            self.logvar,
            self.z,
            dataset_size=self.dataset_size,
        )
        # Difference in loss must equal (10.0 - 6.0) * tc_loss
        expected_diff = (10.0 - 6.0) * out["tc_loss"].item()
        actual_diff = out_beta10["loss"].item() - out["loss"].item()
        self.assertAlmostEqual(actual_diff, expected_diff, places=3)

    def test_adatc_dip_cosine_schedule(self):
        """Verify AdaTC-DIP-VAE loss computes cosine schedule correctly at t=0, t=T_warm/2, and t=T_warm."""
        beta_target = 6.0
        warm_steps = 10000
        loss_fn = AdaTCDIPVAELoss(
            beta_target=beta_target,
            warm_steps=warm_steps,
            lambda_od=10.0,
            lambda_d=1.0,
        )

        # 1. At t = 0: beta(0) must be 1.0 (starts at standard ELBO)
        beta_0 = loss_fn.get_current_beta(0)
        self.assertAlmostEqual(beta_0, 1.0, places=6)

        # 2. At t = T_warm / 2: beta(T_warm / 2) = 1 + (beta_target - 1)/2 = 3.5
        t_half = warm_steps // 2
        beta_half = loss_fn.get_current_beta(t_half)
        expected_half = 1.0 + (beta_target - 1.0) / 2.0
        self.assertAlmostEqual(beta_half, expected_half, places=6)
        self.assertAlmostEqual(beta_half, 3.5, places=6)

        # 3. At t = T_warm: beta(T_warm) must equal beta_target = 6.0
        beta_warm = loss_fn.get_current_beta(warm_steps)
        self.assertAlmostEqual(beta_warm, beta_target, places=6)
        self.assertAlmostEqual(beta_warm, 6.0, places=6)

        # 4. Beyond T_warm: stays at beta_target
        beta_post = loss_fn.get_current_beta(warm_steps + 5000)
        self.assertAlmostEqual(beta_post, beta_target, places=6)

        # Standalone function verification
        self.assertAlmostEqual(get_cosine_annealed_beta(0, 8.0, 20000), 1.0, places=6)
        self.assertAlmostEqual(get_cosine_annealed_beta(10000, 8.0, 20000), 4.5, places=6)
        self.assertAlmostEqual(get_cosine_annealed_beta(20000, 8.0, 20000), 8.0, places=6)

    def test_covariance_penalty_frobenius_norms(self):
        """Verify covariance penalty computes off-diagonal and diagonal Frobenius norms correctly."""
        # Case A: Synthetic uncorrelated means with unit variance
        # Construct U such that sample covariance Cov = I
        # For d=2, B=4:
        # Centered matrix: [[1, 0], [-1, 0], [0, 1], [0, -1]]
        # (1/(4-1)) U^T U = (1/3) * [[2, 0], [0, 2]]
        # To get Cov = I, scale by sqrt(3/2):
        s = math.sqrt(3.0 / 2.0)
        mu_uncorr = torch.tensor(
            [[s, 0.0], [-s, 0.0], [0.0, s], [0.0, -s]], dtype=torch.float32
        )
        # Covariance of mu_uncorr is exactly [[1, 0], [0, 1]]
        total_p, od_p, d_p = compute_covariance_penalty(mu_uncorr, lambda_od=10.0, lambda_d=1.0)

        self.assertAlmostEqual(od_p.item(), 0.0, places=5)
        self.assertAlmostEqual(d_p.item(), 0.0, places=5)
        self.assertAlmostEqual(total_p.item(), 0.0, places=5)

        # Case B: Known correlated matrix
        # Let B=3, d=2:
        # mu = [[1, 2], [2, 1], [3, 3]]
        # Mean = [2, 2]
        # Centered: [[-1, 0], [0, -1], [1, 1]]
        # U^T U = [[ (-1)^2 + 0 + 1^2,  (-1)*0 + 0*(-1) + 1*1 ],
        #          [ 1*1,               0^2 + (-1)^2 + 1^2    ]]
        #       = [[2, 1], [1, 2]]
        # Cov = (1 / (3 - 1)) * U^T U = [[1.0, 0.5], [0.5, 1.0]]
        # Off-diagonal: [[0, 0.5], [0.5, 0]]
        # Off-diag squared Frobenius norm: 0.5^2 + 0.5^2 = 0.50
        # Diagonal: [1.0, 1.0] -> (1-1)^2 + (1-1)^2 = 0.00
        mu_known = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], dtype=torch.float32)
        total_pen, od_pen, d_pen = compute_covariance_penalty(
            mu_known, lambda_od=10.0, lambda_d=2.0
        )

        self.assertAlmostEqual(od_pen.item(), 0.50, places=5)
        self.assertAlmostEqual(d_pen.item(), 0.00, places=5)
        self.assertAlmostEqual(total_pen.item(), 10.0 * 0.50 + 2.0 * 0.00, places=5)

    def test_autograd_backward_all_losses(self):
        """Autograd test: Gradients backward pass through all 4 loss functions without exploding or NaNs."""
        loss_modules = [
            ("Standard VAE", StandardVAELoss(loss_type="bce"), lambda loss_fn, out, x, step: loss_fn(out.x_recon, x, out.mu, out.logvar)),
            ("Beta-VAE", BetaVAELoss(beta=4.0, loss_type="bce"), lambda loss_fn, out, x, step: loss_fn(out.x_recon, x, out.mu, out.logvar)),
            ("Beta-TCVAE", BetaTCVAELoss(beta=6.0, dataset_size=1000), lambda loss_fn, out, x, step: loss_fn(out.x_recon, x, out.mu, out.logvar, out.z, dataset_size=1000)),
            ("AdaTC-DIP-VAE", AdaTCDIPVAELoss(beta_target=6.0, warm_steps=1000, lambda_od=10.0, lambda_d=1.0, dataset_size=1000), lambda loss_fn, out, x, step: loss_fn(out.x_recon, x, out.mu, out.logvar, out.z, current_step=step, dataset_size=1000)),
        ]

        for name, loss_fn, call_loss in loss_modules:
            with self.subTest(loss_name=name):
                # Fresh model and optimizer
                model = VAE(in_channels=1, latent_dim=self.latent_dim)
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                optimizer.zero_grad()

                x = torch.rand(self.batch_size, 1, 64, 64)
                recon_x, mu, logvar, z = model(x)
                # Encapsulate output
                class ModelOut:
                    pass
                out = ModelOut()
                out.x_recon, out.mu, out.logvar, out.z = recon_x, mu, logvar, z

                loss_dict = call_loss(loss_fn, out, x, 500)
                loss_val = loss_dict["loss"]

                self.assertFalse(torch.isnan(loss_val), f"{name} loss is NaN")
                self.assertFalse(torch.isinf(loss_val), f"{name} loss is Inf")

                loss_val.backward()

                # Check gradients for every single parameter
                for p_name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, f"Gradient is None for {p_name} in {name}")
                    self.assertFalse(
                        torch.isnan(param.grad).any(),
                        f"Gradient has NaN for {p_name} in {name}",
                    )
                    self.assertFalse(
                        torch.isinf(param.grad).any(),
                        f"Gradient has Inf for {p_name} in {name}",
                    )
                    # Gradients must not explode
                    grad_norm = param.grad.norm().item()
                    self.assertLess(
                        grad_norm, 1e6, f"Gradient exploding ({grad_norm}) for {p_name} in {name}"
                    )

                optimizer.step()


class TestDisentanglementMetrics(unittest.TestCase):
    """Verifies MIG, FactorVAE score, and DCI on synthetic representations with known ground truth."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_mig_synthetic_ground_truth(self):
        """MIG computes correctly on synthetic representations with known ground truth."""
        n_samples = 4000
        latent_dim = 10
        num_factors = 4

        # 4 ground-truth discrete factors with cardinalities [3, 4, 5, 4]
        factor_cardinalities = [3, 4, 5, 4]
        factors = np.zeros((n_samples, num_factors), dtype=np.int64)
        for k, card in enumerate(factor_cardinalities):
            factors[:, k] = np.random.randint(0, card, size=n_samples)

        # Case 1: Perfectly Disentangled Representations
        # Latent dim k is an exact (noisy) copy of factor k, remaining 6 dims are pure noise
        mus_disentangled = np.random.randn(n_samples, latent_dim).astype(np.float32)
        for k in range(num_factors):
            # Scale factor to continuous range with very low noise
            mus_disentangled[:, k] = factors[:, k].astype(np.float32) * 2.0 + np.random.normal(
                0, 0.05, size=n_samples
            )

        mig_res = compute_mig(mus_disentangled, factors, num_bins=20)
        self.assertIn("mig", mig_res)
        self.assertIn("per_factor_mig", mig_res)
        self.assertIn("mi_matrix", mig_res)

        # Disentangled MIG should be very high (>= 0.80)
        self.assertGreater(
            mig_res["mig"], 0.80, f"Expected high MIG for disentangled rep, got {mig_res['mig']}"
        )

        # Case 2: Entangled Representations
        # Coordinate 0 and 1 both encode linear combinations of factor 0 and factor 1
        mus_entangled = np.random.randn(n_samples, latent_dim).astype(np.float32)
        mus_entangled[:, 0] = factors[:, 0] + factors[:, 1]
        mus_entangled[:, 1] = factors[:, 0] - factors[:, 1]
        mig_entangled = compute_mig(mus_entangled, factors, num_bins=20)

        # Factor 0 and 1 have two competitive latent coordinates -> gap collapses
        self.assertLess(
            mig_entangled["per_factor_mig"][0],
            mig_res["per_factor_mig"][0],
            "Entangled factor 0 should have lower MIG than disentangled",
        )

    def test_factorvae_score_synthetic(self):
        """FactorVAE score computes correctly on synthetic representations."""
        n_samples = 3000
        latent_dim = 10
        num_factors = 4

        factors = np.zeros((n_samples, num_factors), dtype=np.int64)
        for k in range(num_factors):
            factors[:, k] = np.random.randint(0, 4, size=n_samples)

        # Perfectly disentangled representations: dimension k encodes factor k
        representations = np.random.randn(n_samples, latent_dim).astype(np.float32)
        for k in range(num_factors):
            representations[:, k] = factors[:, k].astype(np.float32) * 3.0

        res = compute_factorvae_score_from_representations(
            representations=representations,
            factors=factors,
            num_train_votes=400,
            num_eval_votes=200,
            batch_size=32,
            seed=42,
        )

        self.assertIn("factorvae_score", res)
        self.assertIn("active_dims", res)
        # Perfectly disentangled representation should achieve near-perfect FactorVAE score (>= 0.90)
        self.assertGreaterEqual(
            res["factorvae_score"],
            0.90,
            f"Expected FactorVAE score >= 0.90, got {res['factorvae_score']}",
        )

        # Also test with DummyDisentanglementDataset and model
        dataset = DummyDisentanglementDataset(num_samples=256, in_channels=1)
        model = VAE(in_channels=1, latent_dim=10)
        model.eval()
        device = torch.device("cpu")

        fvae_dataset_res = compute_factorvae_score(
            model=model,
            dataset=dataset,
            device=device,
            num_train_votes=50,
            num_eval_votes=30,
            batch_size=16,
            seed=42,
        )
        self.assertIn("factorvae_score", fvae_dataset_res)
        self.assertGreaterEqual(fvae_dataset_res["factorvae_score"], 0.0)
        self.assertLessEqual(fvae_dataset_res["factorvae_score"], 1.0)

    def test_dci_disentanglement_synthetic(self):
        """DCI Disentanglement computes correctly on synthetic representations."""
        n_samples = 2000
        latent_dim = 8
        num_factors = 3

        factors = np.zeros((n_samples, num_factors), dtype=np.int64)
        for k in range(num_factors):
            factors[:, k] = np.random.randint(0, 3, size=n_samples)

        # Case 1: Perfectly Disentangled
        # Dimension k predicts factor k with 100% precision; other dims are Gaussian noise
        mus = np.random.randn(n_samples, latent_dim).astype(np.float32)
        for k in range(num_factors):
            mus[:, k] = factors[:, k].astype(np.float32) * 4.0 + np.random.normal(0, 0.01, size=n_samples)

        split = int(0.7 * n_samples)
        train_mus, test_mus = mus[:split], mus[split:]
        train_factors, test_factors = factors[:split], factors[split:]

        dci_res = compute_dci(
            train_mus=train_mus,
            train_factors=train_factors,
            test_mus=test_mus,
            test_factors=test_factors,
            classifier_type="gradient_boosting",
        )

        self.assertIn("disentanglement", dci_res)
        self.assertIn("completeness", dci_res)
        self.assertIn("informativeness", dci_res)

        # For perfectly disentangled latents, DCI Disentanglement must be very high (>= 0.85)
        self.assertGreater(
            dci_res["disentanglement"],
            0.85,
            f"Expected DCI Disentanglement >= 0.85, got {dci_res['disentanglement']}",
        )
        self.assertGreater(
            dci_res["completeness"],
            0.85,
            f"Expected DCI Completeness >= 0.85, got {dci_res['completeness']}",
        )
        self.assertGreater(
            dci_res["informativeness"],
            0.85,
            f"Expected DCI Informativeness >= 0.85, got {dci_res['informativeness']}",
        )


class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end integration test training 1 epoch on DummyDisentanglementDataset."""

    def test_training_step_integration(self):
        """Test full training step with AdaTC-DIP-VAE on DummyDisentanglementDataset."""
        dataset = DummyDisentanglementDataset(num_samples=32, in_channels=1, image_size=64)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = VAE(in_channels=1, latent_dim=10)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = AdaTCDIPVAELoss(
            beta_target=6.0,
            warm_steps=100,
            lambda_od=10.0,
            lambda_d=1.0,
            dataset_size=len(dataset),
        )

        initial_loss = None
        final_loss = None

        for step, (imgs, _) in enumerate(loader):
            optimizer.zero_grad()
            recon_x, mu, logvar, z = model(imgs)
            loss_dict = loss_fn(
                recon_x=recon_x,
                x=imgs,
                mu=mu,
                logvar=logvar,
                z=z,
                current_step=step,
                dataset_size=len(dataset),
            )
            loss = loss_dict["loss"]
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertFalse(math.isnan(final_loss))


if __name__ == "__main__":
    unittest.main()
