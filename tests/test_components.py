"""
Comprehensive Unit Tests for VAE Disentanglement Framework.

Tests:
1. Models: DisentangledVAE, ConvEncoder, ConvDecoder (in_channels=1, 3; latent_dim=10)
2. Losses: Standard VAE, Beta-VAE, Beta-TCVAE (MWS), AdaTC-DIP-VAE
3. Datasets: DummyDisentanglementDataset, sample_factor_fixed, DataLoader
4. Metrics: MIG, FactorVAE score, DCI Disentanglement & Completeness, Reconstruction MSE
5. End-to-End training and evaluation smoke test.
"""

import unittest
import math
import torch
import numpy as np

from src.models.vae import DisentangledVAE, ConvEncoder, ConvDecoder
from src.losses.vae_losses import (
    compute_reconstruction_loss,
    compute_gaussian_kl,
    compute_mws_kl_decomposition,
    compute_covariance_penalty,
    get_cosine_annealed_beta,
    StandardVAELoss,
    BetaVAELoss,
    BetaTCVAELoss,
    AdaTCDIPVAELoss,
)
from src.datasets.dataset_loader import (
    DummyDisentanglementDataset,
    get_dataloader,
)
from src.metrics.disentanglement_metrics import (
    compute_mig,
    compute_dci,
    compute_factorvae_score,
    compute_reconstruction_mse,
    evaluate_all_metrics,
)


class TestModels(unittest.TestCase):
    def test_encoder_decoder_shapes_1channel(self):
        B, C, H, W = 4, 1, 64, 64
        d = 10
        model = DisentangledVAE(in_channels=C, latent_dim=d)
        x = torch.rand(B, C, H, W)

        recon_x, mu, logvar, z = model(x)

        assert recon_x.shape == (B, C, H, W)
        assert mu.shape == (B, d)
        assert logvar.shape == (B, d)
        assert z.shape == (B, d)
        assert (recon_x >= 0.0).all() and (recon_x <= 1.0).all()

    def test_encoder_decoder_shapes_3channel(self):
        B, C, H, W = 4, 3, 64, 64
        d = 12
        model = DisentangledVAE(in_channels=C, latent_dim=d)
        x = torch.rand(B, C, H, W)

        recon_x, mu, logvar, z = model(x)

        assert recon_x.shape == (B, C, H, W)
        assert mu.shape == (B, d)
        assert logvar.shape == (B, d)
        assert z.shape == (B, d)

    def test_sample(self):
        model = DisentangledVAE(in_channels=1, latent_dim=10)
        samples = model.sample(num_samples=8)
        assert samples.shape == (8, 1, 64, 64)


class TestLosses(unittest.TestCase):
    def test_standard_vae_loss(self):
        B, C, H, W = 4, 1, 64, 64
        d = 10
        x = torch.rand(B, C, H, W)
        recon_x = torch.rand(B, C, H, W)
        mu = torch.randn(B, d)
        logvar = torch.zeros(B, d)

        loss_fn = StandardVAELoss(loss_type="bce")
        out = loss_fn(recon_x, x, mu, logvar)

        assert "loss" in out and "recon_loss" in out and "kl_loss" in out
        assert out["loss"] == out["recon_loss"] + out["kl_loss"]
        assert out["loss"].item() > 0.0

    def test_beta_vae_loss(self):
        B, d = 4, 10
        x = torch.rand(B, 1, 64, 64)
        recon_x = torch.rand(B, 1, 64, 64)
        mu = torch.randn(B, d)
        logvar = torch.zeros(B, d)

        loss_fn = BetaVAELoss(beta=4.0)
        out = loss_fn(recon_x, x, mu, logvar)

        expected = out["recon_loss"] + 4.0 * out["kl_loss"]
        assert torch.isclose(out["loss"], expected)

    def test_beta_tcvae_loss_mws(self):
        B, d = 8, 10
        x = torch.rand(B, 1, 64, 64)
        recon_x = torch.rand(B, 1, 64, 64)
        mu = torch.randn(B, d, requires_grad=True)
        logvar = torch.zeros(B, d, requires_grad=True)
        z = mu + torch.randn_like(mu)

        loss_fn = BetaTCVAELoss(beta=6.0, dataset_size=1000)
        out = loss_fn(recon_x, x, mu, logvar, z)

        assert "mi_loss" in out and "tc_loss" in out and "dw_kl_loss" in out
        # Check backward pass
        out["loss"].backward()
        assert mu.grad is not None and not torch.isnan(mu.grad).any()

    def test_cosine_annealing_schedule(self):
        warm_steps = 1000
        beta_target = 6.0

        # Step 0: should be 1.0
        assert math.isclose(get_cosine_annealed_beta(0, beta_target, warm_steps), 1.0, abs_tol=1e-5)
        # Midpoint: should be 1 + (5/2)*(1 - cos(pi/2)) = 1 + 2.5 = 3.5
        assert math.isclose(get_cosine_annealed_beta(500, beta_target, warm_steps), 3.5, abs_tol=1e-3)
        # End of warm-up: should reach beta_target = 6.0
        assert math.isclose(get_cosine_annealed_beta(1000, beta_target, warm_steps), 6.0, abs_tol=1e-5)
        # Past warm-up: should stay at beta_target = 6.0
        assert math.isclose(get_cosine_annealed_beta(2000, beta_target, warm_steps), 6.0, abs_tol=1e-5)

    def test_adatc_dip_vae_loss(self):
        B, d = 16, 10
        x = torch.rand(B, 1, 64, 64)
        recon_x = torch.rand(B, 1, 64, 64)
        mu = torch.randn(B, d, requires_grad=True)
        logvar = torch.zeros(B, d, requires_grad=True)
        z = mu + torch.randn_like(mu)

        loss_fn = AdaTCDIPVAELoss(
            beta_target=6.0,
            warm_steps=1000,
            lambda_od=10.0,
            lambda_d=1.0,
            dataset_size=1000,
        )

        out = loss_fn(recon_x, x, mu, logvar, z, current_step=0)
        assert out["beta_t"].item() == 1.0
        assert "cov_loss" in out and "off_diag_cov" in out

        out["loss"].backward()
        assert mu.grad is not None and not torch.isnan(mu.grad).any()


class TestDatasets(unittest.TestCase):
    def test_dummy_dataset(self):
        dataset = DummyDisentanglementDataset(num_samples=64, in_channels=1, image_size=64)
        assert len(dataset) == 64

        img, labels = dataset[0]
        assert img.shape == (1, 64, 64)
        assert labels.shape == (5,)

        # Test sample_factor_fixed
        fixed_imgs, fixed_labels = dataset.sample_factor_fixed(factor_idx=0, fixed_value=1, num_samples=16)
        assert fixed_imgs.shape == (16, 1, 64, 64)
        assert (fixed_labels[:, 0] == 1).all()

    def test_dataloader(self):
        dataset = DummyDisentanglementDataset(num_samples=32, in_channels=3)
        loader = get_dataloader(dataset, batch_size=8, shuffle=True)
        batch_imgs, batch_labels = next(iter(loader))
        assert batch_imgs.shape == (8, 3, 64, 64)
        assert batch_labels.shape == (8, 5)


class TestMetrics(unittest.TestCase):
    def test_mig(self):
        # Create perfectly disentangled representation: z_0 aligns with factor 0, z_1 with factor 1
        N = 200
        factors = np.zeros((N, 2), dtype=np.int64)
        factors[:, 0] = np.random.randint(0, 4, size=N)
        factors[:, 1] = np.random.randint(0, 4, size=N)

        mus = np.zeros((N, 4), dtype=np.float32)
        mus[:, 0] = factors[:, 0] * 2.0 + np.random.normal(0, 0.05, size=N)
        mus[:, 1] = factors[:, 1] * 2.0 + np.random.normal(0, 0.05, size=N)
        mus[:, 2] = np.random.normal(0, 1.0, size=N)
        mus[:, 3] = np.random.normal(0, 1.0, size=N)

        res = compute_mig(mus, factors, num_bins=10)
        assert "mig" in res
        assert 0.0 <= res["mig"] <= 1.0
        # For perfectly separated signals, MIG should be comfortably positive
        assert res["mig"] > 0.4

    def test_dci(self):
        N = 300
        train_factors = np.random.randint(0, 3, size=(N, 2))
        test_factors = np.random.randint(0, 3, size=(100, 2))

        train_mus = np.zeros((N, 4), dtype=np.float32)
        train_mus[:, 0] = train_factors[:, 0] * 3.0 + np.random.normal(0, 0.1, size=N)
        train_mus[:, 1] = train_factors[:, 1] * 3.0 + np.random.normal(0, 0.1, size=N)

        test_mus = np.zeros((100, 4), dtype=np.float32)
        test_mus[:, 0] = test_factors[:, 0] * 3.0 + np.random.normal(0, 0.1, size=100)
        test_mus[:, 1] = test_factors[:, 1] * 3.0 + np.random.normal(0, 0.1, size=100)

        res = compute_dci(train_mus, train_factors, test_mus, test_factors)
        assert "disentanglement" in res
        assert "completeness" in res
        assert "informativeness" in res
        assert 0.0 <= res["disentanglement"] <= 1.0
        assert 0.0 <= res["completeness"] <= 1.0
        assert 0.0 <= res["informativeness"] <= 1.0

    def test_factorvae_score(self):
        model = DisentangledVAE(in_channels=1, latent_dim=10)
        dataset = DummyDisentanglementDataset(num_samples=128, in_channels=1)
        device = torch.device("cpu")

        res = compute_factorvae_score(
            model=model,
            dataset=dataset,
            device=device,
            num_train_votes=20,
            num_eval_votes=10,
            batch_size=16,
        )
        assert "factorvae_score" in res
        assert 0.0 <= res["factorvae_score"] <= 1.0

    def test_reconstruction_mse(self):
        model = DisentangledVAE(in_channels=1, latent_dim=10)
        dataset = DummyDisentanglementDataset(num_samples=32, in_channels=1)
        loader = get_dataloader(dataset, batch_size=16)
        device = torch.device("cpu")

        mse = compute_reconstruction_mse(model, loader, device=device, max_samples=32)
        assert isinstance(mse, float)
        assert mse >= 0.0

    def test_evaluate_all_metrics(self):
        model = DisentangledVAE(in_channels=1, latent_dim=10)
        dataset = DummyDisentanglementDataset(num_samples=64, in_channels=1)
        device = torch.device("cpu")

        report = evaluate_all_metrics(
            model=model,
            dataset=dataset,
            device=device,
            num_eval_samples=64,
            batch_size=16,
            include_factorvae=True,
            num_factorvae_train_votes=30,
            num_factorvae_eval_votes=15,
        )
        for key in ["mig", "dci_disentanglement", "dci_completeness", "dci_informativeness", "reconstruction_mse"]:
            assert key in report


class TestTrainingIntegration:
    """Smoke test ensuring all models and losses train for several steps seamlessly."""

    def test_train_steps(self):
        dataset = DummyDisentanglementDataset(num_samples=64, in_channels=1)
        loader = get_dataloader(dataset, batch_size=16, shuffle=True)
        device = torch.device("cpu")

        losses = [
            ("StandardVAE", StandardVAELoss(loss_type="bce")),
            ("BetaVAE", BetaVAELoss(beta=4.0, loss_type="bce")),
            ("BetaTCVAE", BetaTCVAELoss(beta=6.0, dataset_size=64, loss_type="bce")),
            ("AdaTC_DIP_VAE", AdaTCDIPVAELoss(beta_target=6.0, warm_steps=100, lambda_od=10.0, dataset_size=64, loss_type="bce")),
        ]

        for name, loss_fn in losses:
            model = DisentangledVAE(in_channels=1, latent_dim=10).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()

            initial_loss = None
            final_loss = None

            for step, (x, _) in enumerate(loader):
                x = x.to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar, z = model(x)

                if isinstance(loss_fn, AdaTCDIPVAELoss):
                    out = loss_fn(recon_x, x, mu, logvar, z, current_step=step)
                elif isinstance(loss_fn, BetaTCVAELoss):
                    out = loss_fn(recon_x, x, mu, logvar, z)
                else:
                    out = loss_fn(recon_x, x, mu, logvar)

                loss = out["loss"]
                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

                loss.backward()
                optimizer.step()

            assert initial_loss is not None
            assert not np.isnan(final_loss)
            assert not np.isinf(final_loss)



if __name__ == "__main__":
    unittest.main()

