"""
Standard Disentanglement Variational Autoencoder (VAE) Architecture.

Faithfully implements the standard 4-convolutional layer architecture established by:
- Burgess et al., "Understanding Disentangling in beta-VAE" (2018)
- Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning
  of Disentangled Representations" (ICML 2019, Table 2)

Supports single-channel binary datasets (dSprites) and 3-channel RGB datasets (3D Shapes)
with configurable latent dimensions (default: 10).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """
    4-Conv layer encoder for 64x64 images.

    Architecture:
      Input: (B, in_channels, 64, 64)
      Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1) -> ReLU  -> (B, 32, 32, 32)
      Conv2d(32, 32, kernel_size=4, stride=2, padding=1)          -> ReLU  -> (B, 32, 16, 16)
      Conv2d(32, 64, kernel_size=4, stride=2, padding=1)          -> ReLU  -> (B, 64, 8, 8)
      Conv2d(64, 64, kernel_size=4, stride=2, padding=1)          -> ReLU  -> (B, 64, 4, 4)
      Flatten                                                               -> (B, 1024)
      Linear(1024, 256)                                            -> ReLU  -> (B, 256)
      Linear(256, 2 * latent_dim)                                           -> (B, 2 * latent_dim)
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * latent_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent Gaussian parameters (mu, logvar).

        Args:
            x: Input images of shape (B, in_channels, 64, 64)

        Returns:
            mu: Mean vectors of shape (B, latent_dim)
            logvar: Log-variance vectors of shape (B, latent_dim)
        """
        h = self.conv_layers(x)
        h = torch.flatten(h, start_dim=1)
        params = self.fc_layers(h)
        mu = params[:, : self.latent_dim]
        logvar = params[:, self.latent_dim :]
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    4-ConvTranspose layer decoder for reconstructing 64x64 images.

    Architecture:
      Input: (B, latent_dim)
      Linear(latent_dim, 256)                                      -> ReLU  -> (B, 256)
      Linear(256, 64 * 4 * 4)                                      -> ReLU  -> (B, 1024)
      Reshape                                                               -> (B, 64, 4, 4)
      ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  -> ReLU  -> (B, 64, 8, 8)
      ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  -> ReLU  -> (B, 32, 16, 16)
      ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  -> ReLU  -> (B, 32, 32, 32)
      ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1) -> (B, in_channels, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 10,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.output_activation = output_activation

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to reconstructed images.

        Args:
            z: Latent vectors of shape (B, latent_dim)

        Returns:
            recon_x: Reconstructed images of shape (B, in_channels, 64, 64).
                     If output_activation='sigmoid', values are in [0, 1].
                     If output_activation='none', raw unconstrained logits are returned.
        """
        h = self.fc_layers(z)
        h = h.view(-1, 64, 4, 4)
        logits = self.deconv_layers(h)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        elif self.output_activation == "none":
            return logits
        else:
            raise ValueError(f"Unsupported output_activation: {self.output_activation}")


class DisentangledVAE(nn.Module):
    """
    Modular Variational Autoencoder (VAE) for Disentanglement Benchmarks.

    Conforms to the standard Burgess / Locatello architecture:
    - 4 Conv layers with kernel 4, stride 2, padding 1
    - 2 FC layers in encoder & decoder (256 hidden units)
    - 4 ConvTranspose layers in decoder
    - Latent dimension d = 10 (configurable)
    - Fully supports both grayscale (1-channel) and RGB (3-channel) inputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 10,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.output_activation = output_activation

        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            output_activation=output_activation,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard Gaussian reparameterization trick:
        z = mu + eps * exp(0.5 * logvar), with eps ~ N(0, I).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input images x to mean and log-variance."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors z to reconstructed images."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: Input images of shape (B, in_channels, 64, 64)

        Returns:
            recon_x: Reconstructed images of shape (B, in_channels, 64, 64)
            mu: Posterior mean of shape (B, latent_dim)
            logvar: Posterior log-variance of shape (B, latent_dim)
            z: Sampled latent code of shape (B, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate random samples from standard normal prior p(z) = N(0, I).
        """
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


# Convenience alias for benchmark test suites
VAE = DisentangledVAE



# Standard alias
VAE = DisentangledVAE

