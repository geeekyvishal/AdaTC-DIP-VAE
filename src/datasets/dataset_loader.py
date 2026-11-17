"""
Dataset Loaders and Generators for Disentanglement Benchmarking.

Provides:
1. DSpritesDataset: Loads and serves dSprites (737,280 binary 64x64 images, 5 ground-truth factors).
2. Shapes3DDataset: Loads and serves 3D Shapes (480,000 RGB 64x64 images, 6 ground-truth factors).
3. DummyDisentanglementDataset: Fast, in-memory synthetic generator for immediate unit testing
   without requiring 1GB dataset downloads.
"""

import os
import urllib.request
from typing import Tuple, Optional, List, Dict, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
except ImportError:
    h5py = None


DSPYRITES_URL = "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
SHAPES3D_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"


class DSpritesDataset(Dataset):
    """
    dSprites dataset loader.

    Reference:
      Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained
      Variational Framework" (ICLR 2017)

    Ground-truth factors (5 informative factors; color is constant):
      1. Shape (3 values: square=0, ellipse=1, heart=2)
      2. Scale (6 values: 0.5 to 1.0)
      3. Orientation (40 values: 0 to 2*pi)
      4. Position X (32 values: 0 to 1)
      5. Position Y (32 values: 0 to 1)
    """

    FACTOR_NAMES = ["color", "shape", "scale", "orientation", "pos_x", "pos_y"]
    INFORMATIVE_FACTOR_NAMES = ["shape", "scale", "orientation", "pos_x", "pos_y"]
    FACTOR_SIZES = [1, 3, 6, 40, 32, 32]
    INFORMATIVE_FACTOR_SIZES = [3, 6, 40, 32, 32]

    def __init__(
        self,
        root_dir: str = "./data/dsprites",
        filename: str = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        download: bool = False,
        use_discrete_factors: bool = True,
        informative_only: bool = True,
        transform=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.file_path = os.path.join(root_dir, filename)
        self.use_discrete_factors = use_discrete_factors
        self.informative_only = informative_only
        self.transform = transform

        if download and not os.path.exists(self.file_path):
            self._download()

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"dSprites file not found at {self.file_path}. "
                f"Pass download=True or download manually from {DSPYRITES_URL}"
            )

        data = np.load(self.file_path, allow_pickle=True, encoding="latin1")
        # Images: shape (737280, 64, 64), uint8 {0, 1}
        self.images = data["imgs"]
        # Latent classes: (737280, 6), integer indices
        self.latents_classes = data["latents_classes"]
        # Latent values: (737280, 6), continuous values
        self.latents_values = data["latents_values"]

        # Precompute basis for fast coordinate indexing
        self.factor_bases = np.array([3 * 6 * 40 * 32 * 32, 6 * 40 * 32 * 32, 40 * 32 * 32, 32 * 32, 32, 1])

    def _download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        print(f"Downloading dSprites dataset from {DSPYRITES_URL} to {self.file_path}...")
        urllib.request.urlretrieve(DSPYRITES_URL, self.file_path)
        print("Download complete.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, 64, 64)

        if self.use_discrete_factors:
            labels = self.latents_classes[idx]
        else:
            labels = self.latents_values[idx]

        if self.informative_only:
            # Drop the color factor (constant 0)
            labels = labels[1:]

        labels_tensor = torch.from_numpy(labels).float()

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, labels_tensor

    def sample_factor_fixed(
        self, factor_idx: int, fixed_value: int, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of images where factor_idx is held fixed to fixed_value,
        while all other factors vary uniformly at random.
        Crucial for FactorVAE and BetaVAE disentanglement metrics.
        """
        # If informative_only, shift factor_idx by +1 for the 6-factor indexing
        full_factor_idx = factor_idx + 1 if self.informative_only else factor_idx

        # Generate random classes for all factors
        random_classes = np.zeros((num_samples, 6), dtype=np.int64)
        for f, size in enumerate(self.FACTOR_SIZES):
            if f == full_factor_idx:
                random_classes[:, f] = fixed_value
            else:
                random_classes[:, f] = np.random.randint(0, size, size=num_samples)

        # Convert multidimensional factor indices to flat image indices
        flat_indices = np.dot(random_classes, self.factor_bases // self.FACTOR_SIZES)
        flat_indices = np.clip(flat_indices, 0, len(self.images) - 1)

        imgs = self.images[flat_indices]
        imgs_tensor = torch.from_numpy(imgs).unsqueeze(1).float()  # (B, 1, 64, 64)

        if self.use_discrete_factors:
            labels = self.latents_classes[flat_indices]
        else:
            labels = self.latents_values[flat_indices]

        if self.informative_only:
            labels = labels[:, 1:]

        labels_tensor = torch.from_numpy(labels).float()
        return imgs_tensor, labels_tensor


class Shapes3DDataset(Dataset):
    """
    3D Shapes dataset loader.

    Reference:
      Kim & Mnih, "Disentangling by Factorising" (ICML 2018)

    Ground-truth factors (6 factors, 480,000 total images):
      1. Floor hue (10 values)
      2. Wall hue (10 values)
      3. Object hue (10 values)
      4. Scale (8 values)
      5. Shape (4 values)
      6. Orientation (15 values)
    """

    FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
    FACTOR_SIZES = [10, 10, 10, 8, 4, 15]

    def __init__(
        self,
        root_dir: str = "./data/3dshapes",
        filename: str = "3dshapes.h5",
        download: bool = False,
        in_memory: bool = False,
        transform=None,
    ):
        super().__init__()
        if h5py is None:
            raise ImportError("h5py is required to load 3D Shapes. Install with `pip install h5py`.")

        self.root_dir = root_dir
        self.file_path = os.path.join(root_dir, filename)
        self.in_memory = in_memory
        self.transform = transform

        if download and not os.path.exists(self.file_path):
            self._download()

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"3D Shapes file not found at {self.file_path}. "
                f"Pass download=True or download manually from {SHAPES3D_URL}"
            )

        self._h5_file = None
        if self.in_memory:
            with h5py.File(self.file_path, "r") as f:
                print("Loading 3D Shapes dataset into memory (this may take a few seconds)...")
                self.images = np.array(f["images"])
                self.labels = np.array(f["labels"])
            print("3D Shapes loaded into memory.")
        else:
            # On-demand lazy HDF5 loading
            with h5py.File(self.file_path, "r") as f:
                self.length = len(f["images"])
            self.images = None
            self.labels = None

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.file_path, "r")
        return self._h5_file

    def _download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        print(f"Downloading 3D Shapes dataset from {SHAPES3D_URL} to {self.file_path}...")
        urllib.request.urlretrieve(SHAPES3D_URL, self.file_path)
        print("Download complete.")

    def __len__(self) -> int:
        if self.in_memory:
            return len(self.images)
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            img = self.images[idx]
            lbl = self.labels[idx]
        else:
            f = self._get_h5()
            img = f["images"][idx]
            lbl = f["labels"][idx]

        # Convert uint8 (64, 64, 3) -> float32 (3, 64, 64) in [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        lbl_tensor = torch.from_numpy(lbl).float()

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, lbl_tensor

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


class DummyDisentanglementDataset(Dataset):
    """
    Fast synthetic dummy dataset generator for CPU/MPS unit tests and debugging.

    Renders deterministic 64x64 geometric shapes (rectangles, ellipses) with controlled
    factors of variation:
      0. Shape (0: square, 1: circle/ellipse)
      1. Scale (3 sizes: small, medium, large)
      2. Pos X (8 discrete horizontal positions)
      3. Pos Y (8 discrete vertical positions)
      4. Color/Intensity (4 brightness levels)

    Total factor combinations: 2 * 3 * 8 * 8 * 4 = 1536 canonical states.
    Allows testing full training loops and metrics in under 5 seconds without internet downloads.
    """

    FACTOR_NAMES = ["shape", "scale", "pos_x", "pos_y", "intensity"]
    FACTOR_SIZES = [2, 3, 8, 8, 4]

    def __init__(
        self,
        num_samples: int = 512,
        in_channels: int = 1,
        image_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.image_size = image_size

        rng = np.random.RandomState(seed)

        # Sample factor combinations
        self.labels = np.zeros((num_samples, 5), dtype=np.int64)
        for f, size in enumerate(self.FACTOR_SIZES):
            self.labels[:, f] = rng.randint(0, size, size=num_samples)

        # Synthesize images
        self.images = np.zeros(
            (num_samples, in_channels, image_size, image_size), dtype=np.float32
        )

        for i in range(num_samples):
            shape_type = self.labels[i, 0]
            scale_level = self.labels[i, 1]
            pos_x_level = self.labels[i, 2]
            pos_y_level = self.labels[i, 3]
            intensity_level = self.labels[i, 4]

            # Radius / size: 4 to 12 pixels
            radius = 4 + scale_level * 4
            # Center coordinates: 12 to 52 pixels
            cx = 12 + int(pos_x_level * (image_size - 24) / 7.0)
            cy = 12 + int(pos_y_level * (image_size - 24) / 7.0)
            # Intensity in [0.4, 1.0]
            val = 0.4 + intensity_level * 0.2

            y_grid, x_grid = np.ogrid[:image_size, :image_size]

            if shape_type == 0:
                # Square
                mask = (
                    (x_grid >= cx - radius)
                    & (x_grid <= cx + radius)
                    & (y_grid >= cy - radius)
                    & (y_grid <= cy + radius)
                )
            else:
                # Circle / Ellipse
                mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= radius**2

            for c in range(in_channels):
                # Optionally add slight channel hue shifts for 3-channel RGB
                channel_val = val * (0.8 + 0.2 * c) if in_channels == 3 else val
                self.images[i, c, mask] = channel_val

        self.images_tensor = torch.from_numpy(self.images).float()
        self.labels_tensor = torch.from_numpy(self.labels).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images_tensor[idx], self.labels_tensor[idx]

    def sample_factor_fixed(
        self, factor_idx: int, fixed_value: int, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample batch where factor_idx is held fixed to fixed_value.
        Fast in-memory indexer for rapid metric evaluation.
        """
        matching_indices = np.where(self.labels[:, factor_idx] == fixed_value)[0]
        if len(matching_indices) > 0:
            chosen = np.random.choice(matching_indices, size=num_samples, replace=True)
            return self.images_tensor[chosen], self.labels_tensor[chosen]

        # Fallback if no matching sample exists in current buffer: synthesize directly
        images = np.zeros(
            (num_samples, self.in_channels, self.image_size, self.image_size),
            dtype=np.float32,
        )
        labels = np.zeros((num_samples, 5), dtype=np.int64)
        for f, size in enumerate(self.FACTOR_SIZES):
            if f == factor_idx:
                labels[:, f] = fixed_value
            else:
                labels[:, f] = np.random.randint(0, size, size=num_samples)

        y_grid, x_grid = np.ogrid[: self.image_size, : self.image_size]
        for i in range(num_samples):
            shape_type = labels[i, 0]
            radius = 4 + labels[i, 1] * 4
            cx = 12 + int(labels[i, 2] * (self.image_size - 24) / 7.0)
            cy = 12 + int(labels[i, 3] * (self.image_size - 24) / 7.0)
            val = 0.4 + labels[i, 4] * 0.2

            if shape_type == 0:
                mask = (
                    (x_grid >= cx - radius)
                    & (x_grid <= cx + radius)
                    & (y_grid >= cy - radius)
                    & (y_grid <= cy + radius)
                )
            else:
                mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= radius**2

            for c in range(self.in_channels):
                images[i, c, mask] = (
                    val * (0.8 + 0.2 * c) if self.in_channels == 3 else val
                )

        return torch.from_numpy(images).float(), torch.from_numpy(labels).float()


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Helper to instantiate PyTorch DataLoader with standard benchmark defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
