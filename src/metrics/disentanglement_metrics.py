"""
Quantitative Disentanglement Metrics for Variational Autoencoders.

Implements standard benchmark metrics strictly following:
1. Mutual Information Gap (MIG; Chen et al., 2018, Eq. 6)
2. FactorVAE Score (Kim & Mnih, 2018)
3. DCI Disentanglement, Completeness, and Informativeness (Eastwood & Williams, 2018; Locatello et al., 2019)
4. Reconstruction MSE
"""

import math
from typing import Dict, Tuple, Optional, Union, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# 1. Mutual Information Gap (MIG; Chen et al., 2018)
# ---------------------------------------------------------------------------

def compute_discrete_mutual_info(
    z_discrete: np.ndarray, factor: np.ndarray
) -> float:
    """
    Compute discrete mutual information I(Z_j; V_k) from empirical counts.

    Args:
        z_discrete: 1D array of discretized latent representations (N,).
        factor: 1D array of discrete ground-truth factor labels (N,).

    Returns:
        Mutual information in nats.
    """
    n_samples = len(z_discrete)
    if n_samples == 0:
        return 0.0

    # Contingency matrix / joint counts
    unique_z, z_inv = np.unique(z_discrete, return_inverse=True)
    unique_f, f_inv = np.unique(factor, return_inverse=True)

    n_z = len(unique_z)
    n_f = len(unique_f)

    if n_z <= 1 or n_f <= 1:
        return 0.0

    joint_counts = np.zeros((n_z, n_f), dtype=np.float64)
    np.add.at(joint_counts, (z_inv, f_inv), 1.0)

    p_joint = joint_counts / n_samples
    p_z = np.sum(p_joint, axis=1, keepdims=True)
    p_f = np.sum(p_joint, axis=0, keepdims=True)

    # Filter out zero probabilities for numerical stability
    nz = p_joint > 0.0
    p_nz = p_joint[nz]
    p_prod = (p_z @ p_f)[nz]

    mi = np.sum(p_nz * np.log(p_nz / p_prod))
    return float(max(0.0, mi))


def compute_discrete_entropy(factor: np.ndarray) -> float:
    """Compute empirical Shannon entropy H(V_k) in nats."""
    _, counts = np.unique(factor, return_counts=True)
    p = counts / float(len(factor))
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def compute_mig(
    mus: np.ndarray,
    factors: np.ndarray,
    num_bins: int = 20,
) -> Dict[str, Any]:
    """
    Mutual Information Gap (MIG) metric (Chen et al., 2018, Eq. 6).

    Args:
        mus: Latent representations (typically encoder means), shape (N, d).
        factors: Ground-truth factor values or classes, shape (N, K).
        num_bins: Number of histogram bins for discretizing each latent coordinate.

    Returns:
        Dictionary containing:
          - 'mig': Overall mean Mutual Information Gap across all factors.
          - 'per_factor_mig': List of individual factor MIG scores.
          - 'mi_matrix': Full mutual information matrix of shape (d, K).
          - 'entropies': Array of factor entropies H(V_k).
    """
    n_samples, latent_dim = mus.shape
    num_factors = factors.shape[1]

    # Discretize continuous representations into num_bins equal-width bins
    # per coordinate across the empirical range
    mus_discrete = np.zeros_like(mus, dtype=np.int64)
    for j in range(latent_dim):
        col = mus[:, j]
        col_min, col_max = np.min(col), np.max(col)
        if col_max > col_min:
            bins = np.linspace(col_min, col_max, num_bins + 1)
            # np.digitize returns 1..num_bins; shift to 0..num_bins-1
            mus_discrete[:, j] = np.clip(np.digitize(col, bins) - 1, 0, num_bins - 1)
        else:
            mus_discrete[:, j] = 0

    # Compute mutual information matrix: shape (latent_dim, num_factors)
    mi_matrix = np.zeros((latent_dim, num_factors), dtype=np.float64)
    entropies = np.zeros(num_factors, dtype=np.float64)
    gaps = []

    for k in range(num_factors):
        factor_col = factors[:, k]
        h_k = compute_discrete_entropy(factor_col)
        entropies[k] = h_k

        if h_k <= 1e-12:
            # Constant or zero-entropy factor
            gaps.append(0.0)
            continue

        for j in range(latent_dim):
            mi_matrix[j, k] = compute_discrete_mutual_info(
                mus_discrete[:, j], factor_col
            )

        # Sort MI with factor k descending
        sorted_mi = np.sort(mi_matrix[:, k])[::-1]
        top1_mi = sorted_mi[0]
        top2_mi = sorted_mi[1] if latent_dim > 1 else 0.0

        gap = (top1_mi - top2_mi) / h_k
        gaps.append(float(np.clip(gap, 0.0, 1.0)))

    overall_mig = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
    return {
        "mig": overall_mig,
        "per_factor_mig": gaps,
        "mi_matrix": mi_matrix,
        "entropies": entropies,
    }


# ---------------------------------------------------------------------------
# 2. FactorVAE Score (Kim & Mnih, 2018)
# ---------------------------------------------------------------------------

def compute_factorvae_score(
    model: nn.Module,
    dataset: Any,
    device: torch.device,
    num_train_votes: int = 1000,
    num_eval_votes: int = 500,
    batch_size: int = 64,
    variance_threshold: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    FactorVAE metric (Kim & Mnih, 2018; Locatello et al., 2019 Appendix C).

    Protocol:
      1. Estimate empirical variance s_j^2 of each latent coordinate on a sample batch.
         Exclude collapsed dimensions with s_j^2 < variance_threshold.
      2. For each vote:
         - Select factor k uniformly at random.
         - Fix factor k and randomly sample remaining factors for batch_size images.
         - Encode images to means mu.
         - Compute empirical variance of each dimension normalized by overall s_j^2.
         - Latent coordinate d* = argmin_j (var_j / s_j^2) votes for factor k.
      3. Fit a majority-vote classifier on training votes; evaluate accuracy on eval votes.

    Args:
        model: Trained VAE model.
        dataset: Dataset providing `sample_factor_fixed(factor_idx, fixed_val, num_samples)`.
        device: torch.device.
        num_train_votes: Number of votes for classifier training (default 1000; literature uses 10000).
        num_eval_votes: Number of votes for classifier evaluation (default 500; literature uses 5000).
        batch_size: Number of images per vote (default 64).
        variance_threshold: Threshold to exclude collapsed latent dimensions.
        seed: Random seed.

    Returns:
        Dictionary containing:
          - 'factorvae_score': Majority vote test accuracy in [0, 1].
          - 'active_dims': Indices of non-collapsed latent dimensions.
    """
    model.eval()
    rng = np.random.RandomState(seed)

    # Step 1: Estimate empirical global variances across 2000 samples
    sample_images = []
    n_init = min(2000, len(dataset))
    init_indices = rng.choice(len(dataset), size=n_init, replace=False)
    for idx in init_indices:
        img, _ = dataset[idx]
        sample_images.append(img)
    sample_tensor = torch.stack(sample_images).to(device)

    with torch.no_grad():
        mu_sample, _ = model.encode(sample_tensor)
        global_variances = torch.var(mu_sample, dim=0).cpu().numpy()

    # Identify active dimensions
    active_dims = np.where(global_variances >= variance_threshold)[0]
    if len(active_dims) == 0:
        # Fallback to all dimensions if all are below threshold
        active_dims = np.arange(len(global_variances))

    active_global_vars = global_variances[active_dims]

    num_factors = (
        len(dataset.INFORMATIVE_FACTOR_SIZES)
        if hasattr(dataset, "INFORMATIVE_FACTOR_SIZES")
        else len(dataset.FACTOR_SIZES)
    )
    factor_sizes = (
        dataset.INFORMATIVE_FACTOR_SIZES
        if hasattr(dataset, "INFORMATIVE_FACTOR_SIZES")
        else dataset.FACTOR_SIZES
    )

    def _generate_votes(num_votes: int) -> Tuple[np.ndarray, np.ndarray]:
        votes = np.zeros(num_votes, dtype=np.int64)
        targets = np.zeros(num_votes, dtype=np.int64)

        for v in range(num_votes):
            k = rng.randint(0, num_factors)
            fixed_val = rng.randint(0, factor_sizes[k])

            batch_imgs, _ = dataset.sample_factor_fixed(
                factor_idx=k, fixed_value=fixed_val, num_samples=batch_size
            )
            batch_imgs = batch_imgs.to(device)

            with torch.no_grad():
                mu_batch, _ = model.encode(batch_imgs)
                batch_vars = torch.var(mu_batch, dim=0).cpu().numpy()

            norm_vars = batch_vars[active_dims] / active_global_vars
            argmin_idx = active_dims[np.argmin(norm_vars)]

            votes[v] = argmin_idx
            targets[v] = k

        return votes, targets

    train_votes, train_targets = _generate_votes(num_train_votes)
    eval_votes, eval_targets = _generate_votes(num_eval_votes)

    # Train majority-vote classifier
    latent_dim = len(global_variances)
    majority_map = {}
    for d_idx in range(latent_dim):
        assigned_targets = train_targets[train_votes == d_idx]
        if len(assigned_targets) > 0:
            majority_map[d_idx] = int(np.bincount(assigned_targets).argmax())
        else:
            majority_map[d_idx] = 0

    predictions = np.array([majority_map.get(v, 0) for v in eval_votes])
    accuracy = float(np.mean(predictions == eval_targets))

    return {
        "factorvae_score": accuracy,
        "active_dims": active_dims.tolist(),
        "global_variances": global_variances.tolist(),
    }


def compute_factorvae_score_from_representations(
    representations: np.ndarray,
    factors: np.ndarray,
    num_train_votes: int = 1000,
    num_eval_votes: int = 500,
    batch_size: int = 64,
    variance_threshold: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    FactorVAE metric computed directly from latent representation matrix and ground-truth factors.
    Useful for synthetic representations and evaluations where data is already encoded.

    Args:
        representations: Latent representations matrix of shape (N, d).
        factors: Discrete ground-truth factors matrix of shape (N, K).
        num_train_votes: Number of votes for majority classifier training.
        num_eval_votes: Number of votes for classifier evaluation.
        batch_size: Number of representations per vote.
        variance_threshold: Threshold to exclude collapsed latent dimensions.
        seed: Random seed.

    Returns:
        Dictionary containing 'factorvae_score', 'active_dims', and 'global_variances'.
    """
    rng = np.random.RandomState(seed)
    n_samples, latent_dim = representations.shape
    num_factors = factors.shape[1]

    # Global empirical variances per dimension
    global_variances = np.var(representations, axis=0)
    active_dims = np.where(global_variances >= variance_threshold)[0]
    if len(active_dims) == 0:
        active_dims = np.arange(latent_dim)
    active_global_vars = global_variances[active_dims]

    # Pre-index factor values to sample fast
    factor_indices = {}
    for k in range(num_factors):
        unique_vals = np.unique(factors[:, k])
        factor_indices[k] = {val: np.where(factors[:, k] == val)[0] for val in unique_vals}

    def _generate_rep_votes(num_votes: int) -> Tuple[np.ndarray, np.ndarray]:
        votes = np.zeros(num_votes, dtype=np.int64)
        targets = np.zeros(num_votes, dtype=np.int64)
        for v in range(num_votes):
            k = rng.randint(0, num_factors)
            vals = list(factor_indices[k].keys())
            val = vals[rng.randint(0, len(vals))]
            pool = factor_indices[k][val]
            chosen = rng.choice(pool, size=batch_size, replace=(len(pool) < batch_size))
            batch_reps = representations[chosen]
            batch_vars = np.var(batch_reps, axis=0)
            norm_vars = batch_vars[active_dims] / np.maximum(active_global_vars, 1e-12)
            argmin_idx = active_dims[np.argmin(norm_vars)]
            votes[v] = argmin_idx
            targets[v] = k
        return votes, targets

    train_votes, train_targets = _generate_rep_votes(num_train_votes)
    eval_votes, eval_targets = _generate_rep_votes(num_eval_votes)

    majority_map = {}
    for d_idx in range(latent_dim):
        assigned_targets = train_targets[train_votes == d_idx]
        if len(assigned_targets) > 0:
            majority_map[d_idx] = int(np.bincount(assigned_targets).argmax())
        else:
            majority_map[d_idx] = 0

    predictions = np.array([majority_map.get(v, 0) for v in eval_votes])
    accuracy = float(np.mean(predictions == eval_targets))

    return {
        "factorvae_score": accuracy,
        "active_dims": active_dims.tolist(),
        "global_variances": global_variances.tolist(),
    }



# ---------------------------------------------------------------------------
# 3. DCI Disentanglement, Completeness, Informativeness (Eastwood & Williams, 2018)
# ---------------------------------------------------------------------------

def compute_dci(
    train_mus: np.ndarray,
    train_factors: np.ndarray,
    test_mus: np.ndarray,
    test_factors: np.ndarray,
    classifier_type: str = "gradient_boosting",
) -> Dict[str, Any]:
    """
    DCI framework: Disentanglement, Completeness, and Informativeness.
    (Eastwood & Williams, 2018; Locatello et al., 2019 Appendix C)

    Protocol:
      1. For each ground-truth factor k, fit a predictive model (GBDT or Random Forest)
         mapping latent codes z to factor class v_k.
      2. Extract feature importances R_{k, j} >= 0 (shape K x d).
      3. Disentanglement D:
           Entropy of each column j (normalized across factors k): H(P_{.|j})
           D_j = 1 - H(P_{.|j}) / log(K)
           Overall D = sum_j rho_j * D_j, where rho_j is relative importance of latent j.
      4. Completeness C:
           Entropy of each row k (normalized across latents j): H(Q_{.|k})
           C_k = 1 - H(Q_{.|k}) / log(d)
           Overall C = sum_k omega_k * C_k, where omega_k is relative importance of factor k.
      5. Informativeness I:
           Mean test classification accuracy across all factors.

    Args:
        train_mus: Training latent codes, shape (N_train, d).
        train_factors: Training ground-truth factors, shape (N_train, K).
        test_mus: Test latent codes, shape (N_test, d).
        test_factors: Test ground-truth factors, shape (N_test, K).
        classifier_type: 'random_forest' (fast, robust) or 'gradient_boosting' (Locatello default).

    Returns:
        Dictionary containing 'disentanglement', 'completeness', 'informativeness',
        and the full (K, d) importance matrix.
    """
    n_train, latent_dim = train_mus.shape
    num_factors = train_factors.shape[1]

    importance_matrix = np.zeros((num_factors, latent_dim), dtype=np.float64)
    test_accuracies = []

    for k in range(num_factors):
        y_train = train_factors[:, k].astype(np.int64)
        y_test = test_factors[:, k].astype(np.int64)

        if len(np.unique(y_train)) <= 1:
            importance_matrix[k, :] = 0.0
            test_accuracies.append(1.0)
            continue

        if classifier_type == "gradient_boosting":
            clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)

        clf.fit(train_mus, y_train)
        importance_matrix[k, :] = clf.feature_importances_

        preds = clf.predict(test_mus)
        test_accuracies.append(accuracy_score(y_test, preds))

    # --- Disentanglement Score ---
    # R: shape (num_factors, latent_dim)
    r_total = np.sum(importance_matrix)
    if r_total <= 1e-12:
        return {
            "disentanglement": 0.0,
            "completeness": 0.0,
            "informativeness": float(np.mean(test_accuracies)),
            "importance_matrix": importance_matrix,
        }

    # Relative importance of each latent dimension rho_j: sum_k R_{kj} / R_total
    rho = np.sum(importance_matrix, axis=0) / r_total

    disentanglement_per_dim = np.zeros(latent_dim, dtype=np.float64)
    log_k = math.log(num_factors) if num_factors > 1 else 1.0

    for j in range(latent_dim):
        col_sum = np.sum(importance_matrix[:, j])
        if col_sum > 1e-12 and num_factors > 1:
            p_j = importance_matrix[:, j] / col_sum
            # Shannon entropy in base K: - sum(p * log(p)) / log(K)
            nonzero_p = p_j[p_j > 0.0]
            entropy_j = -np.sum(nonzero_p * np.log(nonzero_p)) / log_k
            disentanglement_per_dim[j] = 1.0 - entropy_j
        else:
            # If a coordinate has zero importance or num_factors == 1, disentanglement is 1
            disentanglement_per_dim[j] = 1.0

    overall_disentanglement = float(np.sum(rho * disentanglement_per_dim))

    # --- Completeness Score ---
    # Relative importance of each factor omega_k: sum_j R_{kj} / R_total
    omega = np.sum(importance_matrix, axis=1) / r_total
    completeness_per_factor = np.zeros(num_factors, dtype=np.float64)
    log_d = math.log(latent_dim) if latent_dim > 1 else 1.0

    for k in range(num_factors):
        row_sum = np.sum(importance_matrix[k, :])
        if row_sum > 1e-12 and latent_dim > 1:
            q_k = importance_matrix[k, :] / row_sum
            nonzero_q = q_k[q_k > 0.0]
            entropy_k = -np.sum(nonzero_q * np.log(nonzero_q)) / log_d
            completeness_per_factor[k] = 1.0 - entropy_k
        else:
            completeness_per_factor[k] = 1.0

    overall_completeness = float(np.sum(omega * completeness_per_factor))
    overall_informativeness = float(np.mean(test_accuracies))

    return {
        "disentanglement": overall_disentanglement,
        "completeness": overall_completeness,
        "informativeness": overall_informativeness,
        "disentanglement_per_dim": disentanglement_per_dim.tolist(),
        "completeness_per_factor": completeness_per_factor.tolist(),
        "importance_matrix": importance_matrix,
    }


# ---------------------------------------------------------------------------
# 4. Reconstruction MSE
# ---------------------------------------------------------------------------

def compute_reconstruction_mse(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = 2000,
) -> float:
    """
    Compute average pixel-wise Mean Squared Error between input images and reconstructions.

    Args:
        model: Trained VAE model.
        data_loader: DataLoader providing batches (images, labels).
        device: torch.device.
        max_samples: Maximum number of samples to evaluate on (for speed).

    Returns:
        Scalar pixel MSE in [0, 1].
    """
    model.eval()
    total_squared_error = 0.0
    total_pixels = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            recon_x = model.decode(mu)

            batch_error = torch.sum((x - recon_x).pow(2)).item()
            total_squared_error += batch_error
            total_pixels += x.numel()

            if max_samples is not None and total_pixels >= max_samples * x.shape[1] * x.shape[2] * x.shape[3]:
                break

    return float(total_squared_error / max(1, total_pixels))


# ---------------------------------------------------------------------------
# 5. Full Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_all_metrics(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_eval_samples: int = 10000,
    batch_size: int = 128,
    include_factorvae: bool = True,
    num_factorvae_train_votes: int = 800,
    num_factorvae_eval_votes: int = 400,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation harness:
    Computes MIG, DCI Disentanglement, Completeness, Informativeness,
    FactorVAE Score, and Reconstruction MSE in a unified single pass.
    """
    model.eval()
    n_samples = min(num_eval_samples, len(dataset))

    # Collect representations and labels
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_mus = []
    all_factors = []

    collected = 0
    with torch.no_grad():
        for x, labels in eval_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            all_mus.append(mu.cpu().numpy())
            all_factors.append(labels.numpy())
            collected += len(x)
            if collected >= n_samples:
                break

    all_mus = np.concatenate(all_mus, axis=0)[:n_samples]
    all_factors = np.concatenate(all_factors, axis=0)[:n_samples]

    # Split into 80% train / 20% test for DCI
    split_idx = int(0.8 * n_samples)
    train_mus, test_mus = all_mus[:split_idx], all_mus[split_idx:]
    train_factors, test_factors = all_factors[:split_idx], all_factors[split_idx:]

    # 1. MIG
    mig_results = compute_mig(all_mus, all_factors)

    # 2. DCI
    dci_results = compute_dci(train_mus, train_factors, test_mus, test_factors)

    # 3. FactorVAE score
    factorvae_score = None
    if include_factorvae and hasattr(dataset, "sample_factor_fixed"):
        try:
            fvae_res = compute_factorvae_score(
                model=model,
                dataset=dataset,
                device=device,
                num_train_votes=num_factorvae_train_votes,
                num_eval_votes=num_factorvae_eval_votes,
                batch_size=64,
            )
            factorvae_score = fvae_res["factorvae_score"]
        except Exception as e:
            print(f"Warning: FactorVAE score computation skipped: {e}")

    # 4. Reconstruction MSE
    mse = compute_reconstruction_mse(model, eval_loader, device=device, max_samples=n_samples)

    report = {
        "mig": mig_results["mig"],
        "dci_disentanglement": dci_results["disentanglement"],
        "dci_completeness": dci_results["completeness"],
        "dci_informativeness": dci_results["informativeness"],
        "factorvae_score": factorvae_score,
        "reconstruction_mse": mse,
    }
    return report
