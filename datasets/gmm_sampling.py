"""GMM-based sampling for motion windows.

The implementation deliberately uses NumPy only (rather than scikit-learn) so it
works with Air-IO's existing requirements.  Features are computed from data that
is already resident in ``SeqeuncesMotionDataset``; no model inference is needed.
"""

import hashlib
import os

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


FEATURE_NAMES = (
    "gt_speed_mean", "gt_speed_std", "gt_speed_p95", "gt_speed_max",
    "gt_acc_mean", "gt_acc_std", "gt_acc_p95", "gt_delta_v",
    "gt_displacement", "gt_path_length",
    "imu_acc_norm_mean", "imu_acc_norm_std", "imu_acc_norm_p95",
    "imu_gyro_norm_mean", "imu_gyro_norm_std", "imu_gyro_norm_p95",
    "imu_acc_jerk_rms", "imu_gyro_jerk_rms",
    "imu_acc_x_std", "imu_acc_y_std", "imu_acc_z_std",
    "imu_gyro_x_std", "imu_gyro_y_std", "imu_gyro_z_std",
)

GMM_DEFAULTS = {
    "components": 12,
    "iterations": 100,
    "seed": 17,
    "balance_power": 1.0,
    "rarity_strength": 0.15,
    "min_weight": 0.2,
    "max_weight": 5.0,
}


def _numpy(value):
    return value.detach().cpu().double().numpy() if torch.is_tensor(value) else np.asarray(value, dtype=np.float64)


def _summary_norm(x):
    n = np.linalg.norm(x, axis=-1)
    return np.mean(n), np.std(n), np.percentile(n, 95)


def extract_window_features(dataset):
    """Return [number of windows, 24] GT+IMU feature matrix."""
    rows = []
    for seq_id, begin, end in dataset.index_map:
        vel = _numpy(dataset.gt_velo[seq_id][begin:end + 1])
        pos = _numpy(dataset.gt_pos[seq_id][begin:end + 1])
        acc = _numpy(dataset.acc[seq_id][begin:end])
        gyro = _numpy(dataset.gyro[seq_id][begin:end])
        dt = _numpy(dataset.dt[seq_id][begin:end]).reshape(-1)
        dt = np.maximum(dt[: max(len(vel) - 1, 0)], 1e-6)

        speed = np.linalg.norm(vel, axis=-1)
        gt_acc = np.diff(vel, axis=0) / dt[:, None]
        gt_acc_norm = np.linalg.norm(gt_acc, axis=-1)
        acc_stats = _summary_norm(acc)
        gyro_stats = _summary_norm(gyro)
        acc_jerk = np.diff(acc, axis=0) / np.maximum(dt[1:, None], 1e-6)
        gyro_jerk = np.diff(gyro, axis=0) / np.maximum(dt[1:, None], 1e-6)
        path = np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum()
        rows.append([
            speed.mean(), speed.std(), np.percentile(speed, 95), speed.max(),
            gt_acc_norm.mean(), gt_acc_norm.std(), np.percentile(gt_acc_norm, 95),
            np.linalg.norm(vel[-1] - vel[0]), np.linalg.norm(pos[-1] - pos[0]), path,
            *acc_stats, *gyro_stats,
            np.sqrt(np.mean(acc_jerk ** 2)), np.sqrt(np.mean(gyro_jerk ** 2)),
            *acc.std(axis=0), *gyro.std(axis=0),
        ])
    features = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite values found while extracting GMM sampling features")
    return features


def extract_window_provenance(dataset):
    """Return arrays that trace every feature row back to its source window."""
    sequence_ids, sequence_names = [], []
    begin_frames, end_frames, start_times, end_times = [], [], [], []
    names = getattr(dataset, "dataset_names", None)
    for seq_id, begin, end in dataset.index_map:
        sequence_ids.append(seq_id)
        sequence_names.append(str(names[seq_id]) if names is not None else str(seq_id))
        begin_frames.append(begin)
        end_frames.append(end)
        timestamps = dataset.ts[seq_id] if seq_id < len(dataset.ts) else None
        if timestamps is None or len(timestamps) == 0:
            start_times.append(np.nan)
            end_times.append(np.nan)
        else:
            ts = _numpy(timestamps).reshape(-1)
            start_times.append(ts[min(begin, len(ts) - 1)])
            end_times.append(ts[min(end, len(ts) - 1)])
    return {
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int64),
        "sequence_names": np.asarray(sequence_names, dtype=str),
        "begin_frames": np.asarray(begin_frames, dtype=np.int64),
        "end_frames": np.asarray(end_frames, dtype=np.int64),
        "start_timestamps": np.asarray(start_times, dtype=np.float64),
        "end_timestamps": np.asarray(end_times, dtype=np.float64),
    }


def _robust_scale(x, clip=8.0):
    center = np.median(x, axis=0)
    q25, q75 = np.percentile(x, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return np.clip((x - center) / scale, -clip, clip), center, scale


def _bounded_mean_one(values, lower, upper, iterations=60):
    """Scale positive values to mean one while respecting hard bounds."""
    if lower > 1 or upper < 1 or lower <= 0 or lower > upper:
        raise ValueError("Weight bounds must satisfy 0 < min_weight <= 1 <= max_weight")
    lo, hi = 0.0, max(1.0, upper / max(values.min(), 1e-12))
    while np.clip(values * hi, lower, upper).mean() < 1.0:
        hi *= 2.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if np.clip(values * mid, lower, upper).mean() < 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(values * ((lo + hi) / 2.0), lower, upper)


def fit_diagonal_gmm(x, components=12, iterations=100, seed=17, reg=1e-4):
    """Small, deterministic EM implementation returning posterior and log p(x)."""
    n, dims = x.shape
    k = min(int(components), n)
    if k < 1:
        raise ValueError("Cannot fit a GMM to an empty dataset")
    rng = np.random.default_rng(seed)
    # Farthest-point seeding gives more useful motion modes than random centres.
    means = [x[rng.integers(n)]]
    closest = np.sum((x - means[0]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(closest))
        means.append(x[idx])
        closest = np.minimum(closest, np.sum((x - x[idx]) ** 2, axis=1))
    means = np.asarray(means)
    variances = np.repeat((np.var(x, axis=0) + reg)[None, :], k, axis=0)
    mixture = np.full(k, 1.0 / k)

    for _ in range(iterations):
        log_prob = -0.5 * (np.sum(np.log(2 * np.pi * variances), axis=1)[None, :] +
                           np.sum((x[:, None, :] - means[None, :, :]) ** 2 /
                                  variances[None, :, :], axis=2))
        logits = log_prob + np.log(np.maximum(mixture, 1e-12))[None, :]
        peak = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - peak)
        posterior = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        count = posterior.sum(axis=0) + 1e-8
        new_means = posterior.T @ x / count[:, None]
        delta = x[:, None, :] - new_means[None, :, :]
        variances = np.maximum(np.einsum("nk,nkd->kd", posterior, delta ** 2) /
                               count[:, None], reg)
        if np.max(np.abs(new_means - means)) < 1e-5:
            means = new_means
            break
        means = new_means
        mixture = count / count.sum()

    # Re-evaluate with the final parameters (the convergence branch updates the
    # means immediately before leaving the loop).
    log_prob = -0.5 * (np.sum(np.log(2 * np.pi * variances), axis=1)[None, :] +
                       np.sum((x[:, None, :] - means[None, :, :]) ** 2 /
                              variances[None, :, :], axis=2))
    logits = log_prob + np.log(np.maximum(mixture, 1e-12))[None, :]
    peak = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - peak)
    posterior = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    log_density = (peak[:, 0] + np.log(exp_logits.sum(axis=1)))
    return posterior, log_density, means, variances, mixture


def compute_gmm_weights(dataset, components=12, iterations=100, seed=17,
                        balance_power=1.0, rarity_strength=0.15,
                        min_weight=0.2, max_weight=5.0):
    """Compute bounded weights balancing motion modes and rare within-mode samples."""
    raw = extract_window_features(dataset)
    x, center, scale = _robust_scale(raw)
    posterior, log_density, means, variances, mixture = fit_diagonal_gmm(
        x, components, iterations, seed
    )
    # Soft assignment avoids discontinuities for windows on a cluster boundary.
    balance = np.sum(posterior / np.maximum(mixture, 1e-8)[None, :] ** balance_power, axis=1)
    rarity = -log_density
    rarity = (rarity - np.median(rarity)) / (np.percentile(rarity, 75) - np.percentile(rarity, 25) + 1e-6)
    weights = balance * np.exp(rarity_strength * np.clip(rarity, -3, 3))
    weights = _bounded_mean_one(weights, min_weight, max_weight)
    details = {
        "features": raw, "feature_names": np.asarray(FEATURE_NAMES),
        "center": center, "scale": scale, "posterior": posterior,
        "labels": posterior.argmax(axis=1), "means": means,
        "variances": variances, "mixture": mixture,
    }
    details.update(extract_window_provenance(dataset))
    return weights, details


def dataset_fingerprint(dataset):
    identity = (dataset.index_map, getattr(dataset, "dataset_names", None),
                [tuple(v.shape) for v in dataset.acc])
    value = repr(identity).encode("utf8")
    return hashlib.sha256(value).hexdigest()[:16]


def settings_fingerprint(settings):
    """Identify every setting that affects the cached sampling weights."""
    identity = (tuple(FEATURE_NAMES), tuple(
        (name, settings[name]) for name in GMM_DEFAULTS
    ))
    return hashlib.sha256(repr(identity).encode("utf8")).hexdigest()[:16]


def gmm_settings(conf):
    """Return normalized GMM arguments from a HOCON block or mapping."""
    return {name: conf.get(name, default) for name, default in GMM_DEFAULTS.items()}


def build_gmm_sampler(dataset, conf):
    """Build a replacement sampler from a HOCON ``train.gmm_sampling`` block."""
    cache = str(conf.get("cache_path", ""))
    fingerprint = dataset_fingerprint(dataset)
    settings = gmm_settings(conf)
    expected_settings = settings_fingerprint(settings)
    if cache and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as saved:
            valid = (
                len(saved["weights"]) == len(dataset)
                and str(saved["fingerprint"]) == fingerprint
                and "settings_fingerprint" in saved
                and str(saved["settings_fingerprint"]) == expected_settings
            )
            if not valid:
                raise ValueError(
                    f"Stale GMM cache {cache}; regenerate it for this dataset/config"
                )
            weights = saved["weights"].copy()
    else:
        weights, details = compute_gmm_weights(dataset, **settings)
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.savez_compressed(
                cache, weights=weights, fingerprint=fingerprint,
                settings_fingerprint=expected_settings, **details,
            )
    generator = torch.Generator().manual_seed(int(conf.get("seed", 17)))
    count = int(round(len(dataset) * float(conf.get("epoch_multiplier", 1.0))))
    print(f"GMM sampler: {len(weights)} windows, weight range "
          f"[{weights.min():.3f}, {weights.max():.3f}], {count} draws/epoch")
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), count,
                                 replacement=True, generator=generator)
