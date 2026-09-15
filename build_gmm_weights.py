"""Precompute and inspect GT+IMU GMM sampling weights."""

import argparse
import csv
import os

import numpy as np
from pyhocon import ConfigFactory

from datasets import SeqeuncesMotionDataset
from datasets.gmm_sampling import compute_gmm_weights, dataset_fingerprint


def _pca_2d(values):
    """Project values to two dimensions without adding a sklearn dependency."""
    values = values - values.mean(axis=0, keepdims=True)
    _, singular_values, axes = np.linalg.svd(values, full_matrices=False)
    projection = values @ axes[:2].T
    explained = singular_values ** 2
    explained = explained[:2] / max(explained.sum(), 1e-12)
    if projection.shape[1] == 1:
        projection = np.column_stack((projection, np.zeros(len(projection))))
        explained = np.append(explained, 0.0)
    return projection, explained


def drop_incomplete_tail_windows(dataset):
    """Drop short infevaluate remainders so all feature rows are comparable."""
    expected = {}
    for seq_id in range(len(dataset.dataset_names)):
        lengths = [end - begin for sid, begin, end in dataset.index_map if sid == seq_id]
        if lengths:
            values, counts = np.unique(lengths, return_counts=True)
            expected[seq_id] = int(values[np.argmax(counts)])
    before = len(dataset.index_map)
    dataset.index_map = [window for window in dataset.index_map
                         if window[2] - window[1] == expected.get(window[0])]
    removed = before - len(dataset.index_map)
    if removed:
        print(f"Dropped {removed} incomplete tail windows from diversity analysis")


def _dominant_feature(details, cluster_id):
    raw = details["features"]
    scaled = np.clip((raw - details["center"]) / details["scale"], -8, 8)
    profile = np.median(scaled[details["labels"] == cluster_id], axis=0)
    index = int(np.argmax(np.abs(profile)))
    return str(details["feature_names"][index]), float(profile[index])


def _short_sequence_name(sequence):
    """Compact a dataset/condition/flight path for dashboard tick labels."""
    parts = str(sequence).split("/")
    if len(parts) < 3:
        return str(sequence)
    dataset = parts[0].replace("T-Lab_", "").replace("_dataset", "")
    return f"{dataset} · {parts[-2]} · {parts[-1].replace('flight_', 'F')}"


def write_provenance_reports(details, weights, prefix):
    """Write lossless window traceability and an aggregated flight/component report."""
    window_path = prefix + "_windows.csv"
    cluster_path = prefix + "_clusters.csv"
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    names = details["feature_names"].astype(str)
    fields = ["window_index", "component", "sequence_id", "sequence", "begin_frame",
              "end_frame", "start_timestamp", "end_timestamp", "sampling_weight", *names]
    with open(window_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for i, feature_row in enumerate(details["features"]):
            writer.writerow([
                i, int(details["labels"][i]), int(details["sequence_ids"][i]),
                details["sequence_names"][i], int(details["begin_frames"][i]),
                int(details["end_frames"][i]), details["start_timestamps"][i],
                details["end_timestamps"][i], weights[i], *feature_row,
            ])

    with open(cluster_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component", "sequence", "window_count", "component_share_pct",
                         "sequence_share_pct", "dominant_feature", "median_robust_z",
                         "representative_feature_value", "representative_window",
                         "begin_frame", "end_frame", "start_timestamp", "end_timestamp"])
        labels = details["labels"]
        sequences = details["sequence_names"].astype(str)
        scaled = np.clip((details["features"] - details["center"]) / details["scale"], -8, 8)
        for component in np.unique(labels):
            component_mask = labels == component
            feature, z_score = _dominant_feature(details, component)
            feature_index = int(np.flatnonzero(details["feature_names"].astype(str) == feature)[0])
            centroid = np.median(scaled[component_mask], axis=0)
            for sequence in np.unique(sequences[component_mask]):
                mask = component_mask & (sequences == sequence)
                indices = np.flatnonzero(mask)
                representative = indices[np.argmin(np.sum((scaled[indices] - centroid) ** 2, axis=1))]
                writer.writerow([
                    int(component), sequence, len(indices),
                    100 * len(indices) / component_mask.sum(),
                    100 * len(indices) / np.sum(sequences == sequence), feature, z_score,
                    details["features"][representative, feature_index], int(representative),
                    int(details["begin_frames"][representative]),
                    int(details["end_frames"][representative]),
                    details["start_timestamps"][representative],
                    details["end_timestamps"][representative],
                ])
    return window_path, cluster_path


def create_dashboard(details, weights, output, max_points=20000, seed=17):
    """Save a visual diversity report and return its path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = details["features"]
    names = details["feature_names"].astype(str)
    scaled = np.clip((raw - details["center"]) / details["scale"], -8, 8)
    # Preserve the true robust z-scores for diagnostics. The clipped values
    # above remain the representation used by the fitted GMM and PCA plots.
    unscaled_profiles = (raw - details["center"]) / details["scale"]
    labels = details["labels"]
    cluster_ids, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)
    shown = (rng.choice(len(raw), max_points, replace=False)
             if len(raw) > max_points else np.arange(len(raw)))

    joint, joint_var = _pca_2d(scaled)
    gt, gt_var = _pca_2d(scaled[:, :10])
    imu, imu_var = _pca_2d(scaled[:, 10:])
    cmap = plt.get_cmap("tab20", max(len(cluster_ids), 1))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)

    def scatter(ax, points, variance, title):
        plot = ax.scatter(points[shown, 0], points[shown, 1], c=labels[shown],
                          cmap=cmap, s=5, alpha=0.45, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)")
        return plot

    scatter(axes[0, 0], joint, joint_var, "Joint GT + IMU motion modes")
    scatter(axes[0, 1], gt, gt_var, "Ground-truth kinematic diversity")
    scatter(axes[0, 2], imu, imu_var, "IMU excitation diversity")

    axes[1, 0].bar(cluster_ids, counts, color=[cmap(i) for i in cluster_ids])
    axes[1, 0].axhline(len(labels) / len(cluster_ids), color="black", ls="--",
                       lw=1, label="uniform target")
    axes[1, 0].set(title="Original windows per GMM component",
                   xlabel="component", ylabel="number of windows")
    axes[1, 0].legend()

    sequences = details["sequence_names"].astype(str)
    sequence_names = np.unique(sequences)
    contribution = np.array([[np.sum((labels == component) & (sequences == sequence))
                              for component in cluster_ids] for sequence in sequence_names])
    contribution = 100 * contribution / np.maximum(contribution.sum(axis=0), 1)
    # A square-root colour normalization keeps small-but-real contributors
    # visible without losing the few dominant cells. Exact values are printed
    # in cells where there is enough room to read them.
    from matplotlib.colors import PowerNorm
    flight_image = axes[1, 1].imshow(
        contribution, aspect="auto", cmap="viridis",
        norm=PowerNorm(gamma=0.55, vmin=0, vmax=max(1, contribution.max())),
    )
    axes[1, 1].set(title="Share of each component supplied by each flight",
                   xlabel="component", ylabel="dataset / flight")
    axes[1, 1].set_xticks(np.arange(len(cluster_ids)))
    axes[1, 1].set_xticklabels(cluster_ids)
    axes[1, 1].set_yticks(np.arange(len(sequence_names)))
    axes[1, 1].set_yticklabels([_short_sequence_name(s) for s in sequence_names],
                               fontsize=6)
    axes[1, 1].set_xticks(np.arange(-.5, len(cluster_ids), 1), minor=True)
    axes[1, 1].set_yticks(np.arange(-.5, len(sequence_names), 1), minor=True)
    axes[1, 1].grid(which="minor", color="white", linewidth=.35, alpha=.55)
    axes[1, 1].tick_params(which="minor", bottom=False, left=False)
    for row in range(len(sequence_names)):
        for column in range(len(cluster_ids)):
            value = contribution[row, column]
            if value >= 5:
                text_color = "black" if value > contribution.max() * .55 else "white"
                axes[1, 1].text(column, row, f"{value:.0f}", ha="center", va="center",
                                fontsize=5.5, color=text_color)
    fig.colorbar(flight_image, ax=axes[1, 1], label="component windows (%)")

    profiles = np.stack([np.median(unscaled_profiles[labels == i], axis=0)
                         for i in cluster_ids])
    feature_limit = max(3.0, float(np.max(np.abs(profiles))))
    image = axes[1, 2].imshow(profiles, aspect="auto", cmap="coolwarm",
                              vmin=-feature_limit, vmax=feature_limit)
    axes[1, 2].set_title("Median standardized feature by component")
    axes[1, 2].set_xlabel("feature  (ground truth  |  IMU)")
    axes[1, 2].set_ylabel("component")
    # Separate tick positions and labels for compatibility with Matplotlib 3.3
    # and other versions whose set_xticks/set_yticks do not accept labels.
    axes[1, 2].set_xticks(np.arange(len(names)))
    axes[1, 2].set_xticklabels(names, rotation=90, fontsize=7)
    axes[1, 2].set_yticks(np.arange(len(cluster_ids)))
    axes[1, 2].set_yticklabels(cluster_ids)
    # Make the GT (first 10 columns) / IMU boundary visually explicit.
    axes[1, 2].axvline(9.5, color="black", linewidth=1.2)
    axes[1, 2].set_xticks(np.arange(-.5, len(names), 1), minor=True)
    axes[1, 2].set_yticks(np.arange(-.5, len(cluster_ids), 1), minor=True)
    axes[1, 2].grid(which="minor", color="white", linewidth=.3, alpha=.5)
    axes[1, 2].tick_params(which="minor", bottom=False, left=False)
    for row in range(len(cluster_ids)):
        for column in range(len(names)):
            value = profiles[row, column]
            # Mid-range colours are light; extremes need white text for contrast.
            text_color = "white" if abs(value) > feature_limit * .55 else "black"
            axes[1, 2].text(column, row, f"{value:.1f}", ha="center", va="center",
                            fontsize=4.2, color=text_color)
    fig.colorbar(image, ax=axes[1, 2], label="median robust z-score")
    split_name = str(details.get("dataset_split", "train"))
    fig.suptitle(f"Air-IO {split_name}-data diversity ({len(raw):,} windows)", fontsize=16)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/TLab/finetune_motion_body.conf")
    parser.add_argument(
        "--dataset-split", default="train",
        choices=("train", "test", "eval", "inference", "infevaluate"),
        help="dataset config block to inspect; 'infevaluate' aliases 'inference'",
    )
    parser.add_argument("--output", default="experiments/tlab_finetune/gmm_weights.npz")
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--balance-power", type=float, default=1.0)
    parser.add_argument("--rarity-strength", type=float, default=0.15)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument(
        "--plot", default=None,
        help="dashboard PNG path; defaults beside --output (use 'none' to disable)",
    )
    parser.add_argument("--max-plot-points", type=int, default=20000)
    parser.add_argument("--report-prefix", default=None,
                        help="prefix for per-window and per-cluster provenance CSVs")
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.config)
    config_split = "inference" if args.dataset_split == "infevaluate" else args.dataset_split
    if config_split not in conf.dataset:
        raise ValueError(f"Dataset config has no '{config_split}' block")
    dataset_config = conf.dataset[config_split]
    if config_split == "inference":
        # This report needs aligned GT, velocity and raw IMU only. Do not make
        # whole-dataset inspection depend on an optional inference-orientation
        # pickle, and do not mutate the configuration file on disk.
        dataset_config.put("rot_type", "gtrot")
    dataset = SeqeuncesMotionDataset(
        data_set_config=dataset_config, device="cpu"
    )
    drop_incomplete_tail_windows(dataset)
    weights, details = compute_gmm_weights(
        dataset, args.components, args.iterations, args.seed,
        args.balance_power, args.rarity_strength, args.min_weight, args.max_weight,
    )
    details["dataset_split"] = np.asarray(args.dataset_split)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, weights=weights,
                        fingerprint=dataset_fingerprint(dataset), **details)
    labels, counts = np.unique(details["labels"], return_counts=True)
    print(f"Saved {len(weights)} weights to {args.output}")
    print("GMM component populations:", dict(zip(labels.tolist(), counts.tolist())))
    print(f"weights: min={weights.min():.3f}, mean={weights.mean():.3f}, "
          f"p95={np.percentile(weights, 95):.3f}, max={weights.max():.3f}")
    report_prefix = args.report_prefix or os.path.splitext(args.output)[0] + "_provenance"
    window_report, cluster_report = write_provenance_reports(details, weights, report_prefix)
    print(f"Saved window provenance to {window_report}")
    print(f"Saved cluster/flight summary to {cluster_report}")
    plot_path = args.plot
    if plot_path is None:
        plot_path = os.path.splitext(args.output)[0] + "_dashboard.png"
    if str(plot_path).lower() != "none":
        create_dashboard(details, weights, plot_path, args.max_plot_points, args.seed)
        print(f"Saved diversity dashboard to {plot_path}")


if __name__ == "__main__":
    main()
