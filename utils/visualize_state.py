import os

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch


def _euroc_to_tlab_xyz(xyz):
    """Convert vectors from EuRoC frame [x, y, z] to TLab frame."""
    return torch.stack((xyz[..., 2], -xyz[..., 1], xyz[..., 0]), dim=-1)

def visualize_motion(save_prefix, save_folder, outstate, infstate, ts=None, label="AirIO"):
    """Visualize velocity over time.

    Parameters
    ----------
    save_prefix : str
        Prefix for the saved figure name.
    save_folder : str
        Directory where the figure will be saved.
    outstate : dict
        Output state from integration containing ground truth data.
    infstate : dict
        Predicted state from AirIO network.
    ts : Tensor or ndarray, optional
        1-D timestamps corresponding to trajectory samples. When ``None`` the
        index of the samples will be used.
    label : str, default "AirIO"
        Label for the predicted trajectory in the plots.
    """
    ### visualize gt&netoutput velocity.
    # Position/trajectory plotting intentionally disabled.
    # Keep the original extraction logic here for quick re-enable if needed.
    # gt_x, gt_y, gt_z = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    # airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)

    # Position-vs-time plotting also disabled.
    # t_pos = t_plot[1:]
    # gt_x = gt_x[1:]
    # gt_y = gt_y[1:]
    # gt_z = gt_z[1:]
    vel_gt_tlab = _euroc_to_tlab_xyz(outstate["vel_gt"][0].cpu())
    net_vel_tlab = _euroc_to_tlab_xyz(infstate["net_vel"][0].cpu())
    v_gt_x, v_gt_y, v_gt_z = torch.split(vel_gt_tlab[::50, :], 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(net_vel_tlab[::50, :], 1, dim=1)

    # Determine time axis. Use relative time when timestamps are absolute
    # (e.g., UNIX epoch) to avoid unreadable axis offsets such as +1.78e9.
    if ts is not None:
        t_plot = torch.as_tensor(ts).cpu().to(torch.float64)
        if t_plot.numel() > 0:
            t_plot = t_plot - t_plot[0]
        t_vel = t_plot[::50]
    else:
        t_plot = torch.arange(outstate["vel_gt"][0].shape[0], dtype=torch.float64)
        t_vel = t_plot[::50]

    fig_vel = plt.figure(figsize=(14, 6))
    gs_vel = GridSpec(3, 1)

    ax_vx = fig_vel.add_subplot(gs_vel[0, 0])
    ax_vy = fig_vel.add_subplot(gs_vel[1, 0])
    ax_vz = fig_vel.add_subplot(gs_vel[2, 0])

    # Visualize velocity
    ax_vx.plot(t_vel, airVel_x, label=label)
    ax_vx.plot(t_vel, v_gt_x, label="Ground Truth")
    ax_vx.set_ylabel("v_x")

    ax_vy.plot(t_vel, airVel_y, label=label)
    ax_vy.plot(t_vel, v_gt_y, label="Ground Truth")
    ax_vy.set_ylabel("v_y")

    ax_vz.plot(t_vel, airVel_z, label=label)
    ax_vz.plot(t_vel, v_gt_z, label="Ground Truth")
    ax_vz.set_ylabel("v_z")

    for ax in [ax_vx, ax_vy, ax_vz]:
        ax.set_xlabel("time")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend()

    # Disabled position figure (kept as reference):
    # fig_pos = plt.figure(figsize=(10, 6))
    # gs = GridSpec(3, 3)
    # ax_traj = fig_pos.add_subplot(gs[:, :])
    # ax_traj.plot(airTraj_x, airTraj_y, label=label)
    # ax_traj.plot(gt_x, gt_y, label="Ground Truth")
    # ax_traj.set_xlabel("X axis")
    # ax_traj.set_ylabel("Y axis")
    # ax_traj.legend()
    # fig_pos.tight_layout()
    # plt.savefig(os.path.join(save_folder, save_prefix + "_position.png"), dpi=600)
    # plt.close(fig_pos)

    fig_vel.tight_layout()
    plt.savefig(os.path.join(save_folder, save_prefix + "_velocity.png"), dpi=600)
    plt.close(fig_vel)

WINDOWS = {
    "flight_1": (34.20, 37.20),
    "flight_2": (52.02, 55.02),
    "flight_3": (47.95, 50.95),
    "flight_4": (36.06, 39.06),
    "flight_5": (31.88, 34.88),
    "flight_6": (36.24, 39.24),
    "flight_7": (33.60, 36.60),
    "flight_8": (46.23, 49.23),
    "flight_9": (35.43, 38.43),
    "flight_10": (39.39, 42.39),
    "flight_11": (36.87, 39.87),
    "flight_12": (33.18, 36.18),
    "flight_13": (48.96, 51.96),
}

def visualize_window_results(
    save_prefix,
    save_folder,
    outstate,
    infstate,
    ts,
    label="AirIO",
):
    """
    Plot velocity only within the predefined three-second evaluation window.
    """

    import re

    # Handles names such as:
    # flight_1
    # /path/to/flight_1
    # flight_1/something
    prefix_string = str(save_prefix).replace("\\", "/")

    match = re.search(r"flight_\d+", prefix_string)

    if match is None:
        print(
            f"[Window plot] Could not extract flight name from: "
            f"{save_prefix!r}"
        )
        return

    flight_name = match.group(0)

    if flight_name not in WINDOWS:
        print(
            f"[Window plot] No predefined window for {flight_name}. "
            f"Available names: {list(WINDOWS.keys())}"
        )
        return

    start_t, end_t = WINDOWS[flight_name]

    # Convert timestamps to CPU tensor.
    ts = torch.as_tensor(ts).detach().cpu().flatten().to(torch.float64)

    if ts.numel() < 2:
        print(f"[Window plot] Invalid timestamps for {flight_name}.")
        return

    # The predefined windows are relative to the beginning of the flight.
    relative_ts = ts - ts[0]

    print(
        f"[Window plot] {flight_name}: timestamp range "
        f"{relative_ts[0].item():.3f} to "
        f"{relative_ts[-1].item():.3f} s; requested window "
        f"{start_t:.3f} to {end_t:.3f} s"
    )

    # Load tensors and remove batch dimension.
    # Position tensors are intentionally not used while position plotting
    # is disabled, but keep the original lines as comments for reuse.
    # gt_pos = _euroc_to_tlab_xyz(outstate["poses_gt"][0].detach().cpu())
    # pred_pos = _euroc_to_tlab_xyz(infstate["poses"][0].detach().cpu())

    gt_vel = _euroc_to_tlab_xyz(outstate["vel_gt"][0].detach().cpu())
    pred_vel = _euroc_to_tlab_xyz(infstate["net_vel"][0].detach().cpu())

    print(
        f"[Window plot] Lengths: ts={len(ts)}, "
        f"gt_vel={len(gt_vel)}, pred_vel={len(pred_vel)}"
    )

    # ---------------------------------------------------------
    # Velocity alignment
    # ---------------------------------------------------------
    velocity_length = min(
        len(relative_ts),
        len(gt_vel),
        len(pred_vel),
    )

    vel_time = relative_ts[:velocity_length]
    gt_vel = gt_vel[:velocity_length]
    pred_vel = pred_vel[:velocity_length]

    vel_mask = (
        (vel_time >= start_t)
        & (vel_time <= end_t)
    )

    if vel_mask.sum().item() < 2:
        print(
            f"[Window plot] No velocity samples found for {flight_name} "
            f"inside {start_t:.2f}–{end_t:.2f} s."
        )
        return

    vel_time = vel_time[vel_mask]
    gt_vel = gt_vel[vel_mask]
    pred_vel = pred_vel[vel_mask]

    # Downsample only for plotting.
    plot_stride = 1

    vel_time_plot = vel_time[::plot_stride]
    gt_vel_plot = gt_vel[::plot_stride]
    pred_vel_plot = pred_vel[::plot_stride]

    # Disabled position alignment (kept as reference):
    # pred_pos_time = relative_ts[1:]
    # position_length = min(
    #     len(pred_pos_time),
    #     len(pred_pos),
    #     max(len(gt_pos) - 1, 0),
    # )
    # pred_pos_time = pred_pos_time[:position_length]
    # pred_pos = pred_pos[:position_length]
    # gt_pos = gt_pos[1:1 + position_length]
    # pos_mask = (
    #     (pred_pos_time >= start_t)
    #     & (pred_pos_time <= end_t)
    # )
    # if pos_mask.sum().item() < 2:
    #     print(
    #         f"[Window plot] No position samples found for {flight_name} "
    #         f"inside {start_t:.2f}-{end_t:.2f} s."
    #     )
    #     return
    # pred_pos = pred_pos[pos_mask]
    # gt_pos = gt_pos[pos_mask]

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    window_folder = os.path.join(
        save_folder,
        "window_results",
    )
    os.makedirs(window_folder, exist_ok=True)

    # ---------------------------------------------------------
    # Velocity
    # ---------------------------------------------------------
    # Disabled position plot (kept as reference):
    # fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
    # ax_pos.plot(pred_pos[:, 0], pred_pos[:, 1], label=label)
    # ax_pos.plot(gt_pos[:, 0], gt_pos[:, 1], label="Ground Truth")
    # ax_pos.set_xlabel("X position (m)")
    # ax_pos.set_ylabel("Y position (m)")
    # ax_pos.set_title(
    #     f"{flight_name}: horizontal trajectory ({start_t:.2f}-{end_t:.2f} s)"
    # )
    # ax_pos.grid(True)
    # ax_pos.legend()
    # ax_pos.axis("equal")
    # fig_pos.tight_layout()
    # position_path = os.path.join(window_folder, f"{flight_name}_window_position.png")
    # fig_pos.savefig(position_path, dpi=600, bbox_inches="tight")
    # plt.close(fig_pos)

    fig_vel, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    velocity_labels = ["v_x", "v_y", "v_z"]

    for axis_index, axis in enumerate(axes):
        axis.plot(
            vel_time_plot,
            pred_vel_plot[:, axis_index],
            label=label,
        )
        axis.plot(
            vel_time_plot,
            gt_vel_plot[:, axis_index],
            label="Ground Truth",
        )

        axis.set_ylabel(f"{velocity_labels[axis_index]} (m/s)")
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time from flight start (s)")

    fig_vel.suptitle(
        f"{flight_name}: velocity "
        f"({start_t:.2f}–{end_t:.2f} s)"
    )

    fig_vel.tight_layout()

    velocity_path = os.path.join(
        window_folder,
        f"{flight_name}_window_velocity.png",
    )

    fig_vel.savefig(
        velocity_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig_vel)

    print(f"[Window plot] Saved velocity plot: {velocity_path}")
    print(
        f"[Window plot] Absolute output directory: "
        f"{os.path.abspath(window_folder)}"
    )

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot=None, save_folder=None):
    gt_euler = np.unwrap(pp.SO3(gt_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
    outstate_euler = np.unwrap(pp.SO3(out_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi

    legend_list = ["roll", "pitch","yaw"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Orientation Comparison")
    for i in range(3):
        axs[i].plot(outstate_euler[:, i], color="b", linewidth=0.9)
        axs[i].plot(gt_euler[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["raw_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)

    if inf_rot is not None:
        infstate_euler = np.unwrap(pp.SO3(inf_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
        for i in range(3):
            axs[i].plot(infstate_euler[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                    "raw_" + legend_list[i],
                    "gt_" + legend_list[i],
                    "AirIMU_" + legend_list[i],
                ]
            )
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + "_orientation_compare.png"), dpi=600
        )
    plt.show()
    plt.close()


def visualize_velocity(save_prefix, gtstate, outstate, refstate=None, save_folder=None):
    gtstate = _euroc_to_tlab_xyz(torch.as_tensor(gtstate).detach().cpu())
    outstate = _euroc_to_tlab_xyz(torch.as_tensor(outstate).detach().cpu())
    if refstate is not None:
        refstate = _euroc_to_tlab_xyz(torch.as_tensor(refstate).detach().cpu())

    legend_list = ["x", "y", "z"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Velocity Comparison")
    for i in range(3):
        axs[i].plot(outstate[:, i], color="b", linewidth=0.9)
        axs[i].plot(gtstate[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["AirIO_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)
    
    if refstate is not None:
        for i in range(3):
            axs[i].plot(refstate[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                "AirIO_" + legend_list[i], 
                "gt_" + legend_list[i],
                "IOnet" + legend_list[i],
                ]
            )

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + ".png"), dpi=600
        )
    plt.show()
    plt.close()