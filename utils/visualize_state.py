import os

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pypose as pp
import torch


def _euroc_to_tlab_xyz(xyz):
    """Convert vectors from EuRoC frame [x, y, z] to TLab frame."""
    return torch.stack((xyz[..., 2], -xyz[..., 1], xyz[..., 0]), dim=-1)


def _save_figure(fig, out_path, **savefig_kwargs):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, **savefig_kwargs)


def visualize_motion(
    save_prefix,
    save_folder,
    outstate,
    infstate,
    ts=None,
    label="AirIO",
    save_in_flight_folder=False,
):
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
    vel_gt = torch.as_tensor(outstate["vel_gt"][0]).detach().cpu()
    net_vel = torch.as_tensor(infstate["net_vel"][0]).detach().cpu()

    velocity_length = min(len(vel_gt), len(net_vel))
    vel_gt = vel_gt[:velocity_length]
    net_vel = net_vel[:velocity_length]

    # Determine time axis. Use relative time when timestamps are absolute
    # (e.g., UNIX epoch) to avoid unreadable axis offsets such as +1.78e9.
    if ts is not None:
        t_plot = torch.as_tensor(ts).cpu().to(torch.float64)
        if t_plot.numel() > 0:
            t_plot = t_plot - t_plot[0]
        t_vel = t_plot[:velocity_length:50]
    else:
        t_plot = torch.arange(velocity_length, dtype=torch.float64)
        t_vel = t_plot[::50]

    velocity_axes = [
        ("x", "Velocity X"),
        ("y", "Velocity Y"),
        ("z", "Velocity Z"),
    ]

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

    for axis_index, (axis_suffix, axis_title) in enumerate(velocity_axes):
        fig_vel, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(
            t_vel,
            net_vel[::50, axis_index],
            color="blue",
            label=label,
            linewidth=1.2,
        )
        ax.plot(
            t_vel,
            vel_gt[::50, axis_index],
            color="red",
            label="GT",
            linewidth=1.2,
        )

        ax.set_title(f"{axis_title} - Original GT Frame")
        ax.set_xlabel("time")
        ax.set_ylabel(f"{axis_title} (m/s, original GT frame)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="major", linewidth=0.8, alpha=0.45)
        ax.grid(True, which="minor", linewidth=0.4, alpha=0.25)
        ax.legend()

        fig_vel.tight_layout()
        if save_in_flight_folder:
            out_path = os.path.join(save_folder, save_prefix, f"velocity_{axis_suffix}.png")
        else:
            out_path = os.path.join(save_folder, f"{save_prefix}_velocity_{axis_suffix}.png")

        _save_figure(fig_vel, out_path, dpi=600)
        plt.close(fig_vel)

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
        _save_figure(
            fig,
            os.path.join(save_folder, save_prefix + "_orientation_compare.png"),
            dpi=600,
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
        _save_figure(
            fig,
            os.path.join(save_folder, save_prefix + ".png"),
            dpi=600,
        )
    plt.show()
    plt.close()
