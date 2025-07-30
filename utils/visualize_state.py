import os

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch

def visualize_motion(save_prefix, save_folder, outstate, infstate, ts=None, label="AirIO"):
    """Visualize trajectory, velocity and position over time.

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
    ### visualize gt&netoutput velocity, 2d trajectory.
    gt_x, gt_y, gt_z = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)

    v_gt_x, v_gt_y, v_gt_z = torch.split(outstate["vel_gt"][0][::50, :].cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(infstate["net_vel"][0][::50, :].cpu(), 1, dim=1)

    # Determine time axis
    if ts is not None:
        t_plot = torch.as_tensor(ts).cpu()
        t_vel = t_plot[::50]
    else:
        t_plot = torch.arange(len(gt_x))
        t_vel = t_plot[::50]

    # Poses predicted by integrate_pos start from the first delta time step,
    # hence they correspond to timestamps ``ts[1:]`` when ``ts`` is provided.
    t_pos = t_plot[1:]
    gt_x = gt_x[1:]
    gt_y = gt_y[1:]
    gt_z = gt_z[1:]

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(3, 3)

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_vx = fig.add_subplot(gs[0, 1])
    ax_vy = fig.add_subplot(gs[1, 1])
    ax_vz = fig.add_subplot(gs[2, 1])
    ax_px = fig.add_subplot(gs[0, 2])
    ax_py = fig.add_subplot(gs[1, 2])
    ax_pz = fig.add_subplot(gs[2, 2])

    # Visualize trajectory (X vs Y)
    ax_traj.plot(airTraj_x, airTraj_y, label=label)
    ax_traj.plot(gt_x, gt_y, label="Ground Truth")
    ax_traj.set_xlabel("X axis")
    ax_traj.set_ylabel("Y axis")
    ax_traj.legend()

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

    # Visualize position over time
    ax_px.plot(t_pos, airTraj_x, label=label)
    ax_px.plot(t_pos, gt_x, label="Ground Truth")
    ax_px.set_ylabel("x")

    ax_py.plot(t_pos, airTraj_y, label=label)
    ax_py.plot(t_pos, gt_y, label="Ground Truth")
    ax_py.set_ylabel("y")

    ax_pz.plot(t_pos, airTraj_z, label=label)
    ax_pz.plot(t_pos, gt_z, label="Ground Truth")
    ax_pz.set_ylabel("z")

    for ax in [ax_vx, ax_vy, ax_vz, ax_px, ax_py, ax_pz]:
        ax.set_xlabel("time")
        ax.legend()

    fig.tight_layout()
    save_prefix += "_state.png"
    plt.savefig(os.path.join(save_folder, save_prefix), dpi=600)
    plt.close()

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
            os.path.join(save_folder, save_prefix + "_orientation_compare.png"), dpi=300
        )
    plt.show()
    plt.close()


def visualize_velocity(save_prefix, gtstate, outstate, refstate=None, save_folder=None):
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
            os.path.join(save_folder, save_prefix + ".png"), dpi=300
        )
    plt.show()
    plt.close()