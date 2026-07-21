import csv
import os

import torch


def save_flight_velocity_csv(save_dir, flight_name, time_tensor, vel_tensor):
    os.makedirs(save_dir, exist_ok=True)

    time_np = torch.as_tensor(time_tensor).detach().cpu().numpy().reshape(-1)
    vel_np = torch.as_tensor(vel_tensor).detach().cpu().numpy()

    length = min(len(time_np), len(vel_np))
    time_np = time_np[:length]
    vel_np = vel_np[:length]

    out_path = os.path.join(save_dir, f"{flight_name}_velocity.csv")
    with open(out_path, "w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["time", "vel_x", "vel_y", "vel_z"])
        for t, v in zip(time_np, vel_np):
            writer.writerow([float(t), float(v[0]), float(v[1]), float(v[2])])

    print(f"saved velocity csv: {out_path}")
