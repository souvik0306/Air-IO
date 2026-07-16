import os
import pickle

import numpy as np
import pypose as pp
import torch

from .dataset import Sequence


class TLab(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81007,
        remove_g=False,
        imu_file="imu.csv",
        gt_file="data.csv",
        align_tolerance=1e-4,
        **kwargs,
    ):
        super(TLab, self).__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.data = {}
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)
        self.imu_file = imu_file
        self.gt_file = gt_file
        self.align_tolerance = align_tolerance

        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path)
        self.load_gt(data_path)
        self.use_aligned_ground_truth()

        self.data["time"] = torch.tensor(self.data["time"], dtype=torch.double)
        self.data["gt_time"] = torch.tensor(self.data["gt_time"], dtype=torch.double)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        self.data["gyro"] = torch.tensor(self.data["gyro"], dtype=torch.double)
        self.data["acc"] = torch.tensor(self.data["acc"], dtype=torch.double)

        self.set_orientation(rot_path, data_name, rot_type)
        self.update_coordinate(coordinate, mode)
        self.remove_gravity(remove_g)

    def get_length(self):
        return self.data["time"].shape[0]

    def load_imu(self, folder):
        """Load TLab IMU samples with the fixed schema used by imu.csv."""
        path = os.path.join(folder, self.imu_file)
        imu_data = self.read_csv(
            path, ["time", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
        )
        self.data["time"] = imu_data["time"]
        self.data["gyro"] = np.column_stack(
            [imu_data["gyro_x"], imu_data["gyro_y"], imu_data["gyro_z"]]
        )
        self.data["acc"] = np.column_stack(
            [imu_data["acc_x"], imu_data["acc_y"], imu_data["acc_z"]]
        )

    def load_gt(self, folder):
        """Load aligned TLab ground truth pose, quaternion, and velocity."""
        path = os.path.join(folder, self.gt_file)
        gt_data = self.read_csv(
            path,
            [
                "time",
                "pos_x",
                "pos_y",
                "pos_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "vel_x",
                "vel_y",
                "vel_z",
            ],
        )
        self.data["gt_time"] = gt_data["time"]
        self.data["pos"] = np.column_stack(
            [gt_data["pos_x"], gt_data["pos_y"], gt_data["pos_z"]]
        )
        self.data["quat"] = self.normalize_quat(
            np.column_stack(
                [
                    gt_data["quat_w"],
                    gt_data["quat_x"],
                    gt_data["quat_y"],
                    gt_data["quat_z"],
                ]
            )
        )
        self.data["velocity"] = np.column_stack(
            [gt_data["vel_x"], gt_data["vel_y"], gt_data["vel_z"]]
        )

    def use_aligned_ground_truth(self):
        """Use GT directly because TLab IMU and GT rows are already time-aligned."""
        if self.data["time"].shape[0] != self.data["gt_time"].shape[0]:
            raise ValueError(
                f"Aligned TLab mode expects the same number of IMU and GT rows for "
                f"{self.data_name}, got {self.data['time'].shape[0]} and "
                f"{self.data['gt_time'].shape[0]}."
            )

        max_time_error = np.max(np.abs(self.data["time"] - self.data["gt_time"]))
        if max_time_error > self.align_tolerance:
            raise ValueError(
                f"Aligned TLab mode expects matching timestamps for {self.data_name}; "
                f"max difference is {max_time_error:.6g}s. Increase align_tolerance "
                "only if this is expected."
            )

        self.data["gt_translation"] = torch.tensor(
            self.data["pos"], dtype=torch.double
        )
        self.data["velocity"] = torch.tensor(self.data["velocity"], dtype=torch.double)
        self.data["gt_orientation"] = pp.SO3(
            torch.tensor(self.wxyz_to_xyzw(self.data["quat"]), dtype=torch.double)
        )

    @staticmethod
    def read_csv(path, required_columns):
        """Read a fixed-schema TLab CSV and drop rows with non-finite values."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        data = np.genfromtxt(
            path,
            dtype=float,
            delimiter=",",
            names=True,
            comments=None,
            encoding="utf-8",
        )
        if data.ndim == 0:
            data = data.reshape(1)
        missing_columns = [
            column for column in required_columns if column not in data.dtype.names
        ]
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {missing_columns}"
            )
        rows = np.column_stack([data[column] for column in required_columns])
        return data[np.all(np.isfinite(rows), axis=1)]

    def normalize_quat(self, quat):
        """Keep rotation quaternions unit-length before constructing SO3 objects."""
        quat = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError(f"Found zero-norm quaternion in {self.gt_file}.")
        return quat / norm

    @staticmethod
    def wxyz_to_xyzw(quat):
        """Convert TLab/EuRoC-style wxyz quaternions to pypose/scipy xyzw."""
        rot = np.zeros_like(quat)
        rot[:, 3] = quat[:, 0]
        rot[:, :3] = quat[:, 1:]
        return rot

    def update_coordinate(self, coordinate, mode):
        """Match EuRoC behavior for global/body-coordinate training targets."""
        if coordinate is None:
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = (
                        self.data["gt_orientation"].Inv() @ self.data["velocity"]
                    )
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise e

    def set_orientation(self, exp_path, data_name, rotation_type):
        """Optionally replace GT orientation with AirIMU or preintegration output."""
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, "rb") as file:
                loaded_data = pickle.load(file)

            state = loaded_data[data_name]
            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state["airimu_rot"]
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state["inte_rot"]
            else:
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            print(f"The file {exp_path} was not found.")
            raise

    def remove_gravity(self, remove_g):
        """Optionally remove gravity from acceleration if enabled in the config."""
        if remove_g is True:
            print("gravity has been removed")
            self.data["acc"] -= self.g_vector
