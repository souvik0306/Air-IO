# output the trajectory comparisons between NN, EKF and ground truth
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import argparse
import numpy as np
import torch

from pyhocon import ConfigFactory
from datasets import SeqDataset
from utils import CPU_Unpickler, interp_xyz
from utils.velocity_integrator import Velocity_Integrator, integrate_pos
from utils.visualize_state import visualize_nn_ekf_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--exp", type=str, default="experiments/euroc/motion_body", help="Path for AirIO netoutput")
    parser.add_argument("--seqlen", type=int, default=1000, help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir", type=str, default="./EKFresult/loss_result", help="Directory where EKF results are saved")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")
    args = parser.parse_args()

    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference

    net_result_path = os.path.join(args.exp, "net_output.pickle")
    if os.path.isfile(net_result_path):
        with open(net_result_path, "rb") as handle:
            inference_state_load = CPU_Unpickler(handle).load()
    else:
        raise Exception(f"Unable to load the network result: {net_result_path}")

    for data_conf in dataset_conf.data_list:
        for data_name in data_conf.data_drive:
            dataset = SeqDataset(
                data_conf.data_root,
                data_name,
                args.device,
                name=data_conf.name,
                duration=args.seqlen,
                step_size=args.seqlen,
                drop_last=False,
                conf=dataset_conf,
            )

            init = dataset.get_init_value()
            inference_state = inference_state_load[data_name]
            gt_ts = dataset.data["time"]
            vel_ts = inference_state["ts"]

            if "coordinate" in dataset_conf.keys():
                if dataset_conf["coordinate"] == "body_coord":
                    rotation = dataset.data["gt_orientation"]
                    net_vel = interp_xyz(gt_ts, vel_ts[:, 0], inference_state["net_vel"])
                    net_vel = rotation * net_vel
                elif dataset_conf["coordinate"] == "glob_coord":
                    net_vel = interp_xyz(gt_ts, vel_ts[:, 0], inference_state["net_vel"])
            else:
                net_vel = interp_xyz(gt_ts, vel_ts[:, 0], inference_state["net_vel"])

            dt = gt_ts[1:] - gt_ts[:-1]
            data_inte = {"vel": net_vel, "dt": dt}
            integrator_vel = Velocity_Integrator(init["pos"]).to(args.device).double()
            inf_outstate = integrate_pos(
                integrator_vel,
                data_inte,
                init,
                dataset,
                device=args.device,
            )

            gt_pos = dataset.data["gt_translation"]

            if data_conf.name == "BlackBird":
                save_prefix = os.path.dirname(data_name).split('/')[1]
            else:
                save_prefix = data_name

            ekf_file = os.path.join(args.savedir, f"{save_prefix}_ekf_result.npy")
            if not os.path.isfile(ekf_file):
                raise FileNotFoundError(f"EKF result not found: {ekf_file}")
            ekf_result = np.load(ekf_file)
            ekf_pos = torch.tensor(ekf_result[:, 6:9])

            # align time with network output
            ts = gt_ts[1:]
            gt_pos = gt_pos[1:]
            nn_pos = inf_outstate["poses"][0]
            ekf_pos = ekf_pos[1:]

            visualize_nn_ekf_motion(save_prefix, args.savedir, gt_pos, nn_pos, ekf_pos, ts)
