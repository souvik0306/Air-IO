# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import argparse
import pypose as pp
import torch
import torch.utils.data as Data

from pyhocon import ConfigFactory
from datasets import SeqInfDataset, SeqDataset, imu_seq_collate

from utils import CPU_Unpickler, integrate
from utils import build_dataset_save_prefix
from utils.visualize_state import visualize_rotations
import pickle


def integrate_dataset_orientation(dataset, init, gravity, device):
    """Integrate one full raw IMU sequence from its initial state."""
    seq_len = dataset.seqlen
    integrator = pp.module.IMUPreintegrator(
        init['pos'], init['rot'], init['vel'], gravity=gravity, reset=False
    ).to(device).double()

    state = integrator(
        dt=dataset.data["dt"][:seq_len][None].to(device),
        gyro=dataset.data["gyro"][:seq_len][None].to(device),
        acc=dataset.data["acc"][:seq_len][None].to(device),
    )
    orientations = torch.cat(
        [init["rot"][None, ...].cpu(), state["rot"].cpu()], dim=-2
    )
    return orientations[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataconf", type=str, help="the configuration of the dataset")
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Optional directory of AirIMU outputs. If omitted, only inte_rot is saved.",
    )
    parser.add_argument("--savedir",type=str,default = "./result/loss_result/orientations",help = "save directory")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu, Default is cuda:0")

    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    if "rot_type" in dataset_conf and dataset_conf.rot_type not in [None, "None"]:
        print(
            f"Ignoring rot_type={dataset_conf.rot_type} while generating "
            "orientation_output.pickle."
        )
        dataset_conf["rot_type"] = None

    # Load the AirIMU network results when requested. Integration-only orientation
    # generation does not need an AirIMU net_output.pickle.
    inference_state_load = None
    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")
    else:
        print("No AirIMU --exp provided; saving raw IMU integrated orientation only.")
    
    # Create the output folder
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)
    save_states = {}

    # Process each dataset
    for data_conf in dataset_conf.data_list:
        for data_name in data_conf.data_drive:
            print(f"dataset: {data_conf.name}, sequence: {data_name}")
            save_key = build_dataset_save_prefix(data_conf, data_name)
            save_cur_state = {}
            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=200, step_size=200, drop_last=False, conf = dataset_conf)
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()

            if inference_state_load is not None:
                # DataLoader for the raw IMU data
                loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
                # DataLoader for the AirIMU corrected data
                inference_key = save_key if save_key in inference_state_load else data_name
                inference_state = inference_state_load[inference_key]
                dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device = args.device, name = data_conf.name,duration=200, step_size=200, drop_last=False, conf = dataset_conf)
                infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1,
                                            collate_fn=imu_seq_collate,
                                            shuffle=False, drop_last=False)
            else:
                loader = None
                infloader = None

            if loader is not None:
                integrator_outstate = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'], init['vel'],gravity=gravity,
                    reset=False
                ).to(args.device).double()
                integrator_infstate = pp.module.IMUPreintegrator(
                    init['pos'], init['rot'], init['vel'], gravity = gravity,
                    reset=False
                ).to(args.device).double()
            
                # Integrate the raw data and the AirIMU corrected data.
                outstate = integrate(
                    integrator_outstate, loader, init,
                    device=args.device, gtinit=False, save_full_traj=True,
                    use_gt_rot=False
                )
                infstate = integrate(
                    integrator_infstate, infloader, init,
                    device=args.device, gtinit=False, save_full_traj=True,
                    use_gt_rot=False
                )
                inte_rot = outstate['orientations'][0]
            else:
                inte_rot = integrate_dataset_orientation(
                    dataset, init, gravity, args.device
                )
                outstate = None
                infstate = None
            
            # Save the results
            save_cur_state["inte_rot"] = inte_rot
            if infstate is not None:
                save_cur_state["airimu_rot"] = infstate['orientations'][0]
            save_states[save_key] = save_cur_state

            # Visualize the results 
            if data_conf.name == "BlackBird":
                plot_prefix = os.path.dirname(data_name).split('/')[1]
            else:
                plot_prefix = save_key

            if outstate is not None:
                inf_rot = infstate['orientations'][0] if infstate is not None else None
                visualize_rotations(plot_prefix,outstate['orientations_gt'][0],outstate['orientations'][0],inf_rot=inf_rot,save_folder=folder)

        net_result_path = os.path.join(folder, 'orientation_output.pickle')
        print("save orientation, ", net_result_path)
        with open(net_result_path, 'wb') as handle:
            pickle.dump(save_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
