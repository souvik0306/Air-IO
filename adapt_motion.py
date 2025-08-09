import argparse
import os
import copy



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EuRoC/motion_body.conf",
        help="config file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda or cpu"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to pretrained checkpoint (default: best_model.ckpt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of adaptation epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--trainable",
        type=str,
        default="veldecoder,velcov_decoder",
        help="comma separated list of layers to finetune",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="print basic statistics to gauge dataset diversity",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="optional path to save adapted weights",
    )
    return parser.parse_args()


def load_checkpoint(network, conf, ckpt_path, device):
    import torch
    if ckpt_path is None:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(
            ckpt_path, map_location=torch.device(device), weights_only=True
        )
        network.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded state dict {ckpt_path} in epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")


def freeze_early_modules(network):
    for name in ["cnn", "gru1", "gru2", "feature_encoder", "ori_encoder"]:
        if hasattr(network, name):
            for p in getattr(network, name).parameters():
                p.requires_grad = False
    # freeze any remaining parameters
    for p in network.parameters():
        p.requires_grad = False


def set_trainable_layers(network, layer_names):
    params = []
    for name in layer_names:
        if hasattr(network, name):
            module = getattr(network, name)
            for p in module.parameters():
                p.requires_grad = True
            params += list(module.parameters())
    return params


def build_dataloader(conf):
    import torch.utils.data as Data
    from datasets import collate_fcs, SeqeuncesMotionDataset
    dataset_conf = copy.deepcopy(conf.dataset.test)
    dataset_conf.data_list[0]["window_size"] = 200
    dataset_conf.data_list[0]["step_size"] = 20

    if "collate" in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs["base"]

    dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf)
    loader = Data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    return dataset, loader


def analyze_dataset(dataset):
    import torch
    acc = torch.cat(dataset.acc, dim=0)
    gyro = torch.cat(dataset.gyro, dim=0)
    print(
        f"Loaded {len(dataset)} windows from {len(dataset.acc)} sequences",
        flush=True,
    )
    print(f"Acceleration std: {acc.std(0).tolist()}")
    print(f"Gyro std: {gyro.std(0).tolist()}")


def main():
    args = parse_args()

    import torch
    from pyhocon import ConfigFactory
    from model import net_dict
    from model.losses import get_motion_loss
    from utils import move_to

    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf["general"]["exp_dir"] = os.path.join(conf.general.exp_dir, conf_name)

    network = net_dict[conf.train.network](conf.train).to(args.device).double()
    load_checkpoint(network, conf, args.ckpt, args.device)

    freeze_early_modules(network)
    trainable_layers = [n.strip() for n in args.trainable.split(",") if n]
    params = set_trainable_layers(network, trainable_layers)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    dataset, loader = build_dataloader(conf)
    if args.analyze:
        analyze_dataset(dataset)

    for epoch in range(args.epochs):
        network.train()
        for data, _, label in loader:
            data, label = move_to([data, label], args.device)
            rot = label["gt_rot"][:, :-1, :].Log().tensor()
            inte_state = network(data, rot)
            gt_label = network.get_label(label["gt_vel"])
            loss_state = get_motion_loss(inte_state, gt_label, conf.train)

            optimizer.zero_grad()
            loss_state["loss"].backward()
            optimizer.step()
        print(f"epoch {epoch}: loss {loss_state['loss'].item():.6f}")

    if args.save_path is not None:
        torch.save({"model_state_dict": network.state_dict()}, args.save_path)
        print(f"saved adapted model to {args.save_path}")


if __name__ == "__main__":
    main()
