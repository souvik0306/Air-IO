import os
import torch
import argparse
from pyhocon import ConfigFactory
from model import net_dict
from datasets import collate_fcs, SeqeuncesMotionDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='path to .ckpt file')
    parser.add_argument('--onnx', type=str, required=True, help='output .onnx file path')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for dummy input')
    parser.add_argument('--seqlen', type=int, default=1000, help='window size for dummy input')
    args = parser.parse_args()

    # Load config and model
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    model = net_dict[conf.train.network](conf.train).to(args.device).double()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Prepare dataset and loader to get a real batch
    dataset_conf = conf.dataset.inference
    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs['base']
    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen
    data_conf = dataset_conf.data_list[0]
    path = data_conf.data_drive[0]
    eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size,
                                              shuffle=False, collate_fn=collate_fn, drop_last=False)
    # Get one batch
    data, _, label = next(iter(eval_loader))
    data, label = data, label
    # Prepare dummy inputs
    dummy_data = data
    dummy_rot = label['gt_rot'][:, :-1, :].Log().tensor()

    # Move to device and double
    dummy_data = {k: v.to(args.device).double() for k, v in dummy_data.items()}
    dummy_rot = dummy_rot.to(args.device).double()

    # If your model expects a dict for data, you may need to adapt this part
    # For ONNX export, flatten dict input if needed
    # Example: If your model expects (data, rot)
    torch.onnx.export(
        model,
        (dummy_data, dummy_rot),
        args.onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['data', 'rot'],
        output_names=['output'],
        dynamic_axes={'data': {0: 'batch'}, 'rot': {0: 'batch'}}
    )
    print(f"Exported ONNX model to {args.onnx}")
    
    
# !python convert_to_onnx.py --config configs/EuRoC/motion_body.conf --ckpt path/to/best_model.ckpt --onnx path/to/output_model.onnx --device cuda:0 --batch_size 1 --seqlen 1000