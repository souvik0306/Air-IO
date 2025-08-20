import os
import argparse
import pickle
import tqdm
import torch
import numpy as np
import onnxruntime as ort
from pyhocon import ConfigFactory
from datasets import collate_fcs, SeqeuncesMotionDataset

def move_to_numpy(data):
    """Convert dict of tensors to float32 NumPy arrays."""
    return {k: v.cpu().numpy().astype(np.float32) for k, v in data.items()}

def run_onnx_inference(onnx_session, loader):
    evaluate_states = {}
    for data, _, label in tqdm.tqdm(loader):
        # Prepare inputs
        data_np = move_to_numpy(data)
        rot = label['gt_rot'][:, :-1, :].Log().tensor().cpu().numpy().astype(np.float32)
        ort_inputs = {
            'acc': data_np['acc'],
            'gyro': data_np['gyro'],
            'rot': rot,
        }
        # Run ONNX inference
        outputs = onnx_session.run(['net_vel', 'cov'], ort_inputs)
        inte_state = {
            'net_vel': torch.from_numpy(outputs[0]),
            'cov': torch.from_numpy(outputs[1]),
            'ts': data['ts'],
        }
        for k, v in inte_state.items():
            if k not in evaluate_states:
                evaluate_states[k] = []
            evaluate_states[k].append(v)
    for k, v in evaluate_states.items():
        evaluate_states[k] = torch.cat(v, dim=-2)
    return evaluate_states

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--onnx', type=str, required=True, help='path to ONNX model')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=1000, help='window size.')
    parser.add_argument('--whole', default=True, action="store_true", help='estimate the whole seq')
    args = parser.parse_args(); print(args)

    conf = ConfigFactory.parse_file(args.config)
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    dataset_conf = conf.dataset.inference

    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs['base']

    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen

    onnx_session = ort.InferenceSession(args.onnx)

    net_out_result = {}
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            if args.whole:
                dataset_conf["mode"] = "inference"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf['general']['exp_dir']
            eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
            eval_loader = torch.utils.data.DataLoader(
                dataset=eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False
            )
            inference_state = run_onnx_inference(onnx_session, eval_loader)
            inference_state['ts'] = inference_state['ts']
            inference_state['net_vel'] = inference_state['net_vel'][0]  # TODO: batch size != 1
            inference_state['cov'] = inference_state['cov'][0]  # TODO: batch size != 1
            net_out_result[path] = inference_state

    net_result_path = os.path.join(conf['general']['exp_dir'], 'onnx_network.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
