import os
import sys
import torch
import argparse
import pickle
import tqdm
import numpy as np

from pyhocon import ConfigFactory
from datasets import collate_fcs, SeqeuncesMotionDataset
from model import net_dict
from utils import move_to, save_state

def inference_pt(network, loader, confs):
    network.eval()
    evaluate_states = {}
    with torch.no_grad():
        for data, _, label in tqdm.tqdm(loader):
            data, label = move_to([data, label], confs.device)
            rot = label['gt_rot'][:, :-1, :].Log().tensor()
            inte_state = network(data, rot)
            inte_state['ts'] = network.get_label(data['ts'][..., None])[0]
            save_state(evaluate_states, inte_state)
        for k, v in evaluate_states.items():
            evaluate_states[k] = torch.cat(v, dim=-2)
    return evaluate_states

def inference_onnx(onnx_session, loader, confs):
    import onnxruntime as ort
    evaluate_states = {}
    for data, _, label in tqdm.tqdm(loader):
        data_np = {k: v.cpu().numpy() for k, v in data.items()}
        rot = label['gt_rot'][:, :-1, :].Log().tensor().cpu().numpy()
        ort_inputs = {'acc': data_np['acc'], 'gyro': data_np['gyro'], 'rot': rot}
        outputs = onnx_session.run(['net_vel', 'cov'], ort_inputs)
        inte_state = {
            'net_vel': torch.from_numpy(outputs[0]),
            'cov': torch.from_numpy(outputs[1]),
            'ts': data['ts'],
        }
        save_state(evaluate_states, inte_state)
    for k, v in evaluate_states.items():
        evaluate_states[k] = torch.cat(v, dim=-2)
    return evaluate_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--model', type=str, required=True, help='path to .pt or .onnx model')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=1000, help='window size.')
    parser.add_argument('--whole', default=True, action="store_true", help='estimate the whole seq')
    args = parser.parse_args(); print(args)

    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    conf['device'] = args.device
    dataset_conf = conf.dataset.inference

    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs['base']

    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen

    # Model loading
    ext = os.path.splitext(args.model)[-1]
    if ext == '.pt':
        network = net_dict[conf.train.network](conf.train).to(args.device).double()
        state_dict = torch.load(args.model, map_location=args.device)
        network.load_state_dict(state_dict)
        run_inference = lambda loader: inference_pt(network, loader, conf.train)
    elif ext == '.onnx':
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(args.model)
        run_inference = lambda loader: inference_onnx(onnx_session, loader, conf.train)
    else:
        raise ValueError("Unsupported model format. Use .pt or .onnx")

    net_out_result = {}
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            if args.whole:
                dataset_conf["mode"] = "inference"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
            eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size,
                                                      shuffle=False, collate_fn=collate_fn, drop_last=False)
            inference_state = run_inference(eval_loader)
            if not "cov" in inference_state.keys():
                inference_state["cov"] = torch.zeros_like(inference_state["net_vel"])
            inference_state['ts'] = inference_state['ts']
            inference_state['net_vel'] = inference_state['net_vel'][0] #TODO: batch size != 1
            net_out_result[path] = inference_state

    net_result_path = os.path.join(conf.general.exp_dir, 'net_output.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
# !python inference_motion_pt_onnx.py --config path/to/your_config.conf --model path/to/your_model.pt