import os
import torch
import io, pickle
import numpy as np
from inspect import currentframe, getframeinfo
import wandb
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def save_state(out_states:dict, in_state:dict):
    for k, v in in_state.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            save_state(out_states=out_states, in_state=v)
        elif k in out_states.keys():
            out_states[k].append(v)
        else:
            out_states[k] = [v]


def _get_conf_value(conf, key, default=None):
    if hasattr(conf, key):
        return getattr(conf, key)
    try:
        return conf[key]
    except (KeyError, TypeError):
        return default


def make_dataset_save_prefix(data_root, data_name):
    dataset_root = os.path.basename(os.path.normpath(str(data_root)))
    data_name = str(data_name)
    if not dataset_root:
        return data_name
    return os.path.join(dataset_root, data_name)


def build_dataset_save_prefix(data_conf, data_name):
    if _get_conf_value(data_conf, "name") != "TLab":
        return data_name
    return make_dataset_save_prefix(_get_conf_value(data_conf, "data_root"), data_name)


def get_orientation_state(loaded_data, data_root, data_name):
    save_key = make_dataset_save_prefix(data_root, data_name)
    if save_key in loaded_data:
        return loaded_data[save_key]
    if data_name in loaded_data:
        return loaded_data[data_name]
    raise KeyError(f"Orientation data not found for {data_name!r} or {save_key!r}")


def Gaussian_noise(num_nodes, sigma_x=0.05 ,sigma_y=0.05, sigma_z=0.05):
    std = torch.stack([torch.ones(num_nodes)*sigma_x, torch.ones(num_nodes)*sigma_y, torch.ones(num_nodes)*sigma_z], dim=-1)
    return torch.normal(mean = 0, std = std)

def move_to(obj, device):
    if torch.is_tensor(obj):return obj.to(device)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj).to(device)
    else:
        raise TypeError("Invalid type for move_to", type(obj))

def qinterp(qs, t, t_int):
    qs = R.from_quat(qs.numpy())
    slerp = Slerp(t, qs)
    interp_rot = slerp(t_int).as_quat()
    return torch.tensor(interp_rot)
def interp_xyz(time, opt_time, xyz):
    intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
    intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
    intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return torch.tensor(inte_xyz)
def lookAt(dir_vec, up = torch.tensor([0.,0.,1.], dtype=torch.float64), source = torch.tensor([0.,0.,0.], dtype=torch.float64)):
    '''
    dir_vec: the tensor shall be (1)
    return the rotation matrix of the 
    '''
    if not isinstance(dir_vec, torch.Tensor):
        dir_vec = torch.tensor(dir_vec)
    def normalize(x):
        length = x.norm()
        if length< 0.005:
            length = 1
            warnings.warn('Normlization error that the norm is too small')
        return x/length
            
    zaxis = normalize(dir_vec - source)
    xaxis = normalize(torch.cross(zaxis, up))
    yaxis = torch.cross(xaxis, zaxis)

    m = torch.tensor([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [zaxis[0], zaxis[1], zaxis[2]],
    ])

    return m

def cat_state(in_state:dict):
    pop_list = []
    for k, v in in_state.items():
        if len(v[0].shape) > 2:
            in_state[k] = torch.cat(v,  dim=-2)
        else:
            pop_list.append(k)
    for k in pop_list:
        in_state.pop(k)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def write_board(writer, objs, epoch_i, header = ''):
    # writer = SummaryWriter(log_dir=conf.general.exp_dir)
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                writer.add_scalar(os.path.join(header, k), v, epoch_i)
    elif isinstance(objs, float):
        writer.add_scalar(header, v, epoch_i)

def write_wandb(header, objs, epoch_i):
    if isinstance(objs, dict):
        for k, v in objs.items():
            # Per-drive metrics are printed to the job log only; avoid creating
            # a separate W&B chart for every flight.
            if isinstance(v, float) and not k.startswith("dataset/") and not k.startswith("speed/"):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)


class DatasetLossTracker:
    """Accumulate losses and optional exact motion metrics by data drive."""

    def __init__(self, loader, confs, loss_fn, track_motion_metrics=False):
        self.dataset_names = loader.dataset.dataset_names
        self.confs = confs
        self.loss_fn = loss_fn
        self.track_motion_metrics = track_motion_metrics
        self.totals = {}
        self.speed_totals = {}
        self.motion_totals = {}
        self.speed_motion_totals = {}
        self.overall_motion_total = None
        self.speed_bins = self._get_speed_bins(confs)

    @staticmethod
    def _get_speed_bins(confs):
        """Return strictly increasing speed-bin edges in m/s."""
        edges = _get_conf_value(confs, "speed_profile_bins", [0, 1, 2, 3, 4])
        edges = [float(edge) for edge in edges]
        if not edges or edges[0] != 0.0:
            edges.insert(0, 0.0)
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("speed_profile_bins must be strictly increasing")
        return edges

    @staticmethod
    def _speed_profile_name(lower, upper=None):
        def format_edge(edge):
            return str(int(edge)) if edge.is_integer() else f"{edge:g}"

        if upper is None:
            return f">={format_edge(lower)}mps"
        return f"{format_edge(lower)}-{format_edge(upper)}mps"

    @staticmethod
    def _slice(state, mask):
        if torch.is_tensor(state):
            return state[mask].detach()
        if isinstance(state, dict):
            return {
                key: DatasetLossTracker._slice(value, mask)
                for key, value in state.items()
            }
        return state

    @staticmethod
    def _new_motion_total(prediction):
        options = {"device": prediction.device, "dtype": torch.float64}
        return {
            "count": 0,
            "squared_error": torch.zeros((), **options),
            "prediction": torch.zeros(3, **options),
            "target": torch.zeros(3, **options),
            "prediction_squared": torch.zeros(3, **options),
            "target_squared": torch.zeros(3, **options),
            "cross_product": torch.zeros(3, **options),
        }

    @classmethod
    def _update_motion_total(cls, total, prediction, target):
        prediction = prediction.detach().reshape(-1, 3).to(torch.float64)
        target = target.detach().reshape(-1, 3).to(torch.float64)
        if total is None:
            total = cls._new_motion_total(prediction)

        error = prediction - target
        total["count"] += prediction.shape[0]
        total["squared_error"] += error.square().sum()
        total["prediction"] += prediction.sum(dim=0)
        total["target"] += target.sum(dim=0)
        total["prediction_squared"] += prediction.square().sum(dim=0)
        total["target_squared"] += target.square().sum(dim=0)
        total["cross_product"] += (prediction * target).sum(dim=0)
        return total

    @staticmethod
    def _motion_metrics(total):
        count = total["count"]
        eps = 1e-8
        prediction_ss = (
            total["prediction_squared"]
            - total["prediction"].square() / count
        ).clamp_min(0.0)
        target_ss = (
            total["target_squared"] - total["target"].square() / count
        ).clamp_min(0.0)
        covariance = (
            total["cross_product"]
            - total["prediction"] * total["target"] / count
        )
        correlation_denominator = torch.sqrt(prediction_ss * target_ss)
        correlation = torch.where(
            correlation_denominator > eps,
            covariance / correlation_denominator,
            torch.zeros_like(covariance),
        ).clamp(-1.0, 1.0)
        gain = total["cross_product"] / (
            total["target_squared"] + eps
        )
        rmse = torch.sqrt(total["squared_error"] / count)
        return rmse.item(), correlation.tolist(), gain.tolist()

    def _add_motion_metrics(self, output, prefix, total):
        rmse, correlation, gain = self._motion_metrics(total)
        # Preserve the existing /loss API while making it the exact aggregate
        # RMSE rather than an average of per-batch RMSE values.
        output[f"{prefix}/loss"] = rmse
        output[f"{prefix}/rmse"] = rmse
        for index, axis in enumerate("xyz"):
            output[f"{prefix}/correlation_{axis}"] = correlation[index]
            output[f"{prefix}/gain_{axis}"] = gain[index]

    def update(self, dataset_ids, prediction, target):
        with torch.no_grad():
            net_velocity = prediction.get("net_vel") if isinstance(prediction, dict) else None
            if self.track_motion_metrics:
                if net_velocity is None or net_velocity.shape[-1] != 3:
                    raise ValueError(
                        "Motion metrics require prediction['net_vel'] and a final xyz axis"
                    )
                self.overall_motion_total = self._update_motion_total(
                    self.overall_motion_total, net_velocity, target
                )

            for dataset_id in dataset_ids.unique():
                mask = dataset_ids == dataset_id
                source = self.dataset_names[dataset_id.item()]
                loss = self.loss_fn(
                    self._slice(prediction, mask),
                    self._slice(target, mask),
                    self.confs,
                )["loss"].mean().item()
                count = mask.sum().item()
                entry = self.totals.setdefault(source, [0.0, 0])
                entry[0] += loss * count
                entry[1] += count
                if self.track_motion_metrics:
                    self.motion_totals[source] = self._update_motion_total(
                        self.motion_totals.get(source), net_velocity[mask], target[mask]
                    )

            # A sample is assigned using its mean ground-truth speed over the
            # window. The loss itself is unchanged and is recomputed on only
            # the samples belonging to that profile.
            sample_speed = torch.linalg.vector_norm(target, dim=-1)
            if sample_speed.ndim > 1:
                sample_speed = sample_speed.mean(dim=tuple(range(1, sample_speed.ndim)))

            for index, lower in enumerate(self.speed_bins):
                upper = (self.speed_bins[index + 1]
                         if index + 1 < len(self.speed_bins) else None)
                mask = sample_speed >= lower
                if upper is not None:
                    mask &= sample_speed < upper
                count = mask.sum().item()
                if not count:
                    continue
                loss = self.loss_fn(
                    self._slice(prediction, mask),
                    self._slice(target, mask),
                    self.confs,
                )["loss"].mean().item()
                profile = self._speed_profile_name(lower, upper)
                entry = self.speed_totals.setdefault(profile, [0.0, 0])
                entry[0] += loss * count
                entry[1] += count
                if self.track_motion_metrics:
                    self.speed_motion_totals[profile] = self._update_motion_total(
                        self.speed_motion_totals.get(profile),
                        net_velocity[mask],
                        target[mask],
                    )

    def overall_motion_metrics(self):
        if self.overall_motion_total is None:
            return {}
        rmse, correlation, gain = self._motion_metrics(self.overall_motion_total)
        metrics = {"loss": rmse}
        for index, axis in enumerate("xyz"):
            metrics[f"correlation_{axis}"] = correlation[index]
            metrics[f"gain_{axis}"] = gain[index]
        return metrics

    def metrics(self):
        dataset_metrics = {
            f"dataset/{source}/loss": loss_sum / count
            for source, (loss_sum, count) in self.totals.items()
            if count
        }
        speed_metrics = {
            f"speed/{profile}/loss": loss_sum / count
            for profile, (loss_sum, count) in self.speed_totals.items()
            if count
        }
        metrics = {**dataset_metrics, **speed_metrics}
        if self.track_motion_metrics:
            for source, total in self.motion_totals.items():
                self._add_motion_metrics(metrics, f"dataset/{source}", total)
            for profile, total in self.speed_motion_totals.items():
                self._add_motion_metrics(metrics, f"speed/{profile}", total)
        return metrics


def print_dataset_losses(split, metrics):
    for key, value in metrics.items():
        if key.startswith("dataset/") and key.endswith("/loss"):
            dataset_name = key[len("dataset/") : -len("/loss")]
            print(f"{split} loss [{dataset_name}]: {value:f}")
        elif key.startswith("speed/") and key.endswith("/loss"):
            speed_profile = key[len("speed/") : -len("/loss")]
            print(f"{split} loss [speed {speed_profile}]: {value:f}")
        elif key.startswith(("dataset/", "speed/")):
            group = key.split("/", 1)[0]
            name, metric = key.rsplit("/", 1)
            name = name[len(group) + 1:]
            print(f"{split} {metric} [{group} {name}]: {value:f}")

def save_ckpt(network, optimizer, scheduler, epoch_i, test_loss, conf, save_best = False):
    if epoch_i%conf.train.save_freq==conf.train.save_freq-1:
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt"%epoch_i))

    if save_best:
        print("saving the best model", test_loss)
        torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"))
    
    torch.save({
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
        }, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))


def report_hasNan(x):
    cf = currentframe().f_back
    res = torch.any(torch.isnan(x)).cpu().item()
    if res: print(f"[hasnan!] {getframeinfo(cf).filename}:{cf.f_lineno}")


def report_hasNeg(x):
    cf = currentframe().f_back
    res = torch.any(x < 0).cpu().item()
    if res: print(f"[hasneg!] {getframeinfo(cf).filename}:{cf.f_lineno}")


    
