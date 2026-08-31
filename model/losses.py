import torch
import torch.nn.functional as F
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan
import numpy as np

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def weighted_hybrid_motion_loss(pred, targ, delta=0.05,
                                speed_start=2.0, speed_range=2.0,
                                speed_gain=2.0, magnitude_weight=0.0,
                                mse_start=2.5, mse_end=3.5):
    """Blend velocity Huber into MSE as the target speed increases.

    The speed weights are normalized over the batch and temporal dimensions, so
    enabling them does not change the average scale of the velocity objective.
    The MSE fraction is zero through ``mse_start``, one from ``mse_end``, and
    linearly interpolated between them.
    """
    dist = pred - targ
    speed = torch.linalg.vector_norm(targ, dim=-1)
    ramp = ((speed - speed_start) / speed_range).clamp(0.0, 1.0)
    weights = 1.0 + speed_gain * ramp.square()
    weights = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)

    # At 3.0 m/s, defaults give an equal Huber/MSE blend.
    mse_fraction = ((speed - mse_start) / (mse_end - mse_start)).clamp(0.0, 1.0)
    huber = F.huber_loss(pred, targ, reduction='none', delta=delta).mean(dim=-1)
    mse = F.mse_loss(pred, targ, reduction='none').mean(dim=-1)
    hybrid = (1.0 - mse_fraction) * huber + mse_fraction * mse
    loss = (weights * hybrid).mean()

    if magnitude_weight:
        pred_speed = torch.linalg.vector_norm(pred, dim=-1)
        magnitude_loss = F.huber_loss(
            pred_speed, speed, reduction='mean', delta=delta)
        loss = loss + magnitude_weight * magnitude_loss

    return loss, dist

def get_motion_loss(inte_state, label, confs):
    ## The state loss for evaluation
    loss, cov_loss = 0, {}
    loss_fc = loss_fc_list[confs.loss]
    
    if "speed_weighted_loss" in confs and confs.speed_weighted_loss:
        vel_loss, vel_dist = weighted_hybrid_motion_loss(
            inte_state['net_vel'], label,
            delta=confs.huber_delta,
            speed_start=confs.speed_weight_start,
            speed_range=confs.speed_weight_range,
            speed_gain=confs.speed_weight_gain,
            magnitude_weight=confs.mag_weight,
            mse_start=confs.mse_start_speed,
            mse_end=confs.mse_end_speed,
        )
    else:
        vel_loss, vel_dist = motion_loss_(loss_fc, inte_state['net_vel'],label)

    # Apply the covariance loss
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss += confs.weight * vel_loss
    return {'loss':loss, 'cov_loss':cov_loss}


def get_motion_RMSE(inte_state, label, confs):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())
    cov_loss = 0
    dist = (inte_state['net_vel'] - label)
    dist = torch.mean(dist,dim=-2)
    loss = _RMSE(dist)[None,...]
    
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()
    
    return {'loss': loss, 
            'dist': dist.norm(dim=-1).mean(),
            'cov_loss': cov_loss}
