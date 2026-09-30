import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan
import numpy as np

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def normalized_squared_velocity_loss(pred, targ, s0=0.25):
    """Squared velocity-vector error normalized by target speed."""
    dist = pred - targ
    speed = torch.linalg.vector_norm(targ, dim=-1)
    loss = (dist.square().mean(dim=-1) / (s0 + speed)).mean()
    return loss, dist

def concordance_correlation_loss(pred, targ, tau=0.1, eps=1e-8):
    """Variance-weighted, per-axis CCC loss over each window's time axis."""
    if pred.shape != targ.shape:
        raise ValueError(
            f"pred and targ must have the same shape, got {pred.shape} and {targ.shape}"
        )
    if pred.ndim < 2 or pred.shape[-1] != 3:
        raise ValueError(
            "pred and targ must have shape (..., time, 3) for x/y/z velocity"
        )

    pred_mean = pred.mean(dim=-2, keepdim=True)
    targ_mean = targ.mean(dim=-2, keepdim=True)
    pred_centered = pred - pred_mean
    targ_centered = targ - targ_mean
    pred_var = pred_centered.square().mean(dim=-2)
    targ_var = targ_centered.square().mean(dim=-2)
    covariance = (pred_centered * targ_centered).mean(dim=-2)
    mean_difference = pred_mean.squeeze(-2) - targ_mean.squeeze(-2)

    ccc = (2.0 * covariance) / (
        pred_var + targ_var + mean_difference.square() + eps
    )
    axis_weight = targ_var / (targ_var + tau ** 2)
    axis_loss = (axis_weight * (1.0 - ccc)).sum(dim=-1) / (
        axis_weight.sum(dim=-1) + eps
    )
    window_variance = targ_var.mean(dim=-1)
    activity_weight = window_variance / (window_variance + tau ** 2)
    return (activity_weight * axis_loss).mean()

def get_motion_loss(inte_state, label, confs):
    ## The state loss for evaluation
    loss, cov_loss = 0, {}
    if confs.loss == "normalized_squared_velocity":
        vel_loss, vel_dist = normalized_squared_velocity_loss(
            inte_state['net_vel'], label, s0=confs.s0,
        )
        if "ccc_weight" in confs and confs.ccc_weight:
            vel_loss = vel_loss + confs.ccc_weight * concordance_correlation_loss(
                inte_state['net_vel'], label, tau=confs.ccc_tau,
            )
    else:
        loss_fc = loss_fc_list[confs.loss]
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
