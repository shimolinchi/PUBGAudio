"""Auxiliary duplicating PIT for up to three same-class external sources.

For two sources the six onto mappings to three tracks are considered; a single
source is duplicated on all tracks; three sources use all six permutations.
Distance is zero on inactive targets, as in the single-task paper output.
"""
import itertools
import torch
from torch.nn import functional as F


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def adpit_loss(prediction, target, active_count, mask, error='mse'):
    # prediction/target: B,T,N=3,C,4. active_count/mask: B,T,C.
    if prediction.shape != target.shape or prediction.shape[2] != 3:
        raise ValueError('ADPIT target shape mismatch')
    if torch.any((active_count < 0) | (active_count > 3)):
        raise ValueError('At most three sources per class')
    a, b, c = [target[:, :, i] for i in range(3)]
    mappings = {0:[(0,0,0)], 1:[(0,0,0)],
        2:[p for p in itertools.product(range(2), repeat=3) if len(set(p)) == 2],
        3:list(itertools.permutations(range(3)))}
    result = torch.zeros_like(active_count, dtype=prediction.dtype)
    for count, choices in mappings.items():
        losses = []
        for indices in choices:
            candidate = torch.stack([[a,b,c][i] for i in indices], dim=2)
            delta = prediction - candidate
            value = delta.square() if error == 'mse' else delta.abs()
            losses.append(value.mean(dim=(2,4)))
        best = torch.stack(losses).amin(dim=0)
        result = torch.where(active_count == count, best, result)
    return masked_mean(result, mask)


def spatial_adpit_loss(prediction, batch, config):
    """Match tracks once using the sum of distinct activity/angle/distance terms."""
    target, count, mask = batch['target'], batch['counts'], batch['mask']
    loc = batch.get('localization_mask', (target[...,:3].square().sum(-1)>0).to(target.dtype))
    norm = torch.linalg.vector_norm(prediction[...,:3], dim=-1)
    direction = F.normalize(prediction[...,:3], dim=-1, eps=1e-4)
    mappings = {0:[(0,0,0)],1:[(0,0,0)],
        2:[p for p in itertools.product(range(2),repeat=3) if len(set(p))==2],
        3:list(itertools.permutations(range(3)))}
    result = torch.zeros_like(count, dtype=prediction.dtype)
    spatial_frames = (loc.sum(dim=2)>0).to(mask.dtype)*mask
    spatial_scale = mask.sum()/spatial_frames.sum().clamp_min(1)
    for n, choices in mappings.items():
        losses = []
        for indices in choices:
            truth = target[:,:,list(indices)]
            active = (truth[...,:3].square().sum(-1)>0).to(prediction.dtype)
            loc_weight = loc[:,:,list(indices)]*active
            activity = (norm-active).square().mean(dim=2)
            angle = (1-(direction*truth[...,:3]).sum(-1).clamp(-1,1))*loc_weight
            distance = F.smooth_l1_loss(prediction[...,3],truth[...,3],reduction='none')*loc_weight
            # Each observed positive contributes equally, regardless of quiet
            # negatives. Missing spatial evidence still trains class presence.
            denom = loc_weight.sum(dim=2).clamp_min(1)
            value = config.get('activity_weight',1.)*activity
            value += spatial_scale*config.get('direction_weight',2.)*angle.sum(dim=2)/denom
            value += spatial_scale*config.get('distance_weight',.2)*distance.sum(dim=2)/denom
            losses.append(value)
        best = torch.stack(losses).amin(dim=0)
        result = torch.where(count==n,best,result)
    return masked_mean(result, mask)


def training_loss(outputs, batch, self_weight=.5, loss_config=None):
    loss_config = loss_config or {}
    if loss_config.get('mode') == 'spatial_balanced':
        if 'accddoa' not in outputs:raise ValueError('Spatial balanced loss requires the multi-track head')
        external = spatial_adpit_loss(outputs['accddoa'], batch, loss_config)
    elif 'accddoa' in outputs:
        external = adpit_loss(outputs['accddoa'], batch['target'], batch['counts'], batch['mask'])
    else:
        # MT cannot represent multiple sources from one class. Never average them.
        valid = batch['mask'] * (batch['counts'] <= 1)
        target = batch['target'][:, :, 0]
        external = masked_mean((outputs['accdoa'] - target[..., :3]).square().mean(-1), valid)
        external += masked_mean((outputs['distance'] - target[..., 3]).square(), valid)
    own = external * 0
    if 'self_logits' in outputs:
        own = masked_mean(F.binary_cross_entropy_with_logits(outputs['self_logits'], batch['self_target'], reduction='none'), batch['self_mask'])
    return external + loss_config.get('self_weight', self_weight) * own, {'external': external.detach(), 'self': own.detach()}
