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


def training_loss(outputs, batch, self_weight=.5):
    if 'accddoa' in outputs:
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
    return external + self_weight * own, {'external': external.detach(), 'self': own.detach()}
