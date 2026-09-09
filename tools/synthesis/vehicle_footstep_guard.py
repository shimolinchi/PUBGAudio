"""Suppress whole external-footstep actors using actual rendered vehicle power."""
import numpy as np


def settings(cfg):
    c = cfg.get('external_footsteps', {}).get('vehicle_guard')
    if c:
        for key in ['self_vehicle_keep_probability','loud_vehicle_keep_probability','minimum_affected_fraction']:
            if not np.isfinite(c[key]) or not 0 <= c[key] <= 1:
                raise ValueError('Invalid vehicle/footstep probability or fraction: '+key)
        for key in ['self_vehicle_power_floor','vehicle_power_floor','minimum_overlap_seconds']:
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError('Invalid vehicle/footstep threshold: '+key)
        if not np.isfinite(c['vehicle_over_foot_db']):
            raise ValueError('Invalid relative vehicle level')
    return c


def active_power(power):
    return power >= np.maximum(np.max(power, initial=0)*10**(-35/10), 1e-9)


def vehicle_powers(scene, powers):
    vehicles = np.array([t['kind']=='vehicle' for t in scene['tracks']], bool)
    own = np.array([t['kind']=='vehicle' and t['source_role']=='self' for t in scene['tracks']], bool)
    return powers[vehicles].sum(axis=0), powers[own].sum(axis=0)


def interference(foot_power, vehicle_power, cabin_power, cfg):
    c = settings(cfg)
    active = active_power(foot_power)
    cabin = cabin_power >= c['self_vehicle_power_floor']
    loud = (vehicle_power >= c['vehicle_power_floor']) & (
        vehicle_power >= foot_power*10**(c['vehicle_over_foot_db']/10))
    return active & cabin, active & loud


def keep_track(track, power, vehicle_power, cabin_power, cfg):
    c = settings(cfg)
    own, loud = interference(power, vehicle_power, cabin_power, cfg)
    minimum = c['minimum_overlap_seconds']/.01
    active_count = int(active_power(power).sum())
    own_conflict = own.sum() >= minimum
    loud_conflict = (loud.sum() >= minimum and
        loud.sum()/max(active_count, 1) >= c['minimum_affected_fraction'])
    probability = c['self_vehicle_keep_probability'] if own_conflict else c['loud_vehicle_keep_probability']
    draw = track['vehicle_overlap_admission_draw']
    if not 0 <= draw < 1:
        raise ValueError('Missing or invalid deterministic admission draw')
    keep = not (own_conflict or loud_conflict) or draw < probability
    return keep, dict(decision='external_footstep_vehicle_overlap', track_id=track['id'],
        selected=keep, probability=probability if own_conflict or loud_conflict else 1.,
        draw=draw, own_vehicle_overlap_frames=int(own.sum()),
        loud_vehicle_overlap_frames=int(loud.sum()), active_footstep_frames=active_count,
        evidence='rendered 32ms power windows at 10ms hop, summed vehicle power',
        reason='self_vehicle' if own_conflict else 'loud_vehicle' if loud_conflict else 'no_vehicle_conflict')
