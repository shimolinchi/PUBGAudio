"""Provisional finite footstep range; metadata radii are not measured audibility.

Keep the old HRIR/lowpass and replace only the distance amplitude response.
Each event uses its own gait profile, including its decay across a gait change.
"""
import numpy as np
from motion_synthesis import SpatialRenderer


def settings(cfg):
    return cfg.get('footstep_acoustics') if cfg else None


def radius_m(acoustics, gait):
    return acoustics['profiles'][gait]['cutoff_m_estimate']


def gain_at_distance(distance_m, acoustics, gait):
    """Amplitude relative to the same dry signal at the reference distance."""
    d = np.asarray(distance_m, dtype=np.float64)
    radius = radius_m(acoustics, gait)
    ref = acoustics['reference_distance_m']
    start = acoustics['taper_start_fraction'] * radius
    if not 0 < ref < start < radius or acoustics['amplitude_exponent'] <= 0:
        raise ValueError('Invalid finite footstep attenuation profile')
    q = np.clip((d - start) / (radius - start), 0, 1)
    taper = 1 - q*q*(3 - 2*q)
    gain = (ref / np.maximum(d, ref)) ** acoustics['amplitude_exponent'] * taper
    return np.where(d >= radius, 0., gain)


class FootstepSpatialRenderer(SpatialRenderer):
    def __init__(self, base, acoustics, gait):
        self.__dict__ = base.__dict__.copy()  # Share immutable HRIR/FFT tables.
        self.acoustics, self.gait = acoustics, gait

    def distance_response(self, distance):
        _, cutoff = super().distance_response(distance)
        metres = distance * self.acoustics['metres_per_scene_unit_assumed']
        return gain_at_distance(metres, self.acoustics, self.gait), cutoff


def detailed_labels(tracks, centres, distance_units, cfg):
    """Additional acoustic estimates, separate from rendered activity/SIR masks."""
    ac = settings(cfg)
    shape = distance_units.shape
    radii = np.full(shape, np.nan, np.float32)
    gains = np.full(shape, np.nan, np.float32)
    scheduled = np.zeros(shape, np.uint8)
    outside = np.zeros(shape, np.uint8)
    metres = np.full(shape, np.nan, np.float32)
    for i, tr in enumerate(tracks):
        if tr['kind'] != 'footsteps':
            continue
        for e in tr['events']:
            scheduled[i] |= ((centres + .016 >= e['start']) &
                             (centres - .016 < e['start'] + e['duration'])).astype(np.uint8)
        if tr['source_role'] == 'self':
            metres[i] = 0
            gains[i] = 1
            continue
        metres[i] = distance_units[i] * ac['metres_per_scene_unit_assumed']
        # Carry the last gait through pauses; position is still annotated there.
        roles = np.full(len(centres), tr['events'][0]['role'] if tr['events'] else 'walk', dtype='<U20')
        for s in sorted(tr['states'], key=lambda s: s['start']):
            if s['state'] in ac['profiles']:
                roles[centres >= s['start']] = s['state']
        for gait in np.unique(roles):
            use = roles == gait
            radii[i, use] = radius_m(ac, gait)
            gains[i, use] = gain_at_distance(metres[i, use], ac, gait)
        # If tails of two gait families overlap, report the larger possible radius/gain.
        event_radius = np.zeros(len(centres))
        event_gain = np.zeros(len(centres))
        for e in tr['events']:
            use = (centres + .016 >= e['start']) & (centres - .016 < e['start'] + e['duration'])
            event_radius[use] = np.maximum(event_radius[use], radius_m(ac, e['role']))
            event_gain[use] = np.maximum(event_gain[use], gain_at_distance(metres[i, use], ac, e['role']))
        use = scheduled[i].astype(bool)
        radii[i, use] = event_radius[use]
        gains[i, use] = event_gain[use]
        outside[i] = (metres[i] >= radii[i]).astype(np.uint8)
    return dict(track_distance_m_estimate=metres, track_footstep_cutoff_m_estimate=radii,
                track_footstep_distance_gain_estimate=gains,
                track_footstep_emission_scheduled=scheduled,
                track_outside_footstep_radius=outside)
