"""V4 motion, native confusers and explicit observability (not identity inference)."""
from functools import lru_cache
import numpy as np
import self_audio_synthesis as v3
from self_audio_synthesis import RATE, trajectory


class SourceLibrary(v3.SourceLibrary):
    @lru_cache(maxsize=8)
    def weather(self, sid):
        source = self.sources[sid]
        if source['role'] != 'weather':
            raise ValueError('Background must be verified weather')
        x = v3.base.read_wave(self.root / source['wav_file'])
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x -= x.mean(axis=0, dtype=np.float64)
        gain = min(.045 / max(float(np.sqrt(np.mean(x.astype(np.float64)**2))), 1e-12),
                   .35 / max(float(np.max(np.abs(x))), 1e-12))
        return x * gain


def render_background(meta, library, seconds):
    original = library.weather(meta['source_id'])
    positions = np.arange(round(seconds * RATE)) + meta['phase_samples']
    x = np.column_stack([v3.base.read_cycle(v3.base.seamless_cycle(original[:, ear], round(.25*RATE)), positions)
                         for ear in range(2)])
    x *= np.float32(10 ** (meta['gain_db'] / 20))
    for ear in range(2):
        v3.base.edge_fade(x[:, ear], round(.15*RATE))
    return np.pad(x, ((64,0),(0,0)))[:len(x)]


def curve_track(track, seconds, rng):
    if track['class_index'] == 2 or track.get('source_role') == 'self':
        return
    old = track['trajectory']['heading_knots_unwrapped_degrees']
    heading = old[0][1]
    bend = float(rng.uniform(50,95)) * int(rng.choice([-1,1]))
    spec = track['trajectory']
    spec['heading_knots_unwrapped_degrees'] = [[0,heading], [.3*seconds,heading+bend],
        [.65*seconds,heading-bend], [seconds,heading+.25*bend]]
    # Keep external paths clear of the listener; rotate curve, never infer identity.
    for _ in range(4):
        if trajectory(spec, seconds)['distance'].min() >= 4:
            break
        for knot in spec['heading_knots_unwrapped_degrees']:
            knot[1] += 90
    if trajectory(spec, seconds)['distance'].min() < 4:
        spec['heading_knots_unwrapped_degrees'] = old


def action_vehicle(track, seconds, rng, tyres, collisions, collision_probability=.4):
    life = track['lifecycle']
    move, stop = life['move_seconds'], life['final_stop_seconds']
    duration = stop - move
    peak = max(v for _,v in track['trajectory']['speed_knots'])
    first_stop = move + .58 * duration
    brake_duration = min(.5, .07 * duration)
    times = [move, move+.24*duration, move+.265*duration, move+.43*duration,
             first_stop-brake_duration, first_stop, move+.64*duration,
             move+.83*duration, stop-min(.7,.1*duration), stop]
    values = [0,peak,peak,.65*peak,.82*peak,0,0,.62*peak,.5*peak,0]
    track['trajectory']['speed_knots'] = [[0.,0.],[life['startup_seconds'],0.]] + list(map(list,zip(times,values))) + [[seconds,0.]]
    track['emergency_brake_seconds'] = [first_stop-brake_duration, first_stop]
    track['motion_policy'] = 'short_cruise_variable_load_emergency_stop_resume'
    for role in ['brake','skid']:
        candidates = [s['id'] for s in tyres if s['role'] == role]
        if candidates:
            track['layers'][role] = str(rng.choice(candidates))
    if collisions and rng.random() < collision_probability:
        track['events'].append(dict(source_id=str(rng.choice([s['id'] for s in collisions])),
            onset_sample=round(first_stop*RATE), gain=float(rng.uniform(.65,1.1)), role='collision'))
        track['collision_note'] = 'native generic collision layer; not a calibrated vehicle-body simulation'


def frame_labels(metadata, motions, backgrounds, seconds, cfg, powers):
    k = len(metadata)
    if k:
        z = v3.frame_labels(None, metadata, motions, seconds, cfg, powers=powers[:k])
    else:
        dummy = dict(class_index=2, source_role='external')
        spec = dict(initial_xy=[100.,0.],speed_knots=[[0,0],[seconds,0]],
                    heading_knots_unwrapped_degrees=[[0,0],[seconds,0]])
        z = v3.frame_labels(None,[dummy],[trajectory(spec,seconds)],seconds,cfg,powers=np.zeros_like(powers[:1]))
        for key in list(z):
            if key.startswith('track_'):
                z[key] = z[key][:0]
    total = powers.sum(axis=0)
    own_car = np.asarray([m.get('source_role') == 'self' and m['class_index']==1 for m in metadata],bool)
    cabin_power = powers[:k][own_car].sum(axis=0)
    own_power = np.zeros_like(z['self_activity'],dtype=np.float64)
    observable = np.zeros_like(z['track_activity'],dtype=np.uint8)
    collision = np.zeros_like(observable)
    # Retain all simulator events as physical truth; hidden positives lose their
    # loss weight, rather than becoming incorrect supervised negative examples.
    for i,meta in enumerate(metadata):
        cls = meta['class_index']
        active = z['track_activity'][i].astype(bool)
        ratio = powers[i] / np.maximum(total - powers[i],1e-15)
        visible = ratio >= 10 ** (cfg['minimum_sir_db']/10)
        if meta.get('source_role') != 'self':
            if cls == 0:
                visible &= powers[i] / np.maximum(cabin_power,1e-15) >= 10 ** (cfg['cabin_foot_sir_db']/10)
            elif cls == 1:
                far = z['track_distance_units'][i] >= cfg['cabin_far_vehicle_distance']
                visible &= ~far | (powers[i] / np.maximum(cabin_power,1e-15) >= 10 ** (cfg['cabin_vehicle_sir_db']/10))
            hidden = np.flatnonzero(active & ~visible)
            sectors = z['track_sector_index'][i,hidden]
            z['activity_loss_mask'][hidden,cls,sectors] = 0
            z['range_loss_mask'][hidden,cls,sectors] = 0
        else:
            own_power[:,cls] += powers[i] * active
        observable[i] = (active & visible).astype(np.uint8)
        times = z['frame_right_edge_seconds'] - .016 - 64/RATE
        for event in meta['events']:
            if event['role'] == 'collision':
                start = event['onset_sample']/RATE
                collision[i,(times>=start)&(times<start+event['duration_seconds'])] = 1
    own_visible = own_power / np.maximum(total[:,None]-own_power,1e-15) >= 10 ** (cfg['minimum_sir_db']/10)
    z['self_activity_loss_mask'][(z['self_activity']!=0)&~own_visible] = 0
    any_mask = z['activity_loss_mask'].max(axis=2)
    any_mask[:,:3] = np.where(z['any_class_activity'][:,:3]!=0,
        (z['activity']*z['activity_loss_mask']).max(axis=2)[:,:3] |
        (z['self_activity']*z['self_activity_loss_mask']),1)
    z.update(any_class_activity_loss_mask=any_mask,track_observable=observable,
        track_collision_activity=collision,background_power=powers[k:].sum(axis=0).astype(np.float32),
        self_vehicle_power=cabin_power.astype(np.float32),
        weather_present=np.full(len(total),bool(backgrounds),np.uint8))
    return z
