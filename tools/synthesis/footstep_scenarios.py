"""Opt-in long external-footstep scenes and deterministic scene sampling."""
import copy
import numpy as np
import footstep_acoustics as foot


def scene_profile(cfg, seed):
    profiles = cfg.get('scene_profiles')
    if not profiles:
        return 'mixed', cfg['clip_seconds']
    names = list(profiles['weights'])
    weights = np.asarray(list(profiles['weights'].values()), float)
    if (not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0
            or set(names) - {'mixed', 'external_footsteps_only'}):
        raise ValueError('Invalid scene profile weights')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 612]))
    name = names[int(rng.choice(len(names), p=weights/weights.sum()))]
    seconds = profiles[name]['clip_seconds'] if name != 'mixed' else cfg['clip_seconds']
    if seconds < 5 or seconds % 5:
        raise ValueError('Scene duration must be a positive multiple of five seconds')
    return name, seconds


def scene_configuration(cfg, seed):
    name, seconds = scene_profile(cfg, seed)
    if name == 'mixed':
        return cfg, name
    result = copy.deepcopy(cfg)
    spec = cfg['scene_profiles'][name]
    result['clip_seconds'] = seconds
    result['footsteps'].update(spec['gait_bouts'])
    result['listener_view'].update(copy.deepcopy(spec['listener_view']))
    return result, name


def _unit_loop(kind, u):
    theta = u * 2*np.pi
    if kind == 'oval':
        return np.column_stack((np.cos(theta), np.sin(theta)))
    if kind == 'figure_eight':
        return np.column_stack((np.cos(theta), .8*np.sin(2*theta)))
    if kind != 'bezier_loop':
        raise ValueError('Unknown local footstep curve')
    # Four C1-continuous cubic Bezier segments, confined to [-1,1]^2.
    nodes = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[1.,0.]])
    tangents = np.array([[0.,.6],[-.6,0.],[0.,-.6],[.6,0.],[0.,.6]])
    q = (u % 1)*4
    segment = np.minimum(q.astype(int), 3)
    t = (q-segment)[:, None]
    a, d = nodes[segment], nodes[segment+1]
    b, c = a+tangents[segment], d-tangents[segment+1]
    return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d


def local_patrol(planner, track, spec, gait):
    """Map a smooth local loop into a bounded annular sector, then use arc length.

    Speed controls progress; pauses stop position. This does not clamp coordinates,
    teleport a source, or slow the actor to stretch a recording.
    """
    shape = planner.choice(spec['curve_weights'], 'foot_focus_curve')
    mode = planner.choice(spec['distance_mode_weights'], 'foot_focus_distance_mode')
    lo, hi = spec['radius_fraction_by_mode'][mode]
    radius = foot.radius_m(foot.settings(planner.cfg), gait)
    radius /= planner.cfg['footstep_acoustics']['metres_per_scene_unit_assumed']
    if not 0 < lo < hi < 1:
        raise ValueError('Footstep patrol must remain inside its gait radius')
    bearing = planner.uniform([0, 2*np.pi])
    span = np.radians(planner.uniform(spec['bearing_span_degrees']))
    phase = planner.uniform([0, 1])
    unit = _unit_loop(shape, np.linspace(0, 1, 16385)+phase)
    radial = radius*(lo+(hi-lo)*(unit[:,0]+1)/2)
    azimuth = bearing+span*unit[:,1]/2
    raw = np.column_stack((radial*np.sin(azimuth), radial*np.cos(azimuth)))
    raw[-1] = raw[0]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    speed = np.asarray(track['path']['speed'])
    travelled = np.r_[0., np.cumsum((speed[:-1]+speed[1:])*planner.dt/2)]
    progress = travelled % arc[-1]
    xy = np.column_stack([np.interp(progress, arc, raw[:,i]) for i in range(2)])
    planner.set_path(track, xy, speed)
    track.update(curve='local_'+shape, distance_mode=mode, listener_bound=False,
        distance_reference_radius_m_estimate=radius*planner.cfg['footstep_acoustics']['metres_per_scene_unit_assumed'],
        curve_parameters=dict(shape=shape, mode=mode, radius_fraction=[lo,hi],
            gait=gait, initial_phase=phase, bearing_radians=bearing,
            bearing_span_radians=span, loop_length_units=float(arc[-1]),
            construction='smooth polar-sector patrol, arc-length speed, fixed world path'))


def populate_footsteps_only(planner):
    spec = planner.cfg['scene_profiles']['external_footsteps_only']
    count = int(planner.choice(spec['source_count_weights'], 'external_foot_source_count'))
    if count not in (1, 2):
        raise ValueError('Footstep-only scenes support one or two actors')
    density = planner.choice(planner.cfg['density_weights'], 'density')
    available = {s['role'] for s in planner.sources if s['source_group_id'].startswith('footsteps:')}
    gaits = {k:v for k,v in planner.cfg['footsteps']['initial_weights'].items() if k in available}
    for index in range(count):
        start = planner.uniform(spec['start_seconds'])
        if index:
            start += planner.uniform(spec['second_source_delay_seconds'])
        end = planner.t[-1]-planner.uniform(spec['end_margin_seconds'])
        gait = planner.choice(gaits, 'foot_focus_actor_gait')
        track = planner.footsteps(start, 'external', end_seconds=end, fixed_gait=gait)
        track['scenario_actor_index'] = index
        local_patrol(planner, track, spec, gait)
    return density


def add_vehicle_admission_draws(planner):
    if not planner.cfg.get('external_footsteps', {}).get('vehicle_guard'):
        return
    for track in planner.tracks:
        if track['kind']=='footsteps' and track['source_role']=='external':
            track['vehicle_overlap_admission_draw'] = float(planner.rng.random())
