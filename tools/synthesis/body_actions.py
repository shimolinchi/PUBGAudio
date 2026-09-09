"""Continuous locomotion with scheduled posture changes and obstacle-linked vaults.

Exact poses are simulation metadata. Shared native transition rustle does not
establish an acoustically identifiable crouch/prone subtype.
"""
import numpy as np
import motion_synthesis as motion

POSTURE = dict(walk='standing', run='standing', sprint='standing',
               crouch_walk='crouched', prone_crawl='prone')


def footsteps(p, start, role='external'):
    c = p.cfg['footsteps']; ac = p.cfg['body_actions']
    end = min(start + p.duration('footsteps'), p.t[-1] - 1)
    groups = [g for g in p.groups if g.startswith('footsteps:')]
    group = p.choice(groups, 'surface')
    available = {s['role'] for s in p.groups[group]}
    weights = {k: v for k, v in c['initial_weights'].items() if k in available and v > 0}
    tr = p.base_track('footsteps', role, 'locomotion', group)
    tr.update(surface=group.split(':')[1], start=start, end=end, posture_states=[], body_action_states=[])
    v = np.zeros(len(p.t)); current = start; gait = p.choice(weights, 'gait')
    phase = p.uniform([0, 1]); step_index = 0; actions = []
    posture = POSTURE[gait]; last_action_end = -np.inf

    def state(a, b, name, pose=None):
        tr['states'].append(dict(start=a, end=b, state=name))
        tr['posture_states'].append(dict(start=a, end=b, state=pose or posture))

    def change_pose(a, target):
        nonlocal posture
        before = posture
        action = {('standing','crouched'):'crouch_down', ('crouched','standing'):'crouch_up',
                  ('standing','prone'):'prone_down', ('prone','standing'):'prone_up',
                  ('crouched','prone'):'crouch_to_prone', ('prone','crouched'):'prone_to_crouch'}[(before,target)]
        sid = p.source('transition_rustle')
        duration = max(p.uniform(ac['transition_seconds']['prone' if 'prone' in (before,target) else 'crouch']),
                       p.by_id[sid]['duration_seconds'])
        fx = p.base_track('action', role, action, 'body:transition')
        fx.update(start=a, end=a+duration, actor_parent=tr['id'], posture_from=before, posture_to=target,
                  semantic_evidence='scheduled_pose; native_shared_transition_rustle_not_pose_unique',
                  acoustic_identity='shared_transition_rustle')
        fx['states'] = [dict(start=a, end=a+duration, state=action)]
        p.event(fx, 'transition_rustle', a, source_id=sid, action=action,
                fine_label_evidence='simulation_schedule_not_unique_native_pose')
        actions.append(fx)
        state(a, a+duration, action, before+'_to_'+target)
        tr['body_action_states'].append(dict(start=a, end=a+duration, state=action))
        p.chains.append(dict(kind='posture_change', actor=tr['id'], effect=fx['id'],
                             start=a, end=a+duration, posture_from=before, posture_to=target))
        posture = target
        return a+duration

    while current < end - .7:
        duration = p.dwell(c['minimum_dwell_seconds'], c['switch_probability_after_minimum'],
                           c['maximum_dwell_seconds'], gait)
        stop = min(end, current+duration); target = p.uniform(c['speed_units_per_second'][gait])
        active = (p.t >= current) & (p.t < stop)
        ramp = min(c['speed_transition_seconds'], (stop-current)/3)
        v[active] = target * motion.smoothstep((p.t[active]-current)/ramp) * motion.smoothstep((stop-p.t[active])/ramp)
        state(current, stop, gait)
        ids = [s['id'] for s in p.groups[group] if s['role'] == gait]
        for i in np.flatnonzero(active):
            phase += (min(3.65, 1.55*v[i]**.47) if v[i] >= .15 else 0) * p.dt
            if phase >= 1:
                phase -= 1; pool = ids[step_index % 2::2] or ids
                p.event(tr, gait, p.t[i], source_id=p.choice(pool, 'foot_variant'),
                        simulated_foot=['left','right'][step_index % 2], foot_side_evidence='simulated_not_native')
                step_index += 1
        # Allow native footstep decay before a new posture/contact action.
        tail = max([stop]+[e['start']+e['duration'] for e in tr['events']])
        pause = p.uniform(c['pause_duration_seconds']) if p.draw(c['pause_probability_after_bout'], 'pause_after_bout') else .3
        current = min(end, max(stop+pause, tail+.08))
        state(stop, current, 'pause')
        if current >= end - c['minimum_dwell_seconds']:
            state(current, end, 'pause'); break
        candidates = {'continue': ac['after_bout_weights']['continue']}
        if posture == 'standing' and current-last_action_end >= ac['minimum_action_gap_seconds'] and end-current >= ac['required_remaining_seconds']:
            if p.roles['transition_rustle']:
                candidates.update({k: ac['after_bout_weights'][k] for k in ['crouch_cycle','prone_cycle']})
            vault_groups = [g for g in p.groups if g.startswith('vault:') and
                            {'vault_grab','vault_climb','vault_contact'} <= {s['role'] for s in p.groups[g]}]
            if vault_groups and p.roles['vault_rustle']:
                candidates['vault'] = ac['after_bout_weights']['vault']
        candidates = {k:w for k,w in candidates.items() if w > 0} or {'continue':1}
        action = p.choice(candidates, 'body_action_after_bout', current)
        if action in ['crouch_cycle','prone_cycle']:
            current = change_pose(current, 'crouched' if action == 'crouch_cycle' else 'prone')
            hold_end = current + p.uniform(ac['posture_hold_seconds'])
            state(current, hold_end, 'pause'); current = change_pose(hold_end, 'standing')
            last_action_end = current
        elif action == 'vault':
            material = p.choice(vault_groups, 'vault_obstacle_material', current)
            fx = p.base_track('action', role, 'vault', material)
            cloth = p.source('vault_rustle'); dur = max(ac['vault_seconds'], p.by_id[cloth]['duration_seconds']+.3)
            fx.update(start=current, end=current+dur, actor_parent=tr['id'], obstacle_material=material.split(':')[1],
                      semantic_evidence='native_vault_media; scheduled_obstacle_and_timing_not_game_animation_reproduction')
            for r, offset in [('vault_grab',0),('vault_climb',.65),('vault_contact',dur-1)]:
                p.event(fx, r, current+offset, group=material)
            p.event(fx, 'vault_rustle', current, source_id=cloth)
            fx['states'] = [dict(start=current, end=current+dur, state='vault')]
            actions.append(fx); state(current, current+dur, 'vault')
            tr['body_action_states'].append(dict(start=current, end=current+dur, state='vault'))
            mask = (p.t >= current) & (p.t < current+dur)
            u = (p.t[mask]-current)/dur
            v[mask] = ac['vault_peak_speed'] * np.sin(np.pi*u)**2
            p.chains.append(dict(kind='vault', actor=tr['id'], effect=fx['id'], start=current, end=current+dur,
                                 obstacle_material=fx['obstacle_material'], obstacle_evidence='planned_synthetic_obstacle'))
            current += dur; last_action_end = current
        else:
            next_gait = p.choice(weights, 'gait_after_pause', current)
            next_posture = POSTURE[next_gait]
            if next_posture != posture:
                if p.roles['transition_rustle'] and end-current >= ac['required_remaining_seconds']:
                    current = change_pose(current, next_posture)
                    last_action_end = current
                else:
                    next_gait = p.choice({k:w for k,w in weights.items() if POSTURE[k] == posture}, 'gait_same_posture', current)
            gait = next_gait
    p.curved(tr, v)
    for fx in actions:
        fx['path'] = tr['path']
        fx['surface'] = tr['surface']
    for chain in p.chains:
        if chain.get('actor') == tr['id'] and chain['kind'] == 'vault':
            mid = (chain['start']+chain['end'])/2
            chain['obstacle_xy'] = [float(np.interp(mid,p.t,tr['path']['xy'][:,i])) for i in range(2)]
    return tr
