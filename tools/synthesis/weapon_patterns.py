"""Persistent trigger patterns, explicit magazines, and non-auto DMR/bolt policy."""


def populate_fire(planner, track, profile, role, group, start, end, attack_target):
    pattern = planner.choice(profile['pattern_weights'], 'firing_pattern')
    if profile['weapon_kind'] in ['dmr', 'bolt'] and 'auto' in pattern:
        raise ValueError('DMR and bolt rifles must not use automatic firing')
    native_mode = 'auto' if 'auto' in pattern else 'burst' if pattern == 'burst' else 'single'
    cursor, ammunition, fired = start, profile['magazine_limit'], 0
    while cursor < end and ammunition:
        if pattern == 'sustained_auto':
            shots = int(planner.choice(list(range(8, min(24, profile['magazine_limit'])+1)), 'automatic_length'))
        elif pattern == 'short_auto':
            shots = int(planner.choice([3, 4, 5, 6], 'automatic_length'))
        elif pattern == 'burst':
            shots = profile['burst_rounds']
        else:
            shots = 1
        began = cursor
        for _ in range(min(shots, ammunition)):
            if cursor >= end:
                break
            planner.event(track, role, cursor, group=group, attack_target=attack_target,
                          firing_pattern=pattern, native_fire_mode=native_mode, shot_index=fired)
            cursor += profile['interval_seconds']
            ammunition -= 1
            fired += 1
        track['states'].append(dict(start=began, end=min(cursor, end), state=pattern))
        pause = planner.uniform(profile['pause_seconds'])
        if cursor < end:
            track['states'].append(dict(start=cursor, end=min(cursor+pause, end), state='trigger_released'))
        cursor += pause
    track.update(start=start, end=end, shots_fired=fired, magazine_limit=profile['magazine_limit'],
                 firing_pattern=pattern, native_fire_mode=native_mode, weapon_kind=profile['weapon_kind'],
                 firing_interval_evidence='documented_synthesis_profile_not_current_patch_ballistics')
    if not ammunition and cursor < end:
        track['states'].append(dict(start=cursor, end=end, state='magazine_empty_no_reload_audio'))
    return track
