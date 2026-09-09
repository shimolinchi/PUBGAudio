"""Generate a small v5.3 release with independent source families and scenes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import causal_scene as plan
from causal_render import Library, render
import motion_synthesis as motion
from project_causal_labels import project
from verify_causal_preview import verify_chain
from air_absorption import configure_spatial
from footstep_scenarios import scene_configuration, scene_profile, populate_footsteps_only, add_vehicle_admission_draws
from confidence_policy import validate_policy, split_scenes, write_plan

STATE = {}
SPLITS = ('train', 'validation', 'test')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, default=plan.serializable,
                                   separators=(',', ':'))+'\n', encoding='utf-8')


def source_partitions(sources, spec):
    if spec.get('split_protocol') == 'known_assets_new_scenes':
        selected = [s for s in sources if s['source_group_id'] in spec['source_groups'] and
            (not s['source_group_id'].startswith('footsteps:') or s['role'] in spec['included_gaits'])]
        if {s['source_group_id'] for s in selected} != set(spec['source_groups']):
            raise ValueError('Missing requested sound types')
        # The same fixed client library is available in every split. This is
        # explicitly scene generalization, not independent-recording evaluation.
        return {split: selected.copy() for split in SPLITS}
    mapping = {}
    for split, groups in spec['source_groups_by_split'].items():
        for group in groups:
            if group in mapping:
                raise ValueError('Recording family crosses partitions: '+group)
            mapping[group] = split
    pools = {s: [] for s in SPLITS}
    pcm_owner = {}
    for source in sources:
        group = source['source_group_id']
        if group not in mapping:
            continue
        if group.startswith('footsteps:') and source['role'] not in spec['included_gaits']:
            continue
        split = mapping[group]
        pcm = source['pcm_sha256']
        if pcm in pcm_owner and pcm_owner[pcm] != split:
            raise ValueError('PCM crosses partitions: '+source['id'])
        pcm_owner[pcm] = split
        pools[split].append(source)
    for split, pool in pools.items():
        found = {s['source_group_id'] for s in pool}
        if found != set(spec['source_groups_by_split'][split]):
            raise ValueError('Missing source groups: '+split)
    return pools


def configuration(spec):
    cfg = plan.config(spec['base_config'])
    cfg.update(seed=spec['seed'], clip_seconds=spec['clip_seconds'],
               purpose='light training release with source and scene isolation',
               display_version='v5.3-light')
    if spec.get('split_protocol')=='known_assets_new_scenes':
        cfg.update(purpose='known client assets with independent synthetic scenes',display_version=cfg.get('display_version','v6'))
    cfg.pop('body_actions', None)
    cfg['footsteps']['initial_weights'] = {k: v for k, v in cfg['footsteps']['initial_weights'].items()
                                          if k in spec['included_gaits']}
    for density, probabilities in cfg['spawn_probability_per_eligible_step'].items():
        cfg['spawn_probability_per_eligible_step'][density] = {
            k: v for k, v in probabilities.items() if k in spec['included_spawn_kinds']}
    multipliers = spec.get('spawn_probability_multipliers', {})
    if set(multipliers)-set(spec['included_spawn_kinds']):
        raise ValueError('Unknown spawn probability multiplier')
    for probabilities in cfg['spawn_probability_per_eligible_step'].values():
        for kind in probabilities:
            value = probabilities[kind]*multipliers.get(kind, 1.)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid spawn probability after multiplier')
            probabilities[kind] = value
    cfg['vehicle']['attack_probability_per_vehicle_with_motion'] = 0
    cfg['vehicle']['available_layers_only'] = True
    cfg['light_release'] = spec
    return cfg


def initialize(root, sources_root, hrir, cfg, pools, targets):
    STATE.update(root=Path(root), cfg=cfg, pools=pools, targets=targets,
                 libs={s: Library(Path(sources_root), p, cfg.get('audio')) for s, p in pools.items()},
                 spatial=configure_spatial(motion.SpatialRenderer(Path(hrir), digest(hrir)), cfg))


def generate(job):
    split, number, seed = job
    started = time.monotonic()
    root = STATE['root']
    cfg, profile = scene_configuration(STATE['cfg'], seed)
    planner = plan.Planner(cfg, STATE['pools'][split], seed)
    density = populate_footsteps_only(planner) if profile=='external_footsteps_only' else planner.random_scene()
    add_vehicle_admission_draws(planner)
    identifier = f'{split}_{number:04d}'
    scene = planner.result(identifier, density)
    scene.update(split=split, seed=seed)
    if cfg.get('scene_profiles'):
        scene.update(sampling_profile=profile, scene_profile_weights=cfg['scene_profiles']['weights'])
    verify_chain(scene, cfg)
    stats = render(scene, STATE['libs'][split], STATE['spatial'], cfg, root/'audio')
    if cfg.get('external_footsteps', {}).get('vehicle_guard'):
        verify_chain(scene, cfg)
    # The opt-in vehicle guard removes whole actors and records its decisions.
    write_json(root/'recipes'/(identifier+'.json'), scene)
    label = root/'audio'/(identifier+'.npz')
    projected_path = root/'audio'/(identifier+'_train3.npz')
    with np.load(label, allow_pickle=False) as z:
        projected = project(z, STATE['targets']['external_targets'], STATE['targets']['self_targets'])
        np.savez_compressed(projected_path, **projected)
        active = z['track_activity'].astype(bool)
        valid = z['track_direction_valid'].astype(bool)
        stats['observable_by_class_role'] = {
            f'{kind}/{role}': int((z['track_observable'][np.array([
                t['kind']==kind and t['source_role']==role for t in scene['tracks']], dtype=bool)]).sum())
            for kind in ['footsteps','vehicle','gunfire'] for role in ['external','self']}
        stats['external_audible_track_frames'] = int((active & valid).sum())
    used = set()
    for track in scene['tracks']:
        used.update(e['source_id'] for e in track['events'])
        used.update(track.get('layers', {}).values())
    allowed = {s['id'] for s in STATE['pools'][split]}
    if not used <= allowed:
        raise ValueError('Scene references another partition')
    row = dict(id=identifier, split=split, density=density, seed=seed,
               duration_seconds=scene['seconds'], sample_rate_hz=32000,
               audio_file=f'audio/{identifier}.wav', label_file=f'audio/{identifier}_train3.npz',
               detailed_label_file=f'audio/{identifier}.npz', recipe_file=f'recipes/{identifier}.json',
               source_ids=sorted(used), audio_sha256=digest(root/'audio'/(identifier+'.wav')),
               label_sha256=digest(projected_path), detailed_label_sha256=digest(label),
               recipe_sha256=digest(root/'recipes'/(identifier+'.json')))
    if cfg.get('scene_profiles'):
        row['sampling_profile'] = profile
    write_json(root/'rows'/(identifier+'.json'), row)
    stats.update(split=split, density=density, elapsed_seconds=round(time.monotonic()-started, 2),
                 track_kinds=dict(Counter(t['kind'] for t in scene['tracks'])),
                 collision_events=sum(e['role']=='collision' for t in scene['tracks'] for e in t['events']))
    if cfg.get('scene_profiles'):
        stats['sampling_profile'] = profile
    write_json(root/'rows'/(identifier+'.stats.json'), stats)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, default=Path('configs/light_dataset_v5_3.json'))
    p.add_argument('--sources', type=Path, default=Path('datasets/sources-v5_3-final'))
    p.add_argument('--hrir', type=Path, default=Path('datasets/hrir/kemar-diffuse.zip'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    cfg = configuration(spec)
    manifest = json.loads((args.sources/'sources.json').read_text(encoding='utf-8'))
    pools = source_partitions(manifest['sources'], spec)
    targets = json.loads(Path('configs/training_targets_v5.json').read_text(encoding='utf-8'))
    if spec.get('confidence_policy_file'):
        policy = json.loads(Path(spec['confidence_policy_file']).read_text(encoding='utf-8'))
        validate_policy(policy)
        # Validate the scene budget before creating any dataset or audio files.
        split_scenes([dict(id=f'validation_{i:04d}') for i in range(spec['split_counts']['validation'])],
                     policy['calibration']['seed'], policy['calibration']['fit_fraction'])
        targets['confidence_policy'] = policy
    selected = [s for pool in pools.values() for s in pool]
    for source in selected:
        if digest(args.sources/source['wav_file']) != source['wav_sha256']:
            raise ValueError('Source changed: '+source['id'])
    identity = dict(spec_sha256=digest(args.spec), source_manifest_sha256=digest(args.sources/'sources.json'),
                    hrir_sha256=digest(args.hrir), targets=targets,
                    generator_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    if args.output.exists():
        if not args.resume:
            raise ValueError('Preserve existing dataset; use a new output directory or --resume')
        if json.loads((args.output/'generation_identity.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume configuration or generator changed')
    for folder in ['recipes','rows','audio','generator_snapshot']:
        (args.output/folder).mkdir(parents=True, exist_ok=True)
    write_json(args.output/'generation_identity.json', identity)
    write_json(args.output/'config.json', cfg)
    write_json(args.output/'training_targets.json', targets)
    write_json(args.output/'source_manifest.json', manifest)
    write_json(args.output/'source_split.json', dict(method=spec.get('split_protocol', 'whole family and PCM isolation'),
               groups={k: sorted({s['source_group_id'] for s in v}) for k,v in pools.items()},
               source_to_splits={sid: [k for k,v in pools.items() if any(s['id']==sid for s in v)]
                                 for sid in {s['id'] for s in selected}},
               source_counts={k:len(v) for k,v in pools.items()},
               independent_recording_evaluation=spec.get('split_protocol')!='known_assets_new_scenes'))
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, args.output/'generator_snapshot'/source.name)
    jobs = [(split, i+1, spec['seed']+j*100000+i) for j,split in enumerate(SPLITS)
            for i in range(spec['split_counts'][split])]
    planned_seconds = sum(scene_profile(cfg, seed)[1] for _,_,seed in jobs)
    print(json.dumps(dict(status='planned', scenes=len(jobs), seconds=planned_seconds,
                          source_counts={k:len(v) for k,v in pools.items()})), flush=True)
    if args.plan_only:
        return
    pending = [job for job in jobs if not (args.output/'rows'/f'{job[0]}_{job[1]:04d}.stats.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
            initargs=(args.output, args.sources, args.hrir, cfg, pools, targets)) as executor:
        for result in executor.map(generate, pending):
            print(json.dumps(result), flush=True)
    rows = [json.loads((args.output/'rows'/f'{s}_{i:04d}.json').read_text()) for s,i,_ in jobs]
    stats = [json.loads((args.output/'rows'/f'{s}_{i:04d}.stats.json').read_text()) for s,i,_ in jobs]
    for split in SPLITS:
        with (args.output/f'manifest_{split}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                if row['split']==split: f.write(json.dumps(row,separators=(',',':'))+'\n')
    write_plan(args.output, targets)
    with (args.output/'recipes.jsonl').open('w', encoding='utf-8') as output:
        for row in rows:
            output.write((args.output/row['recipe_file']).read_text(encoding='utf-8'))
    summary = dict(status='rendered_pending_verification', clips=len(jobs), seconds=sum(r['duration_seconds'] for r in rows),
        split_counts=spec['split_counts'], source_manifest_sha256=identity['source_manifest_sha256'],
        hrir_sha256=identity['hrir_sha256'], results=stats,
        split_density={s:dict(Counter(r['density'] for r in rows if r['split']==s)) for s in SPLITS},
        limitations=spec['notes'])
    if cfg.get('scene_profiles'):
        summary['split_scene_profiles'] = {s:dict(Counter(r['sampling_profile'] for r in rows if r['split']==s)) for s in SPLITS}
    write_json(args.output/'summary.json', summary)
    print(json.dumps(dict(status='generation_complete',clips=len(rows),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':
    main()
