"""Generate v4 curved scenes: cabin masking, short cruise, braking, collisions and weather."""
import argparse
from collections import Counter
import concurrent.futures
import copy
import json
from pathlib import Path
import shutil
import time
import wave
import numpy as np
import generate_self_audio_dataset as base
import scenario_synthesis as motion

STATE = {}


def configuration():
    cfg = base.configuration()
    cfg.update(dataset_id='pubgaudio_scenarios_v4_20260907',
        cabin_foot_keep_probability=.10, cabin_far_vehicle_keep_probability=.05,
        cabin_far_vehicle_distance=90., cabin_foot_sir_db=-12., cabin_vehicle_sir_db=-15.,
        minimum_sir_db=-30., weather_probability=.20, weather_only_probability=.05,
        collision_probability=.40, cruise_plateau_moving_time_fraction=.025,
        masking_threshold_status='configurable_synthetic_assumptions_not_measured_audibility',
        weather_label='nondirectional_background_negative_not_aircraft_or_vehicle')
    return cfg


def sources_and_splits(sources,cfg):
    regular = [s for s in sources if s['role'] not in ['collision','weather']]
    splitting, groups = base.source_pools(regular,cfg)
    auxiliaries = {'collision:common':'train','collision:brdm_light_dry':'validation',
        'collision:snowmobile':'test','weather:sandstorm':'train',
        'weather:wind_pillar':'validation','weather:mode_storm':'test'}
    for s in sources:
        if s['source_group_id'] in auxiliaries:
            split = auxiliaries[s['source_group_id']]
            splitting['source_to_split'][s['id']] = split
            splitting['group_to_split'][s['source_group_id']] = split
    splitting['method'] += '; collision and weather recording families assigned whole to distinct splits'
    return regular,splitting,groups


def adapt_scene(scene,sources,cfg,rng,force_weather=None):
    tracks = scene['tracks']
    own_car = any(t.get('source_role')=='self' and t['class_index']==1 for t in tracks)
    before = len(tracks)
    if own_car and scene['split'] != 'preview':
        kept=[]
        for t in tracks:
            probability=1.
            if t.get('source_role') != 'self':
                if t['class_index']==0:
                    probability=cfg['cabin_foot_keep_probability']
                elif t['class_index']==1 and motion.trajectory(t['trajectory'],30)['distance'].min()>=cfg['cabin_far_vehicle_distance']:
                    probability=cfg['cabin_far_vehicle_keep_probability']
            if rng.random()<probability:
                kept.append(t)
        tracks=scene['tracks']=kept
    scene['cabin_dropped_tracks']=before-len(tracks)
    collisions=[s for s in sources if s['role']=='collision']
    tyres=[s for s in sources if s['source_group_id'].startswith('tyres:')]
    by_id={s['id']:s for s in sources}
    for t in tracks:
        if t['class_index']==1:
            motion.action_vehicle(t,30,rng,tyres,collisions,cfg['collision_probability'])
        motion.curve_track(t,30,rng)
        for event in t['events']:
            if event['role']=='collision':
                event['duration_seconds']=by_id[event['source_id']]['duration_seconds']
    weather=[s for s in sources if s['role']=='weather']
    scene['backgrounds']=[]
    if weather and (force_weather is True or (force_weather is None and rng.random()<cfg['weather_probability'])):
        s=weather[int(rng.integers(len(weather)))]
        scene['backgrounds']=[dict(source_id=s['id'],source_group_id=s['source_group_id'],
            gain_db=float(rng.uniform(-8,2)),phase_samples=int(rng.integers(0,round(s['duration_seconds']*motion.RATE))),
            name=s['source_name'].rsplit('\\',1)[-1],source_role='background',direction_valid=False)]
    return scene


def plan(sources,cfg,splitting,groups,regular,preview=False):
    if preview:
        scenes=base.preview_plan(regular,cfg,groups)
        rng=np.random.default_rng(cfg['seed']+48000)
        for scene in scenes:
            adapt_scene(scene,sources,cfg,rng,force_weather=False)
        car=next(copy.deepcopy(t) for s in scenes for t in s['tracks'] if t['class_index']==1 and t.get('source_role')!='self')
        collision=next(s for s in sources if s['role']=='collision')
        car['trajectory'].update(initial_xy=[25.,25.],speed_knots=[[0,0],[30,0]])
        car['events']=[dict(source_id=collision['id'],role='collision',onset_sample=round(t*motion.RATE),
            duration_seconds=collision['duration_seconds'],gain=1.) for t in [3.,11.,21.]]
        car['collision_only']=True
        car['lifecycle']={key:31. for key in ['startup_seconds','move_seconds','final_stop_seconds','shutdown_seconds','engine_end_seconds']}
        car.pop('emergency_brake_seconds',None)
        scenes.append(dict(id='C01',title='碰撞对照：三次原生撞击，车辆标签，不标枪声',split='preview',density='low',
            duration_seconds=30,sample_rate_hz=motion.RATE,tracks=[car],backgrounds=[]))
        cruise=next(copy.deepcopy(t) for s in scenes for t in s['tracks'] if t['class_index']==1 and t.get('source_role')=='self')
        cruise['events']=[e for e in cruise['events'] if e['role']!='collision']
        cruise.pop('emergency_brake_seconds',None)
        life=cruise['lifecycle']
        cruise['trajectory']['speed_knots']=[[0,0],[life['move_seconds'],0],
            [life['move_seconds']+1.,10.],[life['final_stop_seconds']-1.,10.],[life['final_stop_seconds'],0],[30,0]]
        scenes.append(dict(id='C02',title='巡航比较：自身发动机，和 W01–W03 天气声对照',split='preview',density='high',
            duration_seconds=30,sample_rate_hz=motion.RATE,tracks=[cruise],backgrounds=[]))
        weather=[s for s in sources if s['source_group_id']=='weather:sandstorm']
        for i,source in enumerate(weather,1):
            scene=dict(id=f'W{i:02d}',title='沙尘暴原声：'+source['source_name'].rsplit('\\',1)[-1],
                split='preview',density='high',duration_seconds=30,sample_rate_hz=motion.RATE,tracks=[])
            scenes.append(adapt_scene(scene,[source],cfg,rng,force_weather=True))
        return scenes
    scenes=base.plan(regular,cfg,splitting,groups)
    rng=np.random.default_rng(cfg['seed']+48000)
    for scene in scenes:
        available=[s for s in sources if splitting['source_to_split'].get(s['id'])==scene['split']]
        if rng.random()<cfg['weather_only_probability']:
            scene.update(tracks=[],self_policy='weather_only',layout='weather_only')
            adapt_scene(scene,available,cfg,rng,force_weather=True)
        else:
            adapt_scene(scene,available,cfg,rng)
    return scenes


def init_worker(source_root,manifest,hrir,output,cfg,gain):
    STATE.clear()
    STATE.update(library=motion.SourceLibrary(source_root,manifest['sources']),
        renderer=motion.v3.SpatialRenderer(hrir,cfg['hrir_archive_sha256']),
        output=Path(output),cfg=cfg,gain=gain)


def render(scene):
    signals,paths=[],[]
    for meta in scene['tracks']:
        path=motion.trajectory(meta['trajectory'],scene['duration_seconds'])
        render_meta=dict(meta,class_index=2) if meta.get('collision_only') else meta
        if meta.get('source_role')=='self':
            wet=motion.v3.render_self(render_meta,path,STATE['library'],scene['duration_seconds'])
        else:
            dry=motion.v3.mono_track(render_meta,path,STATE['library'],scene['duration_seconds'])
            wet=STATE['renderer'].render(dry,path)
        signals.append(wet); paths.append(path)
    signals.extend(motion.render_background(b,STATE['library'],scene['duration_seconds']) for b in scene['backgrounds'])
    if not signals:
        raise ValueError('Empty scene: specify a background or an actor')
    return signals,paths


def peak_one(scene):
    signals,_=render(scene)
    mixed=np.sum(signals,axis=0,dtype=np.float32)
    temporary=STATE['output']/'_render_stage'/scene['id']
    np.save(temporary.with_suffix('.npy'),mixed,allow_pickle=False)
    np.save(temporary.with_suffix('.powers.npy'),motion.v3.frame_powers(signals,scene['duration_seconds']),allow_pickle=False)
    return float(np.max(np.abs(mixed)))


def write_one(scene):
    root=STATE['output']; stage=root/'_render_stage'; temporary=stage/scene['id']
    mixed=np.load(temporary.with_suffix('.npy'),allow_pickle=False)*STATE['gain']
    powers=np.load(temporary.with_suffix('.powers.npy'),allow_pickle=False)*STATE['gain']**2
    paths=[motion.trajectory(t['trajectory'],scene['duration_seconds']) for t in scene['tracks']]
    z=motion.frame_labels(scene['tracks'],paths,scene['backgrounds'],scene['duration_seconds'],STATE['cfg'],powers)
    peak=float(np.abs(mixed).max())
    if mixed.shape!=(round(scene['duration_seconds']*motion.RATE),2) or not np.isfinite(mixed).all() or not 0<peak<=.700002:
        raise ValueError('Invalid mixture '+scene['id'])
    audio=f"audio/{scene['split']}/{scene['id']}.wav"; label=f"labels/{scene['split']}/{scene['id']}.npz"
    with wave.open(str(root/audio),'wb') as w:
        w.setparams((2,2,motion.RATE,0,'NONE','not compressed'))
        w.writeframes(np.rint(mixed*32767).astype('<i2').tobytes())
    np.savez_compressed(root/label,**z)
    active=z['track_activity'].astype(bool)
    if np.count_nonzero(active.any(axis=1))!=len(scene['tracks']):
        raise ValueError('Missing actor activity '+scene['id'])
    exterior=active & (z['track_source_role'][:,None]==0)
    ranges=np.bincount(z['track_range_index'][exterior],minlength=3)
    row=dict(scene,audio_file=audio,label_file=label,audio_sha256=base.old.sha_file(root/audio),
        label_sha256=base.old.sha_file(root/label),master_gain=STATE['gain'],peak_dbfs=20*np.log10(peak),
        total_label_frames=len(z['frame_right_edge_seconds']),active_track_range_frame_counts=ranges.tolist(),
        hidden_active_track_frames=int(np.count_nonzero(active & ~z['track_observable'].astype(bool))),
        vehicle_motion_state_frames=np.bincount(z['track_motion_state'][z['track_class_index']==1].ravel(),minlength=5).tolist(),
        collision_events=sum(e['role']=='collision' for t in scene['tracks'] for e in t['events']))
    for p in [temporary.with_suffix('.npy'),temporary.with_suffix('.powers.npy')]:
        if p.resolve().parent!=stage.resolve() or not stage.resolve().is_relative_to(root.resolve()):
            raise ValueError('Invalid stage cleanup path')
        p.unlink()
    return row


def parallel(fn,scenes,args,manifest,cfg,gain):
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
        initargs=(str(args.sources),manifest,str(args.hrir),str(args.output),cfg,gain)) as pool:
        yield from pool.map(fn,scenes,chunksize=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['sources','hrir','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit-per-split',type=int)
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    args.sources,args.hrir,args.output=(x.resolve() for x in [args.sources,args.hrir,args.output])
    cfg=configuration()
    if args.limit_per_split:
        cfg['split_clip_counts']={s:args.limit_per_split for s in cfg['split_clip_counts']}
    manifest=json.loads((args.sources/'sources.json').read_text(encoding='utf-8'))
    regular,splitting,groups=sources_and_splits(manifest['sources'],cfg)
    scenes=plan(manifest['sources'],cfg,splitting,groups,regular,args.preview_only)
    if args.preview_only:
        cfg['split_clip_counts']={'preview':len(scenes)}
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.jsonl').exists():
        raise ValueError('Existing render: use a fresh output directory')
    hashes={p.name:base.old.sha_file(p) for p in Path(__file__).parent.glob('*.py')}
    identity=dict(config=cfg,generator_files_sha256=hashes,source_manifest_sha256=base.old.sha_file(args.sources/'sources.json'),
        source_root=str(args.sources),hrir_file=str(args.hrir),hrir_sha256=base.old.sha_file(args.hrir))
    for name,data in [('generation_identity.json',identity),('config.json',cfg),('source_manifest.json',manifest),('source_split.json',splitting)]:
        base.old.save_json(args.output/name,data)
    base.old.save_jsonl(args.output/'recipes.jsonl',scenes)
    print('PLANNED',len(scenes),'scenes',flush=True)
    if args.plan_only:
        return
    for s in manifest['sources']:
        if base.old.sha_file(args.sources/s['wav_file'])!=s['wav_sha256']:
            raise ValueError('Changed source '+s['id'])
    if shutil.disk_usage(args.output).free<len(scenes)*30*motion.RATE*8+3*1024**3:
        raise ValueError('Insufficient staging disk space')
    (args.output/'_render_stage').mkdir(exist_ok=True)
    for split in cfg['split_clip_counts']:
        for kind in ['audio','labels']:
            (args.output/kind/split).mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); largest=0.
    for i,peak in enumerate(parallel(peak_one,scenes,args,manifest,cfg,1.),1):
        largest=max(largest,peak)
        if i%25==0 or i==len(scenes):
            print('PEAK_SCAN',i,'/',len(scenes),'elapsed_s',round(time.monotonic()-started),flush=True)
    gain=min(.6,.7/largest)
    base.old.save_json(args.output/'master_gain.json',dict(gain=gain,unscaled_max_peak=largest,
        purpose='one_shared_gain_per_dataset_no_clip_or_ear_normalization'))
    rows=[]
    with (args.output/'manifest.jsonl').open('w',encoding='utf-8') as f:
        for i,row in enumerate(parallel(write_one,scenes,args,manifest,cfg,gain),1):
            rows.append(row); f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            if i%25==0 or i==len(scenes):
                f.flush(); print('RENDERED',i,'/',len(scenes),'elapsed_s',round(time.monotonic()-started),flush=True)
    for split in cfg['split_clip_counts']:
        base.old.save_jsonl(args.output/f'manifest_{split}.jsonl',[r for r in rows if r['split']==split])
    ranges=np.sum([r['active_track_range_frame_counts'] for r in rows],axis=0)
    summary=dict(status='rendered_pending_independent_verification',total_clips=len(rows),total_hours=len(rows)*30/3600,
        split_clip_counts=cfg['split_clip_counts'],master_gain=gain,far_active_track_frame_fraction=float(ranges[2]/max(1,ranges.sum())),
        hidden_active_track_frames=sum(r['hidden_active_track_frames'] for r in rows),
        cabin_dropped_tracks=sum(r.get('cabin_dropped_tracks',0) for r in rows),collision_events=sum(r['collision_events'] for r in rows),
        weather_scenes=sum(bool(r['backgrounds']) for r in rows),weather_only_scenes=sum(not r['tracks'] for r in rows),
        vehicle_motion_state_frames=np.sum([r['vehicle_motion_state_frames'] for r in rows],axis=0).tolist(),
        self_policy_counts=dict(Counter(r.get('self_policy','preview') for r in rows)),elapsed_seconds=round(time.monotonic()-started,2))
    base.old.save_json(args.output/'summary.json',summary)
    print('COMPLETE',len(rows),'scenes',flush=True)


if __name__=='__main__':
    main()
