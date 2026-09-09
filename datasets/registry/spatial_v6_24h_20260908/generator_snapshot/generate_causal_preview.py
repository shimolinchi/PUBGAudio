"""Produce inspectable v5 random scenes plus separately marked coverage examples."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import causal_scene as plan
from causal_render import Library,render
import motion_synthesis as motion
from project_causal_labels import project


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,default=plan.serializable,separators=(',',':'))+'\n',encoding='utf-8')


def make_scenes(cfg,sources,count):
    scenes=[]
    for i in range(count):
        p=plan.Planner(cfg,sources,cfg['seed']+i)
        density=p.random_scene(); scenes.append(p.result(f'R{i+1:02d}',density))
    examples=['self_frag','external_flash','self_molotov','c4','curved_gait','coast_collision','tire','vehicle_fire','aircraft','cabin_mask']
    for i,name in enumerate(examples):
        c=copy.deepcopy(cfg)
        if name=='aircraft' and 'event_acoustics' in cfg:
            c['clip_seconds']=100  # At 65+ m/s both ends lie outside the estimated 3 km radius.
        if name in ['coast_collision','tire','vehicle_fire'] and 'event_acoustics' in cfg:
            c['distance_mode_weights']={'near_pass':1}  # Inspect quiet sub-events inside their radius.
        if name in ['curved_gait','coast_collision','tire','vehicle_fire']:
            c['curve_weights']={'bezier':1}
        if name=='coast_collision':
            c['duration_buckets_seconds']['vehicle']=[[28,32]]
            c['vehicle']['attack_probability_per_vehicle_with_motion']=0
        p=plan.Planner(c,sources,cfg['seed']+1000+i)
        if name=='self_frag': p.throwable(4,'self','frag')
        elif name=='external_flash': p.throwable(5,'external','flash',25)
        elif name=='self_molotov': p.throwable(5,'self','molotov')
        elif name=='c4': p.c4(4,'external',28)
        elif name=='curved_gait': p.footsteps(2,'external'); p.footsteps(20,'self')
        elif name=='coast_collision': p.vehicle(1,'external',True,None,True)
        elif name=='tire': p.vehicle(1,'external',True,'tire')
        elif name=='vehicle_fire': p.vehicle(1,'external',False,'fatal')
        elif name=='aircraft': p.aircraft()
        else: p.vehicle(1,'self'); p.footsteps(5,'external'); p.gun(20,'external')
        s=p.result(f'S{i+1:02d}','coverage','coverage_example_not_random_ratio');s['scenario']=name
        s['coverage_config_overrides']={key:c[key] for key in c if c[key]!=cfg[key]}
        scenes.append(s)
    return scenes


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--config',type=Path,default=Path('configs/generation_v5_3.json'))
    a.add_argument('--target-config',type=Path,default=Path('configs/training_targets_v5.json'))
    a.add_argument('--sources',type=Path,default=Path('datasets/sources-v5_3-final'))
    a.add_argument('--hrir',type=Path,default=Path('datasets/hrir/kemar-diffuse.zip'))
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--random-count',type=int,default=8)
    a.add_argument('--plan-only',action='store_true')
    args=a.parse_args();cfg=plan.config(args.config)
    targets=json.loads(args.target_config.read_text(encoding='utf-8'))
    source_manifest=json.loads((args.sources/'sources.json').read_text(encoding='utf-8'));sources=source_manifest['sources']
    args.output.mkdir(parents=True,exist_ok=True)
    scenes=make_scenes(cfg,sources,args.random_count)
    write_json(args.output/'config.json',cfg)
    write_json(args.output/'training_targets.json',targets)
    with (args.output/'recipes.jsonl').open('w',encoding='utf-8') as f:
        for s in scenes: f.write(json.dumps(s,ensure_ascii=False,default=plan.serializable,separators=(',',':'))+'\n')
    print('PLANNED',len(scenes),flush=True)
    if args.plan_only:return
    lib=Library(args.sources,sources)
    h=hashlib.sha256(args.hrir.read_bytes()).hexdigest();spatial=motion.SpatialRenderer(args.hrir,h)
    result=[]
    for s in scenes:
        stats=render(s,lib,spatial,cfg,args.output/'audio')
        with np.load(args.output/'audio'/(s['id']+'.npz'),allow_pickle=False) as z:
            projected=project(z,targets['external_targets'],targets['self_targets']);np.savez_compressed(args.output/'audio'/(s['id']+'_train3.npz'),**projected)
        result.append(stats);print('RENDERED',s['id'],stats,flush=True)
    counts=Counter(t['kind'] for s in scenes for t in s['tracks'])
    summary=dict(schema_version=5,status='rendered_pending_verification',clips=len(scenes),seconds=sum(s['seconds'] for s in scenes),
        random_clips=args.random_count,coverage_clips=len(scenes)-args.random_count,track_kinds=counts,results=result,
        source_manifest_sha256=hashlib.sha256((args.sources/'sources.json').read_bytes()).hexdigest(),hrir_sha256=h,
        limitations=['audition only: no train/test family split','C4 tempo approximation','molotov duration preview assumption','horizontal synthetic coordinates','incomplete 31-category coverage'])
    write_json(args.output/'summary.json',summary)
    snapshot=args.output/'generator_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['generate_causal_preview.py','causal_scene.py','body_actions.py','causal_render.py','footstep_acoustics.py','event_acoustics.py','listener_view.py','project_causal_labels.py','motion_synthesis.py','generate_mixed_dataset.py']:
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    write_json(args.output/'source_manifest.json',source_manifest)


if __name__=='__main__':main()
