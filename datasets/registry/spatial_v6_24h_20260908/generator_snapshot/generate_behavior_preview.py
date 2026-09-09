"""Forward self throws, more frequent view motion, and acceleration-rich vehicles."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import causal_scene as plan
from causal_render import Library,render
from motion_synthesis import SpatialRenderer,smoothstep
from project_causal_labels import project
from generate_causal_preview import write_json
from generate_attenuation_preview import fixed_view
from listener_view import bind_self_throw_direction


def make_scenes(cfg,sources):
    scenes=[]
    for i in range(6):
        p=plan.Planner(cfg,sources,cfg['seed']+i);density=p.random_scene()
        scenes.append(p.result(f'R{i+1:02d}',density))
    controls=[]
    for i,kind in enumerate(['frag','flash','molotov']):
        p=plan.Planner(cfg,sources,cfg['seed']+100+i);p.throwable(4,'self',kind)
        s=p.result(f'T{i+1:02d}','coverage','forward_throw_control_not_random_ratio')
        chain=s['chains'][0];v=fixed_view(s['seconds'],35);t=v['time']
        # Turn after release; the projectile and later fire stay fixed in world space.
        start=chain['release']+.3;finish=start+1.6
        v.update(mode='scripted',yaw_degrees_unwrapped=35+110*smoothstep((t-start)/1.6)-60*smoothstep((t-18)/2.5),
            turns=[dict(start=start,end=finish,from_degrees=35,to_degrees=145),dict(start=18,end=20.5,from_degrees=145,to_degrees=85)])
        s['listener_view']=v;bind_self_throw_direction(s)
        s.update(title={'frag':'自身手雷','flash':'自身闪光弹','molotov':'自身燃烧瓶'}[kind]+' · 向前投出后转头',
            description='出手时朝向35°，沿正前方投出；出手后向右转110°，落点保持世界位置。只把准备动作标为自身，爆炸/燃烧仍是外部位置声音。')
        controls.append(s)
    for i,role in enumerate(['external','self']):
        c=copy.deepcopy(cfg);c['distance_mode_weights']={'near_pass':1};c['curve_weights']={'bezier':1}
        c['duration_buckets_seconds']['vehicle']=[[34,40]];c['vehicle']['attack_probability_per_vehicle_with_motion']=0
        p=plan.Planner(c,sources,cfg['seed']+110+i);p.vehicle(1,role)
        s=p.result(f'V{i+1:02d}','coverage','acceleration_vehicle_control_not_random_ratio')
        s.update(title=('外部曲线车辆' if role=='external' else '自身车辆')+' · 加速过程与短暂停留',
            coverage_config_overrides={k:c[k] for k in c if c[k]!=cfg[k]})
        controls.append(s)
    c=copy.deepcopy(cfg);c['clip_seconds']=120;c['listener_view']['mode_weights']={'natural':1}
    p=plan.Planner(c,sources,cfg['seed']+120);p.footsteps(3,'external');p.throwable(24,'self','molotov')
    p.vehicle(43,'external');p.throwable(85,'self','flash');p.footsteps(96,'self')
    s=p.result('L01','coverage','long_scene_coverage_not_random_ratio')
    s.update(title='两分钟 · 频繁轻晃、大转向与向前投掷',description='按新版概率生成视角和车辆状态。燃烧瓶与闪光弹均按出手瞬间的前方发射；投出后转头不移动落点。',
        coverage_config_overrides={'clip_seconds':120,'listener_view':c['listener_view']})
    controls.append(s)
    c=copy.deepcopy(cfg);c['clip_seconds']=100;c['listener_view']['mode_weights']={'natural':1}
    p=plan.Planner(c,sources,cfg['seed']+130);p.aircraft();s=p.result('A01','coverage','coverage_example_not_random_ratio')
    s.update(title='运输机直线通场 · 频繁视角活动',coverage_config_overrides={'clip_seconds':100,'listener_view':c['listener_view']})
    controls.append(s)
    # Retain a deliberately rare coasting example for coverage, outside random ratios.
    c=copy.deepcopy(cfg);c['duration_buckets_seconds']['vehicle']=[[32,36]];c['distance_mode_weights']={'near_pass':1}
    c['vehicle']['attack_probability_per_vehicle_with_motion']=0
    p=plan.Planner(c,sources,cfg['seed']+140);p.vehicle(1,'external',True,None,True)
    s=p.result('V03','coverage','rare_coast_control_not_random_ratio')
    s.update(title='少量保留的短滑车 · 碰撞对照',coverage_config_overrides={k:c[k] for k in c if c[k]!=cfg[k]})
    controls.append(s)
    return controls+scenes


def main(default_config='configs/generation_v5_2.json',default_sources='datasets/sources-v5',extra_snapshot=()):
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--config',type=Path,default=Path(default_config))
    a.add_argument('--sources',type=Path,default=Path(default_sources));a.add_argument('--hrir',type=Path,default=Path('datasets/hrir/kemar-diffuse.zip'))
    args=a.parse_args();cfg=plan.config(args.config)
    if args.output.exists():raise ValueError('Choose a new directory; never overwrite reviewed samples')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.sources/'sources.json').read_text(encoding='utf-8'));scenes=make_scenes(cfg,manifest['sources'])
    targets=json.loads(Path('configs/training_targets_v5.json').read_text(encoding='utf-8'))
    for name,data in [('config',cfg),('source_manifest',manifest),('training_targets',targets)]:write_json(args.output/(name+'.json'),data)
    with (args.output/'recipes.jsonl').open('w',encoding='utf-8') as f:
        for s in scenes:f.write(json.dumps(s,ensure_ascii=False,default=plan.serializable,separators=(',',':'))+'\n')
    h=hashlib.sha256(args.hrir.read_bytes()).hexdigest();lib=Library(args.sources,manifest['sources']);spatial=SpatialRenderer(args.hrir,h);stats=[]
    for s in scenes:
        stats.append(render(s,lib,spatial,cfg,args.output/'audio'))
        with np.load(args.output/'audio'/(s['id']+'.npz')) as z:np.savez_compressed(args.output/'audio'/(s['id']+'_train3.npz'),**project(z))
        print('RENDERED',s['id'],flush=True)
    write_json(args.output/'summary.json',dict(revision=cfg['revision'],status='rendered_pending_verification',clips=len(scenes),seconds=sum(s['seconds'] for s in scenes),results=stats,
        hrir_sha256=h,source_manifest_sha256=hashlib.sha256((args.sources/'sources.json').read_bytes()).hexdigest(),limitations=['audition only; same v5.1 acoustic calibration limitations','horizontal listener rotation only','formal long-history training not implemented']))
    snapshot=args.output/'generator_snapshot';snapshot.mkdir()
    for name in ['generate_behavior_preview','generate_attenuation_preview','generate_causal_preview','causal_scene','causal_render','listener_view','event_acoustics','footstep_acoustics','project_causal_labels','motion_synthesis','generate_mixed_dataset',*extra_snapshot]:
        shutil.copy2(Path(__file__).with_name(name+'.py'),snapshot/(name+'.py'))
    print('DONE',len(scenes),flush=True)


if __name__=='__main__':main()
