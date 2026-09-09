"""v5.1 audition: finite ranges, continuous listener yaw and paired controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import causal_scene as plan
from generate_causal_preview import make_scenes,write_json
from causal_render import Library,render
from motion_synthesis import SpatialRenderer,smoothstep
from project_causal_labels import project
from footstep_acoustics import gain_at_distance


def fixed_view(seconds,yaw=0):
    t=np.arange(round(seconds/.01)+1)*.01
    return dict(time=t,yaw_degrees_unwrapped=np.full(len(t),yaw),mode='fixed',turns=[],translation='stationary_world_origin',pitch_degrees=0)


def controls(cfg,sources,scenes):
    controls=[]
    for i,gait in enumerate(['walk','run','sprint']):
        c=copy.deepcopy(cfg);c['clip_seconds']=40
        p=plan.Planner(c,sources,cfg['seed']+2000+i)
        ids=[s['id'] for s in sources if s['role']==gait and s['source_group_id']=='footsteps:Concrete'][:6]
        for j,distance in enumerate([5,10,20,30,35,40,45,50,60,80]):
            tr=p.base_track('footsteps','external',gait,'footsteps:Concrete');tr['surface']='Concrete';tr['gain_db']=0
            p.fixed(tr,[0,distance]);start=j*4+.5;tr.update(start=start,end=start+3)
            tr['states']=[dict(start=start,end=start+3,state=gait)]
            for k in range(6):
                p.event(tr,gait,start+k*.45,source_id=ids[k%len(ids)],simulated_foot=['left','right'][k%2])
        s=p.result(f'D{i+1:02d}','coverage','calibration_control_not_random_not_training')
        s.update(title={'walk':'行走','run':'跑步','sprint':'冲刺'}[gait]+' · 5–80 米音量阶梯',
            description='每 4 秒换一个固定距离：5 / 10 / 20 / 30 / 35 / 40 / 45 / 50 / 60 / 80 推定米。同一组混凝土脚步、同一增益、正前方；这是测量对照，不模拟人物瞬移，也不计入随机比例。',
            listener_view=fixed_view(40))
        controls.append(s)
    plane=next(s for s in scenes if s.get('scenario')=='aircraft')
    plane['listener_view']=fixed_view(plane['seconds'])
    plane['title']='运输机 · 世界直线 · 视角不变'
    moving=copy.deepcopy(plane);moving['id']='D04';moving['title']='同一运输机 · 世界直线 · 玩家转头'
    t=np.asarray(moving['listener_view']['time'])
    yaw=100*smoothstep((t-20)/4)-170*smoothstep((t-45)/2)+70*smoothstep((t-70)/5)
    moving['listener_view'].update(mode='scripted',yaw_degrees_unwrapped=yaw,
        turns=[dict(start=20,end=24,from_degrees=0,to_degrees=100),dict(start=45,end=47,from_degrees=100,to_degrees=-70),dict(start=70,end=75,from_degrees=-70,to_degrees=0)])
    moving['selection']='paired_listener_control_not_random_ratio'
    moving['description']='与 S09 的原声、世界轨迹、距离和增益完全相同，只改变玩家朝向。20–24、45–47、70–75 秒转头。世界轨迹保持直线；相对耳朵的角度连续变化。'
    controls.extend([plane,moving])
    c=copy.deepcopy(cfg);c['clip_seconds']=30
    p=plan.Planner(c,sources,cfg['seed']+2010);p.gun(2,'external',np.array([0.,35.]));p.gun(13,'external',np.array([35.,0.]))
    s=p.result('D05','coverage','coverage_example_not_random_ratio');v=fixed_view(30);t=v['time']
    v.update(mode='scripted',yaw_degrees_unwrapped=90*smoothstep((t-4)/1.5)-180*smoothstep((t-15)/2),
        turns=[dict(start=4,end=5.5,from_degrees=0,to_degrees=90),dict(start=15,end=17,from_degrees=90,to_degrees=-90)])
    s.update(listener_view=v,title='固定枪口 · 玩家向右 / 向左转头',description='枪口世界位置固定，玩家水平转头；训练方位和音频同步更新。枪声原生截止距离尚未确认。')
    controls.append(s)
    c=copy.deepcopy(cfg);c['clip_seconds']=120;c['listener_view']['mode_weights']={'natural':1}
    p=plan.Planner(c,sources,cfg['seed']+2020)
    p.footsteps(4,'external');p.throwable(25,'self','molotov');p.footsteps(58,'external')
    p.vehicle(76,'external',force_coast=True);p.footsteps(95,'self')
    s=p.result('D06','coverage','long_scene_coverage_not_random_ratio')
    s.update(title='两分钟连续场景 · 轻晃、转向与停留',description='视角按自然概率生成：以小幅晃动为主，间有转向和停留；包含曲线脚步、燃烧瓶完整过程、滑车与自身声。长音频供后续从同一时间线截取历史窗口；本批未用于训练。',
             coverage_config_overrides={'clip_seconds':120,'listener_view':c['listener_view']})
    controls.append(s)
    return controls+[s for s in scenes if s is not plane]


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--config',type=Path,default=Path('configs/generation_v5_1.json'))
    a.add_argument('--sources',type=Path,default=Path('datasets/sources-v5'))
    a.add_argument('--hrir',type=Path,default=Path('datasets/hrir/kemar-diffuse.zip'))
    args=a.parse_args();cfg=plan.config(args.config)
    if args.output.exists():raise ValueError('Use a fresh output directory; preserve earlier auditions and feedback')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.sources/'sources.json').read_text(encoding='utf-8'));sources=manifest['sources']
    scenes=controls(cfg,sources,make_scenes(cfg,sources,6))
    with (args.output/'recipes.jsonl').open('w',encoding='utf-8') as f:
        for s in scenes:f.write(json.dumps(s,ensure_ascii=False,default=plan.serializable,separators=(',',':'))+'\n')
    targets=json.loads(Path('configs/training_targets_v5.json').read_text(encoding='utf-8'))
    for name,data in [('config',cfg),('source_manifest',manifest),('training_targets',targets)]:write_json(args.output/(name+'.json'),data)
    lib=Library(args.sources,sources);h=hashlib.sha256(args.hrir.read_bytes()).hexdigest();spatial=SpatialRenderer(args.hrir,h)
    stats=[]
    for scene in scenes:
        result=render(scene,lib,spatial,cfg,args.output/'audio')
        with np.load(args.output/'audio'/(scene['id']+'.npz'),allow_pickle=False) as z:
            np.savez_compressed(args.output/'audio'/(scene['id']+'_train3.npz'),**project(z,targets['external_targets'],targets['self_targets']))
        stats.append(result);print('RENDERED',scene['id'],json.dumps(result),flush=True)
    snapshot=args.output/'generator_snapshot';snapshot.mkdir()
    names=['generate_attenuation_preview','generate_causal_preview','causal_scene','causal_render','listener_view','event_acoustics','footstep_acoustics','project_causal_labels','motion_synthesis','generate_mixed_dataset']
    for name in names:shutil.copy2(Path(__file__).with_name(name+'.py'),snapshot/(name+'.py'))
    write_json(args.output/'summary.json',dict(revision=cfg['revision'],status='rendered_pending_verification',clips=len(scenes),seconds=sum(s['seconds'] for s in scenes),results=stats,
        hrir_sha256=h,source_manifest_sha256=hashlib.sha256((args.sources/'sources.json').read_bytes()).hexdigest(),
        limitations=['audition only; not a train/test release','metre conversion and gait mapping assumed','gain shape not native curve','some event radii missing; unknown fallback documented','horizontal rotation only; no listener translation/pitch','no original near/mid/far layer reconstruction']))
    rows=[]
    for d in [5,10,20,30,35,40,45,50,55,60,80]:
        rows.append(dict(distance_m_estimate=d,**{g:float(gain_at_distance(d,cfg['footstep_acoustics'],g)) for g in cfg['footstep_acoustics']['profiles']}))
    write_json(args.output/'footstep_gain_table.json',rows)
    print('DONE',len(scenes),flush=True)


if __name__=='__main__':main()
