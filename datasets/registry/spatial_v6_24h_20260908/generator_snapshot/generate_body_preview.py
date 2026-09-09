"""Audition reduced walking, increased running, and linked posture/vault actions."""
import copy
import causal_scene as plan
import generate_behavior_preview as base


def make_scenes(cfg,sources):
    scenes=[]
    names={'crouch_cycle':'停步蹲下 → 保持 → 起身 → 继续移动',
           'prone_cycle':'停步趴下 → 保持 → 站起 → 继续移动',
           'vault':'接近障碍 → 抓扶翻越 → 接触 → 继续移动'}
    for j,action in enumerate(names):
        for i,role in enumerate(['external','self']):
            c=copy.deepcopy(cfg);c['clip_seconds']=40;c['duration_buckets_seconds']['footsteps']=[[30,34]]
            c['footsteps']['initial_weights']={'walk':.75,'run':1.25}
            c['body_actions']['after_bout_weights']={'continue':0,action:1}
            c['distance_mode_weights']={'near_pass':1};c['curve_weights']={'bezier':1}
            c['listener_view']['mode_weights']={'natural':1}
            # Supply zero weights for all named choices, preserving auditable legal filtering.
            for k in cfg['body_actions']['after_bout_weights']:c['body_actions']['after_bout_weights'].setdefault(k,0)
            p=plan.Planner(c,sources,cfg['seed']+100+j*2+i);p.footsteps(1,role)
            s=p.result(f'{["C","P","V"][j]}{i+1:02d}','coverage','body_action_control_not_random_ratio')
            s.update(title=('自身 · ' if role=='self' else '外部曲线 · ')+names[action],
                description='专项覆盖，不计随机比例。姿态标签来自生成计划；蹲起/趴起共用原生转换衣物声，不能据此宣称音色能区分细动作。翻越使用原生抓扶、攀越、接触和衣物素材。',
                coverage_config_overrides={k:c[k] for k in c if c[k]!=cfg[k]})
            scenes.append(s)
    for i in range(4):
        p=plan.Planner(cfg,sources,cfg['seed']+i);density=p.random_scene()
        scenes.append(p.result(f'R{i+1:02d}',density))
    return scenes


if __name__=='__main__':
    base.make_scenes=make_scenes
    base.main('configs/generation_v5_3.json','datasets/sources-v5_3-final',('body_actions','generate_body_preview'))
