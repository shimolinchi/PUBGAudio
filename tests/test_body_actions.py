import copy
import json
from pathlib import Path
import numpy as np
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/synthesis'))
from causal_scene import Planner,config
from causal_render import labels
from project_causal_labels import project
from pubg_audio.dataset import aligned_targets
from verify_causal_preview import verify_chain
from prepare_body_sources import TRANSITION_MEDIA_IDS


@pytest.fixture
def setup():
    src=ROOT/'datasets/sources-v5_3-final/sources.json'
    if not src.exists():pytest.skip('Native local client sources are not distributed')
    return config(ROOT/'configs/generation_v5_3.json'),json.loads(src.read_text(encoding='utf-8'))['sources']


def test_weights_reduce_only_walk_not_actor_spawn():
    old=config(ROOT/'configs/generation_v5_2.json');new=config(ROOT/'configs/generation_v5_3.json')
    assert old['spawn_probability_per_eligible_step']==new['spawn_probability_per_eligible_step']
    a=old['footsteps']['initial_weights'];b=new['footsteps']['initial_weights']
    assert sum(a.values())==sum(b.values())
    assert b['walk']<a['walk'] and b['run']>a['run']
    assert all(b[k]==a[k] for k in ['sprint','crouch_walk','prone_crawl'])


def test_shared_rustle_uses_event_referenced_media_ids_not_name_aliases(setup):
    _,sources=setup
    assert {s['media_id'] for s in sources if s['role']=='transition_rustle'}==TRANSITION_MEDIA_IDS


@pytest.mark.parametrize('action',['crouch_cycle','prone_cycle','vault'])
@pytest.mark.parametrize('role',['external','self'])
def test_linked_actions_preserve_material_pose_and_target_projection(setup,action,role):
    cfg,sources=setup;cfg=copy.deepcopy(cfg)
    cfg['duration_buckets_seconds']['footsteps']=[[32,34]]
    cfg['footsteps']['initial_weights']={'run':1}
    cfg['body_actions']['after_bout_weights']={k:float(k==action) for k in cfg['body_actions']['after_bout_weights']}
    p=Planner(cfg,sources,731);tr=p.footsteps(1,role);scene=p.result('B','coverage')
    verify_chain(scene,cfg)
    linked=[x for x in p.tracks if x.get('actor_parent')==tr['id']]
    assert linked
    assert all(p.by_id[e['source_id']]['source_group_id']==tr['source_group_id'] for e in tr['events'])
    assert all(x['source_role']==role for x in linked)
    if action!='vault':
        assert linked[0]['posture_from']=='standing' and linked[-1]['posture_to']=='standing'
        for a,b in zip(linked,linked[1:]):assert a['posture_to']==b['posture_from']
        assert all(x['acoustic_identity']=='shared_transition_rustle' for x in linked)
        for x in linked:
            ids=(p.t>=x['start'])&(p.t<=x['end'])
            assert tr['path']['speed'][ids].max(initial=0)==0
    count=(round(cfg['clip_seconds']*32000)-1024)//320+1
    powers=np.zeros((len(p.tracks),count));powers[1:]=.001
    z=labels(scene,powers,cfg);targets=project(z)
    assert np.any(z['track_body_action']!='none')
    assert np.all(z['track_actor_id']==tr['id'])
    aligned=aligned_targets(targets,0,cfg['clip_seconds'])
    assert not aligned['counts'].any() and not targets['self_activity'].any()
    if role=='self':assert not z['track_direction_valid'].any()


def test_missing_native_action_media_does_not_create_fake_sounds(setup):
    cfg,sources=setup;sources=[s for s in sources if s['class_index']!=4]
    p=Planner(cfg,sources,17);tr=p.footsteps(1)
    assert not p.chains and not any(t.get('actor_parent') for t in p.tracks)
    assert all(s['state'] in {'walk','run','sprint','crouch_walk','prone_crawl','pause'} for s in tr['states'])


def test_random_scene_actions_and_geometry_stay_in_bounds(setup):
    cfg,sources=setup
    for seed in range(15):
        p=Planner(cfg,sources,seed);density=p.random_scene();s=p.result(str(seed),density)
        verify_chain(s,cfg)
        for tr in s['tracks']:
            if tr.get('actor_parent'):assert tr['end']<=cfg['clip_seconds']
