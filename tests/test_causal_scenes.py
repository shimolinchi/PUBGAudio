import json
from pathlib import Path
import sys
import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools/synthesis'
sys.path.insert(0,str(TOOLS))
import causal_scene as cs
from causal_render import labels
from project_causal_labels import project
from pubg_audio.dataset import aligned_targets


@pytest.fixture
def planner():
    root=TOOLS.parents[1]
    source=root/'datasets/sources-v5/sources.json'
    if not source.exists(): pytest.skip('Local client assets not shipped in source repository')
    return cs.Planner(cs.config(root/'configs/generation_v5.json'),json.loads(source.read_text(encoding='utf-8'))['sources'],718)


def test_throwable_timer_and_direct_self_only(planner):
    p=planner
    for kind in ['frag','flash','molotov']: p.throwable(3,'self',kind)
    for c in p.chains:
        prep=next(t for t in p.tracks if t['id']==c['prepare'])
        fx=next(t for t in p.tracks if t['id']==c['effect'])
        assert prep['source_role']=='self' and fx['source_role']=='external'
        assert c['pin_or_ignite']<c['release']<=c['detonate']
        assert np.linalg.norm(np.array(c['release_xy'])-c['detonation_xy'])>1
        if c['kind']=='frag': assert c['detonate']==c['fuse_deadline']
        if c['kind']=='flash': assert c['detonate']==min(c['fuse_deadline'],c['first_impact']+.7)
        if c['kind']=='molotov': assert next(e for e in fx['events'] if e['role']=='molotov_fire')['duration']==10


def test_c4_complete_countdown(planner):
    p=planner;p.c4(4,'external',20);c=p.chains[0]
    assert c['activation']==8 and c['detonate']==24
    intervals=np.diff(c['beep_times'])
    assert np.all(intervals[1:]<=intervals[:-1]+1e-8)
    assert c['beep_times'][0]==c['activation'] and c['beep_times'][-1]<c['detonate']


def test_aircraft_no_stop_or_reversal(planner):
    t=planner.aircraft(); xy=t['path']['xy']; delta=np.diff(xy,axis=0)
    assert np.allclose(delta,delta[0])
    assert np.linalg.norm(xy[0])>1500 and np.linalg.norm(xy[-1])>1500
    assert np.min(t['path']['speed'])>60


def test_coasting_retains_speed_and_attack_has_cause(planner):
    p=planner;t=p.vehicle(1,'external',True,'fatal',True)
    path=t['path']; assert np.any((path['engine_on']==0)&(path['speed']>1))
    c=next(c for c in p.chains if c['kind']=='vehicle_attack')
    assert c['detonate']-c['engine_failure']==pytest.approx(5)
    source=next(t for t in p.tracks if t['id']==c['shooter'])
    assert max(e['start'] for e in source['events'])<c['hit']
    assert not any(e['role'].startswith('startup') and e['start']>=c['engine_failure'] for e in t['events'])
    assert next(t for t in p.tracks if t['id']==c['explosion_track'])['kind']=='explosion'


def test_material_and_continuous_geometry(planner):
    t=planner.footsteps(1);p=t['path'];xy=p['xy'];v=p['speed']
    actual=np.linalg.norm(np.diff(xy,axis=0),axis=1)/.01
    assert np.max(np.abs(actual-(v[:-1]+v[1:])/2))<.015
    for e in t['events']:
        assert planner.by_id[e['source_id']]['source_group_id']==t['source_group_id']
    bouts=[s for s in t['states'] if s['state']!='pause']
    assert all(s['end']-s['start']>=3 for s in bouts[:-1])


def test_empty_scene_and_confusers_do_not_become_targets():
    cfg={'masking':{'minimum_sir_db':-30,'cabin_foot_sir_db':-12,'cabin_far_vehicle_sir_db':-15,'cabin_far_distance_units':90}}
    count=(32000-1024)//320+1
    z=labels(dict(tracks=[],seconds=1,classes=cs.KINDS),np.empty((0,count)),cfg)
    out=aligned_targets(project(z),0,1)
    assert out['counts'].sum()==0
    tr=dict(id='E1',kind='explosion',class_index=5,source_role='external',subtype='frag',states=[],
            path=dict(time=[0,1],xy=[[20,10],[20,10]],speed=[0,0],engine_on=[0,0],brake=[0,0]))
    z=labels(dict(tracks=[tr],seconds=1,classes=cs.KINDS),np.ones((1,count))*.01,cfg)
    assert z['track_activity'].all()
    assert aligned_targets(project(z),0,1)['counts'].sum()==0


def test_interference_masks_hidden_targets_without_erasing_details():
    cfg={'masking':{'minimum_sir_db':-30,'cabin_foot_sir_db':-12,'cabin_far_vehicle_sir_db':-15,'cabin_far_distance_units':90}}
    count=(32000-1024)//320+1
    common=dict(states=[],path=dict(time=[0,1],xy=[[120,0],[120,0]],speed=[0,0],engine_on=[0,0],brake=[0,0]))
    foot=dict(common,id='F',kind='footsteps',class_index=0,source_role='external',subtype='walk')
    blast=dict(common,id='B',kind='explosion',class_index=5,source_role='external',subtype='frag')
    z=labels(dict(tracks=[foot,blast],seconds=1,classes=cs.KINDS),np.array([[1e-5]*count,[1.]*count]),cfg)
    assert z['track_activity'][0].all() and not z['track_observable'][0].any()
    targets=aligned_targets(project(z),0,1)
    assert targets['counts'][:,0].min()==1 and targets['mask'][:,0].max()==0
    assert targets['counts'][:,1:].sum()==0
    disabled=aligned_targets(project(z,selected=('vehicle',),self_selected=()),0,1)
    assert disabled['mask'][:,0].max()==0 and disabled['self_mask'].max()==0


def test_weapon_bouts_respect_finite_ammunition(planner):
    for _ in range(8):
        tr=planner.gun(1)
        assert tr['shots_fired']==len(tr['events'])<=tr['magazine_limit']


def test_vehicle_does_not_shut_down_before_startup_finishes(planner):
    for _ in range(12):
        tr=planner.vehicle(1,force_coast=True)
        starting=[s for s in tr['states'] if s['state']=='starting']
        for e in tr['events']:
            if e['role'].startswith('shutdown'):
                assert not any(s['start']<=e['start']<s['end'] for s in starting)
