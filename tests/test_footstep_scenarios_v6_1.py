import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/synthesis'))
import causal_render
from causal_scene import Planner, serializable, KINDS
from configure_footsteps_v6_1 import build
from generate_light_dataset import configuration, source_partitions
from footstep_scenarios import scene_profile, scene_configuration, populate_footsteps_only, add_vehicle_admission_draws
from vehicle_footstep_guard import keep_track
from verify_causal_preview import verify_chain
from project_causal_labels import project
from pubg_audio.dataset import aligned_targets


@pytest.fixture
def cfg():
    return configuration(json.loads((ROOT/'configs/dataset_v6_1.json').read_text(encoding='utf-8')))


def fake_sources():
    return [dict(id=f'{gait}{index}',role=gait,source_group_id='footsteps:Concrete',
                 class_index=0,duration_seconds=.18) for gait in ['walk','run','sprint'] for index in range(4)]


def test_config_preparation_is_reproducible_and_retains_acoustics(cfg):
    base=json.loads((ROOT/'configs/generation_v6.json').read_text())
    generated,spec=build()
    assert generated==json.loads((ROOT/'configs/generation_v6_1.json').read_text())
    assert spec==json.loads((ROOT/'configs/dataset_v6_1.json').read_text())
    for key in ['audio','footstep_acoustics','air_absorption','gunfire']:
        assert cfg[key]==base[key]
    assert cfg['spawn_probability_per_eligible_step']['low']['footsteps']==pytest.approx(.008)
    assert cfg['external_footsteps']['self_probability_when_eligible']==.3


def test_profile_selection_is_seeded_and_durations_are_not_assumed_uniform(cfg):
    selected=[scene_profile(cfg,i) for i in range(2000)]
    assert selected==[scene_profile(cfg,i) for i in range(2000)]
    pure=sum(name=='external_footsteps_only' for name,_ in selected)
    assert 450<pure<550
    assert sum(seconds for _,seconds in selected)==(2000-pure)*120+pure*240
    for name,seconds in selected:
        assert seconds==(240 if name=='external_footsteps_only' else 120)
    broken=copy.deepcopy(cfg);broken['scene_profiles']['weights']['mixed']=-1
    with pytest.raises(ValueError,match='weights'):scene_profile(broken,1)


@pytest.mark.parametrize('shape',['oval','figure_eight','bezier_loop'])
@pytest.mark.parametrize('source_count',[1,2])
def test_long_pure_scene_has_continuous_bounded_sources_and_view(cfg,shape,source_count):
    cfg=copy.deepcopy(cfg)
    cfg['scene_profiles']['weights']={'external_footsteps_only':1}
    focus=cfg['scene_profiles']['external_footsteps_only']
    focus['source_count_weights']={str(source_count):1}
    focus['curve_weights']={shape:1}
    focus['distance_mode_weights']={'near_far':1}
    focus['listener_view']['mode_weights']={'natural':1}
    local,profile=scene_configuration(cfg,937)
    p=Planner(local,fake_sources(),937)
    density=populate_footsteps_only(p);add_vehicle_admission_draws(p)
    scene=p.result('in-memory',density)
    assert scene['seconds']==240 and len(scene['tracks'])==source_count
    verify_chain(scene,local)
    for tr in scene['tracks']:
        assert tr['kind']=='footsteps' and tr['source_role']=='external'
        assert len({e['role'] for e in tr['events']})==1
        assert all(p.by_id[e['source_id']]['source_group_id']==tr['source_group_id'] for e in tr['events'])
        assert tr['end']-tr['start']>200 and len(tr['events'])>150
        assert any(s['state']=='pause' for s in tr['states'])
        xy=np.asarray(tr['path']['xy']);t=np.asarray(tr['path']['time'])
        d=np.linalg.norm(xy,axis=1)/tr['distance_reference_radius_m_estimate']
        assert .1499<=d.min()<.25 and .75<d.max()<=.8501
        stopped=(np.asarray(tr['path']['speed'])[:-1]==0)&(np.asarray(tr['path']['speed'])[1:]==0)
        assert np.max(np.linalg.norm(np.diff(xy,axis=0)[stopped],axis=1),initial=0)<1e-8
    view=scene['listener_view'];turns=view['turns']
    assert len(turns)>80 and any(t['kind']=='large_turn' for t in turns)
    assert np.max(abs(np.diff(view['yaw_degrees_unwrapped'])/.01))<=140.01
    if source_count==2:
        assert scene['tracks'][0]['start']<scene['tracks'][1]['start']
        assert not np.allclose(scene['tracks'][0]['path']['xy'],scene['tracks'][1]['path']['xy'])
    assert cfg['clip_seconds']==120  # no per-scene config mutation leaks to siblings


def test_vehicle_guard_uses_actual_power_and_has_one_track_level_draw(cfg):
    foot=np.full(100,1e-6);loud=np.full(100,1e-4);quiet=np.full(100,1e-8);zero=np.zeros(100)
    tr={'id':'foot','vehicle_overlap_admission_draw':.5}
    assert not keep_track(tr,foot,loud,zero,cfg)[0]
    assert keep_track(tr,foot,quiet,zero,cfg)[0]
    assert not keep_track(tr,foot,quiet,quiet,cfg)[0]  # self even when engine quiet/coasting
    assert keep_track(dict(tr,vehicle_overlap_admission_draw=.015),foot,loud,zero,cfg)[0]
    assert not keep_track(dict(tr,vehicle_overlap_admission_draw=.015),foot,loud,loud,cfg)[0]
    # A single transient does not discard a long footstep actor.
    brief=zero.copy();brief[:20]=1e-3
    assert keep_track(tr,foot,brief,zero,cfg)[0]


def fake_scene(draw=.5,own=False):
    def track(identifier,kind,role,level,start):
        return dict(id=identifier,kind=kind,class_index=0 if kind=='footsteps' else 1,
            source_role=role,subtype='walk' if kind=='footsteps' else 'car',surface='Concrete',
            states=[dict(start=start,end=3,state='walk' if kind=='footsteps' else 'accelerate')],
            events=[dict(role='walk',start=0,duration=3)] if kind=='footsteps' else [],
            path=dict(time=[0,3],xy=[[0,0],[0,0]] if role=='self' else [[20,0],[20,0]],
                speed=[0,0],engine_on=[0,0],brake=[0,0]),level=level,start=start,end=3,
            vehicle_overlap_admission_draw=draw)
    return dict(id='memory-only',seconds=3,classes=KINDS,
        tracks=[track('F','footsteps','external',.001,0),track('V','vehicle','self' if own else 'external',.015,1)],
        decisions=[],chains=[])


def test_later_vehicle_removes_whole_foot_actor_and_keeps_label_order(cfg,monkeypatch,tmp_path):
    import wave
    captured=[];buffers=[]
    real_open=wave.open
    def open_in_memory(*args,**kwargs):
        buffer=io.BytesIO();buffers.append(buffer);return real_open(buffer,'wb')
    monkeypatch.setattr(causal_render.wave,'open',open_in_memory)
    monkeypatch.setattr(causal_render.np,'savez_compressed',lambda path,**labels:captured.append(labels))
    def render_signal(track,library,spatial,seconds,cfg=None):
        t=np.arange(round(seconds*44100))/44100
        signal=(np.sin(2*np.pi*440*t)*track['level']*(t>=track['start'])).astype(np.float32)
        return np.column_stack((signal,signal))
    monkeypatch.setattr(causal_render,'render_track',render_signal)
    for reverse in [False,True]:
        scene=fake_scene()
        if reverse:scene['tracks'].reverse()
        causal_render.render(scene,None,None,cfg,tmp_path)
        assert [tr['id'] for tr in scene['tracks']]==['V']
        assert captured[-1]['track_id'].tolist()==['V']
        assert len(scene['decisions'])==1 and not scene['decisions'][0]['selected']
    assert buffers[0].getvalue()==buffers[1].getvalue()
    # A rare retained conflicting actor has detailed activity but masked supervision.
    scene=fake_scene(draw=.005,own=True)
    causal_render.render(scene,None,None,cfg,tmp_path)
    z=captured[-1]
    assert z['track_activity'][0].any() and z['track_vehicle_masked'][0].any()
    assert not z['track_observable'][0,z['track_vehicle_masked'][0].astype(bool)].any()
    targets=aligned_targets(project(z),0,3)
    assert targets['counts'][15:25,0].min()==1 and targets['mask'][15:25,0].max()==0
    # Rendering the accepted recipe again must not add decisions or change PCM/labels.
    previous=len(scene['decisions']);pcm=buffers[-1].getvalue()
    causal_render.render(scene,None,None,cfg,tmp_path)
    assert len(scene['decisions'])==previous and buffers[-1].getvalue()==pcm
    assert not list(tmp_path.glob('*.wav')) and not list(tmp_path.glob('*.npz'))


def test_old_v6_planning_is_unchanged():
    root=ROOT/'datasets/spatial-v6-24h-20260908'
    if not (root/'recipes/train_0001.json').exists():pytest.skip('Local v6 archive is not distributed')
    cfg=json.loads((root/'config.json').read_text())
    source=json.loads((root/'source_manifest.json').read_text())['sources']
    pools=source_partitions(source,cfg['light_release'])
    p=Planner(cfg,pools['train'],2026090800)
    density=p.random_scene();add_vehicle_admission_draws(p)
    scene=p.result('train_0001',density);scene.update(split='train',seed=2026090800)
    raw=json.dumps(scene,ensure_ascii=False,default=serializable,separators=(',',':'))+'\n'
    # Path.write_text uses CRLF on Windows; compare after universal-newline decoding.
    assert raw==(root/'recipes/train_0001.json').read_text(encoding='utf-8')
