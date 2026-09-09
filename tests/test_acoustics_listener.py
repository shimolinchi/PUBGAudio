import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/synthesis'))
from footstep_acoustics import gain_at_distance
from event_acoustics import EventSpatialRenderer
from listener_view import to_listener,relative_scene,add_world_labels
from causal_render import render_track,labels
from motion_synthesis import SpatialRenderer,RATE
from project_causal_labels import project
import causal_scene as cs

CFG=json.loads((ROOT/'configs/generation_v5_1.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('gait',['walk','run','sprint','crouch_walk','prone_crawl'])
def test_footstep_finite_monotone_and_smooth(gait):
    ac=CFG['footstep_acoustics'];r=ac['profiles'][gait]['cutoff_m_estimate']
    g=gain_at_distance(np.linspace(0,r*2,10001),ac,gait)
    assert np.isfinite(g).all() and np.all(np.diff(g)<=1e-12)
    assert gain_at_distance(5,ac,gait)==1
    assert np.all(gain_at_distance([r,r+1,400,10000],ac,gait)==0)
    assert gain_at_distance(r-.01,ac,gait)<1e-5


def test_yaw_sign_wrap_and_distance_invariance():
    np.testing.assert_allclose(to_listener([0,10],90),[-10,0],atol=1e-12)
    np.testing.assert_allclose(to_listener([10,0],90),[0,10],atol=1e-12)
    yaw=np.linspace(350,370,101);x=to_listener(np.tile([7.,11.],(101,1)),yaw)
    np.testing.assert_allclose(np.linalg.norm(x,axis=1),np.sqrt(170))
    assert np.linalg.norm(np.diff(x,axis=0),axis=1).max()<.05


@pytest.fixture
def spatial():
    h=ROOT/'datasets/hrir/kemar-diffuse.zip'
    if not h.exists():pytest.skip('Local HRIR not distributed')
    return SpatialRenderer(h)


def test_all_known_event_profiles_cutoff_and_unknown_tail(spatial):
    ac=CFG['event_acoustics']
    for role,p in ac['profiles'].items():
        d=np.linspace(1,2*p['cutoff_m_estimate'],1000)
        gain,_=EventSpatialRenderer(spatial,ac,role).distance_response(d)
        assert np.all(np.diff(gain)<=1e-12) and gain[-1]==0
    g,_=EventSpatialRenderer(spatial,ac,'shot').distance_response(np.array([400.,800.,1600.]))
    assert 0<g[2]<g[1]<g[0]


def track(role='external',distance=10):
    return dict(id='T1',kind='footsteps',class_index=0,subtype='walk',source_role=role,gain_db=0,
        events=[dict(role='walk',source_id='source',start=0,duration=1)],
        states=[dict(start=0,end=1,state='walk')],
        path=dict(time=[0,1],xy=[[0,distance],[0,distance]],speed=[0,0],engine_on=[0,0],brake=[0,0]))


class Library:
    def mono(self,sid):return np.random.default_rng(1).normal(0,.03,RATE).astype(np.float32)
    def stereo(self,sid):return np.column_stack((self.mono(sid),self.mono(sid)*.8))


def test_pcm_silent_outside_radius_and_self_unchanged(spatial):
    lib=Library();outside=track(distance=80)
    assert np.max(np.abs(render_track(outside,lib,spatial,1,CFG)))==0
    own=track('self',0)
    np.testing.assert_array_equal(render_track(own,lib,spatial,1,CFG),render_track(own,lib,spatial,1))
    assert np.max(np.abs(render_track(track(),lib,spatial,1,CFG)))>0


def test_labels_keep_world_truth_but_project_listener_coordinates():
    tr=track(distance=80);scene=dict(tracks=[tr],classes=cs.KINDS,seconds=1,
        listener_view=dict(time=[0,1],yaw_degrees_unwrapped=[90,90]))
    power=np.zeros((1,97));z=labels(scene,power,CFG);add_world_labels(z,scene)
    assert z['track_footstep_emission_scheduled'].all() and z['track_outside_footstep_radius'].all()
    assert z['track_activity'].sum()==0 and project(z)['track_activity'].sum()==0
    np.testing.assert_allclose(z['track_world_xy'][0,:,1],80)
    np.testing.assert_allclose(z['track_azimuth_degrees'],270,atol=1e-5)
    np.testing.assert_array_equal(project(z)['track_xy'],z['track_xy'])


def test_new_planner_view_speed_and_aircraft_world_line():
    sources=ROOT/'datasets/sources-v5/sources.json'
    if not sources.exists():pytest.skip('Local game sources not distributed')
    cfg=copy.deepcopy(CFG);cfg['listener_view']['mode_weights']={'natural':1};cfg['clip_seconds']=100
    p=cs.Planner(cfg,json.loads(sources.read_text(encoding='utf-8'))['sources'],781)
    tr=p.aircraft();s=p.result('A','coverage');v=s['listener_view']
    rate=np.abs(np.diff(v['yaw_degrees_unwrapped'])/.01)
    assert rate.max()<=140.001 and rate.max()>0 and np.sum(rate==0)>100
    d=np.diff(tr['path']['xy'],axis=0)
    np.testing.assert_allclose(d,np.broadcast_to(d[0],d.shape),atol=1e-10)
    assert np.linalg.norm(tr['path']['xy'][[0,-1]],axis=1).min()>3000
