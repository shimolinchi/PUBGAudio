import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/synthesis'))
import causal_scene as cs
from listener_view import bind_self_throw_direction,to_listener,make_view


@pytest.mark.parametrize('kind',['frag','flash','molotov'])
def test_self_throw_uses_release_direction_and_does_not_follow_later_yaw(kind):
    source=ROOT/'datasets/sources-v5/sources.json'
    if not source.exists():pytest.skip('Local game sources not distributed')
    cfg=json.loads((ROOT/'configs/generation_v5_2.json').read_text(encoding='utf-8'))
    p=cs.Planner(cfg,json.loads(source.read_text(encoding='utf-8'))['sources'],876)
    p.throwable(3,'self',kind);s=p.result('T','coverage');chain=s['chains'][0]
    r=chain['release'];s['listener_view']=dict(time=[0,r,r+.8,60],yaw_degrees_unwrapped=[40,40,160,160])
    bind_self_throw_direction(s);fx=next(t for t in s['tracks'] if t['id']==chain['effect'])
    relative=to_listener(np.array(chain['detonation_xy'])-chain['release_xy'],40)
    assert abs(relative[0])<1e-10 and relative[1]>0 and fx['source_role']=='external'
    after=np.asarray(fx['path']['xy'])[np.asarray(fx['path']['time'])>=chain['detonate']]
    np.testing.assert_allclose(after,np.broadcast_to(after[0],after.shape),atol=1e-9)
    first=np.asarray(fx['path']['xy']).copy();bind_self_throw_direction(s)
    np.testing.assert_allclose(fx['path']['xy'],first,atol=1e-9)
    changed=to_listener(after[0],160)
    assert np.degrees(np.arctan2(changed[0],changed[1]))==pytest.approx(-120)


def test_forward_binding_does_not_change_an_external_throw():
    source=ROOT/'datasets/sources-v5/sources.json'
    if not source.exists():pytest.skip('Local game sources not distributed')
    cfg=json.loads((ROOT/'configs/generation_v5_2.json').read_text(encoding='utf-8'))
    p=cs.Planner(cfg,json.loads(source.read_text(encoding='utf-8'))['sources'],44)
    p.throwable(2,'external','frag');s=p.result('T','coverage');before=copy.deepcopy(s)
    bind_self_throw_direction(s)
    np.testing.assert_array_equal(s['tracks'][1]['path']['xy'],before['tracks'][1]['path']['xy'])


def test_v52_view_activity_increases_without_exceeding_speed_bound():
    configs=[json.loads((ROOT/f'configs/generation_{v}.json').read_text(encoding='utf-8')) for v in ['v5_1','v5_2']]
    counts=[]
    for cfg in configs:
        total=0
        for seed in range(30):
            p=cs.Planner(cfg,[],seed);v=make_view(p,'natural');total+=len(v['turns'])
            assert np.max(np.abs(np.diff(v['yaw_degrees_unwrapped'])/.01))<=140.01
        counts.append(total)
    assert counts[1]>counts[0]*1.5
