import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from pubg_audio.losses import training_loss
from pubg_audio.evaluate import FrameDiagnostics
from pubg_audio.dataset import aligned_targets
from pubg_audio.augmentation import full_target_plan, crop_targets, mirror_item

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/synthesis'))
from air_absorption import air_kernel
from causal_scene import Planner
from generate_light_dataset import configuration, source_partitions
from footstep_acoustics import gain_at_distance


def release():
    spec=json.loads(Path('configs/dataset_v6_24h.json').read_text())
    cfg=configuration(spec)
    sources=json.loads(Path('datasets/sources-v5_3-final/sources.json').read_text())['sources']
    return cfg,source_partitions(sources,spec)


def batch_for(count):
    target=torch.zeros(1,1,3,3,4)
    vectors=[[0.,1.,0.,.1],[1.,0.,0.,.3],[0.,-1.,0.,.5]]
    for i in range(count):target[0,0,i,0]=torch.tensor(vectors[i])
    counts=torch.zeros(1,1,3,dtype=torch.long);counts[0,0,0]=count
    loc=(target[...,:3].square().sum(-1)>0).float()
    return dict(target=target,counts=counts,localization_mask=loc,mask=torch.ones(1,1,3))


@pytest.mark.parametrize('count',[0,1,2,3])
def test_spatial_adpit_matches_permuted_and_duplicated_sources(count):
    batch=batch_for(count)
    order={0:[0,0,0],1:[0,0,0],2:[1,0,1],3:[2,0,1]}[count]
    pred=batch['target'][:,:,order].clone().requires_grad_()
    loss,_=training_loss(dict(accddoa=pred),batch,loss_config=dict(mode='spatial_balanced'))
    assert loss.item()==pytest.approx(0,abs=1e-7)
    loss.backward();assert torch.isfinite(pred.grad).all()


def test_unknown_direction_retains_presence_without_wrong_spatial_gradient():
    batch=batch_for(1);batch['localization_mask'].zero_()
    pred=batch['target'][:,:,[0,0,0]].clone()
    pred[...,1]*=-1;pred[...,3]+=4;pred.requires_grad_()
    loss,_=training_loss(dict(accddoa=pred),batch,loss_config=dict(mode='spatial_balanced'))
    loss.backward();assert loss.item()==0 and pred.grad.abs().sum()==0
    wrong=pred.detach().clone();wrong[...,:3]*=.3
    loss,_=training_loss(dict(accddoa=wrong),batch,loss_config=dict(mode='spatial_balanced'))
    assert loss.item()>0


def test_front_back_mistake_penalized_despite_correct_class():
    batch=batch_for(1);pred=batch['target'][:,:,[0,0,0]].clone();pred[...,1]*=-1
    stats=FrameDiagnostics(localization_aware=True)
    stats.update(dict(accddoa=pred),batch)
    row=stats.report()['external']['per_class']['footsteps']
    assert row['f1']==1 and row['localization_aware']['f1']==0
    assert row['localization_aware']['fp']==1 and row['localization_aware']['fn']==1


def test_spatial_metric_deduplicates_and_matches_two_sources_and_ignores_ambiguous():
    batch=batch_for(2);pred=batch['target'][:,:,[1,0,1]].clone()
    stats=FrameDiagnostics(localization_aware=True);stats.update(dict(accddoa=pred),batch)
    row=stats.report()['external']['per_class']['footsteps']['localization_aware']
    assert row['tp']==2 and row['fp']==0 and row['fn']==0
    batch['localization_mask'][0,0,1,0]=0
    stats=FrameDiagnostics(localization_aware=True);stats.update(dict(accddoa=pred),batch)
    row=stats.report()['external']['per_class']['footsteps']['localization_aware']
    assert row['ignored_ambiguous_frames']==1 and row['target_instances']==0


def test_air_filter_preserves_common_delay_and_attenuates_smoothly():
    cfg,_=release();s=cfg['air_absorption']
    near,far=air_kernel(1,s),air_kernel(150,s)
    assert np.argmax(abs(near))==64
    np.testing.assert_allclose(far,far[::-1],atol=1e-7)
    response=np.abs(np.fft.rfft(far,4096))
    f=np.fft.rfftfreq(4096,1/44100)
    assert .97 < response[0] < 1.03
    assert .1 < response[np.argmin(abs(f-8000))] < .8
    assert response[np.argmin(abs(f-8000))] < response[np.argmin(abs(f-1000))]


def test_finite_quieter_footsteps_and_all_known_source_types_in_splits():
    cfg,pools=release()
    for gait,radius in [('walk',35),('run',45),('sprint',50)]:
        values=gain_at_distance(np.array([5,20,radius,100]),cfg['footstep_acoustics'],gait)
        assert values[0]>values[1]>values[2]==values[3]==0
    assert cfg['audio']['source_rms_by_class']['0'] < cfg['audio']['source_rms_by_role']['shot']/4
    for role in ['shot','local_shot']:
        test={s['source_group_id'] for s in pools['test'] if s['role']==role}
        train={s['source_group_id'] for s in pools['train'] if s['role']==role}
        assert test and test<=train


def test_full_auto_has_physical_cadence_and_magazine_bound():
    cfg,pools=release();profile=cfg['gunfire']['profiles']['Weapons_AK47']
    profile['pattern_weights']={'sustained_auto':1}
    sources=[s for s in pools['train'] if s['source_group_id']=='gunfire:Weapons_AK47']
    planner=Planner(cfg,sources,4);track=planner.gun(1)
    assert track['firing_pattern']=='sustained_auto'
    assert 8<=len(track['events'])<=profile['magazine_limit']
    gaps=np.diff([e['start'] for e in track['events']])
    assert np.count_nonzero(np.isclose(gaps,profile['interval_seconds'],atol=1e-6))>=7
    assert gaps.min()>=profile['interval_seconds']-1e-6


def test_every_weapon_profile_is_usable_and_manual_rifles_never_auto():
    cfg,pools=release()
    for weapon,profile in cfg['gunfire']['profiles'].items():
        sources=[s for s in pools['train'] if s['source_group_id']=='gunfire:'+weapon]
        role='external' if any(s['role']=='shot' for s in sources) else 'self'
        planner=Planner(cfg,sources,3);track=planner.gun(1,role)
        assert 0<len(track['events'])<=profile['magazine_limit']
        if profile['weapon_kind'] in ['dmr','bolt']:
            assert set(profile['pattern_weights'])=={'single'}
            assert track['native_fire_mode']=='single'
        if weapon in ['Weapons_M16A4','Weapons_Mk47']:
            assert not any('auto' in k for k in profile['pattern_weights'])
