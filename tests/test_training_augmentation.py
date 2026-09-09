from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

from pubg_audio.augmentation import mirror_item,full_target_plan,crop_targets,RandomCropScenes,EpochCropSampler
from pubg_audio.dataset import aligned_targets
from pubg_audio.features import extract,FeatureNormalizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/synthesis'))
from causal_render import labels
from causal_scene import KINDS
from motion_synthesis import frame_powers,RATE


def test_reflection_matches_actual_waveform_swap_before_normalization():
    x=np.random.default_rng(10).normal(0,.02,(24000,2)).astype(np.float32)
    x[:,1]*=.2
    raw=dict(features=extract(x,24000),target=torch.tensor([[[[.6,.8,0.,.75]]]]),
             self_target=torch.ones(10,3),self_mask=torch.zeros(10,3),mask=torch.ones(10,3))
    reflected=mirror_item(raw)
    torch.testing.assert_close(reflected['features'],extract(x[:,::-1].copy(),24000),atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(reflected['target'],torch.tensor([[[[-.6,.8,0.,.75]]]]))
    norm=FeatureNormalizer(torch.full((4,512),.3),torch.full((4,512),2.))
    torch.testing.assert_close(norm(reflected['features']),norm(extract(x[:,::-1].copy(),24000)),atol=2e-5,rtol=2e-5)
    for key in ['features','target','self_target','self_mask','mask']:
        torch.testing.assert_close(mirror_item(reflected)[key],raw[key],atol=0,rtol=0)


@pytest.mark.parametrize('role',['external','self'])
def test_random_crop_plan_equals_direct_projection_and_excludes_outside_pulses(role):
    seconds=12.;x=np.zeros((round(seconds*RATE),2),np.float32)
    for start in [5.095,10.105]:
        x[round(start*RATE):round((start+.002)*RATE)]=.1
    tr=dict(id='T001',kind='footsteps',class_index=0,source_role=role,subtype='pulse',states=[],
        path=dict(time=[0.,seconds],xy=[[10.,0.],[10.,0.]],speed=[0.,0.],engine_on=[0.,0.],brake=[0.,0.]))
    cfg={'masking':{'minimum_sir_db':-30,'cabin_foot_sir_db':-12,'cabin_far_vehicle_sir_db':-15,'cabin_far_distance_units':90}}
    detailed=labels(dict(tracks=[tr],classes=KINDS,seconds=seconds),frame_powers([x],seconds),cfg)
    from project_causal_labels import project
    projected=project(detailed)
    plan=full_target_plan(projected,seconds)
    for tick in [0,1,33,51,70]:
        direct=aligned_targets(projected,tick/10.,5.)
        cached=crop_targets(plan,tick)
        for key in direct:torch.testing.assert_close(cached[key],direct[key],rtol=0,atol=0)
    assert not x[round(5.1*RATE):round(10.1*RATE)].any()
    crop=crop_targets(plan,51)
    if role=='external':assert not (crop['counts']*crop['mask']).any()
    else:assert not (crop['self_target']*crop['self_mask']).any()


def test_augmentation_is_training_only_and_epoch_keys_reproduce_crops(tmp_path):
    raw=SimpleNamespace(root=tmp_path,rows=[{'id':'train_0001','split':'train','duration_seconds':120.}],
                        classes=3,distance_scale=100.)
    dataset=RandomCropScenes(raw,tmp_path,77,crops_per_scene=24)
    first=[dataset.selection(0,i) for i in range(len(dataset))]
    assert first==[dataset.selection(0,i) for i in range(len(dataset))]
    assert first!=[dataset.selection(1,i) for i in range(len(dataset))]
    assert all(0<=tick<=1150 for _,tick,_ in first)
    sampler=EpochCropSampler(len(dataset),77);sampler.set_epoch(3)
    keys=list(sampler)
    assert sorted(keys)==[(3,i) for i in range(len(dataset))]
    assert keys==list(sampler)
    raw.rows[0]['split']='validation'
    with pytest.raises(ValueError,match='training-only'):RandomCropScenes(raw,tmp_path,77)
