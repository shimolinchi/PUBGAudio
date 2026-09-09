"""Exercise real overlapping power windows, not invented centre-only labels."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/synthesis'))
from causal_render import labels
from causal_scene import KINDS
from motion_synthesis import frame_powers, RATE
from project_causal_labels import project
from pubg_audio.dataset import aligned_targets


@pytest.mark.parametrize('role',['external','self'])
@pytest.mark.parametrize('crop_start,pulse_start',[(0.,5.005),(5.,4.995)])
def test_sound_outside_crop_cannot_supervise_its_boundary(role,crop_start,pulse_start):
    seconds=10.;x=np.zeros((round(seconds*RATE),2),np.float32)
    x[round(pulse_start*RATE):round((pulse_start+.002)*RATE)]=.1
    crop=x[round(crop_start*RATE):round((crop_start+5)*RATE)]
    assert not crop.any()
    tr=dict(id='T001',kind='footsteps',class_index=0,source_role=role,subtype='pulse',states=[],
        path=dict(time=[0.,seconds],xy=[[10.,0.],[10.,0.]],speed=[0.,0.],engine_on=[0.,0.],brake=[0.,0.]))
    cfg={'masking':{'minimum_sir_db':-30,'cabin_foot_sir_db':-12,'cabin_far_vehicle_sir_db':-15,'cabin_far_distance_units':90}}
    detailed=labels(dict(tracks=[tr],classes=KINDS,seconds=seconds),frame_powers([x],seconds),cfg)
    batch=aligned_targets(project(detailed),crop_start,5.)
    boundary=-1 if crop_start==0 else 0
    if role=='external':
        assert batch['counts'][boundary,0]==1  # Overlap is present in the raw labels.
        assert not (batch['counts']*batch['mask']).any()
    else:
        assert batch['self_target'][boundary,0]==1
        assert not (batch['self_target']*batch['self_mask']).any()
    assert batch['mask'][boundary].sum()==0 and batch['self_mask'][boundary].sum()==0
    assert batch['mask'][10:40].all() and batch['self_mask'][10:40].all()
