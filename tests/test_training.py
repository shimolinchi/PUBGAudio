import numpy as np
import pytest
import torch
from pubg_audio.model import ModelConfig, PaperSELD
from pubg_audio.features import extract,fit_normalizer
from pubg_audio.losses import adpit_loss,training_loss
from pubg_audio.dataset import aligned_targets

torch.set_num_threads(2)


@pytest.mark.parametrize('mode',['multi_accddoa','multi_task'])
def test_paper_topology_and_backprop(mode):
    model=PaperSELD(ModelConfig(output_mode=mode))
    out=model(torch.randn(1,4,250,512))
    if mode=='multi_accddoa':
        assert out['accddoa'].shape==(1,50,3,13,4)
        loss=out['accddoa'].square().mean()
    else:
        assert out['accdoa'].shape==(1,50,13,3)
        assert out['distance'].shape==(1,50,13)
        assert out['distance'].min()>=0
        loss=out['accdoa'].square().mean()+out['distance'].mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('count',[0,1,2,3])
def test_adpit_permutation_duplication(count):
    target=torch.zeros(1,1,3,1,4)
    for i in range(count):
        target[0,0,i,0]=torch.tensor([i+1.,0,0,(i+1)*.2])
    mapping={0:[0,0,0],1:[0,0,0],2:[1,0,1],3:[2,0,1]}[count]
    prediction=target[:,:,mapping].clone().requires_grad_()
    loss=adpit_loss(prediction,target,torch.tensor([[[count]]]),torch.ones(1,1,1))
    assert loss.item()==0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_masked_positive_has_no_gradient():
    prediction=torch.randn(1,2,3,1,4,requires_grad=True)
    loss=adpit_loss(prediction,torch.zeros_like(prediction),torch.zeros(1,2,1,dtype=torch.long),torch.zeros(1,2,1))
    loss.backward()
    assert loss.item()==0 and prediction.grad.abs().sum()==0


def test_features_silence_and_channel_swap():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(24000*5,2)).astype(np.float32)*.01
    a,b=extract(x,24000),extract(x[:,::-1].copy(),24000)
    assert a.shape==(4,250,512)
    torch.testing.assert_close(a[0],b[0])
    torch.testing.assert_close(a[1],-b[1],atol=2e-5,rtol=1e-5)
    torch.testing.assert_close(a[2],b[2])
    torch.testing.assert_close(a[3],-b[3])
    silent=extract(np.zeros_like(x),24000)
    assert torch.count_nonzero(silent)==0
    norm=fit_normalizer([a])
    before=norm.mean.clone()
    assert torch.isfinite(norm(silent)).all()
    assert torch.equal(before,norm.mean)


def test_targets_distinguish_close_external_and_self_and_crop_boundary():
    n=1000; times=np.arange(n)*.01
    z=dict(frame_right_edge_seconds=times+.016+64/44100,class_supervision_mask=np.array([1,1,1,0]),
        activity_loss_mask=np.ones((n,4,8),np.uint8),self_activity=np.zeros((n,3),np.uint8),
        self_activity_loss_mask=np.ones((n,3),np.uint8),track_class_index=np.array([0,0]),
        track_source_role=np.array([0,1]),track_activity=np.zeros((2,n),np.uint8),
        track_xy=np.zeros((2,n,2),np.float32),track_distance_units=np.zeros((2,n),np.float32))
    z['track_activity'][:,20:30]=1; z['track_activity'][0,500]=1
    z['track_xy'][0,:,0]=2.; z['track_distance_units'][0]=2.
    z['self_activity'][20:30,0]=1
    result=aligned_targets(z,0,5)
    assert result['counts'][:,0].sum()==1
    assert result['self_target'][:,0].sum()==1
    torch.testing.assert_close(result['target'][2,0,0],torch.tensor([1.,0.,0.,.02]))
    assert result['counts'][-1].sum()==0
    next_crop=aligned_targets(z,5,5)
    assert next_crop['counts'][0,0]==1
    z['activity_loss_mask'][20:30,0,0]=0
    assert aligned_targets(z,0,5)['mask'][2,0]==0
