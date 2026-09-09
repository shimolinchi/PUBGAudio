import json
from pathlib import Path
import torch
import pytest

from pubg_audio.cache import prepare_cache
from pubg_audio.features import FeatureNormalizer


class ExampleDataset:
    normalizer=None
    root=Path('.')
    rows=[{'id':'train-example','split':'train'}]
    items=[(0,0.)]
    seconds=5.
    classes=3
    distance_scale=100.

    def __len__(self):return 1

    def __getitem__(self,index):
        return dict(scene_id='train-example',start_seconds=0.,
                    features=torch.arange(4*250*512).reshape(4,250,512).float()/1e6,
                    target=torch.randn(50,3,3,4),counts=torch.zeros(50,3,dtype=torch.int64),
                    mask=torch.zeros(50,3),self_target=torch.ones(50,3),self_mask=torch.zeros(50,3))


def test_cache_preserves_masks_and_normalizes_only_on_read(tmp_path):
    dataset=ExampleDataset()
    cached=prepare_cache(dataset,tmp_path/'cache')
    raw=cached[0]
    assert not raw['mask'].any() and not raw['self_mask'].any()
    assert raw['self_target'].all()
    cached.normalizer=FeatureNormalizer(torch.zeros(4,512),torch.full((4,512),2.))
    torch.testing.assert_close(cached[0]['features'],raw['features']/2)
    unchanged=prepare_cache(dataset,tmp_path/'cache')[0]
    for key in ['features','target','mask','self_target','self_mask']:
        torch.testing.assert_close(unchanged[key],raw[key],rtol=0,atol=0)


def test_cache_rejects_different_training_distance_scale(tmp_path):
    dataset=ExampleDataset()
    prepare_cache(dataset,tmp_path/'cache')
    dataset.distance_scale=50.
    with pytest.raises(ValueError,match='Stale feature cache'):
        prepare_cache(dataset,tmp_path/'cache')
