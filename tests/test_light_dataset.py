import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/synthesis'))
from generate_light_dataset import source_partitions
from event_acoustics import render_components


def test_duplicate_pcm_cannot_cross_source_partitions():
    spec = dict(included_gaits=['run'], source_groups_by_split={
        'train':['footsteps:a'], 'validation':['footsteps:b'], 'test':['footsteps:c']})
    sources = [dict(id=g,source_group_id='footsteps:'+g,role='run',pcm_sha256='duplicate')
               for g in ['a','b','c']]
    with pytest.raises(ValueError,match='PCM crosses'):
        source_partitions(sources,spec)


def test_unavailable_tyre_layer_is_never_requested_from_another_partition():
    calls = []
    def fake_render(track, library, spatial, seconds):
        calls.extend(track['_vehicle_layers'])
        return np.zeros((round(seconds*44100),2),np.float32)
    # No events: this isolates continuous-layer dispatch without requiring assets.
    result = render_components(dict(kind='vehicle',events=[],available_vehicle_layers=['engine','skid']),
        None, type('Spatial',(),{})(), .01, {'event_acoustics':{}}, fake_render)
    assert calls==['engine','skid']
    assert result.shape==(441,2)
