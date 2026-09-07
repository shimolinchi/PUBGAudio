import pytest
import torch
from pubg_audio.evaluate import FrameDiagnostics


def test_duplicate_tracks_count_once_hidden_truth_ignored_and_direction_wrap():
    output = torch.zeros(1,3,3,3,4)
    target = torch.zeros_like(output)
    # Identical three-track predictions are one class presence, including at
    # the 359/1 degree wrap; the last positive is hidden and must be excluded.
    angle = torch.deg2rad(torch.tensor(359.))
    truth_angle = torch.deg2rad(torch.tensor(1.))
    output[0,:,:,0] = torch.tensor([angle.sin(),angle.cos(),0.,1.2])
    target[0,0,0,0] = torch.tensor([truth_angle.sin(),truth_angle.cos(),0.,1.])
    counts = torch.zeros(1,3,3,dtype=torch.long); counts[0,[0,2],0] = 1
    mask = torch.ones(1,3,3); mask[0,2,0] = 0
    stats = FrameDiagnostics()
    stats.update({'accddoa':output},dict(counts=counts,mask=mask,target=target))
    row = stats.report()['external']['per_class']['footsteps']
    assert (row['tp'],row['fp'],row['fn'],row['ignored_frames']) == (1,1,0,1)
    assert row['f1'] == pytest.approx(2/3)
    loc = row['conditional_localization']
    assert loc['detected_single_source_frames'] == 1
    assert loc['angular_mae_degrees'] == pytest.approx(2.,abs=.001)
    assert loc['distance_mae_units'] == pytest.approx(20.,abs=.0001)


def test_absent_class_has_no_invented_recall_or_localization():
    stats = FrameDiagnostics()
    row = stats.report()['external']['per_class']['vehicle']
    assert row['recall'] is None and row['f1'] is None
    assert row['conditional_localization']['angular_mae_degrees'] is None
