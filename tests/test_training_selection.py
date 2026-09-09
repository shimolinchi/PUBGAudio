from copy import deepcopy
from types import SimpleNamespace
import sys
import pytest

from pubg_audio.evaluate import FrameDiagnostics
from pubg_audio.train import external_macro_f1
from pubg_audio.tracking import log_evaluation


def test_detection_macro_does_not_get_dominated_by_vehicle_frame_count():
    report=FrameDiagnostics().report()
    rows=report['external']['per_class']
    for cls,count,score in [('footsteps',10,.2),('vehicle',10000,.9),('gunfire',20,.4)]:
        rows[cls].update(positive_frames=count,f1=score)
    assert external_macro_f1(report)==pytest.approx(.5)
    assert external_macro_f1(FrameDiagnostics().report()) is None


def test_alternative_checkpoint_metrics_cannot_overwrite_primary_stream(monkeypatch):
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(
        Table=lambda **kwargs:kwargs,plot=SimpleNamespace(bar=lambda *args,**kwargs:kwargs)))
    logs=[]
    run=SimpleNamespace(log=logs.append,summary={})
    report=dict(loss={'total':.1},checkpoint_epoch=14,diagnostics=FrameDiagnostics().report())
    log_evaluation(run,report)
    other=deepcopy(report);other['checkpoint_epoch']=25
    log_evaluation(run,other,namespace='test_detection')
    assert run.summary['test/checkpoint_epoch']==15
    assert run.summary['test_detection/checkpoint_epoch']==26
    assert all(k.startswith('test_detection/') for k in logs[1])
