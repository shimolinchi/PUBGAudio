import json
import sys
from types import SimpleNamespace
import pytest
from pubg_audio.tracking import load_binding,start_run


def test_tracking_requires_module_project_binding(tmp_path):
    target=tmp_path/'binding.json'
    target.write_text(json.dumps(dict(entity='user',project='other-project')))
    with pytest.raises(ValueError,match='independent project'):
        load_binding(target)
    target.write_text(json.dumps(dict(application='PUBGAudio',created_for_this_module=True,
                                    entity='user',project='PUBGAudio',project_id='new-id')))
    assert load_binding(target)['project']=='PUBGAudio'


def test_global_project_is_overridden_and_existing_run_cannot_move(tmp_path,monkeypatch):
    target=tmp_path/'binding.json'
    target.write_text(json.dumps(dict(application='PUBGAudio',created_for_this_module=True,
                                    entity='audio-team',project='PUBGAudio',project_id='new-id')))
    calls=[]
    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(define_metric=lambda *a,**k:None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Settings=lambda **kwargs:kwargs))
    monkeypatch.setenv('WANDB_PROJECT','unrelated-project')
    monkeypatch.setenv('WANDB_RUN_ID','unrelated-run')
    start_run(target,tmp_path/'run','audio-test',{},mode='offline')
    assert calls[0]['entity']=='audio-team' and calls[0]['project']=='PUBGAudio'
    assert calls[0]['id']!='unrelated-run'
    receipt=tmp_path/'run'/'wandb_run.json'
    data=json.loads(receipt.read_text()); data['project']='unrelated-project'
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='another W&B project'):
        start_run(target,tmp_path/'run','audio-test',{},mode='offline')
    assert len(calls)==1
