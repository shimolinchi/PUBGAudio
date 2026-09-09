import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import expit

from pubg_audio.confidence import (
    CLASSES, FORMAT, fit_binary, load_calibration, presence_predictions,
    probability, reliability, sha256,
)


def test_raw_scores_do_not_pretend_to_be_probabilities():
    rows = presence_predictions([1.3, .8, .7], 'external')
    assert rows[0]['presence_score'] == 1.3
    assert [r['class_name'] for r in rows] == list(CLASSES)
    assert all(r['presence_probability'] is None for r in rows)
    assert all(r['probability_status'] == 'not_calibrated' for r in rows)
    own = presence_predictions([.8, .8, .8], 'self')
    assert all(r['presence_score'] == .8 and r['presence_probability'] is None for r in own)
    # A paper checkpoint with 13 classes must not receive PUBG class names.
    assert presence_predictions(np.zeros(13), 'external')[0]['class_name'] == 'class_0'


@pytest.mark.parametrize('kind', ['vector_norm', 'sigmoid'])
def test_monotonic_calibration_on_independent_synthetic_samples(kind):
    rng = np.random.default_rng(71)
    scores = rng.uniform(.01, .99, 12000)
    truth = rng.binomial(1, expit(6*scores-4))
    fitted = fit_binary(scores[:6000], truth[:6000], kind)
    predictions = probability(scores[6000:], fitted)
    raw = reliability(scores[6000:], truth[6000:])
    calibrated = reliability(predictions, truth[6000:])
    assert calibrated['brier'] < raw['brier'] - .015
    assert calibrated['ece'] < raw['ece'] / 2
    curve = probability(np.linspace(0, 1, 101), fitted)
    assert np.isfinite(curve).all() and (np.diff(curve) >= 0).all()
    assert (curve > 0).all() and (curve < 1).all()


def test_sparse_class_does_not_get_fabricated_probability():
    fit = fit_binary(np.ones(100)*.2, np.zeros(100), 'vector_norm')
    assert fit['status'] == 'insufficient_labels'
    assert probability([.2], fit) is None
    assert reliability([], []) == dict(frames=0, brier=None, ece=None, bins=[])


@pytest.mark.parametrize('scores,role', [([np.nan], 'external'), ([-.1], 'external'),
    ([1.2], 'self'), ([[.1]], 'external'), ([.1], 'unknown')])
def test_invalid_scores_rejected(scores, role):
    with pytest.raises(ValueError):
        presence_predictions(scores, role)


def test_checkpoint_binding_schema_and_independent_class_probabilities(tmp_path):
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'immutable model fixture')
    scores = np.tile([.1, .8], 30)
    labels = np.tile([0, 1], 30)
    artifact = dict(format=FORMAT, checkpoint_sha256=sha256(checkpoint),
        scope='synthetic_validation_class_presence', classes=list(CLASSES),
        fits={role: {cls: fit_binary(scores, labels, kind) for cls in CLASSES}
              for role, kind in [('external', 'vector_norm'), ('self', 'sigmoid')]})
    path = tmp_path/'calibration.json'
    path.write_text(json.dumps(artifact), encoding='utf-8')
    loaded = load_calibration(path, checkpoint)
    rows = presence_predictions([.8, .8, .8], 'external', loaded)
    assert all(r['presence_probability'] > .9 for r in rows)
    assert sum(r['presence_probability'] for r in rows) > 2.7
    assert all(r['probability_status'] == 'fitted_on_synthetic_validation' for r in rows)
    for defect in ['hash', 'missing_class', 'class_order', 'score_kind', 'parameters']:
        invalid = copy.deepcopy(artifact)
        if defect == 'hash': invalid['checkpoint_sha256'] = 'wrong'
        elif defect == 'missing_class': del invalid['fits']['external']['footsteps']
        elif defect == 'class_order': invalid['classes'].reverse()
        elif defect == 'score_kind': invalid['fits']['self']['footsteps']['score_kind'] = 'vector_norm'
        else: invalid['fits']['external']['footsteps']['scale'] = 0
        path.write_text(json.dumps(invalid), encoding='utf-8')
        with pytest.raises(ValueError): load_calibration(path, checkpoint)


def calibration_cli():
    path = Path(__file__).resolve().parents[1]/'tools/calibrate_presence.py'
    spec = importlib.util.spec_from_file_location('calibrate_presence', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_scene_split_does_not_leak_neighbouring_frames():
    cli = calibration_cli()
    rows = [dict(id=f'scene_{i}') for i in range(11)]
    fit, audit = cli.split_scenes(rows, 7)
    assert not fit & audit and fit | audit == {r['id'] for r in rows}
    assert (fit, audit) == cli.split_scenes(rows[::-1], 7)
    assert len(fit) == 5 and len(audit) == 6
    with pytest.raises(ValueError): cli.split_scenes(rows[:3], 7)
    with pytest.raises(ValueError): cli.split_scenes(rows+rows[:1], 7)


@pytest.mark.parametrize('with_contract', [False, True])
def test_calibration_cli_respects_masks_and_never_needs_test_data(tmp_path, monkeypatch, with_contract):
    cli = calibration_cli()
    rows = [dict(id=f'validation_{i}', split='validation') for i in range(4)]
    manifest = tmp_path/'manifest_validation.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    plan = None
    if with_contract:
        from confidence_policy import write_plan
        policy_path = Path(__file__).resolve().parents[1]/'configs/confidence_v1.json'
        targets = dict(confidence_policy=json.loads(policy_path.read_text()))
        (tmp_path/'training_targets.json').write_text(json.dumps(targets), encoding='utf-8')
        plan = write_plan(tmp_path, targets)
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(dataset_fingerprint={'validation': sha256(manifest)},
        model_config={'classes': 3, 'self_classes': 3}, config={},
        model={}, normalizer={}, epoch=4), checkpoint)
    before = sha256(checkpoint)

    class DummyData:
        def __init__(self, root, split, **kwargs):
            assert split == 'validation'
            self.rows = rows

        def __len__(self): return 4

        def __getitem__(self, i):
            scores = torch.linspace(.1, .9, 50)[:, None].expand(-1, 3)
            labels = (scores > .5).long()
            mask = torch.ones(50, 3)
            mask[-5:] = 0
            return dict(features=scores, counts=labels, self_target=labels,
                mask=mask, self_mask=mask, scene_id=rows[i]['id'])

    class DummyModel:
        def __init__(self, config): pass
        def eval(self): return self
        def load_state_dict(self, state): pass
        def __call__(self, scores):
            vectors = scores[:, :, None, :, None]*torch.tensor([1., 0., 0., 0.])
            return dict(accddoa=vectors.expand(-1, -1, 3, -1, -1), self_logits=torch.logit(scores))

    monkeypatch.setattr(cli, 'SyntheticScenes', DummyData)
    monkeypatch.setattr(cli, 'PaperSELD', DummyModel)
    monkeypatch.setattr(cli, 'FeatureNormalizer', lambda **kwargs: None)
    output = tmp_path/'calibration.json'
    monkeypatch.setattr(cli.sys, 'argv', ['calibrate_presence.py', '--checkpoint', str(checkpoint),
        '--dataset', str(tmp_path), '--output', str(output)])
    cli.main()
    artifact = load_calibration(output, checkpoint)
    assert sha256(checkpoint) == before
    assert artifact['checkpoint_epoch'] == 5
    assert not set(artifact['fit_scene_ids']) & set(artifact['audit_scene_ids'])
    for role in ['external', 'self']:
        for cls in CLASSES:
            fit = artifact['fits'][role][cls]
            assert (fit['frames'], fit['positives'], fit['negatives']) == (90, 40, 50)
            assert artifact['audit'][role][cls]['calibrated']['frames'] == 90
    with pytest.raises(ValueError, match='existing calibration'): cli.main()
    if with_contract:
        assert artifact['fit_scene_ids'] == plan['fit_scene_ids']
        assert artifact['audit_scene_ids'] == plan['audit_scene_ids']
        assert artifact['confidence_policy_sha256'] == plan['policy_sha256']
        monkeypatch.setattr(cli.sys, 'argv', ['calibrate_presence.py', '--checkpoint', str(checkpoint),
            '--dataset', str(tmp_path), '--output', str(tmp_path/'another.json'), '--seed', '1'])
        with pytest.raises(ValueError, match='Do not override'): cli.main()
