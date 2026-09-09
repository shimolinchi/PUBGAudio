import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/synthesis'))
import confidence_policy as policy_module
import generate_light_dataset as generator
from configure_footsteps_v6_1 import build
from pubg_audio.confidence import fit_binary, probability


def write_json(path, value):
    path.write_text(json.dumps(value)+'\n', encoding='utf-8')


def policy():
    return json.loads((ROOT/'configs/confidence_v1.json').read_text(encoding='utf-8'))


def make_dataset(root):
    root.mkdir()
    targets = dict(external_targets=policy_module.CLASSES, self_targets=policy_module.CLASSES,
                   confidence_policy=policy())
    write_json(root/'training_targets.json', targets)
    rows = [dict(id=f'validation_{i}', split='validation', duration_seconds=240,
                 audio_file=f'audio/validation_{i}.wav') for i in range(8)]
    manifest = root/'manifest_validation.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    plan = policy_module.write_plan(root, targets)
    return targets, rows, plan


def test_generated_policy_preserves_hard_truth_and_freezes_only_validation(tmp_path):
    targets, rows, plan = make_dataset(tmp_path/'dataset')
    contract = policy_module.read_contract(tmp_path/'dataset')
    assert contract['plan'] == plan
    assert not set(plan['fit_scene_ids']) & set(plan['audit_scene_ids'])
    assert set(plan['fit_scene_ids']) | set(plan['audit_scene_ids']) == {r['id'] for r in rows}
    assert plan == policy_module.make_plan(rows[::-1], policy(), plan['validation_manifest_sha256'])
    assert targets['confidence_policy']['ground_truth']['generated_confidence_probability'] is None
    assert not plan['model_selection_independent'] and not plan['real_game_calibration']
    assert policy_module.write_plan(tmp_path/'dataset', targets) == plan


@pytest.mark.parametrize('defect', ['manifest', 'overlap', 'test_scene', 'soft_label', 'generation_identity'])
def test_changed_or_leaky_confidence_metadata_fails(tmp_path, defect):
    root = tmp_path/'dataset'
    targets, rows, plan = make_dataset(root)
    if defect == 'manifest':
        with (root/'manifest_validation.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(rows[0], id='extra'))+'\n')
    elif defect == 'overlap':
        plan['audit_scene_ids'][0] = plan['fit_scene_ids'][0]
        write_json(root/'calibration_plan.json', plan)
    elif defect == 'test_scene':
        rows[0]['split'] = 'test'
        with pytest.raises(ValueError, match='Only validation'):
            policy_module.make_plan(rows, policy(), 'dummy')
        return
    elif defect == 'soft_label':
        targets['confidence_policy']['ground_truth']['generated_confidence_probability'] = .8
        write_json(root/'training_targets.json', targets)
    else:
        write_json(root/'generation_identity.json', dict(targets={}))
    with pytest.raises(ValueError): policy_module.read_contract(root)


def test_generator_plan_only_embeds_policy_without_rendering(tmp_path, monkeypatch):
    _, spec = build()
    spec['base_config'] = str(ROOT/spec['base_config'])
    spec['confidence_policy_file'] = str(ROOT/spec['confidence_policy_file'])
    spec['source_groups'] = ['footsteps:Concrete']
    spec['split_counts'] = dict(train=4, validation=4, test=4)
    spec_path = tmp_path/'spec.json'
    write_json(spec_path, spec)
    source_root = tmp_path/'sources'
    source_root.mkdir()
    (source_root/'source.wav').write_bytes(b'not rendered; hash-only fixture')
    write_json(source_root/'sources.json', dict(sources=[dict(id='fixture',
        source_group_id='footsteps:Concrete', role='run', wav_file='source.wav',
        wav_sha256=policy_module.digest(source_root/'source.wav'))]))
    hrir = tmp_path/'hrir.zip'
    hrir.write_bytes(b'hash-only fixture')
    destination = tmp_path/'planned'
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, 'argv', ['generate_light_dataset.py', '--spec', str(spec_path),
        '--sources', str(source_root), '--hrir', str(hrir), '--output', str(destination), '--plan-only'])
    generator.main()
    saved = json.loads((destination/'training_targets.json').read_text())
    assert saved['confidence_policy'] == policy()
    assert json.loads((destination/'generation_identity.json').read_text())['targets'] == saved
    assert not list((destination/'audio').iterdir())
    assert not (destination/'calibration_plan.json').exists()  # Requires the completed manifest.
    assert (destination/'generator_snapshot/confidence_policy.py').exists()


def test_small_validation_budget_and_minimums_are_explicit():
    with pytest.raises(ValueError): policy_module.split_scenes([dict(id=str(i)) for i in range(3)], 1)
    with pytest.raises(ValueError): policy_module.split_scenes([dict(id=str(i)) for i in range(4)], 1, .1)
    fitted = fit_binary([.1, .2, .8, .9], [0, 0, 1, 1], 'vector_norm', 2, 2)
    assert probability([.9], fitted)[0] > .9
    assert fit_binary([.1, .2, .8, .9], [0, 0, 1, 1], 'vector_norm', 3, 2)['status']=='insufficient_labels'
    with pytest.raises(ValueError): fit_binary([], [], 'vector_norm', 0, 0)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_future_training_pipeline_calibrates_frozen_export_and_handles_long_scene(tmp_path, monkeypatch):
    script = load_script('run_spatial_training')
    dataset = tmp_path/'dataset'
    _, _, plan = make_dataset(dataset)
    for split in ['train', 'test']:
        (dataset/f'manifest_{split}.jsonl').write_text('', encoding='utf-8')
    contract = policy_module.read_contract(dataset)
    fingerprints = {s:script.digest(dataset/f'manifest_{s}.jsonl') for s in ['train','validation','test']}
    write_json(dataset/'verification.json', dict(status='passed', manifests_sha256=fingerprints,
        confidence_policy_sha256=contract['policy_sha256'], calibration_plan_sha256=contract['plan_sha256']))
    write_json(dataset/'coverage_verification.json', dict(status='passed',
        every_evaluation_gun_type_and_role_seen_in_training=True, all_available_gun_type_roles_audible_in_training=True,
        all_selected_external_target_types_audible_in_training=True, training_external_types_cover_four_quadrants=True))
    config = tmp_path/'config.json'
    write_json(config, dict(epochs=1, save_localization_checkpoint=True))
    output = tmp_path/'run'
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if 'pubg_audio.train' in command:
            write_json(output/'run.json', dict(status='training_completed', completed_epochs=1))
            write_json(output/'metrics.jsonl', dict(epoch=0, validation_localization_macro_f1=.5,
                validation_external_macro_f1=.6, validation_loss=.2))
            for checkpoint, report in [('best_localization.pt','test_localization.json'),
                ('best_detection.pt','test_detection.json'),('best.pt','test.json')]:
                (output/checkpoint).write_bytes(b'frozen model fixture')
                write_json(output/report, dict(checkpoint_epoch=0, checkpoint_sha256=script.digest(output/checkpoint),
                    dataset_fingerprint=fingerprints, split='test'))
        elif any(c.endswith('export_inference.py') for c in command):
            Path(command[command.index('--output')+1]).write_bytes(b'inference fixture')
        elif any(c.endswith('calibrate_presence.py') for c in command):
            assert 'source_snapshot' in command[1]
            assert Path(command[command.index('--checkpoint')+1]).name == 'model_localization.pt'
            write_json(output/'presence_calibration.json', dict(mock=True))
        elif 'pubg_audio.predict' in command:
            assert command[command.index('--calibration')+1] == str(output/'presence_calibration.json')
            row = dict(presence_probability=.8, probability_status='fitted_on_synthetic_validation')
            frame = dict(self_probabilities=[.9]*3, raw_external_tracks=[dict(raw_xyzd=[0,1,0,.2])],
                external_class_predictions=[row]*3, self_class_predictions=[row]*3)
            write_json(output/'cpu_inference_validation_0000.json', dict(frames=[frame]*2400))

    monkeypatch.setattr(script.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['run_spatial_training.py', '--dataset', str(dataset),
        '--config', str(config), '--output', str(output), '--cache-dir', str(tmp_path/'cache'),
        '--wandb-config', str(tmp_path/'unused-wandb.json')])
    script.main()
    result = json.loads((output/'export_verification.json').read_text())
    assert result['cpu_inference_frames']==2400
    assert result['presence_calibration_file']=='presence_calibration.json'
    assert (output/'source_snapshot/tools/synthesis/confidence_policy.py').exists()
    identity = json.loads((output/'training_identity.json').read_text())
    assert identity['confidence_contract']['plan'] == plan
    assert len(calls)==7
