"""Run a verified v6 release using frozen source, then export all validation selections."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent/'synthesis'))
from confidence_policy import read_contract


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--wandb-config', type=Path, required=True)
    args = parser.parse_args()
    module = Path(__file__).resolve().parents[1]
    dataset, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'last.pt').exists() or (output/'source_snapshot').exists():
        raise ValueError('Use a fresh run directory; refusing to replace a training run')
    verification, coverage = read(dataset/'verification.json'), read(dataset/'coverage_verification.json')
    if verification['status'] != 'passed' or coverage['status'] != 'passed':
        raise ValueError('Both dataset verification gates must pass first')
    if verification['manifests_sha256'] != {s:digest(dataset/f'manifest_{s}.jsonl') for s in ['train','validation','test']}:
        raise ValueError('Dataset manifests changed after verification')
    confidence_contract = read_contract(dataset)
    if confidence_contract is not None:
        for key, expected in [('confidence_policy_sha256', confidence_contract['policy_sha256']),
                              ('calibration_plan_sha256', confidence_contract['plan_sha256'])]:
            if verification.get(key) != expected:
                raise ValueError('Dataset confidence rules were not verified: '+key)
    for key in ['every_evaluation_gun_type_and_role_seen_in_training',
                'all_available_gun_type_roles_audible_in_training',
                'all_selected_external_target_types_audible_in_training',
                'training_external_types_cover_four_quadrants']:
        if not coverage[key]:
            raise ValueError(f'Full release coverage gate missing: {key}')
    config = read(args.config)
    if not config.get('save_localization_checkpoint'):
        raise ValueError('The release must retain the localization validation selection')
    frozen = output/'source_snapshot'
    for source in (module/'src/pubg_audio').glob('*.py'):
        target = frozen/'src/pubg_audio'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ['export_inference.py', 'run_spatial_training.py', 'finalize_training_run.py',
                 'calibrate_presence.py', 'synthesis/confidence_policy.py']:
        target = frozen/'tools'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module/'tools'/name, target)
    shutil.copy2(args.config, output/'training_config.json')
    identity = dict(source_sha256={str(p.relative_to(frozen)): digest(p) for p in frozen.rglob('*.py')},
        dataset=str(dataset), dataset_fingerprint={s: digest(dataset/f'manifest_{s}.jsonl') for s in ['train','validation','test']},
        verification_sha256=digest(dataset/'verification.json'),
        coverage_sha256=digest(dataset/'coverage_verification.json'),
        config_sha256=digest(output/'training_config.json'), python=sys.executable,
        command_budget=dict(epochs=config['epochs'], batch_size=16, workers=4, threads=4))
    if confidence_contract is not None:
        identity['confidence_contract'] = confidence_contract
    write(output/'training_identity.json', identity)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(frozen/'src')
    env['PYTHONUNBUFFERED'] = '1'
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '4'
    env['WANDB_DISABLE_CODE'] = 'true'
    runtime_temp = output/'runtime-temp'
    runtime_temp.mkdir(exist_ok=True)
    env['TMP'] = env['TEMP'] = str(runtime_temp)
    status = dict(status='preparing', dataset=str(dataset), output=str(output), stages_completed=[])

    def stage(name, command):
        status.update(status=name, updated_at=datetime.now(timezone.utc).isoformat())
        write(output/'pipeline_status.json', status)
        print(json.dumps(dict(stage=name, status='started')), flush=True)
        subprocess.run([str(x) for x in command], cwd=module, env=env, check=True)
        status['stages_completed'].append(name)

    try:
        stage('frozen_source_check', [sys.executable, '-c',
            "import pathlib,sys,pubg_audio.train; actual=pathlib.Path(pubg_audio.train.__file__).resolve(); "
            "assert actual==pathlib.Path(sys.argv[1]).resolve(), actual; print('Frozen training source:',actual)",
            frozen/'src/pubg_audio/train.py'])
        stage('training_and_test', [sys.executable, '-u', '-m', 'pubg_audio.train',
            '--dataset', dataset, '--config', output/'training_config.json', '--output', output,
            '--cache-dir', args.cache_dir.resolve(), '--batch-size', '16', '--workers', '4',
            '--device', 'cuda', '--threads', '4', '--evaluate-test', '--wandb-config', args.wandb_config.resolve()])
        info = read(output/'run.json')
        assert info['status'] == 'training_completed' and info['completed_epochs'] == config['epochs']
        records = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        assert len(records) == config['epochs'] and [r['epoch'] for r in records] == list(range(config['epochs']))
        selections = [
            ('localization', 'best_localization.pt', 'model_localization.pt', 'test_localization.json',
             max(records, key=lambda r:r['validation_localization_macro_f1'])['epoch']),
            ('detection', 'best_detection.pt', 'model_detection.pt', 'test_detection.json',
             max(records, key=lambda r:r['validation_external_macro_f1'])['epoch']),
            ('loss', 'best.pt', 'model.pt', 'test.json', min(records, key=lambda r:r['validation_loss'])['epoch'])]
        for name, checkpoint, filename, test_file, epoch in selections:
            report = read(output/test_file)
            assert report['checkpoint_epoch'] == epoch and report['checkpoint_sha256'] == digest(output/checkpoint)
            assert report['dataset_fingerprint'] == identity['dataset_fingerprint'] and report['split'] == 'test'
            stage('export_'+name, [sys.executable, frozen/'tools/export_inference.py',
                '--checkpoint', output/checkpoint, '--output', output/filename])
        calibration_args = []
        if confidence_contract is not None:
            if read_contract(dataset) != confidence_contract:
                raise ValueError('Dataset confidence rules changed during training')
            stage('calibrate_primary_presence', [sys.executable, frozen/'tools/calibrate_presence.py',
                '--checkpoint', output/'model_localization.pt', '--dataset', dataset,
                '--validation-cache', args.cache_dir.resolve()/'validation',
                '--output', output/'presence_calibration.json', '--threads', '4'])
            calibration_args = ['--calibration', output/'presence_calibration.json']
        validation_row = json.loads((dataset/'manifest_validation.jsonl').read_text().splitlines()[0])
        stage('cpu_inference', [sys.executable, '-m', 'pubg_audio.predict',
            '--checkpoint', output/'model_localization.pt', '--audio', dataset/validation_row['audio_file'],
            '--output', output/'cpu_inference_validation_0000.json', '--threads', '4', *calibration_args])
        prediction = read(output/'cpu_inference_validation_0000.json')
        expected_frames = round(validation_row['duration_seconds']*10)
        assert len(prediction['frames']) == expected_frames
        for frame in prediction['frames']:
            assert all(math.isfinite(value) for value in frame['self_probabilities'])
            for track in frame['raw_external_tracks']:
                assert all(math.isfinite(value) for value in track['raw_xyzd'])
            if confidence_contract is not None:
                for field in ['external_class_predictions', 'self_class_predictions']:
                    assert len(frame[field]) == 3
                    for row in frame[field]:
                        probability = row['presence_probability']
                        assert (probability is not None and 0 < probability < 1
                            and row['probability_status']=='fitted_on_synthetic_validation') or (
                            probability is None and row['probability_status']=='insufficient_labels')
        write(output/'export_verification.json', dict(status='passed', completed_epochs=info['completed_epochs'],
            primary_model='model_localization.pt', cpu_inference_frames=expected_frames,
            presence_calibration_file='presence_calibration.json' if confidence_contract else None,
            presence_calibration_sha256=digest(output/'presence_calibration.json') if confidence_contract else None,
            exports={name:dict(file=filename, checkpoint_epoch=epoch+1, sha256=digest(output/filename))
                     for name, _, filename, _, epoch in selections}))
        status.update(status='completed', updated_at=datetime.now(timezone.utc).isoformat())
        write(output/'pipeline_status.json', status)
        print(json.dumps(status), flush=True)
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, updated_at=datetime.now(timezone.utc).isoformat())
        write(output/'pipeline_status.json', status)
        raise


if __name__ == '__main__':
    main()
