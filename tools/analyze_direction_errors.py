"""Post-hoc direction diagnostics for a frozen model; no fitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pubg_audio.dataset import SyntheticScenes
from pubg_audio.features import FeatureNormalizer
from pubg_audio.model import ModelConfig, PaperSELD


CLASSES = ('footsteps', 'vehicle', 'gunfire')


def circular_error(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def summarize(rows):
    if not rows:
        return {'frames': 0}
    error = np.array([r['azimuth_error_deg'] for r in rows])
    matrix = np.zeros((4, 4), dtype=int)
    for r in rows:
        true_sector = int(((r['truth_deg'] + 45) % 360) // 90)
        predicted_sector = int(((r['predicted_deg'] + 45) % 360) // 90)
        matrix[true_sector, predicted_sector] += 1
    return dict(frames=len(rows), azimuth_mae_deg=float(error.mean()),
        angle_3d_mae_deg=float(np.mean([r['angle_3d_error_deg'] for r in rows])),
        median_azimuth_error_deg=float(np.median(error)),
        p90_azimuth_error_deg=float(np.quantile(error, .9)),
        within_20_deg_fraction=float(np.mean(error <= 20)),
        above_90_deg_fraction=float(np.mean(error > 90)),
        front_back_like_fraction=float(np.mean([r['front_back_like'] for r in rows])),
        left_right_like_fraction=float(np.mean([r['left_right_like'] for r in rows])),
        mean_abs_predicted_elevation_deg=float(np.mean([r['predicted_elevation_abs_deg'] for r in rows])),
        true_row_predicted_column_front_right_back_left=matrix.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-test', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    fingerprints = {s: hashlib.sha256((args.dataset / f'manifest_{s}.jsonl').read_bytes()).hexdigest()
                    for s in ['train', 'validation', 'test']}
    assert fingerprints == saved['dataset_fingerprint']
    model = PaperSELD(ModelConfig(**saved['model_config'])).to(args.device).eval()
    model.load_state_dict(saved['model'])
    normalizer = FeatureNormalizer(**saved['normalizer'])
    report = dict(checkpoint_epoch_zero_based=saved['epoch'], checkpoint_epoch_one_based=saved['epoch']+1,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        dataset_fingerprint=fingerprints, threshold=.5,
        scope='Post-hoc frozen-model diagnosis; existing test has been inspected. No tuning, fitting or checkpoint selection.',
        eligibility='Valid, detected, exactly one external source of this class; other classes or self may overlap. Highest xyz-norm track, never truth-based track selection.',
        front_back_like_rule='Azimuth error >60 degrees and error after reflecting predicted y <30 degrees; diagnostic heuristic, not proof of acoustic cause.',
        left_right_like_rule='Azimuth error >60 degrees and error after reflecting predicted x <30 degrees.',
        yaw_rule='Absolute unwrapped listener yaw rate at bin midpoint; observational subsets, not controlled comparisons.',
        splits={})
    for split in ['validation', 'test']:
        dataset = SyntheticScenes(args.dataset, split, normalizer=normalizer)
        views = {}
        for row in dataset.rows:
            scene = json.loads((args.dataset / row['recipe_file']).read_text(encoding='utf-8'))
            view = scene['listener_view']
            time = np.asarray(view['time'])
            yaw = np.asarray(view['yaw_degrees_unwrapped'])
            views[row['id']] = (time, np.abs(np.gradient(yaw, time)))
        records = {c: [] for c in CLASSES}
        with torch.inference_mode():
            for batch_number, batch in enumerate(DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)):
                raw = model(batch['features'].to(args.device))['accddoa'].cpu()
                scores = torch.linalg.vector_norm(raw[..., :3], dim=-1)
                confidence, index = scores.max(dim=2)
                chosen = raw.gather(2, index[:, :, None, :, None].expand(-1, -1, 1, -1, 4)).squeeze(2)
                eligible = (confidence >= .5) & (batch['counts'] == 1) & (batch['mask'] > 0)
                for cls, name in enumerate(CLASSES):
                    for b, f in torch.nonzero(eligible[..., cls]).tolist():
                        pred = chosen[b, f, cls, :3].numpy().astype(float)
                        truth = batch['target'][b, f, 0, cls, :3].numpy().astype(float)
                        pred /= max(np.linalg.norm(pred), 1e-12)
                        truth /= max(np.linalg.norm(truth), 1e-12)
                        pred_angle = float(np.degrees(np.arctan2(pred[0], pred[1])) % 360)
                        true_angle = float(np.degrees(np.arctan2(truth[0], truth[1])) % 360)
                        error = float(circular_error(pred_angle, true_angle))
                        sid = batch['scene_id'][b]
                        at = float(batch['start_seconds'][b]) + (f + .5) * .1
                        vt, speed = views[sid]
                        records[name].append(dict(scene_id=sid, time_seconds=at,
                            truth_deg=true_angle, predicted_deg=pred_angle, azimuth_error_deg=error,
                            angle_3d_error_deg=float(np.degrees(np.arccos(np.clip(np.dot(pred, truth), -1, 1)))),
                            predicted_elevation_abs_deg=float(abs(np.degrees(np.arcsin(np.clip(pred[2], -1, 1))))),
                            front_back_like=bool(error > 60 and circular_error(180 - pred_angle, true_angle) < 30),
                            left_right_like=bool(error > 60 and circular_error(-pred_angle, true_angle) < 30),
                            yaw_speed_deg_s=float(np.interp(at, vt, speed)),
                            true_distance_units=float(batch['target'][b, f, 0, cls, 3]) * saved['config'].get('distance_scale', 100.),
                            self_vehicle_present=bool(batch['self_target'][b, f, 1] > 0)))
                if batch_number % 6 == 0:
                    print(f'{split}: {min((batch_number+1)*8,len(dataset))}/{len(dataset)} windows', flush=True)
        summary = {}
        for name, rows in records.items():
            groups = {
                'yaw_below_5_deg_s': [r for r in rows if r['yaw_speed_deg_s'] < 5],
                'yaw_5_to_30_deg_s': [r for r in rows if 5 <= r['yaw_speed_deg_s'] < 30],
                'yaw_at_least_30_deg_s': [r for r in rows if r['yaw_speed_deg_s'] >= 30],
                'distance_below_40_units': [r for r in rows if r['true_distance_units'] < 40],
                'distance_40_to_100_units': [r for r in rows if 40 <= r['true_distance_units'] < 100],
                'distance_at_least_100_units': [r for r in rows if r['true_distance_units'] >= 100],
                'self_vehicle_present': [r for r in rows if r['self_vehicle_present']],
                'self_vehicle_absent': [r for r in rows if not r['self_vehicle_present']],
            }
            summary[name] = dict(overall=summarize(rows), subsets={k: summarize(v) for k, v in groups.items()},
                per_scene={sid: summarize([r for r in rows if r['scene_id'] == sid]) for sid in views})
        report['splits'][split] = summary
    reference = json.loads(args.reference_test.read_text(encoding='utf-8'))
    assert reference['checkpoint_epoch'] == saved['epoch']
    assert reference['checkpoint_sha256'] == saved['training_checkpoint_sha256']
    for name in CLASSES:
        expected = reference['diagnostics']['external']['per_class'][name]['conditional_localization']
        actual = report['splits']['test'][name]['overall']
        assert actual['frames'] == expected['detected_single_source_frames'], (name, actual, expected)
        assert abs(actual['angle_3d_mae_deg'] - expected['angular_mae_degrees']) < .002, (name, actual, expected)
    report['existing_test_metric_reproduced'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({split: {c: v['overall'] for c, v in classes.items()}
                      for split, classes in report['splits'].items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
