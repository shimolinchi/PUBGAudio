"""Freeze confidence supervision semantics and scene groups, never invent soft labels."""
import hashlib
import json
from pathlib import Path

import numpy as np

POLICY_SCHEMA = 'PUBGAudio-confidence-policy-v1'
PLAN_SCHEMA = 'PUBGAudio-calibration-plan-v1'
CLASSES = ['footsteps', 'vehicle', 'gunfire']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def validate_policy(policy):
    if (policy['schema'] != POLICY_SCHEMA or policy['classes'] != CLASSES
            or policy['roles'] != ['external', 'self'] or policy['output_hop_seconds'] != .1):
        raise ValueError('Unsupported confidence target schema')
    expected = dict(external_presence='counts > 0', external_valid='mask > 0',
        self_presence='self_target > 0', self_valid='self_mask > 0', label_values=[0, 1],
        masked_frames='exclude_from_positive_and_negative_supervision', generated_confidence_probability=None)
    if policy['ground_truth'] != expected:
        raise ValueError('Confidence requires binary truth and task masks, not generated probabilities')
    c = policy['calibration']
    if (c['split'] != 'validation' or c['group_by'] != 'scene_id' or c['use_test_labels'] is not False
            or c['preserve_frame_prevalence'] is not True or c['bind_exact_checkpoint'] is not True):
        raise ValueError('Calibration must preserve validation scene groups and prevalence')
    if not 0 < c['fit_fraction'] < 1 or not isinstance(c['seed'], int) or c['seed'] < 0:
        raise ValueError('Invalid calibration split settings')
    for key in ['minimum_positive_frames', 'minimum_negative_frames']:
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError('Invalid calibration sample minimum')
    expected_inference = dict(emit_presence_score=True, emit_presence_probability=True,
        uncalibrated_probability=None, direction_confidence=None, distance_confidence=None,
        probability_scope='frame_class_role_presence_not_position_or_individual_track')
    if policy['inference'] != expected_inference or policy['workflow'] != dict(
            calibrate_primary_export=True, primary_export='model_localization.pt'):
        raise ValueError('Unsupported confidence inference workflow')


def split_scenes(rows, seed, fit_fraction=.5):
    ids = sorted(r['id'] for r in rows)
    if len(ids) < 4 or len(set(ids)) != len(ids):
        raise ValueError('Need at least four distinct validation scenes')
    if not 0 < fit_fraction < 1:
        raise ValueError('Invalid calibration fit fraction')
    np.random.default_rng(seed).shuffle(ids)
    middle = int(len(ids)*fit_fraction)
    if min(middle, len(ids)-middle) < 2:
        raise ValueError('Need at least two whole scenes on each calibration side')
    return set(ids[:middle]), set(ids[middle:])


def make_plan(rows, policy, manifest_sha256):
    validate_policy(policy)
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('Only validation scenes may be used for calibration')
    c = policy['calibration']
    fit, audit = split_scenes(rows, c['seed'], c['fit_fraction'])
    return dict(schema=PLAN_SCHEMA, policy_sha256=policy_digest(policy),
        validation_manifest_sha256=manifest_sha256, seed=c['seed'],
        fit_scene_ids=sorted(fit), audit_scene_ids=sorted(audit),
        scope='synthetic_validation_class_presence',
        model_selection_independent=False, real_game_calibration=False)


def write_plan(root, targets):
    policy = targets.get('confidence_policy')
    if policy is None:
        return None
    root = Path(root)
    path = root/'manifest_validation.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    plan = make_plan(rows, policy, digest(path))
    destination = root/'calibration_plan.json'
    if destination.exists() and json.loads(destination.read_text(encoding='utf-8')) != plan:
        raise ValueError('Existing calibration plan differs; use a new dataset')
    destination.write_text(json.dumps(plan, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    return plan


def read_contract(root):
    root = Path(root)
    targets_file = root/'training_targets.json'
    targets = json.loads(targets_file.read_text(encoding='utf-8')) if targets_file.exists() else {}
    policy = targets.get('confidence_policy')
    if policy is None:
        if (root/'calibration_plan.json').exists():
            raise ValueError('Calibration plan has no matching target policy')
        return None
    identity_file = root/'generation_identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text(encoding='utf-8'))['targets'] != targets:
        raise ValueError('Confidence target policy differs from generation identity')
    path = root/'manifest_validation.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    expected = make_plan(rows, policy, digest(path))
    plan = json.loads((root/'calibration_plan.json').read_text(encoding='utf-8'))
    if plan != expected:
        raise ValueError('Calibration scene plan or validation manifest changed')
    return dict(policy=policy, plan=plan, policy_sha256=policy_digest(policy),
                plan_sha256=digest(root/'calibration_plan.json'))
