"""Prepare v6.1 probability/config files only; never generate audio or train."""
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def build():
    cfg = json.loads((ROOT/'configs/generation_v6.json').read_text(encoding='utf-8'))
    spec = json.loads((ROOT/'configs/dataset_v6_24h.json').read_text(encoding='utf-8'))
    cfg.update(revision='6.1-external-footsteps-and-vehicle-interference', display_version='v6.1',
               purpose='long external-footstep scenarios and rendered vehicle interference policy')
    cfg['external_footsteps'] = dict(self_probability_when_eligible=.3, maximum_simultaneous=2,
        vehicle_guard=dict(self_vehicle_keep_probability=.01,loud_vehicle_keep_probability=.02,
            self_vehicle_power_floor=1e-8,vehicle_power_floor=1e-6,vehicle_over_foot_db=6.,
            minimum_overlap_seconds=.5,minimum_affected_fraction=.05,
            evidence='Synthesis policy using actual rendered power; thresholds are provisional, not game measurements'))
    view = copy.deepcopy(cfg['listener_view'])
    view.update(mode_weights=dict(fixed=.05,natural=.95),hold_seconds=[.15,.7],
        long_hold_probability=.05,long_hold_seconds=[2,5],
        episode_weights=dict(small_sway=.55,look_around=.15,large_turn=.30))
    cfg['scene_profiles'] = dict(weights=dict(mixed=.75,external_footsteps_only=.25),
        external_footsteps_only=dict(clip_seconds=240,source_count_weights={'1':.65,'2':.35},
            start_seconds=[1,4],second_source_delay_seconds=[8,24],end_margin_seconds=[1,4],
            gait_bouts=dict(minimum_dwell_seconds=6,switch_probability_after_minimum=.08,
                maximum_dwell_seconds=18,pause_probability_after_bout=.45,pause_duration_seconds=[1,4]),
            distance_mode_weights=dict(far=.60,mid_pass=.10,near_pass=.10,near_far=.20),
            radius_fraction_by_mode=dict(far=[.58,.85],mid_pass=[.35,.58],near_pass=[.15,.35],near_far=[.15,.85]),
            curve_weights=dict(oval=1,figure_eight=1,bezier_loop=1),bearing_span_degrees=[45,110],
            listener_view=view))
    spec.update(base_config='configs/generation_v6_1.json',dataset_id='spatial-v6_1-footsteps-pending',
                seed=2026090810,confidence_policy_file='configs/confidence_v1.json')
    spec['spawn_probability_multipliers']['footsteps'] = 1.
    notes = ['v6.1 is script/config preparation only; no dataset or training launched.',
        'Every split uses the same75/25 mixed/pure external-footstep scene probabilities, independent seeds.',
        'Mixed scenes120s; pure footstep scenes240s,1 or2 external actors65/35; total hours must sum actual durations.',
        'External-footstep admission uses rendered power from all vehicles including later arrivals; conflicting whole actors kept1% in self vehicles or2% in loud traffic.',
        'Footstep-only local paths stay inside each actor gait radius, with pauses and independently sampled source geometry; fixed surface/gait per actor.',
        'Existing v6 acoustic levels, ranges, gun/vehicle rules and source split remain; real-game calibration pending.']
    cfg['notes'] += notes
    spec['notes'] = notes
    return cfg, spec


def main():
    cfg, spec = build()
    for name, value in [('generation_v6_1.json',cfg),('dataset_v6_1.json',spec)]:
        (ROOT/'configs'/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Prepared v6.1 configuration files only; audio generation and training were not started.')


if __name__ == '__main__':
    main()
