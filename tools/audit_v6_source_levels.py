"""Audit the actual peak-limited source levels; no files or mix levels are changed."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent/'synthesis'))
import motion_synthesis as motion
from footstep_acoustics import gain_at_distance


def main():
    root = Path('datasets/sources-v5_3-final')
    sources = json.loads((root/'sources.json').read_text())['sources']
    cfg = json.loads(Path('configs/generation_v6.json').read_text())
    spec = json.loads(Path('configs/dataset_v6_24h.json').read_text())
    stats = defaultdict(list)
    for source in sources:
        if source['source_group_id'] not in spec['source_groups']:
            continue
        role = source['role']
        group = 'footsteps' if source['class_index'] == 0 and role in spec['included_gaits'] else 'gunfire' if role in ['shot','local_shot'] else None
        if group is None:
            continue
        x = motion.read_wave(root/source['wav_file']).astype(np.float64)
        x -= x.mean(axis=0)
        rms, peak = float(np.sqrt(np.mean(x*x))), float(np.abs(x).max())
        old_target = .065 if group == 'footsteps' else .09
        target = .025 if group == 'footsteps' else .12
        old = min(old_target/max(rms,1e-12), .55/max(peak,1e-12))
        new = min(target/max(rms,1e-12), .55/max(peak,1e-12))
        stats[group].append(dict(source_id=source['id'], old_dry_rms=rms*old, dry_rms=rms*new,
            dry_peak=peak*new, source_amplitude_ratio=new/old,
            source_and_master_ratio=new*.35/(old*.6)))
    summary = {}
    for group, rows in stats.items():
        summary[group] = dict(sources=len(rows))
        for key in ['old_dry_rms','dry_rms','dry_peak','source_amplitude_ratio','source_and_master_ratio']:
            values = [r[key] for r in rows]
            summary[group][key] = dict(zip(['min','median','max'], map(float,np.quantile(values,[0,.5,1]))))
    distance = [1,5,10,20,30,35,40,45,50,60]
    result = dict(status='measured_from_selected_native_assets', source_levels=summary,
        footstep_gain_by_gait={g:dict(zip(map(str,distance),map(float,gain_at_distance(np.array(distance),cfg['footstep_acoustics'],g))))
                               for g in spec['included_gaits']},
        scope='Dry-source RMS/peak and analytical footstep amplitude only; no claim of mixed-event loudness or calibrated game metres')
    Path('outputs/reports/source_level_audit_v6_20260908.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
