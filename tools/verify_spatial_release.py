"""Additional v6 gates: audible semantic coverage, firing rules, and spatial masks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args();root=args.dataset
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert cfg['light_release']['split_protocol']=='known_assets_new_scenes'
    seen={};report={}
    for split in ['train','validation','test']:
        rows=[json.loads(l) for l in (root/f'manifest_{split}.jsonl').read_text().splitlines()]
        groups=Counter();directions=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        audible=Counter();loc=Counter();patterns=Counter();auto_shots=Counter();max_foot=0.
        for index,row in enumerate(rows):
            scene=json.loads((root/row['recipe_file']).read_text(encoding='utf-8'))
            with np.load(root/row['detailed_label_file'],allow_pickle=False) as z:
                np.testing.assert_array_less(z['track_localization_observable'],z['track_observable']+1)
                assert not (z['track_localization_observable'].astype(bool)&~z['track_direction_valid'].astype(bool)).any()
                for i,tr in enumerate(scene['tracks']):
                    role=tr['source_role'];key=tr['source_group_id']+'|'+role
                    visible=z['track_observable'][i].astype(bool)
                    n=int(visible.sum());groups[key]+=n
                    audible[tr['kind']+'/'+role]+=n
                    loc[tr['kind']+'/'+role]+=int(z['track_localization_observable'][i].sum())
                    if role=='external' and n:
                        az=z['track_azimuth_degrees'][i,visible]
                        directions[key]+=np.bincount(((az+45)//90).astype(int)%4,minlength=4)
                        if tr['kind']=='footsteps':
                            max_foot=max(max_foot,float(z['track_distance_units'][i,visible].max()))
                    if tr['kind']=='gunfire':
                        profile=cfg['gunfire']['profiles'][tr['weapon_asset_id']]
                        assert len(tr['events'])==tr['shots_fired']<=tr['magazine_limit']
                        pattern=tr['firing_pattern'];patterns[pattern]+=1
                        if profile['weapon_kind'] in ['dmr','bolt']:assert pattern=='single'
                        if 'auto' in pattern:assert profile['weapon_kind']=='automatic'
                        if tr['events']:
                            gaps=np.diff([e['start'] for e in tr['events']])
                            assert (gaps>=profile['interval_seconds']-1e-6).all()
                            if 'auto' in pattern:auto_shots[tr['weapon_asset_id']]+=len(tr['events'])
            if index%100==99:
                print(json.dumps(dict(status='coverage_audit',split=split,scenes=index+1)),flush=True)
        # A 32ms energy window and HRIR/filter tail can straddle the hard radius;
        # no footsteps are permitted well beyond the configured 50-unit limit.
        assert max_foot<=50.5,(split,max_foot)
        seen[split]={g for g,n in groups.items() if n>0}
        report[split]=dict(scenes=len(rows),audible_group_frames=dict(groups),
            direction_quadrants={g:v.tolist() for g,v in directions.items()},
            audible_10ms_track_frames=dict(audible),localizable_10ms_track_frames=dict(loc),
            gun_patterns=dict(patterns),automatic_shots_by_weapon=dict(auto_shots),
            maximum_observable_external_footstep_distance_units=max_foot)
    if not args.pilot:
        manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))['sources']
        expected=set()
        for s in manifest:
            if s['source_group_id'] in cfg['light_release']['source_groups'] and s['role'] in ['shot','local_shot']:
                expected.add(s['source_group_id']+'|'+('self' if s['role']=='local_shot' else 'external'))
        assert expected<=seen['train'],('Training gun coverage missing',sorted(expected-seen['train']))
        expected_external={g+'|external' for g in cfg['light_release']['source_groups']
                           if g.startswith(('footsteps:','vehicle:'))}
        expected_external|={g for g in expected if g.endswith('|external')}
        assert expected_external<=seen['train'],('Training external type coverage missing',sorted(expected_external-seen['train']))
        for split in ['validation','test']:
            missing={g for g in seen[split] if g.startswith('gunfire:')}-seen['train']
            assert not missing,('Unseen evaluation gun type/role',split,missing)
        holes={g:q for g,q in report['train']['direction_quadrants'].items()
               if g.startswith(('gunfire:','vehicle:','footsteps:')) and min(q)==0}
        assert not holes,('Training external direction quadrant coverage missing',holes)
        for split in report:
            assert report[split]['gun_patterns'].get('sustained_auto',0)>0
            for kind in ['footsteps','vehicle','gunfire']:
                assert report[split]['localizable_10ms_track_frames'].get(kind+'/external',0)>0
    result=dict(status='passed',protocol='known_assets_new_scenes',independent_source_recordings=False,
        every_evaluation_gun_type_and_role_seen_in_training=not args.pilot,
        all_available_gun_type_roles_audible_in_training=not args.pilot,
        all_selected_external_target_types_audible_in_training=not args.pilot,
        training_external_types_cover_four_quadrants=not args.pilot,
        scope='v6 acoustic, semantic and firing gates; frame counts are correlated and units are not calibrated game metres',
        splits=report)
    (root/'coverage_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='splits'}),flush=True)


if __name__=='__main__':main()
