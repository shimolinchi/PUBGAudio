"""Audit immutable light-release manifests, labels, source isolation and replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'synthesis'))
from causal_render import Library,render
from generate_light_dataset import source_partitions
import motion_synthesis as motion
from air_absorption import configure_spatial
from project_causal_labels import project
from verify_causal_preview import verify_chain
from confidence_policy import read_contract


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--sources',type=Path,default=Path('datasets/sources-v5_3-final'))
    p.add_argument('--hrir',type=Path,default=Path('datasets/hrir/kemar-diffuse.zip'))
    p.add_argument('--replay',action='store_true')
    args=p.parse_args();root=args.dataset
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    identity=json.loads((root/'generation_identity.json').read_text())
    targets=json.loads((root/'training_targets.json').read_text())
    if targets != identity['targets']:
        raise ValueError('Training target definitions changed after generation')
    confidence_contract = read_contract(root)
    extension=json.loads((root/'extension.json').read_text()) if (root/'extension.json').exists() else None
    inherited=set(extension['inherited_scene_ids']) if extension else set()
    parent_cfg=json.loads((root/'parent_config.json').read_text()) if extension else cfg
    if extension:
        for split,sha in extension['parent_manifests_sha256'].items():
            parent_path=root/'parent_manifests'/f'manifest_{split}.jsonl'
            assert digest(parent_path)==sha
            current=(root/f'manifest_{split}.jsonl').read_bytes()
            assert current.startswith(parent_path.read_bytes())
            if split!='train':assert digest(root/f'manifest_{split}.jsonl')==sha
        parent_identity=json.loads((root/'parent_generation_identity.json').read_text())
        for name,sha in parent_identity['generator_sha256'].items():
            assert digest(root/'parent_generator_snapshot'/name)==sha
    assert digest(args.sources/'sources.json')==identity['source_manifest_sha256']
    assert digest(args.hrir)==identity['hrir_sha256']
    for name,sha in identity['generator_sha256'].items():
        assert digest(root/'generator_snapshot'/name)==sha
    manifest=json.loads((args.sources/'sources.json').read_text(encoding='utf-8'))
    pools=source_partitions(manifest['sources'],cfg['light_release'])
    for pool in pools.values():
        for s in pool:assert digest(args.sources/s['wav_file'])==s['wav_sha256']
    observed={}; ids=set(); audio_hashes=set(); peak=0.; replayed=[]; all_rows=[]
    for split in ['train','validation','test']:
        rows=[json.loads(line) for line in (root/f'manifest_{split}.jsonl').read_text().splitlines()]
        assert len(rows)==cfg['light_release']['split_counts'][split]
        allowed={s['id'] for s in pools[split]}
        by_id={s['id']:s for s in pools[split]}
        counts=Counter();curves=Counter();distances=Counter();densities=Counter()
        for row in rows:
            assert row['id'] not in ids and row['split']==split
            ids.add(row['id']); densities[row['density']]+=1
            assert set(row['source_ids'])<=allowed
            for field,key in [('audio_file','audio_sha256'),('label_file','label_sha256'),
                              ('detailed_label_file','detailed_label_sha256'),('recipe_file','recipe_sha256')]:
                assert digest(root/row[field])==row[key],row['id']
            scene=json.loads((root/row['recipe_file']).read_text(encoding='utf-8'))
            verify_chain(scene,parent_cfg if row['id'] in inherited else cfg)
            actual_sources={e['source_id'] for t in scene['tracks'] for e in t['events']}
            actual_sources.update(sid for t in scene['tracks'] for sid in t.get('layers',{}).values())
            assert actual_sources==set(row['source_ids'])
            for t in scene['tracks']:
                assert t['kind'] in ['footsteps','vehicle','gunfire']
                if t['kind']=='footsteps':
                    assert all(by_id[e['source_id']]['source_group_id']==t['source_group_id'] for e in t['events'])
                if t['source_role']=='external' and t['kind'] in ['footsteps','vehicle']:
                    curves[t['curve']]+=1;distances[t['distance_mode']]+=1
                counts['tracks/'+t['kind']+'/'+t['source_role']]+=1
                counts['collision_events']+=sum(e['role']=='collision' for e in t['events'])
            with wave.open(str(root/row['audio_file']),'rb') as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(32000,2,2)
                assert w.getnframes()==round(row['duration_seconds']*32000)
                pcm=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.int32)
                peak=max(peak,float(np.abs(pcm).max()/32767))
                # Independent empty scenes may both be silent; duplicated
                # non-silent recordings must never masquerade as new scenes.
                if np.any(pcm):
                    assert row['audio_sha256'] not in audio_hashes
                    audio_hashes.add(row['audio_sha256'])
            with np.load(root/row['detailed_label_file'],allow_pickle=False) as z, np.load(root/row['label_file'],allow_pickle=False) as coarse:
                n=(round(scene['seconds']*32000)-1024)//320+1
                assert z['track_xy'].shape==(len(scene['tracks']),n,2)
                distance=np.linalg.norm(z['track_xy'],axis=-1)
                np.testing.assert_allclose(distance,z['track_distance_units'],rtol=1e-5)
                np.testing.assert_allclose(distance,np.linalg.norm(z['track_world_xy'],axis=-1),rtol=2e-6,atol=1e-4)
                valid=z['track_direction_valid'].astype(bool)
                az=np.degrees(np.arctan2(z['track_xy'][...,0],z['track_xy'][...,1]))%360
                expected=(z['track_world_azimuth_degrees']-z['listener_yaw_degrees'][None,:])%360
                assert np.max(np.abs((az-expected+180)%360-180)[valid],initial=0)<.002
                own=z['track_source_role']=='self'
                assert not valid[own].any() and (distance[own]==0).all()
                assert (z['track_observable']<=z['track_activity']).all()
                for name,value in project(z,targets['external_targets'],targets['self_targets']).items():
                    assert np.array_equal(value,coarse[name]),(row['id'],name)
                for i,t in enumerate(scene['tracks']):
                    counts['observable_10ms/'+t['kind']+'/'+t['source_role']]+=int(z['track_observable'][i].sum())
            all_rows.append(row)
        for kind in ['footsteps','vehicle','gunfire']:
            for role in ['external','self']:
                assert counts['observable_10ms/'+kind+'/'+role]>0,(split,kind,role,'no positives')
        observed[split]=dict(scenes=len(rows),seconds=sum(r['duration_seconds'] for r in rows),
                            density=dict(densities),curves=dict(curves),distance_modes=dict(distances),counts=dict(counts))
        print(json.dumps(dict(status='split_verified',split=split,scenes=len(rows))),flush=True)
        if args.replay:
            replay_rows=[rows[0]]
            if extension and split=='train':replay_rows.append(next(r for r in rows if r['id'] not in inherited))
            for replay_row in replay_rows:
                scene=json.loads((root/replay_row['recipe_file']).read_text(encoding='utf-8'))
                output=root/'verification_replay'
                replay_cfg = parent_cfg if scene['id'] in inherited else cfg
                render(scene,Library(args.sources,pools[split],replay_cfg.get('audio')),
                       configure_spatial(motion.SpatialRenderer(args.hrir,identity['hrir_sha256']), replay_cfg),
                       replay_cfg,output)
                assert digest(output/(scene['id']+'.wav'))==replay_row['audio_sha256']
                with np.load(output/(scene['id']+'.npz')) as a,np.load(root/replay_row['detailed_label_file']) as b:
                    assert set(a.files)==set(b.files)
                    for key in a.files:assert np.array_equal(a[key],b[key],equal_nan=a[key].dtype.kind=='f'),key
                replayed.append(scene['id'])
                print('EXACT_REPLAY '+scene['id'],flush=True)
    assert peak<.999
    report=dict(status='passed',clips=len(all_rows),seconds=sum(r['duration_seconds'] for r in all_rows),
                source_files_verified=len({s['id'] for p in pools.values() for s in p}),
                source_and_pcm_overlap=len(set(s['pcm_sha256'] for s in pools['train']) & set(s['pcm_sha256'] for s in pools['test'])),
                evaluation_protocol=cfg['light_release'].get('split_protocol','independent_source_families'),
                exact_replays=replayed,pcm_peak=peak,observed=observed,
                manifests_sha256={s:digest(root/f'manifest_{s}.jsonl') for s in pools},
                scope='three-class light synthetic release; all sources/recipes/PCM/geometry/projections checked; not a listening assessment')
    if confidence_contract is not None:
        report['confidence_policy_sha256'] = confidence_contract['policy_sha256']
        report['calibration_plan_sha256'] = confidence_contract['plan_sha256']
    if extension:
        report['extension']=dict(inherited_scenes=len(inherited),new_scenes=len(extension['new_scene_ids']),
                                 held_out_manifests_unchanged=True)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='observed'}),flush=True)


if __name__=='__main__':main()
