"""Audit all WAV/NPZ, source isolation and exact reproducibility of selected v4 scenes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import wave
import numpy as np
import generate_scenario_dataset as gen
from export_motion_labels import audit_and_export


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--reproduce',type=int,default=12)
    p.add_argument('--sources',type=Path,help='Override original source location after moving the project')
    p.add_argument('--hrir',type=Path,help='Override original HRIR location after moving the project')
    args=p.parse_args(); root=args.dataset.resolve()
    rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    by_id={s['id']:s for s in manifest['sources']}
    splitting=json.loads((root/'source_split.json').read_text(encoding='utf-8'))
    identity=json.loads((root/'generation_identity.json').read_text(encoding='utf-8'))
    counters=Counter(); used={}; hashes={}; weather_only=0
    for row in rows:
        audio=root/row['audio_file']; label=root/row['label_file']
        if gen.base.old.sha_file(audio)!=row['audio_sha256'] or gen.base.old.sha_file(label)!=row['label_sha256']:
            raise ValueError('Output changed '+row['id'])
        with wave.open(str(audio),'rb') as w:
            assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(2,2,44100,30*44100)
            samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
            assert samples.size==30*44100*2 and np.max(np.abs(samples.astype(np.int32)))<=22938
        with np.load(label,allow_pickle=False) as z:
            assert z['track_observable'].shape==z['track_activity'].shape
            assert not np.any(z['track_observable']>z['track_activity'])
            assert not z['activity_loss_mask'][:,3].any()
            if not row['tracks']:
                assert row['backgrounds'] and not z['activity'].any() and not z['self_activity'].any()
                weather_only+=1
            for i,track in enumerate(row['tracks']):
                cls=track['class_index']; own=track.get('source_role')=='self'
                assert bool(z['track_source_role'][i])==own
                hidden=np.flatnonzero(z['track_activity'][i]&~z['track_observable'][i].astype(bool))
                if not own:
                    assert not z['activity_loss_mask'][hidden,cls,z['track_sector_index'][i,hidden]].any()
                ids=list(track['layers'].values())+[e['source_id'] for e in track['events']]
                for sid in ids:
                    source=by_id[sid]
                    if cls==0:
                        assert source['source_group_id']==track['source_group_id']
                    if source['role']=='collision':
                        assert cls==1
                        counters['collision_source_uses']+=1
                    used.setdefault(sid,set()).add(row['split'])
                if cls==0:
                    sides=[e['simulated_foot'] for e in track['events']]
                    assert all(a!=b for a,b in zip(sides,sides[1:]))
                if cls==1 and 'emergency_brake_seconds' in track and not track.get('collision_only'):
                    a,b=track['emergency_brake_seconds']; assert 0<b-a<=.501
                    counters['emergency_brake_tracks']+=1
            for bg in row['backgrounds']:
                assert by_id[bg['source_id']]['role']=='weather' and not bg['direction_valid']
                used.setdefault(bg['source_id'],set()).add(row['split'])
        counters['clips']+=1
        if counters['clips']%500==0:
            print('VERIFY',counters['clips'],'/',len(rows),flush=True)
    for sid,splits in used.items():
        if splits!={'preview'}:
            assert len(splits)==1 and splitting['source_to_split'][sid] in splits
            assert splitting['group_to_split'][by_id[sid]['source_group_id']] in splits
        sha=by_id[sid]['pcm_sha256']
        hashes.setdefault(sha,set()).update(splits-{'preview'})
    assert all(len(splits)<=1 for splits in hashes.values())
    frame_audit=audit_and_export(root)
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    source_root=args.sources or Path(identity['source_root'])
    for source in manifest['sources']:
        assert gen.base.old.sha_file(source_root/source['wav_file'])==source['wav_sha256']
    gain=json.loads((root/'master_gain.json').read_text(encoding='utf-8'))['gain']
    gen.init_worker(str(source_root),manifest,str(args.hrir or identity['hrir_file']),root,cfg,gain)
    # Spread the replay across splits, then include weather and cabin scenes.
    indices=set(np.linspace(0,len(rows)-1,min(args.reproduce,len(rows)),dtype=int).tolist())
    for criterion in [lambda r:not r['tracks'],lambda r:r.get('self_policy')=='mixed_self_vehicle',
                      lambda r:any(e['role']=='collision' for t in r['tracks'] for e in t['events'])]:
        indices.update([next((i for i,r in enumerate(rows) if criterion(r)),0)])
    replayed=[]
    for index in sorted(indices):
        row=rows[index]; signals,paths=gen.render(row)
        mixed=np.sum(signals,axis=0,dtype=np.float32)*gain
        expected=np.rint(mixed*32767).astype('<i2').tobytes()
        with wave.open(str(root/row['audio_file']),'rb') as w:
            assert expected==w.readframes(w.getnframes()),'PCM replay '+row['id']
        powers=gen.motion.v3.frame_powers(signals,30)*gain**2
        computed=gen.motion.frame_labels(row['tracks'],paths,row['backgrounds'],30,cfg,powers)
        with np.load(root/row['label_file'],allow_pickle=False) as z:
            assert set(z.files)==set(computed)
            for key in z.files:
                np.testing.assert_array_equal(computed[key],z[key],err_msg=row['id']+':'+key)
        replayed.append(row['id'])
    report=dict(status='passed',counts=dict(counters),weather_only_scenes=weather_only,
        source_family_and_pcm_split_isolation=True,all_frame_geometry_passed=frame_audit['status']=='passed',
        exact_pcm_and_all_labels_reproduced=replayed,original_audio_unchanged=True,
        real_game_audibility_not_validated=True)
    gen.base.old.save_json(root/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
