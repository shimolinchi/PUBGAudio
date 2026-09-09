"""Verify v5 detailed data, event chains, selected targets, and deterministic replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
from causal_render import Library, render
import motion_synthesis as motion
from project_causal_labels import project


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_chain(scene, cfg):
    tracks={t['id']:t for t in scene['tracks']}
    linked_effects=set()
    for c in scene['chains']:
        if c['kind'] in ['frag','flash','molotov']:
            prep,fx=tracks[c['prepare']],tracks[c['effect']]
            assert c['pin_or_ignite']<c['release']<=c['detonate']
            assert np.linalg.norm(np.array(c['release_xy'])-c['detonation_xy'])>1
            assert fx['source_role']=='external' and 'owner' not in fx
            assert prep['kind']=='action'
            if cfg['throwable'].get('self_throws_forward_at_release') and prep['source_role']=='self':
                view=scene['listener_view'];yaw=np.interp(c['release'],view['time'],view['yaw_degrees_unwrapped'])
                delta=np.array(c['detonation_xy'])-c['release_xy']
                expected=np.array([np.sin(np.radians(yaw)),np.cos(np.radians(yaw))])
                np.testing.assert_allclose(delta/np.linalg.norm(delta),expected,atol=1e-8)
                # Rotation after release must not drag the landed effect with the listener.
                after=np.array(fx['path']['xy'])[np.array(fx['path']['time'])>=c['detonate']]
                np.testing.assert_allclose(after,np.broadcast_to(after[0],after.shape),atol=1e-7)
            if c['kind']=='frag': assert np.isclose(c['detonate'],c['fuse_deadline'])
            if c['kind']=='flash': assert np.isclose(c['detonate'],min(c['fuse_deadline'],c['first_impact']+cfg['throwable']['flash_impact_delay_seconds']))
            if c['kind']=='molotov': assert next(e for e in fx['events'] if e['role']=='molotov_fire')['duration']==cfg['throwable']['molotov_burn_seconds']
            linked_effects.add(fx['id'])
        elif c['kind']=='c4':
            assert np.isclose(c['detonate']-c['activation'],cfg['c4']['fuse_seconds'])
            assert np.isclose(c['activation']-tracks[c['prepare']]['start'],cfg['c4']['deploy_seconds'])
            intervals=np.diff(c['beep_times'])
            assert np.all(np.diff(intervals)<=1e-7)
            assert c['beep_times'][0]==c['activation'] and c['beep_times'][-1]<c['detonate']
            assert tracks[c['warning']]['kind']=='warning'
            linked_effects.add(c['effect'])
        elif c['kind']=='vehicle_attack':
            assert tracks[c['shooter']]['kind']=='gunfire'
            assert max(e['start'] for e in tracks[c['shooter']]['events'])<c['hit']
            if c['outcome']=='fatal':
                assert np.isclose(c['detonate']-c['engine_failure'],cfg['vehicle']['destroy_delay_seconds'])
                assert not any(e['role'].startswith('startup') and e['start']>=c['engine_failure'] for e in tracks[c['vehicle']]['events'])
                linked_effects.update([c['fire_track'],c['explosion_track']])
        elif c['kind'] in ['posture_change','vault']:
            actor,fx=tracks[c['actor']],tracks[c['effect']]
            assert actor['kind']=='footsteps' and fx['kind']=='action'
            assert fx['actor_parent']==actor['id'] and fx['source_role']==actor['source_role']
            assert actor['start']<=c['start']<c['end']<=actor['end']
            np.testing.assert_allclose(fx['path']['xy'],actor['path']['xy'])
            assert not any(c['start']<=e['start']<c['end'] for e in actor['events'])
            assert all(e['start']+e['duration']<=c['end']+1e-6 for e in fx['events'])
            if c['kind']=='posture_change':
                assert c['posture_from']!=c['posture_to']
                assert all(e['role']=='transition_rustle' for e in fx['events'])
            else:
                assert [e['role'] for e in fx['events'][:3]]==['vault_grab','vault_climb','vault_contact']
                mid=(c['start']+c['end'])/2
                np.testing.assert_allclose(c['obstacle_xy'],[np.interp(mid,actor['path']['time'],np.array(actor['path']['xy'])[:,i]) for i in range(2)])
    for t in tracks.values():
        if t['kind'] in ['explosion','fire']: assert t['id'] in linked_effects, (scene['id'],t['id'],'orphan effect')
        if t['kind']=='gunfire': assert len(t['events'])==t['shots_fired']<=t['magazine_limit']
        if t['kind']=='aircraft':
            xy=np.array(t['path']['xy']);delta=np.diff(xy,axis=0)
            assert np.allclose(delta,delta[0],atol=1e-8)
            assert np.linalg.norm(xy[0])>1500 and np.linalg.norm(xy[-1])>1500
        if t['kind'] in ['footsteps','vehicle'] and t['source_role']!='self':
            path=t['path']; speed=np.array(path['speed'])
            actual=np.linalg.norm(np.diff(path['xy'],axis=0),axis=1)/np.diff(path['time'])
            assert np.max(abs(actual-(speed[1:]+speed[:-1])/2))<.04
    if scene.get('scenario')=='coast_collision':
        t=next(t for t in tracks.values() if t['kind']=='vehicle')
        hit=next(e['start'] for e in t['events'] if e['role']=='collision')
        assert np.interp(hit,t['path']['time'],t['path']['engine_on'])==0
        assert np.interp(hit,t['path']['time'],t['path']['speed'])>1


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('root',type=Path)
    a.add_argument('--sources',type=Path,default=Path('datasets/sources-v5'))
    a.add_argument('--hrir',type=Path,default=Path('datasets/hrir/kemar-diffuse.zip'))
    a.add_argument('--replay',nargs='*',default=[])
    args=a.parse_args();root=args.root
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    targets=json.loads((root/'training_targets.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    scenes=[json.loads(l) for l in (root/'recipes.jsonl').read_text(encoding='utf-8').splitlines()]
    source_file=args.sources/'sources.json'
    assert digest(source_file)==summary['source_manifest_sha256']
    sources=json.loads(source_file.read_text(encoding='utf-8'))['sources'];source_by_id={s['id']:s for s in sources}
    for s in sources: assert digest(args.sources/s['wav_file'])==s['wav_sha256'],s['id']
    assert digest(args.hrir)==summary['hrir_sha256']
    snapshot_hashes={}
    for path in (root/'generator_snapshot').glob('*.py'):
        assert digest(path)==digest(Path(__file__).with_name(path.name)),f'Code changed since rendering: {path.name}'
        snapshot_hashes[path.name]=digest(path)
    rows=[];density=Counter();hidden=Counter();random_far=0;random_external=0
    for scene in scenes:
        sid=scene['id'];wav=root/'audio'/(sid+'.wav');label=root/'audio'/(sid+'.npz');coarse=root/'audio'/(sid+'_train3.npz')
        with wave.open(str(wav),'rb') as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(32000,2,2)
            assert w.getnframes()==round(scene['seconds']*32000)
            pcm=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.int32)
            peak=float(np.abs(pcm).max()/32767)
            assert peak<.999
        verify_chain(scene,cfg)
        with np.load(label,allow_pickle=False) as z, np.load(coarse,allow_pickle=False) as selected:
            count=(round(scene['seconds']*32000)-1024)//320+1;k=len(scene['tracks'])
            assert z['track_xy'].shape==(k,count,2)
            for name in z.files:
                if z[name].dtype.kind in 'f':
                    if name in ['track_distance_m_estimate','track_footstep_cutoff_m_estimate','track_footstep_distance_gain_estimate']:
                        assert not np.isinf(z[name]).any(),(sid,name)
                    else:assert np.isfinite(z[name]).all(),(sid,name)
            distance=np.linalg.norm(z['track_xy'],axis=2)
            assert np.allclose(distance,z['track_distance_units'],rtol=1e-5)
            az=np.mod(np.degrees(np.arctan2(z['track_xy'][:,:,0],z['track_xy'][:,:,1])),360)
            valid=z['track_direction_valid'].astype(bool)
            difference=np.mod(az-z['track_azimuth_degrees']+180,360)-180
            assert np.abs(difference[valid]).max(initial=0)<.001
            self_ids=z['track_source_role']=='self';weather=z['track_class_index']==scene['classes'].index('weather')
            assert not z['track_direction_valid'][self_ids|weather].any()
            assert np.all(distance[self_ids]==0)
            assert np.all(z['track_observable']<=z['track_activity'])
            if 'listener_yaw_degrees' in z:
                world=z['track_world_xy'];world_distance=np.linalg.norm(world,axis=2)
                np.testing.assert_allclose(distance,world_distance,rtol=2e-6,atol=1e-4)
                expected_az=(z['track_world_azimuth_degrees']-z['listener_yaw_degrees'][None,:])%360
                delta=(az-expected_az+180)%360-180
                assert np.max(abs(delta[valid]),initial=0)<.002,(sid,'listener angle')
                external_foot=(z['track_class_index']==0)&~self_ids
                assert np.isfinite(z['track_footstep_cutoff_m_estimate'][external_foot]).all()
                assert np.isfinite(z['track_footstep_distance_gain_estimate'][external_foot]).all()
            expected=project(z,targets['external_targets'],targets['self_targets'])
            assert set(expected)==set(selected.files)
            for name,value in expected.items(): assert np.array_equal(value,selected[name]),(sid,name)
            if scene['selection']=='independent_random':
                density[scene['density']]+=1
                active=z['track_activity'].astype(bool)&valid
                random_external+=int(active.sum());random_far+=int((active&(distance>=np.sqrt(30*90))).sum())
            for i,t in enumerate(scene['tracks']):
                hidden[t['kind']]+=int((z['track_activity'][i].astype(bool)&~z['track_observable'][i].astype(bool)).sum())
                if t['kind']=='footsteps':
                    assert all(source_by_id[e['source_id']]['source_group_id']==t['source_group_id'] for e in t['events'])
            if not any(t['kind'] in ['footsteps','vehicle','gunfire'] for t in scene['tracks']):
                assert np.all(selected['track_class_index']==255)
        rows.append(dict(id=sid,wav_sha256=digest(wav),detailed_labels_sha256=digest(label),projection_sha256=digest(coarse),pcm_peak=peak))
    replayed=[]
    if args.replay:
        lib=Library(args.sources,sources);spatial=motion.SpatialRenderer(args.hrir,summary['hrir_sha256'])
        for sid in args.replay:
            scene=next(s for s in scenes if s['id']==sid)
            dest=root/'_verification_replay';render(scene,lib,spatial,cfg,dest)
            assert digest(dest/(sid+'.wav'))==digest(root/'audio'/(sid+'.wav')),sid
            with np.load(dest/(sid+'.npz'),allow_pickle=False) as replay,np.load(root/'audio'/(sid+'.npz'),allow_pickle=False) as original:
                assert set(replay.files)==set(original.files)
                for name in original.files: assert np.array_equal(original[name],replay[name],equal_nan=original[name].dtype.kind=='f'),(sid,name)
            replayed.append(sid);print('EXACT_REPLAY',sid,flush=True)
    result=dict(status='passed',clips=len(scenes),seconds=sum(s['seconds'] for s in scenes),source_files_verified=len(sources),
        exact_replays=replayed,random_density=dict(density),random_external_active_frames=random_external,random_far_active_frames=random_far,
        random_far_frame_fraction=random_far/max(random_external,1),hidden_track_frames=dict(hidden),files=rows,
        config_sha256=digest(root/'config.json'),training_target_config_sha256=digest(root/'training_targets.json'),
        recipes_sha256=digest(root/'recipes.jsonl'),generator_sha256=snapshot_hashes,
        human_listening_status='pending_user_feedback',training_split_status='audition_only_no_train_test_split')
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary['status']='machine_verified_pending_user_listening'
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','generator_sha256']},ensure_ascii=False))


if __name__=='__main__':main()
