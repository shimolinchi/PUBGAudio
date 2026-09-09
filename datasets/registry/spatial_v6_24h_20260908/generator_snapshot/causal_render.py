"""Native source rendering and 10 ms physical/audible labels for v5 audition."""
from functools import lru_cache
from pathlib import Path
import wave
import numpy as np
from scipy.signal import resample_poly, lfilter
import motion_synthesis as m
import footstep_acoustics as foot
from event_acoustics import render_components
from listener_view import relative_scene,add_world_labels,to_listener


class Library(m.SourceLibrary):
    def __init__(self, root, sources, audio_settings=None):
        super().__init__(root, sources)
        self.audio_settings = audio_settings or {}

    @lru_cache(maxsize=128)
    def stereo(self, sid):
        source=self.sources[sid]
        x=m.read_wave(self.root/source['wav_file'])
        if x.shape[1]==1: x=np.repeat(x,2,axis=1)
        x=x-x.mean(axis=0,dtype=np.float64)
        role=source['role']
        target=.065 if source['class_index']==0 else .085
        if role in ['shot','local_shot']: target=.09
        elif role=='weather': target=.035
        elif role=='aircraft_pass': target=.16
        elif role=='c4_beep': target=.055
        elif role in ['molotov_fire','vehicle_fire']: target=.05
        elif role=='transition_rustle': target=.035
        elif role=='vault_rustle': target=.04
        target = self.audio_settings.get('source_rms_by_role', {}).get(role,
            self.audio_settings.get('source_rms_by_class', {}).get(str(source['class_index']), target))
        gain=min(target/max(float(np.sqrt(np.mean(x.astype(np.float64)**2))),1e-12),.55/max(float(abs(x).max()),1e-12))
        return (x*gain).astype(np.float32)

    @lru_cache(maxsize=128)
    def mono(self,sid):
        return self.stereo(sid).mean(axis=1)


def render_track(track,library,spatial,seconds,cfg=None):
    n=round(seconds*m.RATE); time=np.arange(n)/m.RATE
    direct=track['source_role']=='self' or track['kind']=='weather'
    if not direct and track['kind']=='footsteps' and foot.settings(cfg):
        out=np.zeros((n,2),np.float32)
        for gait in sorted({e['role'] for e in track['events']}):
            part=dict(track,events=[e for e in track['events'] if e['role']==gait])
            renderer=foot.FootstepSpatialRenderer(spatial,foot.settings(cfg),gait)
            out+=render_track(part,library,renderer,seconds)
        return out
    if not direct and cfg and cfg.get('event_acoustics'):
        return render_components(track,library,spatial,seconds,cfg,render_track)
    x=np.zeros((n,2) if direct else n,np.float32)
    for e in track['events']:
        src=library.stereo(e['source_id']) if direct else library.mono(e['source_id'])
        count=round(e['duration']*m.RATE)
        if e.get('loop'):
            if direct:
                src=np.column_stack([m.read_cycle(m.seamless_cycle(src[:,ear],round(.12*m.RATE)),np.arange(count)) for ear in range(2)])
            else:
                src=m.read_cycle(m.seamless_cycle(src,round(.12*m.RATE)),np.arange(count))
            fade=min(round(.08*m.RATE),count//4)
            if fade:
                env=np.ones(count); env[:fade]=m.smoothstep(np.arange(fade)/fade); env[-fade:]=env[:fade][::-1]
                src=src*env[:,None] if direct else src*env
        else:
            src=src[:count]
        start=round(e['start']*m.RATE); left=max(start,0); right=min(start+len(src),n)
        if right>left: x[left:right]+=src[left-start:right-start]
    p=track['path']; pt=np.asarray(p['time'])
    interp=lambda key: np.interp(time,pt,p[key])
    selected_layers=track.get('_vehicle_layers',track.get('available_vehicle_layers',['engine','roll','brake','skid']))
    if track['kind']=='vehicle' and selected_layers:
        smooth=lambda x,tau: lfilter([1-np.exp(-1/(m.RATE*tau))],[1,-np.exp(-1/(m.RATE*tau))],x)
        speed=interp('speed'); braking=interp('brake'); layers=track['layers']
        continuous=np.zeros(n,np.float32)
        if 'engine' in selected_layers:
            # Each component has its own distance envelope. Do not synthesize
            # the expensive engine again for the roll/brake/skid-only passes.
            on=smooth(interp('engine_on'),.025)
            accel=np.gradient(speed,1/m.RATE); throttle=np.clip((accel+.1)/2.5,0,1)
            rpm=smooth(np.clip(speed/32,0,1),.08)
            cursor=track['engine_phase_samples']+np.cumsum(.96+.83*rpm+.2*throttle)
            engine=m.read_cycle(library.cycle(layers['idle'],True),cursor)
            engine_lowpass = library.audio_settings.get('engine_low_throttle_cutoff_hz', 2000)
            soft=m.fft_convolve(engine,m.lowpass(engine_lowpass))[64:64+n]
            continuous=(soft*(1-throttle)+engine*throttle)*(.65+.24*rpm+.26*throttle)*on
        for role in ['roll','brake','skid']:
            if role not in selected_layers:continue
            src=library.cycle(layers[role]); y=m.read_cycle(src,np.arange(n)+track['engine_phase_samples'])
            # Tyres remain audible when the engine is off and the vehicle coasts.
            env=.22*m.smoothstep(speed/8) if role=='roll' else .5*braking*m.smoothstep(speed/.7)
            continuous=continuous+y*env
        x+=continuous[:,None] if direct else continuous
    x*=np.float32(10**(track['gain_db']/20))
    if direct:
        # Same documented common delay as the spatial lowpass, preserving native stereo.
        return np.pad(x,((64,0),(0,0)))[:n]
    return spatial.render(x,dict(time=pt,xy=np.asarray(p['xy'])))


def labels(scene,powers,cfg):
    tracks=scene['tracks']; k=len(tracks); seconds=scene['seconds']
    count=(round(seconds*32000)-1024)//320+1
    ends=(1024+np.arange(count)*320)/32000; centres=ends-.016-64/m.RATE
    z=dict(frame_right_edge_seconds=ends,frame_center_scene_seconds=centres)
    shape=(k,count)
    xy=np.zeros((k,count,2),np.float32); speeds=np.zeros(shape,np.float32); engine=np.zeros(shape,np.float32); brake=np.zeros(shape,np.float32)
    states=np.full(shape,'inactive',dtype='<U28')
    for i,tr in enumerate(tracks):
        p=tr['path']
        for a in range(2): xy[i,:,a]=np.interp(centres,p['time'],np.asarray(p['xy'])[:,a])
        for dst,key in [(speeds,'speed'),(engine,'engine_on'),(brake,'brake')]: dst[i]=np.interp(centres,p['time'],p[key])
        for s in tr['states']: states[i,(centres>=s['start'])&(centres<s['end'])]=s['state']
    if scene.get('listener_view'):
        view=scene['listener_view'];yaw=np.interp(centres,view['time'],view['yaw_degrees_unwrapped'])
        xy=to_listener(xy,yaw[None,:]).astype(np.float32)
    distance=np.linalg.norm(xy,axis=2); az=np.mod(np.degrees(np.arctan2(xy[:,:,0],xy[:,:,1])),360)
    direct=np.array([t['source_role']=='self' for t in tracks],bool); weather=np.array([t['kind']=='weather' for t in tracks],bool)
    valid=np.broadcast_to((~direct&~weather)[:,None],shape).copy() & (distance>1e-4)
    sector=((az+22.5)//45).astype(np.int8)%8; sector[~valid]=-1
    ranges=np.digitize(distance,[np.sqrt(8*30),np.sqrt(30*90)]).astype(np.int8); ranges[~valid]=-1
    maximum=powers.max(axis=1)[:,None] if k else np.zeros((0,1))
    active=powers>=np.maximum(maximum*10**(-35/10),1e-9)
    total=powers.sum(axis=0); cabin=powers[np.array([t['kind']=='vehicle' and t['source_role']=='self' for t in tracks],bool)].sum(axis=0)
    visible=active & (powers/np.maximum(total-powers,1e-15)>=10**(cfg['masking']['minimum_sir_db']/10))
    for i,tr in enumerate(tracks):
        if direct[i]: continue
        if tr['kind']=='footsteps': visible[i]&=powers[i]/np.maximum(cabin,1e-15)>=10**(cfg['masking']['cabin_foot_sir_db']/10)
        elif tr['kind']=='vehicle': visible[i]&=(distance[i]<cfg['masking']['cabin_far_distance_units']) | (powers[i]/np.maximum(cabin,1e-15)>=10**(cfg['masking']['cabin_far_vehicle_sir_db']/10))
    classes=len(scene['classes']); activity=np.zeros((count,classes,8),np.uint8); mask=np.ones_like(activity)
    self_activity=np.zeros((count,classes),np.uint8); self_mask=np.ones_like(self_activity)
    mask[:,scene['classes'].index('weather')]=0
    for i,tr in enumerate(tracks):
        cls=tr['class_index']; ids=np.flatnonzero(active[i])
        if direct[i]:
            self_activity[ids,cls]=1; self_mask[ids[~visible[i,ids]],cls]=0
        elif not weather[i]:
            ids=ids[valid[i,ids]]; activity[ids,cls,sector[i,ids]]=1
            hidden=ids[~visible[i,ids]]; mask[hidden,cls,sector[i,hidden]]=0
    loc_cfg = cfg.get('localization_supervision')
    if loc_cfg:
        sir = powers/np.maximum(total-powers, 1e-15)
        localizable = visible & valid & (sir >= 10**(loc_cfg['minimum_sir_db']/10))
        z['track_localization_observable'] = localizable.astype(np.uint8)
    z.update(activity=activity,activity_loss_mask=mask,self_activity=self_activity,self_activity_loss_mask=self_mask,
        track_xy=xy,track_distance_units=distance,track_azimuth_degrees=az,track_direction_valid=valid.astype(np.uint8),
        track_sector_index=sector,track_range_index=ranges,track_locomotion_speed=speeds,
        track_speed=np.where(direct[:,None],0,speeds),track_engine_on=engine,track_brake_control=brake,
        track_activity=active.astype(np.uint8),track_observable=visible.astype(np.uint8),track_power=powers.astype(np.float32),
        track_state=states,track_class_index=np.array([t['class_index'] for t in tracks],np.uint8),
        track_id=np.array([t['id'] for t in tracks],dtype='<U12'),
        track_source_role=np.array([t['source_role'] for t in tracks],dtype='<U16'),
        track_subtype=np.array([t['subtype'] for t in tracks],dtype='<U48'),
        track_surface=np.array([t.get('surface','') for t in tracks],dtype='<U24'),
        track_weapon_asset_id=np.array([t.get('weapon_asset_id','') for t in tracks],dtype='<U48'),
        classes=np.array(scene['classes'],dtype='<U20'),self_vehicle_power=cabin.astype(np.float32))
    if foot.settings(cfg):
        z.update(foot.detailed_labels(tracks,centres,distance,cfg))
    if cfg.get('body_actions'):
        posture=np.full(shape,'not_applicable',dtype='<U32')
        action=np.full(shape,'none',dtype='<U24')
        for i,tr in enumerate(tracks):
            for key,dst in [('posture_states',posture),('body_action_states',action)]:
                for s in tr.get(key,[]):dst[i,(centres>=s['start'])&(centres<s['end'])]=s['state']
            if tr.get('actor_parent'):
                ids=(centres>=tr['start'])&(centres<tr['end']);action[i,ids]=tr['subtype']
        z.update(track_posture=posture,track_body_action=action,
                 track_actor_id=np.array([tr.get('actor_parent',tr['id']) for tr in tracks],dtype='<U12'),
                 track_semantic_evidence=np.array([tr.get('semantic_evidence','') for tr in tracks],dtype='<U128'))
    return z


def render(scene,library,spatial,cfg,path):
    seconds=scene['seconds']; output=Path(path); output.mkdir(parents=True,exist_ok=True)
    n=round(seconds*m.RATE); mix=np.zeros((n,2),np.float32); powers=[]
    relative=relative_scene(scene)
    for track in relative['tracks']:
        signal=render_track(track,library,spatial,seconds,cfg)*cfg['audio']['master_gain']
        mix+=signal
        powers.append(m.frame_powers([signal],seconds)[0])
    if powers: power=np.asarray(powers)
    else: power=np.zeros((0,(round(seconds*32000)-1024)//320+1))
    z=labels(scene,power,cfg)
    add_world_labels(z,scene)
    x=resample_poly(mix,320,441,axis=0)
    peak=float(np.max(np.abs(x)))
    if peak >= .999:
        raise ValueError(f'Clipping risk: {scene["id"]}, peak {peak}; reduce fixed global gain and rerender all clips')
    pcm=np.rint(x*32767).astype('<i2')
    with wave.open(str(output/(scene['id']+'.wav')),'wb') as w:
        w.setnchannels(2);w.setsampwidth(2);w.setframerate(32000);w.writeframes(pcm.tobytes())
    np.savez_compressed(output/(scene['id']+'.npz'),**z)
    return dict(id=scene['id'],peak=peak,tracks=len(scene['tracks']),seconds=seconds,
                external_active_frames=int((z['track_activity']*z['track_direction_valid']).sum()),
                far_external_active_frames=int((z['track_activity']*z['track_direction_valid']*(z['track_distance_units']>=np.sqrt(30*90))).sum()))
