"""World +Y is initial forward; positive yaw rotates clockwise towards +X."""
import numpy as np
from motion_synthesis import smoothstep


def to_listener(xy, yaw_degrees):
    xy=np.asarray(xy); a=np.radians(yaw_degrees); c=np.cos(a);s=np.sin(a)
    return np.stack((xy[...,0]*c-xy[...,1]*s,xy[...,0]*s+xy[...,1]*c),axis=-1)


def make_view(planner,forced_mode=None):
    c=planner.cfg.get('listener_view')
    if not c:return None
    mode=forced_mode or planner.choice(c['mode_weights'],'listener_view_mode')
    initial=planner.uniform(c['initial_yaw_degrees'])
    yaw=np.full(len(planner.t),initial);turns=[];current=0.;value=initial
    if mode!='fixed':
        while current < planner.t[-1]:
            long=planner.draw(c['long_hold_probability'],'listener_long_hold')
            current+=planner.uniform(c['long_hold_seconds'] if long else c['hold_seconds'])
            kind=planner.choice(c['episode_weights'],'listener_episode')
            spec=c['episodes'][kind]
            amount=planner.uniform(spec['amplitude_degrees'])*planner.choice([-1,1],'listener_turn_sign')
            duration=max(planner.uniform(spec['seconds']),
                         1.5*abs(amount)/c['maximum_angular_speed_degrees_per_second'])
            if current+duration>planner.t[-1]:break
            yaw+=amount*smoothstep((planner.t-current)/duration)
            turns.append(dict(start=current,end=current+duration,from_degrees=value,to_degrees=value+amount,kind=kind))
            value+=amount;current+=duration
            if kind=='small_sway' and planner.draw(c['small_sway_return_probability'],'listener_sway_return'):
                back=-amount*planner.uniform(c['return_fraction']);current+=planner.uniform([.15,.6])
                if current+duration>planner.t[-1]:break
                yaw+=back*smoothstep((planner.t-current)/duration)
                turns.append(dict(start=current,end=current+duration,from_degrees=value,to_degrees=value+back,kind='sway_return'))
                value+=back;current+=duration
    return dict(time=planner.t,yaw_degrees_unwrapped=yaw,mode=mode,turns=turns,
                translation='stationary_world_origin',pitch_degrees=0)


def relative_scene(scene):
    view=scene.get('listener_view')
    if not view:return scene
    tracks=[]
    for tr in scene['tracks']:
        p=tr['path'];yaw=np.interp(p['time'],view['time'],view['yaw_degrees_unwrapped'])
        tracks.append(dict(tr,path=dict(p,xy=to_listener(p['xy'],yaw))))
    result=dict(scene,tracks=tracks)
    result.pop('listener_view')  # Coordinates are already transformed; prevent a second rotation.
    return result


def bind_self_throw_direction(scene):
    """Freeze a self throw's world ray at release; later head motion cannot steer it."""
    view=scene.get('listener_view')
    if not view:return
    tracks={t['id']:t for t in scene['tracks']}
    for chain in scene['chains']:
        if chain['kind'] not in ['frag','flash','molotov']:continue
        if tracks[chain['prepare']]['source_role']!='self':continue
        yaw=float(np.interp(chain['release'],view['time'],view['yaw_degrees_unwrapped']))
        angle=np.radians(yaw);forward=np.array([np.sin(angle),np.cos(angle)])
        origin=np.asarray(chain['release_xy']);fx=tracks[chain['effect']]
        distances=np.linalg.norm(np.asarray(fx['path']['xy'])-origin,axis=1)
        fx['path']=dict(fx['path'],xy=origin+distances[:,None]*forward)
        radius=np.linalg.norm(np.asarray(chain['detonation_xy'])-origin)
        chain['detonation_xy']=origin+radius*forward
        chain.update(release_listener_yaw_degrees=yaw,release_forward_world=forward,
                     launch_direction_binding='listener_yaw_at_release_then_fixed_world_path')


def add_world_labels(z,scene):
    view=scene.get('listener_view')
    if not view:return
    t=z['frame_center_scene_seconds'];world=np.zeros_like(z['track_xy'])
    for i,tr in enumerate(scene['tracks']):
        for a in range(2):world[i,:,a]=np.interp(t,tr['path']['time'],np.asarray(tr['path']['xy'])[:,a])
    yaw=np.interp(t,view['time'],view['yaw_degrees_unwrapped'])
    z.update(track_world_xy=world,listener_yaw_degrees=yaw.astype(np.float32),
             track_world_azimuth_degrees=np.mod(np.degrees(np.arctan2(world[...,0],world[...,1])),360).astype(np.float32))
