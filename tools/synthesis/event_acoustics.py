"""Event-specific outer taper; no claim to reproduce native Wwise curves."""
import numpy as np
from motion_synthesis import SpatialRenderer


def profile_key(role,source):
    if role=='collision' and 'BRDM' in source.get('source_name',''):return 'collision_brdm'
    return role


class EventSpatialRenderer(SpatialRenderer):
    def __init__(self,base,acoustics,role):
        self.__dict__=base.__dict__.copy()
        self.acoustics,self.role=acoustics,role

    def distance_response(self,distance):
        d=np.maximum(distance*self.acoustics['metres_per_scene_unit_assumed'],1.)
        gain,cutoff=super().distance_response(d)
        # np.interp used to hold the final gain forever. Continue its last slope.
        exponent=-np.log(.045/.22)/np.log(400/90)
        gain*=np.minimum(1.,(400/d)**exponent)
        p=self.acoustics['profiles'].get(self.role)
        if p:
            radius=p['cutoff_m_estimate'];start=self.acoustics['taper_start_fraction']*radius
            q=np.clip((d-start)/(radius-start),0,1)
            gain*=1-q*q*(3-2*q)
            gain=np.where(d>=radius,0.,gain)
        return gain,cutoff


def render_components(track,library,spatial,seconds,cfg,legacy_render):
    n=round(seconds*44100);out=np.zeros((n,2),np.float32);ac=cfg['event_acoustics']
    groups={}
    for e in track['events']:
        key=profile_key(e['role'],library.sources[e['source_id']])
        groups.setdefault(key,[]).append(e)
    for key,events in groups.items():
        part=dict(track,events=events,_vehicle_layers=[])
        out+=legacy_render(part,library,EventSpatialRenderer(spatial,ac,key),seconds)
    if track['kind']=='vehicle':
        for key in track.get('available_vehicle_layers',['engine','roll','brake','skid']):
            part=dict(track,events=[],_vehicle_layers=[key])
            out+=legacy_render(part,library,EventSpatialRenderer(spatial,ac,key),seconds)
    return out


def annotate_recipes(scene,cfg,sources):
    ac=cfg.get('event_acoustics')
    if not ac:return
    for tr in scene['tracks']:
        direct=tr['source_role']=='self' or tr['kind']=='weather'
        tr['spatial_mode']='listener_bound_direct' if direct else 'world_source_listener_yaw_relative'
        for e in tr['events']:
            key=profile_key(e['role'],sources[e['source_id']])
            if tr['kind']=='footsteps':profile=cfg['footstep_acoustics']['profiles'][key]
            else:profile=ac['profiles'].get(key)
            e['attenuation']=dict(applied=not direct,profile=key,
                radius_m_estimate=profile['cutoff_m_estimate'] if profile and not direct else None,
                evidence=profile.get('evidence','gait_event_radius_with_inferred_mapping') if profile else 'radius_unknown_continuing_tail')
        if tr['kind']=='vehicle':
            tr['layer_attenuation']={k:dict(applied=not direct,**ac['profiles'][k]) for k in tr.get('available_vehicle_layers',['engine','roll','brake','skid'])}
