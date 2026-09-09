"""Extract local event radii and build a versioned, explicitly provisional profile."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    info=json.loads(a.metadata.read_text(encoding='utf-8-sig'))['SoundBanksInfo']
    events={}
    for bank in info['SoundBanks']:
        for e in bank.get('IncludedEvents',[]):
            row=dict(bank=bank['ShortName'],event=e['Name'],event_id=e['Id'],object_path=e.get('ObjectPath',''),
                     max_attenuation_raw=float(e['MaxAttenuation']) if 'MaxAttenuation' in e else None)
            events[e['Name']]=row
    report=dict(metadata_sha256=hashlib.sha256(a.metadata.read_bytes()).hexdigest(),
        schema_version=info['SchemaVersion'],soundbank_version=info['SoundbankVersion'],
        interpretation='Raw event radius, not a measured hearing threshold or a recovered attenuation curve. Missing/zero values are not silence.',
        centimetres_per_metre_assumed=100,runtime_attenuation_scaling_factor_assumed=1,
        events=[e for e in events.values() if e['event'].startswith(('Footstep_','Weapon_','Vehicle_','HandBomb_','FireBomb_','Grenade_','CarePackage_','Ambience3D_SandStorm','Ambience3D_Blizzard'))])
    dest=root/'outputs/reports/attenuation_metadata_20260907.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    c=json.loads((root/'configs/generation_v5.json').read_text(encoding='utf-8'))
    c.update(revision='5.1-attenuation-listener-yaw',seed=202609073)
    c['notes'] += ['Footstep metre scale and event-radius cutoffs are provisional; runtime scaling and original curves are not recovered.',
                   'Other known event radii bound a tapered legacy curve; missing radii retain a decreasing tail without an invented cutoff.',
                   'World trajectories stay physical; output positions and azimuths used for training are relative to listener yaw. Horizontal rotation only.']
    ac=dict(reference_distance_m=5,amplitude_exponent=1,taper_start_fraction=.7,
            metres_per_scene_unit_assumed=1,profiles={},
            closest_radius_fraction={'far':[.55,.8],'near_pass':[.08,.22],'mid_pass':[.3,.5]},
            evidence='Metadata event maximum / 100 assuming UE centimetres and scaling factor 1. Gait mapping from source naming is inferred. Gain shape is a proposal, not measured PUBG.',
            metadata_sha256=report['metadata_sha256'])
    for role,stem in dict(walk='Stand_Slow',run='Stand_Normal',sprint='Stand_Fast',crouch_walk='Crouch_Slow',prone_crawl='Prone_Normal').items():
        e=events['Footstep_'+stem+'_Remote_Shoes']
        ac['profiles'][role]=dict(cutoff_m_estimate=e['max_attenuation_raw']/100,event=e['event'],event_id=e['event_id'])
    c['footstep_acoustics']=ac
    mapping={
      'engine':'Vehicle_Dacia_Engine_Remote','startup':'Vehicle_Dacia_Engine_Remote','shutdown':'Vehicle_Dacia_Engine_Stop',
      'roll':'Vehicle_Surface_Roll_Concrete','brake':'Vehicle_Surface_Brake_Concrete','skid':'Vehicle_Surface_Spin_Dirt',
      'collision':'Vehicle_Common_Impact','collision_brdm':'Vehicle_BRDM_Impact_Metal','tire_burst':'Vehicle_Common_TireExplosion',
      'vehicle_explosion':'Vehicle_Delayed_Explosion_Basic_Car','vehicle_fire':'Vehicle_Delayed_Explosion_Basic_Car',
      'vehicle_damage':'Vehicle_Delayed_Explosion_Basic_Car','frag_explosion':'Grenade_Explosion',
      'molotov_impact':'FireBomb_Impact','molotov_fire':'FireBomb_Blaze','c4_beep':'HandBomb_C4_Beep',
      'c4_attach':'HandBomb_C4_Attach','c4_switch_1':'HandBomb_C4_SwitchOn_Step01',
      'c4_switch_2':'HandBomb_C4_SwitchOn_Step02','c4_switch_3':'HandBomb_C4_SwitchOn_Step03',
      'c4_explosion':'HandBomb_C4_Explosion','aircraft_pass':'CarePackage_Aircraft'}
    profiles={}
    for role,name in mapping.items():
        e=events[name]
        if not e['max_attenuation_raw'] or e['max_attenuation_raw']<0:raise ValueError(name)
        status='event_radius_upper_bound_source_family_mapping'
        if role=='skid':status='proxy_from_surface_spin_event_not_native_skid_radius'
        if role in ['vehicle_fire','vehicle_damage']:status='composite_event_upper_bound_child_radius_unknown'
        profiles[role]=dict(cutoff_m_estimate=e['max_attenuation_raw']/100,event=name,event_id=e['event_id'],evidence=status)
    c['event_acoustics']=dict(profiles=profiles,taper_start_fraction=.7,metres_per_scene_unit_assumed=1,
        unknown_radius_roles=['shot','local_shot','pin','cook','flash_explosion','molotov_ignite'],
        fallback='continue_last_log_distance_slope_beyond_400_no_fixed_floor_no_invented_cutoff',
        weather='diffuse_listener_surrounding_bed_no_point_radius; sandstorm700m/blizzard400m are event envelopes, not a point-source gain rule',
        metadata_sha256=report['metadata_sha256'])
    c['listener_view']=dict(mode_weights={'fixed':.25,'natural':.75},
        initial_yaw_degrees=[0,360],hold_seconds=[2,8],long_hold_probability=.2,long_hold_seconds=[10,25],
        episode_weights={'small_sway':.6,'look_around':.25,'large_turn':.15},
        episodes={'small_sway':{'amplitude_degrees':[.5,6],'seconds':[.5,2]},
                  'look_around':{'amplitude_degrees':[8,35],'seconds':[1,3]},
                  'large_turn':{'amplitude_degrees':[60,160],'seconds':[.8,2]}},
        small_sway_return_probability=.65,return_fraction=[.4,1],
        maximum_angular_speed_degrees_per_second=140,rotation_easing='smoothstep_unwrapped_yaw',
        translation='stationary_listener_world_origin',pitch_degrees=0)
    c['formal_dataset_proposal']=dict(status='design_only_not_generated_not_enabled_in_training',
        clip_length_seconds_weights={'120':.25,'180':.5,'300':.25},
        proposed_history_seconds=[10,20,30],current_baseline_history_seconds=5,
        proposed_target='last_time_step_after_past_only_history',pre_roll_seconds=30,
        split_unit='source_family_and_parent_long_scene_before_cropping',storage='one_continuous_recording_random_crops_no_duplicate_audio',
        duration_budget='count_hours_not_clip_count; 25h regeneration not requested')
    (root/'configs/generation_v5_1.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(dest),events=len(report['events']),profiles=len(profiles))))


if __name__=='__main__':main()
