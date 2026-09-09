"""Write the reproducible v6 24-hour release and training probability tables."""
from collections import defaultdict
import json
from pathlib import Path


def main():
    sources = json.loads(Path('datasets/sources-v5_3-final/sources.json').read_text())['sources']
    groups = defaultdict(list)
    for source in sources:
        groups[source['source_group_id']].append(source)
    cfg = json.loads(Path('configs/generation_v5_3.json').read_text())
    cfg['revision'] = '6.0-known-types-spatial-loss-and-natural-firing'
    cfg['display_version'] = 'v6'
    cfg['air_absorption'] = dict(frequencies_hz=[125,250,500,1000,2000,4000,8000],
        energy_coefficients_per_m=[.0001,.0003,.0006,.001,.0017,.0041,.0135],
        distance_grid=[1,8,30,60,90,150,250,400,800,1600],
        reference='https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html',
        evidence='Physical energy attenuation table for 20 C / 50-70% RH; not a measured PUBG sound engine curve',
        implementation='129-tap linear-phase FIR; amplitude exp(-energy_coefficient*distance/2); common delay64')
    cfg['audio'].update(master_gain=.35, source_rms_by_class={'0':.025},
        source_rms_by_role={'shot':.12,'local_shot':.12}, engine_low_throttle_cutoff_hz=6000)
    cfg['footstep_acoustics']['profiles']['sprint']['cutoff_m_estimate'] = 50.
    cfg['footstep_acoustics']['evidence'] += '; v6 sprint limit conservatively capped at50 by user request, not a newly measured universal game radius.'
    cfg['localization_supervision'] = dict(minimum_sir_db=-10,
        scope='Direction and distance supervision requires audible spatial evidence; presence retains its existing mask. No distance-only exclusion.')
    cfg['distance_mode_weights'] = dict(far=.7, near_pass=.1, mid_pass=.1, approach=.05, recede=.05)
    cfg['stationary_distance']['far_probability'] = .7
    cfg['stationary_distance']['near_units'] = [8,60]
    cfg['notes'] += ['v6 uses an explicit known-client-assets/new-scenes evaluation: original PCM can appear across splits, scenes never do.',
        'Far events remain dominant (70%); 20% near/mid and10% approach/recede improve coverage.',
        'Gun/foot source RMS targets .12/.025; no clip-wise output normalization; master .35.',
        'Standing walk/run/sprint maximum assumed distances35/45/50; no claim of exact current game audibility.',
        'Manual, DMR and bolt weapons never synthesize automatic fire; fixed client sample tails may overlap between physical shots.',
        'Engine low-throttle filter6kHz and physical air absorption replace undocumented aggressive2kHz/remote brick lowpasses; real-game calibration still pending.']
    auto = set('AK47 Ace32 Bizon FamasG2 G36C Glock18C Groza HK416 JS9 K2 M249 M762 MP5K MicroUzi O12 QBZ SCAR_L Scorpion Steyr_AUG_A3 Thompson UMP Vector DP28'.split())
    dmr = set('Dragunov FNFAL M14EBR MK12 Mini14 QBU SKS VSS L6'.split())
    bolt = set('AWM Kar98k M24 MosinNagant Winchester1894'.split())
    shotgun = set('Berreta686 SawedOffShotgun Winchester1897 DP12 Saiga12'.split())
    magazines = dict(AWM=5,Kar98k=5,M24=5,MosinNagant=5,Winchester1894=8,L6=10,
        Berreta686=2,SawedOffShotgun=2,Winchester1897=5,DP12=14,Saiga12=5,O12=12,
        M249=75,DP28=47,Vector=19,MicroUzi=25,UMP=25,Thompson=30,Bizon=53,
        M1911=7,NagantM1895=7,Rhino=6,DesertEagle=7,BerretaM9=15,Glock18C=17,Scorpion=20,
        Dragunov=10,FNFAL=10,M14EBR=10,MK12=20,Mini14=20,QBU=10,SKS=10,VSS=10,Mk47=20)
    profiles = {}
    selected = []
    for group, ss in groups.items():
        if group.startswith('gunfire:') and group != 'gunfire:Weapons_RocketLancher':
            weapon = group.split(':')[1]; short = weapon.removeprefix('Weapons_')
            kind = 'automatic' if short in auto else 'dmr' if short in dmr else 'bolt' if short in bolt else 'shotgun' if short in shotgun else 'semi'
            interval = 1.8 if kind=='bolt' else .25 if kind=='dmr' else .8 if kind=='shotgun' else .09 if kind=='automatic' else .2
            weights = {'single':.2,'short_auto':.3,'sustained_auto':.5} if kind=='automatic' else {'single':1.}
            burst_rounds = 0
            if short in ['M16A4','Mk47']:
                kind='burst_rifle';interval=.1;weights={'single':.4,'burst':.6};burst_rounds=3 if short=='M16A4' else 2
            if short=='L6': interval=1.
            if short=='O12': interval=.25
            if short in ['Vector','MicroUzi','Glock18C']: interval=.065
            profiles[weapon] = dict(weapon_kind=kind,interval_seconds=interval,
                magazine_limit=magazines.get(short,30),pattern_weights=weights,burst_rounds=burst_rounds,
                pause_seconds=[.15,.7] if kind in ['dmr','semi'] else [.4,1.6] if kind=='bolt' else [.7,2.2],
                burst_choices=[1],evidence='Bounded synthesis cadence; not a claim to reproduce current patch weapon stats')
            selected.append(group)
        elif group.startswith('vehicle:') and {'idle','startup','shutdown','startup_fpp','shutdown_fpp'} <= {s['role'] for s in ss}:
            selected.append(group)
        elif group.startswith(('footsteps:','tyres:','collision:')):
            selected.append(group)
    cfg['gunfire']['profiles']=profiles
    spec=dict(base_config='configs/generation_v6.json',dataset_id='spatial-v6-24h-20260908',seed=2026090800,
        clip_seconds=120,split_counts=dict(train=600,validation=60,test=60),
        included_spawn_kinds=['footsteps','vehicle','gunfire'],included_gaits=['walk','run','sprint'],
        spawn_probability_multipliers=dict(footsteps=.6,vehicle=1.2,gunfire=1.2),
        split_protocol='known_assets_new_scenes',source_groups=sorted(selected),
        notes=cfg['notes'][-6:]+['Unsupported startup/shutdown Mirado and rocket launch audio omitted; local-only gun assets remain self-only.'])
    training=dict(model=dict(classes=3,output_mode='multi_accddoa',self_classes=3),seed=2026090800,
        learning_rate=.001,epochs=400,early_stopping_enabled=False,patience=75,
        save_detection_checkpoint=True,save_localization_checkpoint=True,distance_scale=100.,
        loss=dict(mode='spatial_balanced',activity_weight=1.,direction_weight=2.,distance_weight=.2,self_weight=.5),
        selection=dict(metric='validation_localization_macro_f1',angular_tolerance_deg=20,duplicate_tolerance_deg=15),
        cache_train_features=False,normalization_max_crops=1200,
        augmentation=dict(enabled=True,mirror_probability=.5,crops_per_scene=12),
        purpose='v6 combined update: known target types, scene independence, quieter finite-range footsteps, physical air loss, automatic gun patterns, task-masked angular training; paper topology and5s context retained.')
    for name, value in [('generation_v6.json',cfg),('dataset_v6_24h.json',spec),('pubg_spatial_v6_400epochs.json',training)]:
        Path('configs',name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(weapon_groups=len(profiles),vehicle_groups=sum(g.startswith('vehicle:') for g in selected),hours=24)))


if __name__ == '__main__':main()
