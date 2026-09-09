"""Read-only preparation of native causally linked event media for v5 previews."""
import re
import prepare_synthetic_sources as base


def choose_sources(banks):
    out = []
    for bank in banks:
        counts = {}
        for media in sorted(bank.get('IncludedMemoryFiles', []), key=lambda m: m['ShortName']):
            name = media['ShortName']; leaf = name.rsplit('\\', 1)[-1]
            role = group = None
            cls = 4
            if bank['ShortName'] == 'HandBombBank':
                rules = [
                    (r'Handbomb_Common_PinOff_\d+.wav', 'pin', 'throwable:common'),
                    (r'Handbomb_Common_Cook_\d+.wav', 'cook', 'throwable:common'),
                    (r'Handbomb_Grenade_Explosion_Closest_TPP_\d+.wav', 'frag_explosion', 'throwable:frag'),
                    (r'Handbomb_FlashBang_Explosion_\d+.wav', 'flash_explosion', 'throwable:flash'),
                    (r'Handbmob_Firebomb_On_\d+.wav', 'molotov_ignite', 'throwable:molotov'),
                    (r'Handbmob_Firebomb_Impact_\d+.wav', 'molotov_impact', 'throwable:molotov'),
                    (r'Handbmob_Firebomb_Blaze_\d+.wav', 'molotov_fire', 'throwable:molotov'),
                    (r'Handbomb_C4_Attach_\d+.wav', 'c4_attach', 'explosive:c4'),
                    (r'Handbomb_C4_SwitchOn_Step01_\d+.wav', 'c4_switch_1', 'explosive:c4'),
                    (r'Handbomb_C4_SwitchOn_Step02_\d+.wav', 'c4_switch_2', 'explosive:c4'),
                    (r'Handbomb_C4_SwitchOn_Step03_\d+.wav', 'c4_switch_3', 'explosive:c4'),
                    (r'Handbomb_C4_Beep_\d+.wav', 'c4_beep', 'explosive:c4'),
                    (r'Handbomb_C4_Explosion_Close_\d+.wav', 'c4_explosion', 'explosive:c4'),
                ]
                for pattern, candidate_role, candidate_group in rules:
                    if re.fullmatch(pattern, leaf, re.I):
                        role, group = candidate_role, candidate_group
                        break
            elif bank['ShortName'] == 'Vehicle_Common':
                cls = 1
                for pattern, candidate_role in [
                    (r'Vehicle_Common_TireExplosion_Explosion_\d+.wav', 'tire_burst'),
                    (r'Vehicle_Delayed_Explosion_Common_Fire_TPP_\d+.wav', 'vehicle_fire'),
                    (r'Vehicle_Delayed_Explosion_Damaged_Medium_TPP_\d+.wav', 'vehicle_damage'),
                    (r'Vehicle_Delayed_Explosion_BlowUp_TPP_\d+.wav', 'vehicle_explosion'),
                ]:
                    if re.fullmatch(pattern, leaf, re.I):
                        role, group = candidate_role, 'vehicle_event:' + candidate_role
                        break
            elif bank['ShortName'] == 'DirectingBank' and leaf == 'Directing_CarePackage_Aircraft_C130_Flying_Loop_01.wav':
                cls, role, group = 3, 'aircraft_pass', 'aircraft:c130'
            elif bank['ShortName'] == 'FootStepsBank':
                match = re.search(r'FootSteps_NEW\\(Crouch_Walk|Prone_Normal)\\\1_(Concrete|Dirt|Fabric|Rock|Snow)\\', name)
                if match:
                    cls, role, group = 0, 'crouch_walk' if match[1] == 'Crouch_Walk' else 'prone_crawl', 'footsteps:' + match[2]
            if role is None:
                continue
            key = (group, role)
            if counts.get(key, 0) >= 4:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append(dict(id=f"src_{bank['Id']}_{media['Id']}", class_index=cls,
                source_group_id=group, mode='event', role=role, bank=bank['ShortName'],
                bank_id=int(bank['Id']), media_id=int(media['Id']), source_name=name,
                source_semantics='native_named_media; availability_does_not_verify_game_event_timing'))
    return out


if __name__ == '__main__':
    base.choose_sources = choose_sources
    base.main()
