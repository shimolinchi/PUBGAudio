"""Copy/decode named local character-transition and vault media, read-only."""
import re
import prepare_synthetic_sources as base

# Exact media IDs reached from CharacterBank event 1103459313, not filename aliases.
TRANSITION_MEDIA_IDS = {355685805,888384367,519224851,804096818,277443246,
                        990125138,1073122829,792282903,30403813}


def choose_sources(banks):
    out = []
    for bank in banks:
        counts = {}
        for media in sorted(bank.get('IncludedMemoryFiles', []), key=lambda m: m['ShortName']):
            name = media['ShortName']; leaf = name.rsplit('\\', 1)[-1]
            role = group = None
            if bank['ShortName'] == 'Weapons_Common' and int(media['Id']) in TRANSITION_MEDIA_IDS:
                role, group = 'transition_rustle', 'body:transition'
                evidence = 'CharacterBank HIRC event 1103459313 -> action 334636408 -> container 500554510; shared transition rustle, exact pose not identified by sound'
                limit = 9
            elif bank['ShortName'] == 'CharacterBank_Vaulting':
                for pattern, candidate in [
                    (r'LedgeGrab_Hand_Touch_(Concrete|Metal|Wood|Snow)_Normal_\d+\.wav', 'vault_grab'),
                    (r'LedgeGrab_Kick_(Concrete|Metal|Wood|Snow)_Normal_MilitaryBoots_\d+\.wav', 'vault_climb'),
                    (r'LedgeGrab_Foot_Impact_(Concrete|Metal|Wood|Snow)_Normal_MilitaryBoots_\d+\.wav', 'vault_contact'),
                ]:
                    match = re.fullmatch(pattern, leaf)
                    if match:
                        role, group = candidate, 'vault:' + match[1]
                        break
                if re.fullmatch(r'LedgeGrab_Rustle_Long_Normal_\d+\.wav', leaf):
                    role, group = 'vault_rustle', 'vault:cloth'
                evidence = 'named native media in CharacterBank_Vaulting; synthetic linked sequence, not recovered animation timing'
                limit = 3
            if role is None:
                continue
            key = (role, group)
            if counts.get(key, 0) >= limit:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append(dict(id=f"src_{bank['Id']}_{media['Id']}", class_index=4,
                source_group_id=group, mode='event', role=role, bank=bank['ShortName'],
                bank_id=int(bank['Id']), media_id=int(media['Id']), source_name=name,
                source_semantics=evidence))
    return out


if __name__ == '__main__':
    base.choose_sources = choose_sources
    base.main()
