"""Select verified native crash and weather media; never relabel generic wind as blizzard."""
import re
import prepare_synthetic_sources as base


def choose_sources(banks):
    out=[]
    for bank in banks:
        for media in bank.get('IncludedMemoryFiles',[]):
            name=media['ShortName']; leaf=name.rsplit('\\',1)[-1]
            role=group=None
            if re.fullmatch(r'Vehicle_Common_Crash_\d+\.wav',leaf):
                role,group='collision','collision:common'
            elif re.fullmatch(r'Vehicle_Impact_BRDM_Light_Dry_\d+\.wav',leaf):
                role,group='collision','collision:brdm_light_dry'
            elif re.fullmatch(r'Vehicle_SnowMobile_Crash_\d+\.wav',leaf):
                role,group='collision','collision:snowmobile'
            elif re.fullmatch(r'Ambience3D_SandStorm_Loop_(Inner|Outer|Center)_01\.wav',leaf):
                role,group='weather','weather:sandstorm'
            elif leaf=='Ambience_2D_Wind_PillarCompound.wav':
                role,group='weather','weather:wind_pillar'
            elif leaf=='Mode_SLB_Ambience2D_Dynamic_Storm.wav':
                role,group='weather','weather:mode_storm'
            if role:
                out.append(dict(id=f"src_{bank['Id']}_{media['Id']}",class_index=1 if role=='collision' else 4,
                    source_group_id=group,mode='event' if role=='collision' else 'ambience',role=role,
                    bank=bank['ShortName'],bank_id=int(bank['Id']),media_id=int(media['Id']),source_name=name,
                    source_semantics='native_named_media; weather_is_nondirectional_background'))
    return out


if __name__=='__main__':
    base.choose_sources=choose_sources
    base.main()
