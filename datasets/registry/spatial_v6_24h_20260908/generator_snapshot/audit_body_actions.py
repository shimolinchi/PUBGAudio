"""Audit native HIRC source identity and sample actual locomotion plans."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from causal_scene import Planner, config
from prepare_body_sources import TRANSITION_MEDIA_IDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank-xml',type=Path,required=True)
    p.add_argument('--sources',type=Path,default=Path('datasets/sources-v5_3-final'))
    p.add_argument('--output',type=Path,default=Path('outputs/reports'))
    p.add_argument('--seeds',type=int,default=200);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    r=ET.parse(a.bank_xml).getroot()
    objects={int(f.get('value')):o for o in r.iter('object') for f in o.findall('field') if f.get('name')=='ulID'}
    def trace(i,seen=None):
        seen=set() if seen is None else seen
        if i in seen:return []
        seen.add(i);o=objects[i];fields=[(f.get('name'),f.get('value')) for f in o.iter('field')]
        rows=[dict(object_id=i,type=o.get('name'),source_ids=[int(v) for k,v in fields if k=='sourceID'])]
        for k,v in fields:
            if k in ['ulActionID','idExt','ulChildID']:rows+=trace(int(v),seen)
        return rows
    traces={str(i):trace(i) for i in [1103459313,4269572827,4158240715]}
    assert len(traces['4269572827'])==1
    sources=json.loads((a.sources/'sources.json').read_text(encoding='utf-8'))['sources']
    media={mid for node in traces['1103459313'] for mid in node['source_ids']}
    assert media==TRANSITION_MEDIA_IDS=={s['media_id'] for s in sources if s['role']=='transition_rustle'}
    new=[s for s in sources if s['role'].startswith('vault_') or s['role']=='transition_rustle']
    audit=dict(bank_id=2143169776,bank_file_sha256=hashlib.sha256(a.bank_xml.with_suffix('').read_bytes()).hexdigest(),
        parser='bnnm/wwiser v20260808 read-only XML dump',parser_source='https://github.com/bnnm/wwiser',traces=traces,
        finding='Transitions_Ruslte and Vaulting_Rustle share 9 native media; Prone_Ruslte has zero actions in this bank. No unique crouch/prone audio identity established. Exact pose tags are simulated.',
        shared_attenuation_id=515586057,attenuation_max_native_distance=3000,
        metres_conversion='centimetres and runtime scale1 assumed, not calibrated',
        new_sources=len(new),roles=dict(Counter(s['role'] for s in new)))
    (a.output/'body_source_audit_20260907.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    result=dict(seeds=a.seeds,scope='planning only; locomotion duration fractions exclude pauses/body actions, not game statistics')
    for version in ['v5_2','v5_3']:
        cfg=config(f'configs/generation_{version}.json');times=Counter();actions=Counter();decisions=Counter()
        for seed in range(a.seeds):
            planner=Planner(cfg,sources,seed);tr=planner.footsteps(1)
            for state in tr['states']:
                if state['state'] in cfg['footsteps']['initial_weights']:times[state['state']]+=state['end']-state['start']
            actions.update(t['subtype'] for t in planner.tracks if t.get('actor_parent'))
            decisions.update(d['selected'] for d in planner.decisions if d['decision']=='body_action_after_bout')
        result[version]=dict(locomotion_seconds=dict(times),duration_fraction={k:v/sum(times.values()) for k,v in times.items()},
                             body_action_counts=dict(actions),after_bout_selections=dict(decisions))
    (a.output/'body_action_probability_audit_20260907.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({v:result[v]['duration_fraction'] for v in ['v5_2','v5_3']}))


if __name__=='__main__':main()
