"""Make a self-contained source-map player for locally rendered causal scenes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def main():
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);args=a.parse_args();root=args.root
    scenes=[json.loads(l) for l in (root/'recipes.jsonl').read_text(encoding='utf-8').splitlines()]
    probabilities=json.loads((root/'config.json').read_text(encoding='utf-8'))
    counts=Counter();eligible=Counter();skipped=Counter();density=Counter();rejected=Counter();retained_tracks=Counter()
    for s in scenes:
        if 'listener_view' in s:
            view=s['listener_view'];ids=np.arange(0,len(view['time']),5)
            s['listener_yaw']=[view['yaw_degrees_unwrapped'][j] for j in ids]
            s['listener_mode']=view['mode']
            s.pop('listener_view')
        if s['selection']=='independent_random':
            density[s['density']]+=1
            retained_tracks.update(t['kind'] for t in s['tracks'])
            for d in s['decisions']:
                if d['decision'] in ['cabin_foot_admission','cabin_far_car_admission'] and not d['selected']:
                    rejected[d['decision']]+=1
                if d['decision'].startswith('spawn:'):
                    key=d['decision'].split(':')[1]
                    if d.get('eligible') is False: skipped[key]+=1
                    else:
                        eligible[key]+=1;counts[key]+=bool(d['selected'])
        s.pop('decisions')
        with np.load(root/'audio'/(s['id']+'.npz'),allow_pickle=False) as labels:
            for i,tr in enumerate(s['tracks']):
                p=tr.pop('path'); idx=np.arange(0,len(p['time']),5)
                tr['points']=[[p['time'][j],*p['xy'][j],p['speed'][j],p['engine_on'][j]] for j in idx]
                tr['observable']=labels['track_observable'][i,::5].tolist()
                tr['active']=labels['track_activity'][i,::5].tolist()
                if 'track_footstep_cutoff_m_estimate' in labels:
                    for key,field in [('cutoff','track_footstep_cutoff_m_estimate'),('distance_gain','track_footstep_distance_gain_estimate'),('outside','track_outside_footstep_radius')]:
                        tr[key]=[float(v) if np.isfinite(v) else None for v in labels[field][i,::5]]
    data=dict(dataset_id=root.name,scenes=scenes,probabilities=probabilities,
              observed=dict(density=density,spawn_successes=counts,eligible=eligible,skipped=skipped,rejected=rejected,retained_tracks=retained_tracks))
    template=Path(__file__).with_name('causal_preview.html').read_text(encoding='utf-8')
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    (root/'试听与轨迹.html').write_text(template.replace('__DATA__',encoded),encoding='utf-8')
    print(json.dumps(dict(clips=len(scenes),file=str(root/'试听与轨迹.html'),observed=data['observed']),ensure_ascii=False))


if __name__=='__main__':main()
