"""Select coarse training targets; retain the original detailed labels and audio."""
import argparse
import json
from pathlib import Path
import numpy as np

BASE_CLASSES = ('footsteps','vehicle','gunfire')


def project(labels, selected=BASE_CLASSES, self_selected=None):
    self_selected = selected if self_selected is None else self_selected
    names=list(labels['classes'])
    if any(s not in BASE_CLASSES for s in [*selected,*self_selected]):
        raise ValueError('Select subsets of the existing footsteps/vehicle/gunfire slots')
    indices=[names.index(s) for s in BASE_CLASSES]
    lookup={old:new for new,old in enumerate(indices) if BASE_CLASSES[new] in selected}
    supervision=np.array([s in selected for s in BASE_CLASSES],np.uint8)
    self_supervision=np.array([s in self_selected for s in BASE_CLASSES],np.uint8)
    z={k:labels[k] for k in ['frame_right_edge_seconds','track_xy','track_distance_units','track_activity']}
    if 'track_localization_observable' in labels:
        z['track_localization_observable'] = labels['track_localization_observable']
    z.update(activity_loss_mask=labels['activity_loss_mask'][:,indices]*supervision[None,:,None],
             self_activity=labels['self_activity'][:,indices],
             self_activity_loss_mask=labels['self_activity_loss_mask'][:,indices]*self_supervision[None,:],
             class_supervision_mask=supervision,
             track_class_index=np.array([lookup.get(int(c),255) for c in labels['track_class_index']],np.uint8),
             track_source_role=np.array([1 if r=='self' else 0 for r in labels['track_source_role']],np.uint8))
    return z


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('input',type=Path);a.add_argument('output',type=Path)
    a.add_argument('--config',type=Path,default=Path('configs/training_targets_v5.json'))
    args=a.parse_args()
    if args.input.resolve()==args.output.resolve():
        raise ValueError('Preserve the detailed input; choose a separate projection output')
    c=json.loads(args.config.read_text(encoding='utf-8'))
    with np.load(args.input,allow_pickle=False) as labels:
        np.savez_compressed(args.output,**project(labels,c['external_targets'],c['self_targets']))


if __name__=='__main__':main()
