"""Load immutable 30 s WAV/NPZ scenes and aligned five-second training crops."""
import json
from pathlib import Path
import wave

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import extract


def read_crop(path, start, seconds):
    with wave.open(str(path), 'rb') as w:
        rate = w.getframerate()
        if w.getnchannels() != 2 or w.getsampwidth() != 2:
            raise ValueError('Dataset must be stereo PCM16')
        w.setpos(round(start * rate))
        x = np.frombuffer(w.readframes(round(seconds * rate)), dtype='<i2').reshape(-1,2).astype(np.float32) / 32768
    if len(x) != round(seconds * rate):
        raise ValueError('Crop exceeds recording')
    return x, rate


def aligned_targets(labels, start, seconds, classes=3, distance_scale=100.):
    frames = round(seconds * 10)
    target = np.zeros((frames,3,classes,4), np.float32)
    counts = np.zeros((frames,classes), np.int64)
    mask = np.ones((frames,classes), np.float32)
    own_target = np.zeros((frames,3), np.float32)
    own_mask = np.ones((frames,3), np.float32)
    label_time = labels['frame_right_edge_seconds'] - .016 - 64/44100
    for frame in range(frames):
        # A non-overlapping 100 ms target bin avoids leaking labels across crops.
        # Feature windows overlap boundaries; retain short events in each bin.
        begin, end = start + frame * .1, start + (frame + 1) * .1
        idx = np.flatnonzero((label_time >= begin) & (label_time < end))
        if not len(idx):
            mask[frame] = 0; own_mask[frame] = 0
            continue
        if 'self_activity' in labels:
            own_target[frame] = labels['self_activity'][idx].max(axis=0)
            own_mask[frame] = labels['self_activity_loss_mask'][idx].min(axis=0)
        for cls in range(classes):
            if cls >= len(labels['class_supervision_mask']) or not labels['class_supervision_mask'][cls]:
                mask[frame,cls] = 0
                continue
            # Any hidden positive of this class makes its negative slots ambiguous.
            if not labels['activity_loss_mask'][idx,cls].all():
                mask[frame,cls] = 0
            values = []
            for actor, actor_cls in enumerate(labels['track_class_index']):
                if int(actor_cls) != cls or labels['track_source_role'][actor] != 0:
                    continue
                active_idx = idx[labels['track_activity'][actor,idx].astype(bool)]
                if not len(active_idx):
                    continue
                j = int(active_idx[len(active_idx)//2])
                xy = labels['track_xy'][actor,j].astype(np.float64)
                distance = float(labels['track_distance_units'][actor,j])
                # Our coordinates: x right, y front, z up. All sources are horizontal.
                direction = xy / max(np.linalg.norm(xy),1e-9)
                values.append([*direction,0.,distance/distance_scale])
            if len(values) > 3:
                mask[frame,cls] = 0  # Never silently discard a fourth source.
            counts[frame,cls] = min(len(values),3)
            for actor,value in enumerate(values[:3]):
                target[frame,actor,cls] = value
    return {k:torch.from_numpy(v) for k,v in dict(target=target,counts=counts,mask=mask,
        self_target=own_target,self_mask=own_mask).items()}


class SyntheticScenes(Dataset):
    def __init__(self, root, split, seconds=5., classes=3, distance_scale=100.,
                 normalizer=None, max_scenes=None):
        if split not in ['train','validation','test'] or seconds != 5:
            raise ValueError('Use a fixed split and five-second crops')
        self.root = Path(root).resolve()
        rows = [json.loads(l) for l in (self.root/f'manifest_{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        if any(r['split'] != split for r in rows):
            raise ValueError('Manifest split mismatch')
        self.rows = rows[:max_scenes] if max_scenes else rows
        self.seconds, self.classes, self.distance_scale = seconds, classes, distance_scale
        self.normalizer = normalizer
        self.items = [(i,float(start)) for i,r in enumerate(self.rows) for start in np.arange(0,r['duration_seconds']-seconds+1e-7,seconds)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scene, start = self.items[index]
        row = self.rows[scene]
        x, rate = read_crop(self.root/row['audio_file'],start,self.seconds)
        features = extract(x,rate)
        if self.normalizer:
            features = self.normalizer(features)
        with np.load(self.root/row['label_file'],allow_pickle=False) as z:
            labels = {k:z[k] for k in z.files}
        item = aligned_targets(labels,start,self.seconds,self.classes,self.distance_scale)
        item.update(features=features,scene_id=row['id'],start_seconds=start)
        return item
