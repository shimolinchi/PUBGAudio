"""Training-only stereo reflection and reproducible, label-aligned time crops."""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import dataset as dataset_module
from .dataset import aligned_targets, read_crop
from .features import extract

TARGET_KEYS=('target','counts','mask','self_target','self_mask','localization_mask')


def mirror_item(item):
    """Reflect raw (not normalized) features and every external target track."""
    result=dict(item)
    result['features']=item['features'].clone()
    result['features'][1].neg_()  # ILD: log(L/R) becomes log(R/L).
    result['features'][3].neg_()  # sin(IPD); cosine and mean magnitude stay fixed.
    result['target']=item['target'].clone()
    result['target'][...,0].neg_()  # x right; preserve y, z and distance.
    return result


def full_target_plan(labels, seconds, classes=3, distance_scale=100.):
    result=aligned_targets(labels,0.,seconds,classes,distance_scale)
    support_end=np.asarray(labels['frame_right_edge_seconds'],dtype=np.float64)
    window=float(labels.get('power_window_seconds',.032))
    times=support_end-window/2-64/44100
    starts=np.full(len(result['target']),np.inf)
    ends=np.full_like(starts,-np.inf)
    for i in range(len(starts)):
        idx=(times>=i*.1)&(times<(i+1)*.1)
        if idx.any():
            starts[i]=(support_end[idx]-window).min()
            ends[i]=support_end[idx].max()
    result['support_start']=torch.from_numpy(starts)
    result['support_end']=torch.from_numpy(ends)
    return result


def crop_targets(plan, start_tick, frames=50):
    """Slice aligned targets and reject any analysis window beyond this crop."""
    start=start_tick/10.;end=start+frames/10.
    selection=slice(start_tick,start_tick+frames)
    result={k:plan[k][selection].clone() for k in TARGET_KEYS if k in plan}
    if len(result['target'])!=frames:raise ValueError('Crop exceeds target plan')
    valid=(plan['support_start'][selection]>=start-1e-9)&(plan['support_end'][selection]<=end+1e-9)
    result['mask'][~valid]=0
    result['self_mask'][~valid]=0
    if 'localization_mask' in result:result['localization_mask'][~valid]=0
    return result


def prepare_target_plans(dataset, directory):
    directory=Path(directory)
    if any(r['split']!='train' for r in dataset.rows):
        raise ValueError('Random crop augmentation is training-only')
    identity=dict(schema=1,rows=dataset.rows,classes=dataset.classes,distance_scale=dataset.distance_scale,
        label_code=hashlib.sha256(Path(dataset_module.__file__).read_bytes()).hexdigest(),
        augmentation_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    directory.mkdir(parents=True,exist_ok=True)
    meta=directory/'identity.json'
    if meta.exists():
        if json.loads(meta.read_text())!=identity:raise ValueError('Stale augmented target cache')
    else:
        if list(directory.glob('*.pt')):raise ValueError('Unidentified target cache')
        meta.write_text(json.dumps(identity,sort_keys=True)+'\n',encoding='utf-8')
    for i,row in enumerate(dataset.rows):
        source=dataset.root/row['label_file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['label_sha256']:
            raise ValueError('Training labels changed')
        path=directory/(row['id']+'.pt')
        if not path.exists():
            with np.load(source,allow_pickle=False) as z:
                plan=full_target_plan({k:z[k] for k in z.files},row['duration_seconds'],dataset.classes,dataset.distance_scale)
            temporary=path.with_suffix('.tmp')
            torch.save(plan,temporary);temporary.replace(path)
        if (i+1)%16==0 or i+1==len(dataset.rows):
            print(json.dumps(dict(status='preparing_augmented_targets',scenes=i+1,total=len(dataset.rows))),flush=True)
    return directory


class EpochCropSampler(Sampler):
    """Carry epoch in each work item; worker count/prefetch cannot change crops."""
    def __init__(self,length,seed):
        self.length,self.seed,self.epoch=length,seed,0

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __len__(self):return self.length

    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,917]))
        return iter((self.epoch,int(i)) for i in rng.permutation(self.length))


class RandomCropScenes(Dataset):
    def __init__(self,dataset,directory,seed,mirror_probability=.5,crops_per_scene=12):
        if any(row['split']!='train' for row in dataset.rows):
            raise ValueError('Random crop augmentation is training-only')
        if not 0<=mirror_probability<=1 or not isinstance(crops_per_scene,int) or crops_per_scene<1:
            raise ValueError('Invalid augmentation settings')
        if any(row['duration_seconds']<5 for row in dataset.rows):raise ValueError('Scene shorter than five seconds')
        self.root,self.rows,self.seconds=dataset.root,dataset.rows,5.
        self.classes,self.distance_scale=dataset.classes,dataset.distance_scale
        self.seed,self.mirror_probability=seed,float(mirror_probability)
        self.items=[(i,n) for i in range(len(self.rows)) for n in range(crops_per_scene)]
        self.directory=Path(directory)
        self.normalizer=None
        self._plans=OrderedDict()

    def __len__(self):return len(self.items)

    def selection(self,epoch,index):
        scene,_=self.items[index]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(epoch),int(index),314]))
        maximum=math.floor((self.rows[scene]['duration_seconds']-5.)*10+1e-7)
        return scene,int(rng.integers(maximum+1)),bool(rng.random()<self.mirror_probability)

    def __getitem__(self,key):
        if not isinstance(key,tuple) or len(key)!=2:
            raise ValueError('Use EpochCropSampler to make training augmentations reproducible')
        epoch,index=key
        scene,tick,mirrored=self.selection(epoch,index)
        row=self.rows[scene];sid=row['id']
        if sid not in self._plans:
            self._plans[sid]=torch.load(self.directory/(sid+'.pt'),map_location='cpu',weights_only=True)
        self._plans.move_to_end(sid)
        plan=self._plans[sid]
        if len(self._plans)>16:self._plans.popitem(last=False)
        item=crop_targets(plan,tick)
        start=tick/10.
        audio,rate=read_crop(self.root/row['audio_file'],start,5.)
        item.update(features=extract(audio,rate),scene_id=sid,start_seconds=start,
                    augmentation_mirrored=mirrored,augmentation_epoch=int(epoch))
        if mirrored:item=mirror_item(item)
        if self.normalizer is not None:item['features']=self.normalizer(item['features'])
        return item
