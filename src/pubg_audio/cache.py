"""Optional immutable feature/target cache, keyed by data and adapter code."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import dataset as dataset_module, features as feature_module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_identity(dataset):
    return dict(schema_version=1, scenes=dataset.rows, seconds=dataset.seconds,
                classes=dataset.classes, distance_scale=dataset.distance_scale,
                dataset_code=sha(dataset_module.__file__), feature_code=sha(feature_module.__file__),
                torch_version=str(torch.__version__), crops=len(dataset))


def prepare_cache(dataset, directory):
    if dataset.normalizer is not None:
        raise ValueError('Cache raw features; fit normalization on training data only')
    directory=Path(directory)
    identity=cache_identity(dataset)
    directory.mkdir(parents=True,exist_ok=True)
    meta=directory/'identity.json'
    if meta.exists():
        if json.loads(meta.read_text(encoding='utf-8')) != identity:
            raise ValueError('Stale feature cache: data or adapter changed; select a fresh directory')
    else:
        if list(directory.glob('*.pt')):
            raise ValueError('Unidentified feature cache')
        meta.write_text(json.dumps(identity,separators=(',',':'))+'\n',encoding='utf-8')
    for row in dataset.rows:
        for path_key,hash_key in [('audio_file','audio_sha256'),('label_file','label_sha256')]:
            if hash_key in row and sha(dataset.root/row[path_key]) != row[hash_key]:
                raise ValueError('Dataset file changed: '+row[path_key])
    for index in range(len(dataset)):
        path=directory/f'{index:06d}.pt'
        if path.exists(): continue
        item=dataset[index]
        temporary=path.with_suffix('.tmp')
        torch.save(item,temporary)
        temporary.replace(path)
        if (index+1)%96==0 or index==len(dataset)-1:
            print(json.dumps(dict(status='caching',cache=directory.name,done=index+1,total=len(dataset))),flush=True)
    return CachedScenes(dataset,directory)


class CachedScenes(Dataset):
    def __init__(self,dataset,directory):
        self.rows,self.items=dataset.rows,dataset.items
        self.normalizer=dataset.normalizer
        self.directory=Path(directory)

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        item=torch.load(self.directory/f'{index:06d}.pt',map_location='cpu',weights_only=True)
        scene,start=self.items[index]
        if item['scene_id']!=self.rows[scene]['id'] or item['start_seconds']!=start:
            raise ValueError('Cached crop identity mismatch')
        if self.normalizer is not None:
            item['features']=self.normalizer(item['features'])
        return item
