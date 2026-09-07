"""Check learnability on two training crops; never use this as a generalization score."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from pubg_audio.dataset import SyntheticScenes
from pubg_audio.evaluate import FrameDiagnostics
from pubg_audio.features import FeatureNormalizer
from pubg_audio.losses import training_loss
from pubg_audio.model import ModelConfig, PaperSELD


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path,required=True,help='Read architecture and train normalization only, not model weights')
    p.add_argument('--scene-id',action='append',required=True,help='Training scene only; choose one balanced crop per scene')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.steps<1:
        p.error('Steps must be positive')
    torch.set_num_threads(4); torch.manual_seed(20260907)
    reference=torch.load(args.reference_checkpoint,map_location='cpu',weights_only=True)
    cfg=ModelConfig(**reference['model_config'])
    dataset=SyntheticScenes(args.dataset,'train',classes=cfg.classes,
        distance_scale=reference['config'].get('distance_scale',100.),normalizer=FeatureNormalizer(**reference['normalizer']))
    chosen=[]
    for scene_id in args.scene_id:
        candidates=[]
        for index,(scene,_) in enumerate(dataset.items):
            if dataset.rows[scene]['id']==scene_id:
                item=dataset[index]
                support=((item['counts']>0)&(item['mask']>0)).sum(0)
                candidates.append((int(support.min()),int(support.sum()),index,item))
        if not candidates:
            raise ValueError('Requested scene is absent from the training split')
        chosen.append(max(candidates,key=lambda x:x[:2])[-1])
    batch={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in default_collate(chosen).items()}
    model=PaperSELD(cfg).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    records=[]; started=time.monotonic()
    for step in range(args.steps+1):
        if step%50==0 or step==args.steps:
            model.eval()
            with torch.no_grad():
                output=model(batch['features']); loss,_=training_loss(output,batch)
                stats=FrameDiagnostics(distance_scale=dataset.distance_scale); stats.update(output,batch)
            diagnostic=stats.report()
            record=dict(step=step,loss=float(loss),external_f1=diagnostic['external']['micro']['f1'])
            records.append(record); print(json.dumps(record),flush=True)
        if step==args.steps:
            break
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss,_=training_loss(model(batch['features']),batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite overfit loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); optimizer.step()
    passed=records[-1]['loss']<records[0]['loss']*.2 and (records[-1]['external_f1'] or 0)>=.95
    report=dict(status='passed' if passed else 'needs_inspection',training_crops=[dict(scene_id=i['scene_id'],start_seconds=i['start_seconds']) for i in chosen],
        reference_checkpoint_usage='architecture and training-only normalization; model initialized from scratch',
        steps=args.steps,seed=20260907,records=records,diagnostics=diagnostic,elapsed_seconds=round(time.monotonic()-started,3),
        scope='memorization sanity check on seen training crops, not validation or test accuracy',weights_saved=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
