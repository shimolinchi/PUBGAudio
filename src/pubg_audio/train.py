"""Train/validate, checkpoint and resume the paper architecture on local scenes."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SyntheticScenes
from .features import FeatureNormalizer, fit_normalizer
from .losses import training_loss
from .model import ModelConfig, PaperSELD
from .tracking import start_run, log_epoch


def run_epoch(model, loader, device, optimizer=None, limit=None):
    model.train(optimizer is not None)
    losses=[]; samples=[]
    for step,batch in enumerate(loader):
        if limit is not None and step >= limit:
            break
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            outputs=model(batch['features'])
            loss,_=training_loss(outputs,batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
        losses.append(float(loss.detach()))
        samples.append(len(batch['features']))
    if not losses:
        raise ValueError('No batches processed')
    return float(np.average(losses,weights=samples)),len(losses)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/pubg_paper_backbone.json'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true',help='Two train batches, one validation batch; no accuracy claim')
    p.add_argument('--resume',type=Path)
    p.add_argument('--wandb-config',type=Path,help='Dedicated PUBGAudio project binding; omitted means local logging only')
    p.add_argument('--wandb-mode',choices=['online','offline'],default='online')
    args=p.parse_args()
    if args.batch_size<1 or args.threads<1 or args.workers<0 or (args.epochs is not None and args.epochs<1):
        p.error('Epochs, batch size and threads must be positive; workers must be nonnegative')
    if not args.resume and (args.output/'last.pt').exists():
        raise ValueError('Existing training run: use --resume or a fresh output directory')
    torch.set_num_threads(args.threads)
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    seed=cfg.get('seed',20260907)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model_cfg=ModelConfig(**cfg['model'])
    if model_cfg.classes != 3:
        raise ValueError('Synthetic adapter supports three labelled classes; paper_13class config is for topology verification')
    kwargs=dict(classes=model_cfg.classes,distance_scale=cfg.get('distance_scale',100.),max_scenes=2 if args.smoke else None)
    dataset_fingerprint={split:hashlib.sha256((args.dataset/f'manifest_{split}.jsonl').read_bytes()).hexdigest()
                         for split in ['train','validation','test']}
    train=SyntheticScenes(args.dataset,'train',**kwargs)
    validation=SyntheticScenes(args.dataset,'validation',**kwargs)
    if set(r['id'] for r in train.rows)&set(r['id'] for r in validation.rows):
        raise ValueError('Scene leakage')
    print(json.dumps(dict(status='initializing',train_scenes=len(train.rows),validation_scenes=len(validation.rows),
        train_crops=len(train),validation_crops=len(validation),device=args.device)),flush=True)
    model=PaperSELD(model_cfg).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.get('learning_rate',.001))
    start,best,stale=0,float('inf'),0
    if args.resume:
        saved=torch.load(args.resume,map_location=args.device,weights_only=True)
        if saved['model_config'] != model_cfg.to_dict():
            raise ValueError('Checkpoint architecture mismatch')
        if saved['dataset'] != str(args.dataset.resolve()) or saved['config'] != cfg:
            raise ValueError('Resume dataset/config mismatch')
        if saved.get('dataset_fingerprint') != dataset_fingerprint:
            raise ValueError('Dataset manifests changed since checkpoint')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        normalizer=FeatureNormalizer(**saved['normalizer'])
        start,best,stale=saved['epoch']+1,saved['best_validation_loss'],saved['stale_epochs']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    else:
        print('Fitting normalization on training data only',flush=True)
        normalizer=fit_normalizer(train[i]['features'] for i in range(min(6,len(train)) if args.smoke else len(train)))
    train.normalizer=validation.normalizer=normalizer
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    val_loader=DataLoader(validation,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    args.output.mkdir(parents=True,exist_ok=True)
    epochs=args.epochs or (1 if args.smoke else cfg.get('epochs',250))
    tracker=start_run(args.wandb_config,args.output/'tracking-live',args.output.name,
        dict(config=cfg,dataset=args.dataset.name,dataset_fingerprint=dataset_fingerprint,
             batch_size=args.batch_size,device=args.device,smoke_only=args.smoke),mode=args.wandb_mode) if args.wandb_config else None
    with tracker if tracker is not None else nullcontext():
        for epoch in range(start,start+epochs):
            started=time.monotonic()
            train_loss,steps=run_epoch(model,loader,args.device,optimizer,2 if args.smoke else None)
            val_loss,val_steps=run_epoch(model,val_loader,args.device,None,1 if args.smoke else None)
            improved=val_loss<best
            best=min(best,val_loss); stale=0 if improved else stale+1
            record=dict(epoch=epoch,train_loss=train_loss,validation_loss=val_loss,train_batches=steps,
                        validation_batches=val_steps,smoke_only=args.smoke,elapsed_seconds=round(time.monotonic()-started,3))
            with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),normalizer=normalizer.state_dict(),
                epoch=epoch,best_validation_loss=best,stale_epochs=stale,model_config=model_cfg.to_dict(),
                config=cfg,dataset=str(args.dataset.resolve()),dataset_fingerprint=dataset_fingerprint,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None)
            temporary=args.output/'last.tmp.pt'
            torch.save(state,temporary); temporary.replace(args.output/'last.pt')
            if improved:
                torch.save(state,args.output/'best.pt')
            print(json.dumps(record),flush=True)
            if tracker is not None:
                log_epoch(tracker,record)
            if stale>=cfg.get('patience',75):
                break
        info=dict(status='smoke_passed' if args.smoke else 'training_completed',
            model_config=model_cfg.to_dict(),parameters=sum(p.numel() for p in model.parameters()),
            dataset=str(args.dataset.resolve()),device=args.device,torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',
            completed_epochs=epoch+1,best_validation_loss=best,batch_size=args.batch_size,
            workers=args.workers,threads=args.threads,seed=seed,
            train_scenes=len(train.rows),validation_scenes=len(validation.rows),
            train_crops=len(train),validation_crops=len(validation),real_game_accuracy_not_evaluated=True)
        (args.output/'run.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        if tracker is not None:
            tracker.summary.update(dict(best_validation_loss=best,completed_epochs=epoch+1,real_game_accuracy_not_evaluated=True))


if __name__=='__main__':
    main()
