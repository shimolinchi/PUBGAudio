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
from .tracking import start_run, log_epoch, log_evaluation
from .evaluate import FrameDiagnostics, evaluate_checkpoint
from .cache import prepare_cache
from .augmentation import RandomCropScenes, EpochCropSampler, prepare_target_plans


def external_macro_f1(report):
    rows=report['external']['per_class'].values()
    values=[r['f1'] for r in rows if r['positive_frames']>0 and r['f1'] is not None]
    return float(np.mean(values)) if values else None


def run_epoch(model, loader, device, optimizer=None, limit=None, diagnostics=None, loss_config=None):
    model.train(optimizer is not None)
    losses=[]; samples=[]
    for step,batch in enumerate(loader):
        if limit is not None and step >= limit:
            break
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            outputs=model(batch['features'])
            loss,_=training_loss(outputs,batch,loss_config=loss_config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
        losses.append(float(loss.detach()))
        samples.append(len(batch['features']))
        if diagnostics is not None:
            diagnostics.update(outputs,batch)
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
    p.add_argument('--cache-dir',type=Path,help='Cache unnormalized features and targets; identity checked before reuse')
    p.add_argument('--evaluate-test',action='store_true',help='Evaluate best validation checkpoint once after training and log to the same run')
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
    raw_train=train
    validation=SyntheticScenes(args.dataset,'validation',**kwargs)
    if set(r['id'] for r in train.rows)&set(r['id'] for r in validation.rows):
        raise ValueError('Scene leakage')
    if args.cache_dir:
        if cfg.get('cache_train_features',True):
            train=prepare_cache(train,args.cache_dir/'train')
        validation=prepare_cache(validation,args.cache_dir/'validation')
    print(json.dumps(dict(status='initializing',train_scenes=len(train.rows),validation_scenes=len(validation.rows),
        train_crops=len(train),validation_crops=len(validation),device=args.device)),flush=True)
    model=PaperSELD(model_cfg).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.get('learning_rate',.001))
    start,best,stale=0,float('inf'),0
    best_detection,best_detection_epoch=-1.,None
    best_localization,best_localization_epoch=-1.,None
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
        best_detection=saved.get('best_detection_macro_f1',-1.)
        best_detection_epoch=saved.get('best_detection_epoch')
        best_localization=saved.get('best_localization_macro_f1',-1.)
        best_localization_epoch=saved.get('best_localization_epoch')
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    else:
        print('Fitting normalization on training data only',flush=True)
        normalizer_count=min(6,len(train)) if args.smoke else min(cfg.get('normalization_max_crops',len(train)),len(train))
        normalizer_indices=np.linspace(0,len(train)-1,normalizer_count,dtype=int)
        normalizer=fit_normalizer(train[int(i)]['features'] for i in normalizer_indices)
    augmentation=cfg.get('augmentation',{})
    sampler=None
    if augmentation.get('enabled',False):
        allowed={'enabled','mirror_probability','crops_per_scene'}
        if set(augmentation)-allowed:raise ValueError('Unknown augmentation setting')
        directory=(args.cache_dir if args.cache_dir else args.output/'cache')/'random_targets'
        prepare_target_plans(raw_train,directory)
        train=RandomCropScenes(raw_train,directory,seed,
            mirror_probability=augmentation.get('mirror_probability',.5),
            crops_per_scene=augmentation.get('crops_per_scene',12))
        sampler=EpochCropSampler(len(train),seed)
        print(json.dumps(dict(status='training_augmentation_ready',train_scenes=len(train.rows),
            sampled_crops_per_epoch=len(train),base_fixed_crops=len(raw_train),crop_grid_seconds=.1,
            mirror_probability=train.mirror_probability)),flush=True)
    train.normalizer=validation.normalizer=normalizer
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=args.workers,
                      persistent_workers=args.workers>0)
    val_loader=DataLoader(validation,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                          persistent_workers=args.workers>0)
    args.output.mkdir(parents=True,exist_ok=True)
    epochs=args.epochs or (1 if args.smoke else cfg.get('epochs',250))
    tracker=start_run(args.wandb_config,args.output/'tracking-live',args.output.name,
        dict(config=cfg,dataset=args.dataset.name,dataset_fingerprint=dataset_fingerprint,
             batch_size=args.batch_size,device=args.device,smoke_only=args.smoke),mode=args.wandb_mode) if args.wandb_config else None
    with tracker if tracker is not None else nullcontext():
        for epoch in range(start,start+epochs):
            started=time.monotonic()
            if sampler is not None:sampler.set_epoch(epoch)
            train_loss,steps=run_epoch(model,loader,args.device,optimizer,2 if args.smoke else None,loss_config=cfg.get('loss'))
            selection=cfg.get('selection',{})
            diagnostics=FrameDiagnostics(.5,cfg.get('distance_scale',100.),bool(selection),
                selection.get('angular_tolerance_deg',20),selection.get('duplicate_tolerance_deg',15))
            val_loss,val_steps=run_epoch(model,val_loader,args.device,None,1 if args.smoke else None,diagnostics,cfg.get('loss'))
            improved=val_loss<best
            best=min(best,val_loss); stale=0 if improved else stale+1
            detection=external_macro_f1(diagnostics.report())
            detection_improved=detection is not None and detection>best_detection
            if detection_improved:
                best_detection,best_detection_epoch=detection,epoch
            localization=diagnostics.report().get('localization_macro_f1')
            localization_improved=localization is not None and localization>best_localization
            if localization_improved:best_localization,best_localization_epoch=localization,epoch
            record=dict(epoch=epoch,train_loss=train_loss,validation_loss=val_loss,train_batches=steps,
                        validation_batches=val_steps,smoke_only=args.smoke,elapsed_seconds=round(time.monotonic()-started,3),
                        validation_diagnostics=diagnostics.report(),validation_external_macro_f1=detection,
                        validation_localization_macro_f1=localization,
                        learning_rate=optimizer.param_groups[0]['lr'])
            with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),normalizer=normalizer.state_dict(),
                epoch=epoch,best_validation_loss=best,stale_epochs=stale,model_config=model_cfg.to_dict(),
                config=cfg,dataset=str(args.dataset.resolve()),dataset_fingerprint=dataset_fingerprint,
                best_detection_macro_f1=best_detection,best_detection_epoch=best_detection_epoch,
                best_localization_macro_f1=best_localization,best_localization_epoch=best_localization_epoch,
                validation_loss_at_checkpoint=val_loss,validation_external_macro_f1_at_checkpoint=detection,
                validation_localization_macro_f1_at_checkpoint=localization,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None)
            temporary=args.output/'last.tmp.pt'
            torch.save(state,temporary); temporary.replace(args.output/'last.pt')
            if improved:
                torch.save(dict(state,selection_metric='validation_loss'),args.output/'best.pt')
            if cfg.get('save_detection_checkpoint',False) and detection_improved:
                torch.save(dict(state,selection_metric='validation_external_macro_f1'),args.output/'best_detection.pt')
            if cfg.get('save_localization_checkpoint',False) and localization_improved:
                torch.save(dict(state,selection_metric='validation_localization_macro_f1'),args.output/'best_localization.pt')
            console={k:v for k,v in record.items() if k!='validation_diagnostics'}
            console['validation_f1']={role:{cls:row['f1'] for cls,row in diagnostics.report()[role]['per_class'].items()}
                                      for role in ['external','self']}
            print(json.dumps(console),flush=True)
            if tracker is not None:
                log_epoch(tracker,record)
            if cfg.get('early_stopping_enabled',True) and stale>=cfg.get('patience',75):
                break
        info=dict(status='smoke_passed' if args.smoke else 'training_completed',
            model_config=model_cfg.to_dict(),parameters=sum(p.numel() for p in model.parameters()),
            dataset=str(args.dataset.resolve()),device=args.device,torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',
            completed_epochs=epoch+1,best_validation_loss=best,batch_size=args.batch_size,
            cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
            workers=args.workers,threads=args.threads,seed=seed,
            early_stopping_enabled=cfg.get('early_stopping_enabled',True),
            best_detection_macro_f1=best_detection,best_detection_epoch=best_detection_epoch+1 if best_detection_epoch is not None else None,
            best_localization_macro_f1=best_localization,best_localization_epoch=best_localization_epoch+1 if best_localization_epoch is not None else None,
            train_scenes=len(train.rows),validation_scenes=len(validation.rows),
            train_crops=len(train),validation_crops=len(validation),
            train_base_fixed_crops=len(raw_train),augmentation=augmentation,
            training_sample_mode='random_5s_windows_on_100ms_grid' if sampler is not None else 'fixed_5s_windows',
            real_game_accuracy_not_evaluated=True)
        (args.output/'run.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        if tracker is not None:
            tracker.summary.update(dict(best_validation_loss=best,completed_epochs=epoch+1,real_game_accuracy_not_evaluated=True,
                best_detection_macro_f1=best_detection,best_detection_epoch=best_detection_epoch+1 if best_detection_epoch is not None else None,
                best_localization_macro_f1=best_localization,
                best_localization_epoch=best_localization_epoch+1 if best_localization_epoch is not None else None))
        if args.evaluate_test and not args.smoke:
            report=evaluate_checkpoint(args.output/'best.pt',args.dataset,'test',args.device,args.batch_size,args.workers,.5)
            (args.output/'test.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            print(json.dumps(dict(status='held_out_test_completed',checkpoint_epoch=report['checkpoint_epoch']+1,
                f1={role:{cls:row['f1'] for cls,row in report['diagnostics'][role]['per_class'].items()}
                    for role in ['external','self']})),flush=True)
            if tracker is not None:log_evaluation(tracker,report)
            if cfg.get('save_detection_checkpoint',False):
                report=evaluate_checkpoint(args.output/'best_detection.pt',args.dataset,'test',args.device,args.batch_size,args.workers,.5)
                report['checkpoint_selection']='maximum validation external macro F1; test not used for selection'
                (args.output/'test_detection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
                print(json.dumps(dict(status='detection_checkpoint_test_completed',checkpoint_epoch=report['checkpoint_epoch']+1,
                    f1={role:{cls:row['f1'] for cls,row in report['diagnostics'][role]['per_class'].items()}
                        for role in ['external','self']})),flush=True)
                if tracker is not None:log_evaluation(tracker,report,namespace='test_detection')
            if cfg.get('save_localization_checkpoint',False):
                report=evaluate_checkpoint(args.output/'best_localization.pt',args.dataset,'test',args.device,args.batch_size,args.workers,.5)
                report['checkpoint_selection']='maximum validation localization-aware macro F1; test not used for selection'
                (args.output/'test_localization.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
                if tracker is not None:log_evaluation(tracker,report,namespace='test_localization')


if __name__=='__main__':
    main()
