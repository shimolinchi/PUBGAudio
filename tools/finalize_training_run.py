"""Complete local evaluation and the existing W&B run without optimizing again."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--wandb-config',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=p.parse_args()
    info=read(args.run/'run.json')
    if info['status']!='training_completed':raise ValueError('Training must already be complete')
    records=[json.loads(s) for s in (args.run/'metrics.jsonl').read_text().splitlines()]
    assert len(records)==info['completed_epochs']
    assert [r['epoch'] for r in records]==list(range(len(records)))
    if not (args.run/'tracking-live/wandb_run.json').exists():
        raise ValueError('An existing project-bound W&B run is required')
    history_hash=digest(args.run/'metrics.jsonl')
    fingerprints={split:digest(args.dataset/f'manifest_{split}.jsonl') for split in ('train','validation','test')}
    reports=[];evaluated=[]
    # A fresh process uses a workspace-owned staging directory for SDK media.
    # This changes neither OS-wide settings nor any original training artifact.
    staging=(args.run/'tracking-finalize-temp').resolve();staging.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(staging)
    import torch
    from pubg_audio.evaluate import evaluate_checkpoint
    torch.set_num_threads(4)
    selections=[('test','best.pt','test.json'),('test_detection','best_detection.pt','test_detection.json')]
    if (args.run/'best_localization.pt').exists():
        selections.append(('test_localization','best_localization.pt','test_localization.json'))
    for namespace,checkpoint,filename in selections:
        path=args.run/filename
        if namespace=='test':chosen=min(records,key=lambda r:r['validation_loss'])
        elif namespace=='test_detection':chosen=max(records,key=lambda r:r['validation_external_macro_f1'])
        else:chosen=max(records,key=lambda r:r['validation_localization_macro_f1'])
        if path.exists():report=read(path)
        else:
            report=evaluate_checkpoint(args.run/checkpoint,args.dataset,'test',args.device,info['batch_size'],0,.5)
            evaluated.append(filename)
        assert report['checkpoint_sha256']==digest(args.run/checkpoint)
        assert report['checkpoint_epoch']==chosen['epoch']
        assert report['dataset_fingerprint']==fingerprints
        assert report['diagnostics']['threshold']==.5 and report['split']=='test'
        if namespace=='test_detection':
            report['checkpoint_selection']='maximum validation external macro F1; test not used for selection'
        elif namespace=='test_localization':
            report['checkpoint_selection']='maximum validation localization-aware macro F1; test not used for selection'
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        reports.append((namespace,report))
    from pubg_audio.tracking import start_run,log_evaluation
    tracker=start_run(args.wandb_config,args.run/'tracking-live',args.run.name,{},mode='online')
    with tracker:
        tracker.summary.update(dict(completed_epochs=info['completed_epochs'],best_validation_loss=info['best_validation_loss'],
            best_detection_macro_f1=info['best_detection_macro_f1'],best_detection_epoch=info['best_detection_epoch'],
            real_game_accuracy_not_evaluated=True,final_evaluation_completed=True))
        if info.get('best_localization_epoch') is not None:
            tracker.summary.update(dict(best_localization_macro_f1=info['best_localization_macro_f1'],
                best_localization_epoch=info['best_localization_epoch']))
        for namespace,report in reports:log_evaluation(tracker,report,namespace)
    assert digest(args.run/'metrics.jsonl')==history_hash,'Finalization must not change training history'
    receipt=dict(status='final_evaluation_and_tracking_completed',completed_epochs=info['completed_epochs'],
        optimizer_updates_added=0,training_metrics_sha256=history_hash,evaluated_now=evaluated,
        existing_reports_reused=[f for _,_,f in selections if f not in evaluated],
        checkpoint_epochs={name:report['checkpoint_epoch']+1 for name,report in reports},
        wandb_url=read(args.run/'tracking-live/wandb_run.json')['url'])
    (args.run/'finalization.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
