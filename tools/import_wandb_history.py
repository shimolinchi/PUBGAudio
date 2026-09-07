"""Import the recorded training and sanity-check histories into the dedicated project."""
import argparse
import json
from pathlib import Path

from pubg_audio.tracking import start_run,log_epoch,log_evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,default=Path('configs/wandb.local.json'))
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--overfit-report',type=Path)
    p.add_argument('--mode',choices=['online','offline'],default='online')
    p.add_argument('--tracking-output',type=Path,help='Default: run-dir/tracking-history')
    args=p.parse_args()
    output=args.tracking_output or args.run_dir/'tracking-history'
    marker=output/'import_complete.json'
    if marker.exists():
        print(marker.read_text(encoding='utf-8')); return
    records=[json.loads(l) for l in (args.run_dir/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    info=json.loads((args.run_dir/'run.json').read_text(encoding='utf-8'))
    test=json.loads((args.run_dir/'test.json').read_text(encoding='utf-8'))
    config={k:v for k,v in info.items() if k!='dataset'}
    config.update(dataset=Path(info['dataset']).name,imported_from_local_history=True,
        augmentation='existing synthetic scenes only; no added online augmentation',
        diagnostics='fixed-threshold frame classification, not official SELD scores',
        dataset_fingerprint=test['dataset_fingerprint'])
    run=start_run(args.binding,output,'small-72scenes-5epochs',config,'history-import',args.mode)
    try:
        completed_steps=run.step
        for index,record in enumerate(records):
            if index>=completed_steps:
                log_epoch(run,record)
        log_evaluation(run,test)
        run.summary['best_validation_loss']=info['best_validation_loss']
        run.summary['best_epoch']=test['checkpoint_epoch']+1
        run.notes='72 synthetic scenes; five full epochs. External classes have no detections at threshold 0.5. Not ready for game inference. Uploaded from preserved local logs.'
        training_url=run.url if args.mode=='online' else None
    except Exception:
        run.finish(exit_code=1); raise
    run.finish()
    sanity_url=None
    if args.overfit_report:
        report=json.loads(args.overfit_report.read_text(encoding='utf-8'))
        sanity=start_run(args.binding,output/'sanity','overfit-2-training-crops',
            dict(scope=report['scope'],seed=report['seed'],steps=report['steps'],
                 training_crops=report['training_crops']), 'sanity-check',args.mode)
        try:
            sanity.define_metric('optimizer_step')
            sanity.define_metric('sanity/*',step_metric='optimizer_step')
            completed_steps=sanity.step
            for index,record in enumerate(report['records']):
                if index>=completed_steps:
                    sanity.log({'optimizer_step':record['step'],'sanity/loss':record['loss'],
                                'sanity/seen_training_f1':record['external_f1']})
            sanity.summary['status']=report['status']
            sanity.notes='Memorization check only. These are seen training crops; F1 here is not test accuracy. No weights retained.'
            sanity_url=sanity.url if args.mode=='online' else None
        except Exception:
            sanity.finish(exit_code=1); raise
        sanity.finish()
    result=dict(status='uploaded' if args.mode=='online' else 'offline_prepared',training_url=training_url,sanity_url=sanity_url)
    marker.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
