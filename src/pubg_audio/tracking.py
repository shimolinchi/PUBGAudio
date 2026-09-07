"""Explicit, project-bound W&B logging; no inference from global project settings."""
import json
from pathlib import Path
import uuid


def load_binding(path):
    binding=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if binding.get('application')!='PUBGAudio' or not binding.get('created_for_this_module'):
        raise ValueError('Use the independent project created by tools/setup_wandb.py')
    if not all(isinstance(binding.get(k),str) and binding[k] for k in ['entity','project','project_id']):
        raise ValueError('Incomplete W&B project binding')
    return binding


def start_run(binding_path, output, name, config, job_type='training', mode='online'):
    import wandb
    binding=load_binding(binding_path)
    if mode not in ['online','offline']:
        raise ValueError('Explicit W&B mode required')
    if mode=='online':
        project=wandb.Api(timeout=20).project(binding['project'],entity=binding['entity'])
        if project.id!=binding['project_id']:
            raise ValueError('W&B project identity changed')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    receipt_path=output/'wandb_run.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt['entity'],receipt['project'],receipt['mode'])!=(binding['entity'],binding['project'],mode):
            raise ValueError('Refusing to move this run into another W&B project or mode')
    else:
        receipt=dict(entity=binding['entity'],project=binding['project'],mode=mode,id=uuid.uuid4().hex[:12])
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    run=wandb.init(entity=binding['entity'],project=binding['project'],id=receipt['id'],
        resume='allow' if mode=='online' else None,mode=mode,name=name,job_type=job_type,
        group='sanity-checks' if job_type=='sanity-check' else 'small-baseline' if job_type=='history-import' else 'training',
        dir=str(output),config=config,
        settings=wandb.Settings(disable_git=True,save_code=False,console='off',
                                x_disable_stats=job_type!='training'))
    receipt['url']=run.url if mode=='online' else None
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    run.define_metric('epoch')
    run.define_metric('train/*',step_metric='epoch')
    run.define_metric('validation/*',step_metric='epoch')
    return run


def log_epoch(run, record):
    run.log({'epoch':record['epoch']+1,'train/loss':record['train_loss'],
        'validation/loss':record['validation_loss'],'timing/epoch_seconds':record.get('elapsed_seconds',0),
        'train/batches':record['train_batches'],'validation/batches':record['validation_batches']})


def log_evaluation(run, report):
    import wandb
    rows=[]
    scalars={'test/loss':report['loss']['total'],
             'test/checkpoint_epoch':report['checkpoint_epoch']+1,
             'test/threshold':report['diagnostics']['threshold']}
    for role in ['external','self']:
        for cls, values in report['diagnostics'][role]['per_class'].items():
            rows.append([role,cls,*[values[k] for k in ['precision','recall','f1','tp','fp','fn','valid_frames']]])
            for key in ['precision','recall','f1','tp','fp','fn']:
                if values[key] is not None:
                    scalars[f'test/{role}/{cls}/{key}']=values[key]
            for key,value in values.get('conditional_localization',{}).items():
                if value is not None:
                    scalars[f'test/{role}/{cls}/{key}']=value
    columns=['role','class','precision','recall','f1','tp','fp','fn','valid_frames']
    table=wandb.Table(columns=columns,data=rows)
    charts={'test/per_class':table}
    for role in ['external','self']:
        subset=wandb.Table(columns=columns,data=[r for r in rows if r[0]==role])
        charts[f'test/{role}_f1']=wandb.plot.bar(subset,'class','f1',title=f'{role}: held-out frame F1 at 0.5')
    run.log(dict(scalars,**charts))
    run.summary.update(scalars)
    run.summary['real_game_accuracy_not_evaluated']=True
    run.summary['external_no_detections']=report['diagnostics']['external']['micro']['tp']==0 and report['diagnostics']['external']['micro']['fp']==0
