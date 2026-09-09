"""Explicit, project-bound W&B logging; no inference from global project settings."""
import json
from pathlib import Path
import tempfile
import uuid


def load_binding(path):
    binding=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if binding.get('application')!='PUBGAudio' or not binding.get('created_for_this_module'):
        raise ValueError('Use the independent project created by tools/setup_wandb.py')
    if not all(isinstance(binding.get(k),str) and binding[k] for k in ['entity','project','project_id']):
        raise ValueError('Incomplete W&B project binding')
    return binding


def start_run(binding_path, output, name, config, job_type='training', mode='online'):
    binding=load_binding(binding_path)
    if mode not in ['online','offline']:
        raise ValueError('Explicit W&B mode required')
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    staging=output/'media-temp';staging.mkdir(parents=True,exist_ok=True)
    # W&B creates its media TemporaryDirectory during import, before init().
    tempfile.tempdir=str(staging)
    import wandb
    if mode=='online':
        project=wandb.Api(timeout=20).project(binding['project'],entity=binding['entity'])
        if project.id!=binding['project_id']:
            raise ValueError('W&B project identity changed')
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
    values={'epoch':record['epoch']+1,'train/loss':record['train_loss'],
        'validation/loss':record['validation_loss'],'timing/epoch_seconds':record.get('elapsed_seconds',0),
        'train/batches':record['train_batches'],'validation/batches':record['validation_batches']}
    if record.get('validation_external_macro_f1') is not None:
        values['validation/external_macro_f1']=record['validation_external_macro_f1']
    if record.get('validation_localization_macro_f1') is not None:
        values['validation/localization_macro_f1']=record['validation_localization_macro_f1']
    if 'learning_rate' in record:values['train/learning_rate']=record['learning_rate']
    for role,section in record.get('validation_diagnostics',{}).items():
        if role not in ['external','self']:continue
        for cls,metrics in section['per_class'].items():
            for key in ['precision','recall','f1']:
                if metrics[key] is not None:values[f'validation/{role}/{cls}/{key}']=metrics[key]
            for key,value in metrics.get('conditional_localization',{}).items():
                if value is not None:values[f'validation/{role}/{cls}/{key}']=value
            for key,value in metrics.get('localization_aware',{}).items():
                if value is not None:values[f'validation/{role}/{cls}/localized_{key}']=value
    run.log(values)


def log_evaluation(run, report, namespace='test'):
    if namespace not in ['test','test_detection','test_localization']:
        raise ValueError('Use a separate named test stream for each validation selection rule')
    import wandb
    rows=[]
    scalars={f'{namespace}/loss':report['loss']['total'],
             f'{namespace}/checkpoint_epoch':report['checkpoint_epoch']+1,
             f'{namespace}/threshold':report['diagnostics']['threshold']}
    for role in ['external','self']:
        for cls, values in report['diagnostics'][role]['per_class'].items():
            rows.append([role,cls,*[values[k] for k in ['precision','recall','f1','tp','fp','fn','valid_frames']]])
            for key in ['precision','recall','f1','tp','fp','fn']:
                if values[key] is not None:
                    scalars[f'{namespace}/{role}/{cls}/{key}']=values[key]
            for key,value in values.get('conditional_localization',{}).items():
                if value is not None:
                    scalars[f'{namespace}/{role}/{cls}/{key}']=value
            for key,value in values.get('localization_aware',{}).items():
                if value is not None:scalars[f'{namespace}/{role}/{cls}/localized_{key}']=value
    if report['diagnostics'].get('localization_macro_f1') is not None:
        scalars[f'{namespace}/localization_macro_f1']=report['diagnostics']['localization_macro_f1']
    columns=['role','class','precision','recall','f1','tp','fp','fn','valid_frames']
    table=wandb.Table(columns=columns,data=rows)
    charts={f'{namespace}/per_class':table}
    for role in ['external','self']:
        subset=wandb.Table(columns=columns,data=[r for r in rows if r[0]==role])
        charts[f'{namespace}/{role}_f1']=wandb.plot.bar(subset,'class','f1',title=f'{namespace} {role}: held-out frame F1 at 0.5')
    run.log(dict(scalars,**charts))
    run.summary.update(scalars)
    run.summary['real_game_accuracy_not_evaluated']=True
    key='external_no_detections' if namespace=='test' else f'{namespace}/external_no_detections'
    run.summary[key]=report['diagnostics']['external']['micro']['tp']==0 and report['diagnostics']['external']['micro']['fp']==0
