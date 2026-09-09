"""Held-out, masked diagnostics; these are not official DCASE/SELD scores."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import SyntheticScenes
from .features import FeatureNormalizer
from .losses import training_loss
from .model import ModelConfig, PaperSELD
from .spatial_metrics import SpatialFrameMetrics

CLASSES = ('footsteps', 'vehicle', 'gunfire')


def binary_summary(tp, fp, fn, tn):
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        positive_frames=tp+fn, valid_frames=tp+fp+fn+tn)


class FrameDiagnostics:
    def __init__(self, threshold=.5, distance_scale=100., localization_aware=False,
                 angular_tolerance_deg=20., duplicate_tolerance_deg=15.):
        self.threshold, self.distance_scale = threshold, distance_scale
        self.confusion = {role: torch.zeros(3,4,dtype=torch.int64) for role in ['external','self']}
        self.ignored = {role: torch.zeros(3,dtype=torch.int64) for role in self.confusion}
        self.localization = [dict(frames=0, angular_sum=0., distance_sum=0.) for _ in CLASSES]
        self.spatial = SpatialFrameMetrics(threshold,angular_tolerance_deg,duplicate_tolerance_deg) if localization_aware else None

    def update(self, outputs, batch):
        outputs = {k:v.detach().cpu() for k,v in outputs.items()}
        batch = {k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        if 'accddoa' in outputs:
            raw = outputs['accddoa']
            scores = torch.linalg.vector_norm(raw[...,:3],dim=-1)
            confidence, selected = scores.max(dim=2)
            chosen = raw.gather(2,selected[:,:,None,:,None].expand(-1,-1,1,-1,4)).squeeze(2)
        else:
            confidence = torch.linalg.vector_norm(outputs['accdoa'],dim=-1)
            chosen = torch.cat([outputs['accdoa'],outputs['distance'][...,None]],dim=-1)
        predicted = confidence >= self.threshold
        cases = [('external',predicted,batch['counts']>0,batch['mask']>0)]
        if 'self_logits' in outputs:
            cases.append(('self',outputs['self_logits'].sigmoid()>=self.threshold,
                          batch['self_target']>0,batch['self_mask']>0))
        for role, prediction, truth, valid in cases:
            for cls in range(3):
                p,t,m = prediction[...,cls],truth[...,cls],valid[...,cls]
                values = [p&t&m,p&~t&m,~p&t&m,~p&~t&m]
                self.confusion[role][cls] += torch.tensor([int(v.sum()) for v in values])
                self.ignored[role][cls] += int((~m).sum())
        # Localization is conditional on detecting one isolated source. Choose
        # the highest-confidence track without consulting its true direction.
        eligible = predicted & (batch['counts']==1) & (batch['mask']>0)
        if 'localization_mask' in batch:
            eligible &= batch['localization_mask'][:,:,0,:]>0
        if self.spatial is not None:
            self.spatial.update(raw if 'accddoa' in outputs else chosen[:,:,None],batch)
        for cls in range(3):
            p = chosen[...,cls,:][eligible[...,cls]]
            t = batch['target'][:,:,0,cls][eligible[...,cls]]
            if not len(p):
                continue
            direction = torch.nn.functional.normalize(p[:,:3],dim=-1)
            truth = torch.nn.functional.normalize(t[:,:3],dim=-1)
            angle = torch.rad2deg(torch.acos((direction*truth).sum(-1).clamp(-1,1)))
            distances = (p[:,3]-t[:,3]).abs()*self.distance_scale
            stat = self.localization[cls]
            stat['frames'] += len(p)
            stat['angular_sum'] += float(angle.sum())
            stat['distance_sum'] += float(distances.sum())

    def report(self):
        result = dict(threshold=self.threshold, frame_seconds=.1,
            external_activity_rule='maximum xyz norm over ADPIT tracks; duplicate tracks count once',
            localization_rule='detected single-source frames with spatial supervision when available; highest-confidence track, no truth-based selection',
            distance_units='uncalibrated synthetic units, not measured metres')
        for role, counts in self.confusion.items():
            per_class = {}
            for i, cls in enumerate(CLASSES):
                row = binary_summary(*counts[i].tolist())
                row['ignored_frames'] = int(self.ignored[role][i])
                if role == 'external':
                    s = self.localization[i]; n = s['frames']
                    row['conditional_localization'] = dict(detected_single_source_frames=n,
                        angular_mae_degrees=s['angular_sum']/n if n else None,
                        distance_mae_units=s['distance_sum']/n if n else None)
                    if self.spatial is not None:row['localization_aware'] = self.spatial.report(i)
                per_class[cls] = row
            result[role] = dict(per_class=per_class,micro=binary_summary(*counts.sum(0).tolist()))
        if self.spatial is not None:
            values=[self.spatial.report(i)['f1'] for i in range(3) if self.spatial.report(i)['target_instances']>0]
            result['localization_macro_f1']=sum(values)/len(values) if values else None
            result['localization_aware_scope']='100ms source matching within angular tolerance; no distance gate; not official DCASE metrics'
        return result


def evaluate_checkpoint(checkpoint, root, split='test', device='cpu', batch_size=4, workers=0, threshold=.5):
    checkpoint,root=Path(checkpoint),Path(root)
    if split not in ['validation','test'] or not 0<threshold<1 or batch_size<1 or workers<0:
        raise ValueError('Invalid evaluation settings')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    fingerprints = {name:hashlib.sha256((root/f'manifest_{name}.jsonl').read_bytes()).hexdigest()
                    for name in ['train','validation','test']}
    if saved.get('dataset_fingerprint') != fingerprints:
        raise ValueError('Dataset manifests differ from the checkpoint; refusing held-out evaluation')
    cfg = ModelConfig(**saved['model_config'])
    if cfg.classes!=3:
        raise ValueError('Synthetic evaluation supports the three PUBG classes')
    scale = saved['config'].get('distance_scale',100.)
    dataset = SyntheticScenes(root,split,classes=cfg.classes,distance_scale=scale,
                              normalizer=FeatureNormalizer(**saved['normalizer']))
    if not len(dataset):
        raise ValueError('Empty evaluation split')
    model = PaperSELD(cfg).to(device)
    model.load_state_dict(saved['model']); model.eval()
    selection=saved['config'].get('selection',{})
    stats = FrameDiagnostics(threshold,scale,bool(selection),selection.get('angular_tolerance_deg',20),selection.get('duplicate_tolerance_deg',15))
    weighted = dict(total=0.,external=0.,self=0.)
    samples = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers):
            batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            outputs = model(batch['features'])
            loss, parts = training_loss(outputs,batch,loss_config=saved['config'].get('loss'))
            if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in outputs.values()):
                raise ValueError('Nonfinite test output')
            n = len(batch['features']); samples += n
            for name, value in dict(total=loss,**parts).items():
                weighted[name] += float(value)*n
            stats.update(outputs,batch)
    report = dict(status='evaluated',scope='small synthetic held-out diagnostics, not official SELD scores',
        checkpoint_epoch=saved['epoch'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        split=split,scenes=len(dataset.rows),crops=samples,dataset_fingerprint=fingerprints,
        device=device,torch_version=str(torch.__version__),batch_size=batch_size,
        loss={k:v/samples for k,v in weighted.items()},loss_aggregation='crop-weighted batch masked objective',
        diagnostics=stats.report(),real_game_accuracy_not_evaluated=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threshold',type=float,default=.5,help='Fixed diagnostic threshold; do not tune on test data')
    args = p.parse_args()
    if not 0<args.threshold<1 or args.batch_size<1 or args.workers<0 or args.threads<1:
        p.error('Invalid threshold, batch size, workers or threads')
    torch.set_num_threads(args.threads)
    report=evaluate_checkpoint(args.checkpoint,args.dataset,args.split,args.device,args.batch_size,args.workers,args.threshold)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
