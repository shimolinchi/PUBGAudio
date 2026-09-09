"""Fit class-presence calibration on validation scenes; never access test labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pubg_audio.confidence import CLASSES,FORMAT,sha256,fit_binary,probability,reliability
from pubg_audio.dataset import SyntheticScenes
from pubg_audio.cache import prepare_cache
from pubg_audio.features import FeatureNormalizer
from pubg_audio.model import ModelConfig,PaperSELD
sys.path.insert(0,str(Path(__file__).resolve().parent/'synthesis'))
from confidence_policy import read_contract, split_scenes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation-cache',type=Path)
    parser.add_argument('--seed',type=int,help='Legacy datasets only; new datasets freeze their calibration scene split')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve an existing calibration; choose a new output')
    if args.threads < 1: raise ValueError('Threads must be positive')
    torch.set_num_threads(args.threads)
    fingerprint=sha256(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    manifest_fingerprint=sha256(args.dataset/'manifest_validation.jsonl')
    if saved['dataset_fingerprint']['validation']!=manifest_fingerprint:
        raise ValueError('Validation manifest differs from checkpoint')
    cfg=ModelConfig(**saved['model_config'])
    if cfg.classes!=3 or cfg.self_classes!=3: raise ValueError('Calibration requires the three-class external/self model')
    data=SyntheticScenes(args.dataset,'validation',distance_scale=saved['config'].get('distance_scale',100.))
    contract=read_contract(args.dataset)
    seed=args.seed if args.seed is not None else 20260908
    minimums={}
    if contract is not None:
        policy=contract['policy']['calibration']
        if args.seed is not None and args.seed!=policy['seed']:
            raise ValueError('Do not override the generated calibration scene split')
        seed=policy['seed']
        fit_ids=set(contract['plan']['fit_scene_ids'])
        audit_ids=set(contract['plan']['audit_scene_ids'])
        minimums={k:policy[k] for k in ['minimum_positive_frames','minimum_negative_frames']}
    else:
        fit_ids,audit_ids=split_scenes(data.rows,seed)
    if args.validation_cache: data=prepare_cache(data,args.validation_cache)
    data.normalizer=FeatureNormalizer(**saved['normalizer'])
    model=PaperSELD(cfg).eval();model.load_state_dict(saved['model'])
    records={part:{role:{cls:[[],[]] for cls in CLASSES} for role in ['external','self']} for part in ['fit','audit']}
    with torch.inference_mode():
        for i,batch in enumerate(DataLoader(data,batch_size=16,shuffle=False,num_workers=0)):
            out=model(batch['features'])
            norm=out['accddoa'][...,:3].norm(dim=-1).amax(dim=2) if 'accddoa' in out else out['accdoa'].norm(dim=-1)
            scores={'external':norm.numpy(),'self':out['self_logits'].sigmoid().numpy()}
            truths={'external':batch['counts'].numpy()>0,'self':batch['self_target'].numpy()>0}
            masks={'external':batch['mask'].numpy()>0,'self':batch['self_mask'].numpy()>0}
            for row,sid in enumerate(batch['scene_id']):
                part='fit' if sid in fit_ids else 'audit'
                for role in scores:
                    for col,cls in enumerate(CLASSES):
                        valid=masks[role][row,:,col]
                        record=records[part][role][cls]
                        record[0].extend(scores[role][row,:,col][valid].tolist())
                        record[1].extend(truths[role][row,:,col][valid].tolist())
            if (i+1)%15==0: print(json.dumps(dict(validation_batches=i+1)),flush=True)
    if sha256(args.checkpoint)!=fingerprint: raise ValueError('Checkpoint changed during calibration; use a frozen export')
    if sha256(args.dataset/'manifest_validation.jsonl')!=manifest_fingerprint:
        raise ValueError('Validation manifest changed during calibration')
    if read_contract(args.dataset)!=contract:
        raise ValueError('Dataset confidence contract changed during calibration')
    fits={};audit={}
    for role,kind in [('external','vector_norm'),('self','sigmoid')]:
        fits[role]={};audit[role]={}
        for cls in CLASSES:
            scores,truth=records['fit'][role][cls]
            fit=fit_binary(scores,truth,kind,**minimums);fits[role][cls]=fit
            scores,truth=records['audit'][role][cls]
            calibrated=probability(scores,fit)
            audit[role][cls]=dict(raw_bounded_score_baseline=reliability(np.clip(scores,0,1),truth),
                calibrated=reliability(calibrated,truth) if calibrated is not None else None)
    result=dict(format=FORMAT,checkpoint_sha256=fingerprint,checkpoint_epoch=saved['epoch']+1,
        classes=list(CLASSES),scope='synthetic_validation_class_presence',fits=fits,audit=audit,
        fit_scene_ids=sorted(fit_ids),audit_scene_ids=sorted(audit_ids),seed=seed,
        confidence_policy_sha256=contract['policy_sha256'] if contract else None,
        calibration_plan_sha256=contract['plan_sha256'] if contract else None,
        validation_manifest_sha256=manifest_fingerprint,
        event_definition='At least one audible/unmasked source of this class and role in a 100 ms frame; not per-track or position correctness',
        limitations=['Audit scenes excluded from calibrator fitting but validation was used for model selection.',
                    'Correlated synthetic frames; not real-game calibration or independent real-game validation.',
                    'No direction/distance confidence; no test labels used; natural calibration class frequencies retained.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='calibration_fitted',output=str(args.output),checkpoint_epoch=saved['epoch']+1)),flush=True)


if __name__=='__main__':main()
