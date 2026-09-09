"""Offline five-second inference; raw ADPIT tracks can contain duplicated detections."""
import argparse
import json
from pathlib import Path
import math
import wave
import numpy as np
import torch
from .features import extract,FeatureNormalizer
from .model import ModelConfig,PaperSELD
from .dataset import read_crop
from .confidence import load_calibration, presence_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--calibration',type=Path,help='Optional class-presence calibration fitted for this exact checkpoint')
    args=p.parse_args(); torch.set_num_threads(args.threads)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    calibration=load_calibration(args.calibration,args.checkpoint)
    config=ModelConfig(**checkpoint['model_config'])
    model=PaperSELD(config).eval(); model.load_state_dict(checkpoint['model'])
    normalize=FeatureNormalizer(**checkpoint['normalizer'])
    with wave.open(str(args.audio),'rb') as w:
        seconds=w.getnframes()/w.getframerate()
    if seconds<5 or abs(seconds/5-round(seconds/5))>1e-6:
        raise ValueError('Use a whole number of five-second windows; tail padding is not implemented')
    frames=[]
    with torch.inference_mode():
        for start in np.arange(0,seconds,5.):
            audio,rate=read_crop(args.audio,float(start),5.)
            out=model(normalize(extract(audio,rate))[None])
            raw=out['accddoa'][0].numpy() if 'accddoa' in out else np.concatenate([
                out['accdoa'][0].numpy(),out['distance'][0].numpy()[...,None]],axis=-1)[:,None]
            own=out['self_logits'][0].sigmoid().tolist() if 'self_logits' in out else [[]]*len(raw)
            for i,tracks in enumerate(raw):
                predictions=[]
                for slot,classes in enumerate(tracks):
                    for cls,value in enumerate(classes):
                        activity=float(np.linalg.norm(value[:3]))
                        predictions.append(dict(slot=slot,class_index=cls,activity_vector_norm=activity,
                            azimuth_degrees=math.degrees(math.atan2(float(value[0]),float(value[1])))%360,
                            distance_units=float(value[3])*checkpoint['config'].get('distance_scale',100.),
                            raw_xyzd=value.tolist(),direction_confidence=None,distance_confidence=None))
                scores=np.linalg.norm(tracks[...,:3],axis=-1).max(axis=0)
                frames.append(dict(start_seconds=round(float(start)+i*.1,4),raw_external_tracks=predictions,
                    self_probabilities=own[i],
                    external_class_predictions=presence_predictions(scores,'external',calibration),
                    self_class_predictions=presence_predictions(own[i],'self',calibration)))
    result=dict(audio=str(args.audio),checkpoint=str(args.checkpoint),hop_seconds=.1,
        noncausal_window_seconds=5.,adpit_tracks_not_deduplicated=True,real_game_accuracy_not_evaluated=True,
        confidence_schema='class-presence-v1',probability_scope=calibration['scope'] if calibration else None,
        calibration_file=str(args.calibration) if args.calibration else None,
        legacy_self_probabilities_are_uncalibrated=True,
        position_estimates_have_no_calibrated_uncertainty=True,frames=frames)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(frames=len(frames),output=str(args.output))))


if __name__=='__main__':
    main()
