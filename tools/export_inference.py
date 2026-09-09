"""Export model plus preprocessing metadata without optimizer/resume state."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.checkpoint.resolve()==args.output.resolve() or args.output.exists():
        raise ValueError('Keep the training checkpoint; select a new output file')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    keys=['model','normalizer','model_config','config','epoch','dataset','dataset_fingerprint','best_validation_loss']
    result={k:saved[k] for k in keys}
    for key in ['selection_metric','validation_loss_at_checkpoint','validation_external_macro_f1_at_checkpoint','validation_localization_macro_f1_at_checkpoint']:
        if key in saved:result[key]=saved[key]
    result.update(format='PUBGAudio-inference-v1',
        training_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        input_context_seconds=5.,output_hop_seconds=.1,noncausal=True,
        external_classes=['footsteps','vehicle','gunfire'],self_classes=['footsteps','vehicle','gunfire'],
        distance_units='uncalibrated synthetic units',real_game_accuracy_not_evaluated=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output)
    reloaded=torch.load(args.output,map_location='cpu',weights_only=True)
    for name,tensor in saved['model'].items():
        torch.testing.assert_close(reloaded['model'][name],tensor,rtol=0,atol=0)
    for key in ['mean','std']:
        torch.testing.assert_close(reloaded['normalizer'][key],saved['normalizer'][key],rtol=0,atol=0)
    print(json.dumps(dict(status='exported_and_verified',output=str(args.output),bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),checkpoint_epoch=saved['epoch']+1)),flush=True)


if __name__=='__main__':main()
