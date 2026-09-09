"""Read-only train-label/validation-score audit; never tune or modify a run."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/train-spatial-v6-400epochs-20260908'
CACHE = ROOT / 'datasets/cache-spatial-v6-20260908'
sys.path.insert(0, str(RUN / 'source_snapshot/src'))
from pubg_audio.features import FeatureNormalizer
from pubg_audio.model import ModelConfig, PaperSELD


def main():
    torch.set_num_threads(2)
    output = ROOT / 'outputs/reports/footstep_diagnosis_v6_20260908'
    output.mkdir(exist_ok=True)
    checkpoint_bytes = (RUN / 'last.pt').read_bytes()
    snapshot = output / 'checkpoint_snapshot.pt'
    snapshot.write_bytes(checkpoint_bytes)
    checkpoint = torch.load(snapshot, map_location='cpu', weights_only=True)
    model = PaperSELD(ModelConfig(**checkpoint['model_config'])).eval()
    model.load_state_dict(checkpoint['model'])
    normalize = FeatureNormalizer(**checkpoint['normalizer'])
    result = dict(checkpoint_epoch=checkpoint['epoch']+1,
                  checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                  scope='All training target plans and full validation inference; no test data, no parameter updates')
    classes = ['footsteps', 'vehicle', 'gunfire']
    train_valid = np.zeros(3, dtype=np.int64)
    train_positive = np.zeros(3, dtype=np.int64)
    bad_target_norms = np.zeros(3, dtype=np.int64)
    plans = sorted((CACHE / 'random_targets').glob('*.pt'))
    for path in plans:
        item = torch.load(path, map_location='cpu', weights_only=True)
        valid = item['mask'].numpy()>0
        positive = item['counts'].numpy()>0
        train_valid += valid.sum(axis=0)
        train_positive += (positive & valid).sum(axis=0)
        norms = torch.linalg.vector_norm(item['target'][..., :3], dim=-1).numpy()
        for cls in range(3):
            n = item['counts'][:, cls].numpy()
            for slot in range(3):
                expected = (n>slot) & valid[:, cls]
                bad_target_norms[cls] += np.count_nonzero(expected & (abs(norms[:, slot, cls]-1)>1e-4))
    result['training_plan_counts'] = {cls:dict(valid_frames=int(train_valid[i]),
        positive_frames=int(train_positive[i]),positive_fraction=float(train_positive[i]/train_valid[i]),
        bad_active_target_direction_norms=int(bad_target_norms[i])) for i,cls in enumerate(classes)}
    print(json.dumps(dict(stage='training_labels',scenes=len(plans),counts=result['training_plan_counts'])),flush=True)

    files = sorted((CACHE / 'validation').glob('*.pt'))
    values = {cls:dict(scores=[],truth=[],distances=[]) for cls in classes}
    started = time.monotonic()
    with torch.inference_mode():
        for begin in range(0,len(files),16):
            items = [torch.load(p,map_location='cpu',weights_only=True) for p in files[begin:begin+16]]
            features = torch.stack([normalize(x['features']) for x in items])
            scores = model(features)['accddoa'][...,:3].norm(dim=-1).amax(dim=2).numpy()
            for k,item in enumerate(items):
                for i,cls in enumerate(classes):
                    valid = item['mask'][:,i].numpy()>0
                    truth = item['counts'][:,i].numpy()>0
                    values[cls]['scores'].append(scores[k,:,i][valid])
                    values[cls]['truth'].append(truth[valid])
                    values[cls]['distances'].append(item['target'][:,0,i,3].numpy()[valid]*100.)
            if (begin//16+1)%15==0 or begin+16>=len(files):
                print(json.dumps(dict(stage='validation_predictions',done=min(begin+16,len(files)),total=len(files),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    result['validation'] = {}
    for cls,arrays in values.items():
        scores,truth,distances = [np.concatenate(arrays[k]) for k in ['scores','truth','distances']]
        quantiles = lambda a: {str(q):float(np.quantile(a,q)) for q in [.05,.25,.5,.75,.95,.99]} if len(a) else None
        record = dict(valid_frames=len(truth),positive_frames=int(truth.sum()),
            positive_scores=quantiles(scores[truth]),negative_scores=quantiles(scores[~truth]),
            positive_first_source_distances=quantiles(distances[truth]),fixed_threshold_diagnostics={})
        for threshold in [.1,.2,.3,.4,.5]:
            predicted=scores>=threshold
            tp,fp,fn=[int(x.sum()) for x in [predicted&truth,predicted&~truth,~predicted&truth]]
            record['fixed_threshold_diagnostics'][str(threshold)]=dict(tp=tp,fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
        result['validation'][cls] = record
    result['inference_seconds']=round(time.monotonic()-started,2)
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='completed',checkpoint_epoch=result['checkpoint_epoch'],report=str(output/'audit.json'),footsteps=result['validation']['footsteps'])),flush=True)


if __name__ == '__main__':
    main()
