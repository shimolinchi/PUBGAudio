"""Class-aware, azimuth-gated multi-source frame score for model selection.

This is a local 100 ms metric, not the complete official DCASE score. Distance
does not gate matches because this application's distance scale is provisional.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class SpatialFrameMetrics:
    def __init__(self, threshold=.5, angular_tolerance_deg=20., duplicate_tolerance_deg=15.):
        if not 0 < duplicate_tolerance_deg <= angular_tolerance_deg < 180:
            raise ValueError('Invalid angular tolerances')
        self.threshold, self.tolerance, self.duplicate = threshold, angular_tolerance_deg, duplicate_tolerance_deg
        self.counts = np.zeros((3, 3), dtype=np.int64)  # TP, FP, FN
        self.ignored = np.zeros(3, dtype=np.int64)

    @staticmethod
    def errors(a, b):
        return abs((a[:, None]-b[None, :]+180)%360-180)

    def update(self, raw, batch):
        raw = raw.numpy()
        target = batch['target'].numpy()
        counts, valid = batch['counts'].numpy(), batch['mask'].numpy()>0
        loc = batch.get('localization_mask')
        loc = loc.numpy() if loc is not None else np.linalg.norm(target[...,:3],axis=-1)>0
        scores = np.linalg.norm(raw[...,:3],axis=-1)
        for cls in range(3):
            for b,t in np.argwhere(valid[...,cls] & ((counts[...,cls]>0) | (scores[...,cls].max(axis=2)>=self.threshold))):
                n = int(counts[b,t,cls])
                if n and not np.all(loc[b,t,:n,cls]>0):
                    self.ignored[cls] += 1
                    continue
                truth = np.degrees(np.arctan2(target[b,t,:n,cls,0],target[b,t,:n,cls,1]))
                candidates = sorted(np.flatnonzero(scores[b,t,:,cls]>=self.threshold),
                                    key=lambda i: -scores[b,t,i,cls])
                angles = []
                for slot in candidates:
                    angle = np.degrees(np.arctan2(raw[b,t,slot,cls,0],raw[b,t,slot,cls,1]))
                    if all(abs((angle-a+180)%360-180)>self.duplicate for a in angles):
                        angles.append(angle)
                matched = 0
                if n and angles:
                    errors = self.errors(np.asarray(angles),truth)
                    # Penalize invalid edges first so assignment maximizes the
                    # number of valid matches, then minimizes their angle error.
                    i,j = linear_sum_assignment(errors + (errors>self.tolerance)*1000.)
                    matched = int((errors[i,j]<=self.tolerance).sum())
                self.counts[cls] += [matched,len(angles)-matched,n-matched]

    def report(self, cls):
        tp,fp,fn = map(int,self.counts[cls])
        return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            target_instances=tp+fn,ignored_ambiguous_frames=int(self.ignored[cls]),
            angular_tolerance_deg=self.tolerance,duplicate_tolerance_deg=self.duplicate)
