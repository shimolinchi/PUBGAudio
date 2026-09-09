"""Frame/class presence scores and optional checkpoint-bound probability calibration."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

CLASSES = ('footsteps', 'vehicle', 'gunfire')
FORMAT = 'PUBGAudio-presence-calibration-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def transform(scores, kind):
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite confidence score')
    if kind == 'vector_norm':
        if (values < 0).any(): raise ValueError('Negative vector norm')
        return values
    if kind == 'sigmoid':
        if ((values < 0) | (values > 1)).any(): raise ValueError('Invalid sigmoid score')
        return logit(np.clip(values, 1e-6, 1-1e-6))
    raise ValueError('Unknown score kind')


def fit_binary(scores, labels, kind, minimum_positive_frames=20, minimum_negative_frames=20):
    """Monotonic Platt-style fit with natural class frequencies, not class weighting."""
    if any(not isinstance(n, int) or n < 1 for n in [minimum_positive_frames, minimum_negative_frames]):
        raise ValueError('Calibration sample minimums must be positive integers')
    x = transform(scores, kind)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.isin(y, [0,1]).all():
        raise ValueError('Expected aligned binary labels and scores')
    counts = dict(frames=len(y), positives=int(y.sum()), negatives=int(len(y)-y.sum()))
    if counts['positives'] < minimum_positive_frames or counts['negatives'] < minimum_negative_frames:
        return dict(status='insufficient_labels', score_kind=kind, **counts)
    centre, scale = float(x.mean()), max(float(x.std()), 1e-6)
    standardized = (x-centre)/scale
    def objective(params):
        a,b = params
        z = a*standardized+b
        residual = expit(z)-y
        loss = np.mean(np.logaddexp(0,z)-y*z)+1e-6*a*a
        gradient = np.array([np.mean(residual*standardized)+2e-6*a, residual.mean()])
        return float(loss), gradient
    fitted = minimize(objective, [1.,float(logit(y.mean()))], jac=True,
        bounds=[(0.,100.),(-100.,100.)], method='L-BFGS-B')
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError('Probability calibration did not converge')
    return dict(status='fitted', score_kind=kind, centre=centre, scale=scale,
                slope=float(fitted.x[0]), intercept=float(fitted.x[1]), **counts)


def probability(scores, fit):
    x = transform(scores, fit['score_kind'])
    if fit['status'] != 'fitted': return None
    params = [fit[k] for k in ['centre','scale','slope','intercept']]
    if not np.isfinite(params).all() or fit['scale'] <= 0 or fit['slope'] < 0:
        raise ValueError('Invalid calibration parameters')
    return np.clip(expit(fit['slope']*(x-fit['centre'])/fit['scale']+fit['intercept']),1e-6,1-1e-6)


def reliability(probabilities, labels):
    p,y = np.asarray(probabilities),np.asarray(labels)
    if (p.ndim != 1 or y.shape != p.shape or not np.isfinite(p).all()
            or ((p < 0) | (p > 1)).any() or not np.isin(y, [0,1]).all()):
        raise ValueError('Expected aligned probabilities and binary labels')
    if not len(y): return dict(frames=0,brier=None,ece=None,bins=[])
    bins=[];ece=0.
    for i in range(10):
        mask=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
        if not mask.any(): continue
        confidence,observed=float(p[mask].mean()),float(y[mask].mean())
        n=int(mask.sum());ece+=n/len(y)*abs(confidence-observed)
        bins.append(dict(lower=i/10,upper=(i+1)/10,frames=n,mean_probability=confidence,observed_fraction=observed))
    return dict(frames=len(y),brier=float(np.mean((p-y)**2)),ece=ece,bins=bins)


def load_calibration(path, checkpoint):
    if path is None: return None
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value['format']!=FORMAT or value['checkpoint_sha256']!=sha256(checkpoint):
        raise ValueError('Calibration does not match this exact checkpoint file')
    if value['scope']!='synthetic_validation_class_presence':
        raise ValueError('Unsupported probability calibration scope')
    if value['classes']!=list(CLASSES): raise ValueError('Calibration class order mismatch')
    for role,kind in [('external','vector_norm'),('self','sigmoid')]:
        if set(value['fits'][role]) != set(CLASSES):
            raise ValueError('Calibration must contain every class exactly once')
        for fit in value['fits'][role].values():
            if fit['score_kind']!=kind: raise ValueError('Calibration score kind mismatch')
            if fit['status']=='fitted': probability([.5],fit)
            elif fit['status']!='insufficient_labels': raise ValueError('Unknown calibration status')
    return value


def presence_predictions(scores, role, calibration=None):
    kind='vector_norm' if role=='external' else 'sigmoid' if role=='self' else None
    values=transform(scores,kind)
    if values.ndim != 1: raise ValueError('Expected one score per class')
    if calibration is not None and len(values) != len(CLASSES):
        raise ValueError('Calibration class count mismatch')
    result=[]
    for cls,score in enumerate(scores):
        name=CLASSES[cls] if len(values)==len(CLASSES) else f'class_{cls}'
        fit=calibration['fits'][role].get(name) if calibration else None
        p=probability([score],fit) if fit is not None else None
        result.append(dict(class_index=cls,class_name=name,presence_score=float(score),score_kind=kind,
            presence_probability=float(p[0]) if p is not None else None,
            probability_status='fitted_on_synthetic_validation' if p is not None else 'insufficient_labels' if fit else 'not_calibrated'))
    return result
