"""Four-channel binaural features: magnitude, ILD, cosine/sine IPD.

24 kHz / 40 ms Hamming / 20 ms hop / 1024 FFT. Pad only the final
half-window, so five seconds gives exactly 250 frames (50 after pooling).
Normalization is fitted on training features and frozen for other splits.
"""
import math
import numpy as np
from scipy.signal import resample_poly
import torch

SAMPLE_RATE = 24000
WINDOW, HOP, FFT = 960, 480, 1024


def extract(stereo: np.ndarray, sample_rate: int) -> torch.Tensor:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError('Expected stereo PCM; do not duplicate mono training recordings')
    x = np.asarray(stereo, np.float32)
    if sample_rate != SAMPLE_RATE:
        gcd = math.gcd(sample_rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // gcd, sample_rate // gcd, axis=0).astype(np.float32)
    wave = torch.from_numpy(x.T.copy())
    count = math.ceil(wave.shape[-1] / HOP)
    needed = (count - 1) * HOP + WINDOW
    wave = torch.nn.functional.pad(wave, (0, max(0, needed - wave.shape[-1])))
    frames = wave.unfold(-1, WINDOW, HOP)
    window = torch.hamming_window(WINDOW, periodic=True)
    spectrum = torch.fft.rfft(frames * window, n=FFT)[..., 1:]
    mag = spectrum.abs() / window.sum()
    left, right = mag[0], mag[1]
    mean_mag = (left + right) * .5
    ild = 20 * torch.log10((left + 1e-8) / (right + 1e-8))
    cross = spectrum[0] * spectrum[1].conj()
    valid = (left > 1e-8) & (right > 1e-8)
    phase = cross / cross.abs().clamp_min(1e-12)
    return torch.stack([mean_mag, ild, torch.where(valid, phase.real, 0.), torch.where(valid, phase.imag, 0.)]).float()


class FeatureNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean).detach().cpu().float().reshape(4, 1, 512)
        self.std = torch.as_tensor(std).detach().cpu().float().reshape(4, 1, 512).clamp_min(1e-5)

    def __call__(self, features):
        return (features - self.mean) / self.std

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}


def fit_normalizer(features):
    total = torch.zeros((4, 512), dtype=torch.float64)
    total2 = torch.zeros_like(total)
    count = 0
    for x in features:
        x = x.double()
        total += x.sum(1)
        total2 += x.square().sum(1)
        count += x.shape[1]
    if not count:
        raise ValueError('Empty training split')
    mean = total / count
    return FeatureNormalizer(mean, (total2 / count - mean.square()).clamp_min(0).sqrt())
