"""Smooth, physically motivated air loss; not a measured PUBG transfer curve.

Energy attenuation coefficients (1/m), 20 C and 50--70% RH, from the
pyroomacoustics room documentation. Amplitude is exp(-coefficient*d/2).
The common linear-phase delay stays 64 samples, as in the old renderer.
"""
import numpy as np


def air_kernel(distance, settings, rate=44100):
    frequencies = np.fft.rfftfreq(4096, 1/rate)
    knots = np.asarray(settings['frequencies_hz'], float)
    coefficients = np.asarray(settings['energy_coefficients_per_m'], float)
    if (knots.ndim != 1 or len(knots) != len(coefficients) or
            not np.all(np.diff(knots) > 0) or np.any(coefficients < 0)):
        raise ValueError('Invalid air absorption table')
    alpha = np.interp(frequencies, knots, coefficients)
    high = frequencies > knots[-1]
    exponent = np.log(coefficients[-1]/coefficients[-2])/np.log(knots[-1]/knots[-2])
    alpha[high] = coefficients[-1]*(frequencies[high]/knots[-1])**exponent
    alpha[0] = 0
    amplitude = np.exp(-.5*alpha*max(float(distance)-1., 0.))
    centered = np.fft.fftshift(np.fft.irfft(amplitude, n=4096))
    h = centered[2048-64:2048+65]*np.hamming(129)
    return (h/h.sum()).astype(np.float32)


def configure_spatial(spatial, cfg):
    settings = cfg.get('air_absorption')
    if settings:
        spatial.air_distance_grid = np.asarray(settings['distance_grid'], float)
        spatial.air_fft = np.fft.rfft(np.array([air_kernel(d, settings) for d in spatial.air_distance_grid]),
                                       spatial.nfft, axis=1)
    return spatial
