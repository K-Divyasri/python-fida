"""make_lipid_basis.py -- lipid-band basis for L2 water/lipid removal.

Faithful (not bit-exact) substitute for FID-A's sim_onepulse-based createLipipBasis
(which is stochastic anyway): random Lorentzian peaks in the lipid ppm band with
random linewidth + phase; FID -> fftshift(fft).
"""
import numpy as np
from mrsi_common import TWO_PI


def make_lipid_basis(Nf, spectralWidth, lipidComponents=1000, lineWidthRange=(1, 80),
                     lipidPPMRange=(0.3, 1.9), txfrq_MHz=123.25, ppm_ref=4.65, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dt = 1.0 / spectralWidth
    t = np.arange(Nf) * dt
    fids = np.zeros((Nf, lipidComponents), complex)
    for k in range(lipidComponents):
        lw = rng.uniform(*lineWidthRange)                 # Hz
        ppm = rng.uniform(*lipidPPMRange)
        phi = rng.uniform(-np.pi, np.pi)
        f0 = -(ppm - ppm_ref) * txfrq_MHz                 # Hz offset from ref
        fids[:, k] = np.exp(1j * (TWO_PI * f0 * t + phi)) * np.exp(-np.pi * lw * t)
    return np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
