"""Circular statistics for the side chain torsions (chi angles) used by MDPack's hybrid codec.

The hybrid codec stores one static chi prior per residue and chi (the angle the shared packer starts from); the
residuals it codes are Cartesian. This module estimates that prior from the trajectory's own chi angles:
  mean      circular mean (normalised mean of (sin, cos)); this is the maximum likelihood mean direction of a single
            von Mises distribution, and was MDPack's behaviour before the flag existed (default).
  vonmises  mode of a von Mises mixture (K = 3 components initialised at the g+, t, g- wells, EM with the Banerjee
            et al. kappa approximation): the densest angle, which for a side chain that jumps between rotamers is a
            real rotamer and not the in between direction a single circular mean gives.
  empirical mode of a smoothed histogram (5 degree bins, circular smoothing over +-10 degrees).
  gaussian  arithmetic mean of the angles in (-180, 180] (not circular; reference only: wrong for wells near 180).
pi periodic chis (canonicalised to the cos >= 0 half circle) are handled on the doubled angle."""
import numpy as np

MODELS = ("mean", "vonmises", "empirical", "gaussian")


def _a1inv(r):  # inverse of A1(kappa) = I1/I0 (Banerjee et al. 2005 approximation)
    r = np.clip(r, 1e-6, 0.999)
    return r * (2 - r ** 2) / (1 - r ** 2)


def vonmises_mixture(theta, K=3, iters=50, init=None):
    """theta [M, T] radians -> weights [M,K], mu [M,K], kappa [M,K] (vectorised EM over the M series)."""
    from scipy.special import i0e
    M, T = theta.shape
    mu = np.tile(np.asarray(init if init is not None else np.linspace(-np.pi, np.pi, K, endpoint=False), float), (M, 1))
    kap = np.full((M, K), 4.0); w = np.full((M, K), 1.0 / K)
    for _ in range(iters):
        ll = np.log(w)[:, :, None] + kap[:, :, None] * (np.cos(theta[:, None] - mu[:, :, None]) - 1) - np.log(2 * np.pi * i0e(kap))[:, :, None]
        ll -= ll.max(1, keepdims=True); g = np.exp(ll); g /= g.sum(1, keepdims=True)  # responsibilities [M,K,T]
        n = g.sum(2) + 1e-12; w = n / T
        S, C = (g * np.sin(theta[:, None])).sum(2), (g * np.cos(theta[:, None])).sum(2)
        mu = np.arctan2(S, C); kap = np.clip(_a1inv(np.hypot(S, C) / n), 1e-3, 500)
    return w, mu, kap


def _mode(w, mu, kap, grid=720):
    from scipy.special import i0e
    g = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    dens = (w[:, :, None] * np.exp(kap[:, :, None] * (np.cos(g[None, None] - mu[:, :, None]) - 1)) / (2 * np.pi * i0e(kap))[:, :, None]).sum(1)
    return g[dens.argmax(1)]


def chi_prior(chi_sc, chi_mask, pi_periodic, model="mean"):
    """chi_sc [T,R,4,2] (sin, cos), already canonicalised; chi_mask [R,4]; pi_periodic [R,4] -> prior [R,4,2]."""
    if model not in MODELS: raise ValueError(f"angle_model must be one of {MODELS}")
    cs = np.asarray(chi_sc, np.float64); T, R = cs.shape[:2]
    th = np.arctan2(cs[..., 0], cs[..., 1]).reshape(T, -1).T  # [R*4, T]
    per = np.asarray(pi_periodic, bool).reshape(-1); th2 = np.where(per[:, None], 2 * th, th)
    th2 = np.angle(np.exp(1j * th2))
    if model == "mean":
        m = np.arctan2(np.sin(th2).mean(1), np.cos(th2).mean(1))
    elif model == "gaussian":
        m = th2.mean(1)
    elif model == "vonmises":
        w, mu, kap = vonmises_mixture(th2, 3, init=np.deg2rad([60.0, 180.0, -60.0])); m = _mode(w, mu, kap)
    else:
        b = np.floor((th2 + np.pi) / (2 * np.pi) * 72).astype(int) % 72
        h = np.stack([np.bincount(r, minlength=72) for r in b]).astype(float)
        h = sum(np.roll(h, s, 1) for s in (-2, -1, 0, 1, 2)); m = (h.argmax(1) + 0.5) / 72 * 2 * np.pi - np.pi
    m = np.where(per, m / 2, m)
    out = np.stack([np.sin(m), np.cos(m)], -1).reshape(R, 4, 2)
    out = np.where((per.reshape(R, 4) & (out[..., 1] < 0))[..., None], -out, out)  # back to the cos >= 0 half circle
    return np.where(np.asarray(chi_mask, bool)[..., None], out, np.array([0.0, 1.0])).astype(np.float32)


def concentration(chi_sc, chi_mask):
    """mean resultant length per chi [R,4] (1 = rigid, 0 = uniform) and the von Mises kappa it implies."""
    cs = np.asarray(chi_sc, np.float64); r = np.hypot(cs[..., 0].mean(0), cs[..., 1].mean(0))
    return np.where(chi_mask, r, np.nan), np.where(chi_mask, _a1inv(r), np.nan)
