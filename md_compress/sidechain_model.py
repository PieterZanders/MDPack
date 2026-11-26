"""Specialized side-chain model: ONE network shared by all 20 residue types.

    inputs : local backbone frame (N, C, O expressed in the residue's own N-CA-C frame,
             i.e. rotation/translation invariant), residue-type embedding, and a
             placement/context vector produced by the mapping (packing) network.
    trunk  : small residual MLP  -> hidden h
    head   : SideChainHead       -> chi angles as (sin, cos), up to 4, masked per type
    builder: SideChainBuilder    -> deterministic NeRF placement of heavy atoms from
                                    ideal internal coordinates (no parameters)

Because only the head decides *how* chis are produced, swapping the deterministic
model for a generative one is a one-line change:

    model = SideChainModel(head="deterministic")   # default, regression
    model = SideChainModel(head="vonmises")        # conditional autoregressive von Mises mixture

To add a real normalizing flow on the torus (e.g. circular rational-quadratic
spline coupling flow, Rezende et al. 2020; or a RealNVP on (sin,cos)-embedded
angles), subclass `SideChainHead`, implement `forward` (point estimate + optional
NLL when `chi_true` is given) and `sample`, then register it in HEADS. Nothing else
in the pipeline (mapping net, builder, losses, codec, inference) needs to change,
since all downstream code only consumes chi (sin, cos).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import residue_constants as rc
from .geometry import SideChainBuilder, frames_from_backbone, to_local


def mlp(d_in, d_hidden, d_out, n_layers=2, act=nn.SiLU):
    layers, d = [], d_in
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, d_hidden), act()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- heads
class SideChainHead(nn.Module):
    """Abstract interface for chi prediction heads.

    forward(h, chi_mask, chi_true=None, chi_ref=None) -> dict(
        chi_sc : [...,4,2] point estimate (sin, cos), unit-norm, zero where masked
        nll    : scalar negative log-likelihood (only for probabilistic heads, when chi_true given)
    )
    sample(h, chi_mask, n_samples, chi_ref=None) -> [n_samples, ...,4,2]

    chi_ref (optional, [...,4,2]) is a per-residue prior rotamer (e.g. from the stored reference
    structure). Heads predict a rotation relative to it; without it they predict absolute chis.
    """
    probabilistic = False

    def forward(self, h, chi_mask, chi_true=None, chi_ref=None):  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, h, chi_mask, n_samples: int = 1, chi_ref=None):
        out = self.forward(h, chi_mask, chi_ref=chi_ref)["chi_sc"]
        return out.expand(n_samples, *out.shape)


def rotate_sc(a, b):
    """(sin,cos) of angle(a) + angle(b)."""
    return torch.stack([a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
                        a[..., 1] * b[..., 1] - a[..., 0] * b[..., 0]], -1)


def identity_sc(shape, like):
    z = like.new_zeros(*shape, rc.MAX_CHI, 2)
    z[..., 1] = 1.0
    return z


class DeterministicChiHead(SideChainHead):
    """Regresses (sin, cos) per chi and normalises to the unit circle."""

    def __init__(self, d_hidden):
        super().__init__()
        self.lin = nn.Linear(d_hidden, rc.MAX_CHI * 2)
        with torch.no_grad():  # start at "no rotation" relative to the prior
            self.lin.weight.mul_(0.1)
            self.lin.bias.copy_(torch.tensor([0.0, 1.0] * rc.MAX_CHI))

    def forward(self, h, chi_mask, chi_true=None, chi_ref=None):
        x = self.lin(h).view(*h.shape[:-1], rc.MAX_CHI, 2)
        delta = F.normalize(x, dim=-1, eps=1e-6)
        chi_sc = (rotate_sc(delta, chi_ref) if chi_ref is not None else delta) * chi_mask[..., None]
        # unnormalised norm, used for an AF2-style norm regulariser
        return dict(chi_sc=chi_sc, chi_raw_norm=x.norm(dim=-1), nll=None)


class VonMisesMixtureHead(SideChainHead):
    """Conditional generative head: p(chi1..chi4 | h) = prod_k p(chi_k | h, chi_<k),
    each factor a K-component von Mises mixture. Exact likelihood, cheap sampling,
    multimodal (rotamers). `forward` returns the mode of the most probable component
    (deterministic), `sample` draws rotamers."""
    probabilistic = True

    def __init__(self, d_hidden, n_components=3, d_mid=64):
        super().__init__()
        self.K = n_components
        self.net = mlp(d_hidden + rc.MAX_CHI * 2 + rc.MAX_CHI, d_mid, self.K * 4)

    def _params(self, h, prev_sc, k, chi_ref=None):
        onehot = F.one_hot(torch.full(h.shape[:-1], k, device=h.device), rc.MAX_CHI).to(h.dtype)
        p = self.net(torch.cat([h, prev_sc.flatten(-2), onehot], -1)).view(*h.shape[:-1], self.K, 4)
        logits = p[..., 0]
        mu = torch.atan2(p[..., 1], p[..., 2])
        if chi_ref is not None:  # component means relative to the prior rotamer
            mu = mu + torch.atan2(chi_ref[..., k, 0], chi_ref[..., k, 1])[..., None]
        kappa = F.softplus(p[..., 3]) + 1e-3
        return logits, mu, kappa

    @staticmethod
    def _vm_logpdf(x, mu, kappa):
        # log I0(k) = log(i0e(k)) + k  (numerically stable)
        return kappa * torch.cos(x - mu) - math.log(2 * math.pi) - (torch.log(torch.special.i0e(kappa)) + kappa)

    def forward(self, h, chi_mask, chi_true=None, chi_ref=None):
        prev = torch.zeros(*h.shape[:-1], rc.MAX_CHI, 2, device=h.device, dtype=h.dtype)
        out, nll = [], 0.0
        for k in range(rc.MAX_CHI):
            logits, mu, kappa = self._params(h, prev, k, chi_ref)
            best = logits.argmax(-1, keepdim=True)
            mode = torch.gather(mu, -1, best).squeeze(-1)
            sc_mode = torch.stack([torch.sin(mode), torch.cos(mode)], -1)
            if chi_true is not None:
                x = torch.atan2(chi_true[..., k, 0], chi_true[..., k, 1])
                lp = torch.logsumexp(F.log_softmax(logits, -1) + self._vm_logpdf(x[..., None], mu, kappa), -1)
                nll = nll - (lp * chi_mask[..., k]).sum()
                nxt = chi_true[..., k, :]          # teacher forcing
            else:
                nxt = sc_mode
            out.append(sc_mode)
            prev = prev.clone()
            prev[..., k, :] = nxt * chi_mask[..., k, None]
        chi_sc = torch.stack(out, -2) * chi_mask[..., None]
        if chi_true is not None:
            nll = nll / chi_mask.sum().clamp(min=1)
        else:
            nll = None
        return dict(chi_sc=chi_sc, nll=nll, chi_raw_norm=None)

    @torch.no_grad()
    def sample(self, h, chi_mask, n_samples: int = 1, chi_ref=None):
        h = h.expand(n_samples, *h.shape)
        chi_mask = chi_mask.expand(n_samples, *chi_mask.shape)
        if chi_ref is not None:
            chi_ref = chi_ref.expand(*h.shape[:-1], rc.MAX_CHI, 2)
        prev = torch.zeros(*h.shape[:-1], rc.MAX_CHI, 2, device=h.device, dtype=h.dtype)
        for k in range(rc.MAX_CHI):
            logits, mu, kappa = self._params(h, prev, k, chi_ref)
            comp = torch.distributions.Categorical(logits=logits).sample()[..., None]
            m = torch.gather(mu, -1, comp).squeeze(-1)
            kp = torch.gather(kappa, -1, comp).squeeze(-1)
            x = torch.distributions.VonMises(m, kp).sample()
            prev[..., k, :] = torch.stack([torch.sin(x), torch.cos(x)], -1) * chi_mask[..., k, None]
        return prev


HEADS = {"deterministic": DeterministicChiHead, "vonmises": VonMisesMixtureHead}


# ----------------------------------------------------------------------------- model
class SideChainModel(nn.Module):
    """Shared side-chain model. ~31k parameters with defaults (deterministic head)."""

    def __init__(self, d_ctx=32, d_aa=16, d_hidden=80, head="deterministic", **head_kw):
        super().__init__()
        self.aa_emb = nn.Embedding(rc.NUM_RESTYPES, d_aa)
        d_in = 9 + d_aa + d_ctx + rc.MAX_CHI * 2  # local N,C,O (CA = origin) + type + context + prior chi
        self.inp = nn.Linear(d_in, d_hidden)
        self.block1 = mlp(d_hidden, d_hidden, d_hidden)
        self.block2 = mlp(d_hidden, d_hidden, d_hidden)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.head: SideChainHead = HEADS[head](d_hidden, **head_kw)
        self.builder = SideChainBuilder()

    def features(self, backbone, aatype, ctx, chi_ref=None):
        R, t = frames_from_backbone(backbone[..., 0, :], backbone[..., 1, :], backbone[..., 2, :])
        loc = to_local(backbone[..., [0, 2, 3], :], R, t).flatten(-2)  # [...,R,9]
        emb = self.aa_emb(aatype).expand(*backbone.shape[:-2], -1)
        ref = (chi_ref.expand(*backbone.shape[:-2], rc.MAX_CHI, 2).flatten(-2) if chi_ref is not None
               else loc.new_zeros(*backbone.shape[:-2], rc.MAX_CHI * 2))
        return torch.cat([loc, emb, ctx, ref], -1)

    def trunk(self, backbone, aatype, ctx, chi_ref=None):
        x = self.inp(self.features(backbone, aatype, ctx, chi_ref))
        x = self.norm1(x + self.block1(F.silu(x)))
        x = self.norm2(x + self.block2(F.silu(x)))
        return x

    def forward(self, backbone, aatype, ctx, chi_true=None, sample=False, n_samples=1, chi_ref=None):
        """backbone [B,R,4,3], aatype [R], ctx [B,R,d_ctx], chi_ref optional prior [R,4,2]."""
        h = self.trunk(backbone, aatype, ctx, chi_ref)
        chi_mask = self.builder.chi_mask[aatype].expand(*h.shape[:-1], rc.MAX_CHI)
        if chi_ref is not None:
            chi_ref = chi_ref.expand(*h.shape[:-1], rc.MAX_CHI, 2)
        out = self.head(h, chi_mask, chi_true, chi_ref=chi_ref)
        if sample:
            chi_sc = self.head.sample(h, chi_mask, n_samples, chi_ref=chi_ref)  # [S,B,R,4,2]
            bb = backbone.expand(n_samples, *backbone.shape)
            out["samples_atom14"], _ = self.builder(bb, aatype, chi_sc)
            out["samples_chi_sc"] = chi_sc
        atom14, mask = self.builder(backbone, aatype, out["chi_sc"])
        out.update(atom14=atom14, atom_mask=mask, chi_mask=chi_mask, h=h)
        return out
