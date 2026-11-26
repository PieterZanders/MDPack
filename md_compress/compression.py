"""Compression workflow.

    frame  --align (Kabsch on CA to a reference)-->  aligned backbone  + rigid (quat 4 + trans 3)
           --Encoder-->  latent  (per residue: [R, d]   or   per frame: [d])
    latent --Decoder-->  aligned backbone (N, CA, C, O)
           --SideChainPacker (mapping GNN + shared side-chain model, refinement passes)-->
           full heavy-atom atom14  --undo rigid-->  original frame

Stored per frame: latents + 7 floats of rigid motion. Stored once: topology, reference backbone,
model weights (amortised over the trajectory).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import residue_constants as rc
from .geometry import kabsch, place_oxt
from .mapping_net import SideChainPacker


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat().astype(np.float32)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix().astype(np.float32)


class BackboneCodec(nn.Module):
    """Base: holds the reference backbone and the rigid alignment logic."""

    def __init__(self, n_res: int, latent_dim: int):
        super().__init__()
        self.n_res, self.latent_dim = n_res, latent_dim
        self.register_buffer("ref", torch.zeros(n_res, 4, 3))
        self.register_buffer("scale", torch.ones(()))
        # optional per-residue prior rotamers (stored once with the reference; see MDCompressor)
        chi_ref = torch.zeros(n_res, rc.MAX_CHI, 2)
        chi_ref[..., 1] = 1.0
        self.register_buffer("chi_ref", chi_ref)

    @torch.no_grad()
    def fit_reference(self, bb: torch.Tensor, n_iter: int = 3):
        """bb [T,R,4,3]: iterative mean structure (align on CA -> average)."""
        ref = bb[0]
        for _ in range(n_iter):
            aligned, _, _ = self._align_to(bb, ref)
            ref = aligned.mean(0)
        self.ref.copy_(ref)
        aligned, _, _ = self._align_to(bb, ref)
        self.scale.copy_((aligned - ref).std().clamp(min=0.1))

    @staticmethod
    def _align_to(bb, ref):
        T, R = bb.shape[:2]
        R_, t_ = kabsch(bb[:, :, 1], ref[None, :, 1].expand(T, R, 3))
        aligned = (bb.reshape(T, -1, 3) @ R_.transpose(-1, -2) + t_[:, None]).reshape(bb.shape)
        return aligned, R_, t_

    def align(self, bb):
        """-> aligned [T,R,4,3], rotation [T,3,3], translation [T,3]  (aligned = x R^T + t)."""
        return self._align_to(bb, self.ref)

    @staticmethod
    def unalign(x, R_, t_):
        """Inverse of align for any atom array x [T,...,3]."""
        sh = x.shape
        return ((x.reshape(sh[0], -1, 3) - t_[:, None]) @ R_).reshape(sh)

    def encode(self, aligned_bb):
        raise NotImplementedError

    def decode(self, z, aatype):
        raise NotImplementedError

    def floats_per_frame(self):
        raise NotImplementedError


class ResidueLatentCodec(BackboneCodec):
    """Per-residue latent [R, d]. 1-D convolutions along the sequence on displacements from the
    reference (aligned frame). Weights are size-independent (do not depend on R)."""

    def __init__(self, n_res, latent_dim=4, d_hidden=64, kernel=5, n_layers=3):
        super().__init__(n_res, latent_dim)
        self.aa = nn.Embedding(rc.NUM_RESTYPES, 8)
        pad = kernel // 2

        def stack(d_in, d_out):
            layers, d = [], d_in
            for _ in range(n_layers):
                layers += [nn.Conv1d(d, d_hidden, kernel, padding=pad), nn.SiLU()]
                d = d_hidden
            layers.append(nn.Conv1d(d, d_out, 1))
            return nn.Sequential(*layers)

        self.enc = stack(12 + 8 + 12, latent_dim)
        self.dec = stack(latent_dim + 8 + 12, 12)

    def _ctx(self, aatype, T):
        a = self.aa(aatype).T  # [8,R]
        r = (self.ref - self.ref[:, 1].mean(0)).reshape(self.n_res, 12).T / 10.0  # ref coords (nm-ish)
        return torch.cat([a, r], 0)[None].expand(T, -1, -1)

    def encode(self, aligned_bb, aatype):
        T = aligned_bb.shape[0]
        d = ((aligned_bb - self.ref) / self.scale).reshape(T, self.n_res, 12).transpose(1, 2)
        return self.enc(torch.cat([d, self._ctx(aatype, T)], 1)).transpose(1, 2)  # [T,R,d]

    def decode(self, z, aatype):
        T = z.shape[0]
        d = self.dec(torch.cat([z.transpose(1, 2), self._ctx(aatype, T)], 1)).transpose(1, 2)
        return self.ref + d.reshape(T, self.n_res, 4, 3) * self.scale

    def floats_per_frame(self):
        return self.n_res * self.latent_dim + 7


class FrameLatentCodec(BackboneCodec):
    """Per-frame latent [d] (nonlinear PCA-like MLP autoencoder). Much higher compression for long
    trajectories, but the weights scale with R and are specific to one topology."""

    def __init__(self, n_res, latent_dim=32, d_hidden=256):
        super().__init__(n_res, latent_dim)
        D = n_res * 12
        self.enc = nn.Sequential(nn.Linear(D, d_hidden), nn.SiLU(), nn.Linear(d_hidden, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, d_hidden), nn.SiLU(), nn.Linear(d_hidden, D))

    def encode(self, aligned_bb, aatype=None):
        T = aligned_bb.shape[0]
        return self.enc(((aligned_bb - self.ref) / self.scale).reshape(T, -1))

    def decode(self, z, aatype=None):
        return self.ref + self.dec(z).view(-1, self.n_res, 4, 3) * self.scale

    def floats_per_frame(self):
        return self.latent_dim + 7


CODECS = {"residue": ResidueLatentCodec, "frame": FrameLatentCodec}


class MDCompressor(nn.Module):
    """Full model = backbone codec + side-chain packer."""

    def __init__(self, n_res, codec="residue", codec_kw=None, packer_kw=None, use_chi_prior=True):
        """use_chi_prior: feed per-residue reference rotamers (circular mean of training chis, stored
        once like the reference backbone) to the side-chain model, which then predicts deviations.
        Set False for a generic packer that sees only backbone + sequence."""
        super().__init__()
        self.codec_type = codec
        self.use_chi_prior = use_chi_prior
        self.codec = CODECS[codec](n_res, **(codec_kw or {}))
        self.packer = SideChainPacker(**(packer_kw or {}))

    @torch.no_grad()
    def fit_chi_prior(self, chi_sc, chi_mask, aatype):
        """chi_sc [T,R,4,2] training chis -> circular mean (pi-periodic chis canonicalised first)."""
        from .losses import canonicalize_chi
        m = F.normalize(canonicalize_chi(chi_sc, aatype).mean(0), dim=-1)
        m = torch.where(chi_mask[..., None] > 0, m, torch.tensor([0.0, 1.0]))
        self.codec.chi_ref.copy_(m)

    def chi_prior(self):
        return self.codec.chi_ref if self.use_chi_prior else None

    def pack(self, bb, aatype, chain_index, chain_break, **kw):
        return self.packer(bb, aatype, chain_index, chain_break, chi_ref=self.chi_prior(), **kw)

    def decode_full(self, z, aatype, chain_index, chain_break, n_refine=None, chi_true=None):
        bb = self.codec.decode(z, aatype)
        outs = self.pack(bb, aatype, chain_index, chain_break, chi_true=chi_true, n_refine=n_refine)
        return bb, outs

    @torch.no_grad()
    def compress(self, atom14_or_bb, aatype, batch=64):
        """Frames [T,R,14|4,3] -> dict of numpy arrays to store (latents float16 + rigid)."""
        bb = atom14_or_bb[:, :, :4]
        zs, qs, ts = [], [], []
        for s in range(0, bb.shape[0], batch):
            aligned, R_, t_ = self.codec.align(bb[s:s + batch])
            zs.append(self.codec.encode(aligned, aatype))
            qs.append(rotmat_to_quat(R_.numpy()))
            ts.append(t_.numpy())
        return dict(latents=torch.cat(zs).numpy().astype(np.float16), quat=np.concatenate(qs),
                    trans=np.concatenate(ts).astype(np.float32))

    @torch.no_grad()
    def decompress(self, stored, aatype, chain_index, chain_break, batch=16, n_refine=None):
        """stored latents + rigid -> atom14 [T,R,14,3] in the ORIGINAL frame, and OXT [T,R,3]."""
        z_all = torch.as_tensor(stored["latents"].astype(np.float32))
        out_x, out_oxt = [], []
        for s in range(0, z_all.shape[0], batch):
            z = z_all[s:s + batch]
            _, outs = self.decode_full(z, aatype, chain_index, chain_break, n_refine=n_refine)
            x = outs[-1]["atom14"] * outs[-1]["atom_mask"][..., None]
            R_ = torch.as_tensor(quat_to_rotmat(stored["quat"][s:s + batch]))
            t_ = torch.as_tensor(stored["trans"][s:s + batch])
            x = self.codec.unalign(x, R_, t_)
            out_x.append(x)
            out_oxt.append(place_oxt(x[:, :, :4]))
        return torch.cat(out_x).numpy(), torch.cat(out_oxt).numpy()


def compression_report(n_heavy_atoms, codec: BackboneCodec, n_frames, n_model_params,
                       latent_bytes=2, n_res=None):
    """Compression ratio vs float32 full-atom heavy coordinates."""
    raw = n_heavy_atoms * 3 * 4
    lat = codec.latent_dim * (codec.n_res if isinstance(codec, ResidueLatentCodec) else 1)
    per_frame = lat * latent_bytes + 7 * 4
    overhead = n_model_params * 4 + codec.n_res * (12 + 8) * 4  # weights (fp32) + ref backbone + ref chis
    r = dict(raw_bytes_per_frame=raw, stored_bytes_per_frame=per_frame,
             ratio_per_frame=raw / per_frame,
             overhead_bytes=overhead,
             ratio_incl_model_this_traj=(raw * n_frames) / (per_frame * n_frames + overhead),
             ratio_incl_model_100k_frames=(raw * 1e5) / (per_frame * 1e5 + overhead))
    return r
