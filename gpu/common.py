"""Shared helpers for the multi-protein GPU runs (mdCATH).

One set of shared weights is trained across many proteins. Per-protein *stored* data (reference
backbone, its scale, mean rotamers = chi prior) live in buffers of the codec; `bind(model, dom)`
swaps them in before each batch, so a single MDCompressor serves every topology.
"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from md_compress import residue_constants as rc  # noqa: E402
from md_compress.compression import MDCompressor  # noqa: E402
from md_compress.geometry import SideChainBuilder, kabsch  # noqa: E402
from md_compress.losses import canonicalize_chi  # noqa: E402

VARIANTS = {
    # side-chain stack as originally specified (<100k params)
    "S": dict(n_refine=2),
    # larger capacity probe (~0.74M params)
    "L": dict(n_refine=2, d_ctx=64, mapping_kw=dict(d_node=128, d_edge=64, n_layers=4),
              sidechain_kw=dict(d_hidden=256, d_aa=32)),
}
BUFFERS = ("codec.ref", "codec.scale", "codec.chi_ref")


def build_model(variant="S", prior=True, device="cuda"):
    m = MDCompressor(1, codec="residue", codec_kw=dict(latent_dim=4), packer_kw=VARIANTS[variant], use_chi_prior=prior)
    return m.to(device)


def weights(model):
    """state_dict without the per-protein buffers (those are stored per topology)."""
    return {k: v for k, v in model.state_dict().items() if k not in BUFFERS}


def load_weights(model, sd, prefix=None):
    sd = {k: v for k, v in sd.items() if k not in BUFFERS and (prefix is None or k.startswith(prefix))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


class Domain:
    def __init__(self, name, aatype, atom_mask, chain_index, chain_break, n_heavy, device, coords=None, off=None, T=None):
        self.name, self.device = name, device
        self.aatype = torch.as_tensor(aatype, device=device).long()
        self.atom_mask = torch.as_tensor(atom_mask, device=device).float()
        self.chain_index = torch.as_tensor(chain_index, device=device).long()
        self.chain_break = torch.as_tensor(chain_break, device=device).bool()
        self.n_heavy = int(n_heavy)
        self.R = int(self.aatype.shape[0])
        self.coords, self.off, self.T = coords, off, T
        if coords is not None and T is None:
            self.T = coords.shape[0]
        self.ref = self.scale = self.chi_ref = None
        self.chi_mask = None

    def frames(self, idx, mm=None):
        idx = np.sort(np.asarray(idx))
        if self.coords is not None:
            x = self.coords[idx]
        else:
            rows = (self.off + idx[:, None] * self.R + np.arange(self.R)[None]).reshape(-1)
            x = np.asarray(mm[rows]).reshape(len(idx), self.R, 14, 3)
        return torch.as_tensor(np.ascontiguousarray(x), device=self.device)


@torch.no_grad()
def fit_buffers(dom, x, builder, n_iter=3):
    """Per-protein stored data from the protein's own trajectory x [T,R,14,3] (as at compression time)."""
    bb = x[:, :, :4]
    T, R = bb.shape[:2]

    def align(ref):
        Rk, tk = kabsch(bb[:, :, 1], ref[None, :, 1].expand(T, R, 3))
        return (bb.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(bb.shape)

    ref = bb[0]
    for _ in range(n_iter):
        ref = align(ref).mean(0)
    al = align(ref)
    dom.ref = ref
    dom.scale = (al - ref).std().clamp(min=0.1)
    chi, chi_mask = builder.chis_from_atom14(x, dom.aatype)
    m = F.normalize(canonicalize_chi(chi, dom.aatype).mean(0), dim=-1)
    m = torch.where(chi_mask[0][..., None] > 0, m, torch.tensor([0.0, 1.0], device=x.device))
    dom.chi_ref, dom.chi_mask = m, chi_mask[0]


def bind(model, dom):
    c = model.codec
    c.n_res = dom.R
    c.ref, c.scale, c.chi_ref = dom.ref, dom.scale, dom.chi_ref


def load_npz_domain(path, device, max_frames=None, replica=None, stride=1):
    z = np.load(path)
    x = z["coords"]
    sel = np.arange(len(x))
    if replica is not None and "replica" in z:
        sel = sel[z["replica"] == replica]
    sel = sel[::stride]
    if max_frames:
        sel = sel[:max_frames]
    name = os.path.basename(path)[:-4]
    d = Domain(name, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], device,
               coords=x[sel].astype(np.float32))
    d.seq = str(z["seq"]) if "seq" in z else None
    return d, z


def type_mode_rotamers(chis, aatypes, chi_masks, n_bins=36):
    """No-per-protein-storage baseline: for each residue type and chi, the centre of the most populated
    10-degree bin over the TRAINING set (a one-rotamer library). Returns [21,4,2] (sin,cos)."""
    hist = torch.zeros(rc.NUM_RESTYPES, rc.MAX_CHI, n_bins)
    for chi, aat, cm in zip(chis, aatypes, chi_masks):
        c = canonicalize_chi(chi, aat).cpu()
        ang = torch.atan2(c[..., 0], c[..., 1])  # [T,R,4]
        b = ((ang + np.pi) / (2 * np.pi) * n_bins).long().clamp(0, n_bins - 1)
        w = cm.cpu().expand_as(b)
        a = aat.cpu()[None, :, None].expand_as(b)
        k = torch.arange(rc.MAX_CHI)[None, None].expand_as(b)
        hist.index_put_((a.reshape(-1), k.reshape(-1), b.reshape(-1)), w.reshape(-1).float(), accumulate=True)
    centre = (hist.argmax(-1).float() + 0.5) / n_bins * 2 * np.pi - np.pi
    out = torch.stack([torch.sin(centre), torch.cos(centre)], -1)
    return out
