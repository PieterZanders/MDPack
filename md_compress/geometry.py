"""Differentiable geometry utilities (pure PyTorch): frames, dihedrals, NeRF
side-chain builder, Kabsch superposition, RBF features."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import residue_constants as rc

EPS = 1e-8


def normalize(v: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def frames_from_backbone(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor):
    """Gram-Schmidt residue frame. Returns rotation R [...,3,3] (columns = e1,e2,e3) and origin CA.
    e1 along CA->C, e2 in the N-CA-C plane, e3 = e1 x e2. Rotation-equivariant; local coords
    x_loc = R^T (x - CA) are rotation/translation invariant."""
    e1 = normalize(c - ca)
    u = n - ca
    e2 = normalize(u - (u * e1).sum(-1, keepdim=True) * e1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1), ca


def to_local(x: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x [...,A,3], R [...,3,3], t [...,3] -> local coords [...,A,3]."""
    return torch.einsum("...ij,...ai->...aj", R, x - t[..., None, :])


def to_global(x_loc: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...aj->...ai", R, x_loc) + t[..., None, :]


def dihedral_sincos(a, b, c, d):
    """Returns (sin, cos) of the dihedral a-b-c-d, shape [...,2]."""
    b0 = a - b
    b1 = normalize(c - b)
    b2 = d - c
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
    nrm = torch.sqrt(x * x + y * y + EPS)
    return torch.stack([y / nrm, x / nrm], dim=-1)


def dihedral(a, b, c, d):
    sc = dihedral_sincos(a, b, c, d)
    return torch.atan2(sc[..., 0], sc[..., 1])


def nerf_place(a, b, c, bond, angle, tor_sin, tor_cos):
    """Natural Extension Reference Frame: place D such that |CD|=bond, angle(BCD)=angle,
    dihedral(ABCD)=torsion. All inputs broadcast; returns [...,3]."""
    bc = normalize(c - b)
    nrm = normalize(torch.cross(b - a, bc, dim=-1))
    m = torch.cross(nrm, bc, dim=-1)
    d2x = -bond * torch.cos(angle)
    d2y = bond * torch.sin(angle) * tor_cos
    d2z = bond * torch.sin(angle) * tor_sin
    return c + d2x[..., None] * bc + d2y[..., None] * m + d2z[..., None] * nrm


class SideChainBuilder(nn.Module):
    """Deterministic, differentiable NeRF builder: (backbone, restype, chi sin/cos) -> atom14 coords.

    No learnable parameters: the ideal internal coordinates come from residue_constants.
    Any model that outputs chi (sin, cos) - deterministic MLP head or a sampled flow head -
    can reuse this builder unchanged.
    """

    def __init__(self):
        super().__init__()
        t = rc.build_tables()
        for k in ["atom_mask", "ref_idx", "bond", "angle", "chi_idx", "torsion", "chi_mask",
                  "chi_atom_idx", "chi_pi_periodic", "swap_perm", "vdw"]:
            self.register_buffer(k, torch.as_tensor(t[k]))  # persistent: calibration is saved

    def forward(self, backbone: torch.Tensor, aatype: torch.Tensor, chi_sc: torch.Tensor):
        """backbone [...,R,4,3] (N,CA,C,O); aatype [R] or [...,R]; chi_sc [...,R,4,2] (sin,cos).
        Returns atom14 [...,R,14,3] (unused slots zero) and mask [...,R,14]."""
        shape = backbone.shape[:-2]
        aat = aatype.expand(shape) if aatype.dim() < len(shape) else aatype
        atoms = [backbone[..., i, :] for i in range(4)]
        ref = self.ref_idx[aat]  # [...,R,14,3]
        bond, ang = self.bond[aat], self.angle[aat]
        chi_idx, tors = self.chi_idx[aat], self.torsion[aat]
        mask = self.atom_mask[aat]
        for s in range(4, rc.ATOM14):
            stack = torch.stack(atoms, dim=-2)  # [...,R,s,3]
            r = ref[..., s, :].clamp(max=s - 1)
            pa = torch.gather(stack, -2, r[..., 0, None, None].expand(*r.shape[:-1], 1, 3)).squeeze(-2)
            pb = torch.gather(stack, -2, r[..., 1, None, None].expand(*r.shape[:-1], 1, 3)).squeeze(-2)
            pc = torch.gather(stack, -2, r[..., 2, None, None].expand(*r.shape[:-1], 1, 3)).squeeze(-2)
            k = chi_idx[..., s]
            off = tors[..., s]
            chi = torch.gather(chi_sc, -2, k.clamp(min=0)[..., None, None].expand(*k.shape, 1, 2)).squeeze(-2)
            use_chi = (k >= 0).to(chi.dtype)
            # torsion = chi + offset (if chi-driven) else fixed value
            s_chi = chi[..., 0] * torch.cos(off) + chi[..., 1] * torch.sin(off)
            c_chi = chi[..., 1] * torch.cos(off) - chi[..., 0] * torch.sin(off)
            t_sin = use_chi * s_chi + (1 - use_chi) * torch.sin(off)
            t_cos = use_chi * c_chi + (1 - use_chi) * torch.cos(off)
            x = nerf_place(pa, pb, pc, bond[..., s], ang[..., s], t_sin, t_cos)
            atoms.append(x * mask[..., s, None])
        return torch.stack(atoms, dim=-2), mask

    def chis_from_atom14(self, atom14: torch.Tensor, aatype: torch.Tensor):
        """Ground-truth chi (sin,cos) [...,R,4,2] and chi mask [...,R,4] from coordinates."""
        shape = atom14.shape[:-2]
        aat = aatype.expand(shape) if aatype.dim() < len(shape) else aatype
        idx = self.chi_atom_idx[aat]  # [...,R,4,4]
        g = torch.gather(atom14[..., None, :, :].expand(*shape, 4, rc.ATOM14, 3), -2,
                         idx[..., None].expand(*idx.shape, 3))  # [...,R,4,4,3]
        sc = dihedral_sincos(g[..., 0, :], g[..., 1, :], g[..., 2, :], g[..., 3, :])
        m = self.chi_mask[aat]
        return sc * m[..., None], m


def kabsch(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor | None = None):
    """Optimal rotation/translation superposing P onto Q. P,Q [...,N,3], w [...,N].
    Returns R [...,3,3], t [...,3] with P_aligned = P @ R^T + t."""
    if w is None:
        w = torch.ones(P.shape[:-1], dtype=P.dtype, device=P.device)
    w = w[..., None]
    ws = w.sum(-2, keepdim=True).clamp(min=EPS)
    pc = (P * w).sum(-2, keepdim=True) / ws
    qc = (Q * w).sum(-2, keepdim=True) / ws
    H = ((P - pc) * w).transpose(-1, -2) @ (Q - qc)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    t = (qc - pc @ R.transpose(-1, -2)).squeeze(-2)
    return R, t


def rbf(d: torch.Tensor, d_min=0.0, d_max=12.0, n=16):
    mu = torch.linspace(d_min, d_max, n, device=d.device, dtype=d.dtype)
    sigma = (d_max - d_min) / n
    return torch.exp(-(((d[..., None] - mu) / sigma) ** 2))


def backbone_dihedrals(bb: torch.Tensor, chain_break: torch.Tensor):
    """phi, psi, omega (sin,cos) per residue with validity mask.
    bb [...,R,4,3]; chain_break [R] bool: True if residue i is NOT bonded to i-1.
    Returns feats [...,R,6], mask [...,R,3]."""
    N, CA, C = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
    Cp = torch.roll(C, 1, dims=-2)
    Nn = torch.roll(N, -1, dims=-2)
    CAn = torch.roll(CA, -1, dims=-2)
    phi = dihedral_sincos(Cp, N, CA, C)
    psi = dihedral_sincos(N, CA, C, Nn)
    omg = dihedral_sincos(CA, C, Nn, CAn)
    prev_ok = (~chain_break).float()
    next_ok = torch.roll(prev_ok, -1, dims=-1)
    next_ok[..., -1] = 0.0
    prev_ok = prev_ok.clone()
    prev_ok[..., 0] = 0.0
    m = torch.stack([prev_ok, next_ok, next_ok], -1).expand(*bb.shape[:-2], 3)
    feats = torch.cat([phi * m[..., 0:1], psi * m[..., 1:2], omg * m[..., 2:3]], -1)
    return feats, m


def place_oxt(bb: torch.Tensor):
    """Terminal OXT from N, CA, C, O (planar carboxylate, opposite O)."""
    N, CA, C, O = bb.unbind(-2)
    psi_o = dihedral_sincos(N, CA, C, O)
    return nerf_place(N, CA, C, torch.tensor(1.25), torch.tensor(math.radians(117.0)),
                      -psi_o[..., 0], -psi_o[..., 1])


@torch.no_grad()
def calibrate_builder(builder: SideChainBuilder, atom14: torch.Tensor, aatype: torch.Tensor,
                      atom_mask: torch.Tensor):
    """Optional: replace ideal bond/angle/torsion-offset tables by their mean values measured on
    reference coordinates (e.g. the training frames of this force field). Circular means are used
    for torsions/offsets. Residue types absent from the data keep the ideal values.
    atom14 [T,R,14,3]; aatype [R]; atom_mask [R,14]."""
    chi, _ = builder.chis_from_atom14(atom14, aatype)  # [T,R,4,2]
    for ri in range(rc.NUM_RESTYPES):
        sel = (aatype == ri).nonzero().flatten()
        if len(sel) == 0:
            continue
        x = atom14[:, sel]  # [T,n,14,3]
        for s in range(4, rc.ATOM14):
            if builder.atom_mask[ri, s] == 0 or atom_mask[sel, s].min() == 0:
                continue
            a, b, c = builder.ref_idx[ri, s].tolist()
            pa, pb, pc, pd = x[..., a, :], x[..., b, :], x[..., c, :], x[..., s, :]
            builder.bond[ri, s] = (pd - pc).norm(dim=-1).mean()
            v1, v2 = normalize(pb - pc), normalize(pd - pc)
            builder.angle[ri, s] = torch.acos((v1 * v2).sum(-1).clamp(-1, 1)).mean()
            tor = dihedral(pa, pb, pc, pd)
            k = builder.chi_idx[ri, s].item()
            if k >= 0:
                ch = torch.atan2(chi[:, sel, k, 0], chi[:, sel, k, 1])
                tor = tor - ch
            builder.torsion[ri, s] = torch.atan2(torch.sin(tor).mean(), torch.cos(tor).mean())
    return builder
