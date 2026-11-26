"""Losses and metrics: backbone, side-chain (local-frame, alignment-free, symmetry-aware),
chi (symmetry-aware), soft clash, plus evaluation metrics."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import residue_constants as rc
from .geometry import SideChainBuilder, frames_from_backbone, kabsch, to_local

_TABLES = None


def tables(device="cpu"):
    global _TABLES
    if _TABLES is None:
        b = SideChainBuilder()
        _TABLES = {k: getattr(b, k) for k in ["swap_perm", "chi_pi_periodic", "vdw", "chi_mask", "atom_mask"]}
    return {k: v.to(device) for k, v in _TABLES.items()}


def backbone_loss(pred_bb, true_bb):
    """MSE (Å^2) over N, CA, C, O in the aligned frame."""
    return ((pred_bb - true_bb) ** 2).sum(-1).mean()


def _swapped(x, aatype):
    perm = tables(x.device)["swap_perm"][aatype]  # [R,14]
    return torch.gather(x, -2, perm.expand(*x.shape[:-1])[..., None].expand(*x.shape))


def sidechain_local_sq_err(pred, true, pred_bb, true_bb, aatype, mask):
    """Per-residue mean squared error of side-chain atoms, each expressed in its OWN residue frame
    (pred in predicted frame, truth in true frame) -> alignment-free. Min over symmetric swaps.
    Returns err [B,R] and per-residue sc atom counts [R]."""
    Rp, tp = frames_from_backbone(pred_bb[..., 0, :], pred_bb[..., 1, :], pred_bb[..., 2, :])
    Rt, tt = frames_from_backbone(true_bb[..., 0, :], true_bb[..., 1, :], true_bb[..., 2, :])
    lp = to_local(pred, Rp, tp)
    lt = to_local(true, Rt, tt)
    lt_sw = to_local(_swapped(true, aatype), Rt, tt)
    m = mask.clone()
    m[..., :4] = 0
    e1 = (((lp - lt) ** 2).sum(-1) * m).sum(-1)
    e2 = (((lp - lt_sw) ** 2).sum(-1) * m).sum(-1)
    return torch.minimum(e1, e2), m.sum(-1)


def sidechain_local_loss(pred, true, pred_bb, true_bb, aatype, mask):
    e, n = sidechain_local_sq_err(pred, true, pred_bb, true_bb, aatype, mask)
    return e.sum() / (n.sum() * e.shape[0]).clamp(min=1)


def sidechain_global_loss(pred, true, aatype, mask):
    """Squared error of side-chain atoms in the common (aligned) frame, symmetry-aware."""
    m = mask.clone()
    m[..., :4] = 0
    e1 = (((pred - true) ** 2).sum(-1) * m).sum(-1)
    e2 = (((pred - _swapped(true, aatype)) ** 2).sum(-1) * m).sum(-1)
    return torch.minimum(e1, e2).sum() / (m.sum() * pred.shape[0]).clamp(min=1)


def canonicalize_chi(chi_sc, aatype):
    """Map pi-periodic chis to cos >= 0 half-circle (used as flow-head targets)."""
    per = tables(chi_sc.device)["chi_pi_periodic"][aatype][..., None]
    flip = (chi_sc[..., 1:2] < 0).float() * per
    return chi_sc * (1 - 2 * flip)


def chi_loss(pred_sc, true_sc, aatype, chi_mask):
    """1 - cos(delta), pi-periodic chis use 1 - |cos(delta)|."""
    cosd = (pred_sc * true_sc).sum(-1)
    per = tables(pred_sc.device)["chi_pi_periodic"][aatype]
    cosd = torch.where(per > 0, cosd.abs(), cosd)
    return ((1 - cosd) * chi_mask).sum() / chi_mask.sum().clamp(min=1)


def chi_abs_err_deg(pred_sc, true_sc, aatype):
    cosd = (pred_sc * true_sc).sum(-1)
    sind = pred_sc[..., 0] * true_sc[..., 1] - pred_sc[..., 1] * true_sc[..., 0]
    d = torch.atan2(sind, cosd).abs()
    per = tables(pred_sc.device)["chi_pi_periodic"][aatype]
    d = torch.where(per > 0, torch.minimum(d, math.pi - d), d)
    return torch.rad2deg(d)


def clash_pairs(ca, chain_index, cutoff=14.0):
    """Residue pairs (b,i,j), i<j, CA-CA < cutoff, excluding sequence neighbours (|i-j|<=1, same chain)."""
    B, R, _ = ca.shape
    d = torch.cdist(ca, ca)
    idx = torch.arange(R, device=ca.device)
    same = chain_index[:, None] == chain_index[None, :]
    near = (idx[None, :] - idx[:, None]).abs() <= 1
    ok = (d < cutoff) & (idx[:, None] < idx[None, :]) & ~(same & near)
    return ok.nonzero(as_tuple=True)


def clash_terms(atom14, aatype, mask, chain_index, tol=1.5, pairs=None):
    """Inter-residue heavy-atom overlaps: overlap = r_a + r_b - tol - d (pairs where at least one
    atom is a side-chain atom; CYS SG-SG excluded to allow disulfides).
    Returns overlap tensor [P,14,14] (>=0) and validity mask."""
    vdw = tables(atom14.device)["vdw"][aatype]  # [R,14]
    B, R = atom14.shape[:2]
    if pairs is None:
        pairs = clash_pairs(atom14[..., 1, :].detach(), chain_index)
    b, i, j = pairs
    mask = mask.expand(B, R, rc.ATOM14)
    xi, xj = atom14[b, i], atom14[b, j]
    d = torch.sqrt(((xi[:, :, None] - xj[:, None]) ** 2).sum(-1) + 1e-8)
    sc_i = torch.arange(rc.ATOM14, device=atom14.device) >= 4
    valid = (mask[b, i][:, :, None] * mask[b, j][:, None]) > 0
    valid &= sc_i[None, :, None] | sc_i[None, None, :]
    cys = rc.RESTYPE_ORDER["CYS"]
    ss = (aatype[i] == cys) & (aatype[j] == cys)
    valid[:, 4, 4] &= ~ss
    ov = F.relu(vdw[i][:, :, None] + vdw[j][:, None] - tol - d) * valid
    return ov, valid


def clash_loss(atom14, aatype, mask, chain_index, tol=1.5):
    ov, _ = clash_terms(atom14, aatype, mask, chain_index, tol)
    return (ov ** 2).sum() / atom14.shape[0] / max(atom14.shape[1], 1)


@torch.no_grad()
def clash_score(atom14, aatype, mask, chain_index, overlap=0.4):
    """Number of heavy-atom pairs overlapping by more than `overlap` Å (vdW radii, no tolerance)
    per 1000 heavy atoms per frame."""
    ov, valid = clash_terms(atom14, aatype, mask, chain_index, tol=0.0)
    n = (ov > overlap).sum().item()
    n_atoms = mask.expand(atom14.shape[:3]).sum().item()
    return 1000.0 * n / max(n_atoms, 1)


@torch.no_grad()
def evaluate(pred_atom14, true_atom14, pred_chi, true_chi, aatype, atom_mask, chi_mask, chain_index):
    """Metrics in Å / degrees. Frames are superposed on backbone (N,CA,C,O) first."""
    T, R = true_atom14.shape[:2]
    m = atom_mask.expand(T, R, rc.ATOM14)
    bbm = m[..., :4].reshape(T, -1)
    Rk, tk = kabsch(pred_atom14[:, :, :4].reshape(T, -1, 3), true_atom14[:, :, :4].reshape(T, -1, 3), bbm)
    P = (pred_atom14.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(pred_atom14.shape)
    d2 = ((P - true_atom14) ** 2).sum(-1)
    # symmetry-aware side-chain error
    d2_sw = ((P - _swapped(true_atom14, aatype)) ** 2).sum(-1)
    scm = m.clone()
    scm[..., :4] = 0
    res_e = torch.minimum((d2 * scm).sum(-1), (d2_sw * scm).sum(-1))
    bb_rmsd = ((d2[..., :4] * m[..., :4]).sum((1, 2)) / m[..., :4].sum((1, 2))).sqrt()
    sc_rmsd = (res_e.sum(-1) / scm.sum((1, 2))).sqrt()
    all_rmsd = (((d2[..., :4] * m[..., :4]).sum((1, 2)) + res_e.sum(-1)) / m.sum((1, 2))).sqrt()
    loc_e, n = sidechain_local_sq_err(pred_atom14, true_atom14, pred_atom14[:, :, :4], true_atom14[:, :, :4],
                                      aatype, atom_mask)
    sc_local = (loc_e.sum(-1) / n.sum()).sqrt()
    err = chi_abs_err_deg(pred_chi, true_chi, aatype)
    cm = chi_mask.expand(err.shape)
    chi_mae = [((err[..., k] * cm[..., k]).sum() / cm[..., k].sum().clamp(min=1)).item() for k in range(4)]
    chi1_acc = (((err[..., 0] < 40) * cm[..., 0]).sum() / cm[..., 0].sum()).item()
    return dict(
        backbone_rmsd=bb_rmsd.mean().item(),
        sidechain_rmsd=sc_rmsd.mean().item(),
        sidechain_local_rmsd=sc_local.mean().item(),
        all_heavy_rmsd=all_rmsd.mean().item(),
        chi_mae_deg=chi_mae,
        chi1_acc_40deg=chi1_acc,
        clash_per_1000_atoms=clash_score(P, aatype, atom_mask, chain_index),
    )
