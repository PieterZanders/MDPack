"""Mapping / packing network: residue-graph GNN (pure PyTorch message passing).

Graph: nodes = residues, edges = pairs with CA-CA distance < `cutoff` (10 Å), built per
frame. All features are SE(3)-invariant (distances via RBF, neighbour positions and
orientations expressed in the receiving residue's local frame), so outputs are invariant
and the NeRF builder makes the final coordinates equivariant.

Iterative refinement ("recycling"):
  pass 0 : backbone + sequence only (side-chain features are zeros)
  pass 1..n : the same network additionally sees the previous pass's side chains:
              predicted chi (sin,cos), side-chain centroid in the local frame (node features)
              and centroid-centroid distance + minimum heavy-atom distance between the
              side chain of i and all atoms of j (edge features -> packing awareness).
Weights are shared across passes (a pass embedding tells the net which pass it is), so
refinement costs no extra parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import residue_constants as rc
from .geometry import backbone_dihedrals, frames_from_backbone, rbf, to_local
from .sidechain_model import SideChainModel, mlp


def build_radius_graph(ca: torch.Tensor, cutoff: float = 10.0, max_neighbors: int | None = None):
    """ca [B,R,3] -> edge_index (src j, dst i) over flattened B*R nodes, plus (b, i, j)."""
    B, R, _ = ca.shape
    d = torch.cdist(ca, ca)
    adj = (d < cutoff) & ~torch.eye(R, dtype=torch.bool, device=ca.device)
    if max_neighbors is not None and max_neighbors < R - 1:
        kth = d.masked_fill(~adj, float("inf")).topk(max_neighbors, dim=-1, largest=False).values[..., -1:]
        adj &= d <= kth
    b, i, j = adj.nonzero(as_tuple=True)
    return b, i, j


class MPLayer(nn.Module):
    """h_i <- LN(h_i + phi_u([h_i, mean_j phi_m([h_i, h_j, e_ij])]))."""

    def __init__(self, d, d_edge):
        super().__init__()
        self.msg = mlp(2 * d + d_edge, d, d)
        self.upd = mlp(2 * d, d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, src, dst, e):
        m = self.msg(torch.cat([h[dst], h[src], e], -1))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        deg = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype).index_add_(
            0, dst, torch.ones_like(m[:, :1]))
        agg = agg / deg.clamp(min=1)
        return self.norm(h + self.upd(torch.cat([h, agg], -1)))


class MappingNet(nn.Module):
    N_RBF = 16
    N_SC_RBF = 8
    MAX_SEP = 16

    def __init__(self, d_node=48, d_edge=32, d_ctx=32, n_layers=3, cutoff=10.0, n_passes_max=4,
                 max_neighbors=None):
        super().__init__()
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        self.aa_emb = nn.Embedding(rc.NUM_RESTYPES, 16)
        self.pass_emb = nn.Embedding(n_passes_max, 8)
        d_node_in = 16 + 9 + 6 + 3 + 8 + (rc.MAX_CHI * 2 + 3)  # type, local bb, dihedrals, masks, pass, sc
        self.node_in = nn.Linear(d_node_in, d_node)
        self.sep_emb = nn.Embedding(2 * self.MAX_SEP + 2, 8)
        d_edge_in = self.N_RBF + 3 + 9 + 8 + 2 * self.N_SC_RBF + 3
        self.edge_in = nn.Linear(d_edge_in, d_edge)
        self.layers = nn.ModuleList([MPLayer(d_node, d_edge) for _ in range(n_layers)])
        self.out = nn.Linear(d_node, d_ctx)

    def forward(self, backbone, aatype, chain_index, chain_break, prev=None, pass_idx=0):
        """backbone [B,R,4,3]; prev = dict(atom14 [B,R,14,3], atom_mask [R,14], chi_sc [B,R,4,2]) or None.
        Returns context vectors [B,R,d_ctx]."""
        B, R = backbone.shape[:2]
        N, CA, C = backbone[..., 0, :], backbone[..., 1, :], backbone[..., 2, :]
        Rot, t = frames_from_backbone(N, CA, C)
        loc_bb = to_local(backbone[..., [0, 2, 3], :], Rot, t).flatten(-2)
        dih, dmask = backbone_dihedrals(backbone, chain_break)
        if prev is None:
            sc_node = backbone.new_zeros(B, R, rc.MAX_CHI * 2 + 3)
            centroid = None
        else:
            m = prev["atom_mask"].clone()
            m[..., :4] = 0
            m = m.expand(B, R, rc.ATOM14)
            cnt = m.sum(-1, keepdim=True)
            centroid = ((prev["atom14"] * m[..., None]).sum(-2) / cnt.clamp(min=1))
            centroid = torch.where(cnt > 0, centroid, CA)  # GLY: centroid = CA
            cen_loc = to_local(centroid[..., None, :], Rot, t).squeeze(-2)
            sc_node = torch.cat([prev["chi_sc"].flatten(-2), cen_loc], -1)
        x = torch.cat([self.aa_emb(aatype).expand(B, R, -1), loc_bb, dih, dmask,
                       self.pass_emb.weight[pass_idx].expand(B, R, -1), sc_node], -1)
        h = self.node_in(x).reshape(B * R, -1)

        b, i, j = build_radius_graph(CA, self.cutoff, self.max_neighbors)
        dst, src = b * R + i, b * R + j
        Rf, tf = Rot.reshape(B * R, 3, 3), t.reshape(B * R, 3)
        vec = CA.reshape(B * R, 3)[src] - tf[dst]
        dist = vec.norm(dim=-1)
        dir_loc = torch.einsum("eij,ei->ej", Rf[dst], vec) / dist[:, None].clamp(min=1e-6)
        rel_rot = torch.einsum("eki,ekj->eij", Rf[dst], Rf[src]).flatten(-2)
        same_chain = chain_index[i] == chain_index[j]
        sep = (j - i).clamp(-self.MAX_SEP, self.MAX_SEP) + self.MAX_SEP
        sep = torch.where(same_chain, sep, torch.full_like(sep, 2 * self.MAX_SEP + 1))
        if centroid is None:
            sc_edge = vec.new_zeros(len(dst), 2 * self.N_SC_RBF + 3)
        else:
            cf = centroid.reshape(B * R, 3)
            cvec = cf[src] - cf[dst]
            cd = cvec.norm(dim=-1)
            cdir = torch.einsum("eij,ei->ej", Rf[dst], cvec) / cd[:, None].clamp(min=1e-6)
            # min distance between side chain of i and every atom of j (packing / clash signal)
            xa = prev["atom14"].reshape(B * R, rc.ATOM14, 3)
            ma = prev["atom_mask"].expand(B, R, rc.ATOM14).reshape(B * R, rc.ATOM14)
            msc = ma.clone()
            msc[:, :4] = 0
            dd = torch.cdist(xa[dst], xa[src])
            valid = (msc[dst][:, :, None] * ma[src][:, None, :]) > 0
            dmin = dd.masked_fill(~valid, 99.0).amin((-1, -2)).clamp(max=20.0)
            sc_edge = torch.cat([rbf(cd, 0, 16, self.N_SC_RBF), rbf(dmin, 0, 8, self.N_SC_RBF), cdir], -1)
        e = torch.cat([rbf(dist, 0, self.cutoff, self.N_RBF), dir_loc, rel_rot, self.sep_emb(sep), sc_edge], -1)
        e = self.edge_in(e)
        for layer in self.layers:
            h = layer(h, src, dst, e)
        return self.out(h).view(B, R, -1)


class SideChainPacker(nn.Module):
    """Mapping network + shared side-chain model with iterative refinement.

    forward(...) returns a list with one prediction dict per pass (last = final). Previous
    predictions are detached before being fed back (recycling-style), which keeps memory flat
    in the number of passes.
    """

    def __init__(self, n_refine=2, d_ctx=32, mapping_kw=None, sidechain_kw=None):
        super().__init__()
        self.n_refine = n_refine
        self.mapping = MappingNet(d_ctx=d_ctx, n_passes_max=max(4, n_refine + 1), **(mapping_kw or {}))
        self.sidechain = SideChainModel(d_ctx=d_ctx, **(sidechain_kw or {}))

    def forward(self, backbone, aatype, chain_index, chain_break, chi_true=None, n_refine=None,
                sample=False, n_samples=1, chi_ref=None):
        n_refine = self.n_refine if n_refine is None else n_refine
        outs, prev = [], None
        for p in range(n_refine + 1):
            ctx = self.mapping(backbone, aatype, chain_index, chain_break, prev=prev, pass_idx=p)
            last = p == n_refine
            out = self.sidechain(backbone, aatype, ctx, chi_true=chi_true, sample=sample and last,
                                 n_samples=n_samples, chi_ref=chi_ref)
            outs.append(out)
            prev = dict(atom14=out["atom14"].detach(), atom_mask=out["atom_mask"],
                        chi_sc=out["chi_sc"].detach())
        return outs
