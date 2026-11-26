"""Shared helpers for the benchmark: topology from PDB, identical metrics for every method, result I/O.

Metrics (all on the same heavy atoms, Angstrom, reconstruction `rec` vs original `x`, both [T,N,3]):
  rmsd_lab        per frame RMSD in the lab frame (no superposition), symmetry aware per residue
                  (ASP/GLU/PHE/TYR/ARG equivalent atom labels may be swapped), mean over frames; same definition as
                  report section 15.
  rmsd_fit        same, after a per frame least squares superposition of rec onto x on all heavy atoms (relevant for
                  methods that do not keep the global frame).
  bb_rmsd_lab     protein backbone N, CA, C, O (and nucleic backbone P, O5', C5', C4', C3', O3' when present).
  sc_rmsd_lab     all other atoms of polymer residues (protein side chains, nucleic sugars and bases), symmetry aware.
  other_rmsd_lab  non polymer atoms (lipids, ligands), when present.
  max_abs_err     max over frames, atoms, coordinates of |rec - x| (lab frame, no symmetry swap).
  rmsf_r          Pearson r of per atom RMSF (all heavy atoms, each trajectory fitted on its own iterative mean of
                  the representative atoms), as in section 15.
  rmsip10         root mean square inner product of the top 10 PCA modes of the representative atoms
                  (both trajectories superposed on the mean of the original), as in section 15.
Representative atoms: protein CA plus nucleic C4'; if none, all heavy atoms (capped at 2000 by striding).
"""
import json, os, time
import numpy as np
import torch

PROT_BB = {"N", "CA", "C", "O", "OXT"}
NUC_BB = {"P", "OP1", "OP2", "O1P", "O2P", "O5'", "C5'", "C4'", "C3'", "O3'"}
AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
      "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "LYN", "ASH", "GLH", "MSE", "NLE", "CYM"}
NUC = {"DA", "DC", "DG", "DT", "A", "C", "G", "U", "DA5", "DA3", "DC5", "DC3", "DG5", "DG3", "DT5", "DT3", "A5", "A3",
       "C5", "C3", "G5", "G3", "U5", "U3", "RA", "RC", "RG", "RU", "ADE", "CYT", "GUA", "THY", "URA"}
SWAPS = {"ASP": [("OD1", "OD2")], "GLU": [("OE1", "OE2")], "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
         "TYR": [("CD1", "CD2"), ("CE1", "CE2")], "ARG": [("NH1", "NH2")]}
BENCH = "/workspace/md_compress/benchmark"


def read_pdb_topology(path):
    names, resn, resk, chain = [], [], [], []
    last, rid = None, -1
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")): continue
        key = (ln[21], ln[22:27], ln[17:21].strip())
        if key != last: rid += 1; last = key
        names.append(ln[12:16].strip()); resn.append(ln[17:21].strip()); resk.append(rid); chain.append(ln[21])
    names, resn, resk = np.array(names), np.array(resn), np.array(resk)
    N = len(names)
    prot = np.array([r in AA for r in resn]); nuc = np.array([r in NUC for r in resn])
    bb = (prot & np.isin(names, list(PROT_BB))) | (nuc & np.isin(names, list(NUC_BB)))
    sc = (prot | nuc) & ~bb; other = ~(prot | nuc)
    swap = np.arange(N)
    for r in np.unique(resk[prot]):
        idx = np.where(resk == r)[0]; rn = resn[idx[0]]
        for a, b in SWAPS.get(rn, []):
            ia, ib = idx[names[idx] == a], idx[names[idx] == b]
            if len(ia) and len(ib): swap[ia[0]], swap[ib[0]] = ib[0], ia[0]
    rep = (prot & (names == "CA")) | (nuc & (names == "C4'"))
    if rep.sum() < 3: rep = np.ones(N, bool)
    ri = np.where(rep)[0]
    if len(ri) > 2000: rep = np.zeros(N, bool); rep[ri[:: int(np.ceil(len(ri) / 2000))]] = True
    return dict(names=names, resn=resn, resk=resk, prot=prot, nuc=nuc, bb=bb, sc=sc, other=other, swap=swap, rep=rep,
                n_atoms=N, n_res=int(resk.max() + 1))


def kabsch_t(P, Q, w=None):
    """rotation R, translation t with P @ R.T + t ~ Q, batched [T,N,3] (torch float64)."""
    if w is None: w = torch.ones(P.shape[1], dtype=P.dtype)
    w = (w / w.sum())[None, :, None]
    pc = (P * w).sum(1, keepdim=True); qc = (Q * w).sum(1, keepdim=True)
    H = ((P - pc) * w).transpose(1, 2) @ (Q - qc)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = qc - pc @ R.transpose(1, 2)
    return R, t


def superpose(P, Q, w=None):
    R, t = kabsch_t(P, Q, w); return P @ R.transpose(1, 2) + t


def iter_mean(X, n=5):
    ref = X[0]
    for _ in range(n): ref = superpose(X, ref[None].expand_as(X)).mean(0)
    return ref


def _sym_sq(rec, x, top):
    """per atom squared error [T,N], symmetry aware per residue (min over the two labellings)."""
    e1 = ((rec - x) ** 2).sum(-1)
    e2 = ((rec[:, top["swap"]] - x) ** 2).sum(-1)
    if (top["swap"] == np.arange(top["n_atoms"])).all(): return e1
    rk = torch.as_tensor(top["resk"]); R = top["n_res"]
    s1 = torch.zeros(e1.shape[0], R, dtype=e1.dtype).index_add_(1, rk, e1)
    s2 = torch.zeros(e1.shape[0], R, dtype=e1.dtype).index_add_(1, rk, e2)
    use = (s2 < s1)[:, rk]
    return torch.where(use, e2, e1)


@torch.no_grad()
def metrics(rec, x, top, chunk=250):
    """rec, x numpy [T,N,3] (Angstrom)."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64); rec = torch.as_tensor(np.asarray(rec), dtype=torch.float64)
    T, N, _ = x.shape
    out = {}
    lab, fit, groups = [], [], {k: [] for k in ("bb", "sc", "other")}
    maxe = 0.0
    for s in range(0, T, chunk):
        xr, rr = x[s:s + chunk], rec[s:s + chunk]
        e = _sym_sq(rr, xr, top); lab.append(e.mean(1).sqrt())
        for g in groups:
            m = torch.as_tensor(top[g])
            if m.any(): groups[g].append(e[:, m].mean(1).sqrt())
        rf = superpose(rr, xr); fit.append(_sym_sq(rf, xr, top).mean(1).sqrt())
        maxe = max(maxe, float((rr - xr).abs().max()))
    out["rmsd_lab"] = float(torch.cat(lab).mean()); out["rmsd_fit"] = float(torch.cat(fit).mean())
    for g, v in groups.items():
        if v: out[f"{g}_rmsd_lab"] = float(torch.cat(v).mean())
    out["max_abs_err"] = maxe
    rep = torch.as_tensor(top["rep"])

    def rmsf(X):
        ref = iter_mean(X[:, rep]); R, t = kabsch_t(X[:, rep], ref[None].expand(T, -1, 3))
        Xf = X @ R.transpose(1, 2) + t
        return ((Xf - Xf.mean(0)) ** 2).sum(-1).mean(0).sqrt()
    ft, fr = rmsf(x), rmsf(rec)
    out["rmsf_r"] = float(np.corrcoef(ft.numpy(), fr.numpy())[0, 1]); out["rmsf_mae"] = float((ft - fr).abs().mean())
    mt = iter_mean(x[:, rep]); A = superpose(x[:, rep], mt[None].expand(T, -1, 3)); B = superpose(rec[:, rep], mt[None].expand(T, -1, 3))

    def pcs(F_):
        Z = (F_ - F_.mean(0)).reshape(T, -1)
        U, S, Vt = torch.linalg.svd(Z, full_matrices=False)
        return Vt[:10].T
    Vt, Vr = pcs(A), pcs(B)
    out["rmsip10"] = float((((Vt.T @ Vr) ** 2).sum() / 10).sqrt())
    return out


class Results:
    """incremental JSON store: key -> record."""
    def __init__(self, path):
        self.path = path; self.d = json.load(open(path)) if os.path.exists(path) else {}

    def has(self, k): return k in self.d

    def put(self, k, v):
        self.d[k] = v; tmp = self.path + ".tmp"; json.dump(self.d, open(tmp, "w"), indent=1); os.replace(tmp, self.path)


def fsize(*paths): return int(sum(os.path.getsize(p) for p in paths))


class Timer:
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): self.s = time.perf_counter() - self.t
