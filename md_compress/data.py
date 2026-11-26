"""Trajectory I/O with MDAnalysis.

* `load_trajectory`  -> TrajectoryData with atom14 coords [T,R,14,3], masks, restypes
* `TrajectoryData.write`  writes reconstructed atom14 frames back to PDB/DCD/XTC
  using the original topology (heavy atoms only; hydrogens are dropped).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import residue_constants as rc


@dataclass
class TrajectoryData:
    coords: np.ndarray        # [T,R,14,3] float32 (Å), zeros where missing
    atom_mask: np.ndarray     # [R,14] 1 if atom present in topology (and in template)
    aatype: np.ndarray        # [R] int64
    chain_index: np.ndarray   # [R] int64
    chain_break: np.ndarray   # [R] bool, residue not bonded to previous one
    resnames: list = field(default_factory=list)
    topology: str | None = None
    # mapping atom14 slot -> topology heavy-atom index (for writing); -1 if absent
    slot_to_atom: np.ndarray | None = None
    oxt_atoms: np.ndarray | None = None  # [R] topology index of OXT/OT2 or -1
    universe: object = None

    @property
    def n_frames(self):
        return self.coords.shape[0]

    @property
    def n_res(self):
        return self.coords.shape[1]

    def n_heavy_atoms(self):
        return int(self.atom_mask.sum() + (self.oxt_atoms >= 0).sum())

    def heavy_atom_indices(self):
        """Topology indices of all atoms this pipeline reconstructs (mapped slots + OXT), sorted."""
        idx = np.concatenate([self.slot_to_atom[self.slot_to_atom >= 0], self.oxt_atoms[self.oxt_atoms >= 0]])
        return np.sort(idx)

    def torch(self, device="cpu"):
        return dict(
            coords=torch.as_tensor(self.coords, device=device),
            atom_mask=torch.as_tensor(self.atom_mask, device=device),
            aatype=torch.as_tensor(self.aatype, device=device),
            chain_index=torch.as_tensor(self.chain_index, device=device),
            chain_break=torch.as_tensor(self.chain_break, device=device),
        )

    def write(self, atom14: np.ndarray, filename: str, oxt: np.ndarray | None = None):
        """Write [T,R,14,3] (or [R,14,3]) reconstructed coordinates using the original topology.
        Output contains the protein heavy atoms of the topology; format from the extension."""
        import MDAnalysis as mda

        if atom14.ndim == 3:
            atom14 = atom14[None]
        u = self.universe
        try:
            u.trajectory
        except AttributeError:  # topology-only universe: attach a dummy in-memory frame
            from MDAnalysis.coordinates.memory import MemoryReader
            u.load_new(np.zeros((1, u.atoms.n_atoms, 3), np.float32), format=MemoryReader)
        heavy = u.atoms[self.heavy_atom_indices()]
        sub = mda.Merge(heavy)
        # map original indices -> position in heavy group
        pos_of = {a.index: i for i, a in enumerate(heavy)}
        sel = np.where(self.slot_to_atom >= 0)
        tgt = np.array([pos_of[i] for i in self.slot_to_atom[sel]])
        oxt_res = np.where(self.oxt_atoms >= 0)[0]
        oxt_tgt = np.array([pos_of[i] for i in self.oxt_atoms[oxt_res]], dtype=int)
        with mda.Writer(filename, n_atoms=sub.atoms.n_atoms) as w:
            for f in range(atom14.shape[0]):
                xyz = np.zeros((sub.atoms.n_atoms, 3), np.float32)
                xyz[tgt] = atom14[f][sel]
                if len(oxt_res):
                    if oxt is not None:
                        xyz[oxt_tgt] = oxt[f if oxt.ndim == 3 else None][..., oxt_res, :].reshape(-1, 3)
                    else:  # crude fallback: copy O
                        xyz[oxt_tgt] = atom14[f][oxt_res, 3]
                sub.atoms.positions = xyz
                w.write(sub.atoms)
        return filename


def is_hydrogen(atom) -> bool:
    """Hydrogen by name (H*, 1HB, 2HG2, ...) or by element/mass."""
    n = atom.name.lstrip("0123456789")
    if n.startswith("H"):
        return True
    try:
        if str(atom.element).upper() in ("H", "D"):
            return True
    except Exception:
        pass
    return False


def load_trajectory(topology: str, trajectory: str | None = None, selection: str = "protein",
                    stride: int = 1, max_frames: int | None = None, start: int = 0,
                    coords: np.ndarray | None = None) -> TrajectoryData:
    """Load protein heavy atoms in atom14 layout. Solvent, ions, ligands and hydrogens are dropped
    (selection='protein' + standard residue filter + hydrogen filter). Accepts any MDAnalysis
    topology/trajectory pair (PDB/GRO/PSF/TPR + XTC/DCD/TRR/NC...), or a topology plus an in-memory `coords` array."""
    import MDAnalysis as mda

    u = mda.Universe(topology, trajectory) if trajectory else mda.Universe(topology)
    if coords is not None:  # in-memory coordinates [T, n_atoms_topology, 3] (e.g. mdCATH h5 datasets)
        from MDAnalysis.coordinates.memory import MemoryReader
        u.load_new(np.asarray(coords, np.float32), format=MemoryReader)
    prot = u.select_atoms(selection)
    # standard residues only (incl. force-field aliases); ligands / non-standard residues are skipped
    residues = prot.residues[[rc.restype_index(r.resname) != rc.UNK_INDEX for r in prot.residues]]
    R = len(residues)
    aatype = np.array([rc.restype_index(r.resname) for r in residues], np.int64)
    tables = rc.build_tables()
    atom_mask = np.zeros((R, rc.ATOM14), np.float32)
    slot_to_atom = -np.ones((R, rc.ATOM14), np.int64)
    oxt_atoms = -np.ones(R, np.int64)
    for i, res in enumerate(residues):
        rn = rc.RESTYPES[aatype[i]] if aatype[i] < len(rc.RESTYPES) else "GLY"
        names = rc.atom14_names(rn)
        slot = {n: s for s, n in enumerate(names) if n}
        for a in res.atoms:
            if is_hydrogen(a):
                continue
            if a.name in ("OT2", "OXT", "O2", "OC2"):
                oxt_atoms[i] = a.index
                continue
            cname = rc.canonical_atom_name(res.resname, a.name)
            if cname in slot:
                slot_to_atom[i, slot[cname]] = a.index
                atom_mask[i, slot[cname]] = 1.0
    atom_mask *= tables["atom_mask"][aatype]
    # chain bookkeeping
    seg = np.array([r.segment.segid for r in residues])
    _, chain_index = np.unique(seg, return_inverse=True)
    chain_break = np.zeros(R, bool)
    chain_break[0] = True
    chain_break[1:] = chain_index[1:] != chain_index[:-1]
    # gaps from skipped residues
    ri = np.array([r.resindex for r in residues])
    chain_break[1:] |= (ri[1:] - ri[:-1]) != 1

    frames = []
    idx = slot_to_atom.copy()
    valid = idx >= 0
    has_coords = hasattr(u, "trajectory") and getattr(u.trajectory, "n_frames", 0) > 0
    if has_coords:
        for k, ts in enumerate(u.trajectory[start::stride]):
            if max_frames and k >= max_frames:
                break
            x = np.zeros((R, rc.ATOM14, 3), np.float32)
            x[valid] = u.atoms.positions[idx[valid]]
            frames.append(x)
        coords = np.stack(frames)
        # geometric chain breaks (C(i-1)-N(i) > 2 Å in first frame)
        d = np.linalg.norm(coords[0, 1:, 0] - coords[0, :-1, 2], axis=-1)
        chain_break[1:] |= d > 2.0
    else:  # topology without coordinates (e.g. PSF only, for decompression)
        coords = np.zeros((0, R, rc.ATOM14, 3), np.float32)
    return TrajectoryData(coords=coords, atom_mask=atom_mask, aatype=aatype, chain_index=chain_index,
                          chain_break=chain_break, resnames=[r.resname for r in residues],
                          topology=topology, slot_to_atom=slot_to_atom, oxt_atoms=oxt_atoms, universe=u)


def split_frames(n_frames: int, test_every: int = 5):
    """Interleaved split: every `test_every`-th frame is held out (frames are time-correlated,
    so this measures interpolation, not generalisation to new conformational states)."""
    idx = np.arange(n_frames)
    test = idx[idx % test_every == test_every - 1]
    train = np.setdiff1d(idx, test)
    return train, test
