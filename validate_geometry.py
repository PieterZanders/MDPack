"""Oracle test of the ideal-geometry NeRF table: rebuild side chains from the TRUE backbone and the
TRUE chi angles. Any remaining error is the floor set by ideal bond lengths/angles/fixed torsions."""
import sys
import numpy as np
import torch
from md_compress.data import load_trajectory
from md_compress.geometry import SideChainBuilder
from md_compress import residue_constants as rc

top = sys.argv[1] if len(sys.argv) > 1 else "data/adk.psf"
trj = sys.argv[2] if len(sys.argv) > 2 else "data/adk_dims.dcd"
d = load_trajectory(top, trj, max_frames=10)
b = SideChainBuilder()
x = torch.as_tensor(d.coords)
aat = torch.as_tensor(d.aatype)
m = torch.as_tensor(d.atom_mask)
chi, cm = b.chis_from_atom14(x, aat)
rec, _ = b(x[..., :4, :], aat, chi)
err = ((rec - x) ** 2).sum(-1).sqrt() * m  # [T,R,14]
sc = m.clone(); sc[:, :4] = 0
print(f"oracle side-chain RMSD (true backbone + true chi): {((err**2*sc).sum()/ (sc.sum()*x.shape[0])).sqrt():.3f} Å")
for ri, rn in enumerate(rc.RESTYPES):
    sel = (aat == ri)
    if sel.sum() == 0 or rn in ("GLY",):
        continue
    e = err[:, sel][..., 4:]; mm = sc[sel][:, 4:]
    per_atom = ((e**2 * mm).sum((0, 1)) / (mm.sum(0) * x.shape[0]).clamp(min=1)).sqrt()
    names = rc.atom14_names(rn)[4:]
    print(f"{rn}: max {per_atom.max():.3f} Å  " + " ".join(f"{n}:{v:.2f}" for n, v in zip(names, per_atom) if n))

from md_compress.geometry import calibrate_builder
calibrate_builder(b, x, aat, m)
rec, _ = b(x[..., :4, :], aat, chi)
err = ((rec - x) ** 2).sum(-1).sqrt() * m
print(f"after calibrate_builder on these frames: {((err**2*sc).sum()/ (sc.sum()*x.shape[0])).sqrt():.3f} Å")
