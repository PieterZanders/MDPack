#!/usr/bin/env python3
"""Export the benchmark test set: 16 mdCATH test domains, first T=1000 archived frames (320 K replicas first, same
archive order as report sections 14 and 15), protein heavy atoms in atom14 order (no hydrogens, no OXT).
Writes per domain: <d>.pdb (topology, frame 0), <d>_T1000.npy float32 [T,N,3] in Angstrom (the ground truth),
<d>_meta.json. T=250 uses the first 250 frames of the same array."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/workspace/md_compress")
from md_compress import residue_constants as rc
SRC = "/workspace/md_compress_gpu_data/data_long"; OUT = "/workspace/bench_data"; T = 1000
doms = json.load(open("/workspace/md_compress/results_gpu/frames/selected_domains.json"))
ELEM = lambda n: n[0]
info = {}
for d in doms:
    z = np.load(f"{SRC}/{d}.npz")
    arch = np.where(z["heldout"] == 0)[0][:T]
    X = z["coords"][arch]; mk = z["atom_mask"] > 0; aat = z["aatype"]; ch = z["chain_index"]
    temps = z["temp"][arch]; reps = z["replica"][arch]
    xyz = X[:, mk].astype(np.float32)  # [T,N,3]
    np.save(f"{OUT}/{d}_T{T}.npy", xyz)
    lines, k = [], 0
    for r in range(len(aat)):
        rn = rc.RESTYPES[aat[r]] if aat[r] < 20 else "UNK"
        names = rc.atom14_names(rn)
        for s in range(14):
            if not mk[r, s]: continue
            k += 1; x, y, zz = xyz[0, k - 1]
            lines.append(f"ATOM  {k:5d} {names[s]:<4s} {rn:3s} A{r+1:4d}    {x:8.3f}{y:8.3f}{zz:8.3f}  1.00  0.00          {ELEM(names[s]):>2s}")
    lines += ["TER", "END"]
    open(f"{OUT}/{d}.pdb", "w").write("\n".join(lines) + "\n")
    grid = float(np.abs(np.round(xyz * 100) - xyz * 100).max())
    info[d] = dict(n_res=int(len(aat)), n_heavy=int(mk.sum()), T=int(len(xyz)), temps=sorted(set(temps.tolist())),
                   replicas=sorted(set(reps.tolist())), max_dev_from_0p01A_grid_in_units=grid, n_chains=int(len(set(ch.tolist()))))
    print(d, info[d], flush=True)
json.dump(info, open(f"{OUT}/meta.json", "w"), indent=1)
