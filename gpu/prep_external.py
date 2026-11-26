#!/usr/bin/env python
"""External test set = the 8 systems of the CPU study (AdK + 7 ATLAS proteins), converted to the same
npz format as the mdCATH domains (all frames used for evaluation, same strides as multi.py)."""
import os, subprocess, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from md_compress.data import load_trajectory  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/data/external"
os.makedirs(OUT, exist_ok=True)
systems = {"ADK_4AKE": (f"{ROOT}/data/adk.psf", f"{ROOT}/data/adk_dims.dcd", 1)}
for pid in ["1pch_A", "3cjs_B", "2ygs_A", "3czz_B", "3fp5_A", "2fip_A", "3klr_A"]:
    d = f"{ROOT}/data/atlas/{pid}"
    if not os.path.exists(f"{d}/{pid}_prod_R1_fit.xtc"):
        subprocess.run([sys.executable, f"{ROOT}/tools/fetch_atlas.py", pid, d], check=False)
    systems[pid] = (f"{d}/{pid}.pdb", f"{d}/{pid}_prod_R1_fit.xtc", 100)
for name, (top, traj, stride) in systems.items():
    path = f"{OUT}/{name}.npz"
    if os.path.exists(path) or not os.path.exists(traj):
        continue
    d = load_trajectory(top, traj, stride=stride)
    seq = "".join("ARNDCQEGHILKMFPSTWYVX"[min(i, 20)] for i in d.aatype)
    np.savez(path, coords=d.coords.astype(np.float32), atom_mask=d.atom_mask, aatype=d.aatype,
             chain_index=d.chain_index, chain_break=d.chain_break, n_heavy=d.n_heavy_atoms(), seq=seq)
    print(name, d.n_res, d.n_frames, flush=True)
