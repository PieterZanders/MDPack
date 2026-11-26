#!/usr/bin/env python
"""Compress / decompress trajectories with a trained checkpoint.

  python infer.py compress   --ckpt ckpt.pt --top data/adk.psf --traj data/adk_dims.dcd --out latents.npz
  python infer.py decompress --ckpt ckpt.pt --top data/adk.psf --latents latents.npz --out recon.dcd

Decompression needs only: checkpoint (weights + reference backbone), topology, latents.npz.
Output: heavy atoms of the topology (hydrogens dropped). Use .pdb for a multi-model PDB,
.dcd/.xtc for binary trajectories (pair with --out-top to also write a heavy-atom PDB topology).
"""
import argparse

import numpy as np
import torch

from md_compress.data import load_trajectory
from md_compress.training import load_checkpoint


def compress(ckpt, top, traj, out, selection="protein"):
    model, cfg = load_checkpoint(ckpt)
    data = load_trajectory(top, traj, selection)
    t = data.torch()
    stored = model.compress(t["coords"], t["aatype"])
    np.savez_compressed(out, **stored, n_res=data.n_res)
    return out, stored


def decompress(ckpt, top, latents, out, selection="protein", out_top=None, n_refine=None):
    model, cfg = load_checkpoint(ckpt)
    data = load_trajectory(top, None, selection)  # topology only (first frame is not used)
    t = data.torch()
    stored = dict(np.load(latents))
    x, oxt = model.decompress(stored, t["aatype"], t["chain_index"], t["chain_break"], n_refine=n_refine)
    data.write(x, out, oxt=oxt)
    if out_top:
        data.write(x[:1], out_top, oxt=oxt[:1])
    return x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["compress", "decompress"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--top", required=True)
    p.add_argument("--traj")
    p.add_argument("--latents")
    p.add_argument("--out", required=True)
    p.add_argument("--out-top")
    p.add_argument("--selection", default="protein")
    a = p.parse_args()
    torch.set_num_threads(8)
    if a.cmd == "compress":
        compress(a.ckpt, a.top, a.traj, a.out, a.selection)
    else:
        decompress(a.ckpt, a.top, a.latents, a.out, a.selection, a.out_top)
    print("wrote", a.out)
