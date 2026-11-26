#!/usr/bin/env python
"""Train an MD compressor on a trajectory.

  python train.py --top data/adk.psf --traj data/adk_dims.dcd --out ckpt.pt \
      --codec residue --latent-dim 4 --head deterministic --stage all

Stages: sidechain (packer on true backbones), codec (backbone AE), e2e (joint fine-tuning),
all (= sidechain -> codec -> e2e), or e2e_scratch (end-to-end from random init).
"""
import argparse
import json

import torch

from md_compress.compression import MDCompressor
from md_compress.data import load_trajectory, split_frames
from md_compress.memory import param_table
from md_compress.training import (LossWeights, evaluate_model, prepare, save_checkpoint, train_codec_stage,
                                  train_end_to_end, train_sidechain_stage)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--top", required=True)
    p.add_argument("--traj", default=None)
    p.add_argument("--selection", default="protein")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", default="ckpt.pt")
    p.add_argument("--codec", default="residue", choices=["residue", "frame"])
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--head", default="deterministic", choices=["deterministic", "vonmises"])
    p.add_argument("--n-refine", type=int, default=2)
    p.add_argument("--no-chi-prior", action="store_true", help="generic packer: no reference rotamers")
    p.add_argument("--stage", default="all", choices=["all", "sidechain", "codec", "e2e", "e2e_scratch"])
    p.add_argument("--epochs-sc", type=int, default=60)
    p.add_argument("--epochs-codec", type=int, default=1500)
    p.add_argument("--epochs-e2e", type=int, default=20)
    p.add_argument("--clash-weight", type=float, default=0.1)
    p.add_argument("--test-every", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--init", default=None, help="checkpoint to continue from")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    data = load_trajectory(a.top, a.traj, a.selection, a.stride)
    t = prepare(data)
    train_idx, test_idx = split_frames(data.n_frames, a.test_every)
    ld = a.latent_dim or (4 if a.codec == "residue" else 32)
    cfg = dict(n_res=data.n_res, codec=a.codec, codec_kw=dict(latent_dim=ld),
               packer_kw=dict(n_refine=a.n_refine, sidechain_kw=dict(head=a.head)),
               use_chi_prior=not a.no_chi_prior)
    model = MDCompressor(**cfg)
    if a.init:
        model.load_state_dict(torch.load(a.init, weights_only=False)["state_dict"])
    else:
        model.codec.fit_reference(t["coords"][train_idx][:, :, :4])
        model.fit_chi_prior(t["chi"][train_idx], t["chi_mask"], t["aatype"])
    w = LossWeights(clash=a.clash_weight)
    print(json.dumps(param_table(model), indent=1))
    if a.stage in ("all", "sidechain"):
        train_sidechain_stage(model, t, train_idx, epochs=a.epochs_sc, w=w)
    if a.stage in ("all", "codec"):
        train_codec_stage(model, t, train_idx, epochs=a.epochs_codec)
    if a.stage in ("all", "e2e", "e2e_scratch"):
        ep = a.epochs_e2e if a.stage != "e2e_scratch" else max(a.epochs_e2e, a.epochs_sc)
        train_end_to_end(model, t, train_idx, epochs=ep, w=w, lr=5e-4 if a.stage != "e2e_scratch" else 2e-3)
    for mode in ("true_bb", "e2e"):
        print(mode, json.dumps(evaluate_model(model, t, test_idx, mode)))
    save_checkpoint(model, a.out, cfg)
    print("saved", a.out)


if __name__ == "__main__":
    main()
