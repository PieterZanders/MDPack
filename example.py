#!/usr/bin/env python
"""Minimal working example on a real small protein trajectory:
adenylate kinase (ADK, 214 residues, 1656 heavy atoms, 98 frames, CHARMM/DIMS open->closed
transition) from the MDAnalysisTests data set (data/adk.psf + data/adk_dims.dcd).

Runs: geometry oracle -> staged training (side-chain stack -> codec -> end-to-end) ->
evaluation on held-out frames -> compression to latents.npz -> decompression from
latents + topology -> PDB/DCD written and re-loaded with MDAnalysis -> memory estimate.
Results are written to results/metrics.json.   CPU-only; ~10-15 min on 8 cores.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
from md_compress import losses as L
from md_compress.compression import MDCompressor, compression_report
from md_compress.data import load_trajectory, split_frames
from md_compress.geometry import SideChainBuilder
from md_compress.memory import analytic_estimate, measure_training_activations, param_table
from md_compress.training import (LossWeights, evaluate_model, prepare, save_checkpoint, train_codec_stage,
                                  train_end_to_end, train_sidechain_stage)
import infer

FAST = "--fast" in sys.argv
EP_SC, EP_CODEC, EP_E2E = (5, 200, 2) if FAST else (60, 1500, 20)
HEAD = "vonmises" if "--vonmises" in sys.argv else "deterministic"
CODEC = "frame" if "--frame" in sys.argv else "residue"
PRIOR = "--no-chi-prior" not in sys.argv
OUT = ("results" + ("_vonmises" if HEAD == "vonmises" else "") + ("_frame" if CODEC == "frame" else "")
       + ("" if PRIOR else "_noprior") + ("_fast" if FAST else ""))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)
torch.manual_seed(0)
log_f = open(f"{OUT}/train_log.txt", "w")


def log(s):
    print(s, flush=True)
    log_f.write(s + "\n")
    log_f.flush()


TOP, TRJ = "data/adk.psf", "data/adk_dims.dcd"
data = load_trajectory(TOP, TRJ)
t = prepare(data)
tr, te = split_frames(data.n_frames, test_every=5)
log(f"ADK: {data.n_res} residues, {data.n_heavy_atoms()} heavy atoms, {data.n_frames} frames "
    f"(train {len(tr)}, test {len(te)} interleaved)")
res = dict(system=dict(name="adenylate kinase (MDAnalysisTests adk.psf/adk_dims.dcd)", n_res=data.n_res,
                       n_heavy_atoms=data.n_heavy_atoms(), n_frames=data.n_frames, n_train=len(tr), n_test=len(te)),
           head=HEAD, codec=CODEC, chi_prior=PRIOR)

# --- 1. geometry floor: true backbone + true chi through the ideal-geometry NeRF builder
b = SideChainBuilder()
x_te = t["coords"][te]
rec, _ = b(x_te[:, :, :4], t["aatype"], t["chi"][te])
res["oracle_true_chi"] = L.evaluate(rec, x_te, t["chi"][te], t["chi"][te], t["aatype"], t["atom_mask"],
                                    t["chi_mask"], t["chain_index"])
res["ground_truth_clash_per_1000_atoms"] = L.clash_score(x_te, t["aatype"], t["atom_mask"], t["chain_index"])
log("oracle (ideal geometry, true chi): " + json.dumps(res["oracle_true_chi"]))

# --- 2. naive baseline: static rotamers = circular-mean chi of training frames on the true backbone
mean_chi = torch.nn.functional.normalize(L.canonicalize_chi(t["chi"][tr], t["aatype"]).mean(0), dim=-1)
mean_chi = mean_chi * t["chi_mask"][..., None]
rec, _ = b(x_te[:, :, :4], t["aatype"], mean_chi.expand_as(t["chi"][te]))
res["baseline_static_mean_chi_true_bb"] = L.evaluate(rec, x_te, mean_chi.expand_as(t["chi"][te]), t["chi"][te],
                                                     t["aatype"], t["atom_mask"], t["chi_mask"], t["chain_index"])
log("baseline static mean chi: " + json.dumps(res["baseline_static_mean_chi_true_bb"]))

# --- 3. model + staged training
cfg = dict(n_res=data.n_res, codec=CODEC, codec_kw=dict(latent_dim=4 if CODEC == "residue" else 32),
           packer_kw=dict(n_refine=2, sidechain_kw=dict(head=HEAD)), use_chi_prior=PRIOR)
model = MDCompressor(**cfg)
model.codec.fit_reference(t["coords"][tr][:, :, :4])
model.fit_chi_prior(t["chi"][tr], t["chi_mask"], t["aatype"])
res["params"] = param_table(model)
log("params: " + json.dumps(res["params"]))
w = LossWeights(clash=0.1)
t0 = time.time()
train_sidechain_stage(model, t, tr, epochs=EP_SC, w=w, log=log)
res["after_stage1_true_bb"] = evaluate_model(model, t, te, "true_bb")
log("stage1 (true bb) test: " + json.dumps(res["after_stage1_true_bb"]))
train_codec_stage(model, t, tr, epochs=EP_CODEC, log=log)
res["after_stage2_e2e"] = evaluate_model(model, t, te, "e2e")
log("stage2 (decoded bb, frozen packer) test: " + json.dumps(res["after_stage2_e2e"]))
train_end_to_end(model, t, tr, epochs=EP_E2E, w=w, log=log)
res["train_time_s"] = time.time() - t0

# --- 4. evaluation on held-out frames
res["test_true_backbone"] = {f"refine_{k}": evaluate_model(model, t, te, "true_bb", n_refine=k) for k in (0, 1, 2)}
res["test_end_to_end"] = evaluate_model(model, t, te, "e2e")
res["train_end_to_end"] = evaluate_model(model, t, tr, "e2e")
for k, v in res["test_true_backbone"].items():
    log(f"test true-bb {k}: " + json.dumps(v))
log("test end-to-end: " + json.dumps(res["test_end_to_end"]))

if HEAD == "vonmises":  # generative head: draw rotamer samples on decoded backbones
    with torch.no_grad():
        model.eval()
        x = t["coords"][te[:4]]
        bb_al, R_, t_ = model.codec.align(x[:, :, :4])
        x_al = (x.reshape(4, -1, 3) @ R_.transpose(-1, -2) + t_[:, None]).reshape(x.shape)
        z = model.codec.encode(bb_al, t["aatype"]).half().float()
        bb = model.codec.decode(z, t["aatype"])
        outs = model.packer(bb, t["aatype"], t["chain_index"], t["chain_break"], chi_ref=model.chi_prior(),
                            sample=True, n_samples=8)
        smp = outs[-1]["samples_chi_sc"]  # [8,4,R,4,2]
        err = torch.stack([L.chi_abs_err_deg(smp[s_], t["chi"][te[:4]], t["aatype"]) for s_ in range(8)])
        cm = t["chi_mask"][..., 0].expand(4, -1)
        res["vonmises_sampling"] = dict(
            n_samples=8,
            chi1_mae_mode=float((L.chi_abs_err_deg(outs[-1]["chi_sc"], t["chi"][te[:4]], t["aatype"])[..., 0] * cm).sum() / cm.sum()),
            chi1_mae_mean_over_samples=float((err[..., 0] * cm).sum() / (cm.sum() * 8)),
            chi1_mae_best_of_8=float((err[..., 0].min(0).values * cm).sum() / cm.sum()),
            sample_sidechain_spread_A=float(outs[-1]["samples_atom14"].std(0).norm(dim=-1)[..., 4:].mean()))
        model.train()
    log("von Mises sampling: " + json.dumps(res["vonmises_sampling"]))

ck = f"{OUT}/ckpt.pt"
save_checkpoint(model, ck, cfg)

# --- 5. compression: store latents, decompress from latents + topology only, write + reload
_, stored = infer.compress(ck, TOP, TRJ, f"{OUT}/latents.npz")
xr = infer.decompress(ck, TOP, f"{OUT}/latents.npz", f"{OUT}/recon.dcd", out_top=f"{OUT}/recon_frame0.pdb")
import MDAnalysis as mda
from MDAnalysis.analysis import rms
u_rec = mda.Universe(f"{OUT}/recon_frame0.pdb", f"{OUT}/recon.dcd")
u_ref = mda.Universe(TOP, TRJ)
ref_heavy = u_ref.atoms[data.heavy_atom_indices()]
rmsds = []
for ts in u_rec.trajectory:
    u_ref.trajectory[ts.frame]
    rmsds.append(rms.rmsd(u_rec.atoms.positions, ref_heavy.positions, superposition=True))
test_rmsd = float(np.mean([rmsds[i] for i in te]))
res["roundtrip"] = dict(recon_pdb=f"{OUT}/recon_frame0.pdb", recon_dcd=f"{OUT}/recon.dcd",
                        reloaded_atoms=u_rec.atoms.n_atoms, reloaded_frames=len(u_rec.trajectory),
                        names_match=bool((u_rec.atoms.names == ref_heavy.names).all()),
                        heavy_rmsd_vs_original_test_frames=test_rmsd,
                        heavy_rmsd_vs_original_all_frames=float(np.mean(rmsds)),
                        latents_npz_bytes=os.path.getsize(f"{OUT}/latents.npz"),
                        original_dcd_bytes=os.path.getsize(TRJ))
log("roundtrip: " + json.dumps(res["roundtrip"]))
n_par = res["params"]["total"]
res["compression"] = compression_report(data.n_heavy_atoms(), model.codec, data.n_frames, n_par)
log("compression: " + json.dumps(res["compression"]))

# --- 6. memory
k_mean = float(((torch.cdist(t["coords"][0, :, 1], t["coords"][0, :, 1]) < 10).sum(-1) - 1).float().mean())
res["mean_neighbors_10A"] = k_mean
meas = {bs: measure_training_activations(model, t, tr[:bs]) / 1e6 for bs in (1, 4)}
res["measured_train_activations_MB_adk"] = meas
res["memory_estimate_300res"] = analytic_estimate(n_res=300, k=round(k_mean))
res["memory_estimate_adk_analytic"] = analytic_estimate(n_res=data.n_res, k=round(k_mean))
log("memory: " + json.dumps({k: res[k] for k in ["mean_neighbors_10A", "measured_train_activations_MB_adk",
                                                  "memory_estimate_300res"]}))
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1)
log(f"done; wrote {OUT}/metrics.json")
