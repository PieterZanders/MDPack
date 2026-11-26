"""Produce PDB files for visualisation: true vs reconstructed frames with per-residue side-chain error
in the B-factor column, plus short multi-model trajectories. Uses the AdK checkpoint from example.py and
trains (or loads) a per-protein model for one ATLAS system (2ygs_A)."""
import json, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(8); torch.set_flush_denormal(True)
import MDAnalysis as mda
from md_compress.data import load_trajectory, split_frames
from md_compress.compression import MDCompressor
from md_compress.training import (prepare, load_checkpoint, save_checkpoint, train_sidechain_stage,
                                  train_codec_stage, train_end_to_end, LossWeights)
from md_compress.geometry import kabsch
from md_compress import losses as L

OUT = "report/viz"
os.makedirs(OUT, exist_ok=True)


def per_residue_sc_err(pred, true, aatype, mask):
    """pred/true [R,14,3] already superposed; returns per-residue side-chain RMSD (swap-aware), 0 for GLY/ALA-free."""
    sw = L._swapped(true[None], aatype)[0]
    m = mask.clone(); m[:, :4] = 0
    e1 = (((pred - true) ** 2).sum(-1) * m).sum(-1)
    e2 = (((pred - sw) ** 2).sum(-1) * m).sum(-1)
    n = m.sum(-1).clamp(min=1)
    return torch.sqrt(torch.minimum(e1, e2) / n), torch.where((e2 < e1)[:, None, None], sw, true)


def write_pdb(data, atom14_frames, path, bfac_res=None, oxt=None):
    """atom14_frames [T,R,14,3] -> PDB (multi-model if T>1), per-residue B-factor optional."""
    data.write(atom14_frames, path, oxt=oxt)
    if bfac_res is not None:
        u = mda.Universe(path)
        u.add_TopologyAttr("tempfactors")
        for k, res in enumerate(u.residues):
            res.atoms.tempfactors = float(bfac_res[k])
        u.atoms.write(path)


def frames_superposed(data, t, model, stored, idx):
    x, oxt = model.decompress({k: v[idx] for k, v in stored.items() if k != "n_res"} if True else stored,
                              t["aatype"], t["chain_index"], t["chain_break"])
    true = t["coords"][idx]
    x = torch.as_tensor(x)
    m = t["atom_mask"]
    T = len(idx)
    bbm = m[:, :4].reshape(-1).expand(T, -1)
    # superpose reconstruction onto truth on backbone
    Rk, tk = kabsch(x[:, :, :4].reshape(T, -1, 3), true[:, :, :4].reshape(T, -1, 3), bbm)
    xs = (x.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(x.shape)
    oxt = torch.as_tensor(oxt) @ Rk.transpose(-1, -2) + tk[:, None]
    return xs, true, oxt.numpy()


def do_system(name, top, traj, stride, ckpt, train_if_missing):
    data = load_trajectory(top, traj, stride=stride)
    t = prepare(data)
    tr, te = split_frames(data.n_frames)
    if not os.path.exists(ckpt):
        assert train_if_missing
        m = MDCompressor(data.n_res, codec_kw=dict(latent_dim=4), packer_kw=dict(n_refine=2))
        m.codec.fit_reference(t["coords"][tr][:, :, :4]); m.fit_chi_prior(t["chi"][tr], t["chi_mask"], t["aatype"])
        w = LossWeights(clash=0.1); q = lambda s: None
        train_sidechain_stage(m, t, tr, epochs=30, w=w, log=q); train_codec_stage(m, t, tr, epochs=600, log=q)
        train_end_to_end(m, t, tr, epochs=8, w=w, log=q)
        save_checkpoint(m, ckpt, dict(n_res=data.n_res, codec="residue", codec_kw=dict(latent_dim=4),
                                      packer_kw=dict(n_refine=2), use_chi_prior=True))
    model, _ = load_checkpoint(ckpt)
    stored = model.compress(t["coords"], t["aatype"])
    f = int(te[0])
    xs, true, oxt = frames_superposed(data, t, model, stored, np.array([f]))
    err, true_sw = per_residue_sc_err(xs[0], true[0], t["aatype"], t["atom_mask"])
    write_pdb(data, true_sw[None].numpy(), f"{OUT}/{name}_true_f{f}.pdb", bfac_res=np.zeros(data.n_res))
    write_pdb(data, xs.numpy(), f"{OUT}/{name}_recon_f{f}.pdb", bfac_res=err.numpy(), oxt=oxt)
    # short trajectories (every 4th frame) for animation, superposed on frame f's truth frame
    idx = np.arange(0, data.n_frames, 4)
    xs_all, true_all, oxt_all = frames_superposed(data, t, model, stored, idx)
    write_pdb(data, xs_all.numpy(), f"{OUT}/{name}_recon_traj.pdb", oxt=oxt_all)
    write_pdb(data, true_all.numpy(), f"{OUT}/{name}_true_traj.pdb")
    info = dict(frame=f, n_res=data.n_res, mean_sc_err=float(err[t["chi_mask"][:, 0] > 0].mean()),
                worst=[(data.resnames[i], int(i) + 1, float(err[i])) for i in torch.argsort(-err)[:8].tolist()],
                traj_frames=idx.tolist())
    print(name, json.dumps(info))
    return info


info = {}
info["adk"] = do_system("adk", "data/adk.psf", "data/adk_dims.dcd", 1, "results/ckpt.pt", False)
info["2ygs_A"] = do_system("2ygs_A", "data/atlas/2ygs_A/2ygs_A.pdb", "data/atlas/2ygs_A/2ygs_A_prod_R1_fit.xtc", 100,
                           "results_multi/ckpt_2ygs_A.pt", True)
json.dump(info, open(f"{OUT}/viz_info.json", "w"), indent=1)
