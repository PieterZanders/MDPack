#!/usr/bin/env python
"""Does the compressed trajectory preserve the dynamics?  (CPU is fine.)

For each held-out protein: encode every frame to fp16 latents, decode the full heavy-atom trajectory,
and compare true vs reconstructed:
  * RMSD vs time (CA, to the first frame and to the average structure), plus per-frame reconstruction
    error (backbone and side-chain heavy atoms, after backbone fit)
  * per-residue RMSF (CA and side-chain heavy atoms) and Pearson r
  * PCA on true CA coordinates: projections on PC1/PC2 (both trajectories), eigenvalue spectra of each
    trajectory's own PCA, RMSIP of the top 10 PCs, total-variance ratio, PC1/PC2 time series
Writes <out>/dynamics.json and one PNG per protein.
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Domain, bind, build_model, fit_buffers, load_weights  # noqa: E402
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import SideChainBuilder, kabsch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--variant", default="S")
ap.add_argument("--no_prior", action="store_true")
ap.add_argument("--systems", nargs="+", required=True, help="npz files (label=path or path)")
ap.add_argument("--out", default="report/gpu_figs")
ap.add_argument("--dt", default="", help="label=ns_per_frame,... time step per system")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_grad_enabled(False)
dev = "cpu"
model = build_model(a.variant, not a.no_prior, dev)
load_weights(model, torch.load(a.ckpt, map_location=dev, weights_only=False)["weights"])
model.eval()
builder = SideChainBuilder()
DT = dict(kv.split("=") for kv in a.dt.split(",") if kv)


def superpose(X, ref, sel):
    """X [T,N,3] fitted on atoms sel to ref [N,3] -> fitted X."""
    T = X.shape[0]
    Rk, tk = kabsch(X[:, sel], ref[None, sel].expand(T, -1, 3))
    return X @ Rk.transpose(-1, -2) + tk[:, None]


def iter_mean(X, sel, n=5):
    ref = X[0]
    for _ in range(n):
        Xf = superpose(X, ref, sel)
        ref = Xf.mean(0)
    return superpose(X, ref, sel), ref


def rmsd(A, B):
    return ((A - B) ** 2).sum(-1).mean(-1).sqrt()


def pca(Xf):
    T = Xf.shape[0]
    Z = (Xf - Xf.mean(0)).reshape(T, -1).double()
    C = Z.T @ Z / (T - 1)
    ev, V = torch.linalg.eigh(C)
    return ev.flip(0), V.flip(1), Z


summary = {}
for spec in a.systems:
    label, path = spec.split("=") if "=" in spec else (os.path.basename(spec)[:-4], spec)
    z = np.load(path)
    x = z["coords"]
    if "replica" in z:
        x = x[z["replica"] == 0]
    x = torch.as_tensor(x.astype(np.float32))
    dom = Domain(label, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], dev)
    fit_buffers(dom, x, builder)
    bind(model, dom)
    recs = []
    for s in range(0, len(x), 32):
        xb = x[s:s + 32]
        bb_al, R_, t_ = model.codec.align(xb[:, :, :4])
        zl = model.codec.encode(bb_al, dom.aatype).half().float()
        _, outs = model.decode_full(zl, dom.aatype, dom.chain_index, dom.chain_break)
        recs.append(model.codec.unalign(outs[-1]["atom14"] * outs[-1]["atom_mask"][..., None], R_, t_))
    rec = torch.cat(recs)
    T, R = x.shape[:2]
    m = dom.atom_mask
    # ---- per-frame reconstruction error (backbone fit), symmetry-aware
    chi, _ = builder.chis_from_atom14(x, dom.aatype)
    err_bb, err_sc = [], []
    for s in range(0, T, 64):
        ev = L.evaluate(rec[s:s + 64], x[s:s + 64], chi[s:s + 64], chi[s:s + 64], dom.aatype, m, dom.chi_mask, dom.chain_index)
    # per-frame values (evaluate averages, so recompute per frame cheaply)
    for f in range(T):
        e = L.evaluate(rec[f:f + 1], x[f:f + 1], chi[f:f + 1], chi[f:f + 1], dom.aatype, m, dom.chi_mask, dom.chain_index)
        err_bb.append(e["backbone_rmsd"]); err_sc.append(e["sidechain_rmsd"])
    # ---- flat heavy-atom arrays
    flat_m = m.reshape(-1) > 0
    Xt = x.reshape(T, -1, 3)[:, flat_m]
    Xr = rec.reshape(T, -1, 3)[:, flat_m]
    slot = torch.arange(14).expand(R, 14).reshape(-1)[flat_m]
    resi = torch.arange(R)[:, None].expand(R, 14).reshape(-1)[flat_m]
    ca = slot == 1
    sc = slot >= 4
    # ---- RMSD vs time (CA)
    Xt_ca, Xr_ca = Xt[:, ca], Xr[:, ca]
    ft0 = superpose(Xt_ca, Xt_ca[0], slice(None)); fr0 = superpose(Xr_ca, Xr_ca[0], slice(None))
    rmsd0_t, rmsd0_r = rmsd(ft0, Xt_ca[0]), rmsd(fr0, Xr_ca[0])
    Ft, mean_t = iter_mean(Xt_ca, slice(None)); Fr, mean_r = iter_mean(Xr_ca, slice(None))
    rmsdav_t, rmsdav_r = rmsd(Ft, mean_t), rmsd(Fr, mean_r)
    # ---- RMSF: fit all heavy atoms on CA to each trajectory's own mean CA
    def rmsf_all(X):
        _, ref = iter_mean(X[:, ca], slice(None))
        Rk, tk = kabsch(X[:, ca], ref[None].expand(T, -1, 3))
        Xf = X @ Rk.transpose(-1, -2) + tk[:, None]
        return ((Xf - Xf.mean(0)) ** 2).sum(-1).mean(0).sqrt()
    ft, fr = rmsf_all(Xt), rmsf_all(Xr)
    rmsf_ca_t, rmsf_ca_r = ft[ca], fr[ca]
    has_sc = torch.zeros(R, dtype=torch.bool); has_sc[resi[sc]] = True
    def per_res_sc(v):
        out = torch.zeros(R); cnt = torch.zeros(R)
        out.index_add_(0, resi[sc], v[sc]); cnt.index_add_(0, resi[sc], torch.ones(int(sc.sum())))
        return (out / cnt.clamp(min=1))[has_sc]
    rmsf_sc_t, rmsf_sc_r = per_res_sc(ft), per_res_sc(fr)
    pr = lambda u, v: float(np.corrcoef(u.numpy(), v.numpy())[0, 1])
    # ---- PCA on true CA (both trajectories superposed on the true mean CA structure)
    Ftt = superpose(Xt_ca, mean_t, slice(None)); Frr = superpose(Xr_ca, mean_t, slice(None))
    ev_t, V_t, Zt = pca(Ftt)
    ev_r, V_r, _ = pca(Frr)
    Zr = (Frr.reshape(T, -1).double() - Ftt.reshape(T, -1).double().mean(0))
    Pt, Pr = Zt @ V_t[:, :2], Zr @ V_t[:, :2]
    k = 10
    rmsip = float(((V_t[:, :k].T @ V_r[:, :k]) ** 2).sum().div(k).sqrt())
    var_ratio = float(ev_r.sum() / ev_t.sum())
    proj_var_ratio = [float(Pr[:, i].var() / Pt[:, i].var()) for i in range(2)]
    pc_r = [pr(Pt[:, i].float(), Pr[:, i].float()) for i in range(2)]
    dt = float(DT.get(label, 1.0))
    res = dict(n_res=R, n_frames=T, ns_per_frame=dt,
               recon_bb_rmsd_mean=float(np.mean(err_bb)), recon_sc_rmsd_mean=float(np.mean(err_sc)),
               rmsd_to_first_mean=dict(true=float(rmsd0_t.mean()), recon=float(rmsd0_r.mean())),
               rmsd_to_first_pearson=pr(rmsd0_t, rmsd0_r),
               rmsf_ca_pearson=pr(rmsf_ca_t, rmsf_ca_r), rmsf_sc_pearson=pr(rmsf_sc_t, rmsf_sc_r),
               rmsf_ca_mean=dict(true=float(rmsf_ca_t.mean()), recon=float(rmsf_ca_r.mean())),
               rmsf_sc_mean=dict(true=float(rmsf_sc_t.mean()), recon=float(rmsf_sc_r.mean())),
               pca_explained_top3=dict(true=(ev_t[:3] / ev_t.sum()).tolist(), recon=(ev_r[:3] / ev_r.sum()).tolist()),
               rmsip_top10=rmsip, total_ca_variance_ratio=var_ratio,
               pc12_variance_ratio=proj_var_ratio, pc12_timeseries_pearson=pc_r)
    summary[label] = res
    print(label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                             if not isinstance(v, (dict, list))}), flush=True)
    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(T) * dt
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a0 = ax[0, 0]
    a0.plot(t, rmsd0_t, c="k", lw=1, label="true, to first frame"); a0.plot(t, rmsd0_r, c="tab:orange", lw=1, label="recon, to first frame")
    a0.plot(t, rmsdav_t, c="k", lw=1, ls="--", label="true, to average"); a0.plot(t, rmsdav_r, c="tab:orange", lw=1, ls="--", label="recon, to average")
    a0.set_xlabel("time (ns)"); a0.set_ylabel("CA RMSD (Å)"); a0.legend(fontsize=7); a0.set_title("RMSD vs time")
    a1 = ax[0, 1]
    a1.plot(t, err_bb, c="tab:blue", lw=1, label=f"backbone (mean {np.mean(err_bb):.2f} Å)")
    a1.plot(t, err_sc, c="tab:red", lw=1, label=f"side chains (mean {np.mean(err_sc):.2f} Å)")
    a1.set_xlabel("time (ns)"); a1.set_ylabel("recon vs true RMSD (Å)"); a1.legend(fontsize=8); a1.set_title("Per-frame reconstruction error")
    a2 = ax[0, 2]
    idx = np.arange(R)
    a2.plot(idx, rmsf_ca_t, c="k", lw=1, label="CA true"); a2.plot(idx, rmsf_ca_r, c="tab:orange", lw=1, label=f"CA recon (r={res['rmsf_ca_pearson']:.2f})")
    sidx = idx[has_sc.numpy()]
    a2.plot(sidx, rmsf_sc_t, c="0.5", lw=0.8, label="side chain true"); a2.plot(sidx, rmsf_sc_r, c="tab:red", lw=0.8, label=f"side chain recon (r={res['rmsf_sc_pearson']:.2f})")
    a2.set_xlabel("residue"); a2.set_ylabel("RMSF (Å)"); a2.legend(fontsize=7); a2.set_title("Per-residue RMSF")
    a3 = ax[1, 0]
    s1 = a3.scatter(Pt[:, 0], Pt[:, 1], c=t, cmap="Greys", s=10, label="true")
    a3.scatter(Pr[:, 0], Pr[:, 1], c=t, cmap="Oranges", s=10, marker="^", label="recon")
    a3.set_xlabel("PC1 (Å)"); a3.set_ylabel("PC2 (Å)"); a3.legend(fontsize=8); a3.set_title("Projection on true PC1/PC2 (color = time)")
    a4 = ax[1, 1]
    n = min(20, len(ev_t))
    a4.bar(np.arange(1, n + 1) - 0.2, (ev_t[:n] / ev_t.sum()).numpy(), width=0.4, color="k", label="true")
    a4.bar(np.arange(1, n + 1) + 0.2, (ev_r[:n] / ev_t.sum()).numpy(), width=0.4, color="tab:orange", label="recon (own PCA)")
    a4.set_xlabel("PC"); a4.set_ylabel("variance / true total"); a4.legend(fontsize=8)
    a4.set_title(f"Spectra: RMSIP(10) {rmsip:.2f}, variance ratio {var_ratio:.2f}")
    a5 = ax[1, 2]
    a5.plot(t, Pt[:, 0], c="k", lw=1, label="PC1 true"); a5.plot(t, Pr[:, 0], c="tab:orange", lw=1, label=f"PC1 recon (r={pc_r[0]:.2f})")
    a5.plot(t, Pt[:, 1], c="0.6", lw=1, label="PC2 true"); a5.plot(t, Pr[:, 1], c="tab:red", lw=1, label=f"PC2 recon (r={pc_r[1]:.2f})")
    a5.set_xlabel("time (ns)"); a5.set_ylabel("projection (Å)"); a5.legend(fontsize=7); a5.set_title("PC time series")
    fig.suptitle(f"{label}: {R} residues, {T} frames, decoded from 4 fp16 numbers per residue (held-out protein)")
    plt.tight_layout(); plt.savefig(f"{a.out}/dynamics_{label}.png", dpi=110); plt.close(fig)
json.dump(summary, open(f"{a.out}/dynamics.json", "w"), indent=1)
