#!/usr/bin/env python
"""Our methods on the benchmark systems, with the shared metrics of bench_lib and encode/decode timings.
Reuses the section 15 code (gpu/residual_coding.py, gpu/rotamer_codes.py, gpu/finetune_traj.py) unchanged:
  ours: model only        chi1..chi4 rotamer codes + frozen shared model + per trajectory residual MLP, no residuals
  ours: hybrid @ b         backbone (and every non protein atom) coded against the per trajectory mean structure with
                           a per frame rigid fit at bound b; side chains predicted by the frozen packer from the decoded
                           backbone (chi1..chi4 codes + MLP), then side chain residuals at bound b; b = 0.005 is lossless
                           on the 0.01 A source grid
  model free mean @ b      every atom against the mean structure + rigid fit, residuals at bound b (no network)
  model free raw @ b       quantised raw coordinates (best of plain / previous frame delta) + lzma, no network
Input per system: <data>/<sys>.ours.npz with atom14 protein arrays (aatype, atom_mask, chain_index, chain_break,
coords [T,R,14,3]) and optional extra [T,M,3] (non protein heavy atoms, appended after the protein atoms in the PDB).
mdCATH domains are read from data_long directly (same frames as bench_data)."""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bench_lib as B
ap = argparse.ArgumentParser()
ap.add_argument("--systems", required=True); ap.add_argument("--T", default="1000,250")
ap.add_argument("--out", default=f"{B.BENCH}/results/ours_raw.json"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--data", default="/workspace/bench_data")
ap.add_argument("--coders", default="", help="comma list restricting the entropy coder search (default: all MAIN_CODERS)")
ap.add_argument("--no-raw", action="store_true", help="skip the model free raw reference (used for the 20k atom membrane system)")
a = ap.parse_args()
os.chdir("/workspace/md_compress"); sys.path.insert(0, "/workspace/md_compress/gpu")
sys.argv = [sys.argv[0], "--threads", str(a.threads)]
import residual_coding as RCD  # noqa: E402  (its argparse sees only --threads)
FT, RC, S = RCD.FT, RCD.RC, RCD.S
from common import Domain, fit_buffers  # noqa: E402
from md_compress.geometry import kabsch  # noqa: E402
torch.set_num_threads(a.threads)
builder = FT.builder; BOUNDS = RCD.BOUNDS; coders = a.coders.split(",") if a.coders else RCD.MAIN_CODERS; HDR = RCD.HDR
RES = B.Results(a.out)
P = lambda t: time.perf_counter()


def load(sysname, T):
    if os.path.exists(f"{a.data}/{sysname}.ours.npz"):
        z = np.load(f"{a.data}/{sysname}.ours.npz"); X = z["coords"][:T]; E = z["extra"][:T] if "extra" in z.files and z["extra"].size else None
    else:
        z = np.load(f"/workspace/md_compress_gpu_data/data_long/{sysname}.npz"); arch = np.where(z["heldout"] == 0)[0][:T]
        X = z["coords"][arch]; E = None
    dom = Domain(sysname, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], int(z["atom_mask"].sum()), "cpu")
    return dom, torch.as_tensor(np.ascontiguousarray(X)), E


def enc_resid(xx, pp, b):
    q, rf = RCD.quantise(xx, pp, b); n, blob, _ = RCD.encode(q, coders); return q, rf, n, blob


def run(sysname, T, m):
    dom, x, E = load(sysname, T)
    if len(x) < T: return
    top = B.read_pdb_topology(f"{a.data}/{sysname}.pdb")
    mk = dom.atom_mask.bool(); Np = int(mk.sum()); M = 0 if E is None else E.shape[1]; N = Np + M
    truth = np.concatenate([x[:, mk].numpy()] + ([E] if M else []), 1).astype(np.float64)
    assert truth.shape[1] == top["n_atoms"], (truth.shape, top["n_atoms"])
    raw_bytes = 12 * N * T; out = {}
    def put(name, nbytes, rec, enc_s, dec_s, extra={}):
        mt = B.metrics(rec, truth, top)
        RES.put(f"{sysname}/T{T}/{name}", dict(system=sysname, T=T, method=name, family="ours" if name.startswith("ours") else "model free",
                bytes=int(nbytes), raw_bytes=raw_bytes, ratio=raw_bytes / nbytes, encode_s=enc_s, decode_s=dec_s, metrics=mt, n_protein_atoms=Np,
                n_other_atoms=M, **extra))
        print(f"{sysname}/T{T}/{name}: {raw_bytes / nbytes:.2f}x rmsd {mt['rmsd_lab']:.3f} max {mt['max_abs_err']:.3f} enc {enc_s:.1f}s dec {dec_s:.1f}s", flush=True)
    # ---------------- model free references (all atoms)
    for b in BOUNDS:
        k = f"model free raw @ {b}"
        if not RES.has(f"{sysname}/T{T}/{k}") and not a.no_raw:
            t0 = time.perf_counter(); q, rf, n, blob = enc_resid(truth, np.zeros_like(truth), b); te = time.perf_counter() - t0
            t0 = time.perf_counter(); rec = rf(RCD.decode(n, blob, q.shape)); td = time.perf_counter() - t0
            put(k, len(blob) + HDR, rec, te, td, dict(coder=n))
    t0 = time.perf_counter()
    xf = torch.as_tensor(truth, dtype=torch.float32); Mn = RCD.iter_mean(xf).float()
    Rk, tk = kabsch(Mn[None].expand(T, -1, 3), xf); pm = (Mn[None] @ Rk.transpose(-1, -2) + tk[:, None]).double().numpy()
    t_mean = time.perf_counter() - t0
    for b in BOUNDS:
        k = f"model free mean @ {b}"
        if not RES.has(f"{sysname}/T{T}/{k}"):
            t0 = time.perf_counter(); q, rf, n, blob = enc_resid(truth, pm, b); te = time.perf_counter() - t0 + t_mean
            t0 = time.perf_counter(); qd = RCD.decode(n, blob, q.shape)
            pmd = (Mn[None] @ Rk.transpose(-1, -2) + tk[:, None]).double().numpy(); rec = rf(qd); td = time.perf_counter() - t0
            put(k, 12 * N + 28 * T + len(blob) + HDR, rec, te, td, dict(coder=n))
    if dom.R == 0 or Np == 0: return
    if all(RES.has(f"{sysname}/T{T}/ours: hybrid @ {b}") for b in BOUNDS) and RES.has(f"{sysname}/T{T}/ours: model only"): return
    # ---------------- learned (protein part)
    aat = dom.aatype.numpy()
    t0 = time.perf_counter()
    tchi, _ = builder.chis_from_atom14(x, dom.aatype); fit_buffers(dom, x, builder); cmn = dom.chi_mask.numpy()
    st4, ns4 = RC.chi_states(S.ang(tchi.numpy()), aat, cmn, nk=4); n_chi_res = cmn.sum(1).astype(int)
    t_codes = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred, base, code, sec, mask4, cref4 = RCD.learned(m, dom, x, tchi, st4, ns4, 4, n_chi_res)
    t_learned = time.perf_counter() - t0
    if not RES.has(f"{sysname}/T{T}/ours: model only"):
        # decode = latents -> backbone decoder + packer + MLP + NeRF (timed separately on the stored latents)
        t1 = time.perf_counter()
        with torch.no_grad():
            fe = RC.frozen_cond(m, dom, x, cref4)  # includes the (cheap) backbone encode; upper bound on decode time
        td = time.perf_counter() - t1
        recp = pred[:, mk].numpy()
        rec = np.concatenate([recp] + ([pm[:, Np:]] if M else []), 1)
        nb = base + (12 * M + 0 if M else 0)
        note = "non protein atoms: mean structure only (no residuals)" if M else ""
        put("ours: model only", nb, rec, t_codes + t_learned, td, dict(code=code, note=note))
        del fe
    # ---------------- hybrid
    slot = torch.arange(14)[None].expand(dom.R, 14); mbb = mk & (slot < 4); msc = mk & (slot >= 4)
    bbsel = mbb[mk].numpy(); scsel = msc[mk].numpy()
    t0 = time.perf_counter()
    ft = RCD.cond_from_bb(m, dom, x[:, :, :4].contiguous(), cref4)
    with torch.enable_grad():
        FT.a.batch = min(48, T); modh, sech, _ = FT.fit("mlp", m, ft, x, dom)
    del ft
    t_fit = time.perf_counter() - t0
    c4 = code; n_bb = int(mbb.sum())
    fixed = 12 * n_bb + 12 * M + 28 * T + 80 * dom.R + c4["zlib_bytes_total"] + c4["mask_bytes"] + c4["state_table_bytes"] + RCD.MLP_BYTES
    xp = truth[:, :Np]; xbbo = np.concatenate([xp[:, bbsel]] + ([truth[:, Np:]] if M else []), 1); xsc = xp[:, scsel]
    pbbo = np.concatenate([pm[:, :Np][:, bbsel]] + ([pm[:, Np:]] if M else []), 1)
    for b in BOUNDS:
        k = f"ours: hybrid @ {b}"
        if RES.has(f"{sysname}/T{T}/{k}"): continue
        t1 = time.perf_counter()
        q1, rf1, n1, blob1 = enc_resid(xbbo, pbbo, b)
        dbb = rf1(q1); bbr = torch.zeros(T, dom.R, 4, 3); bbr[:, mbb[:, :4]] = torch.as_tensor(dbb[:, :n_bb], dtype=torch.float32)
        fh = RCD.cond_from_bb(m, dom, bbr, cref4)
        with torch.no_grad():
            ph = FT.build(m, fh, modh(fh["h"], fh["base"], None, dom.chi_mask.float()), dom) * dom.atom_mask[..., None]
        ph, nbs, frac = RCD.swap_fix(ph, x, dom)
        q2, rf2, n2, blob2 = enc_resid(xsc, ph[:, msc].double().numpy(), b)
        te = time.perf_counter() - t1 + t_codes + t_learned + t_fit + t_mean
        # decode: residual streams, mean + rigid placement, packer on the decoded backbone, swap bits, residuals
        t1 = time.perf_counter()
        q1d = RCD.decode(n1, blob1, q1.shape); dbb2 = rf1(q1d)
        bbr2 = torch.zeros(T, dom.R, 4, 3); bbr2[:, mbb[:, :4]] = torch.as_tensor(dbb2[:, :n_bb], dtype=torch.float32)
        fh2 = RCD.cond_from_bb(m, dom, bbr2, cref4)
        with torch.no_grad():
            ph2 = FT.build(m, fh2, modh(fh2["h"], fh2["base"], None, dom.chi_mask.float()), dom) * dom.atom_mask[..., None]
        q2d = RCD.decode(n2, blob2, q2.shape)
        td = time.perf_counter() - t1
        rec = np.zeros_like(truth); recp = np.zeros((T, Np, 3))
        recp[:, bbsel] = dbb2[:, :n_bb]; recp[:, scsel] = rf2(q2d); rec[:, :Np] = recp
        if M: rec[:, Np:] = dbb2[:, n_bb:]
        put(k, fixed + len(blob1) + len(blob2) + nbs + 2 * HDR, rec, te, td,
            dict(bb_and_other_residual_bytes=len(blob1), sc_residual_bytes=len(blob2), swap_bytes=nbs, fixed_bytes=fixed))


m = FT.load_main()
for s in a.systems.split(","):
    for T in [int(t) for t in a.T.split(",")]:
        t0 = time.time(); run(s, T, m); print(time.strftime("%H:%M:%S"), s, T, f"{time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
