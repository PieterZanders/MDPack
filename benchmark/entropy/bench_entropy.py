#!/usr/bin/env python
"""AEZ-12: the hybrid codec's residual (correction) streams coded with the new context modelled arithmetic coder
(gpu/residual_entropy.py) versus the lzma layout search of section 15 (gpu/residual_coding.py), on exactly the same
quantised residuals. Same systems, frames, atoms, metrics and byte accounting as benchmark/scripts/bench_ours.py.

Rows written per system and T (results/entropy_raw.json, incremental):
  ours: hybrid @ b                 recomputed in this run, lzma search (reference, should reproduce AEZ-11)
  ours: hybrid + AC @ b            same residuals, arithmetic coder
  model free mean @ b              recomputed, lzma search
  model free mean + AC @ b         arithmetic coder
  model free raw + AC @ b          quantised raw coordinates, arithmetic coder (temporal predictor), no network
  ours: hybrid + backbone AE + AC @ b   backbone (and other atoms) coded against the learned backbone autoencoder
                                   (fp16 latents, lossless lzma, counted) instead of the mean structure; side chains as hybrid
  ours: hybrid + backbone AE @ b   same residuals, lzma search
Every AC stream is decoded and compared bit exactly with the encoder's integers; the reconstruction must satisfy
max abs err <= b + 1e-3 A (lossless rows: <= 0.005 + 1e-3 on the 0.01 A grid). A failure raises and is recorded."""
import argparse, json, os, sys, time
import numpy as np, torch
BEN = "/workspace/md_compress/benchmark"; sys.path.insert(0, f"{BEN}/scripts")
import bench_lib as B
ap = argparse.ArgumentParser()
ap.add_argument("--systems", required=True); ap.add_argument("--T", default="1000,250")
ap.add_argument("--out", default=f"{BEN}/entropy/results/entropy_raw.json"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--data", default="/workspace/bench_data"); ap.add_argument("--pdb-dir", default="")
ap.add_argument("--no-ae", action="store_true", help="skip the backbone autoencoder variant")
ap.add_argument("--lzma-coders", default="", help="restrict the lzma search (membrane), default MAIN_CODERS")
ap.add_argument("--no-lzma", action="store_true"); ap.add_argument("--dump", default="")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
os.chdir("/workspace/md_compress"); sys.path.insert(0, "/workspace/md_compress/gpu")
sys.argv = [sys.argv[0], "--threads", str(a.threads)]
import residual_coding as RCD  # noqa: E402
import residual_entropy as RE  # noqa: E402
FT, RC, S = RCD.FT, RCD.RC, RCD.S
from common import Domain, fit_buffers  # noqa: E402
from md_compress.geometry import kabsch  # noqa: E402
torch.set_num_threads(a.threads)
builder = FT.builder; BOUNDS = RCD.BOUNDS; HDR = RCD.HDR
LZ = a.lzma_coders.split(",") if a.lzma_coders else RCD.MAIN_CODERS
RES = B.Results(a.out)
TOL = 1e-3


def load(sysname, T):
    if os.path.exists(f"{a.data}/{sysname}.ours.npz"):
        z = np.load(f"{a.data}/{sysname}.ours.npz"); X = z["coords"][:T]; E = z["extra"][:T] if "extra" in z.files and z["extra"].size else None
    else:
        z = np.load(f"/workspace/md_compress_gpu_data/data_long/{sysname}.npz"); arch = np.where(z["heldout"] == 0)[0][:T]
        X = z["coords"][arch]; E = None
    dom = Domain(sysname, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], int(z["atom_mask"].sum()), "cpu")
    return dom, torch.as_tensor(np.ascontiguousarray(X)), E


def classes(dom, top, Np, N):
    """atom class per atom (decoder knows the topology): protein N, CA, C, O, CB, gamma, delta, epsilon+ = 0..7;
    nucleic backbone 9, nucleic base 10, other (lipid, ligand) 11."""
    mk = dom.atom_mask.bool(); slot = torch.arange(14)[None].expand(dom.R, 14)[mk].numpy()
    c = np.full(N, 11, np.uint8)
    c[:Np] = np.select([slot < 4, slot == 4, slot == 5, slot < 8], [slot, 4, 5, 6], 7)
    if N > Np:
        nuc, bb = top["nuc"][Np:], top["bb"][Np:]
        c[Np:] = np.where(nuc & bb, 9, np.where(nuc, 10, 11))
    return c


def qbase(p, b):
    """float base predictor and step of the residual integers exactly as RCD.quantise defines them"""
    if b == 0.005: return np.round(p / RCD.GRID) * RCD.GRID, RCD.GRID
    return p, 2 * b


def ac_stream(q, cls, p, b, S, Xc=None, Sctx=None):
    """S: encoder side structure of the stream atoms (references are chosen on it and stored); Xc: decoded context atoms."""
    base, st = qbase(p, b)
    t0 = time.perf_counter()
    ref = RE.make_refs(S); R3 = RE.make_refs3(S, Sctx)
    blob, info = RE.encode(q, cls, base, st, ref, R3, Xc); te = time.perf_counter() - t0
    t0 = time.perf_counter(); qd = RE.decode(blob, q.shape, cls, base, st, Xc); td = time.perf_counter() - t0
    if not np.array_equal(qd, q): raise RuntimeError("AC roundtrip mismatch")
    return blob, info, te, td, qd


def lz_stream(q):
    t0 = time.perf_counter(); n, blob, _ = RCD.encode(q, LZ); te = time.perf_counter() - t0
    t0 = time.perf_counter(); qd = RCD.decode(n, blob, q.shape); td = time.perf_counter() - t0
    assert np.array_equal(qd, q)
    return n, blob, te, td


def check(rec, truth, b):
    err = float(np.abs(rec - truth).max())
    if err > b + TOL: raise RuntimeError(f"bound violated: {err} > {b}")
    return err


def run(sysname, T, m):
    dom, x, E = load(sysname, T)
    if len(x) < T: return
    top = B.read_pdb_topology(f"{a.pdb_dir or a.data}/{sysname}.pdb")
    mk = dom.atom_mask.bool(); Np = int(mk.sum()); M = 0 if E is None else E.shape[1]; N = Np + M
    truth = np.concatenate([x[:, mk].numpy()] + ([E] if M else []), 1).astype(np.float64)
    assert truth.shape[1] == top["n_atoms"]
    raw_bytes = 12 * N * T; cls = classes(dom, top, Np, N)
    key = lambda k: f"{sysname}/T{T}/{k}"
    def put(name, nbytes, mt, maxerr, extra):
        RES.put(key(name), dict(system=sysname, T=T, method=name, family="ours" if name.startswith("ours") else "model free", bytes=int(nbytes),
                raw_bytes=raw_bytes, ratio=raw_bytes / nbytes, metrics=mt, checked_max_abs_err=maxerr, n_protein_atoms=Np, n_other_atoms=M, **extra))
        print(f"{sysname}/T{T}/{name}: {raw_bytes / nbytes:.2f}x max {maxerr:.4f} " + " ".join(f"{k}={v:.2f}" for k, v in extra.items() if k.endswith("_s")), flush=True)
    dump = {}
    t0 = time.perf_counter()
    xf = torch.as_tensor(truth, dtype=torch.float32); Mn = RCD.iter_mean(xf).float()
    Rk, tk = kabsch(Mn[None].expand(T, -1, 3), xf); pm = (Mn[None] @ Rk.transpose(-1, -2) + tk[:, None]).double().numpy()
    t_mean = time.perf_counter() - t0
    Mnp = Mn.numpy().astype(np.float64)
    # ---------------- model free raw + AC
    for b in BOUNDS:
        k = f"model free raw + AC @ {b}"
        if RES.has(key(k)): continue
        q, rf = RCD.quantise(truth, np.zeros_like(truth), b)
        blob, info, te, td, qd = ac_stream(q, cls, np.zeros_like(truth), b, Mnp); rec = rf(qd); err = check(rec, truth, b)
        put(k, len(blob) + HDR, B.metrics(rec, truth, top), err, dict(predictor=info["predictor"], encode_s=te, decode_s=td, coder_encode_s=te, coder_decode_s=td))
        if a.dump: dump[f"raw_{b}"] = q
    # ---------------- model free mean (lzma and AC on the same residuals)
    for b in BOUNDS:
        kl, ka = f"model free mean @ {b}", f"model free mean + AC @ {b}"
        if RES.has(key(ka)) and (a.no_lzma or RES.has(key(kl))): continue
        q, rf = RCD.quantise(truth, pm, b)
        blob, info, te, td, qd = ac_stream(q, cls, pm, b, Mnp); rec = rf(qd); err = check(rec, truth, b); mt = B.metrics(rec, truth, top)
        fixed = 12 * N + 28 * T + HDR
        put(ka, fixed + len(blob), mt, err, dict(predictor=info["predictor"], encode_s=te + t_mean, decode_s=td, coder_encode_s=te, coder_decode_s=td, resid_bytes=len(blob)))
        if not a.no_lzma and not RES.has(key(kl)):
            n, lb, tle, tld = lz_stream(q)
            put(kl, fixed + len(lb), mt, err, dict(coder=n, encode_s=tle + t_mean, decode_s=tld, coder_encode_s=tle, coder_decode_s=tld, resid_bytes=len(lb)))
        if a.dump: dump[f"mean_{b}"] = q
    if a.dump: dump["cls"] = cls
    if dom.R == 0 or Np == 0:
        if a.dump: np.savez_compressed(f"{a.dump}/{sysname}_T{T}.npz", **dump)
        return
    hk = [f"ours: hybrid + AC @ {b}" for b in BOUNDS] + ([] if a.no_lzma else [f"ours: hybrid @ {b}" for b in BOUNDS])
    if not a.no_ae: hk += [f"ours: hybrid + backbone AE + AC @ {b}" for b in BOUNDS] + ([] if a.no_lzma else [f"ours: hybrid + backbone AE @ {b}" for b in BOUNDS])
    if all(RES.has(key(k)) for k in hk) and not a.dump: return
    # ---------------- hybrid (same code path as bench_ours.py)
    aat = dom.aatype.numpy()
    t0 = time.perf_counter()
    tchi, _ = builder.chis_from_atom14(x, dom.aatype); fit_buffers(dom, x, builder); cmn = dom.chi_mask.numpy()
    st4, ns4 = RC.chi_states(S.ang(tchi.numpy()), aat, cmn, nk=4); n_chi_res = cmn.sum(1).astype(int)
    pred, base, code, sec, mask4, cref4 = RCD.learned(m, dom, x, tchi, st4, ns4, 4, n_chi_res)
    slot = torch.arange(14)[None].expand(dom.R, 14); mbb = mk & (slot < 4); msc = mk & (slot >= 4)
    bbsel = mbb[mk].numpy(); scsel = msc[mk].numpy()
    ft = RCD.cond_from_bb(m, dom, x[:, :, :4].contiguous(), cref4)
    with torch.enable_grad():
        FT.a.batch = min(48, T); modh, sech, _ = FT.fit("mlp", m, ft, x, dom)
    del ft
    t_model = time.perf_counter() - t0
    c4 = code; n_bb = int(mbb.sum())
    fixed = 12 * n_bb + 12 * M + 28 * T + 80 * dom.R + c4["zlib_bytes_total"] + c4["mask_bytes"] + c4["state_table_bytes"] + RCD.MLP_BYTES
    xp = truth[:, :Np]; xbbo = np.concatenate([xp[:, bbsel]] + ([truth[:, Np:]] if M else []), 1); xsc = xp[:, scsel]
    pbbo = np.concatenate([pm[:, :Np][:, bbsel]] + ([pm[:, Np:]] if M else []), 1)
    cls_bbo = np.concatenate([cls[:Np][bbsel], cls[Np:]]); cls_sc = cls[:Np][scsel]
    S_bbo = np.concatenate([Mnp[:Np][bbsel], Mnp[Np:]]); S_sc = Mnp[:Np][scsel]; S_bb = Mnp[:Np][bbsel]
    variants = [("ours: hybrid", pbbo, fixed, 0.0)]; vinfo = {}
    if not a.no_ae and not all(RES.has(key(f"ours: hybrid + backbone AE{x} @ {b}")) for b in BOUNDS for x in ([" + AC"] + ([] if a.no_lzma else [""]))):
        # learned backbone autoencoder: fp16 latents of the per frame aligned backbone, decoded by the shared model and put back in the lab frame
        t0 = time.perf_counter()
        with torch.no_grad():
            fe = RC.frozen_cond(m, dom, x, cref4)
            bbae = m.codec.unalign(fe["bb"], fe["R"], fe["t"]); del fe
        pae = bbae.reshape(T, dom.R, 4, 3)[:, mbb[:, :4]].double().numpy(); del bbae
        lat = min(8 * dom.R * T, RCD.latent_lzma(m, dom, x))
        t_ae = time.perf_counter() - t0
        fixed_ae = lat + 28 * T + 80 * dom.R + c4["zlib_bytes_total"] + c4["mask_bytes"] + c4["state_table_bytes"] + RCD.MLP_BYTES + ((12 * M + 28 * T) if M else 0)
        pbbo_ae = np.concatenate([pae] + ([pm[:, Np:]] if M else []), 1)
        variants.append(("ours: hybrid + backbone AE", pbbo_ae, fixed_ae, t_ae))
        vinfo["ours: hybrid + backbone AE"] = dict(latent_bytes=int(lat), latent_bytes_fp16=8 * dom.R * T, ae_backbone_rmsd_lab=float(np.sqrt(((pae - xp[:, bbsel]) ** 2).sum(-1).mean())))
    for (vname, pbbo, fixed, t_extra), b in [(v, b) for v in variants for b in BOUNDS]:
        kl, ka = f"{vname} @ {b}", f"{vname} + AC @ {b}"
        if RES.has(key(ka)) and (a.no_lzma or RES.has(key(kl))) and not a.dump: continue
        t1 = time.perf_counter()
        q1, rf1 = RCD.quantise(xbbo, pbbo, b)
        dbb = rf1(q1); bbr = torch.zeros(T, dom.R, 4, 3); bbr[:, mbb[:, :4]] = torch.as_tensor(dbb[:, :n_bb], dtype=torch.float32)
        fh = RCD.cond_from_bb(m, dom, bbr, cref4)
        with torch.no_grad():
            ph0 = FT.build(m, fh, modh(fh["h"], fh["base"], None, dom.chi_mask.float()), dom) * dom.atom_mask[..., None]
        ph, nbs, frac = RCD.swap_fix(ph0, x, dom)
        q2, rf2 = RCD.quantise(xsc, ph[:, msc].double().numpy(), b)
        t_pred = time.perf_counter() - t1
        # AC
        blob1, i1, te1, td1, q1d = ac_stream(q1, cls_bbo, pbbo, b, S_bbo)
        dbb2 = rf1(q1d)   # decoded backbone (and other atoms): context of the side chain stream
        blob2, i2, te2, td2, q2d = ac_stream(q2, cls_sc, ph[:, msc].double().numpy(), b, S_sc, Xc=dbb2[:, :n_bb], Sctx=S_bb)
        rec = np.zeros_like(truth); recp = np.zeros((T, Np, 3))
        recp[:, bbsel] = dbb2[:, :n_bb]; recp[:, scsel] = rf2(q2d); rec[:, :Np] = recp
        if M: rec[:, Np:] = dbb2[:, n_bb:]
        err = check(rec, truth, b); mt = B.metrics(rec, truth, top)
        tot = t_model + t_mean + t_pred + t_extra
        put(ka, fixed + len(blob1) + len(blob2) + nbs + 2 * HDR, mt, err,
            dict(bb_and_other_residual_bytes=len(blob1), sc_residual_bytes=len(blob2), swap_bytes=nbs, fixed_bytes=fixed, predictor_bb=i1["predictor"], predictor_sc=i2["predictor"], ac_header_bytes=i1["header_bytes"] + i2["header_bytes"],
                 encode_s=tot + te1 + te2, decode_s=td1 + td2, coder_encode_s=te1 + te2, coder_decode_s=td1 + td2, model_and_prediction_s=tot, **vinfo.get(vname, {})))
        if not a.no_lzma and not RES.has(key(kl)):
            n1, l1, tle1, tld1 = lz_stream(q1); n2, l2, tle2, tld2 = lz_stream(q2)
            put(kl, fixed + len(l1) + len(l2) + nbs + 2 * HDR, mt, err,
                dict(bb_and_other_residual_bytes=len(l1), sc_residual_bytes=len(l2), swap_bytes=nbs, fixed_bytes=fixed, coder_bb=n1, coder_sc=n2,
                     encode_s=tot + tle1 + tle2, decode_s=tld1 + tld2, coder_encode_s=tle1 + tle2, coder_decode_s=tld1 + tld2, model_and_prediction_s=tot, **vinfo.get(vname, {})))
        if a.dump and vname == "ours: hybrid": dump[f"hbb_{b}"] = q1; dump[f"hsc_{b}"] = q2; dump["cls_bbo"] = cls_bbo; dump["cls_sc"] = cls_sc
    if a.dump: np.savez_compressed(f"{a.dump}/{sysname}_T{T}.npz", **dump)


m = FT.load_main()
for s in a.systems.split(","):
    for T in [int(t) for t in a.T.split(",")]:
        t0 = time.time()
        try:
            run(s, T, m)
        except Exception as ex:  # recorded, never silently dropped
            import traceback; traceback.print_exc()
            RES.put(f"{s}/T{T}/ERROR", dict(system=s, T=T, method="ERROR", error=repr(ex)[-600:]))
        print(time.strftime("%H:%M:%S"), s, T, f"{time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
