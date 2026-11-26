#!/usr/bin/env python
"""AEZ-12 ablation of the hybrid's side chain predictor (Pieter): four variants, same backbone stream, same quantiser,
coded with the new AC coder (and the 3 layout lzma search for reference):
  full          rotamer codes (per frame chi1..4 states, flip mask, state table) + shared packer + refitted residual MLP
  no codes      packer prediction with the static per residue chi prior only (codes, mask and state table not stored)
  no MLP        rotamer codes + shared packer, no per trajectory MLP (MLP bytes not stored)
  neither       shared packer with the static chi prior only
The backbone (and non protein atoms) are coded against the mean structure exactly as in the hybrid, so all four share the
backbone stream; only side chain residuals and side data differ. Side data per variant:
  12 n_bb + 12 M + 28 T + 80 R (mean backbone, other atoms mean, rigid transforms, ref + chi prior) [+ codes] [+ MLP 5,960 B]
Per variant and bound: total bytes / ratio, residual bytes (bb stream, sc stream, swap bits), side data bytes, H0 of the
side chain residual integers overall and per stability group, and the mean structure side chain H0 for comparison.
Every AC stream is decoded and checked bit exactly, every reconstruction against its bound.
Output: results/ablation.json (incremental per system, T, variant, bound)."""
import argparse, json, os, sys, time
import numpy as np, torch
BEN = "/workspace/md_compress/benchmark"; sys.path.insert(0, f"{BEN}/scripts")
import bench_lib as B
ap = argparse.ArgumentParser()
ap.add_argument("--systems", required=True); ap.add_argument("--T", default="1000")
ap.add_argument("--out", default=f"{BEN}/entropy/results/ablation.json"); ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--data", default="/workspace/bench_data"); ap.add_argument("--no-lzma", action="store_true")
a = ap.parse_args()
os.chdir("/workspace/md_compress"); sys.path.insert(0, "/workspace/md_compress/gpu")
sys.argv = [sys.argv[0], "--threads", str(a.threads)]
import residual_coding as RCD  # noqa: E402
import residual_entropy as RE  # noqa: E402
FT, RC, S = RCD.FT, RCD.RC, RCD.S
from common import Domain, fit_buffers  # noqa: E402
from md_compress.geometry import kabsch  # noqa: E402
torch.set_num_threads(a.threads)
builder = FT.builder; BOUNDS = RCD.BOUNDS; HDR = RCD.HDR; TOL = 1e-3
LZ3 = ["none-nat-vint-lzma", "none-nat-planes-lzma", "dt-nat-planes-lzma"]
RES = B.Results(a.out)
VARS = ("full", "no codes", "no MLP", "neither")


def H(v):
    if v.size == 0: return float("nan")
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def load(sysname, T):
    if os.path.exists(f"{a.data}/{sysname}.ours.npz"):
        z = np.load(f"{a.data}/{sysname}.ours.npz"); X = z["coords"][:T]; E = z["extra"][:T] if "extra" in z.files and z["extra"].size else None
        temps = reps = None
    else:
        z = np.load(f"/workspace/md_compress_gpu_data/data_long/{sysname}.npz"); arch = np.where(z["heldout"] == 0)[0][:T]
        X = z["coords"][arch]; E = None; temps, reps = z["temp"][arch], z["replica"][arch]
    dom = Domain(sysname, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], int(z["atom_mask"].sum()), "cpu")
    return dom, torch.as_tensor(np.ascontiguousarray(X)), E, temps, reps


def classes(dom, top, Np, N):
    mk = dom.atom_mask.bool(); slot = torch.arange(14)[None].expand(dom.R, 14)[mk].numpy()
    c = np.full(N, 11, np.uint8)
    c[:Np] = np.select([slot < 4, slot == 4, slot == 5, slot < 8], [slot, 4, 5, 6], 7)
    if N > Np:
        nuc, bb = top["nuc"][Np:], top["bb"][Np:]
        c[Np:] = np.where(nuc & bb, 9, np.where(nuc, 10, 11))
    return c


def qbase(p, b):
    if b == 0.005: return np.round(p / RCD.GRID) * RCD.GRID, RCD.GRID
    return p, 2 * b


def ac_stream(q, cls, p, b, Sref, Xc=None, Sctx=None):
    base, st = qbase(p, b)
    ref = RE.make_refs(Sref); R3 = RE.make_refs3(Sref, Sctx)
    blob, info = RE.encode(q, cls, base, st, ref, R3, Xc)
    qd = RE.decode(blob, q.shape, cls, base, st, Xc)
    if not np.array_equal(qd, q): raise RuntimeError("AC roundtrip mismatch")
    return len(blob), qd


def lzb(q):
    n, blob, _ = RCD.encode(np.ascontiguousarray(q), LZ3); qd = RCD.decode(n, blob, q.shape); assert np.array_equal(qd, q); return len(blob)


def run(sysname, T, m):
    keys = [f"{sysname}/T{T}/{v} @ {b}" for v in VARS for b in BOUNDS]
    if all(RES.has(k) for k in keys): return
    dom, x, E, temps, reps = load(sysname, T)
    if len(x) < T or dom.R == 0: return
    top = B.read_pdb_topology(f"{a.data}/{sysname}.pdb")
    mk = dom.atom_mask.bool(); Np = int(mk.sum()); M = 0 if E is None else E.shape[1]; N = Np + M
    truth = np.concatenate([x[:, mk].numpy()] + ([E] if M else []), 1).astype(np.float64)
    assert truth.shape[1] == top["n_atoms"]
    raw_bytes = 12 * N * T; cls = classes(dom, top, Np, N); aat = dom.aatype.numpy()
    t0 = time.perf_counter()
    xf = torch.as_tensor(truth, dtype=torch.float32); Mn = RCD.iter_mean(xf).float()
    Rk, tk = kabsch(Mn[None].expand(T, -1, 3), xf); pm = (Mn[None] @ Rk.transpose(-1, -2) + tk[:, None]).double().numpy()
    Mnp = Mn.numpy().astype(np.float64)
    tchi, _ = builder.chis_from_atom14(x, dom.aatype); fit_buffers(dom, x, builder); cmn = dom.chi_mask.numpy(); has = cmn[:, 0] > 0
    # stability groups: mdCATH as section 4.7 (320 K replica 0 frames); MDDB: all frames (single run, one temperature)
    sel = np.where((temps == 320) & (reps == 0))[0] if temps is not None else np.arange(T)
    hs = S.hysteresis_states(S.ang(tchi[sel].numpy()), aat, cmn); occ, rate = S.occ_and_rate(hs, 1.0)
    stb = has & (occ >= S.OCC_STABLE) & (rate <= S.MAX_STABLE_RATE); un = has & (occ < S.OCC_UNSTABLE); inter = has & ~stb & ~un
    slot = torch.arange(14)[None].expand(dom.R, 14); mbb = mk & (slot < 4); msc = mk & (slot >= 4)
    bbsel = mbb[mk].numpy(); scsel = msc[mk].numpy(); n_bb = int(mbb.sum())
    resi = torch.arange(dom.R)[:, None].expand(dom.R, 14)[mk].numpy(); sc_res = resi[scsel]
    groups = dict(stable=stb[sc_res], intermediate=inter[sc_res], unstable=un[sc_res], no_chi=~has[sc_res])
    st4, ns4 = RC.chi_states(S.ang(tchi.numpy()), aat, cmn, nk=4); n_chi_res = cmn.sum(1).astype(int)
    _, _, code, _, _, cref4 = RCD.learned(m, dom, x, tchi, st4, ns4, 4, n_chi_res)
    codes_b = code["zlib_bytes_total"] + code["mask_bytes"] + code["state_table_bytes"]
    cref0 = dom.chi_ref[None].repeat(T, 1, 1, 1).clone()
    crefs = {"codes": cref4, "nocodes": cref0}; mlps = {}
    for nm, cr in crefs.items():
        ft = RCD.cond_from_bb(m, dom, x[:, :, :4].contiguous(), cr)
        with torch.enable_grad():
            FT.a.batch = min(48, T); mlps[nm], _, _ = FT.fit("mlp", m, ft, x, dom)
        del ft
    t_model = time.perf_counter() - t0
    side0 = 12 * n_bb + 12 * M + 28 * T + 80 * dom.R
    side = {"full": side0 + codes_b + RCD.MLP_BYTES, "no codes": side0 + RCD.MLP_BYTES, "no MLP": side0 + codes_b, "neither": side0}
    use = {"full": ("codes", True), "no codes": ("nocodes", True), "no MLP": ("codes", False), "neither": ("nocodes", False)}
    xp = truth[:, :Np]; xbbo = np.concatenate([xp[:, bbsel]] + ([truth[:, Np:]] if M else []), 1); xsc = xp[:, scsel]
    pbbo = np.concatenate([pm[:, :Np][:, bbsel]] + ([pm[:, Np:]] if M else []), 1)
    cls_bbo = np.concatenate([cls[:Np][bbsel], cls[Np:]]); cls_sc = cls[:Np][scsel]
    S_bbo = np.concatenate([Mnp[:Np][bbsel], Mnp[Np:]]); S_sc = Mnp[:Np][scsel]; S_bb = Mnp[:Np][bbsel]
    for b in BOUNDS:
        if all(RES.has(f"{sysname}/T{T}/{v} @ {b}") for v in VARS): continue
        t1 = time.perf_counter()
        q1, rf1 = RCD.quantise(xbbo, pbbo, b)
        nb_ac, q1d = ac_stream(q1, cls_bbo, pbbo, b, S_bbo); nb_lz = None if a.no_lzma else lzb(q1)
        dbb = rf1(q1d); bbr = torch.zeros(T, dom.R, 4, 3); bbr[:, mbb[:, :4]] = torch.as_tensor(dbb[:, :n_bb], dtype=torch.float32)
        qm_sc, _ = RCD.quantise(xsc, pm[:, :Np][:, scsel], b)
        mean_sc = dict(H0=H(qm_sc[1:]), groups={g: H(qm_sc[1:, s]) for g, s in groups.items()},
                       rms_A=float(np.sqrt((qm_sc.astype(np.float64) ** 2).mean()) * (RCD.GRID if b == 0.005 else 2 * b)))
        bb_H0 = H(q1[1:])
        fh = {nm: RCD.cond_from_bb(m, dom, bbr, cr) for nm, cr in crefs.items()}
        for v in VARS:
            k = f"{sysname}/T{T}/{v} @ {b}"
            if RES.has(k): continue
            cn, um = use[v]; f_ = fh[cn]
            with torch.no_grad():
                chi = mlps[cn](f_["h"], f_["base"], None, dom.chi_mask.float()) if um else f_["base"]
                ph0 = FT.build(m, f_, chi, dom) * dom.atom_mask[..., None]
            ph, nbs, frac = RCD.swap_fix(ph0, x, dom)
            psc = ph[:, msc].double().numpy()
            q2, rf2 = RCD.quantise(xsc, psc, b)
            ns_ac, q2d = ac_stream(q2, cls_sc, psc, b, S_sc, Xc=dbb[:, :n_bb], Sctx=S_bb)
            ns_lz = None if a.no_lzma else lzb(q2)
            rec = np.zeros_like(truth); recp = np.zeros((T, Np, 3)); recp[:, bbsel] = dbb[:, :n_bb]; recp[:, scsel] = rf2(q2d); rec[:, :Np] = recp
            if M: rec[:, Np:] = dbb[:, n_bb:]
            err = float(np.abs(rec - truth).max())
            if err > b + TOL: raise RuntimeError(f"bound violated {k}: {err}")
            tot_ac = side[v] + nb_ac + ns_ac + nbs + 2 * HDR
            tot_lz = None if a.no_lzma else side[v] + nb_lz + ns_lz + nbs + 2 * HDR
            step = RCD.GRID if b == 0.005 else 2 * b
            RES.put(k, dict(system=sysname, T=T, variant=v, bound=b, raw_bytes=raw_bytes, checked_max_abs_err=err,
                            bytes_ac=int(tot_ac), ratio_ac=raw_bytes / tot_ac, bytes_lzma=tot_lz, ratio_lzma=None if tot_lz is None else raw_bytes / tot_lz,
                            side_bytes=int(side[v]), codes_bytes=int(codes_b) if cn == "codes" else 0, mlp_bytes=RCD.MLP_BYTES if um else 0,
                            bb_resid_ac=int(nb_ac), sc_resid_ac=int(ns_ac), bb_resid_lzma=nb_lz, sc_resid_lzma=ns_lz, swap_bytes=int(nbs), swap_frac=frac,
                            sc_H0=H(q2[1:]), sc_H0_groups={g: H(q2[1:, s]) for g, s in groups.items()}, bb_H0=bb_H0,
                            sc_rms_A=float(np.sqrt((q2.astype(np.float64) ** 2).mean()) * step), mean_sc=mean_sc,
                            n_sc_atoms={g: int(s.sum()) for g, s in groups.items()}, n_res=dom.R, n_atoms=N, n_bb=n_bb, n_other=M,
                            model_s=t_model, bound_s=time.perf_counter() - t1))
            print(f"{k}: AC {raw_bytes / tot_ac:.2f}x lz {'' if tot_lz is None else f'{raw_bytes / tot_lz:.2f}x'} sc {ns_ac} side {side[v]} H0sc {H(q2[1:]):.3f} (mean {mean_sc['H0']:.3f}) err {err:.4f}", flush=True)
        del fh


m = FT.load_main()
for s in a.systems.split(","):
    for T in [int(t) for t in a.T.split(",")]:
        try: run(s, T, m)
        except Exception:
            import traceback; traceback.print_exc()
print("done", flush=True)
