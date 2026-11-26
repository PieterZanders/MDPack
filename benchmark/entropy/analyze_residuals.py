#!/usr/bin/env python
"""AEZ-12 predictor analysis: for the same quantiser (step 2b, or the 0.01 A grid for lossless), the same coders and the
same bound, how much does each predictor shrink the residual integers?  mdCATH protein domains, T = 1000.

Predictors (all produce an integer prediction error stream E [T,N,3] that a decoder can invert in frame/atom order):
  none                 q = round(x / step)                         (raw grid, no prediction)
  temporal             q_t - q_{t-1} on the raw grid                (previous frame)
  MDCompress level 2   q_t(i) - [q_t(ref) + q_{t-1}(i) - q_{t-1}(ref)] on the raw grid (ref: nearest earlier atom)
  MDCompress level 4   q_t(i) - round(translate and rotate prediction from 3 reference atoms), raw grid
  mean structure       round((x - p_mean) / step), p_mean = iterative Kabsch mean with per frame rigid fit
  hybrid               backbone vs mean structure, side chains vs packer prediction from the decoded backbone (+ swap fix)
  hybrid + backbone AE backbone vs learned autoencoder, side chains vs packer prediction from that decoded backbone
Per predictor, bound and atom group (backbone; side chains of stable / intermediate / unstable / no chi residues):
  (statistics over frames 2..T, frame 1 having no history) RMS of E times the step (A), zeroth order entropy H0 and context entropy H(E | bucket(|E| same atom, previous frame)) in
  bits per coordinate, histogram of E. Per predictor and bound, coded bytes of the backbone and side chain streams with
  (a) lzma layout search (3 layouts), (b) the context modelled coder with its internal prediction switched off ("AC, no
  internal prediction": the pure probability model), (c) the full new coder on the model predictors' residuals.
Fixed costs (mean structure, latents, rigid transforms, codes, MLP) are reported separately per predictor.
Output: results/predictor_analysis.json (incremental, per domain)."""
import argparse, json, os, sys, time, lzma
import numpy as np, torch
BEN = "/workspace/md_compress/benchmark"; sys.path.insert(0, f"{BEN}/scripts")
import bench_lib as B
ap = argparse.ArgumentParser()
ap.add_argument("--systems", required=True); ap.add_argument("--T", type=int, default=1000)
ap.add_argument("--out", default=f"{BEN}/entropy/results/predictor_analysis.json"); ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
os.chdir("/workspace/md_compress"); sys.path.insert(0, "/workspace/md_compress/gpu")
sys.argv = [sys.argv[0], "--threads", str(a.threads)]
import residual_coding as RCD  # noqa: E402
import residual_entropy as RE  # noqa: E402
FT, RC, S = RCD.FT, RCD.RC, RCD.S
from common import Domain, fit_buffers  # noqa: E402
from md_compress.geometry import kabsch  # noqa: E402
torch.set_num_threads(a.threads)
builder = FT.builder; BOUNDS = RCD.BOUNDS
LZ3 = ["none-nat-vint-lzma", "none-nat-planes-lzma", "dt-nat-planes-lzma"]
RES = B.Results(a.out)
BK = np.array([0, 1, 2, 3, 5, 9, 17, 33, 1 << 30])


def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def Hctx(E):
    """H(E_t | bucket(|E_{t-1}|)) for the same atom and axis, frames 1..T-1"""
    cur = E[1:].ravel(); ctx = np.searchsorted(BK, np.abs(E[:-1]).ravel(), side="right")
    h = 0.0
    for c in np.unique(ctx):
        sel = ctx == c; h += sel.mean() * H(cur[sel])
    return float(h)


def stats(E, step):
    return dict(rms_A=float(np.sqrt((E.astype(np.float64) ** 2).mean()) * step), H0=H(E), Hctx=Hctx(E), frac_zero=float((E == 0).mean()),
                hist={int(k): int(v) for k, v in zip(*np.unique(np.clip(E, -40, 40), return_counts=True))})


def lzb(E):
    n, b, _ = RCD.encode(np.ascontiguousarray(E, np.int64), LZ3); return len(b)


def acb(q, cls, base, step, Sref, Xc=None, Sctx=None, force=None):
    ref = RE.make_refs(Sref); R3 = RE.make_refs3(Sref, Sctx)
    blob, info = RE.encode(q, cls, base, step, ref, R3, Xc, force=force)
    if force is None:
        qd = RE.decode(blob, q.shape, cls, base, step, Xc); assert np.array_equal(qd, q)
    return len(blob)


def mdc_errors(qr, Sref):
    """MDCompress style prediction errors on the raw integer grid (level 2 and level 4 analogues)"""
    T, N, _ = qr.shape
    ref = RE.make_refs(Sref); rr = np.where(ref < 0, 0, ref); v = (ref >= 0)[None, :, None]
    qp = np.concatenate([np.zeros((1, N, 3), qr.dtype), qr[:-1]])
    e2 = qr - (qr[:, rr] * v + qp - qp[:, rr] * v)
    R3 = RE.make_refs3(Sref); fC = RE.features(qr, np.zeros(qr.shape), 1.0, R3)
    e4 = qr - np.floor(fC + 0.5).astype(qr.dtype)
    return e2, e4


def run(d):
    z = np.load(f"/workspace/md_compress_gpu_data/data_long/{d}.npz"); arch = np.where(z["heldout"] == 0)[0][:a.T]
    X = torch.as_tensor(np.ascontiguousarray(z["coords"][arch])); temps, reps = z["temp"][arch], z["replica"][arch]
    dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], int(z["atom_mask"].sum()), "cpu")
    T = len(X); mk = dom.atom_mask.bool(); Np = int(mk.sum()); aat = dom.aatype.numpy()
    x = X; truth = x[:, mk].double().numpy()
    slot = torch.arange(14)[None].expand(dom.R, 14); mbb = mk & (slot < 4); msc = mk & (slot >= 4)
    bbsel = mbb[mk].numpy(); scsel = msc[mk].numpy(); n_bb = int(mbb.sum())
    cls_all = np.select([slot[mk].numpy() < 4, slot[mk].numpy() == 4, slot[mk].numpy() == 5, slot[mk].numpy() < 8], [slot[mk].numpy(), 4, 5, 6], 7).astype(np.uint8)
    # stability classes from the 320 K replica 0 frames (same rule as report sections 12 to 15)
    tchi, _ = builder.chis_from_atom14(x, dom.aatype); fit_buffers(dom, x, builder); cmn = dom.chi_mask.numpy(); has = cmn[:, 0] > 0
    r0 = np.where((temps == 320) & (reps == 0))[0]
    hs = S.hysteresis_states(S.ang(tchi[r0].numpy()), aat, cmn); occ, rate = S.occ_and_rate(hs, 1.0)
    stb = has & (occ >= S.OCC_STABLE) & (rate <= S.MAX_STABLE_RATE); un = has & (occ < S.OCC_UNSTABLE); inter = has & ~stb & ~un
    resi = torch.arange(dom.R)[:, None].expand(dom.R, 14)[mk].numpy()
    sc_res = resi[scsel]
    groups_sc = dict(stable=stb[sc_res], intermediate=inter[sc_res], unstable=un[sc_res], no_chi=~has[sc_res])
    # predictors
    xf = torch.as_tensor(truth, dtype=torch.float32); Mn = RCD.iter_mean(xf).float()
    Rk, tk = kabsch(Mn[None].expand(T, -1, 3), xf); pm = (Mn[None] @ Rk.transpose(-1, -2) + tk[:, None]).double().numpy(); Mnp = Mn.numpy().astype(np.float64)
    st4, ns4 = RC.chi_states(S.ang(tchi.numpy()), aat, cmn, nk=4); n_chi_res = cmn.sum(1).astype(int)
    t0 = time.perf_counter()
    pred, base, code, sec, mask4, cref4 = RCD.learned(m, dom, x, tchi, st4, ns4, 4, n_chi_res)
    ft = RCD.cond_from_bb(m, dom, x[:, :, :4].contiguous(), cref4)
    with torch.enable_grad():
        FT.a.batch = min(48, T); modh, _, _ = FT.fit("mlp", m, ft, x, dom)
    del ft
    with torch.no_grad():
        fe = RC.frozen_cond(m, dom, x, cref4); bbae = m.codec.unalign(fe["bb"], fe["R"], fe["t"]); del fe
    pae = bbae.reshape(T, dom.R, 4, 3)[:, mbb[:, :4]].double().numpy(); del bbae
    lat = min(8 * dom.R * T, RCD.latent_lzma(m, dom, x))
    c4 = code; codes = c4["zlib_bytes_total"] + c4["mask_bytes"] + c4["state_table_bytes"]
    fixed = {"none": 0, "temporal": 0, "MDCompress level 2": 0, "MDCompress level 4": 0, "mean structure": 12 * Np + 28 * T,
             "hybrid": 12 * n_bb + 28 * T + 80 * dom.R + codes + RCD.MLP_BYTES, "hybrid + backbone AE": lat + 28 * T + 80 * dom.R + codes + RCD.MLP_BYTES}
    t_model = time.perf_counter() - t0
    out = dict(system=d, T=T, n_atoms=Np, n_bb=n_bb, n_res=dom.R, fixed_bytes=fixed, latent_bytes=int(lat), raw_bytes=12 * Np * T,
               n_sc_atoms={k: int(v.sum()) for k, v in groups_sc.items()}, model_s=t_model, bounds={})
    S_bb = Mnp[bbsel]; S_sc = Mnp[scsel]
    for b in BOUNDS:
        t1 = time.perf_counter(); res = {}
        step = RCD.GRID if b == 0.005 else 2 * b
        qr, _ = RCD.quantise(truth, np.zeros_like(truth), b)
        e2, e4 = mdc_errors(qr, Mnp)
        qt = np.concatenate([qr[:1], np.diff(qr, axis=0)])
        qm, _ = RCD.quantise(truth, pm, b)
        streams = {"none": (qr, None), "temporal": (qt, None), "MDCompress level 2": (e2, None), "MDCompress level 4": (e4, None), "mean structure": (qm, pm)}
        # hybrid and hybrid + AE (side chains from the decoded backbone of each)
        for nm, pbb in (("hybrid", pm[:, bbsel]), ("hybrid + backbone AE", pae)):
            q1, rf1 = RCD.quantise(truth[:, bbsel], pbb, b)
            bbr = torch.zeros(T, dom.R, 4, 3); bbr[:, mbb[:, :4]] = torch.as_tensor(rf1(q1), dtype=torch.float32)
            fh = RCD.cond_from_bb(m, dom, bbr, cref4)
            with torch.no_grad():
                ph0 = FT.build(m, fh, modh(fh["h"], fh["base"], None, dom.chi_mask.float()), dom) * dom.atom_mask[..., None]
            ph, nbs, _ = RCD.swap_fix(ph0, x, dom); psc = ph[:, msc].double().numpy()
            q2, _ = RCD.quantise(truth[:, scsel], psc, b)
            qq = np.zeros_like(qr); qq[:, bbsel] = q1; qq[:, scsel] = q2
            P = np.zeros_like(truth); P[:, bbsel] = pbb; P[:, scsel] = psc
            streams[nm] = (qq, P, rf1(q1), nbs)
        for nm, st in streams.items():
            E = st[0]; E1 = E[1:]   # statistics on frames 2..T: frame 1 has no temporal history (MDCompress codes it as an anchor); bytes include it
            r = dict(stats_bb=stats(E1[:, bbsel], step), stats_sc=stats(E1[:, scsel], step),
                     stats_sc_groups={g: stats(E1[:, scsel][:, sel], step) for g, sel in groups_sc.items() if sel.sum() >= 3})
            r["lzma_bb"] = lzb(E[:, bbsel]); r["lzma_sc"] = lzb(E[:, scsel])
            r["ac0_bb"] = acb(E[:, bbsel], cls_all[bbsel], np.zeros((T, n_bb, 3)), 1.0, S_bb, force="zero")
            r["ac0_sc"] = acb(E[:, scsel], cls_all[scsel], np.zeros((T, int(scsel.sum()), 3)), 1.0, S_sc, force="zero")
            if st[1] is not None or nm == "none":   # full new coder on the predictor's residual integers (for raw: on the grid itself)
                p = st[1] if st[1] is not None else np.zeros_like(truth)
                bse = np.round(p / RCD.GRID) * RCD.GRID if b == 0.005 else p
                r["ac_bb"] = acb(E[:, bbsel], cls_all[bbsel], bse[:, bbsel], step, S_bb)
                if nm.startswith("hybrid"):
                    dbb = st[2]
                    r["ac_sc"] = acb(E[:, scsel], cls_all[scsel], bse[:, scsel], step, S_sc, Xc=dbb, Sctx=S_bb)
                else:
                    r["ac_sc"] = acb(E[:, scsel], cls_all[scsel], bse[:, scsel], step, S_sc)
            if len(st) > 3: r["swap_bytes"] = st[3]
            res[nm] = r
        out["bounds"][str(b)] = res
        print(d, b, f"{time.perf_counter() - t1:.0f}s", " ".join(f"{k}: H0 bb {v['stats_bb']['H0']:.2f} sc {v['stats_sc']['H0']:.2f} lz {v['lzma_bb'] + v['lzma_sc']}" for k, v in res.items()), flush=True)
    RES.put(d, out)


m = FT.load_main()
for d in a.systems.split(","):
    if RES.has(d): continue
    t0 = time.time()
    try:
        run(d)
    except Exception as ex:
        import traceback; traceback.print_exc(); RES.put(f"{d}/ERROR", dict(system=d, error=repr(ex)[-600:]))
    print(time.strftime("%H:%M:%S"), d, f"{time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
