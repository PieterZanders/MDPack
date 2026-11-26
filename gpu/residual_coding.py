#!/usr/bin/env python
"""Error-bounded residual coding on top of the learned reconstruction, and chi3/chi4 rotamer codes (section 15).

Data and archive: the 16 mdCATH test domains of section 14 (results_gpu/frames/selected_domains.json), archived
trajectory = first T frames of the archive order of gpu/frames_scaling.py (320 K replicas 0..3 first, 1 ns spacing).
Main setting T = 1000, also T = 250. Everything is fitted on, and evaluated on, the archived frames themselves
(residual coding is a property of the stored archive, so there is no held-out regime here).

Learned reconstructions (lab frame, shared weights frozen and not counted):
  c12   = chi1+chi2 rotamer codes + frozen model + per-trajectory residual MLP (best variant of sections 13 and 14);
  c1234 = the same with chi1..chi4 codes (gpu/rotamer_codes.py, nk=4) and the MLP refitted with the new state priors.
Residual stream at bound b (Å, max abs error per Cartesian coordinate): q = round((x - p) / 2b), x_rec = p + 2b q,
so |x_rec - x| <= b for every coordinate of every heavy atom (backbone and side chain). b = 0.005 means lossless at
the 0.01 Å precision of the source coordinates (integer residuals on the 0.01 Å grid: x_rec == x).
For the learned predictors a swap bit per symmetric residue and frame (ring / carboxylate / guanidinium labels) picks
the atom labelling of p closest to x before the residual is taken; the swap bits are stored (zlib) and counted.
Entropy coder (all candidates are tried, the smallest is kept and decoded back to verify): transform none or delta
against the previous frame; layout frame-major (T,N,3) or coordinate-major (N,3,T); serialisation zigzag varint or
zigzag byte planes (uint8, or uint16 split into low and high planes); compressor zlib -9 or lzma -9.
Model-free references, same quantiser and same coder:
  raw   = quantised raw coordinates (predictor 0, delta coding gives the previous-frame predictor);
  mean  = per-protein mean structure (fp32, 12 B per atom) placed by a per-frame rigid fit (28 B per frame).
Bytes: learned = T*(8R+28) + 80R + code streams + mask + state table + 5,960 (MLP) [+ residual + swap streams + 8];
raw = residual stream + 8; mean = 12 N + 28 T + residual stream + 8. Raw size = 12 B per heavy atom per frame.
Output: results_gpu/eval/residual_raw.json (one entry per domain and T, resumable); --summarize -> residual_results.json
and report/gpu_figs/fig_error_bounds.png.
"""
import argparse, json, lzma, math, os, sys, time, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0]]
import finetune_traj as FT  # noqa: E402
import rotamer_codes as RC  # noqa: E402
sys.argv = _argv
from common import Domain, fit_buffers  # noqa: E402
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import kabsch  # noqa: E402
import stability_lib as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/md_compress_gpu_data/data_long")
ap.add_argument("--raw", default="results_gpu/eval/residual_raw.json")
ap.add_argument("--out", default="results_gpu/eval/residual_results.json")
ap.add_argument("--only", default="")
ap.add_argument("--levels", default="1000,250")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--coders", default="main", help="main (lzma candidates that won the pilot), all, or a comma list")
ap.add_argument("--summarize", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
builder = FT.builder
BOUNDS = (0.5, 0.3, 0.2, 0.1, 0.005)
GRID = 0.01
FIT_MAX, CLASH_MAX, MLP_BYTES, HDR = 1000, 250, 5960, 8
PRED = ("c12", "c1234", "raw", "mean")  # + "hyb" (handled separately)


# ------------------------------------------------------------------ entropy coder
def zigzag(q):
    q = q.astype(np.int64)
    return ((q << 1) ^ (q >> 63)).astype(np.uint64)


def unzigzag(z):
    z = z.astype(np.int64)
    return (z >> 1) ^ -(z & 1)


def varint_enc(z):
    z = z.ravel()
    nb = 1 + sum((z >= (1 << (7 * k))).astype(np.int64) for k in range(1, 5))
    out = np.zeros(int(nb.sum()), np.uint8); pos = np.concatenate([[0], np.cumsum(nb)[:-1]])
    for k in range(5):
        sel = nb > k
        if not sel.any(): break
        out[pos[sel] + k] = (((z[sel] >> np.uint64(7 * k)) & np.uint64(0x7F)) | ((nb[sel] > k + 1).astype(np.uint64) << np.uint64(7))).astype(np.uint8)
    return out.tobytes()


def varint_dec(b, n):
    u = np.frombuffer(b, np.uint8).astype(np.uint64)
    end = np.where(u < 128)[0]; start = np.concatenate([[0], end[:-1] + 1])
    assert len(end) == n
    out = np.zeros(n, np.uint64)
    for k in range(5):
        sel = start + k <= end
        if not sel.any(): break
        out[sel] |= (u[start[sel] + k] & np.uint64(0x7F)) << np.uint64(7 * k)
    return out


def planes_enc(z):
    z = z.ravel()
    if z.max(initial=0) < 256: return b"\x01" + z.astype(np.uint8).tobytes()
    assert z.max() < 65536
    v = z.astype(np.uint16); return b"\x02" + (v & 0xFF).astype(np.uint8).tobytes() + (v >> 8).astype(np.uint8).tobytes()


def planes_dec(b, n):
    if b[0] == 1: return np.frombuffer(b[1:], np.uint8).astype(np.uint64)
    lo = np.frombuffer(b[1:1 + n], np.uint8).astype(np.uint64); hi = np.frombuffer(b[1 + n:], np.uint8).astype(np.uint64)
    return lo | (hi << np.uint64(8))


COMP = {"zlib": (lambda b: zlib.compress(b, 9), zlib.decompress),
        "lzma": (lambda b: lzma.compress(b, preset=9), lzma.decompress)}
SER = {"vint": (varint_enc, varint_dec), "planes": (planes_enc, planes_dec)}
CODERS = [f"{t}-{l}-{s}-{c}" for t in ("none", "dt") for l in ("fna", "nat") for s in SER for c in COMP]
MAIN_CODERS = ["none-nat-vint-lzma", "none-nat-planes-lzma", "dt-nat-vint-lzma", "dt-nat-planes-lzma", "dt-fna-vint-lzma", "dt-fna-planes-lzma"]


def _arrange(q, t, l):
    if t == "dt": q = np.concatenate([q[:1], np.diff(q, axis=0)], 0)
    return np.ascontiguousarray(q if l == "fna" else q.transpose(1, 2, 0))


def encode(q, coders):
    """q int [T,N,3] -> (best name, bytes, sizes of every candidate)."""
    sizes, best = {}, None
    for name in coders:
        t, l, s, c = name.split("-")
        z = zigzag(_arrange(q, t, l))
        if s == "planes" and z.max(initial=0) >= 65536: continue
        b = COMP[c][0](SER[s][0](z)); sizes[name] = len(b)
        if best is None or len(b) < len(best[1]): best = (name, b)
    return best[0], best[1], sizes


def decode(name, b, shape):
    t, l, s, c = name.split("-")
    T, N, _ = shape; n = T * N * 3
    q = unzigzag(SER[s][1](COMP[c][1](b), n))
    q = q.reshape((T, N, 3) if l == "fna" else (N, 3, T))
    if l == "nat": q = q.transpose(2, 0, 1)
    if t == "dt": q = np.cumsum(q, 0)
    return q


def quantise(x, p, b):
    """x, p float64 [T,N,3]; returns integer residual and reconstruction (|rec - x| <= b)."""
    if b == 0.005:  # lossless on the 0.01 Å source grid
        gp = np.round(p / GRID); q = (np.round(x / GRID) - gp).astype(np.int64)
        return q, lambda qq: (gp + qq) * GRID
    s = 2 * b; q = np.round((x - p) / s).astype(np.int64)
    return q, lambda qq: p + qq * s


# ------------------------------------------------------------------ helpers
def superpose(X, ref):
    Rk, tk = kabsch(X, ref[None].expand(X.shape[0], -1, 3)); return X @ Rk.transpose(-1, -2) + tk[:, None]


def iter_mean(X, n=5):
    ref = X[0]
    for _ in range(n): ref = superpose(X, ref).mean(0)
    return ref


@torch.no_grad()
def dyn_metrics(rec_f, x_f, ca):
    """RMSF (all heavy atoms, CA fit to own mean) and PCA overlap (CA, both on the true mean) of rec vs original."""
    def rmsf(X):
        ref = iter_mean(X[:, ca]); Rk, tk = kabsch(X[:, ca], ref[None].expand(X.shape[0], -1, 3))
        Xf = X @ Rk.transpose(-1, -2) + tk[:, None]
        return ((Xf - Xf.mean(0)) ** 2).sum(-1).mean(0).sqrt()
    ft, fr = rmsf(x_f), rmsf(rec_f)
    mt = iter_mean(x_f[:, ca]); A = superpose(x_f[:, ca], mt); B = superpose(rec_f[:, ca], mt)

    def pcs(F_):
        Z = (F_ - F_.mean(0)).reshape(F_.shape[0], -1).double(); ev, V = torch.linalg.eigh(Z.T @ Z / (len(Z) - 1))
        return ev.flip(0), V.flip(1)
    et, Vt = pcs(A); er, Vr = pcs(B); k = 10
    return dict(rmsf_r=float(np.corrcoef(ft.numpy(), fr.numpy())[0, 1]), rmsf_mae=float((ft - fr).abs().mean()),
                rmsf_ratio=float(fr.sum() / ft.sum()), rmsip10=float(((Vt[:, :k].T @ Vr[:, :k]) ** 2).sum().div(k).sqrt()),
                pca_var_ratio=float(er.sum() / et.sum()))


@torch.no_grad()
def full_metrics(rec, x, tchi, dom, cls, st4, mask4):
    """rec, x [T,R,14,3] float32 (lab frame, original atom labels)."""
    T, R = x.shape[:2]; mk = dom.atom_mask.bool()
    pchi, _ = builder.chis_from_atom14(rec, dom.aatype)
    rm = S.residue_metrics(rec, pchi, x, tchi, dom, FT_tl(x, dom), 1.0)
    old = S.CLASSES; S.CLASSES = tuple(cls); sm = S.summarise(rm, cls, dom.chi_mask.numpy()); S.CLASSES = old
    ci = torch.as_tensor(np.unique(np.linspace(0, T - 1, min(T, CLASH_MAX)).astype(int)))
    sm["clash"] = L.clash_score(rec[ci], dom.aatype, dom.atom_mask, dom.chain_index)
    d = rec - x; dm = d[:, mk]  # [T,N,3]
    ds = rec - L._swapped(x, dom.aatype); e1 = (d ** 2).sum(-1); e2 = (ds ** 2).sum(-1)
    m = dom.atom_mask[None].expand(T, R, 14)
    per_res = torch.minimum((e1 * m).sum(-1), (e2 * m).sum(-1))  # symmetry aware, lab frame, no fit
    sm["all_atom_rmsd_lab"] = float((per_res.sum(1) / m.sum((1, 2))).sqrt().mean())
    bbm = m[..., :4]
    sm["bb_rmsd_lab"] = float((((d[..., :4, :] ** 2).sum(-1) * bbm).sum((1, 2)) / bbm.sum((1, 2))).sqrt().mean())
    sm["max_abs_coord_err"] = float(dm.abs().max()); sm["max_atom_err"] = float(dm.norm(dim=-1).max())
    sm["max_atom_err_sym"] = float(torch.minimum(e1, e2)[m > 0].max().sqrt())
    sm["bb_rmsd_fit"] = bb_rmsd(rec[:, :, :4].reshape(T, R * 4, 3), x, dom)
    # chi3 / chi4 accuracy (pi-periodic aware), unstable residues, all residues, and the chis coded in c1234
    err = L.chi_abs_err_deg(pchi, tchi, dom.aatype).numpy(); cm = dom.chi_mask.numpy() > 0
    ps, _ = RC.chi_states(S.ang(pchi.numpy()), dom.aatype.numpy(), dom.chi_mask.numpy(), nk=4)
    acc = {}
    for grp, sel in (("unstable", cls["unstable"]), ("all", cls["all"])):
        for k in (2, 3):
            ok = sel & cm[:, k]
            if ok.sum(): acc[f"{grp}_chi{k+1}"] = dict(n=int(ok.sum()), within40=float((err[:, ok, k] < 40).mean()), mae=float(err[:, ok, k].mean()))
    for k in (2, 3):
        ok = mask4[:, k]
        if ok.sum():
            acc[f"coded_chi{k+1}"] = dict(n=int(ok.sum()), within40=float((err[:, ok, k] < 40).mean()), mae=float(err[:, ok, k].mean()),
                                          state_agree=float((ps[:, ok, k] == st4[:, ok, k]).mean()))
    sm["chi34"] = acc
    ca = (torch.arange(14)[None].expand(R, 14))[mk] == 1
    sm["dyn"] = dyn_metrics(rec[:, mk], x[:, mk], ca)
    return sm


def FT_tl(x, dom):
    Rt, tt = S.frames_from_backbone(x[..., 0, :], x[..., 1, :], x[..., 2, :])
    tl = S.to_local(x, Rt, tt); return S.canon_local(tl, tl[:1], dom.aatype)


@torch.no_grad()
def bb_rmsd(pbb, x, dom):
    T, R = x.shape[:2]; mk = dom.atom_mask[None, :, :4].expand(T, R, 4).reshape(T, -1)
    Rk, tk = S.kabsch(pbb, x[:, :, :4].reshape(T, -1, 3), mk)
    P = pbb @ Rk.transpose(-1, -2) + tk[:, None]
    return float(((((P - x[:, :, :4].reshape(T, -1, 3)) ** 2).sum(-1) * mk).sum(1) / mk.sum(1)).sqrt().mean())


@torch.no_grad()
def learned(m, dom, x, tchi, st4, ns4, K, n_chi_res):
    """codes over chi1..chiK + frozen model + refitted residual MLP -> lab-frame prediction and byte counts."""
    aat = dom.aatype.numpy()
    mask = RC.build_code(st4[:, :, :K], ns4[:, :K], tuple(range(K))); coded = mask.any(1)
    j = RC.joint(st4[:, :, :K], mask)
    tab, seen = RC.state_table(tchi, aat, j, coded, dom.chi_ref, 3 ** K)
    code = RC.code_bytes(st4[:, :, :K], ns4[:, :K], mask, n_chi_res, seen)
    cref = RC.cref_frames(tab, j, coded, dom.chi_ref)
    fe = RC.frozen_cond(m, dom, x, cref)
    T = len(x); fi = np.sort(np.random.default_rng(0).choice(T, min(T, FIT_MAX), replace=False))
    FT.a.batch = min(48, len(fi))
    ff = fe if len(fi) == T else {k: v[torch.as_tensor(fi)] for k, v in fe.items()}
    with torch.enable_grad():
        mod, sec, _ = FT.fit("mlp", m, ff, x[fi], dom)
    pc = mod(fe["h"], fe["base"], None, dom.chi_mask.float())
    pred = FT.build(m, fe, pc, dom) * dom.atom_mask[..., None]
    code["latent_bytes_fp16"] = 8 * dom.R * T; code["latent_bytes_lzma"] = latent_lzma(m, dom, x)
    base = T * (8 * dom.R + 28) + 80 * dom.R + code["zlib_bytes_total"] + code["mask_bytes"] + code["state_table_bytes"] + MLP_BYTES
    code.update(n_coded_chi3=int(mask[:, 2].sum()) if K > 2 else 0, n_coded_chi4=int(mask[:, 3].sum()) if K > 3 else 0)
    return pred, base, code, round(sec, 1), mask, cref


@torch.no_grad()
def latent_lzma(m, dom, x):
    """lossless lzma of the fp16 latent stream (best of frame-major / latent-major, plain / byte planes)."""
    from common import bind
    bind(m, dom); zs = []
    for s in range(0, len(x), 50):
        bb_al, _, _ = m.codec.align(x[s:s + 50, :, :4]); zs.append(m.codec.encode(bb_al, dom.aatype).half())
    z = torch.cat(zs).numpy().astype(np.float16)
    best = None
    for arr in (z, np.ascontiguousarray(z.transpose(1, 2, 0))):
        u = arr.view(np.uint16)
        for b in (u.tobytes(), (u & 0xFF).astype(np.uint8).tobytes() + (u >> 8).astype(np.uint8).tobytes()):
            n = len(lzma.compress(b, preset=9)); best = n if best is None else min(best, n)
    return best


@torch.no_grad()
def cond_from_bb(m, dom, bb, cref):
    """frozen packer on a given backbone [T,R,4,3] (lab frame) with per-frame chi prior; same dict as frozen_cond."""
    from common import bind
    bind(m, dom); BB, H, C, RR, TT = [], [], [], [], []
    for s in range(0, len(bb), 25):
        m.codec.chi_ref = cref[s:s + 25]
        bb_al, R_, t_ = m.codec.align(bb[s:s + 25])
        outs = m.pack(bb_al, dom.aatype, dom.chain_index, dom.chain_break)
        BB.append(bb_al); H.append(outs[-1]["h"]); C.append(outs[-1]["chi_sc"]); RR.append(R_); TT.append(t_)
    m.codec.chi_ref = dom.chi_ref
    return dict(bb=torch.cat(BB), h=torch.cat(H), base=torch.cat(C), R=torch.cat(RR), t=torch.cat(TT))


def swap_fix(pred, x, dom):
    """per frame and symmetric residue choose the labelling of pred closest to x; returns pred and swap-bit bytes."""
    sw = L._swapped(pred, dom.aatype); m = dom.atom_mask[None, ..., None]
    d1 = (((pred - x) ** 2) * m).sum((-1, -2)); d2 = (((sw - x) ** 2) * m).sum((-1, -2))
    perm = L.tables()["swap_perm"][dom.aatype]; sym = (perm != torch.arange(14)).any(-1)
    use = (d2 < d1) & sym[None]
    out = torch.where(use[..., None, None], sw, pred)
    bits = use[:, sym].numpy().astype(np.uint8)
    nbytes = len(zlib.compress(np.packbits(bits.T, axis=-1).tobytes(), 9)) if sym.any() else 0
    return out, nbytes, float(bits.mean()) if bits.size else 0.0


def run_domain(d, m, raw, coders):
    z = np.load(f"{a.data}/{d}.npz")
    dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], "cpu")
    arch = np.where(z["heldout"] == 0)[0]
    Tmax = max(int(t) for t in a.levels.split(","))
    X = torch.as_tensor(z["coords"][arch[:max(Tmax, 1)]]); temps, reps = z["temp"][arch], z["replica"][arch]
    del z
    aat = dom.aatype.numpy(); mk = dom.atom_mask.bool(); N = int(mk.sum())
    r0 = np.where((temps[:len(X)] == 320) & (reps[:len(X)] == 0))[0]
    tc0, _ = builder.chis_from_atom14(X[r0], dom.aatype)
    fit_buffers(dom, X[r0], builder); cmn = dom.chi_mask.numpy(); has = cmn[:, 0] > 0
    hs = S.hysteresis_states(S.ang(tc0.numpy()), aat, cmn); occ, rate = S.occ_and_rate(hs, 1.0)
    stb = has & (occ >= S.OCC_STABLE) & (rate <= S.MAX_STABLE_RATE); un = has & (occ < S.OCC_UNSTABLE)
    cls = dict(stable=stb, intermediate=has & ~stb & ~un, unstable=un, all=has)
    n_chi_res = cmn.sum(1).astype(int)
    for T in [int(t) for t in a.levels.split(",")]:
        key = f"{d}/T{T}"
        if key in raw or T > len(X): continue
        t0 = time.time()
        x = X[:T].contiguous(); tchi, _ = builder.chis_from_atom14(x, dom.aatype)
        fit_buffers(dom, x, builder)
        st4, ns4 = RC.chi_states(S.ang(tchi.numpy()), aat, cmn, nk=4)
        raw_bytes = 12 * N * T
        r = dict(domain=d, T=T, n_res=dom.R, n_heavy=N, raw_bytes=raw_bytes, class_counts={c: int(v.sum()) for c, v in cls.items()},
                 base_bytes={}, code={}, fit_seconds={}, swap={}, variants={})
        x64 = x[:, mk].double().numpy()
        preds = {}
        mask4 = np.zeros((dom.R, 4), bool)
        for v, K in (("c12", 2), ("c1234", 4)):
            pred, base, code, sec, mask, cref = learned(m, dom, x, tchi, st4, ns4, K, n_chi_res)
            if K == 4: mask4, cref4 = mask, cref
            r["base_bytes"][v], r["code"][v], r["fit_seconds"][v] = base, code, sec
            preds[v] = pred
        for v in ("c12", "c1234"):  # model only (no residuals)
            sm = full_metrics(preds[v], x, tchi, dom, cls, st4, mask4)
            r["variants"][f"{v}"] = dict(bytes=r["base_bytes"][v], ratio=raw_bytes / r["base_bytes"][v], metrics=sm)
        # mean-structure predictor (model free): mean of the archive, rigid fit per frame on all heavy atoms
        xf = x[:, mk]; M = iter_mean(xf).float()
        Rk, tk = kabsch(M[None].expand(T, -1, 3), xf); pm = M[None] @ Rk.transpose(-1, -2) + tk[:, None]
        pmean = torch.zeros_like(x); pmean[:, mk] = pm
        r["base_bytes"]["mean"] = 12 * N + 28 * T; r["base_bytes"]["raw"] = 0
        sources = {}
        for v in ("c12", "c1234"):
            pf, nb, frac = swap_fix(preds[v], x, dom); r["swap"][v] = dict(bytes=nb, frac_swapped=frac)
            sources[v] = (pf[:, mk].double().numpy(), r["base_bytes"][v] + nb)
        sources["raw"] = (np.zeros_like(x64), 0); sources["mean"] = (pm.double().numpy(), r["base_bytes"]["mean"])
        for v in PRED:
            p, base = sources[v]
            for b in BOUNDS:
                q, recf = quantise(x64, p, b)
                name, blob, sizes = encode(q, coders)
                qd = decode(name, blob, q.shape); assert np.array_equal(qd, q), (key, v, b, name)
                rec64 = recf(qd); err = np.abs(rec64 - x64).max()
                rec = torch.zeros_like(x); rec[:, mk] = torch.as_tensor(rec64, dtype=torch.float32)
                sm = full_metrics(rec, x, tchi, dom, cls, st4, mask4)
                tot = base + len(blob) + HDR
                r["variants"][f"{v}@{b}"] = dict(bytes=tot, residual_bytes=len(blob), ratio=raw_bytes / tot, coder=name, coder_sizes=sizes,
                                                  max_abs_err_f64=float(err), frac_zero=float((q == 0).mean()),
                                                  q_absmax=int(np.abs(q).max()), metrics=sm)
                print(f"   {key} {v}@{b}: {raw_bytes / tot:.2f}x resid {len(blob) / T:.0f} B/fr {name} max {err:.4f} rmsd {sm['all_atom_rmsd_lab']:.3f}", flush=True)
        # hybrid: backbone atoms coded against the mean-structure predictor at bound b; side chains predicted by the
        # frozen packer from that decoded backbone (chi1..chi4 codes + MLP fitted on the true backbone), then residuals.
        slot = torch.arange(14)[None].expand(dom.R, 14)
        mbb = mk & (slot < 4); msc = mk & (slot >= 4)
        ft = cond_from_bb(m, dom, x[:, :, :4].contiguous(), cref4)
        with torch.enable_grad():
            FT.a.batch = min(48, T); modh, sech, _ = FT.fit("mlp", m, ft, x, dom)
        r["fit_seconds"]["hyb"] = round(sech, 1); del ft
        c4 = r["code"]["c1234"]
        hyb_fixed = 12 * int(mbb.sum()) + 28 * T + 80 * dom.R + c4["zlib_bytes_total"] + c4["mask_bytes"] + c4["state_table_bytes"] + MLP_BYTES
        r["base_bytes"]["hyb"] = hyb_fixed
        xbb64 = x[:, mbb].double().numpy(); xsc64 = x[:, msc].double().numpy(); pbb64 = pmean[:, mbb].double().numpy()
        for b in BOUNDS:
            q1, rf1 = quantise(xbb64, pbb64, b); n1, blob1, _ = encode(q1, coders)
            assert np.array_equal(decode(n1, blob1, q1.shape), q1)
            bbr = torch.zeros(T, dom.R, 4, 3); bbr[:, mbb[:, :4]] = torch.as_tensor(rf1(q1), dtype=torch.float32)
            fh = cond_from_bb(m, dom, bbr, cref4)
            with torch.no_grad():
                pch = modh(fh["h"], fh["base"], None, dom.chi_mask.float())
                ph = FT.build(m, fh, pch, dom) * dom.atom_mask[..., None]
            ph, nbs, frac = swap_fix(ph, x, dom)
            q2, rf2 = quantise(xsc64, ph[:, msc].double().numpy(), b); n2, blob2, _ = encode(q2, coders)
            assert np.array_equal(decode(n2, blob2, q2.shape), q2)
            rec = torch.zeros_like(x); rec[:, mbb] = torch.as_tensor(rf1(q1), dtype=torch.float32)
            rec[:, msc] = torch.as_tensor(rf2(q2), dtype=torch.float32)
            err = max(np.abs(rf1(q1) - xbb64).max(), np.abs(rf2(q2) - xsc64).max())
            sm = full_metrics(rec, x, tchi, dom, cls, st4, mask4)
            tot = hyb_fixed + len(blob1) + len(blob2) + nbs + 2 * HDR
            r["variants"][f"hyb@{b}"] = dict(bytes=tot, residual_bytes=len(blob1) + len(blob2), bb_residual_bytes=len(blob1),
                                              sc_residual_bytes=len(blob2), swap_bytes=nbs, ratio=raw_bytes / tot, coder=[n1, n2],
                                              max_abs_err_f64=float(err), frac_zero_sc=float((q2 == 0).mean()), metrics=sm)
            print(f"   {key} hyb@{b}: {raw_bytes / tot:.2f}x bb {len(blob1) / T:.0f} sc {len(blob2) / T:.0f} B/fr max {err:.4f}", flush=True)
        r["seconds"] = round(time.time() - t0, 1)
        raw[key] = r
        json.dump(raw, open(a.raw + ".tmp", "w")); os.replace(a.raw + ".tmp", a.raw)
        f = lambda v: math.sqrt(r["variants"][v]["metrics"]["all"]["sc_err"] / r["variants"][v]["metrics"]["all"]["n_sc"])
        print(time.strftime("%H:%M:%S"), key, f"{r['seconds']}s", " ".join(f"{v} {r['variants'][v]['ratio']:.2f}x sc {f(v):.3f}" for v in ("c12", "c1234")), flush=True)


def run():
    m = FT.load_main()
    raw = json.load(open(a.raw)) if os.path.exists(a.raw) else {}
    doms = json.load(open("results_gpu/frames/selected_domains.json"))
    only = [x for x in a.only.split(",") if x]
    coders = CODERS if a.coders == "all" else MAIN_CODERS if a.coders == "main" else a.coders.split(",")
    for d in (only or doms):
        if not os.path.exists(f"{a.data}/{d}.npz"): print("missing", d, flush=True); continue
        run_domain(d, m, raw, coders)
    print("done", flush=True)


if __name__ == "__main__" and not a.summarize:
    run()


# ------------------------------------------------------------------ summary + figure
FAMS = dict(c12="χ1+χ2 codes + model + MLP, then residuals", c1234="χ1 to χ4 codes + model + MLP, then residuals",
            hyb="hybrid: backbone vs mean structure, side chains from packer (χ1 to χ4 codes + MLP), then residuals",
            raw="model free: quantised raw coordinates", mean="model free: mean structure + rigid fit, then residuals")
SHORT = dict(c12="χ1+χ2 codes + model + MLP + residuals", c1234="χ1 to χ4 codes + model + MLP + residuals",
             hyb="hybrid (backbone vs mean, packer side chains) + residuals", raw="model free: raw quantised", mean="model free: mean + rigid fit + residuals")
COLF = dict(c12="tab:red", c1234="tab:purple", hyb="tab:green", raw="#333333", mean="tab:blue")


def load_raw():
    import glob
    raw = {}
    for f in sorted(glob.glob(a.raw.replace(".json", "_part*.json"))) + ([a.raw] if os.path.exists(a.raw) else []):
        raw.update(json.load(open(f)))
    return raw


def pool_variant(rows, v):
    ms = [r["variants"][v]["metrics"] for r in rows]
    sc = lambda c: math.sqrt(sum(m[c]["sc_err"] for m in ms if m[c].get("n_res")) / max(sum(m[c]["n_sc"] for m in ms if m[c].get("n_res")), 1e-9))
    raw_b = [r["raw_bytes"] for r in rows]; st = [r["variants"][v]["bytes"] for r in rows]
    o = dict(ratio_mean=float(np.mean([x / y for x, y in zip(raw_b, st)])), ratio_pooled=sum(raw_b) / sum(st),
             bytes_per_frame_mean=float(np.mean([y / r["T"] for y, r in zip(st, rows)])),
             all_atom_rmsd=float(np.mean([m["all_atom_rmsd_lab"] for m in ms])),
             sc_rmsd=dict(all=sc("all"), stable=sc("stable"), unstable=sc("unstable")),
             bb_rmsd=float(np.mean([m["bb_rmsd_fit"] for m in ms])), bb_rmsd_lab=float(np.mean([m["bb_rmsd_lab"] for m in ms])),
             max_abs_coord_err=float(max(m["max_abs_coord_err"] for m in ms)), max_atom_err=float(max(m["max_atom_err"] for m in ms)),
             max_atom_err_sym=float(max(m["max_atom_err_sym"] for m in ms)),
             clash=float(np.mean([m["clash"] for m in ms])),
             **{k: float(np.mean([m["dyn"][k] for m in ms])) for k in ("rmsf_r", "rmsf_mae", "rmsf_ratio", "rmsip10", "pca_var_ratio")})
    chi = {}
    for g in ("unstable_chi3", "unstable_chi4", "coded_chi3", "coded_chi4"):
        xs = [m["chi34"][g] for m in ms if g in m["chi34"]]
        if xs:
            n = sum(x["n"] for x in xs); chi[g] = dict(n_chis=n, within40=sum(x["within40"] * x["n"] for x in xs) / n, mae=sum(x["mae"] * x["n"] for x in xs) / n)
            if "state_agree" in xs[0]: chi[g]["state_agree"] = sum(x["state_agree"] * x["n"] for x in xs) / n
    o["chi34"] = chi
    rv = [r["variants"][v] for r in rows]
    if "residual_bytes" in rv[0]:
        o["residual_bytes_per_frame_mean"] = float(np.mean([x["residual_bytes"] / r["T"] for x, r in zip(rv, rows)]))
        o["max_abs_err_f64"] = float(max(x["max_abs_err_f64"] for x in rv))
        cs = [x["coder"] if isinstance(x["coder"], str) else "+".join(x["coder"]) for x in rv]
        o["coders"] = {c: cs.count(c) for c in sorted(set(cs))}
    if v.split("@")[0] in ("c12", "c1234"):  # same predictions, latents losslessly lzma coded
        fam = v.split("@")[0]
        st2 = [y - r["code"][fam]["latent_bytes_fp16"] + r["code"][fam]["latent_bytes_lzma"] for y, r in zip(st, rows)]
        o["ratio_mean_lzma_latents"] = float(np.mean([x / y for x, y in zip(raw_b, st2)]))
    return o


def summarize():
    raw = load_raw(); levels = sorted({r["T"] for r in raw.values()}, reverse=True)
    summ = {}
    for T in levels:
        rows = sorted([r for r in raw.values() if r["T"] == T], key=lambda r: r["domain"])
        e = dict(n_domains=len(rows), domains=[r["domain"] for r in rows], variants={},
                 code={f: {k: float(np.mean([r["code"][f][k] for r in rows])) for k in rows[0]["code"][f]} for f in ("c12", "c1234")},
                 swap={f: float(np.mean([r["swap"][f]["frac_swapped"] for r in rows])) for f in ("c12", "c1234")})
        for v in rows[0]["variants"]:
            e["variants"][v] = pool_variant(rows, v)
        # does a learned variant beat the best model-free reference? (per domain, same bound)
        cmp_ = {}
        for b in BOUNDS:
            ref = [max(r["variants"][f"raw@{b}"]["ratio"], r["variants"][f"mean@{b}"]["ratio"]) for r in rows]
            for f in ("c12", "c1234", "hyb"):
                lv = [r["variants"][f"{f}@{b}"]["ratio"] for r in rows]
                cmp_[f"{f}@{b}"] = dict(wins=int(sum(x > y for x, y in zip(lv, ref))), n=len(rows),
                                        gain_mean=float(np.mean([x / y for x, y in zip(lv, ref)])),
                                        gain_min=float(min(x / y for x, y in zip(lv, ref))), gain_max=float(max(x / y for x, y in zip(lv, ref))))
        e["learned_vs_best_model_free"] = cmp_
        e["per_domain"] = {r["domain"]: dict(n_res=r["n_res"], n_heavy=r["n_heavy"], ratio={v: r["variants"][v]["ratio"] for v in r["variants"]})
                           for r in rows}
        summ[T] = e
    out = dict(description=__doc__, bounds=BOUNDS, levels=levels, summary=summ)
    json.dump(out, open(a.out, "w"), indent=1)
    figure(summ); tables(summ)


def tables(summ):
    L_ = []
    for T in summ:
        e = summ[T]["variants"]
        L_.append(f"\n### T = {T} ({summ[T]['n_domains']} domains)\n")
        L_.append("| variant | bound (Å) | ratio | all atom RMSD (Å) | side chain all / stable / unstable (Å) | backbone (Å) | max coord err (Å) | RMSF r | PCA RMSIP | unstable χ3 within 40° |")
        L_.append("|---|---|---|---|---|---|---|---|---|---|")
        for v, s in e.items():
            fam, b = (v.split("@") + ["none"])[:2]
            c3 = s["chi34"].get("unstable_chi3", {}).get("within40", float("nan"))
            L_.append(f"| {fam} | {b} | {s['ratio_mean']:.2f} | {s['all_atom_rmsd']:.3f} | {s['sc_rmsd']['all']:.3f} / {s['sc_rmsd']['stable']:.3f} / {s['sc_rmsd']['unstable']:.3f} | "
                      f"{s['bb_rmsd']:.3f} | {s['max_abs_coord_err']:.3f} | {s['rmsf_r']:.3f} | {s['rmsip10']:.3f} | {c3:.2f} |")
        L_.append("\n| learned variant | wins vs best model free | mean ratio gain | min | max |\n|---|---|---|---|---|")
        for k, c in summ[T]["learned_vs_best_model_free"].items():
            L_.append(f"| {k} | {c['wins']}/{c['n']} | {c['gain_mean']:.3f} | {c['gain_min']:.3f} | {c['gain_max']:.3f} |")
    open("results_gpu/residuals/tables.md", "w").write("\n".join(L_) + "\n")


def figure(summ):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    Ts = sorted(summ, reverse=True); T0 = Ts[0]
    bx = [b for b in BOUNDS]
    for T, ls in zip(Ts, ("-", "--")):
        e = summ[T]["variants"]
        for f in FAMS:
            ax[0].plot(range(len(bx)), [e[f"{f}@{b}"]["ratio_mean"] for b in bx], ls, marker="o", ms=4, c=COLF[f], label=SHORT[f] if T == T0 else None)
    ax[0].set(xlabel="error bound b: max abs error per coordinate (Å); 0.005 = lossless at 0.01 Å", ylabel="compression ratio (×)",
              title=f"A  Ratio vs error bound (solid T = {T0}" + (f", dashed T = {Ts[1]})" if len(Ts) > 1 else ")"))
    ax[0].set_xticks(range(len(bx))); ax[0].set_xticklabels([str(b) for b in bx])
    e = summ[T0]["variants"]
    for f in FAMS:
        ax[1].plot([e[f"{f}@{b}"]["all_atom_rmsd"] for b in bx], [e[f"{f}@{b}"]["ratio_mean"] for b in bx], "o-", ms=4, c=COLF[f])
    for f, mk_ in (("c12", "s"), ("c1234", "D")):
        ax[1].scatter(e[f]["all_atom_rmsd"], e[f]["ratio_mean"], marker=mk_, s=70, c=COLF[f], zorder=3,
                      label=f"{'χ1+χ2' if f == 'c12' else 'χ1 to χ4'} codes + model + MLP, no residuals (max err {e[f]['max_abs_coord_err']:.1f} Å)")
    ax[1].set(xlabel="all atom RMSD, lab frame, no fit (Å); 0 = lossless", ylabel="compression ratio (×)", title=f"B  Ratio vs RMSD, T = {T0}")
    ax[1].legend(fontsize=7, loc="lower right")
    pdm = summ[T0]["per_domain"]
    for f, off in (("hyb", -0.15), ("c1234", 0.15)):
        for i, b in enumerate(bx):
            g = [pdm[d]["ratio"][f"{f}@{b}"] / max(pdm[d]["ratio"][f"raw@{b}"], pdm[d]["ratio"][f"mean@{b}"]) for d in pdm]
            ax[2].scatter(np.full(len(g), i + off) + np.random.default_rng(i).uniform(-0.05, 0.05, len(g)), g, s=12, c=COLF[f],
                          label=("hybrid" if f == "hyb" else "χ1 to χ4 codes + model + MLP + residuals") if i == 0 else None)
    ax[2].axhline(1, c="k", lw=0.8); ax[2].set_xticks(range(len(bx))); ax[2].set_xticklabels([str(b) for b in bx])
    ax[2].set(xlabel="error bound b (Å)", ylabel="ratio / best model free ratio (per domain)", title=f"C  Gain over the best model free coder, T = {T0} (>1: model helps)")
    for A_ in ax: A_.grid(alpha=0.3); A_.title.set_fontsize(9)
    ax[0].legend(fontsize=7, loc="upper right"); ax[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig("report/gpu_figs/fig_error_bounds.png", dpi=200)
    import shutil; shutil.copy("report/gpu_figs/fig_error_bounds.png", "results_gpu/fig_error_bounds.png")


if a.summarize:
    summarize()
