"""md_compress codec: error bounded (or lossless) compression of protein MD trajectories.

    from md_compress.codec import compress, decompress
    blob = compress(coords, aatype, atom_mask, chain_index, chain_break, bound=0.1)     # bytes
    out  = decompress(blob)                                                             # dict, out["coords"]

Modes (stored in one flag byte of the header, so decompress needs no arguments):
  default            backbone (and extra atoms) coded against the per trajectory mean structure, side chains against the
                     SHARED packer's prediction from the decoded backbone (static per residue chi prior), residuals of both
                     streams coded with the context modelled arithmetic coder (gpu/residual_entropy.c).
  backbone="ae"      optional / experimental: backbone coded against the shared backbone autoencoder (fp16 latents stored).
                     In the benchmark it lost at every bound (latents cost more than they save); kept for experiments.
  full_predictor     report mode: per frame rotamer codes + per trajectory residual MLP (5,960 B, fp16) on top of the
                     packer. Pays only for long trajectories (about 1,000 frames and more); see report section 4.7.6.
Bounds: bound in Angstrom (every coordinate within bound + 1e-3 of the input), or bound=0 / "lossless": exact on the 0.01 A
grid (max error 0.005 A; exact when the input lies on that grid, as for mdCATH and MDDB coordinates).
Decoding is bit exact with the same build (the coder's geometric term and the packer use floating point).

Other methods (method=..., header version 2 records the method id and its options; see md_compress/methods.py and
`python -m md_compress.codec list-methods`): xtc, tng, mdcompress, new-coder (model free + our coder), lzma, zstd, bz2.
compress_xyz(xyz [T, N, 3], method, bound) codes any atoms (no protein topology) with those methods.

Input layout: atom14 coordinates [T, R, 14, 3] in Angstrom with atom_mask [R, 14] (as md_compress.data.load_trajectory),
plus optional extra atoms [T, M, 3] (non protein heavy atoms, OXT, ...) coded with the backbone stream.
"""
import io, json, lzma, os, struct, sys, time, zlib
import numpy as np, torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GPU = os.path.join(_ROOT, "gpu")
MAGIC, VERSION = b"MDZ1", 1
F_AE, F_FULL, F_LZMA, F_NORES = 1, 2, 4, 8
_E = {}


def _env(threads=4):
    """lazy import of the research modules (they parse sys.argv and use paths relative to the repository root)."""
    if not _E:
        if _GPU not in sys.path: sys.path.insert(0, _GPU)
        if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
        argv, cwd = sys.argv, os.getcwd()
        sys.argv = [argv[0] if argv else "md_compress", "--threads", str(threads)]; os.chdir(_ROOT)
        try:
            import residual_coding as RCD
            import residual_entropy as RE
            from common import Domain, fit_buffers, bind
            from md_compress.geometry import kabsch
            from md_compress import losses as L
            m = RCD.FT.load_main()
        finally:
            sys.argv = argv; os.chdir(cwd)
        _E.update(RCD=RCD, RE=RE, FT=RCD.FT, RC=RCD.RC, S=RCD.S, Domain=Domain, fit_buffers=fit_buffers, bind=bind, kabsch=kabsch, L=L, m=m)
    torch.set_num_threads(threads)
    return _E


# ---------------------------------------------------------------- small helpers
def _rot_to_quat(R):
    R = np.asarray(R, np.float64); q = np.zeros((len(R), 4))
    for i, M in enumerate(R):
        tr = np.trace(M)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2; q[i] = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2; q[i] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2; q[i] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2; q[i] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def _quat_to_rot(q):
    q = np.asarray(q, np.float64); q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


class _W:
    def __init__(self): self.b = io.BytesIO()
    def sec(self, data): self.b.write(struct.pack("<I", len(data))); self.b.write(data)
    def arr(self, a, dt): self.sec(np.ascontiguousarray(a, dt).tobytes())


class _R:
    def __init__(self, data, off): self.d, self.o = data, off
    def sec(self):
        n, = struct.unpack_from("<I", self.d, self.o); self.o += 4; s = self.d[self.o:self.o + n]; self.o += n; return s
    def arr(self, dt, shape): return np.frombuffer(self.sec(), dt).reshape(shape).copy()


def _classes(atom_mask, M, extra_cls):
    mk = atom_mask > 0; slot = np.broadcast_to(np.arange(14), atom_mask.shape)[mk]
    c = np.select([slot < 4, slot == 4, slot == 5, slot < 8], [slot, 4, 5, 6], 7).astype(np.uint8)
    e = np.full(M, 11, np.uint8) if extra_cls is None else np.asarray(extra_cls, np.uint8)
    return np.concatenate([c, e])


def _lz_enc(RCD, q):
    name, blob, _ = RCD.encode(q, RCD.MAIN_CODERS); nb = name.encode(); return struct.pack("<B", len(nb)) + nb + blob


def _lz_dec(RCD, data, shape):
    n = data[0]; return RCD.decode(data[1:1 + n].decode(), data[1 + n:], shape)


def _qbase(p, b, RCD):
    if b == 0.005: return np.round(p / RCD.GRID) * RCD.GRID, RCD.GRID
    return p, 2 * b


# ---------------------------------------------------------------- shared prediction pieces (identical in encoder and decoder)
def _setup(E, aatype, atom_mask, chain_index, chain_break):
    dom = E["Domain"]("traj", aatype, atom_mask, chain_index, chain_break, int(np.asarray(atom_mask).sum()), "cpu")
    mk = dom.atom_mask.bool(); slot = torch.arange(14)[None].expand(dom.R, 14)
    mbb, msc = mk & (slot < 4), mk & (slot >= 4)
    return dom, mk, mbb, msc, mbb[mk].numpy(), msc[mk].numpy(), int(mbb.sum())


def _sym(E, dom):
    perm = E["L"].tables()["swap_perm"][dom.aatype]; return (perm != torch.arange(14)).any(-1)


class _fixed_threads:
    """predictions that feed the decoder run on PRED_THREADS torch threads in both encoder and decoder: intra op
    parallel reductions otherwise change the last float bits with the thread count, which moves the quantisation base."""
    def __enter__(self): self.n = torch.get_num_threads(); torch.set_num_threads(PRED_THREADS)
    def __exit__(self, *a): torch.set_num_threads(self.n)


PRED_THREADS = 1


@torch.no_grad()
def _sc_prediction(E, dom, dbb, mbb, n_bb, cref, mlp):
    """packer (optionally + MLP) prediction [T,R,14,3] from the decoded backbone dbb [T,n_bb,3] (float64)."""
    with _fixed_threads(): return _sc_prediction_(E, dom, dbb, mbb, n_bb, cref, mlp)


def _sc_prediction_(E, dom, dbb, mbb, n_bb, cref, mlp):
    T = dbb.shape[0]; bbr = torch.zeros(T, dom.R, 4, 3); bbr[:, mbb[:, :4]] = torch.as_tensor(dbb, dtype=torch.float32)
    fh = E["RCD"].cond_from_bb(E["m"], dom, bbr, cref)
    chi = mlp(fh["h"], fh["base"], None, dom.chi_mask.float()) if mlp is not None else fh["base"]
    return E["FT"].build(E["m"], fh, chi, dom) * dom.atom_mask[..., None]


@torch.no_grad()
def _ae_backbone(E, dom, z16, Rq, t):
    """shared backbone autoencoder: fp16 latents [T,R,4] -> lab frame backbone [T,R,4,3] (same chunks as the encoder)."""
    with _fixed_threads(): return _ae_backbone_(E, dom, z16, Rq, t)


def _ae_backbone_(E, dom, z16, Rq, t):
    E["bind"](E["m"], dom); BB = []
    for s in range(0, len(z16), 25):
        bb, _ = E["m"].decode_full(torch.as_tensor(z16[s:s + 25]).float(), dom.aatype, dom.chain_index, dom.chain_break); BB.append(bb)
    bb = torch.cat(BB)
    return E["m"].codec.unalign(bb, torch.as_tensor(Rq, dtype=torch.float32), torch.as_tensor(t))


def _mlp_from(E, w16):
    mod = E["FT"].MLP(); off = 0
    with torch.no_grad():
        for p in mod.parameters():
            n = p.numel(); p.copy_(torch.as_tensor(w16[off:off + n].astype(np.float32)).view_as(p)); off += n
    return mod.eval()


def _tab_from_angles(dom, coded_r, seen_c, ang16):
    """state table [n_coded, 81, 4, 2] (sin, cos) from fp16 chi angles of the seen states (chis beyond the residue's
    chi count and unseen states hold the static prior); identical in encoder and decoder."""
    nchi = dom.chi_mask.numpy().sum(1).astype(int)
    tab = np.zeros((len(coded_r), 81, 4, 2), np.float32); o = 0
    for i, r in enumerate(coded_r):
        tab[i] = dom.chi_ref[r].numpy()[None]
        for sidx in np.where(seen_c[i])[0]:
            a = ang16[o:o + nchi[r]].astype(np.float32); o += nchi[r]
            tab[i, sidx, :nchi[r], 0] = np.sin(a); tab[i, sidx, :nchi[r], 1] = np.cos(a)
    return tab


def _cref_from_codes(E, dom, T, coded_r, st_cols, cols, mask, tab16):
    RC = E["RC"]; R = dom.R
    st = np.zeros((T, R, mask.shape[1]), np.int64)
    for i, (r, k) in enumerate(cols): st[:, r, k] = st_cols[i]
    j = RC.joint(st, mask); tab = dom.chi_ref[:, None].repeat(1, 81, 1, 1).clone()
    if len(coded_r): tab[torch.as_tensor(coded_r)] = torch.as_tensor(tab16.astype(np.float32))
    coded = np.zeros(R, bool); coded[coded_r] = True
    return RC.cref_frames(tab, j, coded, dom.chi_ref)


# ---------------------------------------------------------------- public API
def _compress_hybrid(coords, aatype, atom_mask, chain_index, chain_break, bound=0.1, backbone="mean", full_predictor=False,
                     extra=None, extra_cls=None, threads=4, return_info=False, _keep_rec=False, coder="ac", residuals=True):
    """coords [T,R,14,3] (Angstrom); bound in Angstrom or 0 / 'lossless'; backbone 'mean' (default) or 'ae' (experimental);
    full_predictor: rotamer codes + per trajectory MLP (report mode). Returns bytes (and a size breakdown if return_info)."""
    E = _env(threads); RCD, RE, FT, RC, S, m = E["RCD"], E["RE"], E["FT"], E["RC"], E["S"], E["m"]
    b = 0.005 if bound in (0, "lossless", None) or float(bound) == 0 else float(bound)
    if backbone not in ("mean", "ae"): raise ValueError("backbone must be 'mean' or 'ae'")
    aatype = np.asarray(aatype, np.int64); atom_mask = np.asarray(atom_mask, np.float32)
    chain_index = np.asarray(chain_index, np.int64); chain_break = np.asarray(chain_break, bool)
    X = torch.as_tensor(np.ascontiguousarray(coords, np.float32)); T, R = X.shape[:2]
    Mx = 0 if extra is None else int(np.asarray(extra).shape[1])
    dom, mk, mbb, msc, bbsel, scsel, n_bb = _setup(E, aatype, atom_mask, chain_index, chain_break)
    Np = int(mk.sum())
    truth = np.concatenate([X[:, mk].numpy()] + ([np.asarray(extra, np.float32)] if Mx else []), 1).astype(np.float64)
    cls = _classes(atom_mask, Mx, extra_cls)
    if coder not in ("ac", "lzma"): raise ValueError("coder must be 'ac' or 'lzma'")
    flags = (F_AE if backbone == "ae" else 0) | (F_FULL if full_predictor else 0) | (F_LZMA if coder == "lzma" else 0) | (0 if residuals else F_NORES)
    W = _W(); info = {}
    W.b.write(MAGIC + struct.pack("<BBdIHI", VERSION, flags, b, T, R, Mx))
    topo = zlib.compress(np.concatenate([aatype.astype(np.uint8), np.packbits(atom_mask > 0), chain_index.astype(np.uint16).view(np.uint8),
                                         np.packbits(chain_break), cls[Np:]]).tobytes(), 9)
    W.sec(topo); info["topology"] = len(topo) + 4
    # mean structure (all atoms) and per frame rigid fit
    xf = torch.as_tensor(truth, dtype=torch.float32); Mn = RCD.iter_mean(xf).float()
    Rk, tk = E["kabsch"](Mn[None].expand(T, -1, 3), xf)
    q = _rot_to_quat(Rk.double().numpy()); t32 = tk.numpy().astype(np.float32); Rq = _quat_to_rot(q)
    Mnp = Mn.numpy().astype(np.float64)
    bbo = np.concatenate([np.where(bbsel)[0], Np + np.arange(Mx)])
    Mbbo32 = Mn.numpy()[bbo].astype(np.float32)
    pm_bbo = np.einsum("na,tba->tnb", Mbbo32.astype(np.float64), Rq) + t32.astype(np.float64)[:, None]
    # per trajectory buffers of the shared model (reference backbone, scale, static chi prior)
    E["fit_buffers"](dom, X, FT.builder)
    side0 = 0
    if backbone == "mean" or Mx:
        W.arr(Mbbo32 if backbone == "mean" else Mbbo32[n_bb:], np.float32); W.arr(q, np.float32); W.arr(t32, np.float32)
    W.arr(dom.ref.numpy(), np.float32); W.arr(np.float32([float(dom.scale)]), np.float32); W.arr(dom.chi_ref.numpy(), np.float32)
    W.b.write(np.packbits(dom.chi_mask.numpy() > 0).tobytes())
    info["mean_structure_rigid_buffers"] = len(W.b.getvalue()) - info["topology"] - 4 - struct.calcsize("<BBdIHI")
    pbbo = pm_bbo.copy()
    if backbone == "ae":
        E["bind"](m, dom); Z, QA, TA = [], [], []
        with torch.no_grad():
            for s in range(0, T, 25):
                bb_al, R_, t_ = m.codec.align(X[s:s + 25, :, :4]); Z.append(m.codec.encode(bb_al, dom.aatype).half().numpy()); QA.append(R_.double().numpy()); TA.append(t_.numpy())
        z16 = np.concatenate(Z).astype(np.float16); qa = _rot_to_quat(np.concatenate(QA)); ta = np.concatenate(TA).astype(np.float32); Ra = _quat_to_rot(qa)
        lat = lzma.compress(z16.tobytes(), preset=9); raw = z16.tobytes()
        W.b.write(b"\x01" if len(lat) < len(raw) else b"\x00"); W.sec(lat if len(lat) < len(raw) else raw); W.arr(qa, np.float32); W.arr(ta, np.float32)
        info["ae_latents"] = min(len(lat), len(raw))
        pae = _ae_backbone(E, dom, z16, Ra, ta).reshape(T, R, 4, 3)[:, mbb[:, :4]].double().numpy()
        pbbo[:, :n_bb] = pae
    # backbone stream
    if residuals:
        q1, rf1 = RCD.quantise(truth[:, bbo], pbbo, b)
        base, st = _qbase(pbbo, b, RCD)
        if coder == "ac":
            blob1, i1 = RE.encode(q1, cls[bbo], base, st, RE.make_refs(Mnp[bbo]), RE.make_refs3(Mnp[bbo], None), None)
            q1d = RE.decode(blob1, q1.shape, cls[bbo], base, st, None)
        else:
            blob1 = _lz_enc(RCD, q1); q1d = _lz_dec(RCD, blob1, q1.shape)
        if not np.array_equal(q1d, q1): raise RuntimeError("backbone stream roundtrip mismatch")
        dbb = rf1(q1d)[:, :n_bb]
    else:
        blob1 = b""; dbb = pbbo[:, :n_bb]
    # side chain predictor
    mlp = None
    if full_predictor:
        tchi, _ = FT.builder.chis_from_atom14(X, dom.aatype); aat = aatype; cmn = dom.chi_mask.numpy()
        st4, ns4 = RC.chi_states(S.ang(tchi.numpy()), aat, cmn, nk=4)
        mask = RC.build_code(st4, ns4, (0, 1, 2, 3)); coded = mask.any(1); j = RC.joint(st4, mask)
        tab, seen = RC.state_table(tchi, aat, j, coded, dom.chi_ref, 81)
        coded_r = np.where(coded)[0]; cols = [(r, k) for r in range(R) for k in range(4) if mask[r, k]]
        seen_c = seen[coded_r]; nchi = cmn.sum(1).astype(int)
        tabn = tab[torch.as_tensor(coded_r)].numpy() if len(coded_r) else np.zeros((0, 81, 4, 2), np.float32)
        ang16 = np.array([np.arctan2(tabn[i, sidx, k, 0], tabn[i, sidx, k, 1]) for i, r in enumerate(coded_r) for sidx in np.where(seen_c[i])[0] for k in range(nchi[r])], np.float16)
        tab16 = _tab_from_angles(dom, coded_r, seen_c, ang16)
        stc = np.stack([st4[:, r, k] for r, k in cols]).astype(np.uint8) if cols else np.zeros((0, T), np.uint8)
        csec = zlib.compress(stc.tobytes(), 9)
        W.b.write(np.packbits(mask).tobytes()); W.sec(csec)
        W.sec(np.packbits(seen_c).tobytes()); W.arr(ang16, np.float16)
        cref = _cref_from_codes(E, dom, T, coded_r, stc, cols, mask, tab16)
        ft = RCD.cond_from_bb(m, dom, X[:, :, :4].contiguous(), cref)
        torch.manual_seed(0)
        with torch.enable_grad():
            FT.a.batch = min(48, T); mod, _, _ = FT.fit("mlp", m, ft, X, dom)
        del ft
        w16 = np.concatenate([p.detach().numpy().ravel() for p in mod.parameters()]).astype(np.float16)
        W.arr(w16, np.float16); mlp = _mlp_from(E, w16)
        info["rotamer_codes"] = len(csec) + int(np.ceil(mask.size / 8)) + ang16.nbytes + int(np.ceil(seen_c.size / 8)) + 12
        info["mlp"] = w16.nbytes + 4
    else:
        cref = dom.chi_ref[None].repeat(T, 1, 1, 1).clone()
    ph0 = _sc_prediction(E, dom, dbb, mbb, n_bb, cref, mlp)
    sym = _sym(E, dom); sw = E["L"]._swapped(ph0, dom.aatype); amk = dom.atom_mask[None, ..., None]
    d1 = (((ph0 - X) ** 2) * amk).sum((-1, -2)); d2 = (((sw - X) ** 2) * amk).sum((-1, -2)); use = (d2 < d1) & sym[None]
    ph = torch.where(use[..., None, None], sw, ph0)
    swb = zlib.compress(np.packbits(use[:, sym].numpy().astype(np.uint8).T, axis=-1).tobytes(), 9); W.sec(swb); info["swap_bits"] = len(swb) + 4
    psc = ph[:, msc].double().numpy()
    if residuals:
        q2, rf2 = RCD.quantise(truth[:, :Np][:, scsel], psc, b)
        base2, st2 = _qbase(psc, b, RCD)
        if coder == "ac":
            blob2, i2 = RE.encode(q2, cls[:Np][scsel], base2, st2, RE.make_refs(Mnp[:Np][scsel]), RE.make_refs3(Mnp[:Np][scsel], Mnp[:Np][bbsel]), dbb)
        else:
            blob2 = _lz_enc(RCD, q2)
    else:
        blob2 = b""
    W.sec(blob1); W.sec(blob2)
    info.update(backbone_residuals=len(blob1) + 4, sidechain_residuals=len(blob2) + 4)
    if _keep_rec:  # encoder side reconstruction, for bit exact decoder tests
        rec = np.zeros_like(truth)
        if residuals: rec[:, bbo] = rf1(q1); rec[:, np.where(scsel)[0]] = rf2(q2)
        else: rec[:, bbo] = pbbo; rec[:, np.where(scsel)[0]] = psc
        info["_rec"] = rec
    out = W.b.getvalue(); info.update(total=len(out), raw_float32=12 * truth.shape[1] * T, ratio=12 * truth.shape[1] * T / len(out), flags=flags,
                                      bound=b if residuals else None)
    return (out, info) if return_info else out


def _decompress_hybrid(blob, threads=4):
    """-> dict(coords [T,R,14,3] float32 (zeros where atom_mask is 0), extra [T,M,3] or None, aatype, atom_mask,
    chain_index, chain_break, bound, backbone, full_predictor)."""
    E = _env(threads); RCD, RE, FT, RC, m = E["RCD"], E["RE"], E["FT"], E["RC"], E["m"]
    if blob[:4] != MAGIC: raise ValueError("not an md_compress file")
    ver, flags, b, T, R, Mx = struct.unpack_from("<BBdIHI", blob, 4)
    rd = _R(blob, 4 + struct.calcsize("<BBdIHI"))
    t = np.frombuffer(zlib.decompress(rd.sec()), np.uint8)
    o = 0; aatype = t[o:o + R].astype(np.int64); o += R
    nm = int(np.ceil(R * 14 / 8)); atom_mask = np.unpackbits(t[o:o + nm])[:R * 14].reshape(R, 14).astype(np.float32); o += nm
    chain_index = t[o:o + 2 * R].copy().view(np.uint16).astype(np.int64); o += 2 * R
    nb = int(np.ceil(R / 8)); chain_break = np.unpackbits(t[o:o + nb])[:R].astype(bool); o += nb
    ecls = t[o:o + Mx]
    dom, mk, mbb, msc, bbsel, scsel, n_bb = _setup(E, aatype, atom_mask, chain_index, chain_break)
    Np = int(mk.sum()); cls = _classes(atom_mask, Mx, ecls); ae = bool(flags & F_AE); full = bool(flags & F_FULL)
    lz = bool(flags & F_LZMA); nores = bool(flags & F_NORES)
    nbbo = n_bb + Mx
    if not ae or Mx:
        Mb = rd.arr(np.float32, (nbbo if not ae else Mx, 3)); q = rd.arr(np.float32, (T, 4)); t32 = rd.arr(np.float32, (T, 3)); Rq = _quat_to_rot(q)
        pm = np.einsum("na,tba->tnb", Mb.astype(np.float64), Rq) + t32.astype(np.float64)[:, None]
    dom.ref = torch.as_tensor(rd.arr(np.float32, (R, 4, 3))); dom.scale = torch.as_tensor(rd.arr(np.float32, (1,))[0])
    dom.chi_ref = torch.as_tensor(rd.arr(np.float32, (R, 4, 2)))
    nc = int(np.ceil(R * 4 / 8)); dom.chi_mask = torch.as_tensor(np.unpackbits(np.frombuffer(blob, np.uint8, nc, rd.o))[:R * 4].reshape(R, 4).astype(np.float32)); rd.o += nc
    pbbo = np.zeros((T, nbbo, 3))
    if ae:
        kind = blob[rd.o]; rd.o += 1; lat = rd.sec(); lat = lzma.decompress(lat) if kind == 1 else lat
        z16 = np.frombuffer(lat, np.float16).reshape(T, R, -1); qa = rd.arr(np.float32, (T, 4)); ta = rd.arr(np.float32, (T, 3))
        pbbo[:, :n_bb] = _ae_backbone(E, dom, z16, _quat_to_rot(qa), ta).reshape(T, R, 4, 3)[:, mbb[:, :4]].double().numpy()
        if Mx: pbbo[:, n_bb:] = pm
    else:
        pbbo = pm
    mlp = None
    if full:
        nmk = int(np.ceil(R * 4 / 8)); mask = np.unpackbits(np.frombuffer(blob, np.uint8, nmk, rd.o))[:R * 4].reshape(R, 4).astype(bool); rd.o += nmk
        cols = [(r, k) for r in range(R) for k in range(4) if mask[r, k]]; coded_r = np.where(mask.any(1))[0]
        stc = np.frombuffer(zlib.decompress(rd.sec()), np.uint8).reshape(len(cols), T)
        seen_c = np.unpackbits(np.frombuffer(rd.sec(), np.uint8))[:len(coded_r) * 81].reshape(len(coded_r), 81).astype(bool)
        tab16 = _tab_from_angles(dom, coded_r, seen_c, rd.arr(np.float16, (-1,)))
        cref = _cref_from_codes(E, dom, T, coded_r, stc, cols, mask, tab16)
        w16 = rd.arr(np.float16, (-1,)); mlp = _mlp_from(E, w16)
    else:
        cref = dom.chi_ref[None].repeat(T, 1, 1, 1).clone()
    swb = rd.sec(); blob1 = rd.sec(); blob2 = rd.sec()
    if nores:
        dall = pbbo
    else:
        base, st = _qbase(pbbo, b, RCD)
        q1 = _lz_dec(RCD, blob1, (T, nbbo, 3)) if lz else RE.decode(blob1, (T, nbbo, 3), cls[np.concatenate([np.where(bbsel)[0], Np + np.arange(Mx)])], base, st, None)
        _, rf1 = RCD.quantise(np.zeros_like(pbbo), pbbo, b); dall = rf1(q1)
    dbb = dall[:, :n_bb]
    ph0 = _sc_prediction(E, dom, dbb, mbb, n_bb, cref, mlp)
    sym = _sym(E, dom); ns = int(sym.sum())
    use = torch.zeros(T, R, dtype=torch.bool)
    if ns:
        bits = np.unpackbits(np.frombuffer(zlib.decompress(swb), np.uint8).reshape(ns, -1), axis=-1)[:, :T].T.astype(bool)
        use[:, sym] = torch.as_tensor(bits)
    ph = torch.where(use[..., None, None], E["L"]._swapped(ph0, dom.aatype), ph0)
    psc = ph[:, msc].double().numpy()
    if nores:
        sc = psc
    else:
        base2, st2 = _qbase(psc, b, RCD)
        q2 = _lz_dec(RCD, blob2, psc.shape) if lz else RE.decode(blob2, psc.shape, cls[:Np][scsel], base2, st2, dbb)
        _, rf2 = RCD.quantise(np.zeros_like(psc), psc, b); sc = rf2(q2)
    out = np.zeros((T, R, 14, 3), np.float32); prot = np.zeros((T, Np, 3))
    prot[:, bbsel] = dbb; prot[:, scsel] = sc; out[:, mk.numpy()] = prot
    return dict(coords=out, extra=dall[:, n_bb:].astype(np.float32) if Mx else None, aatype=aatype, atom_mask=atom_mask, chain_index=chain_index,
                chain_break=chain_break, bound=None if nores else b, backbone="ae" if ae else "mean", full_predictor=full,
                coder="lzma" if lz else "ac", residuals=not nores)


# ---------------------------------------------------------------- method dispatch (hybrid = version 1 files, others = version 2)
def _topo_bytes(aatype, atom_mask, chain_index, chain_break, ecls):
    return zlib.compress(np.concatenate([np.asarray(aatype).astype(np.uint8), np.packbits(np.asarray(atom_mask) > 0),
                                         np.asarray(chain_index).astype(np.uint16).view(np.uint8), np.packbits(np.asarray(chain_break, bool)),
                                         np.asarray(ecls, np.uint8)]).tobytes(), 9)


def _topo_parse(data, R, Mx):
    t = np.frombuffer(zlib.decompress(data), np.uint8); o = 0
    aatype = t[o:o + R].astype(np.int64); o += R
    nm = int(np.ceil(R * 14 / 8)); atom_mask = np.unpackbits(t[o:o + nm])[:R * 14].reshape(R, 14).astype(np.float32); o += nm
    chain_index = t[o:o + 2 * R].copy().view(np.uint16).astype(np.int64); o += 2 * R
    nb = int(np.ceil(R / 8)); chain_break = np.unpackbits(t[o:o + nb])[:R].astype(bool); o += nb
    return aatype, atom_mask, chain_index, chain_break, t[o:o + Mx].copy()


_HDR2 = "<BBBIId"


GENERIC = ("lzma", "zstd", "bz2", "gzip")
HYBRID = ("hybrid", "hybrid-ae", "model-only")


def _fns(MT):
    return {k: (getattr(MT, f"enc_{k.replace('-', '')}"), getattr(MT, f"dec_{k.replace('-', '')}"))
            for k in ("xtc", "tng", "mdcompress", "trr", "dcd", "netcdf", "pdb", "h5md", "pcazip", "compresstraj", "cg2all") if hasattr(MT, f"enc_{k}")}


def _encode_method(method, x, cls, bound, threads, opts, topo=None):
    from md_compress import methods as MT
    if method in GENERIC: return MT.enc_generic(method, x, bound=bound, threads=threads, **opts)
    if method == "new-coder": return MT.enc_newcoder(x, cls, _env(threads), bound=bound, **opts)
    if method == "cg2all": return MT.enc_cg2all(x, topo, bound=bound, threads=threads, **opts)
    return _fns(MT)[method][0](x, bound=bound, threads=threads, **opts)


def _decode_method(method, data, T, N, opts, cls, threads, topo=None):
    from md_compress import methods as MT
    if method in GENERIC: return MT.dec_generic(method, data, T, N, opts, threads)
    if method == "new-coder": return MT.dec_newcoder(data, T, N, opts, cls, _env(threads))
    if method == "cg2all": return MT.dec_cg2all(data, T, N, opts, topo, threads)
    return _fns(MT)[method][1](data, T, N, opts, threads)


def _container(method, kind, x, cls, bound, threads, opts, topo=None, R=0, Mx=0, meta=None, top_arrays=None):
    from md_compress import methods as MT
    if method not in MT.METHODS: raise ValueError(f"unknown method {method!r}; choose from hybrid, " + ", ".join(MT.METHODS))
    opts = {k: v for k, v in opts.items() if v is not None}
    x = np.ascontiguousarray(x, np.float32); T, N = x.shape[:2]
    if method == "cg2all" and top_arrays is None: raise ValueError("cg2all needs the protein topology; use compress(...) on atom14 coordinates")
    data, used, maxerr = _encode_method(method, x, cls, bound, threads, opts, top_arrays)
    used["bound"] = maxerr; used.update(meta or {})
    W = _W(); W.b.write(MAGIC + struct.pack(_HDR2, 2, MT.METHODS[method], kind, T, N, np.nan if maxerr is None else maxerr))
    W.sec(json.dumps(used).encode())
    if kind == 0: W.b.write(struct.pack("<HI", R, Mx)); W.sec(topo)
    W.sec(data)
    out = W.b.getvalue()
    info = dict(method=method, options=used, bound=maxerr, payload=len(data), total=len(out), raw_float32=12 * N * T, ratio=12 * N * T / len(out))
    info["breakdown"] = dict(header_options_topology=len(out) - len(data) - 4, payload=len(data) + 4,
                             **{k: used[k] for k in ("decoder_bytes", "latent_bytes") if k in used})
    return out, info


def _metadata(core, info, method, x_truth, groups, R, t_enc, threads, measure, inp, residuals=True):
    """decode the core blob once, measure the errors against the input and append the metadata trailer."""
    from md_compress import report as RP
    T, N = x_truth.shape[:2]; used = info.get("options", {}); bound = info.get("bound")
    res = RP.resolution(method, used, bound, residuals)
    skip = ("total", "raw_float32", "ratio", "flags", "bound", "options", "method", "breakdown", "_rec", "payload")
    bd = info.get("breakdown") or {k: v for k, v in info.items() if k not in skip and isinstance(v, (int, np.integer))}
    meta = dict(schema=RP.SCHEMA, writer="md_compress.codec", codec_version=VERSION, container_version=core[4],
                method=method, options=used, input=inp or {}, system=dict(atoms=int(N), residues=None if R is None else int(R), frames=int(T)),
                size=dict(raw_float32_bytes=int(12 * N * T), breakdown=bd), compress_seconds=round(t_enc, 3), resolution=res)
    if measure:
        t0 = time.time(); r = decompress(core, threads, report=False)
        rec = r["xyz"] if "xyz" in r else np.concatenate([r["coords"][:, np.asarray(r["atom_mask"]) > 0]] + ([r["extra"]] if r.get("extra") is not None else []), 1)
        meta["errors"] = dict(measured_on="the decoded file against the input, at compress time", **RP.error_stats(rec, x_truth, res["error_limit_angstrom"], groups))
        meta["verify_decode_seconds"] = round(time.time() - t0, 3)
        a = meta["errors"]["groups"]["all"]
        info["measured"] = dict(rmsd_lab=a["rmsd_lab"], max_abs_err=a["abs_error"]["max"], atoms_missing=a["atoms_missing"])
    else:
        meta["errors"] = None; meta["errors_note"] = "not measured (measure=False)"
    meta = RP.clean(meta); out = RP.attach(core, meta); meta["size"] = RP.sizes(meta["size"], len(out), len(core))
    info.update(total=len(out), ratio=12 * N * T / len(out), core_bytes=len(core), metadata_bytes=len(out) - len(core),
                ratio_without_metadata=12 * N * T / len(core), metadata=meta)
    return out, info


def compress(coords, aatype, atom_mask, chain_index, chain_break, bound=None, backbone="mean", full_predictor=False,
             extra=None, extra_cls=None, threads=4, return_info=False, method="hybrid", _keep_rec=False, measure=True, metadata=True,
             _input=None, **opts):
    """atom14 protein trajectory -> .mdz bytes.
    method 'hybrid' (default; bound default 0.1, backbone / full_predictor flags) or one of md_compress.methods.METHODS
    ('xtc', 'tng', 'mdcompress', 'new-coder', 'lzma', 'zstd', 'bz2') with that method's options as keyword arguments
    (xtc_precision, tng_precision, mdc_res, mdc_level, mdc_preset, mdc_batch, mdc_history, nc_predictor, level, delta).
    bound: Angstrom, 0 / 'lossless' (0.01 A grid), 'exact' (float32 bits; lzma / zstd / bz2 only) or None (method default)."""
    from md_compress import report as RP
    t0 = time.time(); mk0 = np.asarray(atom_mask) > 0; Mx0 = 0 if extra is None else int(np.asarray(extra).shape[1])
    x_truth = np.concatenate([np.asarray(coords, np.float32)[:, mk0]] + ([np.asarray(extra, np.float32)] if Mx0 else []), 1)
    groups = RP.groups_atom14(atom_mask, Mx0)
    if method in HYBRID:
        coder = opts.pop("coder", None) or "ac"; residuals = opts.pop("residuals", True)
        if opts: raise TypeError(f"{method} takes no options {sorted(opts)}")
        if method == "hybrid-ae": backbone = "ae"
        if method == "model-only": backbone, full_predictor, residuals = "ae", True, False
        r = _compress_hybrid(coords, aatype, atom_mask, chain_index, chain_break, 0.1 if bound is None else bound, backbone, full_predictor,
                             extra, extra_cls, threads, True, _keep_rec, coder, residuals)
        out, info = r
        info["method"] = method
        info["options"] = dict(backbone=backbone, full_predictor=bool(full_predictor), coder=coder, residuals=bool(residuals))
        if metadata:
            out, info = _metadata(out, info, method, x_truth, groups, len(aatype), time.time() - t0, threads, measure, _input, residuals)
        return (out, info) if return_info else out
    if backbone != "mean" or full_predictor: raise ValueError("backbone / full_predictor apply to method 'hybrid' only")
    atom_mask = np.asarray(atom_mask, np.float32); mk = atom_mask > 0; Mx = 0 if extra is None else int(np.asarray(extra).shape[1])
    x = np.concatenate([np.asarray(coords, np.float32)[:, mk]] + ([np.asarray(extra, np.float32)] if Mx else []), 1)
    cls = _classes(atom_mask, Mx, extra_cls)
    topo = _topo_bytes(aatype, atom_mask, chain_index, chain_break, cls[int(mk.sum()):])
    out, info = _container(method, 0, x, cls, bound, threads, opts, topo, len(aatype), Mx,
                           top_arrays=dict(aatype=np.asarray(aatype), atom_mask=atom_mask, chain_index=np.asarray(chain_index), Mx=Mx))
    if metadata: out, info = _metadata(out, info, method, x, groups, len(aatype), time.time() - t0, threads, measure, _input)
    if _keep_rec: info["_rec"] = None
    return (out, info) if return_info else out


def compress_xyz(xyz, method="zstd", bound=None, threads=4, return_info=False, atom_classes=None, meta=None, measure=True, metadata=True,
                 groups=None, n_residues=None, _input=None, **opts):
    """any atoms: xyz [T, N, 3] Angstrom -> .mdz bytes with a non hybrid method (the hybrid needs the protein topology).
    groups: optional {name: atom indices} for the error statistics (default: all atoms)."""
    if method in HYBRID or method == "cg2all": raise ValueError("this method needs the protein topology; use compress(...) on atom14 coordinates")
    t0 = time.time()
    xyz = np.asarray(xyz, np.float32); cls = np.full(xyz.shape[1], 11, np.uint8) if atom_classes is None else np.asarray(atom_classes, np.uint8)
    out, info = _container(method, 1, xyz, cls, bound, threads, opts, meta=meta)
    if metadata: out, info = _metadata(out, info, method, xyz, groups, n_residues, time.time() - t0, threads, measure, _input)
    return (out, info) if return_info else out


def decompress(blob, threads=4, return_info=False, report=True):
    """.mdz bytes -> dict. Hybrid and atom14 files: coords [T,R,14,3], extra, aatype, atom_mask, chain_index, chain_break,
    bound, method (+ backbone, full_predictor); compress_xyz files: xyz [T,N,3], bound, method, options.
    report=True (default): the dict also holds 'report', the stored metadata plus the decompression time (the dict that
    decompress_file writes as JSON); report=False skips it. return_info=True: -> (dict, report)."""
    from md_compress import report as RP
    core, meta = RP.split(bytes(blob)); t0 = time.time()
    r = _decompress_core(core, threads)
    if not (report or return_info): return r
    rep = _report(r, meta, len(blob), len(core), time.time() - t0)
    if report: r["report"] = rep
    return (r, rep) if return_info else r


def _report(r, meta, n_total, n_core, t_dec):
    from md_compress import report as RP
    if meta is None:  # file written before the metadata trailer: what the header tells
        x = r["xyz"] if "xyz" in r else None
        T = (x if x is not None else r["coords"]).shape[0]
        N = x.shape[1] if x is not None else int((np.asarray(r["atom_mask"]) > 0).sum()) + (0 if r.get("extra") is None else r["extra"].shape[1])
        opts = r.get("options") or {k: r.get(k) for k in ("backbone", "full_predictor", "coder", "residuals") if k in r}
        meta = dict(schema=RP.SCHEMA, metadata_present=False, note="file written before metadata was stored: no input names, timings or error statistics",
                    method=r["method"], options=opts, system=dict(atoms=N, residues=None if x is not None else int(len(r["aatype"])), frames=T),
                    size=dict(total_bytes=n_total, metadata_bytes=0, raw_float32_bytes=12 * N * T, ratio=12 * N * T / n_total),
                    resolution=RP.resolution(r["method"], opts, r.get("bound"), r.get("residuals", True)), errors=None)
    else:
        meta = dict(meta, metadata_present=True); meta["size"] = RP.sizes(meta["size"], n_total, n_core)
    meta["decompress_seconds"] = round(t_dec, 3)
    return RP.clean(meta)


def _decompress_core(blob, threads=4):
    if blob[:4] != MAGIC: raise ValueError("not an md_compress file")
    if blob[4] == 1:
        r = _decompress_hybrid(blob, threads); r["method"] = "hybrid"; return r  # hybrid-ae / model-only are hybrid files with flags
    from md_compress import methods as MT
    ver, mid, kind, T, N, bnd = struct.unpack_from(_HDR2, blob, 4)
    rd = _R(blob, 4 + struct.calcsize(_HDR2)); opts = json.loads(rd.sec()); method = MT.NAMES[mid]
    if kind == 0:
        R, Mx = struct.unpack_from("<HI", blob, rd.o); rd.o += 6
        aatype, atom_mask, chain_index, chain_break, ecls = _topo_parse(rd.sec(), R, Mx); cls = _classes(atom_mask, Mx, ecls)
    else:
        cls = np.full(N, 11, np.uint8)
    x = _decode_method(method, rd.sec(), T, N, opts, cls, threads,
                       dict(aatype=aatype, atom_mask=atom_mask, chain_index=chain_index, Mx=Mx) if kind == 0 else None)
    res = dict(method=method, options=opts, bound=None if np.isnan(bnd) else bnd)
    if kind == 1: res["xyz"] = x; return res
    mk = atom_mask > 0; Np = int(mk.sum()); out = np.zeros((T, R, 14, 3), np.float32); out[:, mk] = x[:, :Np]
    res.update(coords=out, extra=x[:, Np:] if Mx else None, aatype=aatype, atom_mask=atom_mask, chain_index=chain_index, chain_break=chain_break,
               backbone=None, full_predictor=None)
    return res


# ---------------------------------------------------------------- files and CLI
OPTS = {"hybrid": ("coder", "residuals"), "hybrid-ae": ("coder",), "model-only": (), "xtc": ("xtc_precision",), "tng": ("tng_precision",),
        "mdcompress": ("mdc_res", "mdc_level", "mdc_preset", "mdc_batch", "mdc_history"), "new-coder": ("nc_predictor", "nc_coder"),
        "lzma": ("level", "delta"), "zstd": ("level", "delta"), "bz2": ("level", "delta"), "gzip": ("level", "delta"),
        "trr": ("trr_double",), "dcd": (), "netcdf": (), "pdb": ("pdb_precision", "pdb_gzip"),
        "h5md": ("h5md_compression", "h5md_level", "h5md_chunk", "h5md_sz3_abs"),
        "pcazip": ("pca_variance", "pca_components", "pca_residual_scale", "pca_eigenvector_scale", "pca_entropy", "pca_frame"),
        "compresstraj": ("ct_compression", "ct_latent", "ct_epochs", "ct_layers", "ct_batch"), "cg2all": ("cg_model", "cg_precision", "cg_device", "cg_store")}


def _load_atoms(top, traj, atoms="protein", **kw):
    """the codec's atom set from files -> dict(x [T,N,3] in codec order, groups, residues, and for protein the atom14 arrays)."""
    from md_compress import report as RP
    if atoms != "protein":
        import MDAnalysis as mda
        u = mda.Universe(top, traj); ag = u.atoms if atoms == "all" else u.select_atoms(atoms)
        sl = slice(kw.get("start", 0), kw.get("stop"), kw.get("stride", 1))
        x = np.stack([ag.positions.copy() for _ in u.trajectory[sl]]).astype(np.float32)
        prot = np.isin(np.arange(u.atoms.n_atoms), u.select_atoms("protein").indices)[ag.indices]
        return dict(x=x, groups=RP.groups_names(ag.names, prot), residues=int(ag.n_residues))
    from md_compress.data import load_trajectory
    d = load_trajectory(top, traj, **kw)
    oxt = np.where(d.oxt_atoms >= 0)[0]; extra = None
    if len(oxt):
        u = d.universe; sl = slice(kw.get("start", 0), kw.get("stop"), kw.get("stride", 1))
        extra = np.stack([u.atoms.positions[d.oxt_atoms[oxt]].copy() for _ in u.trajectory[sl]][:d.n_frames]).astype(np.float32)
    mk = d.atom_mask > 0
    x = np.concatenate([d.coords[:, mk].astype(np.float32)] + ([extra] if extra is not None else []), 1)
    return dict(x=x, groups=RP.groups_atom14(d.atom_mask, 0 if extra is None else extra.shape[1]), residues=int(len(d.aatype)), d=d, extra=extra)


def compress_file(top, traj, out, bound=None, backbone="mean", full_predictor=False, threads=4, method="hybrid", atoms="protein",
                  method_opts=None, **kw):
    """topology + trajectory (any MDAnalysis format) -> .mdz file (with the metadata trailer; decompress_file writes it as JSON).
    atoms='protein' (default): protein heavy atoms in atom14 layout, OXT kept as extra atoms (required for 'hybrid').
    atoms='all' or an MDAnalysis selection string: those atoms as plain coordinates (non hybrid methods only).
    kw: start, stride (frame slicing)."""
    method_opts = method_opts or {}
    if atoms != "protein" and (method in HYBRID or method == "cg2all"): raise ValueError(f"{method} codes protein heavy atoms only; use atoms='protein'")
    t0 = time.time(); L = _load_atoms(top, traj, atoms, **kw)
    inp = dict(topology=top, trajectory=traj, atoms=atoms, stride=kw.get("stride", 1), start=kw.get("start", 0), load_seconds=round(time.time() - t0, 3))
    if atoms != "protein":
        blob, info = compress_xyz(L["x"], method, bound, threads, True, meta=dict(atoms=atoms), groups=L["groups"], n_residues=L["residues"],
                                  _input=inp, **method_opts)
    else:
        d = L["d"]
        blob, info = compress(d.coords, d.aatype, d.atom_mask, d.chain_index, d.chain_break, bound, backbone, full_predictor, extra=L["extra"],
                              threads=threads, return_info=True, method=method, _input=inp, **method_opts)
    with open(out, "wb") as f: f.write(blob)
    return info


def decompress_file(inp, top, out, threads=4, ref=None, report=True):
    """.mdz + the same topology -> trajectory file (format from the extension).
    report=True (default): also write <out stem>.json next to it (stored metadata, decompression time, output format note;
    with ref = the original trajectory, errors recomputed against it); report='path.json': write it there; report=False: no JSON."""
    from md_compress import report as RP
    blob = open(inp, "rb").read()
    r, rep = decompress(blob, threads, return_info=True, report=False)
    t0 = time.time()
    if "xyz" in r:
        import MDAnalysis as mda
        u = mda.Universe(top); atoms = r["options"].get("atoms", "all"); ag = u.atoms if atoms == "all" else u.select_atoms(atoms)
        if ag.n_atoms != r["xyz"].shape[1]: raise ValueError(f"topology selection has {ag.n_atoms} atoms, file has {r['xyz'].shape[1]}")
        with mda.Writer(out, ag.n_atoms) as w:
            for xi in r["xyz"]: ag.positions = xi; w.write(ag)
        rec = r["xyz"]
    else:
        from md_compress.data import load_trajectory
        atoms = "protein"; d = load_trajectory(top)
        oxt = None
        if r["extra"] is not None:
            oxt = np.zeros((r["coords"].shape[0], d.n_res, 3), np.float32); oxt[:, np.where(d.oxt_atoms >= 0)[0]] = r["extra"]
        d.write(r["coords"], out, oxt=oxt)
        rec = np.concatenate([r["coords"][:, np.asarray(r["atom_mask"]) > 0]] + ([r["extra"]] if r["extra"] is not None else []), 1)
    rep["write_seconds"] = round(time.time() - t0, 3)
    rep["output"] = dict(path=out, **RP.output_note(out))
    rep["decompressed_from"] = inp
    if ref is not None:
        st = (rep.get("input") or {}); kw = dict(stride=st.get("stride", 1), start=st.get("start", 0))
        R0 = _load_atoms(top, ref, atoms, **kw); x = R0["x"]
        lim = (rep.get("resolution") or {}).get("error_limit_angstrom")
        if x.shape != rec.shape:
            rep["errors_vs_ref"] = dict(error=f"reference has shape {list(x.shape)}, decoded data {list(rec.shape)} (stride {kw['stride']})")
        else:
            if atoms in ("protein", "all"): back = _load_atoms(top, out, atoms)["x"]
            else:  # the written file holds only the selected atoms
                import MDAnalysis as mda
                ub = mda.Merge(mda.Universe(top).select_atoms(atoms)); ub.load_new(out)
                back = np.stack([ub.atoms.positions.copy() for _ in ub.trajectory]).astype(np.float32)
            rnd = rep["output"]["extra_rounding_max_angstrom"]
            lim_w = None if lim is None or rnd is None else lim + rnd
            rep["errors_vs_ref"] = dict(reference=ref, stride=kw["stride"], start=kw["start"],
                                        decoded=RP.error_stats(rec, x, lim, R0["groups"]),
                                        written_file=dict(note="errors of the trajectory file as written (includes the output format rounding)",
                                                          **RP.error_stats(back, x, lim_w, R0["groups"])))
    r["report"] = rep; r["report_path"] = None
    if report:
        jp = RP.json_path(out) if report is True else str(report)
        with open(jp, "w") as f: json.dump(RP.clean(rep), f, indent=1)
        r["report_path"] = jp
    return r


def methods_help():
    from md_compress import methods as MT
    L = ["methods (python -m md_compress.codec compress --method NAME ...):"]
    for k, (d, f, sec) in MT.HELP.items(): L.append(f"  {k:<12} {d}\n  {'':<12} flags: {f}\n  {'':<12} report: {sec}")
    L.append("common flags: --stride N, --threads N, --atoms protein|all|SELECTION (non hybrid methods may code any atoms)")
    L.append("bound: Angstrom, 'lossless' (0.01 A grid, max error 0.005 A) or 'exact' (float32 bits, lzma / zstd / bz2 only)")
    return "\n".join(L)


def main(argv=None):
    import argparse, time
    ap = argparse.ArgumentParser(prog="python -m md_compress.codec", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=methods_help())
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compress", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=methods_help())
    c.add_argument("--top", required=True); c.add_argument("--traj", required=True); c.add_argument("--out", required=True)
    c.add_argument("--method", default="hybrid", choices=list(OPTS), help="compression method (default hybrid)")
    c.add_argument("--bound", default=None, help="max abs error in Angstrom, 'lossless' (0.01 A grid) or 'exact' (lzma/zstd/bz2); default: method default")
    c.add_argument("--backbone", choices=["mean", "ae"], default="mean", help="hybrid: 'ae' = experimental backbone autoencoder")
    c.add_argument("--full-predictor", action="store_true", help="hybrid: rotamer codes + per trajectory MLP (report mode)")
    g = c.add_argument_group("hybrid"); g.add_argument("--coder", choices=["ac", "lzma"], help="residual coder (default ac; lzma = the report's 'ours: hybrid')")
    g.add_argument("--no-residuals", dest="residuals", action="store_const", const=False, help="prediction only, lossy (as model-only)")
    g = c.add_argument_group("xtc / tng"); g.add_argument("--xtc-precision", type=float, help="xtc: 1/nm grid (default 1000)")
    g.add_argument("--tng-precision", type=int, help="tng: power of ten 1/nm (default 1000)")
    g = c.add_argument_group("mdcompress"); g.add_argument("--mdc-res", type=int, help="resolution in fm (default 1000 = 0.01 A)")
    g.add_argument("--mdc-level", type=int, help="-l level (default 2)"); g.add_argument("--mdc-preset", choices=["default", "archive", "trajectory", "frames"])
    g.add_argument("--mdc-batch", type=int, help="-b frames per batch"); g.add_argument("--mdc-history", type=int, help="-h 1..3")
    g = c.add_argument_group("new-coder"); g.add_argument("--nc-predictor", choices=["mean", "raw"]); g.add_argument("--nc-coder", choices=["ac", "lzma"])
    g = c.add_argument_group("lzma / zstd / bz2 / gzip"); g.add_argument("--level", type=int); g.add_argument("--delta", choices=["time", "none"])
    g = c.add_argument_group("trr / pdb"); g.add_argument("--trr-double", action="store_const", const=True, help="trr in float64")
    g.add_argument("--pdb-precision", type=int, help="decimals 0..3 (default 3)"); g.add_argument("--pdb-gzip", type=int, help="gzip level for the PDB text")
    g = c.add_argument_group("h5md"); g.add_argument("--h5md-compression", choices=["none", "gzip", "sz3"]); g.add_argument("--h5md-level", type=int)
    g.add_argument("--h5md-chunk", type=int, help="frames per chunk"); g.add_argument("--h5md-sz3-abs", type=float, help="SZ3 absolute error bound in nm")
    g = c.add_argument_group("pcazip"); g.add_argument("--pca-variance", type=float); g.add_argument("--pca-components", type=int)
    g.add_argument("--pca-residual-scale", type=float); g.add_argument("--pca-eigenvector-scale", type=float); g.add_argument("--pca-entropy", choices=["lzma"])
    g.add_argument("--pca-frame", choices=["lab", "fitted"], help="lab (default): also store the per frame fit; fitted: pcazip's own output frame (as the report)")
    g = c.add_argument_group("compresstraj"); g.add_argument("--ct-compression", type=int); g.add_argument("--ct-latent", type=int)
    g.add_argument("--ct-epochs", type=int); g.add_argument("--ct-layers"); g.add_argument("--ct-batch", type=int)
    g = c.add_argument_group("cg2all"); g.add_argument("--cg-model"); g.add_argument("--cg-precision", type=float); g.add_argument("--cg-device", choices=["cpu", "cuda"])
    g.add_argument("--cg-store", choices=["xtc", "lzma"], help="bead storage: xtc (default, as the report) or quantise + delta + lzma")
    c.add_argument("--atoms", default="protein", help="protein (atom14 heavy atoms, default), all, or an MDAnalysis selection (non hybrid methods)")
    c.add_argument("--stride", type=int, default=1); c.add_argument("--threads", type=int, default=4)
    d = sp.add_parser("decompress"); d.add_argument("--in", dest="inp", required=True); d.add_argument("--top", required=True); d.add_argument("--out", required=True)
    d.add_argument("--ref", help="optional original trajectory: errors recomputed against it in the JSON report")
    d.add_argument("--report", nargs="?", const=True, default=True, metavar="PATH",
                   help="write the JSON report (on by default, to <out stem>.json; give PATH to write it elsewhere)")
    d.add_argument("--no-report", dest="report", action="store_false", help="do not write the JSON report")
    d.add_argument("--threads", type=int, default=4)
    sp.add_parser("list-methods")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "list-methods": print(methods_help()); return
    if a.cmd == "compress":
        all_opts = sorted({o for v in OPTS.values() for o in v})
        given = {k: getattr(a, k) for k in all_opts if getattr(a, k) is not None}
        bad = [k for k in given if k not in OPTS[a.method]]
        if bad: ap.error(f"--{bad[0].replace('_', '-')} does not apply to method {a.method}")
        if a.method != "hybrid" and (a.backbone != "mean" or a.full_predictor): ap.error("--backbone / --full-predictor apply to method hybrid only")
        if a.atoms != "protein" and (a.method in HYBRID or a.method == "cg2all"): ap.error(f"{a.method} needs --atoms protein")
        bound = None if a.bound is None else (a.bound if a.bound in ("lossless", "exact") else float(a.bound))
        info = compress_file(a.top, a.traj, a.out, bound, a.backbone, a.full_predictor, a.threads, method=a.method, atoms=a.atoms,
                             method_opts=given, stride=a.stride)
        rs = info["metadata"]["resolution"]; ms = info.get("measured")
        ms = (f", NO error limit (--bound ignored): measured RMSD {ms['rmsd_lab']:.3f} A, max error {ms['max_abs_err']:.3f} A" if rs["kind"] == "no_limit"
              else f", {rs['kind']}, error limit {rs['error_limit_angstrom']} A" + (f", measured max error {ms['max_abs_err']:.4f} A" if ms else ""))
        print(f"{a.out}: method {a.method}, {info['total']:,} bytes ({info['metadata_bytes']:,} of them metadata), ratio {info['ratio']:.2f}x vs float32 "
              f"({info['ratio_without_metadata']:.2f}x without metadata){ms}, {time.time() - t0:.1f}s")
        print("metadata stored in the file; decompress writes it as JSON next to the output trajectory (<out stem>.json; --report PATH, --no-report)")
    else:
        if a.ref and not a.report: ap.error("--ref needs the report (drop --no-report)")
        r = decompress_file(a.inp, a.top, a.out, a.threads, ref=a.ref, report=a.report)
        n = (r["xyz"] if "xyz" in r else r["coords"]).shape[0]
        print(f"{a.out}: {n} frames, method {r['report'].get('method', r['method'])}, bound {r['bound']} A, {time.time() - t0:.1f}s" + (f"; report {r['report_path']}" if r["report_path"] else "; no report"))


if __name__ == "__main__":
    main()
