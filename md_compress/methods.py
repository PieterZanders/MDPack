"""Alternative (non hybrid) compression methods for md_compress.codec.

Every method codes a float32 coordinate array x [T, N, 3] in Angstrom into one payload; md_compress.codec wraps the
payload in the .mdz container, whose header records the method id and its options, so decompression needs no flags.

  xtc         GROMACS XTC (libmdaxdr from MDAnalysis). xtc_precision = 1/nm grid (default 1000, i.e. 0.001 nm);
              max error 5 / precision Angstrom; from a bound b the precision is 5 / b.
  tng         GROMACS TNG written by `gmx trjconv` (needs a GROMACS install, for encode and decode). tng_precision is a
              power of ten (default 1000, i.e. 0.001 nm); max error 5 / precision Angstrom; from a bound b the smallest
              power of ten with 5 / precision <= b. TNG stores the precision as decimal digits, so other values are rounded.
  mdcompress  MDCompress v2 binary (Kokot and Deorowicz), as benchmarked. mdc_res in femtometres (default 1000 =
              0.01 Angstrom grid), max error res / 2; from a bound b, res = 2 b 1e5. mdc_level (-l, default 2),
              mdc_preset (default, archive, trajectory, frames), mdc_batch (-b), mdc_history (-h, 1 to 3; 2 and 3
              crashed the v2.0.5 decoder in the benchmark).
  new-coder   model free quantisation + the context modelled arithmetic coder (gpu/residual_entropy.c), no network.
              nc_predictor = 'mean' (default: mean structure after a per frame rigid fit) or 'raw' (coordinates
              themselves; the coder's temporal context does the prediction). bound in Angstrom or lossless (0.01 A grid).
  lzma, zstd, bz2
              generic: quantise to a 2 b grid (bound b) or the 0.01 A grid (lossless) or keep float32 bits (exact),
              time delta (delta='time', default) or none, zigzag, byte planes, then the general purpose compressor
              at `level` (lzma 0 to 9, default 9; zstd 1 to 22, default 19; bz2 1 to 9, default 9).
"""
import bz2, io, json, lzma, os, shutil, struct, subprocess, tempfile
import numpy as np

METHODS = {"xtc": 1, "tng": 2, "mdcompress": 3, "new-coder": 4, "lzma": 5, "zstd": 6, "bz2": 7, "gzip": 8, "trr": 9, "dcd": 10,
           "netcdf": 11, "pdb": 12, "h5md": 13, "pcazip": 14, "compresstraj": 15, "cg2all": 16}
NAMES = {v: k for k, v in METHODS.items()}
GRID = 0.01

HELP = {  # name: (description, flags, report section)
    "hybrid": ("ours, default: shared packer side chains + mean structure backbone + context modelled coder (error bounded)",
               "--bound B|lossless (default 0.1), --backbone mean|ae, --full-predictor, --coder ac|lzma, --no-residuals", "4.6, 4.7.6, 4.7.7"),
    "hybrid-ae": ("ours: hybrid with the backbone autoencoder (= hybrid --backbone ae)", "as hybrid", "4.6.6"),
    "model-only": ("ours: model only (backbone AE latents + rotamer codes + per trajectory MLP, no residuals; lossy, no bound)",
                   "--coder n/a; lossy, ignores --bound", "4.1, 4.2"),
    "new-coder": ("model free: quantised raw or mean structure residuals + our context modelled coder (or lzma), no AI model",
                  "--bound B|lossless, --nc-predictor mean|raw, --nc-coder ac|lzma", "4.6 (model free raw/mean + AC), 4.1 (model free raw/mean)"),
    "xtc": ("GROMACS XTC", "--xtc-precision P (1/nm, default 1000; max error 5/P A) or --bound B (P = 5/B)", "4.1 (XTC precision 10/100/1000)"),
    "tng": ("GROMACS TNG via gmx trjconv (GROMACS needed to encode and decode)", "--tng-precision P (power of ten, default 1000) or --bound B", "4.1 (TNG)"),
    "trr": ("GROMACS TRR, uncompressed float32 reference", "--trr-double (float64)", "4.1 (TRR)"),
    "dcd": ("CHARMM/NAMD DCD, uncompressed float32", "none", "4.1 (DCD)"),
    "netcdf": ("AMBER NetCDF, float32", "none", "4.1 (NetCDF)"),
    "pdb": ("multi model PDB text", "--pdb-precision 0..3 (default 3) or --bound B, --pdb-gzip LEVEL (report: PDB + gzip -9)", "4.1 (PDB, PDB + gzip -9)"),
    "h5md": ("H5MD (HDF5) in the GROMACS layout; lossless gzip or lossy SZ3 (hdf5plugin)",
             "--h5md-compression none|gzip|sz3, --h5md-level 1..9, --h5md-chunk FRAMES, --h5md-sz3-abs NM or --bound B", "4.1 (H5MD), sz3 new"),
    "mdcompress": ("MDCompress v2 binary (external)", "--mdc-res FM or --bound B|lossless, --mdc-level L, --mdc-preset NAME, --mdc-batch N, --mdc-history H",
                   "4.1, 4.6 (MDCompress -l 2/4, archive)"),
    "pcazip": ("pyPcazip 3 (mdplus.pca), PCA with optional quantised residuals; lossy, bound not guaranteed",
               "--pca-variance EV (default 0.75) or --pca-components N, --pca-residual-scale S (per nm, default 200, 0 = PCA only), --pca-eigenvector-scale, --pca-entropy lzma, --pca-frame lab|fitted",
               "4.1 (pcazip)"),
    "compresstraj": ("compresstraj autoencoder trained per trajectory (decoder weights stored); lossy, no bound",
                     "--ct-compression C (default 20) or --ct-latent L, --ct-epochs E (default 200), --ct-layers 4096,1024, --ct-batch 128", "4.1b (compresstraj)"),
    "cg2all": ("coarse grained beads stored, all atoms rebuilt by cg2all; lossy, no bound",
               "--cg-model CalphaBasedModel|ResidueBasedModel|BackboneModel|..., --cg-precision P (1/nm, default 1000), --cg-store xtc|lzma, --cg-device cpu|cuda", "4.1 (cg2all)"),
    "lzma": ("quantise + time delta + byte planes + lzma", "--bound B|lossless|exact, --level 0..9 (default 9), --delta time|none", "generic baseline"),
    "zstd": ("quantise + time delta + byte planes + zstandard", "--bound B|lossless|exact, --level 1..22 (default 19), --delta time|none", "generic baseline"),
    "bz2": ("quantise + time delta + byte planes + bzip2", "--bound B|lossless|exact, --level 1..9 (default 9), --delta time|none", "generic baseline"),
    "gzip": ("quantise + time delta + byte planes + gzip (deflate)", "--bound B|lossless|exact, --level 1..9 (default 9), --delta time|none", "generic baseline"),
}


def _which(env, name, *fallback):
    for p in [os.environ.get(env), shutil.which(name), *fallback]:
        if p and os.path.exists(p): return p
    return None


def gmx_path(): return _which("MD_COMPRESS_GMX", "gmx", "/workspace/conda/envs/gmx/bin/gmx")
def mdc_path(): return _which("MDCOMPRESS_BIN", "mdcompress", "/workspace/bench_tools/mdcbin/mdcompress")


def _run(cmd, stdin=None):
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(" ".join(cmd) + "\n" + r.stderr[-1500:] + r.stdout[-500:])


def _bound(bound):
    """-> ('grid', b) | ('lossless', 0.005) | ('exact', None) | (None, None)"""
    if bound is None: return None, None
    if isinstance(bound, str):
        if bound == "lossless": return "lossless", 0.005
        if bound == "exact": return "exact", None
        bound = float(bound)
    if bound == 0: return "lossless", 0.005
    if bound < 0: raise ValueError("bound must be positive")
    return "grid", float(bound)


# ---------------------------------------------------------------- xtc
def _xtc_write(path, x, prec):
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile
    box = np.zeros((3, 3), np.float32)
    with XTCFile(path, "w") as f:
        for i, xi in enumerate(x): f.write(np.ascontiguousarray(xi / 10, np.float32), box, i, float(i), float(prec))


def _xtc_read(path):
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile
    with XTCFile(path) as f: return np.stack([fr.x for fr in f]).astype(np.float32) * 10


def enc_xtc(x, bound=None, xtc_precision=None, **_):
    kind, b = _bound(bound)
    if kind in ("lossless", "exact"): raise ValueError("xtc is lossy only; use --xtc-precision or a positive bound")
    p = float(xtc_precision) if xtc_precision else (5.0 / b if b else 1000.0)
    with tempfile.TemporaryDirectory() as d:
        _xtc_write(f"{d}/a.xtc", x, p); data = open(f"{d}/a.xtc", "rb").read()
    return data, dict(xtc_precision=p), 5.0 / p


def dec_xtc(data, T, N, opts, threads=4):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/a.xtc", "wb").write(data); return _xtc_read(f"{d}/a.xtc")


# ---------------------------------------------------------------- tng (gmx trjconv)
def _gro(path, N):
    with open(path, "w") as f:
        f.write("md_compress\n%d\n" % N)
        for i in range(N): f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % ((i + 1) % 100000, "MOL", "X", (i + 1) % 100000, 0, 0, 0))
        f.write("%10.5f%10.5f%10.5f\n" % (0, 0, 0))


def enc_tng(x, bound=None, tng_precision=None, **_):
    g = gmx_path()
    if g is None: raise RuntimeError("tng needs GROMACS (gmx); set MD_COMPRESS_GMX or put gmx on PATH")
    kind, b = _bound(bound)
    if kind in ("lossless", "exact"): raise ValueError("tng is lossy only; use --tng-precision or a positive bound")
    p = int(tng_precision) if tng_precision else (int(10 ** np.ceil(np.log10(5.0 / b) - 1e-9)) if b else 1000)
    if p < 1 or 10 ** round(np.log10(p)) != p: raise ValueError("tng_precision must be a power of ten (TNG stores decimal digits)")
    with tempfile.TemporaryDirectory() as d:
        # trjconv takes the TNG precision from an XTC input, so the XTC on the same grid is the lossless carrier
        _xtc_write(f"{d}/a.xtc", x, p); _gro(f"{d}/s.gro", x.shape[1])
        _run([g, "-quiet", "trjconv", "-f", f"{d}/a.xtc", "-s", f"{d}/s.gro", "-o", f"{d}/a.tng"], stdin="0\n")
        data = open(f"{d}/a.tng", "rb").read()
    return data, dict(tng_precision=p), 5.0 / p


def dec_tng(data, T, N, opts, threads=4):
    g = gmx_path()
    if g is None: raise RuntimeError("decoding tng needs GROMACS (gmx); set MD_COMPRESS_GMX or put gmx on PATH")
    import mdtraj as md
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/a.tng", "wb").write(data); _gro(f"{d}/s.gro", N)
        _run([g, "-quiet", "trjconv", "-f", f"{d}/a.tng", "-s", f"{d}/s.gro", "-o", f"{d}/a.trr"], stdin="0\n")
        with md.formats.TRRTrajectoryFile(f"{d}/a.trr") as f: return (f.read()[0] * 10).astype(np.float32)


# ---------------------------------------------------------------- mdcompress
def _trr(path, x):
    import mdtraj as md
    T = len(x)
    with md.formats.TRRTrajectoryFile(path, "w") as f:
        f.write(np.ascontiguousarray(x / 10, np.float32), time=np.arange(T, dtype=np.float32), step=np.arange(T, dtype=np.int32))


def enc_mdcompress(x, bound=None, mdc_res=None, mdc_level=2, mdc_preset=None, mdc_batch=None, mdc_history=None, threads=4, **_):
    m = mdc_path()
    if m is None: raise RuntimeError("mdcompress binary not found; set MDCOMPRESS_BIN or put mdcompress on PATH")
    kind, b = _bound(bound)
    if kind == "exact": raise ValueError("mdcompress is grid based; use a bound or lossless (0.01 A grid)")
    res = int(mdc_res) if mdc_res else (int(round(2 * b * 1e5)) if b else 1000)
    if res < 1000: raise ValueError("mdc_res must be >= 1000 fm (MDCompress minimum)")
    cmd = ["-l", str(int(mdc_level)), "--res", str(res), "-t", str(int(threads))]
    if mdc_preset: cmd += ["--preset", mdc_preset]
    if mdc_batch: cmd += ["-b", str(int(mdc_batch))]
    if mdc_history: cmd += ["-h", str(int(mdc_history))]
    with tempfile.TemporaryDirectory() as d:
        _trr(f"{d}/a.trr", x); open(f"{d}/m.desc", "w").write(f"MOL\tmol_1\t{x.shape[1]}\n")
        _run([m, "compress", "-i", f"{d}/a.trr", "-d", f"{d}/m.desc", "-o", f"{d}/a.mdc"] + cmd)
        data = open(f"{d}/a.mdc", "rb").read()
    return data, dict(mdc_res=res, mdc_level=int(mdc_level), mdc_preset=mdc_preset, mdc_batch=mdc_batch, mdc_history=mdc_history), res / 2e5


def dec_mdcompress(data, T, N, opts, threads=4):
    m = mdc_path()
    if m is None: raise RuntimeError("decoding needs the mdcompress binary; set MDCOMPRESS_BIN or put mdcompress on PATH")
    import mdtraj as md
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/a.mdc", "wb").write(data)
        _run([m, "decompress", "-i", f"{d}/a.mdc", "-o", f"{d}/a.trr", "-t", str(int(threads))])
        with md.formats.TRRTrajectoryFile(f"{d}/a.trr") as f: return (f.read()[0] * 10).astype(np.float32)


# ---------------------------------------------------------------- generic quantise + lzma / zstd / bz2
def _planes(u):  # uint32 [n] -> 4 byte planes
    return np.ascontiguousarray(u.view(np.uint8).reshape(-1, 4).T).tobytes()


def _unplanes(b, n): return np.ascontiguousarray(np.frombuffer(b, np.uint8).reshape(4, n).T).view(np.uint32).ravel()


def _gen_comp(name, raw, level, threads):
    import gzip
    if name == "gzip": return gzip.compress(raw, int(level), mtime=0)
    if name == "lzma": return lzma.compress(raw, preset=int(level))
    if name == "bz2": return bz2.compress(raw, int(level))
    import zstandard as zs
    return zs.ZstdCompressor(level=int(level), threads=int(threads)).compress(raw)


def _gen_decomp(name, data):
    import gzip
    if name == "gzip": return gzip.decompress(data)
    if name == "lzma": return lzma.decompress(data)
    if name == "bz2": return bz2.decompress(data)
    import zstandard as zs
    return zs.ZstdDecompressor().decompress(data)


DEF_LEVEL = {"lzma": 9, "zstd": 19, "bz2": 9, "gzip": 9}


def enc_generic(name, x, bound=None, level=None, delta="time", threads=4, **_):
    kind, b = _bound(bound)
    if kind is None: kind, b = "grid", 0.1
    level = DEF_LEVEL[name] if level is None else int(level)
    if delta not in ("time", "none"): raise ValueError("delta must be 'time' or 'none'")
    x = np.ascontiguousarray(x, np.float32)
    if kind == "exact": u = x.view(np.uint32).astype(np.int64); step = None
    else:
        step = GRID if kind == "lossless" else 2 * b
        u = np.round(x.astype(np.float64) / step).astype(np.int64)
    if delta == "time": u = np.concatenate([u[:1], np.diff(u, axis=0)])
    u = (u & 0xFFFFFFFF).astype(np.uint32).view(np.int32).astype(np.int64)  # wrap to int32
    z = ((u << 1) ^ (u >> 63)).astype(np.uint32)  # zigzag
    data = _gen_comp(name, _planes(z.ravel()), level, threads)
    opts = dict(kind=kind, step=step, level=level, delta=delta)
    return data, opts, (None if kind == "exact" else (0.005 if kind == "lossless" else b))


def dec_generic(name, data, T, N, opts, threads=4):
    z = _unplanes(_gen_decomp(name, data), T * N * 3).astype(np.int64)
    u = ((z >> 1) ^ -(z & 1)).reshape(T, N, 3)
    if opts["delta"] == "time": u = np.cumsum(u, axis=0)
    u = (u & 0xFFFFFFFF).astype(np.uint32)
    if opts["kind"] == "exact": return u.view(np.float32)
    return (u.view(np.int32).astype(np.float64) * opts["step"]).astype(np.float32)


# ---------------------------------------------------------------- model free + new coder
def enc_newcoder(x, cls, E, bound=None, nc_predictor="mean", nc_coder="ac", **_):
    from md_compress import codec as C
    RCD, RE = E["RCD"], E["RE"]
    kind, b = _bound(bound)
    if kind is None: kind, b = "grid", 0.1
    if kind == "exact": raise ValueError("new-coder is grid based; use a bound or lossless (0.01 A grid)")
    if nc_predictor not in ("mean", "raw"): raise ValueError("nc_predictor must be 'mean' or 'raw'")
    import torch
    truth = np.asarray(x, np.float64); T = len(truth)
    xf = torch.as_tensor(truth, dtype=torch.float32); Mn = RCD.iter_mean(xf).float(); Mnp = Mn.numpy().astype(np.float64)
    W = C._W()
    if nc_predictor == "mean":
        Rk, tk = E["kabsch"](Mn[None].expand(T, -1, 3), xf)
        q4 = C._rot_to_quat(Rk.double().numpy()); t32 = tk.numpy().astype(np.float32); M32 = Mn.numpy().astype(np.float32)
        p = np.einsum("na,tba->tnb", M32.astype(np.float64), C._quat_to_rot(q4)) + t32.astype(np.float64)[:, None]
        W.arr(M32, np.float32); W.arr(q4, np.float32); W.arr(t32, np.float32)
    else:
        p = np.zeros_like(truth)
    q, rf = RCD.quantise(truth, p, b)
    base, st = C._qbase(p, b, RCD)
    if nc_coder == "lzma":  # the report's 'model free mean' / 'model free raw' rows (best lzma arrangement per stream)
        blob = C._lz_enc(RCD, q); qd = C._lz_dec(RCD, blob, q.shape)
    elif nc_coder == "ac":
        blob, _ = RE.encode(q, cls, base, st, RE.make_refs(Mnp), RE.make_refs3(Mnp, None), None); qd = RE.decode(blob, q.shape, cls, base, st, None)
    else: raise ValueError("nc_coder must be 'ac' or 'lzma'")
    if not np.array_equal(qd, q): raise RuntimeError("coder roundtrip mismatch")
    W.sec(blob)
    return W.b.getvalue(), dict(nc_predictor=nc_predictor, nc_coder=nc_coder, kind=kind), b


def dec_newcoder(data, T, N, opts, cls, E):
    from md_compress import codec as C
    RCD, RE = E["RCD"], E["RE"]; b = opts["bound"]
    rd = C._R(data, 0)
    if opts["nc_predictor"] == "mean":
        M32 = rd.arr(np.float32, (N, 3)); q4 = rd.arr(np.float32, (T, 4)); t32 = rd.arr(np.float32, (T, 3))
        p = np.einsum("na,tba->tnb", M32.astype(np.float64), C._quat_to_rot(q4)) + t32.astype(np.float64)[:, None]
    else:
        p = np.zeros((T, N, 3))
    base, st = C._qbase(p, b, RCD)
    q = C._lz_dec(RCD, rd.sec(), (T, N, 3)) if opts.get("nc_coder") == "lzma" else RE.decode(rd.sec(), (T, N, 3), cls, base, st, None)
    _, rf = RCD.quantise(np.zeros_like(p), p, b)
    return rf(q).astype(np.float32)


# ================================================================ formats and external codecs added for the full report list
def _tmp_rw(writer, reader, x, ext):
    with tempfile.TemporaryDirectory() as d:
        writer(f"{d}/a.{ext}", x); return open(f"{d}/a.{ext}", "rb").read()


def _tmp_read(data, reader, ext):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/a.{ext}", "wb").write(data); return reader(f"{d}/a.{ext}")


# ---------------------------------------------------------------- trr / dcd / netcdf (float, uncompressed references)
def enc_trr(x, trr_double=False, **_):
    T = len(x)
    if trr_double:
        from MDAnalysis.lib.formats.libmdaxdr import TRRFile
        def w(p, x):
            with TRRFile(p, "w") as f:
                for i, xi in enumerate(x): f.write(np.ascontiguousarray(xi / 10, np.float64), None, None, np.zeros((3, 3)), i, float(i), 0.0, xi.shape[0])
    else:
        def w(p, x): _trr(p, x)
    return _tmp_rw(w, None, x, "trr"), dict(trr_double=bool(trr_double)), None


def dec_trr(data, T, N, opts, threads=4):
    def r(p):
        from MDAnalysis.lib.formats.libmdaxdr import TRRFile
        with TRRFile(p) as f: return np.stack([fr.x for fr in f]).astype(np.float64) * 10
    return _tmp_read(data, r, "trr").astype(np.float32)


def enc_dcd(x, **_):
    import mdtraj as md
    def w(p, x):
        with md.formats.DCDTrajectoryFile(p, "w") as f: f.write(np.ascontiguousarray(x, np.float32))
    return _tmp_rw(w, None, x, "dcd"), {}, None


def dec_dcd(data, T, N, opts, threads=4):
    import mdtraj as md
    def r(p):
        with md.formats.DCDTrajectoryFile(p) as f: return f.read()[0]
    return _tmp_read(data, r, "dcd").astype(np.float32)


def enc_netcdf(x, **_):
    import mdtraj as md
    def w(p, x):
        with md.formats.NetCDFTrajectoryFile(p, "w") as f: f.write(np.ascontiguousarray(x, np.float32))
    return _tmp_rw(w, None, x, "nc"), {}, None


def dec_netcdf(data, T, N, opts, threads=4):
    import mdtraj as md
    def r(p):
        with md.formats.NetCDFTrajectoryFile(p) as f: return f.read()[0]
    return _tmp_read(data, r, "nc").astype(np.float32)


# ---------------------------------------------------------------- pdb (text) and pdb + gzip
def enc_pdb(x, bound=None, pdb_precision=None, pdb_gzip=None, names=None, **_):
    """multi model PDB text, coordinates with pdb_precision decimals (default 3, the PDB format's own precision; max
    error 0.5 10^-p Angstrom). pdb_gzip: gzip level 1..9 (the report's 'PDB + gzip -9'), default none."""
    import gzip
    kind, b = _bound(bound)
    p = int(pdb_precision) if pdb_precision is not None else (3 if not b else max(0, min(3, int(np.ceil(np.log10(0.5 / b) - 1e-9)))))
    if not 0 <= p <= 3: raise ValueError("pdb_precision must be 0..3 (PDB has 8.3f coordinate fields)")
    T, N = x.shape[:2]; names = names or [("CA", "ALA")] * N
    out = io.StringIO(); fmt = f"%8.{p}f%8.{p}f%8.{p}f"
    head = ["ATOM  %5d %-4s %3s A%4d    " % ((i + 1) % 100000, (" " + a if len(a) < 4 else a)[:4], r, ((i // 1) + 1) % 10000) for i, (a, r) in enumerate(names)]
    for t in range(T):
        out.write("MODEL     %4d\n" % (t + 1))
        for i in range(N): out.write(head[i] + fmt % tuple(x[t, i]) + "  1.00  0.00\n")
        out.write("ENDMDL\n")
    data = out.getvalue().encode()
    if pdb_gzip: data = gzip.compress(data, int(pdb_gzip), mtime=0)
    return data, dict(pdb_precision=p, pdb_gzip=pdb_gzip), 0.5 * 10.0 ** (-p)


def dec_pdb(data, T, N, opts, threads=4):
    import gzip
    if opts.get("pdb_gzip"): data = gzip.decompress(data)
    rows = [l for l in data.decode().splitlines() if l.startswith("ATOM")]
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in rows], np.float32).reshape(T, N, 3)


# ---------------------------------------------------------------- H5MD (GROMACS 2025 style layout; gzip or SZ3 lossy)
def enc_h5md(x, bound=None, h5md_compression=None, h5md_level=None, h5md_chunk=None, h5md_sz3_abs=None, **_):
    """H5MD file (h5py): /h5md (version 1.1, author, creator), /particles/system/position/{value [T, N, 3] float32 nm,
    step, time} with unit attributes, as the GROMACS H5MD writer lays it out. h5md_compression: 'none', 'gzip'
    (lossless deflate, h5md_level 1..9, default 4) or 'sz3' (lossy SZ3 filter from hdf5plugin with an absolute error
    bound, h5md_sz3_abs in nm, or from bound b: b / 10 nm). h5md_chunk: frames per chunk (default 1 for gzip as in
    the report's 'gzip 4, chunk 1 frame', 100 for sz3)."""
    import h5py
    kind, b = _bound(bound)
    comp = h5md_compression or ("sz3" if (b and kind == "grid") or h5md_sz3_abs else "gzip")
    if comp not in ("none", "gzip", "sz3"): raise ValueError("h5md_compression must be none, gzip or sz3")
    T, N = x.shape[:2]
    kw = {}; maxerr = None
    if comp == "gzip": kw = dict(compression="gzip", compression_opts=int(h5md_level or 4))
    if comp == "sz3":
        try:
            import hdf5plugin
        except ImportError:
            raise RuntimeError("h5md sz3 needs the hdf5plugin package (pip install hdf5plugin)")
        if kind in ("lossless", "exact"): raise ValueError("sz3 is lossy; use h5md_compression='gzip' for lossless")
        absn = float(h5md_sz3_abs) if h5md_sz3_abs else (b / 10 if b else 0.0001)
        kw = dict(hdf5plugin.SZ3(absolute=absn)); maxerr = absn * 10
    ch = int(h5md_chunk or (100 if comp == "sz3" else 1)); ch = max(1, min(ch, T))
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(f"{d}/a.h5md", "w") as f:
            h = f.create_group("h5md"); h.attrs["version"] = np.array([1, 1], np.int32)
            h.create_group("author").attrs["name"] = "md_compress"; c = h.create_group("creator"); c.attrs["name"] = "md_compress"; c.attrs["version"] = "2"
            pos = f.create_group("particles/system/position")
            v = pos.create_dataset("value", data=np.ascontiguousarray(x / 10, np.float32), chunks=(ch, N, 3), **({} if comp == "none" else kw))
            v.attrs["unit"] = "nm"
            pos.create_dataset("step", data=np.arange(T, dtype=np.int64)); t = pos.create_dataset("time", data=np.arange(T, dtype=np.float64)); t.attrs["unit"] = "ps"
        data = open(f"{d}/a.h5md", "rb").read()
    return data, dict(h5md_compression=comp, h5md_level=h5md_level if comp == "gzip" else None, h5md_chunk=ch,
                      h5md_sz3_abs=kw and comp == "sz3" and absn or None), maxerr


def dec_h5md(data, T, N, opts, threads=4):
    import h5py
    if opts.get("h5md_compression") == "sz3": import hdf5plugin  # noqa: F401  registers the filter
    def r(p):
        with h5py.File(p) as f: return f["particles/system/position/value"][:] * 10
    return _tmp_read(data, r, "h5md").astype(np.float32)


# ---------------------------------------------------------------- pcazip (pyPcazip 3, mdplus.pca)
def enc_pcazip(x, pca_variance=None, pca_components=None, pca_residual_scale=None, pca_eigenvector_scale=None, pca_entropy=None, pca_frame=None, **_):
    """pyPcazip's pcazipsave on x / 10 (nm). pca_variance = explained variance (tool default 0.75); pca_components N
    selects the explained variance of the first N principal components of the superposed trajectory (pcazip then
    picks the number of components itself, which can differ by one); pca_residual_scale (per nm, tool default 200,
    0 = classic PCA only); pca_eigenvector_scale (tool default 100); pca_entropy 'lzma' adds an lzma pass over the
    pcz bytes. pcazip outputs the frames superposed on its average structure; the per frame rigid transform back to
    the input frame (quaternion + translation, 28 B per frame) is stored as well, which the benchmark did not count."""
    from mdplus.pca import pcazipsave, pcazipload
    from md_compress import codec as C
    import torch
    xs = np.asarray(x, np.float32)
    ev = float(pca_variance) if pca_variance else 0.75
    if pca_components:
        xf = torch.as_tensor(xs, dtype=torch.float64); M = xf.mean(0)
        from md_compress.geometry import kabsch
        for _ in range(3):
            R_, t_ = kabsch(xf, M[None].expand_as(xf)); fit = xf @ R_.transpose(-1, -2) + t_[:, None]; M = fit.mean(0)
        Z = (fit - M).reshape(len(xs), -1).numpy(); s = np.linalg.svd(Z, compute_uv=False) ** 2; cv = np.cumsum(s) / s.sum()
        ev = float(cv[min(int(pca_components), len(cv)) - 1]) - 1e-9
    rs = 200 if pca_residual_scale is None else float(pca_residual_scale); es = 100 if pca_eigenvector_scale is None else float(pca_eigenvector_scale)
    with tempfile.TemporaryDirectory() as d:
        pcazipsave(xs / 10.0, f"{d}/a.pcz", explained_variance=ev, residual_scale=rs, eigenvector_scale=es)
        pcz = open(f"{d}/a.pcz", "rb").read(); rec = np.asarray(pcazipload(f"{d}/a.pcz"), np.float64) * 10
    from md_compress.geometry import kabsch
    Rk, tk = kabsch(torch.as_tensor(rec), torch.as_tensor(xs, dtype=torch.float64))
    q4 = C._rot_to_quat(Rk.numpy()); t32 = tk.numpy().astype(np.float32)
    body = lzma.compress(pcz, preset=9) if pca_entropy == "lzma" else pcz
    frame = pca_frame or "lab"
    if frame not in ("lab", "fitted"): raise ValueError("pca_frame must be lab or fitted")
    W = C._W(); W.sec(body)
    if frame == "lab": W.arr(q4, np.float32); W.arr(t32, np.float32)
    return W.b.getvalue(), dict(pca_variance=ev, pca_components=pca_components, pca_residual_scale=rs, pca_eigenvector_scale=es, pca_entropy=pca_entropy,
                                pca_frame=frame), None


def dec_pcazip(data, T, N, opts, threads=4):
    from mdplus.pca import pcazipload
    from md_compress import codec as C
    rd = C._R(data, 0); body = rd.sec()
    if opts.get("pca_entropy") == "lzma": body = lzma.decompress(body)
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/a.pcz", "wb").write(body); rec = np.asarray(pcazipload(f"{d}/a.pcz"), np.float64) * 10
    if opts.get("pca_frame", "lab") == "fitted": return rec.astype(np.float32)  # pcazip's own output frame, as in the report
    q4 = rd.arr(np.float32, (T, 4)); t32 = rd.arr(np.float32, (T, 3))
    return (np.einsum("tna,tba->tnb", rec, C._quat_to_rot(q4)) + t32[:, None]).astype(np.float32)


# ---------------------------------------------------------------- compresstraj (SerpentByte/compresstraj, JCIM 2025)
CT_PATH = os.environ.get("COMPRESSTRAJ_PATH", "/workspace/bench_tools/compresstraj")


def _ct():
    import sys
    if CT_PATH not in sys.path: sys.path.insert(0, CT_PATH)
    from compresstraj.classes import DenseAutoEncoder, RMSDLoss, LightAutoEncoder
    from compresstraj.helpers import set_seed
    return DenseAutoEncoder, RMSDLoss, LightAutoEncoder, set_seed


def enc_compresstraj(x, ct_compression=None, ct_latent=None, ct_epochs=None, ct_layers=None, ct_batch=None, threads=4, **_):
    """compresstraj's own classes and steps (as the benchmark driver): per frame centre of geometry removed and stored
    (float64), global standard scaling, DenseAutoEncoder (layers 4096,1024), RMSD loss, AdamW lr 3e-4, batch 128,
    latent = ceil(3N / ct_compression) (default 20) or ct_latent; trained on this trajectory for ct_epochs (default
    200, the tool's CLI default; the report used 2000 on a GPU). The decoder weights (float32, as the tool saves them)
    are stored in the file, since decompression is impossible without them; that dominates the size."""
    import math, torch, pytorch_lightning as pl
    from torch.utils.data import DataLoader
    DenseAutoEncoder, RMSDLoss, LightAutoEncoder, set_seed = _ct()
    torch.set_num_threads(int(threads)); set_seed()
    xs = np.asarray(x, np.float32); T = len(xs)
    cog = xs.mean(1); xc = xs - cog[:, None]
    mean = float(xc.astype(np.float64).sum() / xc.size); sq = float((xc.astype(np.float64) ** 2).sum() / xc.size); std = math.sqrt(max(sq - mean ** 2, 1e-16))
    Xs = torch.as_tensor(((xc - mean) / std).reshape(T, -1).astype(np.float32)); Nin = Xs.shape[1]
    layers = [int(v) for v in (ct_layers or "4096,1024").split(",")] if isinstance(ct_layers, (str, type(None))) else list(ct_layers)
    c = int(ct_compression or 20); lat = max(int(ct_latent) if ct_latent else int(np.ceil(Nin / c)), 2); ep = int(ct_epochs or 200); bs = int(ct_batch or 128)
    ae = DenseAutoEncoder(N=Nin, latent=lat, layers=layers)
    model = LightAutoEncoder(model=ae, loss_fn=RMSDLoss(), learning_rate=3e-4)
    import logging; logging.getLogger("pytorch_lightning").setLevel(logging.ERROR); logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    tr = pl.Trainer(max_epochs=ep, accelerator="cpu", precision="32-true", enable_checkpointing=False, logger=False, enable_progress_bar=False, enable_model_summary=False)
    tr.fit(model, DataLoader(list(Xs), batch_size=bs, shuffle=True))
    with torch.no_grad():
        z = torch.cat([model.model.encoder.eval()(Xs[i:i + bs]) for i in range(0, T, bs)]).numpy().astype(np.float32)
    buf = io.BytesIO(); torch.save(model.model.decoder.state_dict(), buf)
    from md_compress import codec as C
    W = C._W(); W.sec(buf.getvalue()); W.arr(z, np.float32); W.arr(cog.astype(np.float64), np.float64); W.arr(np.float64([mean, std]), np.float64)
    return W.b.getvalue(), dict(ct_latent=lat, ct_epochs=ep, ct_layers=",".join(map(str, layers)), ct_batch=bs, ct_compression=c,
                                decoder_bytes=len(buf.getvalue()), latent_bytes=int(z.nbytes)), None


def dec_compresstraj(data, T, N, opts, threads=4):
    import torch
    DenseAutoEncoder, _, _, _ = _ct()
    from md_compress import codec as C
    rd = C._R(data, 0); sd = torch.load(io.BytesIO(rd.sec()), map_location="cpu"); lat = opts["ct_latent"]
    z = rd.arr(np.float32, (T, lat)); cog = rd.arr(np.float64, (T, 3)); mean, std = rd.arr(np.float64, (2,))
    ae = DenseAutoEncoder(N=3 * N, latent=lat, layers=[int(v) for v in opts["ct_layers"].split(",")]); ae.decoder.load_state_dict(sd)
    with torch.no_grad(): pos = ae.decoder.eval()(torch.as_tensor(z)).numpy().reshape(T, N, 3)
    pos = pos * std + mean; pos -= pos.mean(axis=1, keepdims=True); pos += cog[:, None]
    return pos.astype(np.float32)


# ---------------------------------------------------------------- cg2all (Heo and Feig): store CG beads, rebuild all atoms
CG2ALL_BIN = os.environ.get("MD_COMPRESS_CG2ALL_BIN", "/workspace/envs/cg2all/bin")
_CG_ALIAS = {("ILE", "CD"): "CD1", ("ANY", "OT1"): "O", ("ANY", "OT2"): "OXT", ("ANY", "O1"): "O", ("ANY", "O2"): "OXT"}


def _aa_pdb(path, topo, x0):
    """all atom PDB (one model) in atom14 order (+ OXT extra atoms on the last residue of each chain)."""
    from md_compress import residue_constants as rc
    aat, mk, ci = topo["aatype"], topo["atom_mask"] > 0, topo["chain_index"]
    L, k, Np = [], 0, int(mk.sum())
    for r in range(len(aat)):
        rn = rc.RESTYPES[aat[r]] if aat[r] < len(rc.RESTYPES) else "UNK"; nm = rc.atom14_names(rn)
        for s in range(14):
            if not mk[r, s]: continue
            a = nm[s]; L.append("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n" % (
                k + 1, (" " + a if len(a) < 4 else a), rn, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[ci[r] % 26], r + 1, *x0[k], a[0])); k += 1
    L.append("END\n"); open(path, "w").write("".join(L))
    return Np


def _cg_run(tool, *args):
    exe = os.path.join(CG2ALL_BIN, tool)
    if not os.path.exists(exe): raise RuntimeError(f"cg2all not found ({exe}); install cg2all and set MD_COMPRESS_CG2ALL_BIN")
    _run([exe, *args])


def enc_cg2all(x, topo, bound=None, cg_model=None, cg_precision=None, cg_device=None, cg_store=None, threads=4, **_):
    """Beads of cg_model (default CalphaBasedModel; any model of convert_all2cg) from the protein atoms, quantised on a
    1/cg_precision nm grid (default 1000), time delta + zigzag + byte planes + lzma. The CG topology PDB (one frame) is
    stored lzma compressed. Lossy with no error bound: --bound is ignored."""
    import mdtraj as md
    model = cg_model or "CalphaBasedModel"; p = float(cg_precision or 1000)
    with tempfile.TemporaryDirectory() as d:
        Np = _aa_pdb(f"{d}/aa.pdb", topo, x[0])
        with md.formats.DCDTrajectoryFile(f"{d}/aa.dcd", "w") as f: f.write(np.ascontiguousarray(x[:, :Np], np.float32))
        _cg_run("convert_all2cg", "-p", f"{d}/aa.pdb", "-o", f"{d}/cg.pdb", "--cg", model)
        _cg_run("convert_all2cg", "-p", f"{d}/aa.pdb", "-d", f"{d}/aa.dcd", "-o", f"{d}/cg.dcd", "--cg", model)
        cgtop = open(f"{d}/cg.pdb", "rb").read()
        with md.formats.DCDTrajectoryFile(f"{d}/cg.dcd") as f: beads = f.read()[0]
    store = cg_store or "xtc"  # xtc = as in the report (beads stored as XTC at cg_precision); lzma = quantise + delta + lzma
    from md_compress import codec as C
    W = C._W(); W.sec(lzma.compress(cgtop, preset=9))
    if store == "xtc":
        W.sec(_tmp_rw(lambda pth, xx: _xtc_write(pth, xx, p), None, beads * 10, "xtc"))
    elif store == "lzma":
        q = np.round(beads.astype(np.float64) / (10.0 / p)).astype(np.int64)
        dq = np.concatenate([q[:1], np.diff(q, axis=0)]); z = ((dq << 1) ^ (dq >> 63)).astype(np.uint32)
        W.sec(lzma.compress(_planes(z.ravel()), preset=9))
    else: raise ValueError("cg_store must be xtc or lzma")
    return W.b.getvalue(), dict(cg_model=model, cg_precision=p, cg_device=cg_device, cg_store=store, n_beads=int(beads.shape[1])), None


def dec_cg2all(data, T, N, opts, topo, threads=4):
    """all atoms rebuilt by convert_cg2all, mapped back to the atom14 order by residue and atom name; atoms cg2all does
    not output are NaN."""
    import mdtraj as md
    from md_compress import codec as C
    from md_compress import residue_constants as rc
    rd = C._R(data, 0); cgtop = lzma.decompress(rd.sec()); nb = opts["n_beads"]
    if opts.get("cg_store", "lzma") == "xtc":
        beads = (_tmp_read(rd.sec(), _xtc_read, "xtc") / 10).astype(np.float32)
    else:
        z = _unplanes(lzma.decompress(rd.sec()), T * nb * 3).astype(np.int64)
        q = np.cumsum(((z >> 1) ^ -(z & 1)).reshape(T, nb, 3), axis=0); beads = (q * (10.0 / opts["cg_precision"])).astype(np.float32)
    dev = opts.get("cg_device") or "cpu"
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/cg.pdb", "wb").write(cgtop)
        with md.formats.DCDTrajectoryFile(f"{d}/cg.dcd", "w") as f: f.write(beads)
        _cg_run("convert_cg2all", "-p", f"{d}/cg.pdb", "-d", f"{d}/cg.dcd", "-o", f"{d}/out.dcd", "-opdb", f"{d}/out.pdb",
                "--cg", opts["cg_model"], "--device", dev, "--proc", str(int(threads)))
        t = md.load(f"{d}/out.dcd", top=f"{d}/out.pdb")
    aat, mk = topo["aatype"], topo["atom_mask"] > 0
    slot = {}; k = 0
    for r in range(len(aat)):
        rn = rc.RESTYPES[aat[r]] if aat[r] < len(rc.RESTYPES) else "UNK"; nm = rc.atom14_names(rn)
        for s in range(14):
            if mk[r, s]: slot[(r, nm[s])] = k; k += 1
    out = np.full((T, N, 3), np.nan, np.float32); src, dst = [], []
    res = list(t.topology.residues)
    for ri, R_ in enumerate(res[:len(aat)]):
        for a in R_.atoms:
            if a.element.symbol == "H": continue
            nmx = _CG_ALIAS.get((R_.name, a.name), _CG_ALIAS.get(("ANY", a.name), a.name))
            if (ri, nmx) in slot: src.append(a.index); dst.append(slot[(ri, nmx)])
    out[:, dst] = t.xyz[:, src] * 10
    return out
