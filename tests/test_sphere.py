"""Tests of the bond sphere mode of the residual coder (gpu/residual_entropy.{c,py}, blob version 5).

    python tests/test_sphere.py          # all cases
    pytest tests/test_sphere.py
    python tests/test_sphere.py --write-fixture   # (re)writes tests/fixtures/sphere_v5_* (only when the format changes)
Checks: (1) the version 5 fixture and the legacy fixtures decode (fixture: SHA-256 of the decoded float32 coordinates);
(2) with the sphere off the coder output is version 4 and identical to an encode that does not pass the option;
(3) exact round trip through compress_file and decompress_file (new-coder raw and mean, lossless and 0.1 A), auto never
larger than off; (4) edge cases: one frame, fewer than three atoms, atoms with no neighbour, collinear neighbours,
negative square root arguments, NaN and infinite bond lengths passed to the C coder (must fall back, never convert NaN)."""
import hashlib, json, os, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIX = os.path.join(ROOT, "tests/fixtures")
TOP = "/workspace/bench_data/1e8rA00.pdb"; XYZ = "/workspace/bench_data/1e8rA00_T1000.npy"
FIXN = "sphere_v5_1e8rA00_T20_newcoder_raw"


def _re():
    from md_compress import codec as C
    return C._env(1)["RE"]


def _xyz20():
    c = np.load(f"{FIX}/legacy_1e8rA00_T20_input.npz")["coords"]; mk = np.abs(c).sum((0, 3)) > 0
    return np.ascontiguousarray(c[:, mk], np.float32)


def _h(a): return hashlib.sha256(np.ascontiguousarray(a, "<f4").tobytes()).hexdigest()


def write_fixture():
    from md_compress.codec import compress_xyz, decompress
    x = _xyz20(); exp = {}
    for bd, sm in (("lossless", "circle"), (0.1, "sphere")):
        blob = compress_xyz(x, "new-coder", bd, 1, nc_predictor="raw", nc_sphere=sm)
        nm = f"{FIXN}_{bd}_{sm}.mdz"; open(f"{FIX}/{nm}", "wb").write(blob)
        exp[nm] = _h(decompress(blob, 1, report=False)["xyz"])
    json.dump(exp, open(f"{FIX}/{FIXN}_expected.json", "w"), indent=1)


def test_fixtures_decode():
    from md_compress.codec import decompress
    exp = json.load(open(f"{FIX}/{FIXN}_expected.json"))
    for nm, h in exp.items():
        blob = open(f"{FIX}/{nm}", "rb").read(); r = decompress(blob, 1, report=False)
        assert _h(r["xyz"]) == h, nm
    x0 = np.load(f"{FIX}/legacy_1e8rA00_T20_input.npz")["coords"]
    for m in ("hybrid", "xtc", "zstd"):
        r = decompress(open(f"{FIX}/legacy_1e8rA00_T20_{m}.mdz", "rb").read(), 1, report=False)
        y = r["coords"] if "coords" in r else r["xyz"]; assert y.shape[0] == x0.shape[0]


def test_off_is_version4():
    RE = _re(); x = _xyz20().astype(np.float64); q = np.round(x / 0.01).astype(np.int64); T, N, _ = q.shape
    base = np.zeros_like(x); cls = np.full(N, 11, np.uint8); M = x.mean(0)
    a, _ = RE.encode(q, cls, base, 0.01, RE.make_refs(M), RE.make_refs3(M)); b, inf = RE.encode(q, cls, base, 0.01, RE.make_refs(M), RE.make_refs3(M), sphere="off")
    assert a == b and a[0] == RE.VERSION and inf["smode"] == "off"
    for sm in ("sphere", "circle", "auto"):
        c, inf = RE.encode(q, cls, base, 0.01, RE.make_refs(M), RE.make_refs3(M), sphere=sm)
        assert np.array_equal(RE.decode(c, q.shape, cls, base, 0.01), q)
        if sm != "auto": assert c[0] == RE.VERSION_SPHERE and inf["smode"] == sm
        else: assert len(c) <= len(a)


def test_auto_counts_option_bytes():
    """nc_sphere auto must count the container option bytes: (1) a margin larger than any saving gives the off blob,
    (2) the measured whole file overhead of the option fields is within methods.sphere_option_bytes(), so a file picked
    by auto is never larger than with off."""
    from md_compress import methods as MT
    from md_compress.codec import compress_xyz
    RE = _re(); x = _xyz20(); xd = x.astype(np.float64); q = np.round(xd / 0.01).astype(np.int64); N = q.shape[1]
    base = np.zeros_like(xd); cls = np.full(N, 11, np.uint8); M = xd.mean(0); args = (q, cls, base, 0.01, RE.make_refs(M), RE.make_refs3(M))
    off, _ = RE.encode(*args); big, inf = RE.encode(*args, sphere="auto", margin=10 ** 9)
    assert big == off and inf["smode"] == "off"
    a0, i0 = RE.encode(*args, sphere="auto"); assert i0["smode"] != "off"   # the sphere wins on this input
    m = i0["smode_sizes"]["off"] - i0["smode_sizes"][i0["smode"]]
    a1, i1 = RE.encode(*args, sphere="auto", margin=m); assert a1 == off   # saving not above the margin: off
    a2, i2 = RE.encode(*args, sphere="auto", margin=m - 1); assert i2["smode"] == i0["smode"]
    # whole file overhead of the option fields (forced mode, so the coder blob sizes are known)
    rec = {}; enc0 = RE.encode
    def spy(*a, **k):
        r = enc0(*a, **k); rec[k.get("sphere")] = len(r[0]); return r
    RE.encode = spy
    try:
        fo = compress_xyz(x, "new-coder", "lossless", 1, nc_predictor="raw", nc_sphere="off")
        fs = compress_xyz(x, "new-coder", "lossless", 1, nc_predictor="raw", nc_sphere="circle")
    finally: RE.encode = enc0
    overhead = (len(fs) - len(fo)) - (rec["circle"] - rec["off"])
    print("option overhead", overhead, "margin", MT.sphere_option_bytes()); assert 0 < overhead <= MT.sphere_option_bytes(), (overhead, MT.sphere_option_bytes())
    fa = compress_xyz(x, "new-coder", "lossless", 1, nc_predictor="raw", nc_sphere="auto"); assert len(fa) <= len(fo) + 8


def _files(tmp, T=40):
    import MDAnalysis as mda, warnings
    warnings.filterwarnings("ignore")
    u = mda.Universe(TOP); x = np.load(XYZ, mmap_mode="r")[:T]; p = os.path.join(tmp, "in.dcd")
    with mda.Writer(p, u.atoms.n_atoms) as w:
        for xi in x: u.atoms.positions = xi; w.write(u.atoms)
    return p


def test_file_roundtrip():
    from md_compress.codec import compress_file, decompress_file, decompress
    import MDAnalysis as mda
    with tempfile.TemporaryDirectory() as tmp:
        dcd = _files(tmp)
        for pred in ("raw", "mean"):
            for bd in ("lossless", 0.1):
                sizes = {}
                for sm in ("off", "auto"):
                    out = os.path.join(tmp, f"o_{pred}_{bd}_{sm}.mdz")
                    compress_file(TOP, dcd, out, bound=bd, method="new-coder", atoms="all", threads=1, method_opts=dict(nc_predictor=pred, nc_sphere=sm))
                    sizes[sm] = os.path.getsize(out)
                    rec = decompress(open(out, "rb").read(), 1, report=False)["xyz"]
                    back = os.path.join(tmp, f"b_{pred}_{bd}_{sm}.dcd"); decompress_file(out, TOP, back, threads=2, report=False)
                    u = mda.Universe(TOP, back); y = np.stack([u.atoms.positions.copy() for _ in u.trajectory])
                    assert np.array_equal(y, rec), (pred, bd, sm)          # file decode == in memory decode, other thread count
                    u0 = mda.Universe(TOP, dcd); x0 = np.stack([u0.atoms.positions.copy() for _ in u0.trajectory])
                    lim = 0.005 if bd == "lossless" else bd
                    assert np.abs(rec.astype(np.float64) - x0).max() <= lim + 1e-5, (pred, bd, sm)
                assert sizes["auto"] <= sizes["off"] + 64, sizes     # never larger than off by more than the option bytes


def _codec_roundtrip(q, cls, base, s, plan=None, sphere="circle"):
    RE = _re(); T, N, _ = q.shape; M = (base + s * q).mean(0)
    b, inf = RE.encode(q, cls, base, s, RE.make_refs(M), RE.make_refs3(M), sphere=sphere, plan=plan)
    assert np.array_equal(RE.decode(b, q.shape, cls, base, s), q)
    return b, inf


def test_edge_cases():
    RE = _re(); rng = np.random.default_rng(1)
    x = _xyz20().astype(np.float64); q = np.round(x / 0.01).astype(np.int64); N = q.shape[1]; cls = np.full(N, 11, np.uint8)
    # one frame: nothing to predict from, stays version 4
    b, inf = _codec_roundtrip(q[:1], cls, np.zeros_like(x[:1]), 0.01); assert b[0] == RE.VERSION
    # fewer than three atoms
    for n in (1, 2):
        _codec_roundtrip(q[:, :n], cls[:n], np.zeros_like(x[:, :n]), 0.01)
    # atoms with no neighbour: far apart gas of atoms (no bond like pairs) plus a molecule
    gas = rng.normal(0, 30, (20, 30, 3)) + np.cumsum(rng.normal(0, 0.5, (20, 30, 3)), 0)
    xg = np.concatenate([x[:, :40], gas], 1); qg = np.round(xg / 0.01).astype(np.int64); cg = np.full(qg.shape[1], 11, np.uint8)
    P = RE.bond_plan(xg); assert (P[40:, 0] < 0).all()
    _codec_roundtrip(qg, cg, np.zeros_like(xg), 0.01)
    # collinear neighbours: a straight chain (angle 180 degrees, circle degenerate) with noise
    t = np.arange(12)[None, :, None] * np.array([1.5, 0, 0]); xl = t + rng.normal(0, 0.02, (20, 12, 3)); ql = np.round(xl / 0.01).astype(np.int64)
    _codec_roundtrip(ql, np.full(12, 11, np.uint8), np.zeros_like(xl), 0.01)
    xl2 = np.broadcast_to(t, (20, 12, 3)).copy(); _codec_roundtrip(np.round(xl2 / 0.01).astype(np.int64), np.full(12, 11, np.uint8), np.zeros_like(xl2), 0.01)
    # negative square root argument: plan pairs at a large varying distance (the stored mean is often smaller than d1^2 + d2^2)
    xs = np.cumsum(rng.normal(0, 0.8, (20, 10, 3)), 0); qs = np.round(xs / 0.01).astype(np.int64)
    Ps = np.full((10, 3), -1, np.int32); Ps[1:, 0] = 0; Ps[1:, 1] = 0; Ps[1:, 2] = 1
    _codec_roundtrip(qs, np.full(10, 11, np.uint8), np.zeros_like(xs), 0.01, plan=Ps)
    # NaN / inf bond lengths and angle cosines straight into the C coder: must fall back and round trip
    M = xs.mean(0); ref, R3 = RE.make_refs(M), RE.make_refs3(M); W = np.zeros((RE.NG, 4), np.int16); W[:, 2] = 64
    base, s, xc, Nc = RE._geo(np.zeros_like(xs), 0.01, None, 10, 20); q32 = np.ascontiguousarray(qs, np.int32)
    for bad in (np.nan, np.inf, -1.0, 0.0):
        rb = np.full(10, bad); ct = np.full(10, bad if bad != 0 else 2.0)
        blob = RE._enc(q32, ref, np.full(10, 11, np.uint8), W, base, s, xc, Nc, np.ascontiguousarray(R3, np.int32).reshape(-1), 2, Ps, rb, ct)
        out = np.zeros((20, 10, 3), np.int32); pay = np.frombuffer(blob, np.uint8).copy()
        rc = RE._lib.re_decode(pay, len(pay), 20, 10, ref, np.full(10, 11, np.uint8), W, base, s, xc, Nc, np.ascontiguousarray(R3, np.int32).reshape(-1), out,
                               2, np.ascontiguousarray(Ps).reshape(-1), rb, ct)
        assert rc == 0 and np.array_equal(out, q32), bad


if __name__ == "__main__":
    if "--write-fixture" in sys.argv: write_fixture(); print("fixture written")
    for f in (test_fixtures_decode, test_off_is_version4, test_edge_cases, test_file_roundtrip):
        f(); print(f.__name__, "PASS", flush=True)
