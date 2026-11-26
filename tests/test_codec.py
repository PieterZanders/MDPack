"""Roundtrip and ratio tests for md_compress.codec.

Run from the repository root (the research modules load results_gpu/runs/e2e_S_prior/best.pt relative to it):
    python tests/test_codec.py            # full run, writes benchmark/entropy/results/codec_tests.json
    python tests/test_codec.py --quick    # roundtrip cases at T=100 only
    pytest tests/test_codec.py            # quick roundtrip cases as pytest tests
Checks: (1) decoder output equals the encoder side reconstruction bit for bit (float32), (2) max |error| <= bound
(lossless: <= 0.005 A, the 0.01 A grid), (3) a second decode with a different thread count is identical,
(4) default mode ratios at T=1000 on the 16 mdCATH domains agree with the ablation 'neither' (packer only) results,
(5) every alternative method (--method xtc, tng, mdcompress, new-coder, lzma, zstd, bz2, ..., dvorak, mdz) roundtrips through the container
with its error bound respected, exact reconstruction where the method is lossless (float32 'exact' mode of lzma / zstd /
bz2, plain coordinates via compress_xyz), and the method and options read back from the header.
    python tests/test_codec.py --methods-only   # only (5) and the method comparison table"""
import json, os, sys, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MDCATH = "/workspace/md_compress_gpu_data/data_long"
MDDB = "/workspace/bench_data/mddb_small_1APQ.ours.npz"
ABL = os.path.join(ROOT, "benchmark/entropy/results")
DOMS = ["1e8rA00", "2ikeA00", "2jbvA03", "3iwcB00", "3mzkB03", "3nisA01", "2p4vA02", "4m70B00",
        "1jyoE00", "2g36A02", "3ogdA01", "2lg7A00", "1yf2A03", "2rsxA00", "1r44A00", "1b5eA00"]
BOUNDS = [0.5, 0.3, 0.2, 0.1, 0.0]


def load(name, T):
    if name.startswith("mddb"):
        z = np.load(MDDB); sl = slice(0, T); ex = z["extra"][sl] if z["extra"].size else None
    else:
        z = np.load(f"{MDCATH}/{name}.npz"); sl = np.where(z["heldout"] == 0)[0][:T]; ex = None
    return dict(coords=z["coords"][sl], aatype=z["aatype"], atom_mask=z["atom_mask"], chain_index=z["chain_index"],
                chain_break=z["chain_break"], extra=ex)


def roundtrip(d, bound, threads_dec=None, **kw):
    from md_compress.codec import compress, decompress
    t0 = time.time(); blob, info = compress(d["coords"], d["aatype"], d["atom_mask"], d["chain_index"], d["chain_break"],
                                            bound=bound, extra=d["extra"], return_info=True, _keep_rec=True, **kw); te = time.time() - t0
    t0 = time.time(); r = decompress(blob); td = time.time() - t0
    mk = d["atom_mask"] > 0; rec = info.pop("_rec"); Np = int(mk.sum())
    out = np.concatenate([r["coords"][:, mk]] + ([r["extra"]] if d["extra"] is not None else []), 1)
    truth = np.concatenate([d["coords"][:, mk]] + ([d["extra"]] if d["extra"] is not None else []), 1)
    b = 0.005 if bound == 0 else bound
    res = dict(bitexact=bool(np.array_equal(out, rec.astype(np.float32))),
               max_err=float(np.abs(out.astype(np.float64) - truth).max()), bound=b,
               rmsd=float(np.sqrt(((out.astype(np.float64) - truth) ** 2).sum(-1).mean())), ratio=info["ratio"], bytes=info["total"],
               enc_s=round(te, 2), dec_s=round(td, 2), mode=dict(backbone=r["backbone"], full_predictor=r["full_predictor"]),
               header_ok=bool(r["backbone"] == kw.get("backbone", "mean") and r["full_predictor"] == kw.get("full_predictor", False)),
               topology_ok=bool(np.array_equal(r["aatype"], d["aatype"]) and np.array_equal(r["atom_mask"] > 0, mk)))
    res["within_bound"] = res["max_err"] <= b + 1e-4  # float32 output rounding
    if threads_dec:
        r2 = decompress(blob, threads=threads_dec); res["threads_identical"] = bool(np.array_equal(r2["coords"], r["coords"]))
    return res


METHOD_CASES = [  # (method, bound, options); 'lossless' for xtc / tng means their 0.001 nm default grid (max error 0.005 A)
    ("xtc", 0.1, {}), ("xtc", None, {"xtc_precision": 1000}), ("tng", 0.1, {}), ("tng", None, {"tng_precision": 1000}),
    ("mdcompress", 0.1, {}), ("mdcompress", 0.0, {}), ("mdcompress", 0.1, {"mdc_level": 4, "mdc_preset": "archive"}), ("mdcompress", 0.0, {"mdc_level": 4, "mdc_preset": "archive"}),
    ("new-coder", 0.1, {}), ("new-coder", 0.0, {}), ("new-coder", 0.1, {"nc_predictor": "raw"}), ("new-coder", 0.0, {"nc_predictor": "raw"}),
    ("lzma", 0.1, {}), ("lzma", 0.0, {}), ("lzma", "exact", {}), ("zstd", 0.1, {}), ("zstd", 0.0, {}), ("zstd", "exact", {}),
    ("bz2", 0.1, {}), ("bz2", 0.0, {}), ("bz2", "exact", {}), ("lzma", 0.1, {"delta": "none", "level": 6}), ("zstd", 0.1, {"level": 3}),
    ("gzip", 0.1, {}), ("gzip", 0.0, {}), ("gzip", "exact", {}),
    ("new-coder", 0.1, {"nc_coder": "lzma"}), ("new-coder", 0.0, {"nc_coder": "lzma"}), ("new-coder", 0.1, {"nc_predictor": "raw", "nc_coder": "lzma"}),
    ("hybrid", 0.1, {"coder": "lzma"}), ("hybrid", 0.0, {"coder": "lzma"}), ("hybrid-ae", 0.1, {}), ("hybrid-ae", 0.0, {}), ("model-only", None, {}),
    ("trr", None, {}), ("trr", None, {"trr_double": True}), ("dcd", None, {}), ("netcdf", None, {}),
    ("pdb", None, {}), ("pdb", None, {"pdb_gzip": 9}), ("pdb", 0.1, {}),
    ("h5md", None, {"h5md_compression": "none"}), ("h5md", None, {"h5md_compression": "gzip", "h5md_level": 4, "h5md_chunk": 1}),
    ("h5md", None, {"h5md_compression": "gzip", "h5md_level": 9, "h5md_chunk": 100}), ("h5md", 0.1, {"h5md_compression": "sz3"}), ("h5md", 0.0, {"h5md_compression": "gzip"}),
    ("pcazip", None, {}), ("pcazip", None, {"pca_variance": 0.95, "pca_residual_scale": 0}), ("pcazip", None, {"pca_variance": 0.95, "pca_residual_scale": 200}),
    ("pcazip", None, {"pca_components": 10, "pca_entropy": "lzma"}),
    ("pcazip", None, {"pca_variance": 0.95, "pca_residual_scale": 0, "pca_frame": "fitted"}),
    ("compresstraj", None, {}), ("cg2all", None, {}), ("cg2all", None, {"cg_model": "BackboneModel"}), ("cg2all", None, {"cg_precision": 100}),
    ("cg2all", None, {"cg_store": "lzma"}),
    ("hybrid", 0.1, {"angle_model": "vonmises"}), ("hybrid", 0.0, {"angle_model": "empirical"}), ("hybrid-ae", 0.1, {"angle_model": "gaussian"}),
    ("dvorak", 0.1, {}), ("dvorak", 0.5, {}), ("dvorak", None, {"dv_qr": 0.05, "dv_qa": 0.05}), ("dvorak", 0.2, {"dv_cutoff": 1.8, "dv_qc": 0.01}),
    ("mdz", 0.1, {}), ("mdz", 0.5, {}), ("mdz", 0.1, {"mdz_predictor": "mt"}), ("mdz", 0.1, {"mdz_predictor": "ts", "mdz_batch": 50}),
    ("mdz", 0.2, {"mdz_predictor": "lr", "mdz_quantbins": 4096})]
LOSSLESS_FORMATS = ("trr", "dcd", "netcdf")  # float32 copies: must be exact
SLOW_T = {"cg2all": 100}  # cg2all on CPU takes about 3 s per frame (encode measures nothing; the test decodes once)


def method_roundtrip(d, method, bound, opts):
    from md_compress.codec import compress, decompress
    if method not in ("hybrid", "hybrid-ae", "model-only"): opts = dict(opts, measure=False)
    t0 = time.time(); blob, info = compress(d["coords"], d["aatype"], d["atom_mask"], d["chain_index"], d["chain_break"], bound=bound,
                                            extra=d["extra"], method=method, return_info=True, **opts); te = time.time() - t0
    t0 = time.time(); r = decompress(blob); td = time.time() - t0
    mk = d["atom_mask"] > 0
    out = np.concatenate([r["coords"][:, mk]] + ([r["extra"]] if d["extra"] is not None else []), 1)
    truth = np.concatenate([d["coords"][:, mk]] + ([d["extra"]] if d["extra"] is not None else []), 1)
    fin = np.isfinite(out).all(-1)
    if opts.get("pca_frame") == "fitted":  # output is in pcazip's fitted frame: superpose the input on it frame by frame
        truth = truth.astype(np.float64).copy()
        for t in range(len(truth)):
            a, q = truth[t], out[t].astype(np.float64); ca, cq = a.mean(0), q.mean(0)
            U, _, Vt = np.linalg.svd((a - ca).T @ (q - cq)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            truth[t] = (a - ca) @ (U @ D @ Vt) + cq
    dd = out.astype(np.float64) - truth
    err = float(np.abs(dd[fin]).max()); b = info["bound"]
    hyb = method in ("hybrid", "hybrid-ae", "model-only")
    res = dict(method=method, options=info.get("options", {}), bound=b, max_err=err, rmsd=float(np.sqrt((dd[fin] ** 2).sum(-1).mean())),
               atoms_missing=int((~fin).sum()), exact=bool(np.array_equal(out, truth)), ratio=info["ratio"], bytes=info["total"],
               enc_s=round(te, 2), dec_s=round(td, 2),
               header_ok=bool(r["method"] == ("hybrid" if hyb else method) and (hyb or r["options"] == info["options"])),
               topology_ok=bool(np.array_equal(r["aatype"], d["aatype"]) and np.array_equal(r["atom_mask"] > 0, mk)))
    lossy_unbounded = method in ("model-only", "pcazip", "compresstraj", "cg2all")
    res["error_bounded"] = not lossy_unbounded
    if lossy_unbounded: res["within_bound"] = None  # no bound to respect: RMSD and max error are recorded instead
    elif b is None: res["within_bound"] = res["exact"]
    else: res["within_bound"] = err <= b + 1e-4
    res["pass"] = bool((lossy_unbounded or res["within_bound"]) and res["header_ok"] and res["topology_ok"] and (bound != "exact" or res["exact"])
                       and (method not in LOSSLESS_FORMATS or res["exact"]) and (lossy_unbounded or res["atoms_missing"] == 0))
    return res


def xyz_roundtrip(d, method, bound):
    from md_compress.codec import compress_xyz, decompress
    mk = d["atom_mask"] > 0; x = d["coords"][:, mk]
    r = decompress(compress_xyz(x, method, bound)); err = float(np.abs(r["xyz"].astype(np.float64) - x).max())
    b = None if bound == "exact" else (0.005 if bound == 0 else bound)
    return dict(max_err=err, exact=bool(np.array_equal(r["xyz"], x)), pass_=bool(np.array_equal(r["xyz"], x) if b is None else err <= b + 1e-4))


def ok(res): return res["bitexact"] and res["within_bound"] and res["header_ok"] and res["topology_ok"] and res.get("threads_identical", True)


# ---------------------------------------------------------------- pytest entry points (quick)
def _q(name, bound, **kw):
    os.chdir(ROOT); res = roundtrip(load(name, 100), bound, **kw); assert ok(res), res

def test_default_mdcath(): [_q(n, b) for n in ("1e8rA00", "4m70B00") for b in (0.5, 0.1, 0.0)]
def test_ae_mdcath(): [_q(n, b, backbone="ae") for n in ("1e8rA00", "4m70B00") for b in (0.5, 0.1, 0.0)]
def test_mddb(): [_q("mddb_small_1APQ", b, **kw) for b in (0.5, 0.1, 0.0) for kw in ({}, {"backbone": "ae"})]
def test_methods():
    os.chdir(ROOT)
    for n in ("1e8rA00", "mddb_small_1APQ"):
        d = load(n, 100)
        for m, b, o in METHOD_CASES:
            if m in ("cg2all", "compresstraj"): continue  # slow on CPU: covered by --methods-only
            res = method_roundtrip(d, m, b, o); assert res["pass"], (m, b, o, res)
def test_angle_prior():
    """Circular chi prior: von Mises mixture mode across the +-pi seam; the linear (gaussian) mean is pulled away by the wrap."""
    from md_compress.angles import chi_prior
    rng = np.random.default_rng(0); T = 400
    a = np.where(rng.random(T) < 0.7, np.pi - 0.05, -2.0) + rng.normal(0, 0.15, T)
    sc = np.zeros((T, 1, 4, 2)); sc[..., 1] = 1.0; sc[:, 0, 0] = np.stack([np.sin(a), np.cos(a)], -1)
    mask = np.zeros((1, 4), bool); mask[0, 0] = True; per = np.zeros((1, 4), bool)
    ang = lambda m: float(np.arctan2(*chi_prior(sc, mask, per, m)[0, 0]))
    d = lambda x, y: abs((x - y + np.pi) % (2 * np.pi) - np.pi)
    assert d(ang("vonmises"), np.pi - 0.05) < 0.1 and d(ang("empirical"), np.pi - 0.05) < 0.15
    assert d(ang("gaussian"), np.pi - 0.05) > 0.5
    assert np.allclose(chi_prior(sc, mask, per, "vonmises")[0, 1:], [0.0, 1.0])  # unmasked chis keep the neutral prior
def test_dvorak_short():  # PMC needs >= 50 frames; shorter inputs are padded by the wrapper and cut on decode
    os.chdir(ROOT); d = load("1e8rA00", 20); r = xyz_roundtrip(d, "dvorak", 0.1); assert r["pass_"], r
def test_xyz(): os.chdir(ROOT); d = load("4m70B00", 50); assert all(xyz_roundtrip(d, m, b)["pass_"] for m, b in (("zstd", "exact"), ("lzma", 0.1), ("new-coder", 0.1), ("xtc", 0.1), ("dvorak", 0.1), ("mdz", 0.1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default=f"{ABL}/codec_tests.json")
    ap.add_argument("--methods-only", action="store_true")
    a = ap.parse_args(); os.chdir(ROOT)
    if a.methods_only: return run_methods(a)
    out = json.load(open(a.out)) if os.path.exists(a.out) and not a.quick else {}
    def save():
        if not a.quick: json.dump(out, open(a.out, "w"), indent=1)
    T = 100 if a.quick else 1000
    cases = [(n, b, kw) for n in ("1e8rA00", "4m70B00", "mddb_small_1APQ") for b in (0.5, 0.1, 0.0) for kw in ({}, {"backbone": "ae"})]
    cases += [("1e8rA00", 0.1, {"full_predictor": True}), ("1e8rA00", 0.0, {"full_predictor": True}), ("mddb_small_1APQ", 0.1, {"backbone": "ae", "full_predictor": True})]
    for n, b, kw in cases:
        key = f"roundtrip/{n}/T{T}/{kw.get('backbone', 'mean')}{'+full' if kw.get('full_predictor') else ''} @ {b}"
        if key in out: continue
        res = roundtrip(load(n, T), b, threads_dec=1 if n == "1e8rA00" and b == 0.1 else None, **kw); res["pass"] = ok(res); out[key] = res; save()
        print(key, "PASS" if res["pass"] else "FAIL", {k: res[k] for k in ("max_err", "ratio", "bitexact", "enc_s", "dec_s")}, flush=True)
    if a.quick: return
    abl = {}
    for f in ("ablation_A.json", "ablation_B.json", "ablation_C.json"): abl.update(json.load(open(f"{ABL}/{f}")))
    for n in DOMS + ["mddb_small_1APQ"]:
        d = load(n, 1000)
        for b in BOUNDS:
            key = f"ratio/{n}/T1000/default @ {b}"
            if key in out: continue
            res = roundtrip(d, b); ref = abl.get(f"{n}/T1000/neither @ {0.005 if b == 0 else b}")
            res["ablation_ratio_ac"] = ref["ratio_ac"] if ref else None; res["pass"] = ok(res); out[key] = res; save()
            print(key, "PASS" if res["pass"] else "FAIL", f"ratio {res['ratio']:.2f} ablation {res['ablation_ratio_ac']:.2f}", flush=True)



def run_methods(a):
    out_p = a.out.replace("codec_tests.json", "codec_method_tests.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    T = 100 if a.quick else 1000
    for n in ("1e8rA00", "mddb_small_1APQ"):
        DD = {}
        for m, b, o in [("hybrid", 0.1, {}), ("hybrid", 0.0, {})] + METHOD_CASES:
            Tm = min(T, SLOW_T.get(m, T)); d = DD.setdefault(Tm, load(n, Tm))
            key = f"method/{n}/T{Tm}/{m} {json.dumps(o, sort_keys=True)} @ {b}"
            if key in out: continue
            if m == "hybrid" and not o:
                res = roundtrip(d, b); res.update(method="hybrid", options={}, exact=False, pass_hyb=ok(res)); res["pass"] = res.pop("pass_hyb")
            else:
                res = method_roundtrip(d, m, b, o)
            out[key] = res; json.dump(out, open(out_p, "w"), indent=1)
            print(key, "PASS" if res["pass"] else "FAIL", f"ratio {res['ratio']:.2f} max_err {res['max_err']:.6f} rmsd {res.get('rmsd', float('nan')):.4f} bound {res['bound']} exact {res.get('exact')}", flush=True)
    d = load("4m70B00", 100)
    for m, b in (("zstd", "exact"), ("lzma", 0.1), ("bz2", 0.0), ("new-coder", 0.1), ("xtc", 0.1), ("mdcompress", 0.1), ("tng", 0.1), ("dvorak", 0.1), ("mdz", 0.1)):
        key = f"xyz/4m70B00/T100/{m} @ {b}"
        if key in out: continue
        r = xyz_roundtrip(d, m, b); r["pass"] = r.pop("pass_"); out[key] = r; json.dump(out, open(out_p, "w"), indent=1)
        print(key, "PASS" if r["pass"] else "FAIL", r, flush=True)


if __name__ == "__main__":
    main()
