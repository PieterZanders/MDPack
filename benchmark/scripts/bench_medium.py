#!/usr/bin/env python
"""Concise MDPack report benchmark: every method on ONE medium protein (default: mdCATH domain 4m70B00, the median heavy
atom count of the 16 T = 1,000 mdCATH benchmark domains; --system selects another domain or an MDDB system), all through the MDPack container (md_compress.codec.compress / decompress, metadata
trailer off; size = container bytes, shared model weights excluded). The reconstruction is mapped back to the PDB atom
order and scored with bench_lib.metrics plus extra dynamics metrics (radius of gyration, RMSD to frame 1 time series,
CA contact frequencies). Resumable: rows already in the output JSON are skipped.
  python bench_medium.py --stage main      # all methods at T = 1000
  python bench_medium.py --stage frames    # frame count study T = 10 .. 1000
  python bench_medium.py --stage angle     # --angle-model vonmises / empirical / gaussian against the default circular mean
  python bench_medium.py --stage tune      # sensitivity of dvorak and mdz to their own flags
  python bench_medium.py --stage slow      # compresstraj (CPU) and cg2all (CPU, T = 100)"""
import argparse, json, os, sys, time
import numpy as np
CWD0 = os.getcwd()  # --out is relative to where the script was started
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import bench_lib as B
import torch
from md_compress.codec import compress, decompress
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="main"); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--system", default="4m70B00", help="mdCATH domain (default 4m70B00: median heavy atom count of the 16 T1000 domains) or an MDDB name")
ap.add_argument("--out", default=None); ap.add_argument("--only", default="")
a = ap.parse_args(); SYS = a.system; RES = B.Results(os.path.join(CWD0, a.out) if a.out else f"{B.BENCH}/results/medium_{SYS}_raw.json")
DATA = os.environ.get("MDPACK_BENCH_DATA", "/workspace/bench_data")  # export_data.py / fetch_mddb.py output
LONG = os.environ.get("MDPACK_MDCATH_LONG", "/workspace/md_compress_gpu_data/data_long")  # mdCATH atom14 conversions
if SYS.startswith("mddb"):
    z = dict(np.load(f"{DATA}/{SYS}.ours.npz")); XP = np.load(f"{DATA}/{SYS}.npy").astype(np.float32); PDB = f"{DATA}/{SYS}.pdb"
else:  # the same 1,000 archive frames as bench_data/<sys>_T1000.npy (first 1,000 non held out frames)
    zl = np.load(f"{LONG}/{SYS}.npz"); arch = np.where(zl["heldout"] == 0)[0][:1000]
    z = {k: zl[k] for k in ("aatype", "atom_mask", "chain_index", "chain_break")}; z["coords"] = zl["coords"][arch]
    z["extra"] = np.zeros((len(arch), 0, 3), np.float32); z["temp"] = zl["temp"][arch]
    XP = np.load(f"{DATA}/{SYS}_T1000.npy").astype(np.float32); PDB = f"{DATA}/{SYS}.pdb"
top = B.read_pdb_topology(PDB)
mk = z["atom_mask"] > 0
def codec_order(c, e): return np.concatenate([c[:, mk]] + ([e] if e is not None and e.size else []), 1)
X0 = codec_order(z["coords"][:2], z["extra"][:2])
d, perm = cKDTree(X0[0]).query(XP[0]); assert d.max() < 2e-3 and len(set(perm)) == len(perm), d.max()  # PDB atom i = codec atom perm[i]
assert np.abs(X0[1][perm] - XP[1]).max() < 2e-3
CA = np.where(top["prot"] & (top["names"] == "CA"))[0]


def extra_metrics(rec, x):
    def rg(X): c = X.mean(1, keepdims=True); return np.sqrt(((X - c) ** 2).sum(-1).mean(1))
    def rmsd0(X):
        t = torch.as_tensor(X[:, CA], dtype=torch.float64); ref = t[:1].expand(len(t), -1, -1)
        f = B.superpose(t, ref); return ((f - ref) ** 2).sum(-1).mean(1).sqrt().numpy()
    def cont(X):
        c = X[:, CA]; out = np.zeros((len(CA), len(CA)))
        for s in range(0, len(X), 100): out += (np.linalg.norm(c[s:s + 100, :, None] - c[s:s + 100, None], axis=-1) < 8.0).sum(0)
        return out / len(X)
    gx, gr = rg(x), rg(rec); r0x, r0r = rmsd0(x), rmsd0(rec); cx, cr = cont(x), cont(rec)
    iu = np.triu_indices(len(CA), 3)
    return dict(rg_mae=float(np.abs(gx - gr).mean()), rg_mean_orig=float(gx.mean()), rg_mean_rec=float(gr.mean()),
                rmsd_to_frame1_mae=float(np.abs(r0x - r0r).mean()),
                rmsd_to_frame1_r=float(np.corrcoef(r0x, r0r)[0, 1]) if r0x.std() > 0 and r0r.std() > 0 else None,
                contact_freq_mae=float(np.abs(cx - cr)[iu].mean()), contact_freq_max=float(np.abs(cx - cr)[iu].max()))


def run(key, method, bound, opts, T):
    if RES.has(key) or (a.only and a.only not in key): return
    c, e = z["coords"][:T], (z["extra"][:T] if z["extra"].size else None); x = XP[:T].astype(np.float64)
    try:
        t0 = time.perf_counter()
        blob, info = compress(c, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], bound=bound, extra=e, method=method,
                              threads=a.threads, return_info=True, measure=False, metadata=False, **opts)
        te = time.perf_counter() - t0
        t0 = time.perf_counter(); r = decompress(blob, threads=a.threads); td = time.perf_counter() - t0
        rec = codec_order(r["coords"], r["extra"])[:, perm].astype(np.float64)
        fin = np.isfinite(rec).all(-1); miss = int((~fin).sum())
        if miss: rec = np.where(fin[..., None], rec, x)  # cg2all: atoms it does not output are filled with the truth and counted
        mt = B.metrics(rec, x, top); mt.update(extra_metrics(rec, x))
        b = info.get("bound")
        row = dict(system=SYS, T=T, method=method, bound_requested=bound, options=info.get("options"), bound=b, bytes=len(blob), raw_bytes=12 * x.shape[1] * T,
                   ratio=12 * x.shape[1] * T / len(blob), encode_s=te, decode_s=td, atoms_missing=miss, metrics=mt,
                   within_bound=(None if b is None or (isinstance(b, float) and np.isnan(b)) else bool(mt["max_abs_err"] <= b + 1e-4)))
        RES.put(key, row)
        print(f"{key}: {row['ratio']:.2f}x max {mt['max_abs_err']:.4f} rmsd {mt['rmsd_lab']:.4f} rmsf_r {mt['rmsf_r']:.5f} enc {te:.1f}s dec {td:.1f}s", flush=True)
    except Exception as ex:
        RES.put(key, dict(system=SYS, T=T, method=method, bound_requested=bound, options=opts, error=str(ex)[-1500:])); print("FAILED", key, str(ex)[-300:], flush=True)


BOUNDS = [0.5, 0.3, 0.2, 0.1]
if a.stage == "main":
    T = 1000
    for b in BOUNDS + [0.0]:
        bl = "lossless" if b == 0 else b
        run(f"T{T}/MDPack default (hybrid) @ {bl}", "hybrid", b, {}, T)
        run(f"T{T}/new-coder raw @ {bl}", "new-coder", b, {"nc_predictor": "raw"}, T)
        run(f"T{T}/new-coder mean @ {bl}", "new-coder", b, {"nc_predictor": "mean"}, T)
        for g in ("lzma", "zstd", "bz2", "gzip"): run(f"T{T}/{g} @ {bl}", g, b, {}, T)
        run(f"T{T}/mdcompress -l 2 @ {bl}", "mdcompress", b, {}, T)
        run(f"T{T}/mdcompress -l 4 archive @ {bl}", "mdcompress", b, {"mdc_level": 4, "mdc_preset": "archive"}, T)
        if b: 
            run(f"T{T}/dvorak @ {bl}", "dvorak", b, {}, T); run(f"T{T}/mdz @ {bl}", "mdz", b, {}, T)
            run(f"T{T}/xtc @ {bl}", "xtc", b, {}, T); run(f"T{T}/tng @ {bl}", "tng", b, {}, T)
            run(f"T{T}/h5md sz3 @ {bl}", "h5md", b, {"h5md_compression": "sz3"}, T)
            run(f"T{T}/pdb @ {bl}", "pdb", b, {}, T)
    run(f"T{T}/xtc precision 1000", "xtc", None, {"xtc_precision": 1000}, T); run(f"T{T}/tng precision 1000", "tng", None, {"tng_precision": 1000}, T)
    run(f"T{T}/model-only", "model-only", None, {}, T)
    for m, o in (("trr", {}), ("dcd", {}), ("netcdf", {}), ("pdb", {}), ("pdb", {"pdb_gzip": 9}), ("h5md", {"h5md_compression": "none"}),
                 ("h5md", {"h5md_compression": "gzip", "h5md_level": 4, "h5md_chunk": 1}), ("h5md", {"h5md_compression": "gzip", "h5md_level": 9, "h5md_chunk": 100}),
                 ("lzma", {}), ("zstd", {})):
        bb = "exact" if m in ("lzma", "zstd") else None
        run(f"T{T}/{m} {json.dumps(o, sort_keys=True)}" + (" @ exact" if bb else ""), m, bb, o, T)
    for ev in (0.7, 0.9, 0.95):
        run(f"T{T}/pcazip EV {ev} PCA only", "pcazip", None, {"pca_variance": ev, "pca_residual_scale": 0}, T)
    run(f"T{T}/pcazip EV 0.95 + residuals 20/nm", "pcazip", None, {"pca_variance": 0.95, "pca_residual_scale": 20}, T)
    run(f"T{T}/pcazip EV 0.75 + residuals 200/nm", "pcazip", None, {"pca_variance": 0.75, "pca_residual_scale": 200}, T)
elif a.stage == "frames":
    for T in (10, 50, 100, 250, 500):
        for b in (0.5, 0.1, 0.0):
            bl = "lossless" if b == 0 else b
            run(f"T{T}/MDPack default (hybrid) @ {bl}", "hybrid", b, {}, T)
            run(f"T{T}/new-coder raw @ {bl}", "new-coder", b, {"nc_predictor": "raw"}, T)
            run(f"T{T}/mdcompress -l 4 archive @ {bl}", "mdcompress", b, {"mdc_level": 4, "mdc_preset": "archive"}, T)
            run(f"T{T}/mdcompress -l 2 @ {bl}", "mdcompress", b, {}, T)
            run(f"T{T}/lzma @ {bl}", "lzma", b, {}, T)
            if b: run(f"T{T}/dvorak @ {bl}", "dvorak", b, {}, T); run(f"T{T}/mdz @ {bl}", "mdz", b, {}, T); run(f"T{T}/xtc @ {bl}", "xtc", b, {}, T)
        run(f"T{T}/model-only", "model-only", None, {}, T)
        run(f"T{T}/pcazip EV 0.9 PCA only", "pcazip", None, {"pca_variance": 0.9, "pca_residual_scale": 0}, T)
elif a.stage == "angle":  # static chi prior: circular statistics (md_compress/angles.py)
    for am in ("vonmises", "empirical", "gaussian"):
        for b in BOUNDS + [0.0]:
            bl = "lossless" if b == 0 else b
            run(f"T1000/MDPack default, angle model {am} @ {bl}", "hybrid", b, {"angle_model": am}, 1000)
        run(f"T1000/model-only, angle model {am}", "model-only", None, {"angle_model": am}, 1000)
elif a.stage == "tune":  # parameter sensitivity of the two new methods (not their defaults; labelled as tuning)
    for b in (0.5, 0.1):
        for qa in (0.03, 0.3):
            run(f"T1000/dvorak qa {qa} @ {b}", "dvorak", b, {"dv_qa": qa}, 1000)
        for bt in (1, 50):
            run(f"T1000/mdz batch {bt} @ {b}", "mdz", b, {"mdz_batch": bt}, 1000)
        for pr in ("mt", "ts"):
            run(f"T1000/mdz {pr} @ {b}", "mdz", b, {"mdz_predictor": pr}, 1000)
elif a.stage == "slow":
    run("T100/cg2all CA", "cg2all", None, {}, 100)
    run("T100/MDPack default (hybrid) @ 0.1", "hybrid", 0.1, {}, 100)
    run("T1000/compresstraj c 20", "compresstraj", None, {}, 1000)
print("done", flush=True)
