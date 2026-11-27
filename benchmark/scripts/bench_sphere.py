#!/usr/bin/env python
"""Bond sphere coder benchmark through the codec (whole .mdz files, metadata trailer included).
Per system, T and bound: new-coder raw and mean with --nc-sphere off (production) and auto, hybrid (default ac coder;
its residual streams stay on version 4, and what auto would save on its two streams is measured separately by wrapping
the coder: the extra encodes are timed and subtracted from the hybrid encode time), and MDCompress through the codec
(level 2 default and level 4 archive preset; the smaller is the 'best' setting). Every file is decoded in the same
process (hybrid: the codec's own verify decode at compress time, which is also part of every encode time); max |error| <= bound is checked, and new-coder off and auto must decode to identical coordinates.
Results are appended to --out (JSON, resumable).
    python benchmark/scripts/bench_sphere.py --systems 1e8rA00,... --T 1000,250 --out results.json"""
import argparse, json, os, sys, time, traceback
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--systems", required=True); ap.add_argument("--T", default="1000,250")
ap.add_argument("--bounds", default="0.5,0.3,0.2,0.1,lossless"); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--methods", default="nc_raw_off,nc_raw_auto,nc_mean_off,nc_mean_auto,hybrid,mdc_l2,mdc_l4a")
a = ap.parse_args()
from md_compress import codec as C
RES = json.load(open(a.out)) if os.path.exists(a.out) else {}


def load(name, T):
    p = f"/workspace/bench_data/{name}.ours.npz"
    if os.path.exists(p):
        z = np.load(p); sl = slice(0, T); ex = z["extra"][sl] if "extra" in z.files and z["extra"].size else None
    else:
        z = np.load(f"/workspace/md_compress_gpu_data/data_long/{name}.npz"); sl = np.where(z["heldout"] == 0)[0][:T]; ex = None
    return dict(coords=z["coords"][sl], aatype=z["aatype"], atom_mask=z["atom_mask"], chain_index=z["chain_index"], chain_break=z["chain_break"], extra=ex)


def flat(d, r):
    mk = d["atom_mask"] > 0
    x = np.concatenate([np.asarray(d["coords"], np.float32)[:, mk]] + ([np.asarray(d["extra"], np.float32)] if d["extra"] is not None else []), 1)
    if "xyz" in r: return x, r["xyz"]
    y = np.concatenate([r["coords"][:, mk]] + ([r["extra"]] if r.get("extra") is not None else []), 1); return x, y


SPEC = {"nc_raw_off": ("new-coder", dict(nc_predictor="raw", nc_sphere="off")), "nc_raw_auto": ("new-coder", dict(nc_predictor="raw", nc_sphere="auto")),
        "nc_mean_off": ("new-coder", dict(nc_predictor="mean", nc_sphere="off")), "nc_mean_auto": ("new-coder", dict(nc_predictor="mean", nc_sphere="auto")),
        "hybrid": ("hybrid", {}), "mdc_l2": ("mdcompress", {}), "mdc_l4a": ("mdcompress", dict(mdc_level=4, mdc_preset="archive"))}

E = C._env(a.threads); RE = E["RE"]; _orig = RE.encode; HY = []; DT = []; _odec = C.decompress


def tdec(*args, **kw):
    t0 = time.perf_counter(); r = _odec(*args, **kw); DT.append(time.perf_counter() - t0); return r


C.decompress = tdec


def wrapped(*args, **kw):
    out = _orig(*args, **kw)
    if kw.get("sphere", "off") == "off":
        t0 = time.perf_counter(); b2, i2 = _orig(*args, **dict(kw, sphere="auto"))
        q, cls, base, s = args[0], args[1], args[2], args[3]; Xc = args[6] if len(args) > 6 else kw.get("Xc")
        assert np.array_equal(RE.decode(b2, np.asarray(q).shape, cls, base, s, Xc), np.asarray(q))
        HY.append(dict(off=len(out[0]), auto=len(b2), smode=i2["smode"], sizes=i2.get("smode_sizes"), N=int(np.asarray(q).shape[1]), extra_s=time.perf_counter() - t0))
    return out


EXCLUDE = {"mddb_membrane_Eprot"}   # dropped from all runs and reports (Pieter, t86u)
for sysn in [x for x in a.systems.split(",") if x not in EXCLUDE]:
    for T in [int(t) for t in a.T.split(",")]:
        d = load(sysn, T)
        if len(d["coords"]) < T: print("skip", sysn, T, flush=True); continue
        x0 = None; dec = {}
        for bd in a.bounds.split(","):
            bound = "lossless" if bd == "lossless" else float(bd); lim = 0.005 if bd == "lossless" else float(bd)
            for mname in a.methods.split(","):
                key = f"{sysn}/T{T}/{bd}/{mname}"
                if key in RES: continue
                method, opts = SPEC[mname]; rec = dict(system=sysn, T=T, bound=bd, method=mname)
                try:
                    HY.clear(); RE.encode = wrapped if method == "hybrid" else _orig
                    t0 = time.perf_counter()
                    blob, info = C.compress(d["coords"], d["aatype"], d["atom_mask"], d["chain_index"], d["chain_break"], bound=bound, extra=d["extra"],
                                            threads=a.threads, return_info=True, method=method, **opts)
                    te = time.perf_counter() - t0; RE.encode = _orig
                    if method == "hybrid":   # the codec's own verify decode (metadata trailer) is the decode measurement; no second decode
                        td = DT[-1]; err = float(info["measured"]["max_abs_err"]); x, _ = flat(d, dict(xyz=np.zeros((1, 1, 3)))); y = None
                    else:
                        t0 = time.perf_counter(); r = C.decompress(blob, a.threads, report=False); td = time.perf_counter() - t0
                        x, y = flat(d, r); err = float(np.abs(y.astype(np.float64) - x).max())
                    rec.update(bytes=len(blob), raw_bytes=12 * x.shape[0] * x.shape[1], ratio=12 * x.shape[0] * x.shape[1] / len(blob),
                               encode_s=te, decode_s=td, max_err=err, within=bool(err <= lim + 1e-5), N=int(x.shape[1]),
                               smode=(info.get("options") or {}).get("nc_smode"))
                    if mname.startswith("nc_"):
                        dec[(bd, mname)] = y
                        if mname.endswith("_auto") and (bd, mname.replace("auto", "off")) in dec:
                            rec["identical_to_off"] = bool(np.array_equal(y, dec.pop((bd, mname.replace("auto", "off"))))); dec.pop((bd, mname), None)
                    if method == "hybrid" and HY:
                        ex = sum(h["extra_s"] for h in HY); rec["encode_s"] = te - ex
                        rec["hybrid_streams"] = HY.copy(); rec["hybrid_auto_bytes"] = len(blob) - sum(h["off"] - h["auto"] for h in HY)
                except Exception as e:
                    RE.encode = _orig; rec.update(error=f"{type(e).__name__}: {e}"[:400]); traceback.print_exc()
                RES[key] = rec; json.dump(RES, open(a.out + ".tmp", "w"), indent=1, default=float); os.replace(a.out + ".tmp", a.out)
                print(time.strftime("%H:%M:%S"), key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("hybrid_streams",)}, flush=True)
print("done", flush=True)
