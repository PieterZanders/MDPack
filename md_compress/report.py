"""Metadata stored in every .mdz file and the JSON report written on decompression.

The metadata is a trailer after the compressed data: zlib(JSON) + uint32 length + b"MDZJ". Files written before the
trailer existed simply have none; they still decode, and their report holds what the header tells."""
import json, os, struct, zlib
import numpy as np

TRAILER = b"MDZJ"
SCHEMA = 1
TOL = 1e-5  # Angstrom, float32 rounding allowance when counting atoms within the error limit
UNBOUNDED = ("model-only", "pcazip", "compresstraj", "cg2all")
EXACT_FORMATS = ("trr", "dcd", "netcdf")
MAX_SAMPLE = 20_000_000  # abs error values kept for median / p99; beyond that a regular subsample is used


def split(blob):
    """-> (core bytes, metadata dict or None)."""
    if len(blob) > 12 and blob[-4:] == TRAILER:
        n = struct.unpack("<I", blob[-8:-4])[0]
        if 0 < n <= len(blob) - 12:
            try: return blob[:-8 - n], json.loads(zlib.decompress(blob[-8 - n:-8]))
            except Exception: pass
    return blob, None


def attach(core, meta):
    """append the metadata. The stored size section holds the core (compressed data) bytes; total bytes, metadata bytes and
    the ratios are filled in from the actual file length when the file is read (sizes() below), so they count every byte."""
    meta["size"]["core_bytes"] = len(core)
    z = zlib.compress(json.dumps(meta, separators=(",", ":")).encode(), 9)
    return core + z + struct.pack("<I", len(z)) + TRAILER


def sizes(size, n_total, n_core):
    raw = size["raw_float32_bytes"]
    return dict(size, core_bytes=n_core, metadata_bytes=n_total - n_core, total_bytes=n_total, ratio=raw / n_total, ratio_without_metadata=raw / n_core)


# ---------------------------------------------------------------- atom groups
def groups_atom14(atom_mask, Mx):
    """atom order of the codec: atom14 slots of present atoms (residue major), then Mx extra (OXT) atoms."""
    j = np.where(np.asarray(atom_mask) > 0)[1]; n = len(j)
    ex = np.arange(n, n + Mx)
    return {"all": np.arange(n + Mx), "backbone": np.concatenate([np.where(j < 4)[0], ex]), "side_chain": np.where(j >= 4)[0],
            "ca": np.where(j == 1)[0]}


def groups_names(names, is_protein):
    """plain coordinates with atom names (MDAnalysis selection): protein N, CA, C, O, OXT = backbone; other protein = side chain."""
    names = np.asarray(names).astype(str); p = np.asarray(is_protein, bool)
    bb = p & np.isin(names, ["N", "CA", "C", "O", "OXT", "OT1", "OT2"])
    g = {"all": np.arange(len(names))}
    if p.any():
        g.update(backbone=np.where(bb)[0], side_chain=np.where(p & ~bb)[0], ca=np.where(p & (names == "CA"))[0])
    if (~p).any() and p.any(): g["non_protein"] = np.where(~p)[0]
    return g


# ---------------------------------------------------------------- error statistics
def _kabsch_rmsd(a, b, w):
    """per frame RMSD after optimal superposition of a on b; a, b [t,n,3], w [t,n] (0 = missing)."""
    ws = w.sum(1, keepdims=True); ws = np.maximum(ws, 1)
    ca = (a * w[..., None]).sum(1, keepdims=True) / ws[..., None]; cb = (b * w[..., None]).sum(1, keepdims=True) / ws[..., None]
    A = (a - ca) * w[..., None]; B = (b - cb) * w[..., None]
    U, S, Vt = np.linalg.svd(np.einsum("tni,tnj->tij", A, B))
    d = np.sign(np.linalg.det(U @ Vt)); S[:, 2] *= d
    e = (A ** 2).sum((1, 2)) + (B ** 2).sum((1, 2)) - 2 * S.sum(1)
    return np.sqrt(np.maximum(e, 0) / ws[:, 0])


def error_stats(rec, truth, limit=None, groups=None, chunk=100):
    """rec, truth [T,N,3] (Angstrom). Absolute errors are per Cartesian component, as the error limit is."""
    T, N = truth.shape[:2]; groups = groups or {"all": np.arange(N)}
    acc = {k: dict(n=0, se=0.0, mx=0.0, sm=0.0, mxd=0.0, within=0, miss=0, fr=[], frs=[], vals=[]) for k, g in groups.items() if len(g)}
    tot = sum(len(groups[k]) for k in acc) * T * 3; step = max(1, int(np.ceil(tot / MAX_SAMPLE)))
    exact = True
    for s in range(0, T, chunk):
        r = np.asarray(rec[s:s + chunk], np.float64); t = np.asarray(truth[s:s + chunk], np.float64)
        exact = exact and np.array_equal(np.asarray(rec[s:s + chunk], np.float32), np.asarray(truth[s:s + chunk], np.float32))
        for k, a in acc.items():
            g = groups[k]; rr, tt = r[:, g], t[:, g]; ok = np.isfinite(rr).all(-1)
            d = np.where(ok[..., None], rr - tt, 0.0); ad = np.abs(d); d2 = (d ** 2).sum(-1)
            a["n"] += int(ok.sum()); a["se"] += float(d2.sum()); a["sm"] += float(ad.sum()); a["miss"] += int((~ok).sum())
            a["mx"] = max(a["mx"], float(ad.max()) if ad.size else 0.0); a["mxd"] = max(a["mxd"], float(np.sqrt(d2.max())) if d2.size else 0.0)
            if limit is not None: a["within"] += int((ok & (ad.max(-1) <= limit + TOL)).sum())
            a["fr"].append(np.sqrt(d2.sum(1) / np.maximum(ok.sum(1), 1)))
            a["frs"].append(_kabsch_rmsd(np.where(ok[..., None], rr, 0.0), tt, ok.astype(np.float64)) if len(g) >= 3 else np.zeros(len(rr)))
            a["vals"].append(ad[ok].ravel()[::step].astype(np.float32))
    out = {}
    for k, a in acc.items():
        fr, frs, v = np.concatenate(a["fr"]), np.concatenate(a["frs"]), np.concatenate(a["vals"])
        n = max(a["n"], 1)
        out[k] = dict(atoms=int(len(groups[k])), rmsd_lab=float(np.sqrt(a["se"] / n)), rmsd_superposed_mean=float(frs.mean()),
                      abs_error=dict(mean=a["sm"] / (3 * n), median=float(np.median(v)) if v.size else 0.0,
                                     p99=float(np.percentile(v, 99)) if v.size else 0.0, max=a["mx"]),
                      max_displacement=a["mxd"],
                      fraction_within_limit=None if limit is None else a["within"] / n,
                      per_frame_rmsd_lab=dict(mean=float(fr.mean()), min=float(fr.min()), max=float(fr.max())),
                      atoms_missing=a["miss"])
    return dict(bit_exact=bool(exact), percentiles_from_subsample=step > 1, tolerance_angstrom=TOL if limit is not None else None,
                abs_error_is="per Cartesian component, Angstrom", groups=out)


# ---------------------------------------------------------------- resolution
def resolution(method, opts, bound, residuals=True):
    o = opts or {}
    if method in UNBOUNDED or not residuals:
        kind = "no_limit"
    elif method in EXACT_FORMATS or o.get("kind") == "exact" or (method == "h5md" and o.get("h5md_compression") != "sz3"):
        kind = "exact"
    elif bound is not None and abs(bound - 0.005) < 1e-12 and method in ("hybrid", "hybrid-ae", "new-coder", "lzma", "zstd", "bz2", "gzip", "mdcompress"):
        kind = "lossless"
    else:
        kind = "bounded"
    p = {}
    if method in ("hybrid", "hybrid-ae", "new-coder", "lzma", "zstd", "bz2", "gzip") and kind in ("bounded", "lossless"):
        p = {"grid": "0.01 Angstrom absolute grid (source precision)"} if kind == "lossless" else {"grid": "residual grid", "step_angstrom": 2 * bound}
    elif method == "xtc": p = {"xtc_precision_per_nm": o.get("xtc_precision"), "step_angstrom": 10 / o["xtc_precision"]} if o.get("xtc_precision") else {}
    elif method == "tng": p = {"tng_precision_per_nm": o.get("tng_precision"), "step_angstrom": 10 / o["tng_precision"]} if o.get("tng_precision") else {}
    elif method == "pdb": p = {"decimals": o.get("pdb_precision"), "step_angstrom": 10.0 ** -o["pdb_precision"]} if o.get("pdb_precision") is not None else {}
    elif method == "mdcompress": p = {"resolution_fm": o.get("mdc_res"), "step_angstrom": o["mdc_res"] * 1e-5} if o.get("mdc_res") else {}
    elif method == "h5md" and o.get("h5md_compression") == "sz3": p = {"sz3_absolute_nm": o.get("h5md_sz3_abs")}
    elif method == "cg2all": p = {"beads": o.get("n_beads"), "bead_precision_per_nm": o.get("cg_precision"), "bead_store": o.get("cg_store"),
                                  "note": "all atoms are rebuilt by cg2all; no error limit"}
    elif method == "pcazip": p = {k: o.get(k) for k in ("pca_variance", "pca_components", "pca_residual_scale", "pca_eigenvector_scale")}
    elif method == "compresstraj": p = {k: o.get(k) for k in ("ct_latent", "ct_compression", "ct_epochs")}
    elif method in EXACT_FORMATS or kind == "exact": p = {"storage": "float64" if o.get("trr_double") else "float32"}
    return dict(kind=kind, error_limit_angstrom=None if kind in ("no_limit", "exact") else bound, stored_precision=p,
                respects_bound=kind != "no_limit")


# ---------------------------------------------------------------- output formats
ROUNDING = {".xtc": (0.005, "XTC written at precision 1000 per nm (0.001 nm grid): adds up to 0.005 Angstrom per coordinate"),
            ".gro": (0.005, "GRO text with 3 decimals in nm: adds up to 0.005 Angstrom per coordinate"),
            ".pdb": (0.0005, "PDB text with 3 decimals in Angstrom: adds up to 0.0005 Angstrom per coordinate"),
            ".dcd": (0.0, "DCD float32: no extra rounding"), ".trr": (0.0, "TRR float32: no extra rounding"),
            ".nc": (0.0, "NetCDF float32: no extra rounding"), ".ncdf": (0.0, "NetCDF float32: no extra rounding"),
            ".h5": (0.0, "HDF5 float32: no extra rounding"), ".h5md": (0.0, "H5MD float32: no extra rounding")}


def output_note(path):
    e = os.path.splitext(path)[1].lower(); r = ROUNDING.get(e)
    return dict(format=e.lstrip(".") or None, extra_rounding_max_angstrom=r[0] if r else None,
                note=r[1] if r else "format rounding unknown for this extension")


def json_path(out):
    return os.path.splitext(out)[0] + ".json"


def clean(o):
    """numpy scalars / arrays -> JSON types."""
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and not np.isfinite(o): return None
    return o
