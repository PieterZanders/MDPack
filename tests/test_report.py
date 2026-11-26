"""Tests of the metadata trailer and the JSON report that decompression always writes.

    python tests/test_report.py          # all cases, writes benchmark/entropy/results/report_tests.json
    pytest tests/test_report.py          # same cases except the slow cg2all and compresstraj ones
Per case: compress_file on a small mdCATH trajectory (1e8rA00, DCD written to a temporary directory), decompress_file with
--ref, then checks (1) <out stem>.json exists, (2) the required fields are present, (3) the stored error statistics equal
the ones recomputed against --ref, (4) the sizes count every byte (total_bytes = file size, ratio = raw / total), (5) the
resolution kind is the expected one. Legacy files (tests/fixtures, written by the codec before the metadata existed)
must still decode and give a JSON with what the header tells. Temporary files are removed."""
import json, os, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TOP = "/workspace/bench_data/1e8rA00.pdb"; XYZ = "/workspace/bench_data/1e8rA00_T1000.npy"
FIX = os.path.join(ROOT, "tests/fixtures")
OUT = os.path.join(ROOT, "benchmark/entropy/results/report_tests.json")
REQ = ("schema", "codec_version", "method", "options", "input", "system", "size", "compress_seconds", "resolution", "errors",
       "decompress_seconds", "output", "metadata_present", "errors_vs_ref")
STAT = ("rmsd_lab", "rmsd_superposed_mean", "abs_error", "max_displacement", "fraction_within_limit", "per_frame_rmsd_lab", "atoms_missing")
# (method, bound, options, atoms, expected resolution kind, frames, output extension, stride)
CASES = [("hybrid", 0.1, {}, "protein", "bounded", 40, ".dcd", 1),
         ("hybrid", "lossless", {}, "protein", "lossless", 40, ".xtc", 2),
         ("hybrid-ae", 0.1, {}, "protein", "bounded", 40, ".dcd", 1),
         ("model-only", None, {}, "protein", "no_limit", 40, ".dcd", 1),
         ("new-coder", 0.1, {"nc_predictor": "raw"}, "protein", "bounded", 40, ".dcd", 1),
         ("xtc", None, {"xtc_precision": 1000}, "protein", "bounded", 40, ".dcd", 1),
         ("tng", 0.05, {}, "protein", "bounded", 40, ".dcd", 1),
         ("mdcompress", 0.1, {}, "protein", "bounded", 40, ".dcd", 1),
         ("zstd", "exact", {}, "all", "exact", 40, ".dcd", 1),
         ("gzip", 0.1, {}, "protein", "bounded", 40, ".dcd", 1),
         ("trr", None, {}, "protein", "exact", 40, ".dcd", 1),
         ("dcd", None, {}, "all", "exact", 40, ".dcd", 1),
         ("netcdf", None, {}, "protein", "exact", 40, ".dcd", 1),
         ("pdb", None, {"pdb_precision": 2}, "protein", "bounded", 40, ".dcd", 1),
         ("h5md", 0.1, {"h5md_compression": "sz3"}, "protein", "bounded", 40, ".dcd", 1),
         ("h5md", None, {"h5md_compression": "gzip"}, "name CA", "exact", 40, ".dcd", 1),
         ("pcazip", None, {}, "protein", "no_limit", 40, ".dcd", 1),
         ("compresstraj", None, {"ct_epochs": 5}, "protein", "no_limit", 40, ".dcd", 1),
         ("cg2all", None, {}, "protein", "no_limit", 5, ".dcd", 1)]
SLOW = ("compresstraj", "cg2all")


def _write_input(path, T):
    import MDAnalysis as mda, warnings
    warnings.filterwarnings("ignore")
    u = mda.Universe(TOP); x = np.load(XYZ, mmap_mode="r")[:T]
    with mda.Writer(path, u.atoms.n_atoms) as w:
        for xi in x: u.atoms.positions = xi; w.write(u.atoms)


def _close(a, b, path=""):
    """stored statistics vs recomputed ones, recursively (floats to 1e-9 relative)."""
    if isinstance(a, dict):
        return all(_close(a[k], b.get(k), f"{path}/{k}") for k in a)
    if isinstance(a, float) and isinstance(b, (float, int)): return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))
    return a == b


def report_case(method, bound, opts, atoms, kind, T, ext, stride, tmp):
    from md_compress.codec import compress_file, decompress_file
    ref = os.path.join(tmp, f"in_{T}.dcd")
    if not os.path.exists(ref): _write_input(ref, T)
    mdz = os.path.join(tmp, "t.mdz"); out = os.path.join(tmp, "out" + ext)
    t0 = time.time(); info = compress_file(TOP, ref, mdz, bound=bound, method=method, atoms=atoms, method_opts=dict(opts), stride=stride)
    r = decompress_file(mdz, TOP, out, ref=ref)
    jp = os.path.splitext(out)[0] + ".json"; exists = os.path.exists(jp)
    J = json.load(open(jp)) if exists else {}
    missing = [k for k in REQ if k not in J]
    g = (J.get("errors") or {}).get("groups", {}); gr = J.get("errors_vs_ref", {}).get("decoded", {}).get("groups", {})
    stat_missing = [f"{n}/{k}" for n, v in g.items() for k in STAT if k not in v]
    fsize = os.path.getsize(mdz); S = J.get("size", {})
    res = dict(method=method, bound=bound, options=opts, atoms=atoms, frames=T, stride=stride, output=ext,
               json_exists=exists, missing_fields=missing, missing_stats=stat_missing, groups=sorted(g),
               stats_match_ref=bool(g) and _close(g, gr), sizes_ok=S.get("total_bytes") == fsize and np.isclose(S.get("ratio", 0), S.get("raw_float32_bytes", 0) / fsize)
               and S.get("metadata_bytes", -1) + S.get("core_bytes", 0) == fsize,
               kind=J.get("resolution", {}).get("kind"), kind_ok=J.get("resolution", {}).get("kind") == kind,
               api_same=J.get("method") == info["metadata"]["method"] and J.get("size", {}).get("total_bytes") == info["total"],
               frames_ok=J.get("system", {}).get("frames") == len(range(0, T, stride)),
               total_bytes=fsize, metadata_bytes=S.get("metadata_bytes"), ratio=S.get("ratio"), ratio_without_metadata=S.get("ratio_without_metadata"),
               rmsd_lab=g.get("all", {}).get("rmsd_lab"), max_abs=g.get("all", {}).get("abs_error", {}).get("max"),
               output_note=J.get("output", {}).get("note"), seconds=round(time.time() - t0, 1))
    if kind == "exact": res["exact_ok"] = bool((J.get("errors") or {}).get("bit_exact"))
    if kind in ("bounded", "lossless"):
        res["within_ok"] = g.get("all", {}).get("fraction_within_limit") == 1.0
    res["pass"] = bool(exists and not missing and not stat_missing and res["stats_match_ref"] and res["sizes_ok"] and res["kind_ok"]
                       and res["api_same"] and res["frames_ok"] and res.get("exact_ok", True) and res.get("within_ok", True))
    for f in os.listdir(tmp):
        if not f.startswith("in_"): os.remove(os.path.join(tmp, f))
    return res


def legacy_cases():
    from md_compress.codec import decompress, decompress_file
    x0 = np.load(f"{FIX}/legacy_1e8rA00_T20_input.npz")["coords"]; out = {}
    for m, lim in (("hybrid", 0.1), ("xtc", 0.005), ("zstd", 0.0)):
        blob = open(f"{FIX}/legacy_1e8rA00_T20_{m}.mdz", "rb").read()
        r, J = decompress(blob, return_info=True)
        mk = np.asarray(r["atom_mask"]) > 0
        err = float(np.abs(r["coords"][:, mk].astype(np.float64) - x0[:, mk]).max())
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "o.mdz"); open(p, "wb").write(blob); o = os.path.join(tmp, "o.dcd")
            rr = decompress_file(p, TOP, o); jex = os.path.exists(os.path.join(tmp, "o.json"))
        out[m] = dict(max_err=err, metadata_present=J["metadata_present"], errors=J["errors"], total_bytes=J["size"]["total_bytes"], json_written=jex,
                      pass_=bool(err <= lim + 1e-5 and J["metadata_present"] is False and J["errors"] is None and J["size"]["total_bytes"] == len(blob) and jex))
    return out


def flag_cases():
    """--report PATH, --no-report (CLI) and decompress(blob, report=True / False) (API)."""
    from md_compress.codec import main, decompress
    out = {}
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        ref = os.path.join(tmp, "in.dcd"); _write_input(ref, 20); mdz = os.path.join(tmp, "t.mdz")
        main(["compress", "--top", TOP, "--traj", ref, "--out", mdz, "--method", "zstd", "--bound", "0.1"])
        main(["decompress", "--in", mdz, "--top", TOP, "--out", os.path.join(tmp, "a.dcd")])
        out["cli default writes <out stem>.json"] = os.path.exists(os.path.join(tmp, "a.json"))
        main(["decompress", "--in", mdz, "--top", TOP, "--out", os.path.join(tmp, "b.dcd"), "--report", os.path.join(tmp, "custom.json")])
        out["cli --report PATH writes PATH only"] = os.path.exists(os.path.join(tmp, "custom.json")) and not os.path.exists(os.path.join(tmp, "b.json"))
        main(["decompress", "--in", mdz, "--top", TOP, "--out", os.path.join(tmp, "c.dcd"), "--no-report"])
        out["cli --no-report writes no JSON"] = os.path.exists(os.path.join(tmp, "c.dcd")) and not os.path.exists(os.path.join(tmp, "c.json"))
        main(["decompress", "--in", mdz, "--top", TOP, "--out", os.path.join(tmp, "d.dcd"), "--report"])
        out["cli bare --report writes <out stem>.json"] = os.path.exists(os.path.join(tmp, "d.json"))
        blob = open(mdz, "rb").read()
        r1 = decompress(blob); r0 = decompress(blob, report=False); r2, J = decompress(blob, return_info=True)
        out["api report=True default"] = "report" in r1 and r1["report"]["method"] == "zstd" and r1["report"]["errors"] is not None
        out["api report=False"] = "report" not in r0 and np.array_equal(r0["coords"], r1["coords"])
        out["api return_info matches file JSON"] = _close({k: v for k, v in json.load(open(os.path.join(tmp, "a.json"))).items()
                                                         if k not in ("decompress_seconds", "write_seconds", "output", "decompressed_from")},
                                                        {k: v for k, v in J.items() if k != "decompress_seconds"})
    return out


def run(cases):
    os.chdir(ROOT); R = {}
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        for c in cases:
            key = f"{c[0]} {json.dumps(c[2], sort_keys=True)} @ {c[1]} atoms={c[3]} T{c[5]} stride{c[7]} out{c[6]}"
            R[key] = report_case(*c, tmp); print(key, "PASS" if R[key]["pass"] else "FAIL", {k: R[key][k] for k in ("ratio", "metadata_bytes", "rmsd_lab", "max_abs", "seconds")}, flush=True)
            if not R[key]["pass"]: print("   ", R[key], flush=True)
    for k, v in flag_cases().items():
        R[f"flags: {k}"] = dict(pass_=bool(v)); print(f"flags: {k}", "PASS" if v else "FAIL", flush=True)
    for v in R.values():
        if "pass_" in v: v["pass"] = v.pop("pass_")
    for m, v in legacy_cases().items():
        R[f"legacy {m}"] = v; v["pass"] = v.pop("pass_"); print(f"legacy {m}", "PASS" if v["pass"] else "FAIL", v, flush=True)
    return R


def test_report_fast():
    R = run([c for c in CASES if c[0] not in SLOW]); assert all(v["pass"] for v in R.values()), {k: v for k, v in R.items() if not v["pass"]}


if __name__ == "__main__":
    R = run(CASES); json.dump(R, open(OUT, "w"), indent=1)
    print(f"{sum(v['pass'] for v in R.values())} of {len(R)} passed -> {OUT}")
