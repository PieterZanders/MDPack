#!/usr/bin/env python
"""Standard trajectory formats and MDCompress on the benchmark systems.
Every method gets the same in-memory float32 heavy-atom array x [T,N,3] (Angstrom) and is timed for writing (encode)
and reading back (decode) on this box's CPU, single process. External binaries (GROMACS trjconv for TNG, MDCompress)
read a TRR (float32, exact) written beforehand; their timings include that TRR read and process start.
Usage: python bench_formats.py --systems a,b --T 1000,250 [--out results/formats_raw.json]"""
import argparse, gzip, os, shutil, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B
import warnings; warnings.filterwarnings("ignore")
import mdtraj as md, MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

ap = argparse.ArgumentParser()
ap.add_argument("--systems", required=True); ap.add_argument("--T", default="1000,250")
ap.add_argument("--data", default="/workspace/bench_data"); ap.add_argument("--out", default=f"{B.BENCH}/results/formats_raw.json")
ap.add_argument("--only", default="")
a = ap.parse_args()
GMX = "/workspace/conda/envs/gmx/bin/gmx"; MDC = "/workspace/bench_tools/mdcbin/mdcompress"
TMP = "/workspace/bench_tmp/fmt"; os.makedirs(TMP, exist_ok=True)
RES = B.Results(a.out)


def universe(pdb, x):
    u = mda.Universe(pdb); u.load_new(x.astype(np.float32), format=MemoryReader, order="fac"); return u


def read_mda(pdb, f):
    u = mda.Universe(pdb, f); return np.stack([ts.positions.copy() for ts in u.trajectory])


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(cmd + "\n" + r.stderr[-2000:] + r.stdout[-1000:])
    return r


def methods(pdb, x, top, base):
    T = len(x); tt = np.arange(T, dtype=np.float32); st = np.arange(T, dtype=np.int32)
    trr = f"{base}.src.trr"
    with md.formats.TRRTrajectoryFile(trr, "w") as f: f.write(x / 10, time=tt, step=st)
    desc = f"{base}.desc"; run(f"{MDC} make_desc -i {pdb} -o {desc}")
    M = {}

    def mdtraj_fmt(ext, cls, kw={}, rd_scale=10.0):
        p = f"{base}.{ext}"
        def enc():
            with cls(p, "w") as f: f.write(x / (10 if rd_scale == 10 else 1), **kw)
        def dec():
            with cls(p) as f: r = f.read()
            return (r[0] if isinstance(r, tuple) else r) * rd_scale
        return enc, dec, [p]
    M["TRR (float32)"] = mdtraj_fmt("trr", md.formats.TRRTrajectoryFile, dict(time=tt, step=st))
    M["DCD (float32)"] = mdtraj_fmt("dcd", md.formats.DCDTrajectoryFile, {}, 1.0)
    M["NetCDF (AMBER, float32)"] = mdtraj_fmt("nc", md.formats.NetCDFTrajectoryFile, {}, 1.0)
    for prec in (3, 2, 1):
        p = f"{base}.p{prec}.xtc"
        def enc(p=p, prec=prec):
            u = universe(pdb, x)
            with mda.coordinates.XTC.XTCWriter(p, u.atoms.n_atoms, precision=prec) as w:
                for ts in u.trajectory: w.write(u.atoms)
        M[f"XTC precision {10 ** prec}"] = (enc, lambda p=p: read_mda(pdb, p), [p])
    p = f"{base}.tng"
    M["TNG (GROMACS trjconv)"] = (lambda p=p: run(f"echo 0 | {GMX} -quiet trjconv -f {trr} -s {pdb} -o {p}"),
                                  lambda p=p: (run(f"echo 0 | {GMX} -quiet trjconv -f {p} -s {pdb} -o {p}.trr"), md.formats.TRRTrajectoryFile(f"{p}.trr").read()[0] * 10)[1], [p])
    p = f"{base}.pdb"
    def enc_pdb(p=p):
        t = md.Trajectory(x / 10, md.load_topology(pdb)); t.save_pdb(p)
    big = x.shape[0] * x.shape[1] > 5_000_000  # mdtraj's PDB reader needs > 10 GB here; stream with MDAnalysis instead
    if big:
        def enc_pdb(p=p):
            u = universe(pdb, x)
            with mda.Writer(p, u.atoms.n_atoms, multiframe=True) as w:
                for ts in u.trajectory: w.write(u.atoms)
        dec_pdb = lambda p=p: read_mda(pdb, p)
    else:
        dec_pdb = lambda p=p: md.load_pdb(p).xyz * 10
    M["PDB (text)"] = (enc_pdb, dec_pdb, [p])
    def enc_pdbgz(p=p):
        enc_pdb(p)
        with open(p, "rb") as fi, gzip.open(p + ".gz", "wb", 9) as fo: shutil.copyfileobj(fi, fo)
    def dec_pdbgz(p=p):
        with gzip.open(p + ".gz", "rb") as fi, open(p + ".d.pdb", "wb") as fo: shutil.copyfileobj(fi, fo)
        return dec_pdb(p + ".d.pdb")
    M["PDB + gzip -9"] = (enc_pdbgz, dec_pdbgz, [p + ".gz"])
    for name, kw in (("H5MD (uncompressed)", {}), ("H5MD (gzip 4, chunk 1 frame)", dict(compression="gzip", compression_opts=4)),
                     ("H5MD (gzip 9, chunk 100 frames)", dict(compression="gzip", compression_opts=9, chunks=(min(100, T), len(top["names"]), 3)))):
        p = f"{base}.{len(M)}.h5md"
        def enc(p=p, kw=kw):
            u = universe(pdb, x)
            with mda.Writer(p, u.atoms.n_atoms, velocities=False, forces=False, n_frames=T, convert_units=False, lengthunit="Angstrom", timeunit="ps", **kw) as w:
                for ts in u.trajectory: w.write(u.atoms)
        M[name] = (enc, lambda p=p: read_mda(pdb, p), [p])
    for lev, res in [(l_, r_) for r_ in (1000, 20000, 40000, 60000, 100000) for l_ in (2, 4)]:
        p = f"{base}.l{lev}r{res}.mdc"
        name = f"MDCompress -l {lev} --res {res} fm" + (" (default)" if (lev, res) == (2, 1000) else "")
        M[name] = (lambda p=p, lev=lev, res=res: run(f"{MDC} compress -i {trr} -d {desc} -o {p} -l {lev} --res {res} -t 1"),
                   lambda p=p: (run(f"{MDC} decompress -i {p} -o {p}.trr -t 1"), md.formats.TRRTrajectoryFile(f"{p}.trr").read()[0] * 10)[1], [p])
    # strongest settings we found in the README: level 4 with the archive preset (history -h 2 or 3 segfaults in the v2.0.5 decoder, see report)
    for res in (1000, 20000, 100000):
        p = f"{base}.arch{res}.mdc"
        M[f"MDCompress -l 4 --preset archive --res {res} fm"] = (
            lambda p=p, res=res: run(f"{MDC} compress -i {trr} -d {desc} -o {p} -l 4 --preset archive --res {res} -t 1"),
            lambda p=p: (run(f"{MDC} decompress -i {p} -o {p}.trr -t 1"), md.formats.TRRTrajectoryFile(f"{p}.trr").read()[0] * 10)[1], [p])
    return M


for sysname in a.systems.split(","):
    pdb = f"{a.data}/{sysname}.pdb"; X = np.load(f"{a.data}/{sysname}_T1000.npy") if os.path.exists(f"{a.data}/{sysname}_T1000.npy") else np.load(f"{a.data}/{sysname}.npy")
    top = B.read_pdb_topology(pdb)
    for T in [int(t) for t in a.T.split(",")]:
        if T > len(X): continue
        x = np.ascontiguousarray(X[:T]); base = f"{TMP}/{sysname}_T{T}"
        raw = 12 * x.shape[1] * T
        for name, (enc, dec, files) in methods(pdb, x, top, base).items():
            key = f"{sysname}/T{T}/{name}"
            if RES.has(key) or (a.only and not any(o in name for o in a.only.split("|"))): continue
            try:
                with B.Timer() as te: enc()
                size = B.fsize(*files)
                with B.Timer() as td: rec = dec()
                rec = np.asarray(rec, np.float64).reshape(x.shape)
                mt = B.metrics(rec, x, top)
                RES.put(key, dict(system=sysname, T=T, method=name, family="format", bytes=size, raw_bytes=raw, ratio=raw / size,
                                  encode_s=te.s, decode_s=td.s, metrics=mt))
                print(f"{key}: {raw / size:.2f}x rmsd {mt['rmsd_lab']:.4f} max {mt['max_abs_err']:.4f} enc {te.s:.2f}s dec {td.s:.2f}s", flush=True)
            except Exception as e:
                RES.put(key, dict(system=sysname, T=T, method=name, family="format", error=str(e)[-1500:]))
                print("FAILED", key, str(e)[-500:], flush=True)
        for f in os.listdir(TMP):
            if f.startswith(f"{sysname}_T{T}"): os.remove(f"{TMP}/{f}")
print("done", flush=True)
