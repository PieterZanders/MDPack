#!/usr/bin/env python
"""cg2all as a compression scheme (run on a Colab GPU). Stored data: the CA trace (CalphaBasedModel) or the N, CA, C
backbone (BackboneModel) of every protein residue, written as XTC at precision 1000 (0.01 A) or 100 (0.1 A); the
size of that XTC is the stored size (cg2all weights are shared, like ours, and not counted). Decode: read the XTC and
run convert_cg2all (official CLI) on the GPU; then map the all atom output (heavy atoms) back to the benchmark atom
order by residue index and atom name. Saves rec_<sys>_<setting>.npy (float32, protein heavy atoms of the benchmark
order, NaN where cg2all did not output a matching atom) and cg_results.json."""
import json, os, subprocess, sys, time
import numpy as np, mdtraj as md
import MDAnalysis as mda
D = "/content/bench/data"; O = "/content/bench/out_cg"; os.makedirs(O, exist_ok=True)
systems = sys.argv[1].split(","); settings = sys.argv[2].split(",")  # CA_p1000, CA_p100, BB_p1000
AA = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
ALIAS = {("ILE", "CD"): "CD1", ("ANY", "OT1"): "O", ("ANY", "OT2"): "OXT", ("ANY", "O1"): "O", ("ANY", "O2"): "OXT"}
res = json.load(open(f"{O}/cg_results.json")) if os.path.exists(f"{O}/cg_results.json") else {}
for s in systems:
    # data packed for transfer as int32 on the 0.01 A grid (the source grid; differences < 1e-4 A)
    _f = f"{D}/{s}_T1000.npz" if os.path.exists(f"{D}/{s}_T1000.npz") else f"{D}/{s}.npz"
    x = (np.load(_f)["q"] * 0.01).astype(np.float32)
    x = x[:1000]; top = md.load_topology(f"{D}/{s}.pdb")
    prot = [a.index for a in top.atoms if a.residue.name in AA]
    ptop = top.subset(prot)
    for st in settings:
        key = f"{s}/{st}"
        if key in res and "error" not in res[key]: continue
        cg, prec = st.split("_p"); prec = int(prec)
        names = ["CA"] if cg == "CA" else ["N", "CA", "C"]
        sel = [a.index for a in ptop.atoms if a.name in names]
        sub = ptop.subset(sel); xs = x[:, prot][:, sel]
        cgpdb = f"{O}/{s}_{st}_cg.pdb"; md.Trajectory(xs[:1] / 10, sub).save_pdb(cgpdb)
        stored = f"{O}/{s}_{st}.xtc"
        u = mda.Universe(cgpdb)
        t0 = time.time()
        with mda.coordinates.XTC.XTCWriter(stored, u.atoms.n_atoms, precision=int(round(np.log10(prec)))) as w:
            for f in xs: u.atoms.positions = f; w.write(u.atoms)
        te = time.time() - t0; nbytes = os.path.getsize(stored)
        # decode
        t0 = time.time()
        dec = md.load(stored, top=cgpdb); dcd = f"{O}/{s}_{st}_cg.dcd"; dec.save_dcd(dcd)
        model = "CalphaBasedModel" if cg == "CA" else "BackboneModel"
        r = subprocess.run(["convert_cg2all", "-p", cgpdb, "-d", dcd, "-o", f"{O}/{s}_{st}_aa.dcd", "-opdb", f"{O}/{s}_{st}_aa.pdb",
                            "--cg", model, "--device", "cuda", "--batch", "8"], capture_output=True, text=True)
        td = time.time() - t0
        if r.returncode:
            res[key] = dict(error=r.stderr[-3000:]); json.dump(res, open(f"{O}/cg_results.json", "w"), indent=1); print("FAIL", key, r.stderr[-1000:], flush=True); continue
        aa = md.load(f"{O}/{s}_{st}_aa.dcd", top=f"{O}/{s}_{st}_aa.pdb")
        # map: residue order index + atom name
        lut = {}
        for a in aa.topology.atoms:
            if a.element is not None and a.element.symbol == "H": continue
            nm = ALIAS.get((a.residue.name, a.name), ALIAS.get(("ANY", a.name), a.name))
            lut[(a.residue.index, nm)] = a.index
        rec = np.full((len(x), len(prot), 3), np.nan, np.float32); miss = 0
        for j, a in enumerate(ptop.atoms):
            k = lut.get((a.residue.index, a.name))
            if k is None: miss += 1; continue
            rec[:, j] = aa.xyz[:, k] * 10
        np.save(f"{O}/rec_{s}_{st}.npy", rec)
        res[key] = dict(system=s, setting=st, cg_model=model, xtc_precision=prec, stored_bytes=nbytes, n_cg_atoms=len(sel),
                        n_protein_atoms=len(prot), n_unmatched=miss, encode_s=te, decode_s=td, n_frames_out=int(aa.n_frames),
                        stderr_tail=r.stderr[-300:])
        for f in (dcd, f"{O}/{s}_{st}_aa.dcd"): os.remove(f)
        json.dump(res, open(f"{O}/cg_results.json", "w"), indent=1)
        print(time.strftime("%H:%M:%S"), key, f"{nbytes} B dec {td:.0f}s miss {miss}", flush=True)
print("done", flush=True)
