#!/usr/bin/env python
"""Download MDDB (MDposit) systems via the REST API, heavy atoms only, ALL frames (AEZ-12 constant temperature scaling; copy of fetch_mddb.py), then build the
benchmark files: <label>.pdb (atom order: protein atoms in atom14 order first, then every other heavy atom),
<label>.npy float32 [T,N,3] Angstrom, <label>.ours.npz (atom14 protein arrays + extra atoms) and mddb_meta.json.
Water and ions are excluded for all methods. Raw downloads are deleted after conversion."""
import json, os, subprocess, sys, urllib.request
import numpy as np
sys.path.insert(0, "/workspace/md_compress"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from md_compress import residue_constants as rc
import mdtraj as md
API = "https://mdposit.mddbr.eu/api/rest/v1/projects"
OUT = "/workspace/bench_data_mddb_long"; TMP = "/workspace/bench_tmp/mddb_long"; os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
SYSTEMS = [("mddb_small_1APQ_long", "MD-A00008", "small protein"),] or [ ("mddb_medium_12CA", "MD-A00001", "medium protein"),
           ("mddb_large_1AYX", "MD-A00009", "large protein"), ("mddb_dna_1d89", "MD-A00178", "DNA"),
           ("mddb_rna_2l8f", "MD-A0017P", "RNA"), ("mddb_membrane_Eprot", "MD-A001SU", "membrane + protein"),
           ("mddb_protdna_FOXA3", "MD-A001FH", "protein + DNA complex")]
WATER = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "TIP5", "SPC", "T3P", "OPC"}
IONS = {"NA", "CL", "K", "MG", "CA", "ZN", "SOD", "CLA", "POT", "Na+", "Cl-", "K+", "NA+", "CL-", "MN", "CAL", "CS", "LI", "RB"}
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
meta = json.load(open(f"{OUT}/mddb_meta.json")) if os.path.exists(f"{OUT}/mddb_meta.json") else {}
for label, acc, kind in SYSTEMS:
    if (only and label not in only) or (label in meta and os.path.exists(f"{OUT}/{label}.npy")): continue
    pm = json.load(urllib.request.urlopen(f"{API}/{acc}", timeout=120)); m = pm["metadata"]
    nf = int(m.get("frameCount") or pm.get("totalFrames") or 1000); step = max(1, nf // 100000)
    sel = "not hydrogen"; q = urllib.parse.quote(sel)
    spdb, sxtc = f"{TMP}/{label}.pdb", f"{TMP}/{label}.xtc"
    subprocess.run(["curl", "-sL", "-m", "600", "-o", spdb, f"{API}/{acc}/files/structure?selection={q}"], check=True)
    subprocess.run(["curl", "-sL", "-m", "3600", "-o", sxtc, f"{API}/{acc}/files/trajectory?format=xtc&frames=1:{nf}:{step}&selection={q}"], check=True)
    top = md.load_topology(spdb)
    X = md.formats.XTCTrajectoryFile(sxtc).read()[0] * 10.0  # Angstrom
    assert X.shape[1] == top.n_atoms, (label, X.shape, top.n_atoms)
    atoms = list(top.atoms)
    keep = [a for a in atoms if a.residue.name not in WATER and a.residue.name not in IONS and a.element.symbol != "H"]
    # protein residues mappable to atom14
    prot_res = []
    for r in top.residues:
        rn = rc.RESNAME_ALIASES.get(r.name.upper(), r.name.upper())
        if r.is_protein and rn in rc.RESTYPES: prot_res.append(r)
    order, aat, amask, chain, slotmap = [], [], [], [], []
    used = set()
    for r in prot_res:
        rn = rc.RESNAME_ALIASES.get(r.name.upper(), r.name.upper()); names = rc.atom14_names(rn)
        idx = {}
        for a in r.atoms:
            if a.element.symbol == "H": continue
            nm = rc.canonical_atom_name(r.name, a.name)
            if nm in names and nm: idx[names.index(nm)] = a.index
        if not all(s in idx for s in range(4)): continue  # no full backbone: treat as other
        aat.append(rc.RESTYPES.index(rn)); chain.append(r.chain.index)
        mk = np.zeros(14, np.float32)
        for s in sorted(idx): mk[s] = 1; order.append(idx[s]); used.add(idx[s])
        amask.append(mk); slotmap.append(idx)
    other = [a.index for a in keep if a.index not in used]
    R = len(aat)
    coords = np.zeros((len(X), R, 14, 3), np.float32)
    for i, idx in enumerate(slotmap):
        for s, ai in idx.items(): coords[:, i, s] = X[:, ai]
    cb = np.zeros(R, bool)
    for i in range(1, R):
        cb[i] = chain[i] != chain[i - 1] or np.linalg.norm(coords[0, i, 0] - coords[0, i - 1, 2]) > 2.0
    if R: cb[0] = True
    full = order + other
    xyz = X[:, full].astype(np.float32)
    np.save(f"{OUT}/{label}.npy", xyz)
    np.savez_compressed(f"{OUT}/{label}.ours.npz", coords=coords, aatype=np.array(aat, np.int64), atom_mask=np.array(amask, np.float32).reshape(R, 14),
                        chain_index=np.array(chain, np.int64), chain_break=cb, extra=X[:, other].astype(np.float32))
    # PDB topology in the benchmark order (names from the source; protein atoms renamed to the atom14 canonical names)
    lines = []; k = 0
    canon = {}
    for idx, r in zip(slotmap, prot_res if len(prot_res) == R else [top.atom(list(i.values())[0]).residue for i in slotmap]):
        pass
    resmap = {}
    for i, idx in enumerate(slotmap):
        rn = rc.RESTYPES[aat[i]]; names = rc.atom14_names(rn)
        for s, ai in idx.items(): canon[ai] = (names[s], rn)
    for ai in full:
        a = top.atom(ai); k += 1
        nm, rn = canon.get(ai, (a.name, a.residue.name))
        ch = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[a.residue.chain.index % 26]
        x, y, z = xyz[0, k - 1]
        nm4 = f" {nm:<3s}" if len(nm) < 4 else nm[:4]
        lines.append(f"ATOM  {k % 100000:5d} {nm4:4s} {rn[:4]:<4s}{ch}{(a.residue.resSeq % 10000):4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {a.element.symbol:>2s}")
    lines += ["END"]; open(f"{OUT}/{label}.pdb", "w").write("\n".join(lines) + "\n")
    grid = float(np.abs(np.round(xyz * 100) - xyz * 100).max())
    meta[label] = dict(accession=acc, kind=kind, name=m.get("NAME"), link=f"https://mdposit.mddbr.eu/#/id/{acc}/overview",
                       api=f"{API}/{acc}", source_frames=nf, stride=step, T=int(len(xyz)), n_heavy=int(xyz.shape[1]),
                       n_protein_atoms_model=int(len(order)), n_protein_residues_model=R, n_other_atoms=len(other),
                       excluded="hydrogens, water, ions", ff=m.get("FF"), program=m.get("PROGRAM"), temp=m.get("TEMP"),
                       max_dev_from_0p01A_grid=grid, resnames_other=sorted({top.atom(i).residue.name for i in other})[:40])
    json.dump(meta, open(f"{OUT}/mddb_meta.json", "w"), indent=1)
    os.remove(spdb); os.remove(sxtc)
    print(label, meta[label], flush=True)
