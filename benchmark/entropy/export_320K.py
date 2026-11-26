"""Same temperature frames scaling data (AEZ-12 follow up): 2lg7A00, the scaling domain with the most 320 K frames (5 replicas x 500 = 2,500).
Writes /workspace/bench_data_320K/<name>.ours.npz (bench_entropy) and <name>.npy + .pdb (bench_formats):
  2lg7A00_320K : 320 K replicas 0,1,2,3,4 concatenated (2,500 frames, replica boundaries at 500, 1000, 1500, 2000)
  boundary test at T = 500, all 320 K:
  2lg7A00_b0   : replica 0 frames 0..499 in order (no discontinuity)
  2lg7A00_p4   : the same 500 frames as 5 blocks of 100 in the order 0,2,4,1,3 (4 jumps of >= 100 ns, identical frame set)
  2lg7A00_p9   : the same 500 frames as 10 blocks of 50 in the order 0,5,1,6,2,7,3,8,4,9 (9 jumps)
  2lg7A00_r4   : frames 0..99 of replicas 0..4 (4 replica boundaries, different content)
The first 1,000 frames of 2lg7A00_320K are checked against data_long (320 K replicas 0 and 1)."""
import os, sys, json, shutil, tempfile, numpy as np, h5py
sys.path.insert(0, "/workspace/md_compress")
from md_compress.data import load_trajectory
d = "2lg7A00"; OUT = "/workspace/bench_data_320K"
g = h5py.File(f"/workspace/mdcath_h5/mdcath_dataset_{d}.h5", "r")[d]
pdb = g["pdbProteinAtoms"][()]; pdb = pdb.decode() if isinstance(pdb, bytes) else str(pdb)
reps = [g["320"][str(r)]["coords"][:] for r in range(5)]; lens = [len(x) for x in reps]
with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f: f.write(pdb); top = f.name
dd = load_trajectory(top, coords=np.concatenate(reps)); os.unlink(top)
C = dd.coords.astype(np.float32); mk = dd.atom_mask > 0
z = np.load(f"/workspace/md_compress_gpu_data/data_long/{d}.npz"); arch = np.where(z["heldout"] == 0)[0][:1000]
assert np.array_equal(C[:1000], z["coords"][arch]), "mismatch with data_long"
assert np.array_equal(dd.atom_mask, z["atom_mask"])
r0 = np.arange(500); blk = lambda n, order: np.concatenate([r0[i * n:(i + 1) * n] for i in order])
sets = {f"{d}_320K": np.arange(len(C)), f"{d}_b0": r0, f"{d}_p4": blk(100, [0, 2, 4, 1, 3]),
        f"{d}_p9": blk(50, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]), f"{d}_r4": np.concatenate([np.arange(100) + 500 * r for r in range(5)])}
meta = dict(domain=d, temperature_K=320, replica_lengths=lens)
for name, idx in sets.items():
    X = C[idx]
    np.savez(f"{OUT}/{name}.ours.npz", coords=X, extra=np.zeros((0,), np.float32), aatype=z["aatype"], atom_mask=z["atom_mask"],
             chain_index=z["chain_index"], chain_break=z["chain_break"])
    np.save(f"{OUT}/{name}.npy", X[:, mk]); shutil.copy(f"/workspace/bench_data_long/{d}.pdb", f"{OUT}/{name}.pdb")
    meta[name] = dict(T=len(idx), source_frames=[int(i) for i in idx[np.r_[0, np.where(np.diff(idx) != 1)[0] + 1]]])
    print(name, X.shape)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
