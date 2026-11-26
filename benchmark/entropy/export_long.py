"""Frames scaling data: first 10,000 archived frames (same archive order as bench_data: 320 K replicas 0..3, then
348, 379, 413, 450 K) of the scaling domains, protein heavy atoms in atom14 order, float32 Angstrom -> /workspace/bench_data_long/<d>.npy
(the first 1,000 frames are identical to /workspace/bench_data/<d>_T1000.npy; checked)."""
import numpy as np, shutil, json
OUT = "/workspace/bench_data_long"; info = {}
for d in ["1e8rA00", "4m70B00", "2lg7A00"]:
    z = np.load(f"/workspace/md_compress_gpu_data/data_long/{d}.npz"); arch = np.where(z["heldout"] == 0)[0][:10000]
    X = z["coords"][arch][:, z["atom_mask"] > 0].astype(np.float32)
    ref = np.load(f"/workspace/bench_data/{d}_T1000.npy"); assert np.array_equal(X[:1000], ref), d
    np.save(f"{OUT}/{d}.npy", X); shutil.copy(f"/workspace/bench_data/{d}.pdb", f"{OUT}/{d}.pdb")
    t, r = z["temp"][arch], z["replica"][arch]
    seg = [(int(t[i]), int(r[i]), int(i)) for i in range(len(t)) if i == 0 or (t[i], r[i]) != (t[i - 1], r[i - 1])]
    info[d] = dict(T=len(X), segments_start=seg); print(d, X.shape)
json.dump(info, open(f"{OUT}/meta.json", "w"), indent=1)
