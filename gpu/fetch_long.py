#!/usr/bin/env python
"""Long per-domain trajectories for the frames-scaling study, from HuggingFace compsciencelab/mdCATH (HTTP range reads).

Archive order at the native 1 ns spacing: 320 K replicas 0,1,2,3, then 348 K replicas 0..4, 379 K 0..4, 413 K 0..4,
450 K 0..4, until --max_frames frames. Held-out (never archived): 320 K replica 4, frames 0..398 every 2nd (200 frames, 2 ns).
Output per domain: /workspace/md_compress_gpu_data/data_long/<dom>.npz (atom14 coords float32, temp, replica, frame index,
heldout flag). Incremental: finished domains are skipped.
"""
import json, os, random, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_{}.h5"
OUT = "/workspace/md_compress_gpu_data/data_long"
MAX = 10000
ORDER = [("320", r) for r in "0123"] + [(t, r) for t in ("348", "379", "413", "450") for r in "01234"]


def resolve(url):
    import requests
    for k in range(8):
        r = requests.head(url, allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 307, 308): return r.headers["Location"]
        if r.status_code == 200: return url
        time.sleep(min(60, 5 * 2 ** k) * (0.5 + random.random()))
    raise RuntimeError(f"resolve {r.status_code}")


def fetch(d):
    import fsspec, h5py
    from md_compress.data import load_trajectory
    path = f"{OUT}/{d}.npz"
    if os.path.exists(path): return d, "cached"
    t0 = time.time()
    with fsspec.open(resolve(URL.format(d)), block_size=8 * 2 ** 20).open() as fh:
        g = h5py.File(fh, "r")[d]
        pdb = g["pdbProteinAtoms"][()]; pdb = pdb.decode() if isinstance(pdb, bytes) else str(pdb)
        xs, meta, n = [], [], 0
        for t, r in ORDER:
            if n >= MAX: break
            if r not in g[t]: continue
            k = min(g[t][r]["coords"].shape[0], MAX - n)
            xs.append(g[t][r]["coords"][:k]); meta += [(int(t), int(r), i, 0) for i in range(k)]; n += k
        h = g["320"]["4"]["coords"][:400][::2]
        xs.append(h); meta += [(320, 4, i, 1) for i in range(0, 400, 2)][:len(h)]
    coords = np.concatenate(xs)
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(pdb); top = f.name
    dd = load_trajectory(top, coords=coords); os.unlink(top)
    m = np.array(meta, np.int32)
    np.savez(path + ".tmp.npz", coords=dd.coords.astype(np.float32), temp=m[:, 0], replica=m[:, 1], frame=m[:, 2], heldout=m[:, 3],
             atom_mask=dd.atom_mask, aatype=dd.aatype, chain_index=dd.chain_index, chain_break=dd.chain_break, n_heavy=dd.n_heavy_atoms())
    os.replace(path + ".tmp.npz", path)
    return d, f"{n} archive + {len(h)} held-out frames, {time.time()-t0:.0f}s"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    doms = json.load(open("results_gpu/frames/selected_domains.json"))
    with ProcessPoolExecutor(3) as ex:
        futs = [ex.submit(fetch, d) for d in doms]
        for f in futs:
            try: print(time.strftime("%H:%M:%S"), *f.result(), flush=True)
            except Exception as e: print("FAILED", repr(e)[:300], flush=True)
