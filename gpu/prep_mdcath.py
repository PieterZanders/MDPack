#!/usr/bin/env python
"""Download a superfamily-diverse subset of mdCATH directly on the GPU VM and convert it to atom14.

Source: HuggingFace compsciencelab/mdCATH (one HDF5 file per CATH domain, CHARMM22* / TIP3P,
5 temperatures x 5 replicas, 1 frame per ns). Only the needed datasets are read with HTTP range
requests (h5py over fsspec), so a 1 GB domain file costs ~10-20 MB of traffic.

Selection: domains of 50..250 residues with a known CATH superfamily (gpu/mdcath_cath.json, built from
cath-domain-list.txt), ONE domain per superfamily, so the train/val/test split is superfamily-disjoint
by construction. Domains whose PDB id matches an external test protein are excluded.
Kept: 320 K, replicas 0 and 1, every 2nd frame (2 ns), heavy atoms only.

Incremental: every domain is written to OUT/domains/<id>.npz as soon as it is done; reruns skip it.
"""
import argparse, io, json, os, random, sys, tempfile, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from md_compress.data import load_trajectory  # noqa: E402

URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_{}.h5"
EXTERNAL_PDB = {"4ake", "1ake", "1pch", "3cjs", "2ygs", "3czz", "3fp5", "2fip", "3klr"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/data/mdcath")
ap.add_argument("--n_train", type=int, default=300)
ap.add_argument("--n_val", type=int, default=30)
ap.add_argument("--n_test", type=int, default=50)
ap.add_argument("--temp", default="320")
ap.add_argument("--replicas", default="0,1")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--min_len", type=int, default=50)
ap.add_argument("--max_len", type=int, default=250)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(f"{a.out}/domains", exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
cath = json.load(open(f"{here}/mdcath_cath.json"))

# ---------------------------------------------------------------- superfamily-disjoint selection
by_sf = {}
for d, v in sorted(cath.items()):
    if v and a.min_len <= v["length"] <= a.max_len and d[:4].lower() not in EXTERNAL_PDB:
        by_sf.setdefault(v["sf"], []).append(d)
rng = random.Random(a.seed)
sfs = sorted(by_sf)
rng.shuffle(sfs)
cands = [(sf, rng.choice(by_sf[sf])) for sf in sfs]
need = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
# candidate pools per split (with 25% spare for download / parsing failures)
pools, p = {}, 0
for split in ("test", "val", "train"):
    k = int(need[split] * 1.25) + 3
    pools[split] = cands[p:p + k]
    p += k

MAN = f"{a.out}/manifest.json"
man = json.load(open(MAN)) if os.path.exists(MAN) else {"domains": {}, "failed": {}}
# forget transient failures (rate limits, dropped connections) so they are retried
man["failed"] = {d: e for d, e in man["failed"].items()
                 if not any(t in e for t in ("FileNotFoundError", "pickle", "ClientPayloadError", "resolve failed", "Timeout"))}


def save_manifest():
    json.dump(man, open(MAN + ".tmp", "w"), indent=1)
    os.replace(MAN + ".tmp", MAN)


def resolve(url, tries=8):
    """HF 'resolve' URL -> signed CDN URL (range requests then go straight to the CDN, which avoids the
    Hub's per-request rate limit). Retries with backoff on 429/5xx."""
    import requests
    for k in range(tries):
        r = requests.head(url, allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 307, 308) and "Location" in r.headers:
            return r.headers["Location"]
        if r.status_code == 200:
            return url
        time.sleep(min(60, 5 * 2 ** k) * (0.5 + random.random()))
    raise RuntimeError(f"resolve failed: HTTP {r.status_code}")


def fetch(dom):
    try:
        return _fetch(dom)
    except Exception as e:  # aiohttp exceptions are not picklable across processes
        raise RuntimeError(repr(e)[:300]) from None


def _fetch(dom):
    import fsspec, h5py
    path = f"{a.out}/domains/{dom}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return dict(n_res=int(z["aatype"].shape[0]), n_frames=int(z["coords"].shape[0]), n_heavy=int(z["n_heavy"]), cached=True)
    t0 = time.time()
    with fsspec.open(resolve(URL.format(dom)), block_size=4 * 2 ** 20).open() as fh:
        h = h5py.File(fh, "r")
        g = h[dom]
        pdb = g["pdbProteinAtoms"][()]
        pdb = pdb.decode() if isinstance(pdb, bytes) else str(pdb)
        xs, reps = [], []
        for r in a.replicas.split(","):
            if r in g[a.temp]:
                c = g[a.temp][r]["coords"][:][::a.stride]
                xs.append(c)
                reps.append(np.full(len(c), int(r), np.int16))
    coords = np.concatenate(xs)
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(pdb)
        top = f.name
    d = load_trajectory(top, coords=coords)
    os.unlink(top)
    if d.n_res < a.min_len or (d.atom_mask[:, 1] == 0).any():
        raise ValueError(f"bad topology ({d.n_res} res)")
    seq = "".join("ARNDCQEGHILKMFPSTWYVX"[min(i, 20)] for i in d.aatype)
    np.savez(path + ".tmp.npz", coords=d.coords.astype(np.float32), replica=np.concatenate(reps),
             atom_mask=d.atom_mask, aatype=d.aatype, chain_index=d.chain_index, chain_break=d.chain_break,
             n_heavy=d.n_heavy_atoms(), seq=seq, pdb=pdb)
    os.replace(path + ".tmp.npz", path)
    return dict(n_res=d.n_res, n_frames=d.n_frames, n_heavy=d.n_heavy_atoms(), seconds=round(time.time() - t0, 1))


t_start = time.time()
for split in ("test", "val", "train"):
    have = [d for d, v in man["domains"].items() if v["split"] == split]
    todo = [(sf, d) for sf, d in pools[split] if d not in man["domains"] and d not in man["failed"]]
    with ProcessPoolExecutor(a.workers) as ex:  # processes: h5py holds a global lock
        futs = {}
        it = iter(todo)
        def submit_next():
            try:
                sf, d = next(it)
            except StopIteration:
                return
            futs[ex.submit(fetch, d)] = (sf, d)
        for _ in range(a.workers):
            submit_next()
        while futs:
            for fut in as_completed(list(futs)):
                sf, d = futs.pop(fut)
                try:
                    info = fut.result()
                    if len(have) < need[split]:
                        man["domains"][d] = dict(split=split, sf=sf, **info)
                        have.append(d)
                except Exception as e:
                    man["failed"][d] = repr(e)[:200]
                save_manifest()
                if len(have) < need[split]:
                    submit_next()
                break
    print(f"{time.strftime('%H:%M:%S')} {split}: {len(have)} domains ({time.time()-t_start:.0f}s)", flush=True)

# ---------------------------------------------------------------- pack training frames into one memmap
doms = sorted(d for d, v in man["domains"].items() if v["split"] == "train")
pk = f"{a.out}/train_pack"
if not os.path.exists(pk + "/index.json") or sorted(json.load(open(pk + "/index.json"))) != doms:
    os.makedirs(pk, exist_ok=True)
    tot = [0]
    shapes = {}
    for d in doms:
        z = np.load(f"{a.out}/domains/{d}.npz")
        shapes[d] = z["coords"].shape[:2]
    n = sum(T * R for T, R in shapes.values())
    mm = np.lib.format.open_memmap(pk + "/coords.npy", "w+", np.float32, (n, 14, 3))
    index, off = {}, 0
    for d in doms:
        T, R = shapes[d]
        mm[off:off + T * R] = np.load(f"{a.out}/domains/{d}.npz")["coords"].reshape(T * R, 14, 3)
        index[d] = dict(off=off, T=T, R=R)
        off += T * R
    mm.flush()
    json.dump(index, open(pk + "/index.json", "w"))
tr = [man["domains"][d] for d in doms]
print(f"packed {len(doms)} train domains, {sum(v['n_frames'] for v in tr)} frames, "
      f"{sum(v['n_frames'] * v['n_res'] for v in tr) * 14 * 3 * 4 / 1e9:.2f} GB; total {time.time()-t_start:.0f}s", flush=True)
