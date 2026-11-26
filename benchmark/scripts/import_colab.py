#!/usr/bin/env python
"""Turn the Colab outputs of compresstraj and cg2all into benchmark records with the shared metrics (bench_lib).
compresstraj: <colab_out>/ct_results.json + <sys>_<setting>/<sys>_decompressed.xtc.
cg2all: <colab_out_cg>/cg_results.json + rec_<sys>_<setting>.npy (protein heavy atoms; metrics on protein atoms only)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B
import MDAnalysis as mda
DATA = "/workspace/bench_data"; CT = "/workspace/bench_colab_out/out"; CG = "/workspace/bench_colab_out/out_cg"


def truth(s):
    f = f"{DATA}/{s}_T1000.npy" if os.path.exists(f"{DATA}/{s}_T1000.npy") else f"{DATA}/{s}.npy"
    return np.load(f)[:1000].astype(np.float64)


def sub_top(s, mask):
    lines = [l for l in open(f"{DATA}/{s}.pdb") if l.startswith(("ATOM", "HETATM"))]
    p = f"/workspace/bench_tmp/sub_{s}.pdb"; open(p, "w").write("".join(l for l, m in zip(lines, mask) if m)); return B.read_pdb_topology(p)


if os.path.exists(f"{CT}/ct_results.json"):
    R = B.Results(f"{B.BENCH}/results/compresstraj_raw.json")
    for key, r in json.load(open(f"{CT}/ct_results.json")).items():
        s, st = key.split("/")
        name = f"compresstraj {'c ' + st[1:] if st[0] == 'c' else 'latent ' + st[1:]}"
        if "error" in r:
            R.put(f"{s}/T1000/{name}", dict(system=s, T=1000, method=name, family="compresstraj", error=r["error"])); continue
        if not os.path.exists(f"{CT}/{s}_{st}/{s}_decompressed.xtc"): print("missing", key); continue
        x = truth(s); top = B.read_pdb_topology(f"{DATA}/{s}.pdb")
        u = mda.Universe(f"{DATA}/{s}.pdb", f"{CT}/{s}_{st}/{s}_decompressed.xtc")
        rec = np.stack([ts.positions.copy() for ts in u.trajectory]).astype(np.float64)
        mt = B.metrics(rec, x, top); raw = 12 * x.shape[1] * len(x)
        lat_b = r["bytes_compressed_pkl"] + r["bytes_cog"] + r["bytes_scaler"]
        common = dict(system=s, T=1000, family="compresstraj", raw_bytes=raw, encode_s=r["encode_s"], decode_s=r["decode_s"], metrics=mt,
                      latent=r["latent"], epochs=r["epochs"], n_params=r["n_params"], n_decoder_params=r["n_decoder_params"],
                      decoder_bytes=r["bytes_decoder_state"], hardware=r["gpu"])
        R.put(f"{s}/T1000/{name} (latents + COG + scaler)", dict(method=f"{name} (latents + COG + scaler)", bytes=lat_b, ratio=raw / lat_b, **common))
        R.put(f"{s}/T1000/{name} (+ decoder weights)", dict(method=f"{name} (+ decoder weights)", bytes=lat_b + r["bytes_decoder_state"],
                                                          ratio=raw / (lat_b + r["bytes_decoder_state"]), **common))
        print(key, f"{raw / lat_b:.1f}x rmsd {mt['rmsd_lab']:.3f}", flush=True)
if os.path.exists(f"{CG}/cg_results.json"):
    R = B.Results(f"{B.BENCH}/results/cg2all_raw.json")
    for key, r in json.load(open(f"{CG}/cg_results.json")).items():
        s, st = key.split("/"); cg, p = st.split("_p")
        name = f"cg2all {'CA trace' if cg == 'CA' else 'N, CA, C backbone'} stored as XTC precision {p}"
        if "error" in r:
            R.put(f"{s}/T1000/{name}", dict(system=s, T=1000, method=name, family="cg2all", error=r["error"])); continue
        if not os.path.exists(f"{CG}/rec_{s}_{st}.npy"): print("missing", key); continue
        x = truth(s); topf = B.read_pdb_topology(f"{DATA}/{s}.pdb"); pm = topf["prot"]
        rec = np.load(f"{CG}/rec_{s}_{st}.npy").astype(np.float64)
        ok = ~np.isnan(rec[0, :, 0])
        xs = x[:, pm]; mask = np.zeros(len(pm), bool); mask[np.where(pm)[0][ok]] = True
        top = sub_top(s, mask)
        mt = B.metrics(rec[:, ok], xs[:, ok], top); raw_p = 12 * int(ok.sum()) * len(x)
        R.put(f"{s}/T1000/{name}", dict(system=s, T=1000, method=name, family="cg2all", bytes=r["stored_bytes"], raw_bytes=raw_p, ratio=raw_p / r["stored_bytes"],
                                        encode_s=r["encode_s"], decode_s=r["decode_s"], metrics=mt, n_atoms_scored=int(ok.sum()),
                                        n_protein_atoms=int(pm.sum()), n_unmatched=int((~ok).sum()), hardware="Colab Tesla T4 GPU",
                                        note="protein heavy atoms only; ratio vs raw float32 of the scored atoms"))
        print(key, f"{raw_p / r['stored_bytes']:.1f}x rmsd {mt['rmsd_lab']:.3f} unmatched {(~ok).sum()}", flush=True)
