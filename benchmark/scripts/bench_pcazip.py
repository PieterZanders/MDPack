#!/usr/bin/env python
"""pyPcazip (Laughton lab, pypcazip 3.0.0 from PyPI; mdplus.pca.pcazipsave / pcazipload, the functions behind the
pyPcazip and pyPcaunzip command line tools) on the benchmark systems.
Settings: PCA only (residual_scale 0, the classic lossy pcazip) at explained variance 0.70, 0.80, 0.90, 0.95, 0.99;
and V3 with quantised residuals (residual_scale 200 per nm = 0.05 A steps, the tool default) at explained variance 0.75
and 0.95, plus residual_scale 20 and 50 (0.5 and 0.2 A steps) at 0.95.
pcazip superposes every frame on the average structure and does not store the global rotation/translation, so its
output is in the fitted frame. For the metrics, each decoded frame is superposed back onto the original frame (this
favours pcazip slightly: the rotation and translation needed for that would cost 28 B per frame and are not counted)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B
from mdplus.pca import pcazipsave, pcazipload
ap = argparse.ArgumentParser(); ap.add_argument("--systems", required=True); ap.add_argument("--T", default="1000,250")
ap.add_argument("--out", default=f"{B.BENCH}/results/pcazip_raw.json"); ap.add_argument("--data", default="/workspace/bench_data")
a = ap.parse_args(); RES = B.Results(a.out); TMP = "/workspace/bench_tmp/pcz"; os.makedirs(TMP, exist_ok=True)
SET = [(ev, 0) for ev in (0.70, 0.80, 0.90, 0.95, 0.99)] + [(0.75, 200), (0.95, 200), (0.95, 50), (0.95, 20)]
for s in a.systems.split(","):
    f = f"{a.data}/{s}_T1000.npy" if os.path.exists(f"{a.data}/{s}_T1000.npy") else f"{a.data}/{s}.npy"
    X = np.load(f); top = B.read_pdb_topology(f"{a.data}/{s}.pdb")
    for T in [int(t) for t in a.T.split(",")]:
        if T > len(X): continue
        x = np.ascontiguousarray(X[:T]); raw = 12 * x.shape[1] * T
        for ev, rs in SET:
            name = f"pcazip EV {ev:.2f}" + (" PCA only" if rs == 0 else f" + residuals (scale {rs}/nm)")
            key = f"{s}/T{T}/{name}"
            if RES.has(key): continue
            p = f"{TMP}/{s}_{T}.pcz"
            try:
                with B.Timer() as te: pcazipsave(x / 10.0, p, explained_variance=ev, residual_scale=rs, eigenvector_scale=100)
                size = os.path.getsize(p)
                with B.Timer() as td: rec = pcazipload(p) * 10.0
                rec = rec.astype(np.float64)
                rec_fit = B.superpose(torch.as_tensor(rec), torch.as_tensor(x, dtype=torch.float64)).numpy()
                mt = B.metrics(rec_fit, x, top)
                n_vecs = int(np.frombuffer(open(p, "rb").read(64), dtype=">i4")[4]) if False else None
                RES.put(key, dict(system=s, T=T, method=name, family="pcazip", bytes=size, raw_bytes=raw, ratio=raw / size,
                                  encode_s=te.s, decode_s=td.s, metrics=mt, explained_variance=ev, residual_scale=rs,
                                  note="metrics after per frame superposition (global frame not stored)"))
                print(f"{key}: {raw / size:.2f}x rmsd_fit {mt['rmsd_fit']:.3f} max {mt['max_abs_err']:.3f} enc {te.s:.1f}s dec {td.s:.1f}s", flush=True)
            except Exception as e:
                RES.put(key, dict(system=s, T=T, method=name, family="pcazip", error=repr(e)[-1500:])); print("FAIL", key, repr(e)[-300:], flush=True)
            if os.path.exists(p): os.remove(p)
print("done", flush=True)
