#!/usr/bin/env python
"""compresstraj (github.com/SerpentByte/compresstraj, JCIM 2025) on a Colab GPU, per benchmark system.
Uses the package's own classes (DenseAutoEncoder with layers 4096,1024, RMSDLoss, LightAutoEncoder with AdamW
lr 3e-4, global StandardScaler statistics, set_seed) and the same steps and defaults as scripts/compress.py and
scripts/decompress.py: centre of geometry removed per frame and stored (float64 .npy), latent = ceil(3N / c) for
compression factor c (default c = 20) or a fixed latent size, batch 128, shuffled, precision 32, max_epochs EPOCHS,
latents pickled per batch of 128 frames, decoding = decoder -> inverse scaling -> recentre -> add stored COG -> XTC
(MDAnalysis XTC writer, as decompress.py). The only change: frames are held in memory instead of being read from
the trajectory file by MDAnalysis at every step (the original per item file reads made one epoch take ~6 s on the
2 core Colab VM); the values fed to the network are identical."""
import json, math, os, pickle, sys, time
import numpy as np, torch
import MDAnalysis as mda
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, "/content/bench/compresstraj")
from compresstraj.classes import DenseAutoEncoder, RMSDLoss, LightAutoEncoder
from compresstraj.helpers import set_seed
torch.set_float32_matmul_precision("medium")
D = "/content/bench/data"; O = "/content/bench/out"; os.makedirs(O, exist_ok=True)
systems = sys.argv[1].split(","); settings = sys.argv[2].split(",")
EPOCHS = int(os.environ.get("EPOCHS", "2000")); BATCH = 128
res = json.load(open(f"{O}/ct_results.json")) if os.path.exists(f"{O}/ct_results.json") else {}


class Frames(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


for s in systems:
    _f = f"{D}/{s}_T1000.npz" if os.path.exists(f"{D}/{s}_T1000.npz") else f"{D}/{s}.npz"
    x = (np.load(_f)["q"] * 0.01).astype(np.float32)[:1000]; pdb = f"{D}/{s}.pdb"
    T, Na, _ = x.shape
    for st in settings:
        key = f"{s}/{st}"
        if key in res and "error" not in res[key]: continue
        # GPU time budget: the fixed latent sweep (l16) runs on four domains spanning the size range only
        if st[0] == "l" and s not in ("1e8rA00", "3mzkB03", "1jyoE00", "1b5eA00"): continue
        od = f"{O}/{s}_{st}"; os.makedirs(od, exist_ok=True)
        try:
            set_seed()
            t0 = time.time()
            cog = x.mean(1)  # centre of geometry per frame (float32 positions, as MDAnalysis)
            xc = x - cog[:, None]
            mean = float(xc.astype(np.float64).sum() / xc.size); sq = float((xc.astype(np.float64) ** 2).sum() / xc.size)
            std = math.sqrt(max(sq - mean ** 2, 1e-16))
            Xs = ((xc - mean) / std).reshape(T, -1).astype(np.float32)
            N = Xs.shape[1]
            latent = int(np.ceil(N / int(st[1:]))) if st[0] == "c" else int(st[1:]); latent = max(latent, 2)
            ae = DenseAutoEncoder(N=N, latent=latent, layers=[4096, 1024])
            model = LightAutoEncoder(model=ae, loss_fn=RMSDLoss(), learning_rate=3e-4)
            trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu", devices=[0], precision="32-true", enable_checkpointing=False,
                                 logger=False, enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model, DataLoader(Frames(Xs), batch_size=BATCH, shuffle=True))
            enc = model.model.encoder.cuda().eval(); lat = []
            with torch.no_grad():
                for b in DataLoader(Frames(Xs), batch_size=BATCH, shuffle=False):
                    lat.append(enc(b.cuda().float()).cpu())
            pickle.dump(lat, open(f"{od}/{s}_compressed.pkl", "wb")); np.save(f"{od}/{s}_cog.npy", cog.astype(np.float64))
            pickle.dump({"mean": mean, "std": std}, open(f"{od}/{s}_scaler.pkl", "wb"))
            torch.save(model.model.decoder.state_dict(), f"{od}/decoder_state.pt")
            te = time.time() - t0
            # decode exactly as decompress.py
            t0 = time.time()
            dec = model.model.decoder.cuda().eval(); lat2 = pickle.load(open(f"{od}/{s}_compressed.pkl", "rb"))
            cg = np.load(f"{od}/{s}_cog.npy"); sc = pickle.load(open(f"{od}/{s}_scaler.pkl", "rb"))
            u = mda.Universe(pdb)
            with mda.Writer(f"{od}/{s}_decompressed.xtc", u.atoms.n_atoms) as w, torch.no_grad():
                for i, l in enumerate(lat2):
                    pos = dec(l.cuda().float()).cpu().numpy().reshape(l.shape[0], -1, 3)
                    pos = pos * sc["std"] + sc["mean"]; pos -= pos.mean(axis=1, keepdims=True)
                    pos += cg[i * BATCH:(i + 1) * BATCH][:, None]
                    for p in pos: u.atoms.positions = p; w.write(u.atoms)
            td = time.time() - t0
            sz = lambda f: os.path.getsize(f"{od}/{f}")
            res[key] = dict(system=s, setting=st, epochs=EPOCHS, latent=latent, n_inputs=N, n_params=sum(p.numel() for p in ae.parameters()),
                            n_decoder_params=sum(p.numel() for p in ae.decoder.parameters()), bytes_compressed_pkl=sz(f"{s}_compressed.pkl"),
                            bytes_cog=sz(f"{s}_cog.npy"), bytes_scaler=sz(f"{s}_scaler.pkl"), bytes_decoder_state=sz("decoder_state.pt"),
                            encode_s=te, decode_s=td, final_train_rmsd_scaled=float(model.train_loss[-1]) if model.train_loss else None,
                            gpu=torch.cuda.get_device_name(0))
            os.remove(f"{od}/decoder_state.pt")
            print(time.strftime("%H:%M:%S"), key, f"enc {te:.0f}s dec {td:.1f}s lat {latent} loss {res[key]['final_train_rmsd_scaled']}", flush=True)
        except Exception as e:
            import traceback; res[key] = dict(error=traceback.format_exc()[-3000:]); print("FAIL", key, repr(e), flush=True)
        json.dump(res, open(f"{O}/ct_results.json", "w"), indent=1)
        torch.cuda.empty_cache()
print("done", flush=True)
