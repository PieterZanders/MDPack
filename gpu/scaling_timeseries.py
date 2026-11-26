#!/usr/bin/env python
"""Per-frame reconstruction error vs simulation time (for the scaling figure, panel D).

Same fixed models and stored per-protein data as eval_gpu / eval_stability. For every system and frame:
bb RMSD and symmetry-corrected sc RMSD (after per-frame backbone superposition, as losses.evaluate), plus
the true backbone RMSD of the frame from the stored reference backbone (how far the conformation has moved).
mdCATH test: replica 0, contiguous, 2 ns apart, up to 250 frames. External: all frames (ATLAS 1 ns; AdK path).
"""
import glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Domain, bind, build_model, fit_buffers, load_npz_domain, load_weights  # noqa: E402
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import SideChainBuilder, kabsch  # noqa: E402

torch.set_num_threads(8); torch.set_flush_denormal(True)
DATA, EXT, RUNS = "/workspace/md_compress_gpu_data/data_mdcath", "/workspace/md_compress_gpu_data/data_external", "results_gpu/runs"
OUT = "results_gpu/eval/error_vs_time.json"
builder = SideChainBuilder()
ck = lambda p: torch.load(p, map_location="cpu", weights_only=False)["weights"]
codec_w = ck(f"{RUNS}/codec_S_prior/best.pt")
MODELS = {}
m = build_model("S", True, "cpu"); load_weights(m, ck(f"{RUNS}/e2e_S_prior/best.pt")); MODELS["e2e_S_prior"] = m.eval()
m = build_model("L", True, "cpu"); load_weights(m, codec_w, "codec."); load_weights(m, ck(f"{RUNS}/packer_L_prior/best.pt"), "packer.")
MODELS["staged_L_prior"] = m.eval()
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


@torch.no_grad()
def run(m, dom, x):
    bind(m, dom); P = []
    for s in range(0, len(x), 25):
        xb = x[s:s + 25]
        bb_al, R_, t_ = m.codec.align(xb[:, :, :4])
        z = m.codec.encode(bb_al, dom.aatype).half().float()
        _, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        P.append(m.codec.unalign(outs[-1]["atom14"], R_, t_))
    return torch.cat(P)


def per_frame(pred, x, dom):
    T, R = x.shape[:2]
    mk = dom.atom_mask.expand(T, R, 14)
    Rk, tk = kabsch(pred[:, :, :4].reshape(T, -1, 3), x[:, :, :4].reshape(T, -1, 3), mk[..., :4].reshape(T, -1))
    P = (pred.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(pred.shape)
    d2 = ((P - x) ** 2).sum(-1); d2s = ((P - L._swapped(x, dom.aatype)) ** 2).sum(-1)
    scm = mk.clone(); scm[..., :4] = 0
    bb = ((d2[..., :4] * mk[..., :4]).sum((1, 2)) / mk[..., :4].sum((1, 2))).sqrt()
    sc = (torch.minimum((d2 * scm).sum(-1), (d2s * scm).sum(-1)).sum(-1) / scm.sum((1, 2))).sqrt()
    return bb.numpy().round(4).tolist(), sc.numpy().round(4).tolist()


def evaluate(key, dom, x_all, x, dt, meta):
    t0 = time.time()
    fit_buffers(dom, x_all, builder)
    T, R = x.shape[:2]
    ref = dom.ref[None].expand(T, R, 4, 3)
    mk = dom.atom_mask[None, :, :4].expand(T, R, 4).reshape(T, -1)
    Rk, tk = kabsch(x[:, :, :4].reshape(T, -1, 3), ref.reshape(T, -1, 3), mk)
    xa = (x[:, :, :4].reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(T, R, 4, 3)
    dref = (((xa - ref) ** 2).sum(-1) * mk.reshape(T, R, 4)).sum((1, 2)) / mk.sum(-1)
    r = dict(meta, n_res=dom.R, n_heavy=int(dom.n_heavy), dt_ns=dt, time_ns=(np.arange(T) * dt).tolist(),
             true_bb_rmsd_from_ref=dref.sqrt().numpy().round(4).tolist(), methods={})
    for name, mdl in MODELS.items():
        bb, sc = per_frame(run(mdl, dom, x), x, dom)
        r["methods"][name] = dict(bb_rmsd=bb, sc_rmsd=sc)
    res[key] = r
    json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
    print(time.strftime("%H:%M:%S"), key, T, f"{time.time()-t0:.0f}s", flush=True)


man = json.load(open(f"{DATA}/manifest.json"))
for d in sorted(d for d, v in man["domains"].items() if v["split"] == "test"):
    key = f"mdcath/{d}"
    if key in res: continue
    z = np.load(f"{DATA}/domains/{d}.npz"); rep0 = np.where(z["replica"] == 0)[0]
    x_all = torch.as_tensor(z["coords"][rep0].astype(np.float32))
    dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], "cpu")
    evaluate(key, dom, x_all, x_all[:250], 2.0, dict(set="mdcath_test"))
for p in sorted(glob.glob(f"{EXT}/*.npz")):
    name = os.path.basename(p)[:-4]; key = f"external/{name}"
    if key in res: continue
    dom, z = load_npz_domain(p, "cpu"); x_all = torch.as_tensor(dom.coords)
    evaluate(key, dom, x_all, x_all, 1.0, dict(set="external", time_axis="nominal (DIMS path)" if name.startswith("ADK") else "1 ns"))
print("done", flush=True)
