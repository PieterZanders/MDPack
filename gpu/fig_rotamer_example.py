#!/usr/bin/env python
"""Example of rotamer hopping: chi1 time series and chi1 well populations, true MD vs reconstructions,
for the most frequently hopping unstable residue and one stable residue of a system."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bind, build_model, fit_buffers, load_npz_domain, load_weights  # noqa: E402
from md_compress import residue_constants as rc  # noqa: E402
from md_compress.geometry import SideChainBuilder  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz, key, dt, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
torch.set_num_threads(4)
raw = json.load(open("results_gpu/eval/stability_raw.json"))[key]["residues"]
dom, _ = load_npz_domain(npz, "cpu", replica=0 if "mdcath" in key else None)
x = torch.as_tensor(dom.coords)[:250]
b = SideChainBuilder(); fit_buffers(dom, torch.as_tensor(dom.coords), b)
tchi, cm = b.chis_from_atom14(x, dom.aatype)
preds = {}
for tag, lab in (("e2e_S_prior", "e2e S, prior"), ("e2e_S_noprior", "e2e S, no prior")):
    m = build_model("S", "noprior" not in tag, "cpu"); load_weights(m, torch.load(f"results_gpu/runs/{tag}/best.pt", map_location="cpu", weights_only=False)["weights"]); m.eval()
    bind(m, dom)
    with torch.no_grad():
        al, _, _ = m.codec.align(x[:, :, :4]); z = m.codec.encode(al, dom.aatype).half().float()
        _, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
    preds[lab] = outs[-1]["chi_sc"]
ang = lambda sc: np.degrees(np.arctan2(sc[..., 0].numpy(), sc[..., 1].numpy()))
cls = np.array(raw["cls"]); rate = np.array(raw["transitions_per_ns"]); nb = np.array(raw["cb_neighbours"])
uns = np.where(cls == "U")[0]; st = np.where(cls == "S")[0]
u = uns[np.argmax(rate[uns])]
s = st[np.argmax(nb[st])] if len(st) else uns[0]
names = rc.RESTYPES
t = np.arange(len(x)) * dt
fig, axes = plt.subplots(2, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[3.2, 1]))
cols = {"MD (true)": "#222222", "e2e S, prior": "#2F64B0", "e2e S, no prior": "#E0A526"}
for row, i in enumerate((u, s)):
    series = {"MD (true)": ang(tchi)[:, i, 0], **{k: ang(v)[:, i, 0] for k, v in preds.items()}}
    ax = axes[row, 0]
    for k, v in series.items():
        ax.plot(t, v, ".", ms=3 if k == "MD (true)" else 2.5, color=cols[k], label=k, alpha=0.9 if k == "MD (true)" else 0.7)
    for c in (60, 180, -60):
        ax.axhline(c, color="#bbb", lw=0.6, ls="--")
    ax.set_ylim(-180, 180); ax.set_yticks([-180, -60, 60, 180]); ax.set_ylabel("χ1 (°)")
    lab = "unstable" if row == 0 else "stable"
    ax.set_title(f"{names[int(dom.aatype[i])]}{i + 1} ({lab}: dominant occupancy {raw['occupancy'][i]:.0%}, "
                 f"{rate[i]:.2f} transitions/ns, {nb[i]} CB neighbours)", fontsize=10)
    ax.set_xlabel("time (ns)" if "mdcath" in key or "ADK" not in key else "frame")
    if row == 0:
        ax.legend(fontsize=8, loc="upper right", markerscale=3)
    ax = axes[row, 1]
    wells = (60, 180, -60)
    for n, (k, v) in enumerate(series.items()):
        d = np.stack([np.abs((v - c + 180) % 360 - 180) for c in wells], -1).argmin(-1)
        p = [(d == w).mean() for w in range(3)]
        ax.bar(np.arange(3) + (n - 1) * 0.27, p, 0.27, color=cols[k], label=k)
    ax.set_xticks(range(3)); ax.set_xticklabels(["p (+60)", "t (180)", "m (−60)"]); ax.set_ylim(0, 1)
    ax.set_title("χ1 well populations", fontsize=10)
plt.suptitle(f"{key.split('/')[-1]}: rotamer hopping, true MD vs reconstruction from fp16 latents", fontsize=11)
plt.tight_layout(); plt.savefig(out, dpi=130); print(out, names[int(dom.aatype[u])], u + 1, names[int(dom.aatype[s])], s + 1)
