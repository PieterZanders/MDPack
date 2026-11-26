#!/usr/bin/env python
"""Per-trajectory side-chain fine-tuning on top of the frozen shared model (E2E S with prior).

Pipeline per trajectory (compression time, original coordinates available):
  1. frozen codec encodes/decodes the backbone (fp16 latents), frozen packer predicts chis with the stored chi prior;
  2. ONLY a small correction is fitted to this trajectory's true frames, minimising the squared symmetry-corrected
     side-chain atom RMSD after backbone superposition (the evaluation metric);
  3. the correction is stored once in fp16 and counted in the per-protein stored data.
Variants:
  bias  per-residue chi offsets (one angle per defined chi), applied as a rotation of the packer output
  mlp   tiny residual MLP shared by all residues: [final trunk features h (80), packer chi sin/cos (8)] -> 4 angle offsets
  head  the final-pass chi output layer (Linear 80 -> 8) unfrozen, initialised from the pretrained weights
Regimes:
  same     stored data (reference, chi prior) and correction fitted on ALL evaluated frames, evaluated on the same frames
  heldout  stored data and correction fitted on the first 80% of frames, evaluated on the last 20%
Baselines: no fine-tune (frozen model) and prior alone (stored mean rotamers placed on the decoded backbone).
Stability classes (stable / intermediate / unstable) come from the TRUE MD over all evaluated frames (same definition as
sidechain_by_stability). Frames: mdCATH test replica 0, contiguous, 2 ns apart, up to 250; external: all frames.
Raw per-system sums go to results_gpu/eval/finetune_raw.json (resumable); --summarize writes finetune_results.json.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Domain, bind, build_model, fit_buffers, load_npz_domain, load_weights  # noqa: E402
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import SideChainBuilder, frames_from_backbone, to_local  # noqa: E402
from md_compress.sidechain_model import rotate_sc  # noqa: E402
import stability_lib as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/md_compress_gpu_data/data_mdcath")
ap.add_argument("--external", default="/workspace/md_compress_gpu_data/data_external")
ap.add_argument("--runs", default="results_gpu/runs")
ap.add_argument("--raw", default="results_gpu/eval/finetune_raw.json")
ap.add_argument("--out", default="results_gpu/eval/finetune_results.json")
ap.add_argument("--max_frames", type=int, default=250)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--only", default="")
ap.add_argument("--summarize", action="store_true")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.set_flush_denormal(True)
builder = SideChainBuilder()
VARIANTS = ("bias", "mlp", "head")
LR = dict(bias=0.05, mlp=3e-3, head=1e-3)
FIT_FRAC = 0.8


# ------------------------------------------------------------------ corrections
class Bias(nn.Module):
    def __init__(self, R):
        super().__init__(); self.theta = nn.Parameter(torch.zeros(R, 4))

    def forward(self, h, base, chi_ref, cm):
        d = torch.stack([self.theta.sin(), self.theta.cos()], -1)
        return rotate_sc(d.expand_as(base), base) * cm[..., None]


class MLP(nn.Module):
    def __init__(self, d_h=80, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_h + 8, hid), nn.SiLU(), nn.Linear(hid, 4))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, base, chi_ref, cm):
        th = self.net(torch.cat([h, base.flatten(-2)], -1))
        d = torch.stack([th.sin(), th.cos()], -1)
        return rotate_sc(d, base) * cm[..., None]


class Head(nn.Module):
    def __init__(self, lin):
        super().__init__(); self.lin = nn.Linear(lin.in_features, lin.out_features)
        self.lin.load_state_dict(lin.state_dict())

    def forward(self, h, base, chi_ref, cm):
        x = F.normalize(self.lin(h).view(*h.shape[:-1], 4, 2), dim=-1, eps=1e-6)
        return rotate_sc(x, chi_ref.expand_as(x)) * cm[..., None]


def n_stored_params(name, mod, cm):
    if name == "bias":
        return int(cm.sum().item())  # one angle per defined chi
    return sum(p.numel() for p in mod.parameters())


# ------------------------------------------------------------------ frozen forward
ck = lambda p: torch.load(p, map_location="cpu", weights_only=False)["weights"]


def load_main():
    m = build_model("S", True, "cpu"); load_weights(m, ck(f"{a.runs}/e2e_S_prior/best.pt"))
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval()


@torch.no_grad()
def frozen(m, dom, x):
    bind(m, dom); BB, H, C, RR, TT = [], [], [], [], []
    for s in range(0, len(x), 25):
        bb_al, R_, t_ = m.codec.align(x[s:s + 25, :, :4])
        z = m.codec.encode(bb_al, dom.aatype).half().float()
        bb, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        BB.append(bb); H.append(outs[-1]["h"]); C.append(outs[-1]["chi_sc"]); RR.append(R_); TT.append(t_)
    return dict(bb=torch.cat(BB), h=torch.cat(H), base=torch.cat(C), R=torch.cat(RR), t=torch.cat(TT))


def build(m, fr, chi_sc, dom):
    atom14, _ = builder(fr["bb"], dom.aatype, chi_sc)
    return m.codec.unalign(atom14, fr["R"], fr["t"])


def sc_loss_setup(m, fr, x, dom):
    """Fixed per-frame superposition of the decoded backbone onto the true one (the correction does not move the
    backbone), so the loss below is exactly the squared symmetry-corrected side-chain RMSD used in evaluation."""
    with torch.no_grad():
        T, R = x.shape[:2]
        bbu = m.codec.unalign(fr["bb"], fr["R"], fr["t"])
        mk = dom.atom_mask.expand(T, R, 14)
        Rk, tk = S.kabsch(bbu.reshape(T, -1, 3), x[:, :, :4].reshape(T, -1, 3), mk[..., :4].reshape(T, -1))
        scm = mk.clone(); scm[..., :4] = 0
    return Rk, tk, scm, L._swapped(x, dom.aatype)


def sc_loss(m, fr, chi_sc, x, dom, setup):
    Rk, tk, scm, xs = setup
    P = build(m, fr, chi_sc, dom)
    P = (P.reshape(P.shape[0], -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(P.shape)
    d2 = ((P - x) ** 2).sum(-1); d2s = ((P - xs) ** 2).sum(-1)
    return torch.minimum((d2 * scm).sum(-1), (d2s * scm).sum(-1)).sum() / scm.sum()


def fit(name, m, fr, x, dom):
    """Fit the correction by minimising the mean squared side-chain atom error (the evaluation metric) on x,
    with Adam on random minibatches of a.batch frames for a.steps steps (fixed budget, no early stopping)."""
    cm = dom.chi_mask.float()
    mod = Bias(dom.R) if name == "bias" else MLP() if name == "mlp" else Head(m.packer.sidechain.head.lin)
    opt = torch.optim.Adam(mod.parameters(), lr=LR[name])
    setup = sc_loss_setup(m, fr, x, dom)
    t0 = time.time(); hist = []
    T = x.shape[0]; g = torch.Generator().manual_seed(0)
    sub = lambda d, i: {k: v[i] for k, v in d.items()}
    for it in range(a.steps):  # Adam on random minibatches of frames
        i = torch.randperm(T, generator=g)[:a.batch]
        fb = sub(fr, i)
        loss = sc_loss(m, fb, mod(fb["h"], fb["base"], dom.chi_ref, cm), x[i], dom, tuple(v[i] for v in setup))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == a.steps - 1: hist.append(round(math.sqrt(loss.item()), 4))
    with torch.no_grad():  # store in fp16
        for p in mod.parameters(): p.copy_(p.half().float())
    return mod, time.time() - t0, hist


@torch.no_grad()
def metrics(pred, pchi, x, tchi, dom, cls, dt):
    Rt, tt = frames_from_backbone(x[..., 0, :], x[..., 1, :], x[..., 2, :])
    tl = to_local(x, Rt, tt); tl = S.canon_local(tl, tl[:1], dom.aatype)
    rm = S.residue_metrics(pred, pchi, x, tchi, dom, tl, dt)
    old = S.CLASSES; S.CLASSES = tuple(cls)
    out = S.summarise(rm, cls, dom.chi_mask.numpy()); S.CLASSES = old
    out["clash"] = L.clash_score(pred, dom.aatype, dom.atom_mask, dom.chain_index)
    return out


def evaluate(key, dom, x, dt, meta, m, raw):
    t0 = time.time()
    tchi_all, _ = builder.chis_from_atom14(x, dom.aatype)
    aat = dom.aatype.numpy()
    fit_buffers(dom, x, builder); cmn = dom.chi_mask.numpy(); has = cmn[:, 0] > 0
    tc = S.ang(tchi_all.numpy()); hs = S.hysteresis_states(tc, aat, cmn); occ, rate = S.occ_and_rate(hs, dt)
    st = has & (occ >= S.OCC_STABLE) & (rate <= S.MAX_STABLE_RATE); un = has & (occ < S.OCC_UNSTABLE)
    cls = dict(stable=st, intermediate=has & ~st & ~un, unstable=un, all=has)
    T = len(x); nf = int(round(FIT_FRAC * T))
    r = dict(meta, n_res=dom.R, n_heavy=dom.n_heavy, n_frames=T, dt_ns=dt, n_fit_heldout=nf,
             n_chi=int(cmn.sum()), class_counts={c: int(v.sum()) for c, v in cls.items()}, regimes={})
    for regime, xf, xe in (("same", x, x), ("heldout", x[:nf], x[nf:])):
        fit_buffers(dom, xf, builder)  # stored reference backbone + chi prior from the fitting frames only
        tf, _ = builder.chis_from_atom14(xf, dom.aatype); te, _ = builder.chis_from_atom14(xe, dom.aatype)
        ff = frozen(m, dom, xf); fe = ff if xe is xf else frozen(m, dom, xe)
        res = {}
        res["no_finetune"] = metrics(build(m, fe, fe["base"], dom), fe["base"], xe, te, dom, cls, dt)
        pri = dom.chi_ref.expand_as(fe["base"]) * dom.chi_mask[..., None]
        res["prior_alone"] = metrics(build(m, fe, pri, dom), pri, xe, te, dom, cls, dt)
        for v in VARIANTS:
            mod, sec, hist = fit(v, m, ff, xf, dom)
            with torch.no_grad():
                pc = mod(fe["h"], fe["base"], dom.chi_ref, dom.chi_mask.float())
            res[v] = metrics(build(m, fe, pc, dom), pc, xe, te, dom, cls, dt)
            npar = n_stored_params(v, mod, dom.chi_mask)
            res[v].update(fit_seconds=round(sec, 2), n_params=npar, extra_bytes_fp16=2 * npar, loss_hist=hist)
        r["regimes"][regime] = res
    r["seconds"] = round(time.time() - t0, 1)
    raw[key] = r
    json.dump(raw, open(a.raw + ".tmp", "w")); os.replace(a.raw + ".tmp", a.raw)
    u = r["regimes"]
    f = lambda reg, meth, c="all": math.sqrt(u[reg][meth][c]["sc_err"] / u[reg][meth][c]["n_sc"]) if u[reg][meth][c].get("n_res") else float("nan")
    print(time.strftime("%H:%M:%S"), key, T, f"{r['seconds']}s", " ".join(
        f"{reg}:" + ",".join(f"{mm[:4]} {f(reg, mm):.2f}" for mm in ("no_finetune", "bias", "mlp", "head")) for reg in ("same", "heldout")), flush=True)


def run():
    m = load_main()
    raw = json.load(open(a.raw)) if os.path.exists(a.raw) else {}
    only = set(a.only.split(",")) - {""}
    man = json.load(open(f"{a.data}/manifest.json"))
    for d in sorted(d for d, v in man["domains"].items() if v["split"] == "test"):
        key = f"mdcath/{d}"
        if key in raw or (only and d not in only): continue
        z = np.load(f"{a.data}/domains/{d}.npz"); rep0 = np.where(z["replica"] == 0)[0]
        x = torch.as_tensor(z["coords"][rep0][:a.max_frames].astype(np.float32))
        dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], "cpu")
        evaluate(key, dom, x, 2.0, dict(set="mdcath_test"), m, raw)
    for p in sorted(glob.glob(f"{a.external}/*.npz")):
        name = os.path.basename(p)[:-4]; key = f"external/{name}"
        if key in raw or (only and name not in only): continue
        dom, z = load_npz_domain(p, "cpu"); x = torch.as_tensor(dom.coords)
        evaluate(key, dom, x, 1.0, dict(set="external", time_axis="nominal (DIMS path)" if name.startswith("ADK") else "1 ns"), m, raw)
    print("done", flush=True)


if __name__ == "__main__" and not a.summarize:
    run()


# ------------------------------------------------------------------ summary + figure
METHODS = ("prior_alone", "no_finetune") + VARIANTS
LABEL = dict(prior_alone="prior alone", no_finetune="no fine-tune", bias="(a) chi bias", mlp="(b) residual MLP",
             head="(c) output layer")


def ratios(r, extra):
    raw = 12 * r["n_heavy"]; pf = 8 * r["n_res"] + 28; one = 80 * r["n_res"] + extra
    am = lambda T: raw * T / (pf * T + one)
    return dict(per_frame=raw / pf, T100=am(100), T1000=am(1000), T_actual=am(r["n_frames"]), one_off_bytes=one)


def pool(rows, meth, reg):
    out = {}
    for c in ("all", "stable", "intermediate", "unstable"):
        xs = [r["regimes"][reg][meth][c] for r in rows if r["regimes"][reg][meth][c].get("n_res")]
        sm = lambda k: sum(x[k] for x in xs)
        if not xs: out[c] = None; continue
        out[c] = dict(n_res=sm("n_res"), sc_rmsd=math.sqrt(sm("sc_err") / sm("n_sc")),
                      chi_mae=[sum(x["chi_err"][k] for x in xs) / max(sum(x["n_chi"][k] for x in xs), 1) for k in range(4)],
                      chi1_within40=sm("chi1_ok") / sum(x["n_chi"][0] for x in xs),
                      rotamer_state_acc=sm("state_ok") / sum(x["n_chi"][0] for x in xs),
                      sc_rmsf_ratio=sm("rmsf_p") / sm("rmsf_t"), chi1_pop_jsd=sm("jsd") / sm("n_res"),
                      transitions_per_ns_true=sm("rate_t") / sm("n_res"), transitions_per_ns_recon=sm("rate_p") / sm("n_res"),
                      transition_rate_ratio=sm("rate_p") / max(sm("rate_t"), 1e-12))
    out["clash_per_1000_atoms"] = float(np.mean([r["regimes"][reg][meth]["clash"] for r in rows]))
    if meth in VARIANTS:
        v = [r["regimes"][reg][meth] for r in rows]
        out["fit_seconds_mean"] = float(np.mean([x["fit_seconds"] for x in v]))
        out["fit_seconds_max"] = float(np.max([x["fit_seconds"] for x in v]))
        out["n_params_mean"] = float(np.mean([x["n_params"] for x in v]))
        out["extra_bytes_fp16_mean"] = float(np.mean([x["extra_bytes_fp16"] for x in v]))
        rr = [ratios(r, r["regimes"][reg][meth]["extra_bytes_fp16"]) for r in rows]
    else:
        rr = [ratios(r, 0) for r in rows]
    out["compression_mean"] = {k: float(np.mean([x[k] for x in rr])) for k in ("per_frame", "T100", "T1000", "T_actual")}
    base = [math.sqrt(r["regimes"][reg]["no_finetune"]["all"]["sc_err"] / r["regimes"][reg]["no_finetune"]["all"]["n_sc"]) for r in rows]
    me = [math.sqrt(r["regimes"][reg][meth]["all"]["sc_err"] / r["regimes"][reg][meth]["all"]["n_sc"]) for r in rows]
    out["systems_improved_vs_no_finetune"] = int(sum(b - x > 0.005 for b, x in zip(base, me)))
    out["systems_worse_vs_no_finetune"] = int(sum(x - b > 0.005 for b, x in zip(base, me)))
    return out


def summarize():
    raw = json.load(open(a.raw))
    keys = [k for k in raw if not k.startswith("_")]
    sets = dict(mdcath_test=[raw[k] for k in keys if raw[k]["set"] == "mdcath_test"],
                external=[raw[k] for k in keys if raw[k]["set"] == "external"])
    summ = {st: dict(n_systems=len(rows), regimes={reg: {mm: pool(rows, mm, reg) for mm in METHODS} for reg in ("same", "heldout")})
            for st, rows in sets.items() if rows}
    per_sys = {}
    for k in keys:
        r = raw[k]; d = dict(set=r["set"], n_res=r["n_res"], n_frames=r["n_frames"], class_counts=r["class_counts"], regimes={})
        for reg in ("same", "heldout"):
            d["regimes"][reg] = {}
            for mm in METHODS:
                x = r["regimes"][reg][mm]
                d["regimes"][reg][mm] = {c: (math.sqrt(x[c]["sc_err"] / x[c]["n_sc"]) if x[c].get("n_res") else None)
                                         for c in ("all", "stable", "intermediate", "unstable")}
                d["regimes"][reg][mm]["clash"] = x["clash"]
                if mm in VARIANTS:
                    d["regimes"][reg][mm].update({q: x[q] for q in ("fit_seconds", "n_params", "extra_bytes_fp16")},
                                                 compression=ratios(r, x["extra_bytes_fp16"]))
        per_sys[k] = d
    out = dict(description=__doc__, main_model="e2e_S_prior (frozen)", fit_frac_heldout=FIT_FRAC,
               fit=dict(loss="squared symmetry-corrected side-chain RMSD after per-frame backbone superposition",
                        optimizer="Adam", steps=a.steps, batch_frames=a.batch, lr=LR, stored_dtype="fp16", device="CPU (8 threads)"),
               variants=dict(bias="one angle offset per defined chi of each residue (2 B each)",
                             mlp="Linear(88,32)+SiLU+Linear(32,4) on [final trunk features, packer chi sin/cos] -> chi offsets, 2,980 params",
                             head="final-pass chi output layer Linear(80,8), 648 params, initialised from the pretrained layer"),
               compression="raw 12 B per heavy atom per frame; per frame 8 B/res + 28 B (unchanged); one-off 80 B/res "
                           "(reference backbone 48 + chi prior 32) + correction bytes; shared weights not counted",
               summary=summ, per_system=per_sys)
    json.dump(out, open(a.out, "w"), indent=1)
    figure(summ)
    return out


def figure(summ):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sets = [s for s in ("mdcath_test", "external") if s in summ]
    fig, ax = plt.subplots(len(sets), 2, figsize=(12, 4.4 * len(sets)), squeeze=False)
    cols = dict(prior_alone="#bbbbbb", no_finetune="#444444", bias="tab:blue", mlp="tab:orange", head="tab:green")
    cls = ("all", "stable", "intermediate", "unstable")
    for i, st in enumerate(sets):
        for j, reg in enumerate(("same", "heldout")):
            A = ax[i, j]; d = summ[st]["regimes"][reg]; w = 0.16
            for k, mm in enumerate(METHODS):
                y = [d[mm][c]["sc_rmsd"] if d[mm][c] else np.nan for c in cls]
                lab = LABEL[mm]
                if mm in VARIANTS:
                    lab += f": +{d[mm]['extra_bytes_fp16_mean']/1000:.1f} kB, {d[mm]['compression_mean']['T1000']:.2f}× at T=1000"
                elif mm == "no_finetune":
                    lab += f": {d[mm]['compression_mean']['T1000']:.2f}× at T=1000"
                b = A.bar(np.arange(4) + (k - 2) * w, y, w, color=cols[mm], label=lab)
                if mm != "prior_alone":
                    for xx, yy in zip(np.arange(4) + (k - 2) * w, y):
                        A.text(xx, yy + 0.02, f"{yy:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
            A.set_xticks(range(4)); A.set_xticklabels([f"{c}\n(n={d['no_finetune'][c]['n_res']})" for c in cls], fontsize=8)
            A.set_ylabel("side-chain RMSD (Å)"); A.set_ylim(0, max(d["prior_alone"]["unstable"]["sc_rmsd"], 2.2) * 1.18)
            ttl = "same frames (fit on all, archival)" if reg == "same" else "held-out (fit first 80%, test last 20%)"
            A.set_title(f"{'mdCATH test (50)' if st == 'mdcath_test' else 'external (8)'}: {ttl}", fontsize=10)
            A.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Per-trajectory fine-tuning on the frozen E2E S model: side-chain RMSD by rotamer stability class", fontsize=11)
    fig.tight_layout(); fig.savefig("report/gpu_figs/fig_finetune.png", dpi=200)
    import shutil; shutil.copy("report/gpu_figs/fig_finetune.png", "results_gpu/fig_finetune.png")


if a.summarize:
    summarize()
