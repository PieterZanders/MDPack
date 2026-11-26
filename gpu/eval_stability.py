#!/usr/bin/env python
"""Side-chain accuracy split by rotamer stability (stable / intermediate / unstable), on CPU.

Classification uses the TRUE MD only. For each residue with chi angles:
  * chi1 is assigned to rotameric wells p (+60), t (180), m (-60); PRO to two puckers (chi1 > 0 / < 0).
  * chi2 is added to the state where it is an sp3-sp3 torsion with three wells (ARG GLN GLU ILE LEU LYS MET).
  * Core-to-core assignment (hysteresis): a frame enters a well only when the angle is within CORE degrees of
    the well centre (PRO: |chi1| >= PRO_CORE); otherwise it keeps the previous state. Noise inside the
    barrier region is therefore not counted as a transition.
  * occupancy = fraction of frames in the dominant joint state; transitions per ns = state changes / time.
  * stable: occupancy >= 0.90 and <= MAX_STABLE_RATE transitions/ns; unstable: occupancy < 0.70;
    intermediate: everything else. not_stable = intermediate + unstable (binary split).
Also: chi1 circular variance (continuous), burial (CB neighbour count within 10 A, buried if >= BURIED_NB).

Per class metrics for every method (pooled over residues): symmetry-corrected sc RMSD after backbone
superposition, chi1-4 MAE, chi1 within 40 deg, rotamer-state accuracy per frame (nearest well of predicted chi1
== nearest well of true chi1; and joint chi1+chi2 state), local side-chain RMSF ratio recon/true (atoms in their
own residue frame, so only side-chain motion counts), Jensen-Shannon divergence (base 2) of chi1 state
populations, and transition rate true vs recon (same hysteresis on predicted chis).
Frames: mdCATH test replica 0, contiguous, all frames up to --max_frames (2 ns apart); external: all frames.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Domain, bind, build_model, fit_buffers, load_npz_domain, load_weights  # noqa: E402
from md_compress import residue_constants as rc  # noqa: E402
from md_compress import losses as L  # noqa: E402
from md_compress.geometry import SideChainBuilder, frames_from_backbone, kabsch, to_local  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/md_compress_gpu_data/data_mdcath")
ap.add_argument("--external", default="/workspace/md_compress_gpu_data/data_external")
ap.add_argument("--runs", default="results_gpu/runs")
ap.add_argument("--out", default="results_gpu/eval/stability_raw.json")
ap.add_argument("--max_frames", type=int, default=250)
ap.add_argument("--only_systems", default="")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.set_flush_denormal(True)
dev = "cpu"
CORE, PRO_CORE, OCC_STABLE, OCC_UNSTABLE, MAX_STABLE_RATE, BURIED_NB = 40.0, 10.0, 0.90, 0.70, 0.02, 16
CRITERIA = dict(
    source="true MD frames only (never predictions)",
    chi1_wells_deg={"p": 60, "t": 180, "m": -60}, pro_chi1_wells="two puckers, sign of chi1",
    chi2_in_state_for=["ARG", "GLN", "GLU", "ILE", "LEU", "LYS", "MET"],
    core_half_width_deg=CORE, pro_core_min_abs_deg=PRO_CORE,
    assignment="core-to-core hysteresis: a frame changes state only when it enters the core of another well",
    stable=f"dominant joint-state occupancy >= {OCC_STABLE} and transitions <= {MAX_STABLE_RATE} per ns",
    unstable=f"dominant occupancy < {OCC_UNSTABLE}", intermediate="otherwise", not_stable="intermediate + unstable",
    excluded="GLY, ALA (no chi)",
    burial=f"buried if >= {BURIED_NB} CB atoms (CA for GLY) within 10 A of the residue's CB in frame 0",
    frames="mdCATH test: replica 0, contiguous, 2 ns apart, up to 250 frames (500 ns); ATLAS: 101 frames 1 ns apart; "
           "AdK: 98 frames of a DIMS open to closed path (time axis nominal, 1 per frame)",
)
SP3_CHI2 = {rc.RESTYPE_ORDER[r] for r in CRITERIA["chi2_in_state_for"]}
PRO = rc.RESTYPE_ORDER["PRO"]
builder = SideChainBuilder()
man = json.load(open(f"{a.data}/manifest.json"))
res = json.load(open(a.out)) if os.path.exists(a.out) else {}


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def save():
    json.dump(res, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)


# ------------------------------------------------------------------ models (same set as eval_gpu.py)
def ck(p):
    return torch.load(p, map_location=dev, weights_only=False)["weights"] if os.path.exists(p) else None


codec_w = ck(f"{a.runs}/codec_S_prior/best.pt")
MODELS = {}
for v in ("S", "L"):
    for prior in (True, False):
        tag = f"{v}_{'prior' if prior else 'noprior'}"
        pw = ck(f"{a.runs}/packer_{tag}/best.pt")
        if pw is not None:
            m = build_model(v, prior, dev); load_weights(m, codec_w, "codec."); load_weights(m, pw, "packer.")
            MODELS[f"staged_{tag}"] = m.eval()
        ew = ck(f"{a.runs}/e2e_{tag}/best.pt")
        if ew is not None:
            m = build_model(v, prior, dev); load_weights(m, ew); MODELS[f"e2e_{tag}"] = m.eval()
_libp = f"{a.runs}/eval/type_mode_rotamers.pt"
if not os.path.exists(_libp):
    _libp = "results_gpu/eval/type_mode_rotamers.pt"
if not os.path.exists(_libp):
    raise SystemExit("missing type_mode_rotamers.pt; rebuild it with gpu/eval_gpu.py")
LIB = torch.load(_libp)
log(f"models {list(MODELS)}")


@torch.no_grad()
def run_model(m, dom, x, mode):
    bind(m, dom)
    P, C = [], []
    for s in range(0, len(x), 25):
        xb = x[s:s + 25]
        bb_al, R_, t_ = m.codec.align(xb[:, :, :4])
        if mode == "true_bb":
            outs = m.pack(bb_al, dom.aatype, dom.chain_index, dom.chain_break)
        else:
            z = m.codec.encode(bb_al, dom.aatype).half().float()
            _, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        P.append(m.codec.unalign(outs[-1]["atom14"], R_, t_)); C.append(outs[-1]["chi_sc"])
    return torch.cat(P), torch.cat(C)


# ------------------------------------------------------------------ rotamer states
def ang(sc):  # (sin,cos) -> degrees in (-180,180]
    return np.degrees(np.arctan2(sc[..., 0], sc[..., 1]))


def nearest_well(chi1, aat):
    """[T,R] chi1 deg -> nearest well index (0 p, 1 t, 2 m; PRO 0/1)."""
    d = np.stack([np.abs((chi1 - c + 180) % 360 - 180) for c in (60, 180, -60)], -1)
    w = d.argmin(-1)
    pro = aat == PRO
    w[:, pro] = (chi1[:, pro] < 0).astype(int)
    return w


def hysteresis_states(chi, aat, chi_mask):
    """chi [T,R,4] deg -> joint state [T,R] with core-to-core assignment."""
    T, R = chi.shape[:2]
    out = np.zeros((T, R), int)
    for k, use in ((0, np.ones(R, bool)), (1, np.isin(aat, list(SP3_CHI2)) & (chi_mask[:, 1] > 0))):
        x = chi[:, :, k]
        d = np.stack([np.abs((x - c + 180) % 360 - 180) for c in (60, 180, -60)], -1)
        core = np.where(d.min(-1) <= CORE, d.argmin(-1), -1)
        near = d.argmin(-1)
        if k == 0:
            pro = aat == PRO
            core[:, pro] = np.where(np.abs(x[:, pro]) >= PRO_CORE, (x[:, pro] < 0).astype(int), -1)
            near[:, pro] = (x[:, pro] < 0).astype(int)
        st = np.zeros((T, R), int)
        st[0] = np.where(core[0] >= 0, core[0], near[0])
        for t in range(1, T):
            st[t] = np.where(core[t] >= 0, core[t], st[t - 1])
        out = out * 3 + np.where(use[None], st, 0)
    return out


def occ_and_rate(states, dt):
    T = states.shape[0]
    occ = np.array([np.bincount(states[:, i]).max() / T for i in range(states.shape[1])])
    trans = (states[1:] != states[:-1]).sum(0)
    return occ, trans / max((T - 1) * dt, 1e-9)


def jsd(p, q):
    m = 0.5 * (p + q)
    f = lambda a, b: np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0).sum(-1)
    return 0.5 * f(p, m) + 0.5 * f(q, m)


def pops(w, n=3):
    return np.stack([(w == k).mean(0) for k in range(n)], -1)  # [R,n]


# ------------------------------------------------------------------ per-residue metrics for one prediction
SWAP = None


def canon_local(xl, ref, aat):
    """choose per frame the symmetric swap closest to ref (local coords [T,R,14,3])."""
    perm = L.tables()["swap_perm"][torch.as_tensor(aat)]
    sw = torch.gather(xl, -2, perm.expand(*xl.shape[:-1])[..., None].expand(*xl.shape))
    d1 = ((xl - ref) ** 2).sum((-1, -2)); d2 = ((sw - ref) ** 2).sum((-1, -2))
    return torch.where((d2 < d1)[..., None, None], sw, xl)


def residue_metrics(pred, pchi, x, tchi, dom, true_loc_c, dt):
    T, R = x.shape[:2]
    aat = dom.aatype
    m = dom.atom_mask.expand(T, R, 14)
    Rk, tk = kabsch(pred[:, :, :4].reshape(T, -1, 3), x[:, :, :4].reshape(T, -1, 3), m[..., :4].reshape(T, -1))
    P = (pred.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]).reshape(pred.shape)
    scm = m.clone(); scm[..., :4] = 0
    d2 = ((P - x) ** 2).sum(-1); d2s = ((P - L._swapped(x, aat)) ** 2).sum(-1)
    sc_err = torch.minimum((d2 * scm).sum(-1), (d2s * scm).sum(-1)).sum(0).numpy()  # [R] summed over frames
    n_sc = scm[0].sum(-1).numpy()  # per frame
    err = L.chi_abs_err_deg(pchi, tchi, aat).numpy()  # [T,R,4]
    # local-frame RMSF of side-chain atoms
    Rp, tp = frames_from_backbone(pred[..., 0, :], pred[..., 1, :], pred[..., 2, :])
    pl = canon_local(to_local(pred, Rp, tp), true_loc_c[:1], aat)
    msf = lambda xl: (((xl - xl.mean(0)) ** 2).sum(-1).mean(0) * scm[0]).sum(-1) / scm[0].sum(-1).clamp(min=1)
    rmsf_p, rmsf_t = msf(pl).sqrt().numpy(), msf(true_loc_c).sqrt().numpy()
    pc, tc = ang(pchi.numpy()), ang(tchi.numpy())
    wp, wt = nearest_well(pc[..., 0], aat.numpy()), nearest_well(tc[..., 0], aat.numpy())
    cm = dom.chi_mask.numpy()
    hp, ht = hysteresis_states(pc, aat.numpy(), cm), hysteresis_states(tc, aat.numpy(), cm)
    _, rate_p = occ_and_rate(hp, dt)
    _, rate_t = occ_and_rate(ht, dt)
    return dict(sc_err=sc_err, n_sc=n_sc, chi_err=err.sum(0), chi1_ok=(err[..., 0] < 40).sum(0),
                state_ok=(wp == wt).sum(0), joint_ok=(hp == ht).sum(0), T=T,
                rmsf_p=rmsf_p, rmsf_t=rmsf_t, jsd=jsd(pops(wt), pops(wp)), rate_p=rate_p, rate_t=rate_t)


CLASSES = ("stable", "intermediate", "unstable", "not_stable", "all")


def summarise(rm, cls, cm):
    """pool residue metrics over a class mask -> sums (for pooling across systems) and means."""
    out = {}
    for c in CLASSES:
        sel = cls[c]
        if sel.sum() == 0:
            out[c] = dict(n_res=0); continue
        cmk = cm[sel]  # [n,4]
        s = dict(n_res=int(sel.sum()), T=rm["T"],
                 sc_err=float(rm["sc_err"][sel].sum()), n_sc=float(rm["n_sc"][sel].sum() * rm["T"]),
                 chi_err=[float((rm["chi_err"][sel, k] * cmk[:, k]).sum()) for k in range(4)],
                 n_chi=[float(cmk[:, k].sum() * rm["T"]) for k in range(4)],
                 chi1_ok=float(rm["chi1_ok"][sel].sum()), state_ok=float(rm["state_ok"][sel].sum()),
                 joint_ok=float(rm["joint_ok"][sel].sum()),
                 rmsf_p=float(rm["rmsf_p"][sel].sum()), rmsf_t=float(rm["rmsf_t"][sel].sum()),
                 jsd=float(rm["jsd"][sel].sum()), rate_p=float(rm["rate_p"][sel].sum()), rate_t=float(rm["rate_t"][sel].sum()))
        out[c] = s
    return out


def evaluate(key, dom, x_all, x, dt, meta):
    t0 = time.time()
    fit_buffers(dom, x_all, builder)
    tchi, _ = builder.chis_from_atom14(x, dom.aatype)
    aat = dom.aatype.numpy(); cm = dom.chi_mask.numpy()
    has = cm[:, 0] > 0
    tc = ang(tchi.numpy())
    hs = hysteresis_states(tc, aat, cm)
    occ, rate = occ_and_rate(hs, dt)
    c1 = np.radians(tc[..., 0]); circvar = 1 - np.hypot(np.cos(c1).mean(0), np.sin(c1).mean(0))
    stable = has & (occ >= OCC_STABLE) & (rate <= MAX_STABLE_RATE)
    unstable = has & (occ < OCC_UNSTABLE)
    inter = has & ~stable & ~unstable
    cb = x[0, :, 4].numpy().copy(); gly = dom.atom_mask[:, 4].numpy() == 0; cb[gly] = x[0, gly, 1].numpy()
    nb = ((np.linalg.norm(cb[:, None] - cb[None], axis=-1) < 10).sum(1) - 1)
    buried = nb >= BURIED_NB
    cls = dict(stable=stable, intermediate=inter, unstable=unstable, not_stable=inter | unstable, all=has)
    for b, bm in (("buried", buried), ("exposed", ~buried)):
        for c in ("stable", "not_stable"):
            cls[f"{c}_{b}"] = cls[c] & bm
    # true local coords (canonical swaps relative to frame 0)
    Rt, tt = frames_from_backbone(x[..., 0, :], x[..., 1, :], x[..., 2, :])
    tl = to_local(x, Rt, tt)
    tl = canon_local(tl, tl[:1], dom.aatype)
    r = dict(meta, n_res=dom.R, n_frames=len(x), ns_per_frame=dt,
             residues=dict(occupancy=occ.round(4).tolist(), transitions_per_ns=rate.round(5).tolist(),
                           chi1_circvar=circvar.round(4).tolist(), cb_neighbours=nb.tolist(), has_chi=has.tolist(),
                           cls=np.select([stable, inter, unstable], ["S", "I", "U"], "-").tolist()),
             methods={})
    global CLASSES
    allc = list(cls)

    def add(name, pred, pchi):
        rm = residue_metrics(pred, pchi, x, tchi, dom, tl, dt)
        global CLASSES
        old = CLASSES; CLASSES = allc
        r["methods"][name] = summarise(rm, cls, cm)
        CLASSES = old

    rec, _ = builder(x[:, :, :4], dom.aatype, tchi); add("floor", rec, tchi)
    mc = dom.chi_ref.expand_as(tchi); rec, _ = builder(x[:, :, :4], dom.aatype, mc); add("static_rotamers", rec, mc)
    lc = (LIB[dom.aatype] * dom.chi_mask[..., None]).expand_as(tchi); rec, _ = builder(x[:, :, :4], dom.aatype, lc)
    add("type_library", rec, lc)
    for name, m in MODELS.items():
        add(f"{name}/e2e", *run_model(m, dom, x, "e2e"))
        if name.startswith("staged"):
            add(f"{name}/true_bb", *run_model(m, dom, x, "true_bb"))
    r["seconds"] = round(time.time() - t0, 1)
    res[key] = r; save()
    n = {c: int(cls[c].sum()) for c in ("stable", "intermediate", "unstable")}
    log(f"{key} ({dom.R} res, {len(x)} fr): {n} {r['seconds']}s")


only = set(a.only_systems.split(",")) - {""}
test = sorted(d for d, v in man["domains"].items() if v["split"] == "test")
for d in test:
    key = f"mdcath/{d}"
    if key in res or (only and d not in only):
        continue
    z = np.load(f"{a.data}/domains/{d}.npz")
    rep0 = np.where(z["replica"] == 0)[0]
    x_all = torch.as_tensor(z["coords"][rep0].astype(np.float32))
    dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], dev)
    evaluate(key, dom, x_all, x_all[:a.max_frames], 2.0, dict(set="mdcath_test"))
import glob  # noqa: E402
for p in sorted(glob.glob(f"{a.external}/*.npz")):
    name = os.path.basename(p)[:-4]
    key = f"external/{name}"
    if key in res or (only and name not in only):
        continue
    dom, z = load_npz_domain(p, dev)
    x_all = torch.as_tensor(dom.coords)
    evaluate(key, dom, x_all, x_all, 1.0, dict(set="external", time_axis="nominal (DIMS path)" if name.startswith("ADK") else "1 ns"))
res["_criteria"] = CRITERIA
save()
log("done")
