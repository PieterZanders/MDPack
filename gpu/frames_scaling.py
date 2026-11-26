#!/usr/bin/env python
"""Error and compression vs number of archived frames T, per-trajectory fitting at every T.

Data (gpu/fetch_long.py): mdCATH test domains at the native 1 ns spacing. Archive order: 320 K replicas 0..3, then
348, 379, 413, 450 K replicas 0..4. The archived trajectory for level T is the first T frames of that order.
Held-out set (fixed for all T): 320 K replica 4, 200 frames 2 ns apart, never archived.
Per T, fitted on the T archived frames only: reference backbone, chi prior, flip mask, per-state chi table, residual MLPs.
Methods: no_finetune; ft_mlp (residual MLP, section 12); code12_prior (chi1+chi2 state codes, state prior only);
code12 (codes + frozen model); code12_mlp (codes + model + MLP refitted with state priors, section 13 recommendation).
Evaluation: (a) same frames: all T frames for T <= 1000; for T > 1000, four contiguous blocks of 250 frames starting
at 0, T/4, T/2, 3T/4 (1,000 frames). (b) held-out frames. Stability classes are fixed per domain from 320 K replica 0
(1 ns spacing, same thresholds as sidechain_by_stability). MLP fits: 400 Adam steps on batches of 48 frames drawn from
min(T, 1000) frames (a random subset for T > 1000), same budget at every T. Clashes on at most 250 evenly spaced frames.
Compression at T: 12 B per heavy atom per frame raw; stored = T*(8R+28) + 80R + flip mask + state table + actual zlib
bytes of the state codes of the T frames (+ 5,960 B per MLP); shared weights not counted.
Output: results_gpu/eval/frames_scaling_raw.json, one entry per domain and T (resumable); --summarize -> frames_scaling.json.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0]]
import finetune_traj as FT  # noqa: E402
import rotamer_codes as RC  # noqa: E402
sys.argv = _argv
from common import Domain, bind, fit_buffers  # noqa: E402
from md_compress import losses as L  # noqa: E402
import stability_lib as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/md_compress_gpu_data/data_long")
ap.add_argument("--raw", default="results_gpu/eval/frames_scaling_raw.json")
ap.add_argument("--out", default="results_gpu/eval/frames_scaling.json")
ap.add_argument("--only", default="")
ap.add_argument("--levels", default="10,50,100,250,500,1000,5000,10000")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--summarize", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
builder = FT.builder
FIT_MAX, BLOCK, CLASH_MAX = 1000, 250, 250
METHODS = ("no_finetune", "ft_mlp", "code12_prior", "code12", "code12_mlp")


def blocks_of(T):
    if T <= 1000: return [np.arange(T)]
    return [np.arange(s, s + BLOCK) for s in (0, T // 4, T // 2, 3 * T // 4)]


@torch.no_grad()
def forward(m, dom, x, cref, mods):
    """frozen codec+packer on x with per-frame chi prior cref [T,R,4,2]; base, state-prior and corrected predictions."""
    bind(m, dom); cm = dom.chi_mask.float()
    out = {k: [] for k in ["base", "prior", "bb", "chi_base"] + list(mods) + ["chi_" + k for k in mods]}
    for s in range(0, len(x), 25):
        m.codec.chi_ref = cref[s:s + 25]
        bb_al, R_, t_ = m.codec.align(x[s:s + 25, :, :4])
        z = m.codec.encode(bb_al, dom.aatype).half().float()
        bb, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        fr = dict(bb=bb, R=R_, t=t_)
        out["base"].append(FT.build(m, fr, outs[-1]["chi_sc"], dom)); out["chi_base"].append(outs[-1]["chi_sc"])
        out["prior"].append(FT.build(m, fr, cref[s:s + 25] * cm[..., None], dom))
        out["bb"].append(m.codec.unalign(bb, R_, t_))
        for k, mod in mods.items():
            pc = mod(outs[-1]["h"], outs[-1]["chi_sc"], None, cm)
            out[k].append(FT.build(m, fr, pc, dom)); out["chi_" + k].append(pc)
    m.codec.chi_ref = dom.chi_ref
    o = {k: torch.cat(v) for k, v in out.items()}
    o["chi_prior"] = cref * cm[None, ..., None]
    return o


@torch.no_grad()
def bb_rmsd(pbb, x, dom):
    T, R = x.shape[:2]; mk = dom.atom_mask[None, :, :4].expand(T, R, 4).reshape(T, -1)
    Rk, tk = S.kabsch(pbb.reshape(T, -1, 3), x[:, :, :4].reshape(T, -1, 3), mk)
    P = pbb.reshape(T, -1, 3) @ Rk.transpose(-1, -2) + tk[:, None]
    return float(((((P - x[:, :, :4].reshape(T, -1, 3)) ** 2).sum(-1) * mk).sum(1) / mk.sum(1)).sqrt().mean())


def _tl(x, dom):
    Rt, tt = S.frames_from_backbone(x[..., 0, :], x[..., 1, :], x[..., 2, :])
    tl = S.to_local(x, Rt, tt)
    return S.canon_local(tl, tl[:1], dom.aatype)


@torch.no_grad()
def metrics_blocks(pred, pchi, x, tchi, dom, cls, dt, blocks):
    """residue sums pooled over contiguous blocks (index arrays into x)."""
    tot = None
    for b in blocks:
        bt = torch.as_tensor(b)
        rm = S.residue_metrics(pred[bt], pchi[bt], x[bt], tchi[bt], dom, _tl(x[bt], dom), dt)
        old = S.CLASSES; S.CLASSES = tuple(cls); sm = S.summarise(rm, cls, dom.chi_mask.numpy()); S.CLASSES = old
        if tot is None:
            tot = sm; continue
        for c in sm:
            if not sm[c].get("n_res"): continue
            for k, v in sm[c].items():
                if k in ("n_res", "T"): continue
                tot[c][k] = [p + q for p, q in zip(tot[c][k], v)] if isinstance(v, list) else tot[c][k] + v
    ci = torch.as_tensor(np.unique(np.linspace(0, len(x) - 1, min(len(x), CLASH_MAX)).astype(int)))
    tot["clash"] = L.clash_score(pred[ci], dom.aatype, dom.atom_mask, dom.chain_index)
    tot["n_blocks"] = len(blocks)
    return tot


def run_domain(d, m, raw):
    z = np.load(f"{a.data}/{d}.npz")
    dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], "cpu")
    arch = np.where(z["heldout"] == 0)[0]; ho = np.where(z["heldout"] == 1)[0]
    X = torch.as_tensor(z["coords"][arch]); XH = torch.as_tensor(z["coords"][ho])
    temps, reps = z["temp"][arch], z["replica"][arch]
    aat = dom.aatype.numpy()
    r0 = np.where((temps == 320) & (reps == 0))[0]
    tc0, _ = builder.chis_from_atom14(X[r0], dom.aatype)
    fit_buffers(dom, X[r0], builder); cmn = dom.chi_mask.numpy(); has = cmn[:, 0] > 0
    hs = S.hysteresis_states(S.ang(tc0.numpy()), aat, cmn); occ, rate = S.occ_and_rate(hs, 1.0)
    stb = has & (occ >= S.OCC_STABLE) & (rate <= S.MAX_STABLE_RATE); un = has & (occ < S.OCC_UNSTABLE)
    cls = dict(stable=stb, intermediate=has & ~stb & ~un, unstable=un, all=has)
    tchiH, _ = builder.chis_from_atom14(XH, dom.aatype); stH, _ = RC.chi_states(S.ang(tchiH.numpy()), aat, cmn)
    n_chi_res = cmn.sum(1).astype(int)
    for T in [int(t) for t in a.levels.split(",")]:
        key = f"{d}/T{T}"
        if key in raw or T > len(X): continue
        t0 = time.time()
        x = X[:T]; tchi, _ = builder.chis_from_atom14(x, dom.aatype)
        fit_buffers(dom, x, builder)
        st, ns = RC.chi_states(S.ang(tchi.numpy()), aat, cmn)
        mask = RC.build_code(st, ns, (0, 1)); coded = mask.any(1)
        j, jH = RC.joint(st, mask), RC.joint(stH, mask)
        tab, seen = RC.state_table(tchi, aat, j, coded, dom.chi_ref)
        code = RC.code_bytes(st, ns, mask, n_chi_res, seen)
        plain = lambda n: dom.chi_ref[None].expand(n, *dom.chi_ref.shape).contiguous()
        cref_code = RC.cref_frames(tab, j, coded, dom.chi_ref); cref_codeH = RC.cref_frames(tab, jH, coded, dom.chi_ref)
        fi = np.sort(np.random.default_rng(0).choice(T, min(T, FIT_MAX), replace=False))
        mods, fit_t = {}, {}
        FT.a.batch = min(48, len(fi))
        for name, cref in (("ft_mlp", plain(T)), ("code12_mlp", cref_code)):
            ff = RC.frozen_cond(m, dom, x[fi], cref[fi].contiguous())
            mods[name], sec, _ = FT.fit("mlp", m, ff, x[fi], dom); fit_t[name] = round(sec, 2); del ff
        tl_ = [(int(t), int(r)) for t, r in zip(temps[:T], reps[:T])]
        segs = sorted(set(tl_), key=tl_.index)
        r = dict(domain=d, T=T, n_res=dom.R, n_heavy=dom.n_heavy, fit_frames=len(fi), fit_seconds=fit_t, code=code,
                 class_counts={c: int(v.sum()) for c, v in cls.items()},
                 archive_segments=[dict(temp_K=t, replica=rr, frames=tl_.count((t, rr))) for t, rr in segs], same={}, heldout={})
        bl = blocks_of(T); ei = np.concatenate(bl); rel, off = [], 0
        for b in bl: rel.append(np.arange(off, off + len(b))); off += len(b)
        for reg, xe, tce, cp, cc, blk, dt in (("same", x[ei], tchi[ei], plain(len(ei)), cref_code[ei].contiguous(), rel, 1.0),
                                               ("heldout", XH, tchiH, plain(len(XH)), cref_codeH, [np.arange(len(XH))], 2.0)):
            o = forward(m, dom, xe, cp, {"ft_mlp": mods["ft_mlp"]})
            res = dict(bb_rmsd=bb_rmsd(o["bb"], xe, dom))
            res["no_finetune"] = metrics_blocks(o["base"], o["chi_base"], xe, tce, dom, cls, dt, blk)
            res["ft_mlp"] = metrics_blocks(o["ft_mlp"], o["chi_ft_mlp"], xe, tce, dom, cls, dt, blk); del o
            o = forward(m, dom, xe, cc, {"code12_mlp": mods["code12_mlp"]})
            res["code12_prior"] = metrics_blocks(o["prior"], o["chi_prior"], xe, tce, dom, cls, dt, blk)
            res["code12"] = metrics_blocks(o["base"], o["chi_base"], xe, tce, dom, cls, dt, blk)
            res["code12_mlp"] = metrics_blocks(o["code12_mlp"], o["chi_code12_mlp"], xe, tce, dom, cls, dt, blk); del o
            r[reg] = res
        r["seconds"] = round(time.time() - t0, 1); r["eval_frames_same"] = int(len(ei))
        raw[key] = r
        json.dump(raw, open(a.raw + ".tmp", "w")); os.replace(a.raw + ".tmp", a.raw)
        f = lambda reg, mm: math.sqrt(r[reg][mm]["all"]["sc_err"] / r[reg][mm]["all"]["n_sc"])
        print(time.strftime("%H:%M:%S"), key, f"{r['seconds']}s coded {code['n_coded_residues']} {code['zlib_bytes_per_frame']:.1f}B/fr bb {r['same']['bb_rmsd']:.3f}/{r['heldout']['bb_rmsd']:.3f}",
              " ".join(f"{reg}:" + ",".join(f"{mm} {f(reg, mm):.2f}" for mm in ("no_finetune", "ft_mlp", "code12_prior", "code12_mlp")) for reg in ("same", "heldout")), flush=True)


def run():
    m = FT.load_main()
    raw = json.load(open(a.raw)) if os.path.exists(a.raw) else {}
    doms = json.load(open("results_gpu/frames/selected_domains.json"))
    only = set(a.only.split(",")) - {""}
    for d in doms:
        if only and d not in only: continue
        if not os.path.exists(f"{a.data}/{d}.npz"): print("missing", d, flush=True); continue
        run_domain(d, m, raw)
    print("done", flush=True)


if __name__ == "__main__" and not a.summarize:
    run()


# ------------------------------------------------------------------ summary + figure
LABEL = dict(no_finetune="no fine-tune", ft_mlp="fine-tune MLP", code12_prior="χ1+χ2 codes, state prior",
             code12="χ1+χ2 codes + model", code12_mlp="χ1+χ2 codes + model + MLP")
COL = dict(no_finetune="#333333", ft_mlp="tab:orange", code12_prior="tab:olive", code12="tab:green", code12_mlp="tab:red")


def stored_bytes(r, meth):
    R, T = r["n_res"], r["T"]; s = T * (8 * R + 28) + 80 * R
    if meth.startswith("code"):
        c = r["code"]; s += c["zlib_bytes_total"] + c["mask_bytes"] + c["state_table_bytes"]
    if meth.endswith("mlp"): s += 5960
    return s


def pool(rows, reg, meth):
    out = {}
    for c in ("all", "stable", "intermediate", "unstable"):
        xs = [r[reg][meth][c] for r in rows if r[reg][meth][c].get("n_res")]
        if not xs: out[c] = None; continue
        sm = lambda k: sum(x[k] for x in xs)
        out[c] = dict(sc_rmsd=math.sqrt(sm("sc_err") / sm("n_sc")), chi1_within40=sm("chi1_ok") / sum(x["n_chi"][0] for x in xs),
                      transition_rate_ratio=sm("rate_p") / max(sm("rate_t"), 1e-12), sc_rmsf_ratio=sm("rmsf_p") / sm("rmsf_t"),
                      chi_mae=[sum(x["chi_err"][k] for x in xs) / max(sum(x["n_chi"][k] for x in xs), 1) for k in range(4)])
    out["clash"] = float(np.mean([r[reg][meth]["clash"] for r in rows]))
    return out


def summarize():
    raw = json.load(open(a.raw)); levels = [int(t) for t in a.levels.split(",")]
    doms = sorted({r["domain"] for r in raw.values()})
    summ = {}
    for T in levels:
        rows = [r for r in raw.values() if r["T"] == T]
        if not rows: continue
        e = dict(n_domains=len(rows), bb_rmsd=dict(same=float(np.mean([r["same"]["bb_rmsd"] for r in rows])),
                                                    heldout=float(np.mean([r["heldout"]["bb_rmsd"] for r in rows]))),
                 code_mean={k: float(np.mean([r["code"][k] for r in rows])) for k in rows[0]["code"]},
                 eval_frames_same=rows[0]["eval_frames_same"], fit_frames=rows[0]["fit_frames"],
                 fit_seconds_median={k: float(np.median([r["fit_seconds"][k] for r in rows])) for k in rows[0]["fit_seconds"]},
                 archive_segments_example=rows[0]["archive_segments"], methods={})
        for mm in METHODS:
            raw_b = [12 * r["n_heavy"] * T for r in rows]; st = [stored_bytes(r, mm) for r in rows]
            e["methods"][mm] = dict(ratio_mean=float(np.mean([x / y for x, y in zip(raw_b, st)])), ratio_pooled=sum(raw_b) / sum(st),
                                    extra_bytes_mean=float(np.mean([stored_bytes(r, mm) - stored_bytes(r, "no_finetune") for r in rows])),
                                    same=pool(rows, "same", mm), heldout=pool(rows, "heldout", mm))
        summ[T] = e
    per = {k: dict(domain=r["domain"], T=r["T"], n_res=r["n_res"], bb_same=r["same"]["bb_rmsd"], bb_heldout=r["heldout"]["bb_rmsd"],
                   ratio={mm: 12 * r["n_heavy"] * r["T"] / stored_bytes(r, mm) for mm in METHODS},
                   sc={reg: {mm: math.sqrt(r[reg][mm]["all"]["sc_err"] / r[reg][mm]["all"]["n_sc"]) for mm in METHODS} for reg in ("same", "heldout")},
                   code=r["code"], archive_segments=r["archive_segments"], fit_seconds=r["fit_seconds"], seconds=r["seconds"]) for k, r in raw.items()}
    out = dict(description=__doc__, domains=doms, levels=levels, summary=summ, per_domain_level=per)
    json.dump(out, open(a.out, "w"), indent=1)
    figure(summ)


def figure(summ):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ts = sorted(summ, key=int); x = [int(t) for t in Ts]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for mm in METHODS:
        ax[0].plot(x, [summ[t]["methods"][mm]["ratio_mean"] for t in Ts], "o-", c=COL[mm], label=LABEL[mm])
        for reg, ls in (("same", "-"), ("heldout", "--")):
            ax[1].plot(x, [summ[t]["methods"][mm][reg]["all"]["sc_rmsd"] for t in Ts], ls, marker="o", ms=3, c=COL[mm],
                       label=LABEL[mm] if reg == "same" else None)
            ax[2].plot(x, [summ[t]["methods"][mm][reg]["unstable"]["sc_rmsd"] for t in Ts], ls, marker="o", ms=3, c=COL[mm])
    for reg, ls in (("same", "-"), ("heldout", "--")):
        ax[1].plot(x, [summ[t]["bb_rmsd"][reg] for t in Ts], ls, marker="s", ms=3, c="tab:blue", label="backbone (all methods)" if reg == "same" else None)
    ax[0].set(xscale="log", xlabel="archived frames T", ylabel="compression ratio (×, all per-protein bytes counted)", title="A  Compression")
    ax[1].set(xscale="log", xlabel="archived frames T", ylabel="RMSD (Å)", title="B  Side-chain RMSD, all residues (solid same frames, dashed held-out)", ylim=(0, None))
    ax[2].set(xscale="log", xlabel="archived frames T", ylabel="side-chain RMSD (Å)", title="C  Unstable residues", ylim=(0, None))
    for A_ in ax:
        A_.grid(alpha=0.3); A_.title.set_fontsize(9)
        A_.axvspan(2000, 12000, color="grey", alpha=0.08)
    ax[2].text(1900, 0.1, "shaded: archive adds\n348 to 450 K frames", fontsize=7.5, color="grey", ha="right")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig("report/gpu_figs/fig_frames_scaling.png", dpi=200)
    import shutil; shutil.copy("report/gpu_figs/fig_frames_scaling.png", "results_gpu/fig_frames_scaling.png")


if a.summarize:
    summarize()
