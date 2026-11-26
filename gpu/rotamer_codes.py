#!/usr/bin/env python
"""Per-frame rotamer-state coding for flipping side chains, on top of the frozen E2E S model with the chi prior.

Compression time (original coordinates available):
  * per chi, rotamer bins with core-to-core hysteresis (+-CORE deg core):
      chi1: g+ (+60), t (180), g- (-60); PRO: two puckers (sign, |chi1| >= 10 deg core).
      chi2: sp3 types (ARG GLN GLU ILE LEU LYS MET): g+/t/g-; HIS TRP ASN: two bins (+90 / -90, +-50 deg core);
            pi-periodic chi2 (PHE TYR ASP): not coded (ring/carboxylate flips are symmetric).
  * a chi is "flipping" if its state changes at least once into a state that then persists >= DWELL frames.
  * stored once per protein: a bit mask of flipping chis (R bits per coded chi slot) and a per-state prior table
    (mean of every chi of the residue over the frames in each observed joint state, fp16 angle per chi).
  * stored per frame: the state of every flipping chi. Sizes reported: packed bits (2 bits for 3 states, 1 for 2),
    order-0 entropy bound, and the real size of zlib -9 on the residue-major uint8 state array (used for ratios).
Decoding: the chi prior of a coded residue in a frame is replaced by its per-state prior; the frozen model then
predicts the offset within the state. Variants: code1 (chi1 states only), code12 (chi1 + chi2), each with the
frozen model or with the state prior alone, and code12 + residual MLP refitted with state-conditioned priors.
Regimes as in finetune_traj.py: same (mask, table, MLP from all frames; evaluate same frames) and heldout
(mask, table, buffers and MLP from the first 80%; stored states of the last 20% used to decode the last 20%).
Raw sums per system: results_gpu/eval/rotamer_code_raw.json (resumable); --summarize writes rotamer_code_results.json
and merges prior_alone / no_finetune / mlp from finetune_raw.json (same frames, same protocol).
"""
import argparse, glob, json, math, os, sys, time, zlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0]]
import finetune_traj as FT  # noqa: E402  (defaults: 400 steps, batch 48, same MLP and loss)
sys.argv = _argv
from common import Domain, bind, fit_buffers, load_npz_domain  # noqa: E402
from md_compress import residue_constants as rc  # noqa: E402
from md_compress import losses as L  # noqa: E402
import stability_lib as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="results_gpu/eval/rotamer_code_raw.json")
ap.add_argument("--out", default="results_gpu/eval/rotamer_code_results.json")
ap.add_argument("--only", default="")
ap.add_argument("--summarize", action="store_true")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
CORE, CORE2, DWELL = 40.0, 50.0, 2
SP3 = {rc.RESTYPE_ORDER[r] for r in ("ARG", "GLN", "GLU", "ILE", "LEU", "LYS", "MET")}
TWO = {rc.RESTYPE_ORDER[r] for r in ("HIS", "TRP", "ASN")}
PRO = rc.RESTYPE_ORDER["PRO"]
ARG, GLN = rc.RESTYPE_ORDER["ARG"], rc.RESTYPE_ORDER["GLN"]
SP3_34 = {2: {rc.RESTYPE_ORDER[r] for r in ("ARG", "LYS", "MET")}, 3: {rc.RESTYPE_ORDER["LYS"]}}
builder = FT.builder


def _bins(x, centers, core):
    d = np.stack([np.abs((x - c + 180) % 360 - 180) for c in centers], -1)
    return np.where(d.min(-1) <= core, d.argmin(-1), -1), d.argmin(-1), len(centers)


def chi_states(chi_deg, aat, cm, nk=2):
    """chi_deg [T,R,4] -> states [T,R,nk] (-1 where not coded) and n_states [R,nk].
    nk=2: chi1, chi2 (section 13). nk=4 adds chi3/chi4 (section 15):
      sp3 chi3: ARG LYS MET (g+/t/g-), sp3 chi4: LYS (g+/t/g-);
      ARG chi4 (CG-CD-NE-CZ, rotation about CD-NE): three bins at +90 / 180 / -90 (core +-40 deg);
      GLN chi3 (amide, not symmetric): two bins at +90 / -90 (core +-50 deg);
      GLU chi3 (carboxylate, pi-periodic, a single broad well after symmetry): not coded."""
    T, R = chi_deg.shape[:2]
    st = -np.ones((T, R, nk), int); ns = np.zeros((R, nk), int)
    for k in range(nk):
        x = chi_deg[:, :, k]
        for r in range(R):
            if cm[r, k] == 0: continue
            if k >= 2:
                if aat[r] in SP3_34[k]:
                    core, near, n = _bins(x[:, r], (60, 180, -60), CORE)
                elif k == 3 and aat[r] == ARG:
                    core, near, n = _bins(x[:, r], (90, 180, -90), CORE)
                elif k == 2 and aat[r] == GLN:
                    core, near, n = _bins(x[:, r], (90, -90), CORE2)
                else:
                    continue
            elif k == 0 and aat[r] == PRO:
                core = np.where(np.abs(x[:, r]) >= 10, (x[:, r] < 0).astype(int), -1); near = (x[:, r] < 0).astype(int); n = 2
            elif k == 0 or aat[r] in SP3:
                d = np.stack([np.abs((x[:, r] - c + 180) % 360 - 180) for c in (60, 180, -60)], -1)
                core = np.where(d.min(-1) <= CORE, d.argmin(-1), -1); near = d.argmin(-1); n = 3
            elif aat[r] in TWO:
                d = np.stack([np.abs((x[:, r] - c + 180) % 360 - 180) for c in (90, -90)], -1)
                core = np.where(d.min(-1) <= CORE2, d.argmin(-1), -1); near = d.argmin(-1); n = 2
            else:
                continue
            s = np.empty(T, int); s[0] = core[0] if core[0] >= 0 else near[0]
            for t in range(1, T):
                s[t] = core[t] if core[t] >= 0 else s[t - 1]
            st[:, r, k] = s; ns[r, k] = n
    return st, ns


def flipping(st):
    """[T] states -> True if some change leads into a state held >= DWELL frames."""
    ch = np.where(st[1:] != st[:-1])[0] + 1
    for c in ch:
        if c + DWELL <= len(st) and np.all(st[c:c + DWELL] == st[c]):
            return True
    return False


def build_code(st, ns, chis_k):
    """mask [R,K] of coded chis among chis_k (e.g. (0,), (0,1) or (0,1,2,3)); K = st.shape[2]."""
    R = st.shape[1]; mask = np.zeros((R, st.shape[2]), bool)
    for r in range(R):
        for k in chis_k:
            if ns[r, k] and flipping(st[:, r, k]): mask[r, k] = True
    return mask


def joint(st, mask):
    """joint coded state per frame/residue (0 for uncoded residues)."""
    j = np.zeros(st.shape[:2], int)
    for k in range(mask.shape[1]):
        j = j * 3 + np.where(mask[None, :, k], st[:, :, k], 0)
    return j


def state_table(tchi_sc, aat, j, coded, glob_ref, n_joint=9):
    """per (residue, joint state) mean chi (sin,cos) [R,n_joint,4,2] from the given frames; unseen -> global prior.
    n_joint = 3**K (9 for chi1+chi2, 81 for chi1..chi4)."""
    tab = glob_ref[:, None].repeat(1, n_joint, 1, 1).clone(); seen = np.zeros((len(aat), n_joint), bool)
    can = L.canonicalize_chi(tchi_sc, torch.as_tensor(aat))
    for r in np.where(coded)[0]:
        for s in np.unique(j[:, r]):
            sel = torch.as_tensor(j[:, r] == s)
            tab[r, s] = torch.nn.functional.normalize(can[sel, r].mean(0), dim=-1); seen[r, s] = True
    return tab * (glob_ref.abs().sum(-1, keepdim=True) > 0)[:, None], seen


def cref_frames(tab, j, coded, glob_ref):
    T = j.shape[0]
    c = glob_ref[None].repeat(T, 1, 1, 1).clone()
    for r in np.where(coded)[0]:
        c[:, r] = tab[r, torch.as_tensor(j[:, r])]
    return c


@torch.no_grad()
def frozen_cond(m, dom, x, cref):
    bind(m, dom); BB, H, C, RR, TT = [], [], [], [], []
    for s in range(0, len(x), 25):
        m.codec.chi_ref = cref[s:s + 25]
        bb_al, R_, t_ = m.codec.align(x[s:s + 25, :, :4])
        z = m.codec.encode(bb_al, dom.aatype).half().float()
        bb, outs = m.decode_full(z, dom.aatype, dom.chain_index, dom.chain_break)
        BB.append(bb); H.append(outs[-1]["h"]); C.append(outs[-1]["chi_sc"]); RR.append(R_); TT.append(t_)
    m.codec.chi_ref = dom.chi_ref
    return dict(bb=torch.cat(BB), h=torch.cat(H), base=torch.cat(C), R=torch.cat(RR), t=torch.cat(TT))


def code_bytes(st, ns, mask, n_chi_res, seen):
    T = st.shape[0]
    cols = [(r, k) for r in range(st.shape[1]) for k in range(mask.shape[1]) if mask[r, k]]
    packed_bits = sum(2 if ns[r, k] == 3 else 1 for r, k in cols)
    if cols:
        arr = np.stack([st[:, r, k] for r, k in cols], 0).astype(np.uint8)  # residue-major
        z = len(zlib.compress(arr.tobytes(), 9))
        ent = sum(-(np.bincount(arr[i]) / T)[np.bincount(arr[i]) > 0] @ np.log2((np.bincount(arr[i]) / T)[np.bincount(arr[i]) > 0]) for i in range(len(cols))) * T / 8
    else:
        z, ent = 0, 0.0
    coded = mask.any(1)
    table_bytes = int(sum(2 * n_chi_res[r] * seen[r].sum() for r in np.where(coded)[0]))
    mask_bytes = mask.shape[1] * math.ceil(st.shape[1] / 8) if len(mask) else 0
    return dict(n_coded_chis=len(cols), n_coded_residues=int(coded.sum()), packed_bytes_per_frame=packed_bits / 8,
                zlib_bytes_total=z, zlib_bytes_per_frame=z / T, entropy0_bytes_per_frame=float(ent) / T,
                mask_bytes=mask_bytes, state_table_bytes=table_bytes)


VARS = {"code1": (0,), "code12": (0, 1)}


def evaluate(key, dom, x, dt, meta, m, raw):
    t0 = time.time()
    aat = dom.aatype.numpy()
    tchi_all, _ = builder.chis_from_atom14(x, dom.aatype)
    fit_buffers(dom, x, builder); cmn = dom.chi_mask.numpy(); has = cmn[:, 0] > 0
    tc = S.ang(tchi_all.numpy()); hs = S.hysteresis_states(tc, aat, cmn); occ, rate = S.occ_and_rate(hs, dt)
    stb = has & (occ >= S.OCC_STABLE) & (rate <= S.MAX_STABLE_RATE); un = has & (occ < S.OCC_UNSTABLE)
    cls = dict(stable=stb, intermediate=has & ~stb & ~un, unstable=un, all=has)
    st_all, ns = chi_states(tc, aat, cmn)
    T = len(x); nf = int(round(FT.FIT_FRAC * T)); n_chi_res = cmn.sum(1).astype(int)
    r = dict(meta, n_res=dom.R, n_heavy=dom.n_heavy, n_frames=T, dt_ns=dt, n_fit_heldout=nf,
             class_counts={c: int(v.sum()) for c, v in cls.items()}, code={}, regimes={})
    for regime, fs, es in (("same", slice(0, T), slice(0, T)), ("heldout", slice(0, nf), slice(nf, T))):
        xf, xe = x[fs], x[es]
        fit_buffers(dom, xf, builder)
        tf, _ = builder.chis_from_atom14(xf, dom.aatype); te, _ = builder.chis_from_atom14(xe, dom.aatype)
        res = {}
        for v, ks in VARS.items():
            mask = build_code(st_all[fs], ns, ks); coded = mask.any(1)
            jf, je = joint(st_all[fs], mask), joint(st_all[es], mask)
            tab, seen = state_table(tf, aat, jf, coded, dom.chi_ref)
            if regime == "same":
                r["code"][v] = code_bytes(st_all, ns, mask, n_chi_res, seen)
                r["code"][v]["coded_unstable"] = int((coded & un).sum()); r["code"][v]["coded_stable"] = int((coded & stb).sum())
            ce = cref_frames(tab, je, coded, dom.chi_ref)
            pri = ce * dom.chi_mask[None, ..., None]
            fe = frozen_cond(m, dom, xe, ce)
            res[f"{v}_prior"] = FT.metrics(FT.build(m, fe, pri, dom), pri, xe, te, dom, cls, dt)
            res[v] = FT.metrics(FT.build(m, fe, fe["base"], dom), fe["base"], xe, te, dom, cls, dt)
            if v == "code12":
                ff = fe if regime == "same" else frozen_cond(m, dom, xf, cref_frames(tab, jf, coded, dom.chi_ref))
                mod, sec, hist = FT.fit("mlp", m, ff, xf, dom)
                with torch.no_grad():
                    pc = mod(fe["h"], fe["base"], None, dom.chi_mask.float())
                res["code12_mlp"] = FT.metrics(FT.build(m, fe, pc, dom), pc, xe, te, dom, cls, dt)
                res["code12_mlp"].update(fit_seconds=round(sec, 2), n_params=2980, extra_bytes_fp16=5960, loss_hist=hist)
        r["regimes"][regime] = res
    r["seconds"] = round(time.time() - t0, 1)
    raw[key] = r
    json.dump(raw, open(a.raw + ".tmp", "w")); os.replace(a.raw + ".tmp", a.raw)
    f = lambda reg, mm: math.sqrt(r["regimes"][reg][mm]["all"]["sc_err"] / r["regimes"][reg][mm]["all"]["n_sc"])
    print(time.strftime("%H:%M:%S"), key, T, f"{r['seconds']}s", "coded", r["code"]["code12"]["n_coded_residues"],
          f"{r['code']['code12']['zlib_bytes_per_frame']:.1f}B/fr",
          " ".join(f"{reg}:" + ",".join(f"{mm} {f(reg, mm):.2f}" for mm in ("code1", "code12", "code12_mlp")) for reg in ("same", "heldout")), flush=True)


def run():
    m = FT.load_main()
    raw = json.load(open(a.raw)) if os.path.exists(a.raw) else {}
    only = set(a.only.split(",")) - {""}
    man = json.load(open(f"{FT.a.data}/manifest.json"))
    for d in sorted(d for d, v in man["domains"].items() if v["split"] == "test"):
        key = f"mdcath/{d}"
        if key in raw or (only and d not in only): continue
        z = np.load(f"{FT.a.data}/domains/{d}.npz"); rep0 = np.where(z["replica"] == 0)[0]
        x = torch.as_tensor(z["coords"][rep0][:FT.a.max_frames].astype(np.float32))
        dom = Domain(d, z["aatype"], z["atom_mask"], z["chain_index"], z["chain_break"], z["n_heavy"], "cpu")
        evaluate(key, dom, x, 2.0, dict(set="mdcath_test"), m, raw)
    for p in sorted(glob.glob(f"{FT.a.external}/*.npz")):
        name = os.path.basename(p)[:-4]; key = f"external/{name}"
        if key in raw or (only and name not in only): continue
        dom, z = load_npz_domain(p, "cpu"); x = torch.as_tensor(dom.coords)
        evaluate(key, dom, x, 1.0, dict(set="external"), m, raw)
    print("done", flush=True)


if __name__ == "__main__" and not a.summarize:
    run()


# ------------------------------------------------------------------ summary + figure
METHODS = ("prior_alone", "no_finetune", "ft_mlp", "code1_prior", "code1", "code12_prior", "code12", "code12_mlp")
LABEL = dict(prior_alone="prior alone", no_finetune="no fine-tune", ft_mlp="fine-tune (b) MLP",
             code1_prior="χ1 states + state prior only", code1="χ1 states + model",
             code12_prior="χ1+χ2 states + state prior only", code12="χ1+χ2 states + model",
             code12_mlp="χ1+χ2 states + model + MLP")


def cost(r, meth):
    """(extra bytes per frame, extra one-off bytes) for a method on system r (archive = same-frame regime)."""
    pf, one = 0.0, 0
    if meth.startswith("code"):
        c = r["code"]["code1" if meth.startswith("code1_") or meth == "code1" else "code12"]
        pf += c["zlib_bytes_per_frame"]; one += c["mask_bytes"] + c["state_table_bytes"]
    if meth.endswith("mlp"):
        one += 5960
    return pf, one


def ratios(r, meth):
    raw = 12 * r["n_heavy"]; pf0 = 8 * r["n_res"] + 28; epf, eone = cost(r, meth)
    pf = pf0 + epf; one = 80 * r["n_res"] + eone
    am = lambda T: raw * T / (pf * T + one)
    return dict(bytes_per_frame=pf, extra_bytes_per_frame=epf, one_off_bytes=one, extra_one_off_bytes=eone,
                per_frame=raw / pf, T100=am(100), T1000=am(1000), T_actual=am(r["n_frames"]))


def pool(rows, meth, reg):
    out = {}
    for c in ("all", "stable", "intermediate", "unstable"):
        xs = [r["regimes"][reg][meth][c] for r in rows if r["regimes"][reg][meth][c].get("n_res")]
        sm = lambda k: sum(x[k] for x in xs)
        out[c] = dict(n_res=sm("n_res"), sc_rmsd=math.sqrt(sm("sc_err") / sm("n_sc")),
                      chi_mae=[sum(x["chi_err"][k] for x in xs) / max(sum(x["n_chi"][k] for x in xs), 1) for k in range(4)],
                      chi1_within40=sm("chi1_ok") / sum(x["n_chi"][0] for x in xs),
                      rotamer_state_acc=sm("state_ok") / sum(x["n_chi"][0] for x in xs),
                      sc_rmsf_ratio=sm("rmsf_p") / sm("rmsf_t"), chi1_pop_jsd=sm("jsd") / sm("n_res"),
                      transition_rate_ratio=sm("rate_p") / max(sm("rate_t"), 1e-12)) if xs else None
    out["clash_per_1000_atoms"] = float(np.mean([r["regimes"][reg][meth]["clash"] for r in rows]))
    rr = [ratios(r, meth) for r in rows]
    out["compression_mean"] = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
    if meth.endswith("mlp"):
        out["fit_seconds_median"] = float(np.median([r["regimes"][reg][meth]["fit_seconds"] for r in rows]))
    return out


def summarize():
    raw = json.load(open(a.raw)); ft = json.load(open("results_gpu/eval/finetune_raw.json"))
    for k, r in raw.items():  # merge baselines from the fine-tune study (same frames, classes and protocol)
        for reg in ("same", "heldout"):
            for src, dst in (("prior_alone", "prior_alone"), ("no_finetune", "no_finetune"), ("mlp", "ft_mlp")):
                r["regimes"][reg][dst] = ft[k]["regimes"][reg][src]
    sets = {s: [r for r in raw.values() if r["set"] == s] for s in ("mdcath_test", "external")}
    summ = {}
    for s, rows in sets.items():
        codes = {v: {q: float(np.mean([r["code"][v][q] for r in rows])) for q in rows[0]["code"][v]} for v in VARS}
        frac = {v: dict(unstable_coded=sum(r["code"][v]["coded_unstable"] for r in rows) / sum(r["class_counts"]["unstable"] for r in rows),
                        stable_coded=sum(r["code"][v]["coded_stable"] for r in rows) / sum(r["class_counts"]["stable"] for r in rows),
                        residues_coded=sum(r["code"][v]["n_coded_residues"] for r in rows) / sum(r["class_counts"]["all"] for r in rows))
                for v in VARS}
        summ[s] = dict(n_systems=len(rows), code_sizes_mean=codes, coded_fractions=frac,
                       regimes={reg: {mm: pool(rows, mm, reg) for mm in METHODS} for reg in ("same", "heldout")})
    per = {k: dict(set=r["set"], n_res=r["n_res"], n_frames=r["n_frames"], code=r["code"],
                   compression={mm: ratios(r, mm) for mm in METHODS},
                   sc_rmsd={reg: {mm: {c: (math.sqrt(r["regimes"][reg][mm][c]["sc_err"] / r["regimes"][reg][mm][c]["n_sc"])
                                          if r["regimes"][reg][mm][c].get("n_res") else None) for c in ("all", "stable", "intermediate", "unstable")}
                                  for mm in METHODS} for reg in ("same", "heldout")}) for k, r in raw.items()}
    out = dict(description=__doc__, main_model="e2e_S_prior (frozen)", dwell_frames=DWELL, core_deg=dict(chi1=CORE, chi2_sp3=CORE, chi2_two_state=CORE2),
               compression="raw 12 B per heavy atom per frame; per frame 8 B/res + 28 B + zlib state codes (total / T); one-off 80 B/res + "
                           "flip mask + per-state prior table (+ 5,960 B for an MLP); shared weights not counted; code rate per frame "
                           "taken from the actual trajectory for T = 100 / 1000",
               baselines_from="results_gpu/eval/finetune_raw.json", summary=summ, per_system=per)
    json.dump(out, open(a.out, "w"), indent=1)
    figure(summ)


def figure(summ):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = dict(prior_alone="#aaaaaa", no_finetune="#333333", ft_mlp="tab:orange", code1_prior="#9ecae1", code1="tab:blue",
               code12_prior="#a1d99b", code12="tab:green", code12_mlp="tab:red")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    for i, s in enumerate(("mdcath_test", "external")):
        d = summ[s]["regimes"]; A = ax[i, 0]
        for mm in METHODS:
            x = d["same"][mm]["compression_mean"]["T1000"]
            A.scatter(x, d["same"][mm]["all"]["sc_rmsd"], s=70, c=col[mm], label=LABEL[mm], zorder=3)
            A.scatter(x, d["heldout"][mm]["all"]["sc_rmsd"], s=70, facecolors="none", edgecolors=col[mm], zorder=3)
            A.plot([x, x], [d["same"][mm]["all"]["sc_rmsd"], d["heldout"][mm]["all"]["sc_rmsd"]], c=col[mm], lw=0.8)
        A.set(xlabel="compression ratio at T = 1000 (×, all extra bytes counted)", ylabel="side-chain RMSD, all residues (Å)",
              title=f"{'mdCATH test (50)' if s == 'mdcath_test' else 'external (8)'}: error vs compression (filled same frames, open held-out)")
        A.title.set_fontsize(9); A.grid(alpha=0.3)
        if i == 0: A.legend(fontsize=7.5, loc="upper left")
        B = ax[i, 1]; w = 0.1
        for k, mm in enumerate(METHODS):
            for j, c in enumerate(("stable", "intermediate", "unstable")):
                y = d["same"][mm][c]["sc_rmsd"]; yh = d["heldout"][mm][c]["sc_rmsd"]
                B.bar(j + (k - 3.5) * w, y, w, color=col[mm], label=LABEL[mm] if j == 0 and i == 0 else None)
                B.plot([j + (k - 3.5) * w] * 2, [y, yh], c="k", lw=0.8); B.scatter(j + (k - 3.5) * w, yh, s=8, c="k", zorder=3)
        B.set_xticks(range(3)); B.set_xticklabels(["stable", "intermediate", "unstable"])
        B.set(ylabel="side-chain RMSD (Å)", title="by stability class (bars same frames, dots held-out)"); B.title.set_fontsize(9)
    fig.tight_layout(); fig.savefig("report/gpu_figs/fig_rotamer_codes.png", dpi=200)
    import shutil; shutil.copy("report/gpu_figs/fig_rotamer_codes.png", "results_gpu/fig_rotamer_codes.png")


if a.summarize:
    summarize()
