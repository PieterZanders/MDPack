#!/usr/bin/env python
"""AEZ-12: section 4.7 (predictor versus coder) from results/predictor_analysis_*.json, the main entropy results and the AEZ-11 results.
Writes entropy/predictor_section.md, figs/fig_residual_hist.png, figs/fig_entropy_vs_bound.png, results/predictor_summary.json."""
import glob, json, os, sys
import numpy as np
BEN = "/workspace/md_compress/benchmark"; E = f"{BEN}/entropy"; R = f"{BEN}/results"
src = sys.argv[1] if len(sys.argv) > 1 else f"{E}/results/predictor_analysis_*.json"
PA = {}
for f in sorted(glob.glob(src)): PA.update({k: v for k, v in json.load(open(f)).items() if "error" not in v})
DOMS = [d for d in json.load(open("/workspace/md_compress/results_gpu/frames/selected_domains.json")) if d in PA]
MDDB = json.load(open("/workspace/bench_data/mddb_meta.json"))
rows = {}
for f in sorted(glob.glob(f"{R}/*_raw*.json")) + sorted(glob.glob("/workspace/md_compress/results/ours_raw_*.json")): rows.update(json.load(open(f)))
ent = {}
for f in sorted(glob.glob(f"{E}/results/entropy_raw_*.json")): ent.update(json.load(open(f)))
BOUNDS = ["0.5", "0.3", "0.2", "0.1", "0.005"]; BL = lambda b: "lossless" if b == "0.005" else f"{b} Å"
PRED = ["none", "temporal", "MDCompress level 2", "MDCompress level 4", "mean structure", "hybrid", "hybrid + backbone AE"]
LAB = {"none": "none (raw grid)", "temporal": "previous frame", "MDCompress level 2": "MDCompress level 2 style", "MDCompress level 4": "MDCompress level 4 style",
       "mean structure": "mean structure + rigid fit", "hybrid": "hybrid (packer)", "hybrid + backbone AE": "hybrid + backbone autoencoder"}
f2 = lambda v, n=2: "n/a" if v is None or not np.isfinite(v) else f"{v:.{n}f}"
mean = lambda xs: float(np.mean(xs)) if len(xs) else np.nan
def g(d, b, p, *path):
    v = PA[d]["bounds"][b][p]
    for k in path: v = v.get(k) if isinstance(v, dict) else None
    return v
def M(b, p, *path):
    xs = [g(d, b, p, *path) for d in DOMS]; xs = [x for x in xs if x is not None]
    return mean(xs) if len(xs) == len(DOMS) else np.nan
L = []; A = L.append; SUM = {}; CMP = {}; GRP = {}; ATT = {}
n = len(DOMS)
A("### 4.7 What the AI predictor buys: residual entropy against baseline predictors at the same coder and bound\n")
A("An error bounded code stores two things: side information that lets the decoder form a prediction, and the quantised difference between the data and that "
  "prediction. With the quantiser fixed (step twice the bound, or the 0.01 Å source grid for lossless) the residual stream costs about the entropy of the "
  "residual integers, times the number of coordinates. A predictor never removes the residual; it narrows its distribution, and a narrower distribution has "
  "lower entropy. If the continuous residual has spread σ and the step is Δ, the entropy per coordinate is roughly log2(σ/Δ) plus a constant while σ is well "
  "above Δ, so a predictor that shrinks σ by a factor k saves about log2(k) bits per coordinate at every bound. At a loose bound those bits are a large "
  "fraction of a small total (residuals collapse onto zero); at a tight bound the same saving is a small fraction of a large total, and what is left is decided "
  "by how much of the remaining structure the coder can exploit. This section measures each step of that argument on the mdCATH test domains "
  f"(T = 1,000; {n} of 16 domains; means over domains), using the same quantiser, the same coders and the same bound for every predictor.\n")
A("Predictors compared (all produce integers a decoder can invert frame by frame): none (the raw coordinate grid); the previous frame; two MDCompress style "
  "predictors on the raw grid, reproduced from its source (level 2: previous position moved by the displacement of an already coded neighbour; level 4: the "
  "local frame of three already coded reference atoms carried from the previous frame to this one); the mean structure with a per frame rigid fit; the hybrid "
  "(backbone against the mean structure, side chains against the packer network's prediction from the decoded backbone); and the hybrid with the backbone "
  "predicted by the learned autoencoder. Statistics are over frames 2 to T (frame 1 has no history; MDCompress codes it as an anchor); coded sizes include "
  "every frame. Coders: the lzma layout search (three layouts; note that two of them take a frame difference, i.e. contain their own previous frame predictor), "
  "the new context modelled coder with its internal prediction switched off ('AC, no prediction': only the adaptive probability model), and the full new coder "
  "(which adds its geometric and temporal prediction on top of the given residuals).\n")
# ---------------- per predictor sums (reference values)
def tot_b(d, b, p, key):
    bb, sc = g(d, b, p, key + "_bb"), g(d, b, p, key + "_sc")
    if bb is None and key == "ac": bb, sc = g(d, b, "none", "ac_bb"), g(d, b, "none", "ac_sc")   # raw grid predictors: full AC = AC on the raw grid
    return bb + sc
def fixed(d, b, p): return PA[d]["fixed_bytes"][p] + (g(d, b, p, "swap_bytes") or 0)
def Hall(d, b, p, q):
    nb, na = PA[d]["n_bb"], PA[d]["n_atoms"]; return (g(d, b, p, "stats_bb", q) * nb + g(d, b, p, "stats_sc", q) * (na - nb)) / na
for b in BOUNDS:
    for p in PRED:
        SUM.setdefault(b, {})[p] = dict(rms_bb=M(b, p, "stats_bb", "rms_A"), rms_sc=M(b, p, "stats_sc", "rms_A"), H0_bb=M(b, p, "stats_bb", "H0"), H0_sc=M(b, p, "stats_sc", "H0"),
            Hctx_bb=M(b, p, "stats_bb", "Hctx"), Hctx_sc=M(b, p, "stats_sc", "Hctx"), frac0_sc=M(b, p, "stats_sc", "frac_zero"),
            H0_all=mean([Hall(d, b, p, "H0") for d in DOMS]), Hctx_all=mean([Hall(d, b, p, "Hctx") for d in DOMS]),
            ideal_kB=mean([(g(d, b, p, "stats_bb", "H0") * PA[d]["n_bb"] + g(d, b, p, "stats_sc", "H0") * (PA[d]["n_atoms"] - PA[d]["n_bb"])) * 3 * PA[d]["T"] / 8 for d in DOMS]) / 1e3,
            lzma_kB=mean([tot_b(d, b, p, "lzma") for d in DOMS]) / 1e3, ac0_kB=mean([tot_b(d, b, p, "ac0") for d in DOMS]) / 1e3, ac_kB=mean([tot_b(d, b, p, "ac") for d in DOMS]) / 1e3,
            fixed_kB=mean([fixed(d, b, p) for d in DOMS]) / 1e3,
            ratio_lzma=mean([PA[d]["raw_bytes"] / (fixed(d, b, p) + tot_b(d, b, p, "lzma")) for d in DOMS]),
            ratio_ac=mean([PA[d]["raw_bytes"] / (fixed(d, b, p) + tot_b(d, b, p, "ac")) for d in DOMS]))
AI = ["hybrid", "hybrid + backbone AE"]; BASE = ["none", "mean structure", "MDCompress level 2", "MDCompress level 4"]
pc = lambda d_, base: f"{d_:+.2f} ({100 * d_ / base:+.0f} %)" if np.isfinite(d_) and base else "n/a"
pk = lambda d_, base: f"{d_:+.1f} ({100 * d_ / base:+.0f} %)" if np.isfinite(d_) and base else "n/a"
# ---------------- 4.7.1 AI predictor versus baselines
A("#### 4.7.1 The AI predictor against each baseline predictor\n")
A("Each row compares an AI predictor (the trained packer inside the hybrid, or the hybrid with the learned backbone autoencoder) with one baseline predictor "
  "at the same bound, with the same quantiser and the same coder. Values are the AI predictor's reduction (baseline minus AI; positive means the AI predictor "
  "is better), absolute and relative to the baseline: entropy in bits per coordinate (backbone, side chains, all atoms), residual stream bytes in kB with lzma, "
  "with the new coder's probability model alone (no internal prediction) and with the full new coder. 'Side information' is the extra fixed data the AI "
  "predictor needs (negative: it costs more), and 'net, full AC' the change of the whole file with the full new coder. For the raw grid baselines the full new "
  "coder applies its own temporal and geometric prediction, so the full AC column compares against the raw grid coded by the full new coder.\n")
for ai in AI:
    A(f"**{LAB[ai]}** versus baselines:\n")
    A("| max error | baseline | H0 backbone | H0 side chains | H0 all atoms | Hctx all atoms | residual bytes, lzma | residual bytes, AC no prediction | residual bytes, full AC | side information | net, full AC |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BOUNDS:
        for bs in BASE:
            x0, x1 = SUM[b][bs], SUM[b][ai]
            net0 = x0["fixed_kB"] + x0["ac_kB"]; net1 = x1["fixed_kB"] + x1["ac_kB"]
            CMP.setdefault(ai, {}).setdefault(b, {})[bs] = dict(dH0_bb=x0["H0_bb"] - x1["H0_bb"], dH0_sc=x0["H0_sc"] - x1["H0_sc"], dH0_all=x0["H0_all"] - x1["H0_all"],
                dHctx_all=x0["Hctx_all"] - x1["Hctx_all"], dlz=x0["lzma_kB"] - x1["lzma_kB"], dac0=x0["ac0_kB"] - x1["ac0_kB"], dac=x0["ac_kB"] - x1["ac_kB"],
                dside=x0["fixed_kB"] - x1["fixed_kB"], dnet=net0 - net1, lz0=x0["lzma_kB"], ac0=x0["ac_kB"], net0=net0, H0all0=x0["H0_all"])
            c = CMP[ai][b][bs]
            A(f"| {BL(b)} | {LAB[bs]} | {pc(c['dH0_bb'], x0['H0_bb'])} | {pc(c['dH0_sc'], x0['H0_sc'])} | {pc(c['dH0_all'], x0['H0_all'])} | {pc(c['dHctx_all'], x0['Hctx_all'])} | "
              f"{pk(c['dlz'], x0['lzma_kB'])} | {pk(c['dac0'], x0['ac0_kB'])} | {pk(c['dac'], x0['ac_kB'])} | {c['dside']:+.1f} | {pk(c['dnet'], net0)} |")
    A("")
# log2 rule
A("Check of the log2(σ ratio) rule on the side chains, packer (hybrid) versus mean structure (the side chain predictor is the only difference between them):\n")
A("| max error | residual RMS, mean structure (Å) | residual RMS, packer (Å) | log2 of the ratio (bits) | measured H0 reduction (bits) | measured Hctx reduction (bits) |")
A("|---|---|---|---|---|---|")
for b in BOUNDS:
    s0, s1 = SUM[b]["mean structure"], SUM[b]["hybrid"]; k = s0["rms_sc"] / s1["rms_sc"]
    A(f"| {BL(b)} | {f2(s0['rms_sc'], 3)} | {f2(s1['rms_sc'], 3)} | {f2(np.log2(k))} | {f2(s0['H0_sc'] - s1['H0_sc'])} | {f2(s0['Hctx_sc'] - s1['Hctx_sc'])} |")
A("")
# ---------------- 4.7.2 side chains by stability
A("#### 4.7.2 Side chains by rotamer stability: where the packer's advantage comes from\n")
A("Reduction of the side chain residual entropy (bits per coordinate, and relative) by the packer (hybrid) against the mean structure and against the MDCompress "
  "level 4 style predictor, per residue class (stable, intermediate, unstable rotamers on the 320 K replica 0 frames, as in the main report; 'no χ': CB atoms of "
  "residues without χ angles).\n")
GR = ["stable", "intermediate", "unstable", "no_chi"]
A("| max error | baseline | " + " | ".join(gg.replace("no_chi", "no χ") for gg in GR) + " |")
A("|---|---|" + "---|" * len(GR))
for b in BOUNDS:
    for bs in ["mean structure", "MDCompress level 4"]:
        cells = []
        for gg in GR:
            dd = [d for d in DOMS if g(d, b, bs, "stats_sc_groups", gg, "H0") is not None and g(d, b, "hybrid", "stats_sc_groups", gg, "H0") is not None]
            h0 = mean([g(d, b, bs, "stats_sc_groups", gg, "H0") for d in dd]); h1 = mean([g(d, b, "hybrid", "stats_sc_groups", gg, "H0") for d in dd])
            GRP.setdefault(b, {}).setdefault(bs, {})[gg] = dict(base=h0, ai=h1, n=len(dd))
            cells.append(pc(h0 - h1, h0))
        A(f"| {BL(b)} | {LAB[bs]} | " + " | ".join(cells) + " |")
A("")
# ---------------- 4.7.3 attribution
A("#### 4.7.3 Attribution: bytes saved by the AI predictor versus by the coder\n")
A("Two by two design per bound: baseline predictor (mean structure) or AI predictor (hybrid), coded with lzma or with the full new coder. Residual bytes only "
  "(kB, mean over domains). The AI predictor's extra side information (rotamer codes, MLP, per residue data, swap bits) is listed separately. The last column "
  "splits the total residual saving from 'mean structure + lzma' to 'hybrid + new coder' between predictor and coder, in both orders.\n")
A("| max error | mean + lzma | hybrid + lzma | mean + AC | hybrid + AC | AI predictor saving with lzma | AI predictor saving with AC | coder saving, mean | coder saving, hybrid | extra side information of the AI predictor | predictor share of the total saving (predictor first / coder first) |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for b in BOUNDS:
    s0, s1 = SUM[b]["mean structure"], SUM[b]["hybrid"]
    ml, hl, ma, ha = s0["lzma_kB"], s1["lzma_kB"], s0["ac_kB"], s1["ac_kB"]
    side = s1["fixed_kB"] - s0["fixed_kB"]; tot = ml - ha
    ATT[b] = dict(ml=ml, hl=hl, ma=ma, ha=ha, side=side, pred_share_pf=(ml - hl) / tot, pred_share_cf=(ma - ha) / tot)
    A(f"| {BL(b)} | {f2(ml, 1)} | {f2(hl, 1)} | {f2(ma, 1)} | {f2(ha, 1)} | {pk(ml - hl, ml)} | {pk(ma - ha, ma)} | {pk(ml - ma, ml)} | {pk(hl - ha, hl)} | {f2(side, 1)} | "
      f"{100 * ATT[b]['pred_share_pf']:.0f} % / {100 * ATT[b]['pred_share_cf']:.0f} % |")
A("")
# ---------------- MDDB (hybrid versus mean structure, lzma and new coder)
def r_(src, s_, m_):
    v = src.get(f"{s_}/T1000/{m_}"); return v["ratio"] if v and "error" not in v else np.nan
prot = [s_ for s_ in MDDB if np.isfinite(r_(rows, s_, "ours: hybrid @ 0.5"))]
MD = {}
for s_ in prot:
    for b in BOUNDS:
        rh = lambda k: (lambda v: v["bb_and_other_residual_bytes"] + v["sc_residual_bytes"] if v else np.nan)(ent.get(f"{s_}/T1000/{k}"))
        rm = lambda k: (lambda v: v["resid_bytes"] if v else np.nan)(ent.get(f"{s_}/T1000/{k}"))
        sh = lambda k: (lambda v: v["fixed_bytes"] + v["swap_bytes"] if v else np.nan)(ent.get(f"{s_}/T1000/{k}"))
        smf = lambda k: (lambda v: v["bytes"] - v["resid_bytes"] if v else np.nan)(ent.get(f"{s_}/T1000/{k}"))
        MD.setdefault(s_, {})[b] = dict(res_h_ac=rh(f"ours: hybrid + AC @ {b}"), res_m_ac=rm(f"model free mean + AC @ {b}"), res_h_lz=rh(f"ours: hybrid @ {b}"), res_m_lz=rm(f"model free mean @ {b}"),
                                        side=sh(f"ours: hybrid + AC @ {b}") - smf(f"model free mean + AC @ {b}"),
                                        hl=r_(rows, s_, f"ours: hybrid @ {b}"), ml=r_(rows, s_, f"model free mean @ {b}"), ha=r_(ent, s_, f"ours: hybrid + AC @ {b}"),
                                        ma=r_(ent, s_, f"model free mean + AC @ {b}"), aa=r_(ent, s_, f"ours: hybrid + backbone AE + AC @ {b}"))
# ---------------- 4.7.4 findings, worded from the numbers
def wins(ai, b, key):
    return [bs for bs in BASE if CMP[ai][b][bs][key] > 0]
A("#### 4.7.4 What the numbers say\n")
h, ae = "hybrid", "hybrid + backbone AE"
c05, c01, cll = CMP[h]["0.5"], CMP[h]["0.1"], CMP[h]["0.005"]
A(f"* **The AI predictor shrinks the residual; it does not remove it.** Against the mean structure, the packer lowers the side chain entropy by "
  f"{pc(c05['mean structure']['dH0_sc'], SUM['0.5']['mean structure']['H0_sc'])} bits per coordinate at 0.5 Å, {pc(c01['mean structure']['dH0_sc'], SUM['0.1']['mean structure']['H0_sc'])} at 0.1 Å "
  f"and {pc(cll['mean structure']['dH0_sc'], SUM['0.005']['mean structure']['H0_sc'])} at lossless; the side chain residual still carries {f2(SUM['0.005']['hybrid']['H0_sc'])} bits per "
  f"coordinate at lossless. The measured reduction tracks log2 of the residual RMS ratio (table above: {f2(np.log2(SUM['0.5']['mean structure']['rms_sc'] / SUM['0.5']['hybrid']['rms_sc']))} "
  f"predicted against {f2(c05['mean structure']['dH0_sc'])} measured at 0.5 Å, {f2(np.log2(SUM['0.005']['mean structure']['rms_sc'] / SUM['0.005']['hybrid']['rms_sc']))} against "
  f"{f2(cll['mean structure']['dH0_sc'])} at lossless). In bits it grows towards tight bounds, because the packer predicts from the backbone decoded at the bound "
  f"(side chain residual RMS {f2(SUM['0.5']['hybrid']['rms_sc'], 3)} Å at 0.5 Å, {f2(SUM['0.005']['hybrid']['rms_sc'], 3)} Å at lossless); relative to the baseline entropy it "
  f"peaks at intermediate bounds and is smallest at lossless, where the remaining entropy is dominated by the fine grid.")
A(f"* **Against the raw grid the AI predictor wins everywhere.** All atom entropy reduction versus no prediction: " + ", ".join(f"{BL(b)} {pc(CMP[h][b]['none']['dH0_all'], CMP[h][b]['none']['H0all0'])}" for b in BOUNDS) +
  " bits; lzma residual bytes: " + ", ".join(f"{BL(b)} {pk(CMP[h][b]['none']['dlz'], CMP[h][b]['none']['lz0'])} kB" for b in BOUNDS) + ".")
for bs in ["mean structure", "MDCompress level 2", "MDCompress level 4"]:
    wl = [b for b in BOUNDS if CMP[h][b][bs]["dlz"] > 0]; wa = [b for b in BOUNDS if CMP[h][b][bs]["dac"] > 0]; wn = [b for b in BOUNDS if CMP[h][b][bs]["dnet"] > 0]
    wh = [b for b in BOUNDS if CMP[h][b][bs]["dH0_all"] > 0]
    A(f"* **Packer (hybrid) versus {LAB[bs]}.** Lower all atom entropy at: {', '.join(BL(b) for b in wh) or 'no bound'}; fewer residual bytes with lzma at: "
      f"{', '.join(BL(b) for b in wl) or 'no bound'}; with the full new coder at: {', '.join(BL(b) for b in wa) or 'no bound'}; smaller whole file (side information "
      f"included, full new coder) at: {', '.join(BL(b) for b in wn) or 'no bound'}. Residual bytes with lzma: " +
      ", ".join(f"{BL(b)} {pk(CMP[h][b][bs]['dlz'], CMP[h][b][bs]['lz0'])}" for b in BOUNDS) + " kB.")
A(f"* **Where the MDCompress style prediction is stronger.** The level 4 style prediction (local frame of three bonded neighbours carried from the previous "
  f"frame) has all atom entropy " + ", ".join(f"{f2(SUM[b]['MDCompress level 4']['H0_all'])}" for b in BOUNDS) + " bits at " + ", ".join(BL(b) for b in BOUNDS) +
  ", against " + ", ".join(f"{f2(SUM[b]['hybrid']['H0_all'])}" for b in BOUNDS) + " for the hybrid. Bonded geometry carried from the previous frame captures fine "
  "detail that a mean structure or a packer accurate to a few tenths of an ångström does not, so the AI predictor alone has the higher entropy at: " + (", ".join(BL(b) for b in BOUNDS if CMP[h][b]["MDCompress level 4"]["dH0_all"] < 0) or "no bound") + ". This, together with its range coder, is why MDCompress wins at tight bounds against hybrid + lzma. The new coder "
  "applies the same geometric prediction to the AI predictor's residuals, and that combination is what overtakes MDCompress at 0.1 Å (section 4.6).")
A(f"* **Attribution.** From mean structure + lzma to hybrid + new coder, the AI predictor's share of the residual saving is " +
  ", ".join(f"{BL(b)} {100 * ATT[b]['pred_share_pf']:.0f} % ({100 * ATT[b]['pred_share_cf']:.0f} % if the coder is changed first)" for b in BOUNDS) + "; the rest is the coder. " +
  (lambda sh: (f"This share stays between {100 * min(sh):.0f} % and {100 * max(sh):.0f} % at all bounds, so the data do not support the stronger form of the argument "
   "that the coder's share grows at tight bounds; in absolute terms both savings grow as the bound tightens. What changes at tight bounds is the comparison "
   "with the MDCompress style geometric predictor (previous bullet), not the split between predictor and coder.") if max(sh) - min(sh) < 0.12 else
   f"The share falls from {100 * sh[0]:.0f} % at 0.5 Å to {100 * sh[-1]:.0f} % at lossless.")([ATT[b]["pred_share_pf"] for b in BOUNDS]))
G = GRP
def gl(bs, gg, cond): return [BL(b) for b in BOUNDS if cond(G[b][bs][gg]["base"] - G[b][bs][gg]["ai"])]
A(f"* **Side chains by rotamer stability.** Against the mean structure the packer's entropy reduction is largest for unstable and intermediate rotamers and "
  f"smallest for stable ones (at 0.1 Å: stable {pc(G['0.1']['mean structure']['stable']['base'] - G['0.1']['mean structure']['stable']['ai'], G['0.1']['mean structure']['stable']['base'])}, "
  f"unstable {pc(G['0.1']['mean structure']['unstable']['base'] - G['0.1']['mean structure']['unstable']['ai'], G['0.1']['mean structure']['unstable']['base'])} bits): a mean "
  "structure cannot follow rotamer jumps, the packer with its per frame rotamer codes can. Against the MDCompress level 4 style predictor the picture differs: "
  f"for stable side chains the packer has the higher entropy at {', '.join(gl('MDCompress level 4', 'stable', lambda x: x < 0)) or 'no bound'} (a stable side chain is well "
  f"predicted by carrying its local geometry from the previous frame), while for unstable side chains it keeps an advantage at {', '.join(gl('MDCompress level 4', 'unstable', lambda x: x > 0)) or 'no bound'}.")
a05 = CMP[ae]["0.5"]["mean structure"]
A(f"* **Backbone autoencoder: best predictor, too expensive to describe.** Its backbone entropy reduction against the mean structure is "
  f"{pc(CMP[ae]['0.5']['mean structure']['dH0_bb'], SUM['0.5']['mean structure']['H0_bb'])} bits per coordinate at 0.5 Å and "
  f"{pc(CMP[ae]['0.005']['mean structure']['dH0_bb'], SUM['0.005']['mean structure']['H0_bb'])} at lossless, and it saves {pk(a05['dac'], a05['ac0'])} kB of residuals with the full new coder at 0.5 Å, "
  f"but its fp16 latents add {-a05['dside']:.1f} kB of side information, so the whole file grows by {-a05['dnet']:.1f} kB ({-100 * a05['dnet'] / a05['net0']:.0f} %). A better predictor only pays "
  "when its own description costs less than the entropy it removes; for these latents it does not, at any bound (section 4.6.6).")
worse_res_ac = [(s_, b) for s_ in prot for b in BOUNDS if MD[s_][b]["res_h_ac"] > MD[s_][b]["res_m_ac"]]
worse_res_lz = [(s_, b) for s_ in prot for b in BOUNDS if MD[s_][b]["res_h_lz"] > MD[s_][b]["res_m_lz"]]
tight_t = [(s_, b) for s_ in prot for b in ("0.2", "0.1", "0.005")]
pays_t = [(s_, b) for s_, b in tight_t if MD[s_][b]["res_m_ac"] - MD[s_][b]["res_h_ac"] > MD[s_][b]["side"]]
loose = [(s_, b) for s_ in prot for b in ("0.5", "0.3") if MD[s_][b]["ml"] > MD[s_][b]["hl"]]
loose_ac = [(s_, b) for s_ in prot for b in ("0.5", "0.3") if MD[s_][b]["ma"] > MD[s_][b]["ha"]]
A(f"* **MDDB proteins: here the claim partly fails.** With lzma the "
  f"mean structure gave a smaller whole file than the hybrid in {len(loose)} of {2 * len(prot)} loose bound cases (0.5 and 0.3 Å), with the new coder in {len(loose_ac)} of "
  f"{2 * len(prot)}. The reason is not only side information: the AI predictor's residual stream itself was larger than the mean structure's in {len(worse_res_ac)} of "
  f"{len(prot) * len(BOUNDS)} system and bound cases with the new coder (lzma: {len(worse_res_lz)}), {sum(b in ('0.5', '0.3') for _, b in worse_res_ac)} of them at 0.5 or 0.3 Å: " +
  "; ".join(f"{MDDB[s_]['accession']} {BL(b)}" for s_, b in worse_res_ac) + f". At 0.2 Å, 0.1 Å and lossless the packer's residual saving exceeded its extra side information in {len(pays_t)} of {len(tight_t)} cases (table below). "
  "The argument therefore holds on the mdCATH test domains (held out proteins from the dataset the packer was trained on); on these MDDB proteins, which come from different simulation setups (programs, force fields, temperatures), the AI predictor does not "
  "shrink the residual at loose bounds. We did not test why; a shift away from the training distribution is a plausible cause, not a demonstrated one.\n")
A("MDDB proteins: residual bytes of the hybrid and of the mean structure with the new coder (kB), and the hybrid's extra side information (kB):\n")
A("| system | " + " | ".join(f"{BL(b)}: hybrid / mean / extra side" for b in BOUNDS) + " |")
A("|---|" + "---|" * len(BOUNDS))
for s_ in prot:
    A(f"| {MDDB[s_]['kind']} ({MDDB[s_]['accession']}) | " + " | ".join(f"{MD[s_][b]['res_h_ac'] / 1e3:.1f} / {MD[s_][b]['res_m_ac'] / 1e3:.1f} / {MD[s_][b]['side'] / 1e3:.1f}" for b in BOUNDS) + " |")
A("")
A("MDDB proteins, ratio hybrid / mean structure, lzma ; new coder ; hybrid + backbone AE with the new coder (T = 1,000):\n")
A("| system | " + " | ".join(BL(b) for b in BOUNDS) + " |")
A("|---|" + "---|" * len(BOUNDS))
for s_ in prot:
    A(f"| {MDDB[s_]['kind']} ({MDDB[s_]['accession']}) | " + " | ".join(f"{f2(MD[s_][b]['hl'])} / {f2(MD[s_][b]['ml'])} ; {f2(MD[s_][b]['ha'])} / {f2(MD[s_][b]['ma'])} ; {f2(MD[s_][b]['aa'])}" for b in BOUNDS) + " |")
A("")
# ---------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COLS = {"temporal": "tab:gray", "MDCompress level 2": "tab:pink", "MDCompress level 4": "tab:red", "mean structure": "tab:blue", "hybrid": "tab:green", "hybrid + backbone AE": "olive", "none": "k"}
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for j, b in enumerate(["0.5", "0.1", "0.005"]):
    for i, part in enumerate(["stats_bb", "stats_sc"]):
        a_ = ax[i, j]
        for p in ["MDCompress level 2", "MDCompress level 4", "mean structure", "hybrid", "hybrid + backbone AE"]:
            hh = {}
            for d in DOMS:
                for k, v in g(d, b, p, part, "hist").items(): hh[int(k)] = hh.get(int(k), 0) + v
            ks = np.array(sorted(hh)); vs = np.array([hh[k] for k in ks], float); vs /= vs.sum()
            a_.plot(ks, vs, c=COLS[p], label=LAB[p] + (" (AI)" if p in AI else " (baseline)"), drawstyle="steps-mid", lw=2.2 if p in AI else 1.0, ls="-" if p in AI else "--")
        a_.set_yscale("log"); a_.set_xlim(-25, 25); a_.grid(alpha=0.3)
        a_.set(title=f"{'backbone' if i == 0 else 'side chains'}, max error {BL(b)}", xlabel="residual integer (clipped at ±40)", ylabel="probability")
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"Residual integer distributions: AI predictors (thick) versus baseline predictors (dashed); mdCATH, {n} domains, T = 1000, frames 2 to T", fontsize=11)
fig.tight_layout(); fig.savefig(f"{BEN}/figs/fig_residual_hist.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
xs = list(range(len(BOUNDS)))
for i, part in enumerate(["bb", "sc"]):
    for p in PRED[1:]:
        ax[i].plot(xs, [SUM[b][p][f"H0_{part}"] for b in BOUNDS], c=COLS[p], marker="o", lw=2.2 if p in AI else 1.0, label=LAB[p] + (" (AI)" if p in AI else ""))
    ax[i].set_xticks(xs); ax[i].set_xticklabels([BL(b) for b in BOUNDS]); ax[i].grid(alpha=0.3)
    ax[i].set(title=("backbone" if part == "bb" else "side chains") + ": zeroth order entropy", xlabel="guaranteed maximum error", ylabel="bits per coordinate")
ax[0].legend(fontsize=7)
for ai, ls in zip(AI, ["-", "--"]):
    for bs in ["none", "mean structure", "MDCompress level 2", "MDCompress level 4"]:
        ax[2].plot(xs, [CMP[ai][b][bs]["dlz"] / CMP[ai][b][bs]["lz0"] * 100 for b in BOUNDS], c=COLS[bs], ls=ls, marker="o", label=f"{LAB[ai]} vs {LAB[bs]}")
ax[2].axhline(0, c="k", lw=0.8); ax[2].set_xticks(xs); ax[2].set_xticklabels([BL(b) for b in BOUNDS]); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=6)
ax[2].set(title="AI predictor's residual byte reduction with lzma (%)", xlabel="guaranteed maximum error", ylabel="reduction vs baseline (%), positive = AI better")
fig.suptitle("Residual entropy and AI predictor savings versus bound (same quantiser and coder for all predictors)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{BEN}/figs/fig_entropy_vs_bound.png", dpi=150); plt.close(fig)
A("![Residual histograms](figs/fig_residual_hist.png)\n")
A("*Figure 8. Distribution of the residual integers at 0.5 Å, 0.1 Å and lossless, backbone (top) and side chains (bottom), log scale; AI predictors thick, baselines dashed. The hybrid's backbone uses the mean structure predictor, so its backbone curve covers the mean structure curve.*\n")
A("![Entropy versus bound](figs/fig_entropy_vs_bound.png)\n")
A("*Figure 9. Left and middle: zeroth order entropy of the residual integers per predictor and bound. Right: the AI predictors' reduction of the lzma coded "
  "residual bytes against each baseline (solid: hybrid, dashed: hybrid + backbone autoencoder).*\n")
# ---------------- 4.7.5 reference table
A("#### 4.7.5 Reference: residual statistics and sizes per predictor\n")
A("Residual RMS (Å), entropies (bits per coordinate), residual stream sizes (kB: ideal H0 size, lzma, AC without internal prediction, full AC), fixed side "
  "information (kB) and whole file ratio with lzma and with the full new coder, mean over domains.\n")
A("| max error | predictor | RMS bb | RMS sc | H0 bb | H0 sc | Hctx bb | Hctx sc | ideal H0 | lzma | AC no prediction | full AC | fixed | ratio lzma | ratio full AC |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for b in BOUNDS:
    for p in PRED:
        x = SUM[b][p]
        A(f"| {BL(b)} | {LAB[p]} | {f2(x['rms_bb'], 3)} | {f2(x['rms_sc'], 3)} | {f2(x['H0_bb'])} | {f2(x['H0_sc'])} | {f2(x['Hctx_bb'])} | {f2(x['Hctx_sc'])} | "
          f"{f2(x['ideal_kB'], 1)} | {f2(x['lzma_kB'], 1)} | {f2(x['ac0_kB'], 1)} | {f2(x['ac_kB'], 1)} | {f2(x['fixed_kB'], 1)} | {f2(x['ratio_lzma'])} | {f2(x['ratio_ac'])} |")
A("")
A("Notes: RMS for the raw grid ('none') is the RMS of the coordinates themselves. The lzma search includes frame difference layouts, so lzma applied to the raw "
  "grid already contains a previous frame predictor (its sizes for 'none' and 'previous frame' coincide). The MDCompress style predictors are our "
  "reimplementation in the integer domain (with our choice of reference atoms), not MDCompress itself; MDCompress's own sizes are in sections 4.1 to 4.6.\n")
open(f"{E}/predictor_section.md", "w").write("\n".join(L) + "\n")
json.dump(dict(stats=SUM, ai_vs_baseline=CMP, side_chain_groups=GRP, attribution=ATT, mddb=MD, domains=DOMS), open(f"{E}/results/predictor_summary.json", "w"), indent=1, default=float)
print("wrote predictor_section.md", n, "domains")
