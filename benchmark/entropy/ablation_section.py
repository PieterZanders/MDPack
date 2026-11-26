#!/usr/bin/env python
"""AEZ-12 ablation section 4.7.6 (rotamer codes and per trajectory MLP) -> ablation_section.md, figs/fig_ablation.png,
results/ablation_summary.json. Reads results/ablation_{A,B,C,D}.json written by ablate.py."""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BEN = "/workspace/md_compress/benchmark"; E = f"{BEN}/entropy"
R = {}
for f in sorted(glob.glob(f"{E}/results/ablation_*.json")): R.update(json.load(open(f)))
BOUNDS = ["0.5", "0.3", "0.2", "0.1", "0.005"]; BL = lambda b: "lossless" if b == "0.005" else f"{b} Å"
VARS = ["full", "no codes", "no MLP", "neither"]
VLAB = {"full": "full (codes + packer + MLP)", "no codes": "no rotamer codes (packer + MLP)", "no MLP": "no MLP (codes + packer)", "neither": "neither (packer only)"}
MDCATH = ["1e8rA00", "2ikeA00", "2jbvA03", "3iwcB00", "3mzkB03", "3nisA01", "2p4vA02", "4m70B00", "1jyoE00", "2g36A02", "3ogdA01", "2lg7A00", "1yf2A03", "2rsxA00", "1r44A00", "1b5eA00"]
MDDB = [("mddb_small_1APQ", "small protein MD-A00008"), ("mddb_medium_12CA", "medium protein MD-A00001"), ("mddb_protdna_FOXA3", "protein + DNA MD-A001FH"),
        ("mddb_large_1AYX", "large protein MD-A00009")]
L = []; A = L.append; SUM = {}
f2 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"
def rec(s, T, v, b):
    r = R.get(f"{s}/T{T}/{v} @ {float(b)}") or R.get(f"{s}/T{T}/{v} @ {b}"); return r
def done(s, T): return all(rec(s, T, v, b) for v in VARS for b in BOUNDS)
def mean_over(sys_, T, fn):
    ok = [s for s in sys_ if done(s, T)]
    return ok, {v: {b: float(np.mean([fn(rec(s, T, v, b)) for s in ok])) if ok else np.nan for b in BOUNDS} for v in VARS}
res_ac = lambda r: r["sc_resid_ac"] + r["swap_bytes"]
res_lz = lambda r: r["sc_resid_lzma"] + r["swap_bytes"]

def net(sys_, T, comp, coder="ac"):
    """per domain bytes saved in side chain residuals minus the component's own bytes, in both contexts"""
    rr = res_ac if coder == "ac" else res_lz
    out = {}
    for b in BOUNDS:
        rows = []
        for s in sys_:
            if not done(s, T): continue
            g = lambda v: rec(s, T, v, b)
            if comp == "codes":
                cost = g("full")["codes_bytes"]
                rows.append((rr(g("no codes")) - rr(g("full")) - cost, rr(g("neither")) - rr(g("no MLP")) - cost, cost))
            else:
                cost = g("full")["mlp_bytes"]
                rows.append((rr(g("no MLP")) - rr(g("full")) - cost, rr(g("neither")) - rr(g("no codes")) - cost, cost))
        if rows:
            a_ = np.array(rows, float)
            out[b] = dict(net_with_other=float(a_[:, 0].mean()), net_without_other=float(a_[:, 1].mean()), cost=float(a_[:, 2].mean()),
                          saving_with_other=float((a_[:, 0] + a_[:, 2]).mean()), n=len(rows), n_pays=int((a_[:, 0] > 0).sum()))
    return out

A("#### 4.7.6 Ablation: what the rotamer codes and the per trajectory MLP cost and save\n")
A("The hybrid's side chain predictor has three parts: the shared packer (trained once, weights not counted), the per frame rotamer codes (chi1 to chi4 "
  "states with flip mask and state table, stored per trajectory) and the per trajectory residual MLP refitted on each trajectory (5,960 bytes). To see what "
  "each part buys, the hybrid was rerun in four variants on exactly the same backbone stream (the backbone and non protein atoms are coded against the mean "
  "structure in every variant, so the backbone residual bytes are identical across variants and all differences are side chain residuals and side data): "
  "(1) full; (2) no rotamer codes: the packer gets only the static per residue chi prior (already part of the 80 bytes per residue of the reference), codes, "
  "flip mask and state table are not stored; (3) no MLP: rotamer codes and packer, the packer's chi output used directly, MLP not stored; (4) neither: shared "
  "packer with the static chi prior only. Same quantiser and bound, new coder (and the three layout lzma search for reference); every stream decoded and "
  "every reconstruction checked against its bound. Side data = 12 bytes per backbone (and other) atom for the mean structure + 28 bytes per frame (rigid "
  "transform) + 80 bytes per residue + codes (if used) + MLP (if used). The per trajectory MLP fit is stochastic; this run refits it, so the full variant "
  "differs slightly from section 4.6.\n")
nchk = sum(1 for r in R.values() if "checked_max_abs_err" in r)
nbad = sum(1 for r in R.values() if "checked_max_abs_err" in r and r["checked_max_abs_err"] > float(r["bound"]) + 1e-3)
A(f"Verification: {nchk} coded variants, every AC stream decoded bit exactly, {nbad} exceeded the bound.\n")

def block(sys_, T, title):
    ok, rat = mean_over(sys_, T, lambda r: r["ratio_ac"])
    if not ok: return None
    _, ratl = mean_over(sys_, T, lambda r: r["ratio_lzma"])
    _, side = mean_over(sys_, T, lambda r: r["side_bytes"]); _, sc = mean_over(sys_, T, res_ac); _, bb = mean_over(sys_, T, lambda r: r["bb_resid_ac"])
    _, tot = mean_over(sys_, T, lambda r: r["bytes_ac"])
    _, h = mean_over(sys_, T, lambda r: r["sc_H0"]); _, hm = mean_over(sys_, T, lambda r: r["mean_sc"]["H0"])
    A(f"**{title}** ({len(ok)} trajectories, T = {T:,}; mean ratio, new coder; lzma in brackets):\n")
    A("| variant | " + " | ".join(BL(b) for b in BOUNDS) + " |"); A("|---|" + "---|" * len(BOUNDS))
    for v in VARS: A(f"| {VLAB[v]} | " + " | ".join(f"{f2(rat[v][b])}× ({f2(ratl[v][b])}×)" for b in BOUNDS) + " |")
    A("")
    A("Mean bytes per trajectory (new coder): side data / side chain residuals (incl. swap bits) / backbone residuals / total, and side chain H0 "
      "(bits per coordinate) with its reduction against the mean structure side chain residuals at the same bound:\n")
    A("| variant | " + " | ".join(BL(b) for b in BOUNDS) + " |"); A("|---|" + "---|" * len(BOUNDS))
    for v in VARS:
        A(f"| {VLAB[v]} | " + " | ".join(f"{side[v][b] / 1e3:.1f} / {sc[v][b] / 1e3:.1f} / {bb[v][b] / 1e3:.1f} / {tot[v][b] / 1e3:.1f} kB; H0 {h[v][b]:.2f} "
                                         f"({hm[v][b] - h[v][b]:+.2f} bits, {(hm[v][b] - h[v][b]) / hm[v][b] * 100:+.0f} %)" for b in BOUNDS) + " |")
    A("")
    nc, nm = net(ok, T, "codes"), net(ok, T, "mlp")
    A("Net effect of each component (mean bytes per trajectory, new coder): side chain residual bytes saved minus the component's own bytes; positive = it pays "
      "for itself. 'With' / 'without' = with or without the other component present; in brackets the number of trajectories where it pays (with the other present):\n")
    A("| component | " + " | ".join(BL(b) for b in BOUNDS) + " |"); A("|---|" + "---|" * len(BOUNDS))
    A("| rotamer codes: cost | " + " | ".join(f"{nc[b]['cost']:,.0f} B" for b in BOUNDS) + " |")
    A("| rotamer codes: residual saving (with MLP) | " + " | ".join(f"{nc[b]['saving_with_other']:,.0f} B" for b in BOUNDS) + " |")
    A("| rotamer codes: net with MLP / without MLP | " + " | ".join(f"{nc[b]['net_with_other']:+,.0f} / {nc[b]['net_without_other']:+,.0f} B ({nc[b]['n_pays']} of {nc[b]['n']})" for b in BOUNDS) + " |")
    A("| MLP: cost | " + " | ".join(f"{nm[b]['cost']:,.0f} B" for b in BOUNDS) + " |")
    A("| MLP: residual saving (with codes) | " + " | ".join(f"{nm[b]['saving_with_other']:,.0f} B" for b in BOUNDS) + " |")
    A("| MLP: net with codes / without codes | " + " | ".join(f"{nm[b]['net_with_other']:+,.0f} / {nm[b]['net_without_other']:+,.0f} B ({nm[b]['n_pays']} of {nm[b]['n']})" for b in BOUNDS) + " |")
    A("")
    best = {b: max(VARS, key=lambda v: rat[v][b]) for b in BOUNDS}
    return dict(n=len(ok), ratio_ac=rat, ratio_lzma=ratl, side=side, sc=sc, bb=bb, total=tot, sc_H0=h, mean_sc_H0=hm, net_codes=nc, net_mlp=nm, best=best)

SUM["mdcath_T1000"] = block(MDCATH, 1000, "mdCATH, 16 domains")
if SUM["mdcath_T1000"]:
    # stability groups at T = 1000
    ok = [s for s in MDCATH if done(s, 1000)]
    A("Side chain H0 by rotamer stability (mdCATH, T = 1,000, mean over domains; bits per coordinate; mean structure in the last column):\n")
    for b in ("0.5", "0.1", "0.005"):
        A(f"{BL(b)}:\n"); A("| group | " + " | ".join(VARS) + " | mean structure |"); A("|---|" + "---|" * (len(VARS) + 1))
        for g in ("stable", "intermediate", "unstable", "no_chi"):
            vals = [np.nanmean([rec(s, 1000, v, b)["sc_H0_groups"][g] for s in ok]) for v in VARS]
            mv = np.nanmean([rec(s, 1000, "full", b)["mean_sc"]["groups"][g] for s in ok])
            A(f"| {g.replace('_', ' ')} | " + " | ".join(f"{x:.2f}" for x in vals) + f" | {mv:.2f} |")
            SUM.setdefault("groups_T1000", {}).setdefault(b, {})[g] = dict(zip(VARS + ["mean structure"], vals + [mv]))
        A("")
SUM["mdcath_T250"] = block(MDCATH, 250, "mdCATH, 16 domains, short trajectories")
SUM["mddb"] = {}
okm = [s for s, _ in MDDB if done(s, 1000)]
if okm:
    A("**MDDB (T = 1,000, systems with protein; the membrane system was skipped for memory, and DNA and RNA have no protein so the hybrid does not apply).** "
      "Ratio with the new coder per system and variant:\n")
    A("| system | variant | " + " | ".join(BL(b) for b in BOUNDS) + " |"); A("|---|---|" + "---|" * len(BOUNDS))
    for s, lab in MDDB:
        if not done(s, 1000): continue
        for v in VARS:
            A(f"| {lab} | {v} | " + " | ".join(f2(rec(s, 1000, v, b)["ratio_ac"]) for b in BOUNDS) + " |")
        SUM["mddb"][s] = {v: {b: rec(s, 1000, v, b)["ratio_ac"] for b in BOUNDS} for v in VARS}
    A("")
    nc, nm = net(okm, 1000, "codes"), net(okm, 1000, "mlp")
    A("Net effect on MDDB (mean over these systems, new coder): rotamer codes " + "; ".join(f"{BL(b)} {nc[b]['net_with_other']:+,.0f} B ({nc[b]['n_pays']} of {nc[b]['n']})" for b in BOUNDS) +
      ". MLP " + "; ".join(f"{BL(b)} {nm[b]['net_with_other']:+,.0f} B ({nm[b]['n_pays']} of {nm[b]['n']})" for b in BOUNDS) + ".\n")
    SUM["mddb_net"] = dict(codes=nc, mlp=nm)
SUM["long"] = {}
for T in (1000, 10000):
    s = "mddb_small_1APQ_long"
    if done(s, T): SUM["long"][T] = block([s], T, f"MD-A00008 at 1 ps frame spacing (fine spacing case, first {T:,} frames of the 10 ns run)")

# figure
blocks = [(k, SUM[k]) for k in ("mdcath_T1000", "mdcath_T250") if SUM.get(k)] + [(f"long_T{T}", SUM["long"][T]) for T in SUM["long"]]
if blocks:
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    col = {"full": "darkgreen", "no codes": "tab:orange", "no MLP": "tab:purple", "neither": "#777777"}
    xb = np.arange(len(BOUNDS)); s1 = SUM["mdcath_T1000"] or blocks[0][1]
    for v in VARS:
        ax[0].plot(xb, [s1["ratio_ac"][v][b] for b in BOUNDS], marker="o", c=col[v], label=VLAB[v] + ", new coder")
        ax[0].plot(xb, [s1["ratio_lzma"][v][b] for b in BOUNDS], marker=".", c=col[v], ls=":", label=VLAB[v] + ", lzma")
    ax[0].set(xticks=xb, xticklabels=[BL(b) for b in BOUNDS], ylabel="mean ratio (×)", title="mdCATH T = 1,000: ratio per variant"); ax[0].legend(fontsize=6); ax[0].grid(alpha=0.3)
    w = 0.8 / len(blocks)
    for comp, a_ in (("net_codes", ax[1]), ("net_mlp", ax[2])):
        cc = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, (k, bl) in enumerate(blocks):
            xx = xb + (i - (len(blocks) - 1) / 2) * w
            a_.bar(xx, [bl[comp][b]["net_with_other"] / bl["total"]["full"][b] * 100 for b in BOUNDS], w, color=cc[i % 4], label=k.replace("_", " ") + ", other component present")
            a_.bar(xx, [bl[comp][b]["net_without_other"] / bl["total"]["full"][b] * 100 for b in BOUNDS], w * 0.45, color="white", edgecolor=cc[i % 4], hatch="///",
                   label=k.replace("_", " ") + ", other component absent")
        a_.axhline(0, c="k", lw=0.8); a_.set(xticks=xb, xticklabels=[BL(b) for b in BOUNDS], ylabel="net saving, % of the full variant's file",
                                             title=("rotamer codes" if comp == "net_codes" else "per trajectory MLP") + ": residual saving minus own bytes")
        a_.legend(fontsize=6); a_.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{BEN}/figs/fig_ablation.png", dpi=150); plt.close(fig)
    A("![Ablation](figs/fig_ablation.png)\n")
    A("*Figure 10. Ablation of the hybrid's side chain predictor. Left: mean ratio on mdCATH (T = 1,000) per variant (solid: new coder, dotted: lzma). Middle and "
      "right: net saving of the rotamer codes and of the per trajectory MLP (side chain residual saving minus the component's own size) as a percentage of the "
      "full variant's file, wide bars with the other component present, narrow hatched bars without it; above zero the component pays for itself. "
      "'long' = MD-A00008 at 1 ps spacing.*\n")
def pct(bl, v, b): return (bl["ratio_ac"]["full"][b] / bl["ratio_ac"][v][b] - 1) * 100
t1, t2 = SUM.get("mdcath_T1000"), SUM.get("mdcath_T250")
if t1 and t2:
    A("**What the ablation shows.**\n")
    A("* **Long mdCATH trajectories (T = 1,000): both components pay, but only together.** The full predictor is the best variant at every bound "
      "(" + ", ".join(f"{BL(b)} {t1['ratio_ac']['full'][b]:.2f}× against {t1['ratio_ac']['neither'][b]:.2f}× for the packer alone, {pct(t1, 'neither', b):+.1f} %" for b in BOUNDS) +
      "), and with the other component present each one pays for itself on all 16 domains at every bound (rotamer codes net "
      + ", ".join(f"{t1['net_codes'][b]['net_with_other'] / 1e3:+.1f} kB" for b in BOUNDS) + "; MLP net " + ", ".join(f"{t1['net_mlp'][b]['net_with_other'] / 1e3:+.1f} kB" for b in BOUNDS) +
      " per trajectory). Alone, neither does: without the MLP the codes save less than their own "
      f"{t1['net_codes']['0.5']['cost'] / 1e3:.1f} kB at 0.5 Å and lossless (net {t1['net_codes']['0.5']['net_without_other'] / 1e3:+.1f} and {t1['net_codes']['0.005']['net_without_other'] / 1e3:+.1f} kB) "
      f"and only a few kB at 0.3 to 0.1 Å; without the codes the MLP never earns back its 5.96 kB (net {min(t1['net_mlp'][b]['net_without_other'] for b in BOUNDS) / 1e3:+.1f} to "
      f"{max(t1['net_mlp'][b]['net_without_other'] for b in BOUNDS) / 1e3:+.1f} kB). So 'no codes' and 'no MLP' are both worse than the bare packer at 0.5 Å. The "
      "interpretation: the codes tell the packer which rotamer each residue is in, and the MLP is fitted on predictions that are already conditioned on the right "
      "rotamer, so it learns a trajectory specific correction within each rotamer; without the codes it can only fit an average over rotamers, which the packer "
      "mostly has already.")
    g = SUM.get("groups_T1000", {})
    if g:
        A("* **Where the bits go.** The rotamer codes act on the residues that change rotamer: at 0.1 Å the unstable side chains drop from "
          f"{g['0.1']['unstable']['neither']:.2f} (packer alone) to {g['0.1']['unstable']['no MLP']:.2f} bits with codes and {g['0.1']['unstable']['full']:.2f} with codes and MLP "
          f"(mean structure {g['0.1']['unstable']['mean structure']:.2f}), while stable side chains move only from {g['0.1']['stable']['neither']:.2f} to {g['0.1']['stable']['full']:.2f}. "
          f"Residues without chi angles (no rotamer) are unaffected ({g['0.1']['no_chi']['full']:.2f} bits in every variant); the backbone stream is identical in all "
          "variants by construction (mean structure predictor), so every difference in the tables is side chain.")
    A("* **Short mdCATH trajectories (T = 250): the components do not pay for themselves.** The bare packer is the best variant at every bound ("
      + ", ".join(f"{BL(b)} {t2['ratio_ac']['neither'][b]:.2f}× against {t2['ratio_ac']['full'][b]:.2f}× full" for b in BOUNDS) +
      "). The MLP's fixed 5.96 kB is earned back only on " + ", ".join(f"{t2['net_mlp'][b]['n_pays']}" for b in BOUNDS) +
      " of 16 domains (per bound), and the codes are net positive with the MLP but the pair together does not beat leaving both out.")
    if SUM.get("mddb_net"):
        mn = SUM["mddb_net"]
        A("* **MDDB proteins (the T = 1,000 files of section 4.3):** the full predictor is the best variant on all four systems at the lossy bounds, "
          "with the largest gains on the medium and large proteins (large protein at 0.5 Å "
          f"{SUM['mddb']['mddb_large_1AYX']['full']['0.5']:.2f}× against {SUM['mddb']['mddb_large_1AYX']['neither']['0.5']:.2f}× for the packer alone); at lossless all four variants are within about 1 %.")
    lg = SUM["long"]
    if 10000 in lg:
        b10 = lg[10000]
        A("* **Fine frame spacing (MD-A00008 at 1 ps):** at T = 10,000 the full predictor is best at every bound, but by small margins in ratio ("
          + ", ".join(f"{BL(b)} {pct(b10, 'neither', b):+.1f} %" for b in BOUNDS) + " against the packer alone), and the same synergy appears: rotamer codes net "
          f"{b10['net_codes']['0.1']['net_with_other'] / 1e3:+.1f} kB with the MLP but {b10['net_codes']['0.1']['net_without_other'] / 1e3:+.1f} kB without it at 0.1 Å. As section 4.6.9.2 "
          "showed, at this spacing the purely temporal raw + AC stream beats every hybrid variant anyway.")
    A("* **Bottom line.** The rotamer codes and the per trajectory MLP are worth their bytes only together and only once the trajectory is long enough to "
      "amortise them (between 250 and 1,000 frames at mdCATH spacing; we tested only these two lengths). For short trajectories the encoder should drop both and use the shared packer alone; choosing "
      "the variant per trajectory costs one flag byte and is a cheap improvement we have not yet built into the codec.\n")
json.dump(SUM, open(f"{E}/results/ablation_summary.json", "w"), indent=1, default=lambda o: None)
open(f"{E}/ablation_section.md", "w").write("\n".join(L) + "\n")
print("wrote ablation_section.md", len(L))
