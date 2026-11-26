#!/usr/bin/env python
"""Pool stability-split side-chain metrics (results_gpu/eval/stability_raw.json), write them into
results_gpu/eval/results.json (per system), results_gpu/summary.json (pooled), summary_stability.md
(also inserted into summary.md between markers) and figures."""
import json, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results_gpu"
raw = json.load(open(f"{R}/eval/stability_raw.json"))
crit = raw.pop("_criteria", None)
CLS = ("stable", "intermediate", "unstable", "not_stable", "all", "stable_buried", "stable_exposed",
       "not_stable_buried", "not_stable_exposed")
METHODS = [("floor", "Ideal-geometry floor (true χ)"), ("static_rotamers", "Mean rotamers (stored prior)"),
           ("type_library", "One rotamer per type"),
           ("staged_S_prior/e2e", "Staged S, prior"), ("e2e_S_prior/e2e", "End-to-end S, prior"),
           ("staged_L_prior/e2e", "Staged L, prior"), ("staged_L_prior/true_bb", "Staged L, prior (true bb)"),
           ("staged_S_noprior/e2e", "Staged S, no prior"), ("e2e_S_noprior/e2e", "End-to-end S, no prior"),
           ("staged_L_noprior/e2e", "Staged L, no prior"), ("staged_S_prior/true_bb", "Staged S, prior (true bb)"),
           ("staged_S_noprior/true_bb", "Staged S, no prior (true bb)"), ("staged_L_noprior/true_bb", "Staged L, no prior (true bb)")]


def finish(sums):
    """sums (one or several summarise() dicts, pooled) -> metric dict"""
    if not sums or sum(s.get("n_res", 0) for s in sums) == 0:
        return dict(n_res=0)
    g = lambda k: float(sum(s[k] for s in sums if s.get("n_res")))
    gi = lambda k, i: float(sum(s[k][i] for s in sums if s.get("n_res")))
    nrt = float(sum(s["n_res"] * s["T"] for s in sums if s.get("n_res")))
    n = g("n_res")
    return dict(n_res=int(n), sc_rmsd=(g("sc_err") / g("n_sc")) ** 0.5,
                chi_mae=[gi("chi_err", i) / gi("n_chi", i) if gi("n_chi", i) else None for i in range(4)],
                chi1_acc40=g("chi1_ok") / nrt, rotamer_state_acc=g("state_ok") / nrt, joint_state_acc=g("joint_ok") / nrt,
                sc_rmsf_local_ratio=g("rmsf_p") / g("rmsf_t"), sc_rmsf_local_true=g("rmsf_t") / n,
                sc_rmsf_local_recon=g("rmsf_p") / n, chi1_population_jsd=g("jsd") / n,
                transitions_per_ns_true=g("rate_t") / n, transitions_per_ns_recon=g("rate_p") / n)


# ------------------------------------------------------------------ per system -> results.json
rpath = f"{R}/eval/results.json"
res = json.load(open(rpath))
for key, r in raw.items():
    if key not in res:
        continue
    cl = np.array(r["residues"]["cls"]); has = np.array(r["residues"]["has_chi"])
    cv = np.array(r["residues"]["chi1_circvar"]); nb = np.array(r["residues"]["cb_neighbours"])
    res[key]["sidechain_stability"] = dict(
        n_frames=r["n_frames"], ns_per_frame=r["ns_per_frame"],
        counts={c: int((cl == k).sum()) for c, k in (("stable", "S"), ("intermediate", "I"), ("unstable", "U"))},
        fractions={c: float((cl == k).sum() / max(has.sum(), 1)) for c, k in (("stable", "S"), ("intermediate", "I"), ("unstable", "U"))},
        residue_class=r["residues"]["cls"], dominant_occupancy=r["residues"]["occupancy"],
        transitions_per_ns=r["residues"]["transitions_per_ns"], chi1_circular_variance=r["residues"]["chi1_circvar"],
        cb_neighbours_10A=r["residues"]["cb_neighbours"], criteria=crit,
        note="metrics under sidechain_by_stability use these contiguous frames, not the 40 frames of the main metrics")
    for m, s in r["methods"].items():
        name, mode = (m.split("/") + [None])[:2]
        tgt = res[key].get(name)
        if tgt is None:
            continue
        if mode:
            tgt = tgt.get(mode)
            if tgt is None:
                continue
        tgt["sidechain_by_stability"] = {c: finish([s[c]]) for c in CLS if c in s}
json.dump(res, open(rpath + ".tmp", "w"), indent=1)
os.replace(rpath + ".tmp", rpath)

# ------------------------------------------------------------------ pooled per set
pooled, lines = {}, []
for set_name, title in (("mdcath_test", "mdCATH test (50 domains, held-out superfamilies)"), ("external", "External set (AdK + 7 ATLAS)")):
    keys = [k for k, v in raw.items() if v.get("set") == set_name]
    if not keys:
        continue
    cl = np.concatenate([raw[k]["residues"]["cls"] for k in keys]); has = cl != "-"
    cv = np.concatenate([raw[k]["residues"]["chi1_circvar"] for k in keys])
    nb = np.concatenate([raw[k]["residues"]["cb_neighbours"] for k in keys])
    occ = np.concatenate([raw[k]["residues"]["occupancy"] for k in keys])
    rate = np.concatenate([raw[k]["residues"]["transitions_per_ns"] for k in keys])
    P = pooled[set_name] = dict(n_proteins=len(keys), n_residues_with_chi=int(has.sum()), classes={}, methods={})
    for c, k in (("stable", "S"), ("intermediate", "I"), ("unstable", "U")):
        sel = cl == k
        P["classes"][c] = dict(n=int(sel.sum()), fraction=float(sel.sum() / has.sum()),
                               exposed_fraction=float((nb[sel] < 16).mean()) if sel.any() else None,
                               mean_occupancy=float(occ[sel].mean()) if sel.any() else None,
                               mean_transitions_per_ns=float(rate[sel].mean()) if sel.any() else None,
                               mean_chi1_circvar=float(cv[sel].mean()) if sel.any() else None)
    P["exposed_fraction_all"] = float((nb[has] < 16).mean())
    for m, _ in METHODS:
        if all(m in raw[k]["methods"] for k in keys):
            P["methods"][m] = {c: finish([raw[k]["methods"][m][c] for k in keys]) for c in CLS}
    fr = P["classes"]
    nfr = sorted({raw[k]["n_frames"] for k in keys})
    lines += [f"### {title}", "",
              f"{has.sum()} residues with χ; frames per protein {nfr[0]} to {nfr[-1]}, contiguous. Classes from the true MD: "
              f"stable {fr['stable']['fraction']:.0%}, intermediate {fr['intermediate']['fraction']:.0%}, unstable {fr['unstable']['fraction']:.0%}. "
              f"Exposed (fewer than 16 CB neighbours within 10 Å): {fr['stable']['exposed_fraction']:.0%} of stable, "
              f"{fr['intermediate']['exposed_fraction']:.0%} of intermediate, {fr['unstable']['exposed_fraction']:.0%} of unstable residues. "
              f"Mean χ1 circular variance: stable {fr['stable']['mean_chi1_circvar']:.2f}, intermediate {fr['intermediate']['mean_chi1_circvar']:.2f}, unstable {fr['unstable']['mean_chi1_circvar']:.2f}.", "",
              "| method | sc RMSD S / I / U (Å) | χ1 within 40° S / I / U | rotamer state acc. S / I / U | sc RMSF recon/true S / I / U | χ1 population JSD S / I / U | transitions/ns true → recon (U) |",
              "|---|---|---|---|---|---|---|"]
    for m, label in METHODS:
        if m not in P["methods"]:
            continue
        q = P["methods"][m]
        f3 = lambda k, fmt: " / ".join(fmt.format(q[c][k]) for c in ("stable", "intermediate", "unstable"))
        lines.append(f"| {label} | {f3('sc_rmsd', '{:.2f}')} | {f3('chi1_acc40', '{:.0%}')} | {f3('rotamer_state_acc', '{:.0%}')} | "
                     f"{f3('sc_rmsf_local_ratio', '{:.2f}')} | {f3('chi1_population_jsd', '{:.3f}')} | "
                     f"{q['unstable']['transitions_per_ns_true']:.3f} → {q['unstable']['transitions_per_ns_recon']:.3f} |")
    lines.append("")
    lines += ["Binary split and burial (sc RMSD Å, χ1 within 40°):", "",
              "| method | stable | not stable | stable buried | stable exposed | not stable buried | not stable exposed |", "|---|---|---|---|---|---|---|"]
    for m, label in METHODS:
        if m not in P["methods"]:
            continue
        q = P["methods"][m]
        cell = lambda c: f"{q[c]['sc_rmsd']:.2f}, {q[c]['chi1_acc40']:.0%}" if q[c]["n_res"] else ""
        lines.append(f"| {label} | " + " | ".join(cell(c) for c in ("stable", "not_stable", "stable_buried", "stable_exposed",
                                                                     "not_stable_buried", "not_stable_exposed")) + " |")
    lines.append("")
hdr = ["## Side-chain accuracy by rotamer stability", "",
       "Residues with χ (GLY, ALA excluded) are classified per trajectory from the TRUE MD only: χ1 wells p (+60°), t (180°), m (−60°) "
       "(PRO: two puckers), plus χ2 for sp3 χ2 (ARG GLN GLU ILE LEU LYS MET); core-to-core assignment with a ±40° core (a frame changes state only when it "
       "enters another well's core). Stable: dominant state occupancy ≥ 90% and ≤ 0.02 transitions per ns. Unstable: occupancy < 70%. Intermediate: the rest. "
       "S / I / U = stable / intermediate / unstable. Rotamer state accuracy: nearest χ1 well of the prediction equals that of the MD, per frame. "
       "sc RMSF is computed with atoms in their own residue frame, so only side-chain motion counts. JSD: Jensen-Shannon divergence (bits) of χ1 well populations per residue. "
       "Transition rates use the same core-to-core assignment on predicted χ. Frames: mdCATH replica 0, contiguous, 2 ns apart; ATLAS 1 ns apart; AdK is a DIMS path (time nominal).", ""]
md = "\n".join(hdr + lines)
open(f"{R}/summary_stability.md", "w").write(md + "\n")
sm = open(f"{R}/summary.md").read()
A, B = "<!-- stability:start -->", "<!-- stability:end -->"
if A in sm:
    sm = sm[:sm.index(A)] + sm[sm.index(B) + len(B):]
open(f"{R}/summary.md", "w").write(sm.rstrip() + "\n\n" + A + "\n" + md + "\n" + B + "\n")
sj = json.load(open(f"{R}/summary.json"))
sj["sidechain_by_stability"] = dict(criteria=crit, pooled=pooled)
json.dump(sj, open(f"{R}/summary.json", "w"), indent=1)
json.dump(dict(criteria=crit, pooled=pooled), open(f"{R}/eval/sidechain_by_stability_summary.json", "w"), indent=1)
print(md)

# ------------------------------------------------------------------ figure: bars per method per class
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
show = [("floor", "floor"), ("static_rotamers", "mean rotamers"), ("type_library", "type library"),
        ("e2e_S_prior/e2e", "e2e S prior"), ("staged_L_prior/e2e", "staged L prior"),
        ("e2e_S_noprior/e2e", "e2e S no prior"), ("staged_L_noprior/e2e", "staged L no prior")]
panels = [("sc_rmsd", "side-chain RMSD (Å)"), ("chi1_acc40", "χ1 within 40°"), ("rotamer_state_acc", "χ1 rotamer state accuracy"),
          ("sc_rmsf_local_ratio", "side-chain RMSF recon / true"), ("chi1_population_jsd", "χ1 population JSD (bits)")]
cols = {"stable": "#2F64B0", "intermediate": "#E0A526", "unstable": "#C0392B"}
fig, axes = plt.subplots(len(pooled), len(panels), figsize=(20, 4.2 * len(pooled)), squeeze=False)
for i, (s, P) in enumerate(pooled.items()):
    ms = [(m, l) for m, l in show if m in P["methods"]]
    y = np.arange(len(ms))
    for j, (k, t) in enumerate(panels):
        ax = axes[i, j]
        for n, c in enumerate(("stable", "intermediate", "unstable")):
            ax.barh(y + (n - 1) * 0.27, [P["methods"][m][c][k] for m, _ in ms], 0.27, color=cols[c], label=c)
        ax.set_yticks(y); ax.set_yticklabels([l for _, l in ms] if j == 0 else []); ax.invert_yaxis()
        if k == "sc_rmsf_local_ratio":
            ax.axvline(1, color="#666", lw=0.8, ls="--")
        ax.set_title(t, fontsize=10); ax.grid(axis="x", alpha=0.3)
        if j == 0:
            ax.set_ylabel({"mdcath_test": "mdCATH test", "external": "external"}[s], fontsize=11)
axes[0, 0].legend(loc="lower right", fontsize=8)
plt.tight_layout()
os.makedirs("report/gpu_figs", exist_ok=True)
plt.savefig(f"{R}/fig_sidechain_stability.png", dpi=130); plt.savefig("report/gpu_figs/fig_sidechain_stability.png", dpi=130)
print("figure written")
