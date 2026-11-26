#!/usr/bin/env python
"""Scaling figure: compression and error vs protein size and number of frames (report/gpu_figs/fig_scaling.png)."""
import json, shutil
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

res = json.load(open("results_gpu/eval/results.json"))
ts = json.load(open("results_gpu/eval/error_vs_time.json"))
keys = [k for k in res if not k.startswith("_") and "compression" in res[k]]
isx = np.array([res[k]["set"] == "external" for k in keys])
R = np.array([res[k]["n_res"] for k in keys]); H = np.array([res[k]["n_heavy"] for k in keys])
ratio = np.array([res[k]["compression"]["ratio"] for k in keys])
hpr = float(np.median(H / R))
pf = lambda r, h=hpr: 12 * h * r / (8 * r + 28)                       # per-frame ratio
amort = lambda r, T, h=hpr: 12 * h * r * T / ((8 * r + 28) * T + 80 * r)  # incl. one-off ref bb + chi prior
out = dict(heavy_atoms_per_residue_median=hpr, formula=dict(
    raw_bytes_per_frame="12 * n_heavy (fp32 xyz)", stored_bytes_per_frame="8 * n_res (4 fp16 latents) + 28 (rigid transform)",
    one_off_per_protein_bytes="80 * n_res (reference backbone 48 B/res + chi prior 32 B/res)", shared_weights="excluded"))

fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
c_m, c_x = "tab:blue", "tab:orange"
# A
a = ax[0, 0]; rr = np.linspace(40, 260, 200)
a.scatter(R[~isx], ratio[~isx], s=18, c=c_m, label="mdCATH test (50)")
a.scatter(R[isx], ratio[isx], s=30, c=c_x, marker="s", label="external (8)")
a.plot(rr, pf(rr), "k--", lw=1, label=f"formula, {hpr:.2f} heavy atoms/res")
a.axhline(1.5 * hpr, c="grey", lw=0.6, ls=":"); a.text(258, 1.5 * hpr + 0.05, "large-protein limit, 12 h / 8 B = 1.5 h", ha="right", va="bottom", fontsize=8, c="grey")
rA = stats.pearsonr(H / R, ratio)[0]
a.set(xlabel="protein size (residues)", ylabel="compression ratio per frame (×)",
      title="A  Compression vs size (per frame, measured)"); a.legend(fontsize=8, loc="lower right")
a.text(0.03, 0.04, f"r with size {stats.pearsonr(R, ratio)[0]:.2f}\nr with heavy atoms/res {rA:.2f}", transform=a.transAxes, fontsize=8, va="bottom")
out["A"] = dict(ratio_min=float(ratio.min()), ratio_max=float(ratio.max()), ratio_median=float(np.median(ratio)),
                r_ratio_vs_size=float(stats.pearsonr(R, ratio)[0]), r_ratio_vs_heavy_per_res=float(rA))
# B
a = ax[0, 1]; T = np.logspace(0, 4, 300); out["B"] = {}
for r_, c in zip((100, 200, 300), ("tab:green", "tab:purple", "tab:red")):
    a.plot(T, amort(r_, T), c=c, label=f"{r_} residues"); a.axhline(pf(r_), c=c, ls=":", lw=0.8)
    s = 8 * r_ + 28; t90 = 9 * 80 * r_ / s
    out["B"][str(r_)] = dict(per_frame=pf(r_), T1=amort(r_, 1), T10=amort(r_, 10), T100=amort(r_, 100), T1000=amort(r_, 1000), frames_to_90pct=t90)
a.axvline(90, c="grey", lw=0.6, ls="--"); a.text(95, 1.2, "≈90 frames: 90% of\nper-frame ratio", fontsize=8, c="grey")
a.set(xscale="log", xlabel="frames stored (T)", ylabel="compression ratio incl. per-protein data (×)", ylim=(0, None),
      title="B  Compression vs frames (analytical; size barely matters)"); a.legend(fontsize=8, loc="upper left")
# C
a = ax[1, 0]; out["C"] = {}
for met, mk, lab in (("sc_rmsd", "o", "side chain"), ("bb_rmsd", "^", "backbone")):
    for meth, ls, alpha in (("e2e_S_prior", "-", 1.0), ("staged_L_prior", "--", 0.35)):
        y = np.array([res[k][meth]["e2e"][met] for k in keys]); fit = stats.linregress(R, y)
        col = "tab:red" if met == "sc_rmsd" else "tab:blue"
        if meth == "e2e_S_prior":
            a.scatter(R[~isx], y[~isx], s=16, c=col, marker=mk, alpha=0.8)
            a.scatter(R[isx], y[isx], s=26, c=col, marker=mk, edgecolors="k", linewidths=0.8)
        a.plot(rr, fit.intercept + fit.slope * rr, c=col, ls=ls, alpha=alpha, lw=1.4,
               label=f"{lab}, {'E2E S' if meth=='e2e_S_prior' else 'staged L'}: r = {fit.rvalue:.2f}, {100*fit.slope:+.2f} Å/100 res")
        sub = {}
        for nm, sel in (("mdcath", ~isx), ("external", isx)):
            f2 = stats.linregress(R[sel], y[sel]); sub[nm] = dict(r=f2.rvalue, p=f2.pvalue, slope_per_100res=100 * f2.slope)
        out["C"][f"{meth}/{met}"] = dict(r=fit.rvalue, p=fit.pvalue, slope_per_100res=100 * fit.slope, intercept=fit.intercept, **sub)
a.set(xlabel="protein size (residues)", ylabel="RMSD (Å)", ylim=(0, None),
      title="C  Error vs size (58 proteins; black edge = external)"); a.legend(fontsize=7.5, loc="upper right", framealpha=0.9); a.set_ylim(0, 2.75)
# D
a = ax[1, 1]; out["D"] = {}
mk_ = [k for k in ts if ts[k]["set"] == "mdcath_test"]
for met, col, lab in (("sc_rmsd", "tab:red", "side chain"), ("bb_rmsd", "tab:blue", "backbone")):
    Tm = max(len(ts[k]["time_ns"]) for k in mk_)
    M = np.full((len(mk_), Tm), np.nan)
    for i, k in enumerate(mk_):
        y = np.array(ts[k]["methods"]["e2e_S_prior"][met]); M[i, :len(y)] = y / y.mean()
    t = np.arange(Tm) * 2.0
    w = 10; nb = Tm // w; tb = t[:nb * w].reshape(nb, w).mean(1)
    Mb = np.nanmean(M[:, :nb * w].reshape(len(mk_), nb, w), 2)
    ok = (~np.isnan(Mb)).sum(0) >= 20; tb, Mb = tb[ok], Mb[:, ok]
    q1, med, q3 = np.nanpercentile(Mb, [25, 50, 75], axis=0)
    a.plot(tb, med, c=col, label=lab); a.fill_between(tb, q1, q3, color=col, alpha=0.15)
    slopes, rs_ref = [], []
    for k in mk_:
        y = np.array(ts[k]["methods"]["e2e_S_prior"][met]); tt = np.array(ts[k]["time_ns"])
        f = stats.linregress(tt, y); slopes.append(100 * 100 * f.slope / y.mean())  # percent of own mean per 100 ns
        rs_ref.append(stats.pearsonr(ts[k]["true_bb_rmsd_from_ref"], y)[0])
    pooled = stats.pearsonr(np.concatenate([np.array(ts[k]["time_ns"]) for k in mk_]),
                            np.concatenate([np.array(ts[k]["methods"]["e2e_S_prior"][met]) / np.mean(ts[k]["methods"]["e2e_S_prior"][met]) for k in mk_]))[0]
    first = np.concatenate([np.array(ts[k]["methods"]["e2e_S_prior"][met][:25]) for k in mk_]).mean()
    last = np.concatenate([np.array(ts[k]["methods"]["e2e_S_prior"][met][-25:]) for k in mk_]).mean()
    out["D"][met] = dict(n_proteins=len(mk_), frames_max=Tm, proteins_per_bin_min=20, median_slope_pct_per_100ns=float(np.median(slopes)),
                         iqr_slope_pct_per_100ns=[float(np.percentile(slopes, 25)), float(np.percentile(slopes, 75))],
                         pooled_r_time_vs_relative_error=float(pooled), mean_first_50ns=float(first), mean_last_50ns=float(last),
                         median_r_error_vs_true_bb_dev_from_ref=float(np.median(rs_ref)))
a.axhline(1, c="grey", lw=0.6)
a.set(xlabel="simulation time (ns), mdCATH replica 0", ylabel="per-frame RMSD / protein's own mean",
      title="D  Error over trajectory time (E2E S, 50 proteins)", ylim=(0.6, 1.4))
a.legend(fontsize=8, title="median, IQR band (20 ns bins)", title_fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("report/gpu_figs/fig_scaling.png", dpi=200)
shutil.copy("report/gpu_figs/fig_scaling.png", "results_gpu/fig_scaling.png")
json.dump(out, open("results_gpu/eval/scaling_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
# size effect on sc RMSD vs exposed fraction (from stability_raw.json)
s = json.load(open("results_gpu/eval/stability_raw.json"))
ks = [k for k in keys if k in s]
Rk = np.array([res[k]["n_res"] for k in ks], float); yk = np.array([res[k]["e2e_S_prior"]["e2e"]["sc_rmsd"] for k in ks])
ex = np.array([np.mean(np.array(s[k]["residues"]["cb_neighbours"]) < 16) for k in ks])
rsd = lambda u, v: u - np.polyval(np.polyfit(v, u, 1), v)
out["C"]["note"] = dict(exposed="fewer than 16 CB within 10 A (frame 0)", r_size_vs_exposed_fraction=float(stats.pearsonr(Rk, ex)[0]),
                        r_sc_vs_exposed_fraction=float(stats.pearsonr(yk, ex)[0]),
                        partial_r_size_sc_given_exposed=float(stats.pearsonr(rsd(Rk, ex), rsd(yk, ex))[0]),
                        partial_p=float(stats.pearsonr(rsd(Rk, ex), rsd(yk, ex))[1]))
json.dump(out, open("results_gpu/eval/scaling_summary.json", "w"), indent=1, default=float)
