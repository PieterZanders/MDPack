#!/usr/bin/env python
"""Tables (markdown + json) and figures from results_gpu/eval/results.json and the run histories."""
import glob, json, os, sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results_gpu"
res = json.load(open(f"{R}/eval/results.json"))
METHODS = [("floor", "Ideal-geometry floor (true χ)"), ("static_rotamers", "Per-protein mean rotamers (stored prior only)"),
           ("type_library", "One rotamer per residue type (no per-protein data)"),
           ("staged_S_prior", "Staged S, χ prior"), ("e2e_S_prior", "End-to-end S, χ prior"),
           ("staged_L_prior", "Staged L, χ prior"), ("e2e_L_prior", "End-to-end L, χ prior"),
           ("staged_S_noprior", "Staged S, no prior"), ("e2e_S_noprior", "End-to-end S, no prior"),
           ("staged_L_noprior", "Staged L, no prior"), ("e2e_L_noprior", "End-to-end L, no prior")]


def get(r, k, mode="e2e"):
    v = r.get(k)
    if v is None:
        return None
    return v[mode] if isinstance(v, dict) and mode in v else (v if "sc_rmsd" in v else None)


def agg(keys, k, mode="e2e"):
    rows = [get(res[x], k, mode) for x in keys]
    rows = [r for r in rows if r]
    if not rows:
        return None
    out = {m: float(np.mean([r[m] for r in rows])) for m in ("bb_rmsd", "sc_rmsd", "all_heavy", "chi1_acc40", "clash")}
    out["chi_mae"] = [float(np.mean([r["chi_mae"][i] for r in rows])) for i in range(4)]
    out["n"] = len(rows)
    return out


summary = {}
lines = []
for set_name, title in (("mdcath_test", "mdCATH test domains (held-out superfamilies)"), ("external", "External set (AdK + 7 ATLAS)")):
    keys = [k for k, v in res.items() if v.get("set") == set_name]
    if not keys:
        continue
    md_clash = float(np.mean([res[k]["md_clash"] for k in keys]))
    comp = np.array([res[k]["compression"]["ratio"] for k in keys])
    n_res = np.array([res[k]["n_res"] for k in keys])
    ident = [res[k].get("max_seq_identity_to_train") for k in keys if res[k].get("max_seq_identity_to_train") is not None]
    S = summary[set_name] = dict(n_proteins=len(keys), n_res_range=[int(n_res.min()), int(n_res.max())],
                                 md_clash=md_clash, compression_mean=float(comp.mean()),
                                 compression_range=[float(comp.min()), float(comp.max())],
                                 max_seq_identity_to_train=dict(mean=float(np.mean(ident)), max=float(np.max(ident))) if ident else None,
                                 methods={})
    lines += [f"### {title}: {len(keys)} proteins, {n_res.min()} to {n_res.max()} residues", "",
              f"Compression (float32 heavy atoms / fp16 latent + rigid): {comp.mean():.1f}x on average "
              f"(range {comp.min():.1f} to {comp.max():.1f}). MD clash reference: {md_clash:.1f} per 1000 atoms." +
              (f" Highest sequence identity to any training domain: mean {np.mean(ident):.0%}, max {np.max(ident):.0%}." if ident else ""), "",
              "| method | bb RMSD (Å) | sc RMSD (Å) | χ1 / χ2 / χ3 / χ4 MAE (°) | χ1 within 40° | clash /1000 |",
              "|---|---|---|---|---|---|"]
    for k, label in METHODS:
        for mode in (("e2e", "true_bb") if k.startswith("staged") else ("e2e",)):
            a = agg(keys, k, mode)
            if a is None:
                continue
            lab = label + (" (true backbone)" if mode == "true_bb" else "")
            S["methods"][f"{k}/{mode}"] = a
            ch = " / ".join(f"{c:.0f}" for c in a["chi_mae"])
            lines.append(f"| {lab} | {a['bb_rmsd']:.3f} | {a['sc_rmsd']:.3f} | {ch} | {a['chi1_acc40']:.0%} | {a['clash']:.1f} |")
    lines.append("")
# per-protein external table
ext = sorted(k for k, v in res.items() if v.get("set") == "external")
if ext:
    lines += ["### External set per protein (sc RMSD Å, end-to-end decode)", "",
              "| protein | res | ratio | floor | mean rotamers | S prior | L prior | S no prior | L no prior |", "|---|---|---|---|---|---|---|---|---|"]
    best_s = "e2e_S_prior" if "e2e_S_prior" in res[ext[0]] else "staged_S_prior"
    for k in ext:
        r = res[k]
        f = lambda m: f"{get(r, m)['sc_rmsd']:.2f}" if get(r, m) else "n/a"
        lines.append(f"| {k.split('/')[1]} | {r['n_res']} | {r['compression']['ratio']:.1f}x | {f('floor')} | {f('static_rotamers')} | "
                     f"{f(best_s)} | {f('e2e_L_prior' if 'e2e_L_prior' in r else 'staged_L_prior')} | "
                     f"{f('e2e_S_noprior' if 'e2e_S_noprior' in r else 'staged_S_noprior')} | {f('e2e_L_noprior' if 'e2e_L_noprior' in r else 'staged_L_noprior')} |")
    lines.append("")
# training runs
runs = {}
for c in sorted(glob.glob(f"{R}/runs/*/config.json")):
    d = os.path.dirname(c)
    cfg = json.load(open(c))
    h = json.load(open(f"{d}/history.json")) if os.path.exists(f"{d}/history.json") else []
    runs[os.path.basename(d)] = dict(config=cfg, steps=h[-1]["step"] if h else 0, minutes=h[-1]["minutes"] if h else 0,
                                     final_val=h[-1]["val"] if h else None,
                                     best_val=min(h, key=lambda x: x["val"].get("sc_rmsd", x["val"]["bb_rmsd"]))["val"] if h else None)
summary["runs"] = runs
lines += ["### Training runs", "", "| run | params (side-chain stack) | steps | train min | best val |", "|---|---|---|---|---|"]
for n, r in runs.items():
    bv = r["best_val"] or {}
    lines.append(f"| {n} | {r['config']['params']['sidechain_stack_total']:,} | {r['steps']:,} | {r['minutes']:.0f} | "
                 + ", ".join(f"{k} {v:.3f}" for k, v in bv.items() if k in ("bb_rmsd", "sc_rmsd", "chi1_acc40")) + " |")
open(f"{R}/summary.md", "w").write("\n".join(lines) + "\n")
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
print("\n".join(lines))

# figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, s in zip(axes, ("mdcath_test", "external")):
        if s not in summary:
            continue
        ms = [(k, v) for k, v in summary[s]["methods"].items() if k.endswith("/e2e")]
        labels = [k.split("/")[0].replace("_", " ") for k, _ in ms]
        ax.barh(labels, [v["sc_rmsd"] for _, v in ms], color=["#888" if not k.startswith(("staged", "e2e")) else
                                                               ("#2a7" if "noprior" not in k else "#c63") for k, _ in ms])
        ax.set_xlabel("side-chain RMSD (Å), decoded from latents")
        ax.set_title({"mdcath_test": "mdCATH held-out superfamilies", "external": "External set (AdK + ATLAS)"}[s])
        ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(f"{R}/fig_sc_rmsd.png", dpi=150)
    # training curves: train loss and validation metrics vs wall-clock minutes for every run
    names = list(runs)
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.5), squeeze=False)
    for j, n in enumerate(names):
        h = json.load(open(f"{R}/runs/{n}/history.json"))
        mins = [x["minutes"] for x in h]
        a = axes[0, j]
        a.plot(mins, [x["train"]["loss"] for x in h], c="k")
        a.set_title(n, fontsize=9); a.set_xlabel("minutes (L4)"); a.set_ylabel("train loss"); a.set_yscale("log")
        b = axes[1, j]
        if "sc_rmsd" in h[0]["val"]:
            b.plot(mins, [x["val"]["sc_rmsd"] for x in h], c="tab:red", label="val sc RMSD (Å)")
            b2 = b.twinx(); b2.plot(mins, [100 * x["val"]["chi1_acc40"] for x in h], c="tab:green", label="χ1 within 40° (%)")
            b2.set_ylabel("χ1 within 40° (%)", color="tab:green", fontsize=8)
            if n.startswith("e2e"):
                b.plot(mins, [x["val"]["bb_rmsd"] for x in h], c="tab:blue", label="val bb RMSD (Å)")
        else:
            b.plot(mins, [x["val"]["bb_rmsd"] for x in h], c="tab:blue", label="val bb RMSD (Å)")
        b.set_xlabel("minutes (L4)"); b.legend(fontsize=7, loc="upper right")
    plt.tight_layout(); plt.savefig(f"{R}/fig_training_curves.png", dpi=110)
except Exception as e:
    print("figures skipped:", e)
