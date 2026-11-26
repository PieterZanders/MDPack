"""Build results_multi/summary.md (+ summary.json) from results_multi/state.json."""
import json
import numpy as np

S = json.load(open("results_multi/state.json"))
sy = S["systems"]
names = list(sy)
f = lambda v, d=2: f"{v:.{d}f}"
out, summ = [], {"systems": {}, "held_out": {}}

out.append("### Systems\n")
out.append("| system | description | residues | heavy atoms | frames used | source |")
out.append("|---|---|---|---|---|---|")
for n in names:
    s = sy[n]
    fr = f"{s['n_frames']} (every {s['stride']}th of 10001, 1 ns spacing)" if s["stride"] > 1 else f"{s['n_frames']}"
    out.append(f"| {n} | {s['desc']} | {s['n_res']} | {s['n_heavy']} | {fr} | {s['source']} |")

out.append("\n### (a) Per-protein interpolation (one model per system, interleaved test frames, with χ prior)\n")
out.append("| system | BB RMSD | SC RMSD | χ1 MAE (°) | χ1 <40° | clash (MD ref) | static-rotamer SC RMSD / χ1 MAE / χ1<40° / clash | ideal-geometry floor SC RMSD / clash | compression (per frame) |")
out.append("|---|---|---|---|---|---|---|---|---|")
for n in names:
    k = f"per_protein/{n}"
    if k not in S:
        continue
    e, st, fl = S[k]["e2e"], sy[n]["static_rotamers"], sy[n]["floor"]
    c = S[k]["compression"]["ratio_per_frame"]
    out.append(f"| {n} | {f(e['bb_rmsd'])} | {f(e['sc_rmsd'])} | {f(e['chi1_mae'],1)} | {100*e['chi1_acc40']:.1f}% | {f(e['clash'],1)} ({f(sy[n]['md_clash'],1)}) | "
               f"{f(st['sc_rmsd'])} / {f(st['chi1_mae'],1)} / {100*st['chi1_acc40']:.1f}% / {f(st['clash'],1)} | {f(fl['sc_rmsd'])} / {f(fl['clash'],1)} | {c:.1f}× |")
    summ["systems"][n] = dict(per_protein=e, static=st, floor=fl, md_clash=sy[n]["md_clash"], compression=S[k]["compression"])

out.append("\n### (b) Held-out proteins (4 folds; shared packer and codec trained on the other 6 systems)\n")
out.append("Prior = mean rotamers of the held-out system, computed from its own trajectory and stored with its topology (R×8 floats, counted in storage). No prior = backbone + sequence only.\n")
out.append("| held-out system | fold | BB RMSD | SC RMSD prior / no prior | χ1 MAE prior / no prior | χ1 <40° prior / no prior | clash prior / no prior (MD ref) | static rotamers SC RMSD / χ1<40° | per-protein model SC RMSD | compression incl. prior |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
rows = []
for fi in range(4):
    fo = S.get(f"fold{fi}")
    if not fo or "eval" not in fo:
        continue
    for n, r in fo["eval"].items():
        if not r["held_out"]:
            continue
        p, q = r["prior"]["e2e"], r["noprior"]["e2e"]
        st = sy[n]["static_rotamers"]
        pp = S.get(f"per_protein/{n}", {}).get("e2e", {})
        comp = r["compression_prior"]
        # per-frame storage incl. the prior amortised over this trajectory
        lat = comp["stored_bytes_per_frame"]
        prior_bytes = sy[n]["n_res"] * 8 * 4
        ratio_prior = comp["raw_bytes_per_frame"] * sy[n]["n_frames"] / (lat * sy[n]["n_frames"] + prior_bytes + sy[n]["n_res"] * 12 * 4)
        out.append(f"| {n} | {fi} | {f(p['bb_rmsd'])} | {f(p['sc_rmsd'])} / {f(q['sc_rmsd'])} | {f(p['chi1_mae'],1)} / {f(q['chi1_mae'],1)} | "
                   f"{100*p['chi1_acc40']:.1f}% / {100*q['chi1_acc40']:.1f}% | {f(p['clash'],1)} / {f(q['clash'],1)} ({f(sy[n]['md_clash'],1)}) | "
                   f"{f(st['sc_rmsd'])} / {100*st['chi1_acc40']:.1f}% | {f(pp.get('sc_rmsd', float('nan')))} | {comp['ratio_per_frame']:.1f}× ({ratio_prior:.1f}× incl. ref+prior, excl. shared weights) |")
        rows.append(dict(name=n, fold=fi, prior=p, noprior=q, static=st, per_protein=pp, ratio=comp["ratio_per_frame"], ratio_incl_ref_prior=ratio_prior))
        summ["held_out"][n] = rows[-1]
if rows:
    m = lambda key, tag: np.mean([r[tag][key] for r in rows])
    out.append(f"| **mean** | | {f(m('bb_rmsd','prior'))} | {f(m('sc_rmsd','prior'))} / {f(m('sc_rmsd','noprior'))} | {f(m('chi1_mae','prior'),1)} / {f(m('chi1_mae','noprior'),1)} | "
               f"{100*m('chi1_acc40','prior'):.1f}% / {100*m('chi1_acc40','noprior'):.1f}% | {f(m('clash','prior'),1)} / {f(m('clash','noprior'),1)} | "
               f"{f(m('sc_rmsd','static'))} / {100*m('chi1_acc40','static'):.1f}% | {f(np.mean([r['per_protein']['sc_rmsd'] for r in rows if r['per_protein']]))} | |")
# in-distribution (training systems, test frames) for reference
out.append("\n### (b, reference) Same fold models evaluated on two of their training systems (test frames)\n")
out.append("| system | fold | BB RMSD | SC RMSD prior / no prior | χ1 <40° prior / no prior |")
out.append("|---|---|---|---|---|")
for fi in range(4):
    fo = S.get(f"fold{fi}")
    if not fo or "eval" not in fo:
        continue
    for n, r in fo["eval"].items():
        if r["held_out"]:
            continue
        p, q = r["prior"]["e2e"], r["noprior"]["e2e"]
        out.append(f"| {n} | {fi} | {f(p['bb_rmsd'])} | {f(p['sc_rmsd'])} / {f(q['sc_rmsd'])} | {100*p['chi1_acc40']:.1f}% / {100*q['chi1_acc40']:.1f}% |")
open("results_multi/summary.md", "w").write("\n".join(out) + "\n")
json.dump(summ, open("results_multi/summary.json", "w"), indent=1)
print("\n".join(out))
