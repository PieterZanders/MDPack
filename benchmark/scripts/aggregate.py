#!/usr/bin/env python
"""Pool every raw result file into results/benchmark_summary.json, markdown tables (results/tables.md) and figures."""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B
R = f"{B.BENCH}/results"; F = f"{B.BENCH}/figs"; os.makedirs(F, exist_ok=True)
DOMS = json.load(open("/workspace/md_compress/results_gpu/frames/selected_domains.json"))
MDDB = json.load(open("/workspace/bench_data/mddb_meta.json"))
rows = {}
# bench_ours.py chdirs to /workspace/md_compress, so relative --out paths land in /workspace/md_compress/results
for f in sorted(glob.glob(f"{R}/*_raw*.json")) + sorted(glob.glob("/workspace/md_compress/results/ours_raw_*.json")):
    rows.update(json.load(open(f)))
# AEZ-12: arithmetic coder rows (only the "+ AC" variants; the lzma rows recomputed in that run stay out of the main tables so AEZ-11 numbers are unchanged)
for f in sorted(glob.glob(f"{B.BENCH}/entropy/results/entropy_raw_*.json")):
    rows.update({k: v for k, v in json.load(open(f)).items() if "+ AC" in k or k.endswith("/ERROR")})
timing = {}
for f in sorted(glob.glob(f"{R}/timing_*.json")):
    timing.update(json.load(open(f)))
ok = {k: v for k, v in rows.items() if "error" not in v}
err = {k: v for k, v in rows.items() if "error" in v}


def family(m):
    if m.startswith("ours"): return "ours"
    if m.startswith("model free"): return "model free"
    if m.startswith("MDCompress"): return "MDCompress"
    if m.startswith("pcazip"): return "pcazip"
    if m.startswith("compresstraj"): return "compresstraj"
    if m.startswith("cg2all"): return "cg2all"
    return "format"


def bound(m):
    for tag in ("@ 0.005", "@ 0.5", "@ 0.3", "@ 0.2", "@ 0.1"):
        if tag in m: return float(tag[2:])
    return None


def guarantee(m):
    """Error guarantee of a method on this data (source coordinates lie on a 0.01 A grid)."""
    f = family(m)
    if f in ("ours", "model free"):
        if "@ 0.005" in m: return "lossless (0.01 A grid)"
        b = bound(m)
        return f"bounded, max {b:g} A" if b else "lossy, no bound"
    if f == "MDCompress":
        res = int(m.split("--res ")[1].split()[0])
        return "lossless (0.01 A grid)" if res <= 1000 else f"bounded, max {res / 2e5:g} A"
    if f == "format":
        if m == "XTC precision 100": return "bounded, max 0.05 A"
        if m == "XTC precision 10": return "bounded, max 0.5 A"
        if m.startswith("XTC") or m.startswith("TNG") or m.startswith("PDB"): return "lossless (0.01 A grid)"
        return "lossless (float32)"
    if f == "pcazip": return "bounded by residual step (fitted frame)" if "resid" in m else "lossy, no bound"
    return "lossy, no bound"


def pool(keys):
    vs = [ok[k] for k in keys]
    mt = lambda q: [v["metrics"].get(q) for v in vs if v["metrics"].get(q) is not None]
    o = dict(n=len(vs), ratio_mean=float(np.mean([v["ratio"] for v in vs])), ratio_pooled=sum(v["raw_bytes"] for v in vs) / sum(v["bytes"] for v in vs),
             bytes_mean=float(np.mean([v["bytes"] for v in vs])), encode_s=float(np.mean([v["encode_s"] for v in vs])), decode_s=float(np.mean([v["decode_s"] for v in vs])))
    for q in ("rmsd_lab", "rmsd_fit", "bb_rmsd_lab", "sc_rmsd_lab", "other_rmsd_lab", "rmsf_r", "rmsip10"):
        if mt(q): o[q] = float(np.mean(mt(q)))
    o["max_abs_err"] = float(max(mt("max_abs_err")))
    return o


summary = {"mdcath": {}, "mddb": {}, "failures": {k: v["error"][-600:] for k, v in err.items()}}
for T in (1000, 250):
    meths = sorted({v["method"] for v in ok.values() if v["system"] in DOMS and v["T"] == T})
    tab = {}
    for m in meths:
        keys = [f"{d}/T{T}/{m}" for d in DOMS if f"{d}/T{T}/{m}" in ok]
        if len(keys) < len(DOMS): continue  # only complete methods
        e = pool(keys); e["family"] = family(m); e["guarantee"] = guarantee(m)
        xtc = [ok[f"{d}/T{T}/XTC precision 1000"]["bytes"] / ok[f"{d}/T{T}/{m}"]["bytes"] for d in DOMS if f"{d}/T{T}/XTC precision 1000" in ok]
        if len(xtc) == len(DOMS): e["ratio_vs_xtc"] = float(np.mean(xtc))
        tk = [timing[f"{d}/T{T}/{m}"] for d in DOMS if f"{d}/T{T}/{m}" in timing]
        if tk: e["clean_encode_s"] = float(np.mean([t["encode_s"] for t in tk])); e["clean_decode_s"] = float(np.mean([t["decode_s"] for t in tk])); e["clean_n"] = len(tk)
        tab[m] = e
    summary["mdcath"][T] = tab
# subset of domains where some methods (compresstraj, cut off by the Colab time limit) are incomplete: pool every method that is complete on that subset
summary["mdcath_subset"] = {}
for T in (1000, 250):
    allm = sorted({v["method"] for v in ok.values() if v["system"] in DOMS and v["T"] == T})
    inc = [m for m in allm if 0 < sum(f"{d}/T{T}/{m}" in ok for d in DOMS) < len(DOMS)]
    for fam in sorted({family(m) for m in inc}):
        sub = [d for d in DOMS if all(f"{d}/T{T}/{m}" in ok for m in inc if family(m) == fam)]
        if not sub: continue
        tab = {}
        for m in allm:
            keys = [f"{d}/T{T}/{m}" for d in sub if f"{d}/T{T}/{m}" in ok]
            if len(keys) < len(sub): continue
            e = pool(keys); e["family"] = family(m); e["guarantee"] = guarantee(m)
            xtc = [ok[f"{d}/T{T}/XTC precision 1000"]["bytes"] / ok[f"{d}/T{T}/{m}"]["bytes"] for d in sub if f"{d}/T{T}/XTC precision 1000" in ok]
            if len(xtc) == len(sub): e["ratio_vs_xtc"] = float(np.mean(xtc))
            tab[m] = e
        summary["mdcath_subset"][f"{T}/{fam}"] = dict(domains=sub, missing=[d for d in DOMS if d not in sub], methods=tab)
for s in MDDB:
    meths = sorted({v["method"] for v in ok.values() if v["system"] == s})
    tab = {}
    for m in meths:
        k = f"{s}/T1000/{m}"
        if k not in ok: continue
        v = ok[k]; e = dict(ratio=v["ratio"], bytes=v["bytes"], encode_s=v["encode_s"], decode_s=v["decode_s"], family=family(m), guarantee=guarantee(m), **v["metrics"])
        if f"{s}/T1000/XTC precision 1000" in ok: e["ratio_vs_xtc"] = ok[f"{s}/T1000/XTC precision 1000"]["bytes"] / v["bytes"]
        if "note" in v: e["note"] = v["note"]
        tab[m] = e
    summary["mddb"][s] = dict(meta=MDDB[s], methods=tab)
json.dump(summary, open(f"{R}/benchmark_summary.json", "w"), indent=1)

# ------------------------------------------------------------------ tables
ORDER = ["ours", "model free", "MDCompress", "format", "pcazip", "compresstraj", "cg2all"]
def srt(tab): return sorted(tab, key=lambda m: (ORDER.index(tab[m]["family"]), -tab[m].get("ratio_mean", tab[m].get("ratio", 0))))
L = []
for T in (1000, 250):
    tab = summary["mdcath"][T]
    if not tab: continue
    L += [f"\n### mdCATH, 16 domains, T = {T} frames (means over domains; max error = max over domains)\n",
          "| method | guarantee | ratio vs raw float32 | ratio vs XTC | all atom RMSD lab (Å) | RMSD fitted (Å) | backbone (Å) | side chain (Å) | max abs err (Å) | RMSF r | PCA RMSIP10 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in srt(tab):
        e = tab[m]
        L.append(f"| {m} | {e['guarantee']} | {e['ratio_mean']:.2f} | {e.get('ratio_vs_xtc', float('nan')):.2f} | {e['rmsd_lab']:.3f} | {e['rmsd_fit']:.3f} | "
                 f"{e.get('bb_rmsd_lab', float('nan')):.3f} | {e.get('sc_rmsd_lab', float('nan')):.3f} | {e['max_abs_err']:.3f} | {e['rmsf_r']:.4f} | {e['rmsip10']:.4f} |")
for s, d in summary["mddb"].items():
    mm = d["meta"]
    L += [f"\n### {mm['kind']}: {s} ({mm['accession']}, {mm['name']}), {mm['n_heavy']} heavy atoms, {mm['T']} frames; model covers {mm['n_protein_atoms_model']} protein atoms ({mm['n_protein_residues_model']} residues), {mm['n_other_atoms']} other atoms\n",
          "| method | guarantee | ratio vs raw | ratio vs XTC | RMSD lab (Å) | RMSD fitted (Å) | backbone (Å) | side chain / base (Å) | other (Å) | max abs err (Å) | RMSF r | RMSIP10 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in srt(d["methods"]):
        e = d["methods"][m]; g = lambda q: e.get(q, float("nan"))
        L.append(f"| {m} | {e['guarantee']} | {e['ratio']:.2f} | {g('ratio_vs_xtc'):.2f} | {e['rmsd_lab']:.3f} | {e['rmsd_fit']:.3f} | {g('bb_rmsd_lab'):.3f} | {g('sc_rmsd_lab'):.3f} | {g('other_rmsd_lab'):.3f} | {e['max_abs_err']:.3f} | {e['rmsf_r']:.4f} | {e['rmsip10']:.4f} |")
for key, d in summary["mdcath_subset"].items():
    T, fam = key.split("/")
    L += [f"\n### mdCATH subset for {fam}, T = {T}: {len(d['domains'])} of 16 domains (missing: {', '.join(d['missing'])}); every method pooled over the same subset\n",
          "| method | guarantee | ratio vs raw float32 | ratio vs XTC | all atom RMSD lab (Å) | RMSD fitted (Å) | backbone (Å) | side chain (Å) | max abs err (Å) | RMSF r | PCA RMSIP10 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tab = d["methods"]
    for m in srt(tab):
        e = tab[m]
        L.append(f"| {m} | {e['guarantee']} | {e['ratio_mean']:.2f} | {e.get('ratio_vs_xtc', float('nan')):.2f} | {e['rmsd_lab']:.3f} | {e['rmsd_fit']:.3f} | "
                 f"{e.get('bb_rmsd_lab', float('nan')):.3f} | {e.get('sc_rmsd_lab', float('nan')):.3f} | {e['max_abs_err']:.3f} | {e['rmsf_r']:.4f} | {e['rmsip10']:.4f} |")
L += ["\n### Failures\n"] + [f"* {k}: {v[:300].strip()}" for k, v in summary["failures"].items()]
open(f"{R}/tables.md", "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"ours": "tab:green", "model free": "tab:blue", "MDCompress": "tab:red", "format": "#555555", "pcazip": "tab:orange", "compresstraj": "tab:purple", "cg2all": "tab:brown"}
def series(tab):
    g = {}
    for m, e in tab.items():
        f = e["family"]
        if f == "ours" and "model only" in m: g.setdefault("ours: model only", []).append((m, e))
        elif f == "ours" and "backbone AE" in m: g.setdefault("ours: hybrid + backbone AE + AC coder", []).append((m, e))
        elif f == "ours": g.setdefault("ours: hybrid + AC coder" if "+ AC" in m else "ours: hybrid + residuals", []).append((m, e))
        elif f == "model free": g.setdefault(("model free mean" if "mean" in m else "model free raw") + (" + AC coder" if "+ AC" in m else ""), []).append((m, e))
        elif f == "MDCompress": g.setdefault("MDCompress -l 4 archive" if "archive" in m else ("MDCompress -l 4" if "-l 4" in m else "MDCompress -l 2"), []).append((m, e))
        elif f == "pcazip": g.setdefault("pcazip PCA only" if "PCA only" in m else "pcazip + residuals", []).append((m, e))
        elif f == "compresstraj": g.setdefault("compresstraj (+ decoder)" if "decoder" in m else "compresstraj (latents)", []).append((m, e))
        elif f == "cg2all": g.setdefault("cg2all", []).append((m, e))
        else: g.setdefault("formats", []).append((m, e))
    return g
SCOL = {"ours: model only": ("tab:green", "*"), "ours: hybrid + residuals": ("tab:green", "o"), "model free mean": ("tab:blue", "s"),
        "model free raw": ("tab:cyan", "s"), "ours: hybrid + AC coder": ("darkgreen", "o"), "ours: hybrid + backbone AE + AC coder": ("olive", "p"), "model free mean + AC coder": ("navy", "s"), "model free raw + AC coder": ("teal", "s"), "MDCompress -l 2": ("tab:red", "v"), "MDCompress -l 4": ("darkred", "^"), "MDCompress -l 4 archive": ("magenta", "^"), "pcazip PCA only": ("tab:orange", "D"),
        "pcazip + residuals": ("gold", "D"), "compresstraj (latents)": ("tab:purple", "P"), "compresstraj (+ decoder)": ("plum", "X"),
        "cg2all": ("tab:brown", "h"), "formats": ("#444444", "x")}
def pareto(tab, title, fn, key_x="rmsd_fit"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for name, pts in series(tab).items():
        c, mk = SCOL[name]
        pts = sorted(pts, key=lambda p: p[1][key_x])
        xs = [p[1][key_x] for p in pts]; ys = [p[1].get("ratio_mean", p[1].get("ratio")) for p in pts]
        for a_ in ax: a_.plot(xs, ys, marker=mk, ls=("--" if "AC coder" in name else "-") if name not in ("formats", "cg2all") else "none", c=c, label=name, ms=7)
        if name == "formats":
            for p, xx, yy in zip(pts, xs, ys): ax[0].annotate(p[0].replace(" (float32)", "").replace("precision ", "p"), (xx, yy), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[0].set(xscale="symlog", yscale="log", xlabel=f"all atom RMSD, {'per frame superposed' if key_x == 'rmsd_fit' else 'lab frame'} (Å, symlog; 0 = lossless)", ylabel="compression ratio vs raw float32 (×, log)")
    ax[0].set_xlim(-0.001, 5); ax[0].set_title(title, fontsize=10)
    ax[1].set(xlim=(-0.02, 0.62), ylim=(0, None), xlabel="all atom RMSD (Å), zoom on 0 to 0.6 Å", ylabel="compression ratio (×)"); ax[1].set_title("zoom: lossless and error bounded region", fontsize=10)
    zy = [e.get("ratio_mean", e.get("ratio")) for e in tab.values() if e[key_x] <= 0.62]
    if zy: ax[1].set_ylim(0, 1.15 * max(zy))
    for a_ in ax: a_.grid(alpha=0.3)
    ax[1].legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(fn, dpi=180); plt.close(fig)
def with_subset(T):
    """16 domain table plus the incomplete families pooled on their own subset (marked in the figure titles)"""
    t = dict(summary["mdcath"][T]); notes = []
    for key, d in summary["mdcath_subset"].items():
        TT, fam = key.split("/")
        if int(TT) != T: continue
        for m, e in d["methods"].items():
            if e["family"] == fam and m not in t: t[m] = e
        notes.append(f"{fam}: {len(d['domains'])} of 16 domains")
    return t, notes
for T in (1000, 250):
    if summary["mdcath"][T]:
        t, notes = with_subset(T)
        pareto(t, f"mdCATH 16 domains, T = {T}: ratio vs error (all methods" + ("; " + "; ".join(notes) if notes else "") + ")", f"{F}/fig_pareto_mdcath_T{T}.png")
# bar chart at comparable error
def pick(tab, cands):
    """best (highest mean ratio) of the listed alternatives that are present"""
    c = [m for m in cands if m in tab]
    return max(c, key=lambda m: tab[m]["ratio_mean"]) if c else None
LEV = [("lossless (0.01 Å grid)", {"ours: hybrid": ["ours: hybrid @ 0.005"], "model free mean": ["model free mean @ 0.005"], "MDCompress (best setting)": ["MDCompress -l 4 --preset archive --res 1000 fm", "MDCompress -l 4 --res 1000 fm", "MDCompress -l 2 --res 1000 fm (default)"],
        "XTC": ["XTC precision 1000"], "TNG": ["TNG (GROMACS trjconv)"], "H5MD gzip": ["H5MD (gzip 9, chunk 100 frames)"], "PDB gzip": ["PDB + gzip -9"], "TRR": ["TRR (float32)"]}),
       ("max error 0.1 Å", {"ours: hybrid": ["ours: hybrid @ 0.1"], "model free mean": ["model free mean @ 0.1"], "MDCompress (best setting)": ["MDCompress -l 4 --preset archive --res 20000 fm", "MDCompress -l 4 --res 20000 fm", "MDCompress -l 2 --res 20000 fm"]}),
       ("max error 0.5 Å", {"ours: hybrid": ["ours: hybrid @ 0.5"], "model free mean": ["model free mean @ 0.5"], "MDCompress (best setting)": ["MDCompress -l 4 --preset archive --res 100000 fm", "MDCompress -l 4 --res 100000 fm", "MDCompress -l 2 --res 100000 fm"], "XTC p10": ["XTC precision 10"]}),
       ("lossy, RMSD about 0.3 to 1 Å", {"ours: model only": ["ours: model only"], "pcazip PCA 95%": ["pcazip EV 0.95 PCA only"], "pcazip PCA 90%": ["pcazip EV 0.90 PCA only"],
        "compresstraj c20 (latents)": ["compresstraj c 20 (latents + COG + scaler)"], "compresstraj c20 (+ decoder)": ["compresstraj c 20 (+ decoder weights)"],
        "cg2all CA p100": ["cg2all CA trace stored as XTC precision 100"], "cg2all CA p1000": ["cg2all CA trace stored as XTC precision 1000"]})]
tab, _notes = with_subset(1000)
if tab:
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    for a_, (lab, cands) in zip(ax, LEV):
        names, vals, errs, cols = [], [], [], []
        for nm, cs in cands.items():
            m = pick(tab, cs)
            if m: names.append(nm); vals.append(tab[m]["ratio_mean"]); errs.append(tab[m]["rmsd_fit"]); cols.append(COL[tab[m]["family"]])
        a_.bar(range(len(vals)), vals, color=cols)
        for i, (v, e_) in enumerate(zip(vals, errs)): a_.text(i, v, f"{v:.1f}×\n{e_:.2f} Å", ha="center", va="bottom", fontsize=7)
        a_.set_xticks(range(len(vals))); a_.set_xticklabels(names, rotation=40, ha="right", fontsize=8); a_.set_title(lab, fontsize=10)
        a_.set_ylabel("ratio vs raw float32 (×)"); a_.set_ylim(0, max(vals + [1]) * 1.25); a_.grid(axis="y", alpha=0.3)
    fig.suptitle("mdCATH 16 domains, T = 1000: compression ratio at comparable error (label: ratio and fitted all atom RMSD)" + (" [" + "; ".join(_notes) + "]" if _notes else ""), fontsize=11)
    fig.tight_layout(); fig.savefig(f"{F}/fig_bars_comparable_error.png", dpi=180); plt.close(fig)
    # timing
    sel = ["ours: hybrid @ 0.1", "ours: hybrid @ 0.005", "ours: model only", "model free mean @ 0.1", "MDCompress -l 2 --res 1000 fm (default)", "MDCompress -l 4 --res 1000 fm",
           "XTC precision 1000", "TNG (GROMACS trjconv)", "TRR (float32)", "PDB (text)", "PDB + gzip -9", "H5MD (uncompressed)", "H5MD (gzip 9, chunk 100 frames)",
           "pcazip EV 0.95 PCA only", "pcazip EV 0.95 + residuals (scale 200/nm)", "compresstraj c 20 (latents + COG + scaler)", "cg2all CA trace stored as XTC precision 1000"]
    sel = [m for m in sel if m in tab]
    fig, ax = plt.subplots(figsize=(12, 6)); y = np.arange(len(sel))
    ge = lambda m, q: tab[m].get(f"clean_{q}", tab[m][q])
    ax.barh(y - 0.2, [ge(m, "encode_s") for m in sel], 0.4, label="encode", color="tab:blue"); ax.barh(y + 0.2, [ge(m, "decode_s") for m in sel], 0.4, label="decode", color="tab:orange")
    ax.set_yticks(y); ax.set_yticklabels([m + (" [GPU T4]" if tab[m]["family"] in ("compresstraj", "cg2all") else "") for m in sel], fontsize=8); ax.set_xscale("log")
    ax.set_xlabel("wall time per trajectory of 1000 frames (s, log); mean over domains"); ax.legend(); ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    ax.set_title("Encode and decode time (box CPU unless marked GPU; compresstraj encode includes training)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{F}/fig_timing.png", dpi=180); plt.close(fig)
# MDDB per system bars
if summary["mddb"]:
    systems = list(summary["mddb"]); cands = [("ours: hybrid @ 0.005", "ours lossless"), ("MDCompress -l 4 --res 1000 fm", "MDCompress lossless"), ("XTC precision 1000", "XTC"),
            ("ours: hybrid @ 0.1", "ours 0.1 Å"), ("MDCompress -l 4 --res 20000 fm", "MDCompress 0.1 Å"), ("ours: hybrid @ 0.5", "ours 0.5 Å"), ("MDCompress -l 4 --res 100000 fm", "MDCompress 0.5 Å"),
            ("model free mean @ 0.5", "mean+res 0.5 Å"), ("pcazip EV 0.95 PCA only", "pcazip 95%")]
    fig, ax = plt.subplots(figsize=(16, 6)); w = 0.8 / len(cands)
    cmap = plt.get_cmap("tab10")
    for i, (m, lab) in enumerate(cands):
        vals = [summary["mddb"][s]["methods"].get(m, {}).get("ratio", np.nan) for s in systems]
        if m.startswith("ours") and "model" not in m:
            vals = [summary["mddb"][s]["methods"].get(m, summary["mddb"][s]["methods"].get(m.replace("ours: hybrid", "model free mean"), {})).get("ratio", np.nan) for s in systems]
        ax.bar(np.arange(len(systems)) + i * w - 0.4 + w / 2, vals, w, label=lab, color=cmap(i))
    ax.set_xticks(range(len(systems))); ax.set_xticklabels([f"{summary['mddb'][s]['meta']['kind']}\n{summary['mddb'][s]['meta']['accession']}" for s in systems], fontsize=8)
    ax.set_ylabel("ratio vs raw float32 (×)"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    ax.set_title("MDDB systems (T = 1000): ratio by method (systems without protein: 'ours' = model free mean + residuals)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{F}/fig_mddb_systems.png", dpi=180); plt.close(fig)
print("ok", len(ok), "records", len(err), "failures")
