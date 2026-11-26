#!/usr/bin/env python3
"""Concise MDPack report (current code, one medium protein) from benchmark/results/medium_<system>_raw.json (+ _side.json) and
benchmark/entropy/results/codec_method_tests.json (test counts). Output: report_mdpack/MDPack_report.{md,html,pdf} and figures.
Run with system python3 (markdown, weasyprint, matplotlib)."""
import json, os, re, datetime
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); OUT = f"{ROOT}/report_mdpack"; os.makedirs(f"{OUT}/figs", exist_ok=True)
import sys
SYS = sys.argv[1] if len(sys.argv) > 1 else "4m70B00"
D = {}
for fn in (f"{ROOT}/benchmark/results/medium_{SYS}_raw.json", f"{ROOT}/benchmark/results/medium_{SYS}_side.json"):
    if os.path.exists(fn): D.update(json.load(open(fn)))
NATOM = next(r["raw_bytes"] // 12 // r["T"] for r in D.values() if "raw_bytes" in r)
SYSLAB = f"mdCATH {SYS}, {NATOM:,} heavy atoms, T = 1,000" if not SYS.startswith("mddb") else f"{SYS}, {NATOM:,} heavy atoms, T = 1,000"
EXTRA = json.load(open(f"{OUT}/report_text.json")) if os.path.exists(f"{OUT}/report_text.json") else {}
BL = ["0.5", "0.3", "0.2", "0.1", "lossless"]


def g(key):
    r = D.get(key); return r if r and "error" not in r else None


def ratio(key, show_err=False):
    r = g(key)
    if r is None: return "not run" if key not in D else "failed"
    s = f"{r['ratio']:.2f}"
    return s


def f(v, n=3):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    return f"{v:.{n}f}"


BOUNDED = [("MDPack default (hybrid)", "MDPack default (`hybrid`)"), ("new-coder mean", "MDPack `new-coder --nc-predictor mean` (model free)"),
           ("new-coder raw", "MDPack `new-coder --nc-predictor raw` (model free)"), ("dvorak", "Dvořák et al. 2020 (PMC 1.0)"), ("mdz", "MDZ (SZ3)"),
           ("mdcompress -l 2", "MDCompress -l 2"), ("mdcompress -l 4 archive", "MDCompress -l 4 --preset archive"),
           ("lzma", "quantise + lzma"), ("zstd", "quantise + zstd"), ("bz2", "quantise + bz2"), ("gzip", "quantise + gzip"),
           ("h5md sz3", "H5MD + SZ3 filter (hdf5plugin)"), ("xtc", "XTC"), ("tng", "TNG"), ("pdb", "PDB text")]
LOSSLESS_ALIAS = {"xtc": "T1000/xtc precision 1000", "tng": "T1000/tng precision 1000"}


def cell(name, b):
    if b == "lossless":
        k = LOSSLESS_ALIAS.get(name, f"T1000/{name} @ lossless")
    else: k = f"T1000/{name} @ {b}"
    r = g(k)
    if r is None:
        return "n/a" if (b == "lossless" and name in ("dvorak", "mdz", "h5md sz3", "pdb")) else ("failed" if k in D else "")
    mx = r["metrics"]["max_abs_err"]; lim = 0.005 if b == "lossless" else float(b)
    s = f"{r['ratio']:.2f}"
    if mx < 0.8 * lim and b != "lossless": s += f" (max {mx:.3f})"
    return s


RMSD_DEF = ("Backbone and side chain RMSD: decompressed against original, per frame RMSD in the lab frame (no superposition), "
            "mean over frames, Å. Backbone = N, CA, C, O of every residue (and the C terminal OXT); side chain = all other protein heavy atoms. "
            "Atoms that are chemically equivalent (for example Asp OD1/OD2, Phe CD1/CD2) are matched in the way that gives the smaller error; "
            "this only changes the numbers of the methods without a bound. Computed from the decoded trajectories of the runs in this report.")


def bbsc(r):
    if r is None: return ""
    m = r["metrics"]; return f"{m['bb_rmsd_lab']:.3f} / {m['sc_rmsd_lab']:.3f}"


def bbsc_cell(name, b):
    k = LOSSLESS_ALIAS.get(name, f"T1000/{name} @ lossless") if b == "lossless" else f"T1000/{name} @ {b}"
    r = g(k)
    return bbsc(r) if r else ("n/a" if b == "lossless" else "")


def best_mark(rows):  # bold the best ratio per column among the bounded rows
    for j in range(1, 6):
        vals = []
        for r in rows:
            m = re.match(r"([0-9.]+)", r[j]); vals.append(float(m.group(1)) if m else -1)
        k = int(np.argmax(vals))
        if vals[k] > 0: rows[k][j] = f"**{rows[k][j]}**"
    return rows


def table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


L = []
A = L.append
A("# MDPack: error bounded compression of molecular dynamics trajectories\n")
A(f"Report on the current code (MDPack, package `md_compress`), {datetime.date.today():%d %B %Y}. All numbers are from one medium sized protein, "
  "measured with the code in this repository; compression ratios are float32 size over file size and exclude the shared model weights.\n")

A("## 1. What MDPack is\n")
A(EXTRA.get("intro", ""))

A("## 2. The default codec\n")
A(EXTRA.get("codec", ""))

A("## 3. Methods and flags\n")
A(EXTRA.get("methods", ""))

A("## 4. The .mdz file and the JSON report\n")
A(EXTRA.get("format", ""))

A("## 5. Running and reproducing\n")
A(EXTRA.get("running", ""))

A("## 6. Test system and measurements\n")
A(EXTRA.get("setup", ""))

# ---------------------------------------------------------------- 7.1 main comparison
A("## 7. Results\n")
A("### 7.1 Every method on the same trajectory (T = 1,000 frames)\n")
A("Compression ratio at each guaranteed maximum absolute error per coordinate. Lossless = the 0.01 Å grid of the source "
  "(max error 0.005 Å; the source coordinates lie on this grid within 0.0005 Å, so it reproduces the stored data). "
  "Where a method's grid gives a smaller error than the column bound, its actual maximum error is shown in brackets "
  "(TNG and PDB precisions are powers of ten; Dvořák's grid is in a rotated local frame). n/a: the method has no lossless mode. "
  "Best per column in bold.\n")
rows = [[lab] + [cell(n, b) for b in BL] for n, lab in BOUNDED]
rows = best_mark(rows)
tm = []
for n, lab in BOUNDED[:7] + [("xtc", "XTC")]:
    r = g(f"T1000/{n} @ 0.1")
    if r: tm.append(f"{lab}: {r['encode_s']:.1f} s / {r['decode_s']:.1f} s")
A(table(["method", "0.5 Å", "0.3 Å", "0.2 Å", "0.1 Å", "lossless"], rows))
A("\nEncode / decode wall time at 0.1 Å (8 core box, 4 threads, container and external process start included): " + "; ".join(tm) + ".\n")
A("Backbone / side chain RMSD (Å) of the same runs (definition below the table):\n")
A(table(["method", "0.5 Å", "0.3 Å", "0.2 Å", "0.1 Å", "lossless"], [[lab] + [bbsc_cell(n, b) for b in BL] for n, lab in BOUNDED]))
A(RMSD_DEF + "\n")
A("![Ratio against error bound](figs/fig_ratio_bound.png)\n")
A("Exact (float32) formats and methods without an error bound, same trajectory:\n")
EX = [("trr {}", "TRR"), ("dcd {}", "DCD"), ("netcdf {}", "AMBER NetCDF"), ('h5md {"h5md_compression": "none"}', "H5MD, uncompressed"),
      ('h5md {"h5md_chunk": 1, "h5md_compression": "gzip", "h5md_level": 4}', "H5MD, gzip 4, 1 frame chunks"),
      ('h5md {"h5md_chunk": 100, "h5md_compression": "gzip", "h5md_level": 9}', "H5MD, gzip 9, 100 frame chunks"),
      ("pdb {}", "PDB text (3 decimals, max 0.0005 Å)"), ('pdb {"pdb_gzip": 9}', "PDB + gzip 9"),
      ("lzma {} @ exact", "float32 bits + lzma"), ("zstd {} @ exact", "float32 bits + zstd"),
      ("model-only", "MDPack `model-only` (no residuals)"),
      ("pcazip EV 0.7 PCA only", "pcazip, 70 % variance, PCA only"), ("pcazip EV 0.9 PCA only", "pcazip, 90 %, PCA only"),
      ("pcazip EV 0.95 PCA only", "pcazip, 95 %, PCA only"), ("pcazip EV 0.95 + residuals 20/nm", "pcazip, 95 % + residuals (20 per nm)"),
      ("pcazip EV 0.75 + residuals 200/nm", "pcazip, 75 % + residuals (200 per nm)"),
      ("compresstraj c 20", "compresstraj, compression 20 (decoder weights counted)"), ("cg2all CA", "cg2all, CA trace in XTC (T = 100)")]
rows = []
for k, lab in EX:
    T = 100 if k.startswith("cg2all") else 1000; r = g(f"T{T}/{k}")
    if r is None: rows.append([lab, "not run" if f"T{T}/{k}" not in D else "failed", "", "", "", "", "", ""]); continue
    m = r["metrics"]; rows.append([lab, f"{r['ratio']:.2f}", f(m["rmsd_lab"]), f(m["bb_rmsd_lab"]), f(m["sc_rmsd_lab"]), f(m["max_abs_err"], 2), f(m["rmsf_r"], 4), f(m["rmsip10"], 4)])
A(table(["method", "ratio", "RMSD all (Å)", "RMSD backbone (Å)", "RMSD side chain (Å)", "max error (Å)", "RMSF r", "RMSIP top 10 PCs"], rows))
A(EXTRA.get("comparison_notes", ""))

# ---------------------------------------------------------------- 7.2 ablation
A("### 7.2 Ablation: model only, model + quantisation, quantisation only\n")
AB = [("model-only", "MDPack model only: backbone AE latents + rotamer codes + per trajectory MLP, no residuals"),
      ("MDPack default (hybrid)", "MDPack model + quantisation: mean structure backbone + shared packer, residuals on the bound grid, MDPack coder (default)"),
      ("new-coder mean", "quantisation only, mean structure predictor + MDPack coder (no learned model)"),
      ("new-coder raw", "quantisation only, raw coordinates + MDPack coder"),
      ("lzma", "quantisation only, raw coordinates, time delta + lzma")]
rows = []
for n, lab in AB:
    if n == "model-only":
        r = g("T1000/model-only")
        rows.append([lab] + ([f"{r['ratio']:.2f} (RMSD {r['metrics']['rmsd_lab']:.2f} Å, max {r['metrics']['max_abs_err']:.1f} Å, no bound)"] + [""] * 4 if r else ["not run"] + [""] * 4))
    else: rows.append([lab] + [cell(n, b) for b in BL])
A(table(["variant", "0.5 Å", "0.3 Å", "0.2 Å", "0.1 Å", "lossless"], rows))
rows = []
for n, lab in AB:
    if n == "model-only": rows.append([lab.split(":")[0], bbsc(g("T1000/model-only")) + " (no bound)", "", "", "", ""])
    else: rows.append([lab.split(":")[0].split(",")[0] if n != "MDPack default (hybrid)" else "MDPack model + quantisation (default)"] + [bbsc_cell(n, b) for b in BL])
rows[2][0] = "quantisation only, mean structure + MDPack coder"; rows[3][0] = "quantisation only, raw + MDPack coder"; rows[4][0] = "quantisation only, raw + lzma"
A("Backbone / side chain RMSD (Å) of the same variants:\n")
A(table(["variant", "0.5 Å", "0.3 Å", "0.2 Å", "0.1 Å", "lossless"], rows))
A(EXTRA.get("ablation_notes", ""))

# ---------------------------------------------------------------- 7.3 fidelity
A("### 7.3 Dynamical fidelity of the decompressed trajectory\n")
A(f"Decompressed against original, all {NATOM:,} heavy atoms unless stated. RMSD lab: per frame, no superposition, mean over frames; "
  "RMSD fit: after per frame superposition; RMSF r and MAE: per atom fluctuations after fitting each trajectory on its own mean CA structure; "
  "RMSIP: overlap of the top 10 CA principal components (1 = identical subspace); Rg MAE: radius of gyration, mean absolute difference per frame; "
  "RMSD(t) r: correlation of the CA RMSD to frame 1 time series; contacts: CA pairs within 8 Å (|i − j| ≥ 3), mean absolute difference of the contact frequency.\n")
FID = [("T1000/MDPack default (hybrid) @ 0.5", "MDPack default, 0.5 Å"), ("T1000/new-coder mean @ 0.5", "new-coder mean, 0.5 Å"),
       ("T1000/dvorak @ 0.5", "Dvořák 2020, 0.5 Å"), ("T1000/mdz @ 0.5", "MDZ, 0.5 Å"), ("T1000/mdcompress -l 2 @ 0.5", "MDCompress -l 2, 0.5 Å"),
       ("T1000/xtc @ 0.5", "XTC, 0.5 Å"), ("T1000/MDPack default (hybrid) @ 0.1", "MDPack default, 0.1 Å"), ("T1000/mdz @ 0.1", "MDZ, 0.1 Å"),
       ("T1000/dvorak @ 0.1", "Dvořák 2020, 0.1 Å"), ("T1000/model-only", "MDPack model only (no bound)"),
       ("T1000/pcazip EV 0.9 PCA only", "pcazip 90 %, PCA only (no bound)"), ("T1000/compresstraj c 20", "compresstraj c 20 (no bound)"),
       ("T100/cg2all CA", "cg2all CA (no bound, T = 100)")]
rows = []
for k, lab in FID:
    r = g(k)
    if r is None: continue
    m = r["metrics"]
    rows.append([lab, f"{r['ratio']:.2f}", f(m["rmsd_lab"]), f(m["rmsd_fit"]), f(m["rmsf_r"], 4), f(m["rmsf_mae"]), f(m["rmsip10"], 4), f(m["rg_mae"]),
                 f(m.get("rmsd_to_frame1_r"), 4), f(m["contact_freq_mae"], 4)])
A(table(["trajectory", "ratio", "RMSD lab", "RMSD fit", "RMSF r", "RMSF MAE", "RMSIP", "Rg MAE", "RMSD(t) r", "contacts MAE"], rows))
A(EXTRA.get("fidelity_notes", ""))

# ---------------------------------------------------------------- 7.4 frames
A("### 7.4 Number of frames\n")
A("The first T frames of the same trajectory (frame spacing unchanged). Ratio at 0.1 Å and lossless; the error of every bounded method "
  "stays at its bound for every T (largest observed max error / bound in the last column); for the model only variant the RMSD is given instead.\n")
TS = [10, 50, 100, 250, 500, 1000]
FM = [("MDPack default (hybrid)", "MDPack"), ("new-coder raw", "new-coder raw"), ("dvorak", "Dvořák"), ("mdz", "MDZ"),
      ("mdcompress -l 2", "MDC -l 2"), ("mdcompress -l 4 archive", "MDC -l 4 arch."), ("xtc", "XTC"), ("lzma", "lzma")]
rows = []; FIG = {}
for T in TS:
    row = [str(T)]; worst = 0
    for b in ("0.1", "lossless"):
        for n, lab in FM:
            if b == "lossless" and n in ("dvorak", "mdz", "xtc"): continue
            r = g(f"T{T}/{n} @ {b}")
            row.append(f"{r['ratio']:.2f}" if r else "")
            if r:
                FIG.setdefault((n, b), []).append((T, r["ratio"])); lim = 0.005 if b == "lossless" else float(b)
                worst = max(worst, r["metrics"]["max_abs_err"] / lim)
    r = g(f"T{T}/model-only"); row.append(f"{r['ratio']:.1f} ({r['metrics']['rmsd_lab']:.2f} Å)" if r else "")
    row.append(f"{worst:.4f}" if worst else ""); rows.append(row)
hd = ["T"] + [f"{lab} 0.1 Å" for n, lab in FM] + [f"{lab} lossless" for n, lab in FM if n not in ("dvorak", "mdz", "xtc")] + ["model only ratio (RMSD)", "max error / bound"]
A(table(hd, rows))
FR = [("MDPack default (hybrid)", "MDPack"), ("new-coder raw", "new-coder raw"), ("dvorak", "Dvořák"), ("mdz", "MDZ"),
      ("mdcompress -l 4 archive", "MDC -l 4 arch."), ("xtc", "XTC")]
rows = []
for T in TS:
    rows.append([str(T)] + [bbsc(g(f"T{T}/{n} @ 0.1")) for n, lab in FR] + [bbsc(g(f"T{T}/model-only"))])
A("Backbone / side chain RMSD (Å) at 0.1 Å against T (bounded methods; lossless runs are on the source grid, RMSD below 0.0001 Å), and of the model only variant:\n")
A(table(["T"] + [lab for n, lab in FR] + ["model only (no bound)"], rows))
A("![Ratio against number of frames](figs/fig_frames.png)\n")
A(EXTRA.get("frames_notes", ""))

# ---------------------------------------------------------------- 7.5 von Mises
A("### 7.5 Circular (von Mises) statistics for the side chain torsions\n")
A(EXTRA.get("vonmises_text", ""))
rows = []
for am, lab in (("mean", "circular mean (default; single von Mises MLE)"), ("vonmises", "von Mises mixture mode (K = 3)"),
                ("empirical", "histogram mode"), ("gaussian", "arithmetic (non circular) mean")):
    pre = "T1000/MDPack default (hybrid)" if am == "mean" else f"T1000/MDPack default, angle model {am}"
    row = [lab]
    for b in BL:
        r = g(f"{pre} @ {b}"); row.append(f"{r['ratio']:.3f}" if r else "")
    r = g("T1000/model-only" if am == "mean" else f"T1000/model-only, angle model {am}")
    row.append(f"{r['metrics']['sc_rmsd_lab']:.3f}" if r else "")
    rows.append(row)
A(table(["static chi prior (`--angle-model`)", "0.5 Å", "0.3 Å", "0.2 Å", "0.1 Å", "lossless", "model only: side chain RMSD (Å)"], rows))
A(EXTRA.get("vonmises_notes", ""))

# ---------------------------------------------------------------- 7.6 tuning of the new methods
A("### 7.6 Sensitivity of the two added methods\n")
rows = []
for k, lab in (("dvorak @ {b}", "Dvořák, default (qa 0.1 rad)"), ("dvorak qa 0.03 @ {b}", "Dvořák, qa 0.03 rad"), ("dvorak qa 0.3 @ {b}", "Dvořák, qa 0.3 rad"),
               ("mdz @ {b}", "MDZ, default (adp, batch 10)"), ("mdz batch 1 @ {b}", "MDZ, batch 1"), ("mdz batch 50 @ {b}", "MDZ, batch 50"),
               ("mdz mt @ {b}", "MDZ, MT forced"), ("mdz ts @ {b}", "MDZ, TS forced")):
    row = [lab]
    for b in ("0.5", "0.1"):
        r = g("T1000/" + k.format(b=b)); row.append(f"{r['ratio']:.2f}" + (f" (max {r['metrics']['max_abs_err']:.3f})" if r else "") if r else "")
    rows.append(row)
A(table(["setting", "0.5 Å", "0.1 Å"], rows))
A(EXTRA.get("tuning_notes", ""))

A("## 8. Caveats\n")
A(EXTRA.get("caveats", ""))

md = "\n".join(L)
md = md.replace(" - ", ", ").replace(" -- ", ", ")
open(f"{OUT}/MDPack_report.md", "w").write(md)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bx = [0.5, 0.3, 0.2, 0.1]
for n, lab in BOUNDED[:7] + [("lzma", "quantise + lzma"), ("h5md sz3", "H5MD + SZ3"), ("xtc", "XTC")]:
    ys = [(b, g(f"T1000/{n} @ {b}")) for b in bx]; ys = [(b, r["ratio"]) for b, r in ys if r]
    if ys: ax.plot([y[0] for y in ys], [y[1] for y in ys], "o-" if n.startswith(("MDPack", "new-coder")) else "s--", label=lab.replace("`", ""), ms=3, lw=1.2)
ax.set_xlabel("guaranteed max abs error (Å)"); ax.set_ylabel("compression ratio"); ax.invert_xaxis(); ax.grid(alpha=0.3)
ax.legend(fontsize=6.5, ncol=2); ax.set_title(SYSLAB)
fig.tight_layout(); fig.savefig(f"{OUT}/figs/fig_ratio_bound.png", dpi=160); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, b in zip(axs, ("0.1", "lossless")):
    for ci, (n, lab) in enumerate(FM):
        v = FIG.get((n, b))
        if v: ax.plot([p[0] for p in v], [p[1] for p in v], "o-", ms=3, label=lab, color=f"C{ci}")
    ax.set_xscale("log"); ax.set_xlabel("frames T"); ax.set_ylabel("compression ratio"); ax.grid(alpha=0.3); ax.set_title(f"{b} Å" if b != "lossless" else "lossless (0.01 Å grid)")
axs[0].legend(fontsize=6.5); axs[1].legend(fontsize=6.5); fig.tight_layout(); fig.savefig(f"{OUT}/figs/fig_frames.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- html / pdf
import markdown
from weasyprint import HTML
css = """@page { size: A4; margin: 16mm 14mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9pt; line-height: 1.35; color: #111; }
h1 { font-size: 17pt; margin-bottom: 4pt; } h2 { font-size: 13pt; border-bottom: 1px solid #ccc; margin-top: 16pt; }
h3 { font-size: 11pt; margin-top: 12pt; } table { border-collapse: collapse; margin: 6pt 0; font-size: 7.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.5pt 3pt; vertical-align: top; } th { background: #eef; }
code, pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.8pt; } pre { background: #f6f6f6; padding: 5pt; white-space: pre-wrap; }
img { max-width: 100%; }"""
html = "<html><head><meta charset='utf-8'><style>" + css + "</style></head><body>" + markdown.markdown(md, extensions=["tables", "fenced_code"]) + "</body></html>"
open(f"{OUT}/MDPack_report.html", "w").write(html)
doc = HTML(string=html, base_url=OUT).render(); doc.write_pdf(f"{OUT}/MDPack_report.pdf")
print(f"{OUT}/MDPack_report.pdf: {len(doc.pages)} pages")
