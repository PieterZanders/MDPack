#!/usr/bin/env python
"""Writes the bond sphere section of the benchmark report (markdown) from the benchmark JSONs (whole .mdz bytes):
  sphere_bench_mdcath.json / sphere_bench_mddb.json   production rows (sphere off), MDCompress, hybrid packer only
  t91_auto_mdcath.json / t91_auto_mddb.json           new-coder with the sphere (auto, bonds from the original coordinates)
  t91_hybrid_full.json                                hybrid with the full predictor
    python benchmark/entropy/sphere_section.py RESULTS_DIR OUT.md"""
import json, os, sys
import numpy as np
R, OUT = sys.argv[1], sys.argv[2]
B = ["0.5", "0.3", "0.2", "0.1", "lossless"]; BL = {"0.5": "0.5 Å", "0.3": "0.3 Å", "0.2": "0.2 Å", "0.1": "0.1 Å", "lossless": "lossless"}
J = lambda n: json.load(open(os.path.join(R, n))) if os.path.exists(os.path.join(R, n)) else {}
oc, om, nc, nm, hf = J("sphere_bench_mdcath.json"), J("sphere_bench_mddb.json"), J("t91_auto_mdcath.json"), J("t91_auto_mddb.json"), J("t91_hybrid_full.json")
DOMS = sorted({k.split("/")[0] for k in oc}); PROT = ["mddb_small_1APQ", "mddb_medium_12CA", "mddb_large_1AYX"]
f2 = lambda v: f"{v:.2f}"
def best(d, s, T, bd, suf): return max(d[f"{s}/T{T}/{bd}/nc_raw_{suf}"]["ratio"], d[f"{s}/T{T}/{bd}/nc_mean_{suf}"]["ratio"])
def bestb(d, s, T, bd, suf): return min(d[f"{s}/T{T}/{bd}/nc_raw_{suf}"]["bytes"], d[f"{s}/T{T}/{bd}/nc_mean_{suf}"]["bytes"])
def mt(vals): return float(np.median(vals))
L = []; A = L.append
A("## 9. Bond sphere prediction in the new coder (coder blob version 5)\n")
A("The new coder gains a prediction from bonded neighbours: for every atom with a bonded neighbour B coded earlier in the same frame, the coordinate axis "
  "with the largest offset from B is coded last and predicted from the sphere of the mean bond length around B (mode sphere); mode sphere plus circle also "
  "predicts the middle axis from the circle given by the bond length and the mean angle at B. Bonds are chosen by the encoder from the original "
  "coordinates (the earlier atom whose distance varies least) and stored in the file with the bond lengths and angles, so no topology is needed. "
  "The encoder codes each file with the sphere off, sphere and sphere plus circle (same predictor) and keeps the smallest; the mode is carried by the "
  "coder blob version, so the rest of the file is the same in every mode and no file is larger than with the sphere off. All sizes are whole .mdz "
  "files including the container and the metadata trailer; ratios are raw float32 bytes over file bytes. Model weights are not part of any ratio.\n")
for T in (1000, 250):
    A(f"### 9.{1 if T == 1000 else 2} mdCATH, 16 test domains, T = {T:,} (mean ratio)\n")
    A("| max error | new-coder, sphere off (best of raw and mean) | new-coder with sphere | raw: off, sphere | mean: off, sphere | MDCompress best setting (-l 4 --preset archive, through the codec) | files larger than sphere off |")
    A("|---|---|---|---|---|---|---|")
    for bd in B:
        if not all(f"{s}/T{T}/{bd}/nc_raw_auto" in nc and f"{s}/T{T}/{bd}/nc_mean_auto" in nc for s in DOMS): continue
        o = np.mean([best(oc, s, T, bd, "off") for s in DOMS]); n = np.mean([best(nc, s, T, bd, "auto") for s in DOMS])
        ro = np.mean([oc[f"{s}/T{T}/{bd}/nc_raw_off"]["ratio"] for s in DOMS]); rn = np.mean([nc[f"{s}/T{T}/{bd}/nc_raw_auto"]["ratio"] for s in DOMS])
        mo = np.mean([oc[f"{s}/T{T}/{bd}/nc_mean_off"]["ratio"] for s in DOMS]); mn = np.mean([nc[f"{s}/T{T}/{bd}/nc_mean_auto"]["ratio"] for s in DOMS])
        md = np.mean([oc[f"{s}/T{T}/{bd}/mdc_l4a"]["ratio"] for s in DOMS])
        big = sum(1 for s in DOMS for p in ("raw", "mean") if (nc[f"{s}/T{T}/{bd}/nc_{p}_auto"].get("smode_sizes") or {}).get(nc[f"{s}/T{T}/{bd}/nc_{p}_auto"]["smode"] or "off", 0) > (nc[f"{s}/T{T}/{bd}/nc_{p}_auto"].get("smode_sizes") or {}).get("off", 1e18))
        A(f"| {BL[bd]} | {f2(o)} | **{f2(n)}** | {f2(ro)}, {f2(rn)} | {f2(mo)}, {f2(mn)} | {f2(md)} | {big} of 32 |")
    A("")
A("Files larger than sphere off counts coder blobs; whole files where the encoder kept the sphere off differ from the production file only by the run time fields of the metadata trailer (a few bytes).\n")
A("MDCompress column: the best single setting over the 16 domains is `-l 4 --preset archive` at every bound once the archive preset is also run at 0.3 and 0.2 Å (resolution 60,000 and 40,000 fm), written through the codec (about 1.5 KB of container and trailer per file). The summary of section 4.6 used native MDCompress files and the archive preset only at 0.5 Å, 0.1 Å and lossless, so its 0.3 and 0.2 Å values (12.55 and 11.70) are plain `-l 4`.\n")
A("### 9.3 MDDB protein systems, T = 1,000 (ratio)\n")
A("| system | atoms | max error | new-coder, sphere off | new-coder with sphere | mode (raw, mean) | MDCompress best setting |")
A("|---|---|---|---|---|---|---|")
for s in PROT:
    for bd in B:
        if f"{s}/T1000/{bd}/nc_raw_auto" not in nm: continue
        o = best(om, s, 1000, bd, "off"); n = best(nm, s, 1000, bd, "auto"); md = max(om[f"{s}/T1000/{bd}/mdc_l4a"]["ratio"], om[f"{s}/T1000/{bd}/mdc_l2"]["ratio"])
        modes = ", ".join(nm[f"{s}/T1000/{bd}/nc_{p}_auto"]["smode"] or "off" for p in ("raw", "mean"))
        A(f"| {s.replace('mddb_', '')} | {om[f'{s}/T1000/{bd}/nc_raw_off']['N']} | {BL[bd]} | {f2(o)} | **{f2(n)}** | {modes} | {f2(md)} |")
A("\nMDCompress here is the better of `-l 2` and `-l 4 --preset archive` per system. The DNA, RNA, protein and DNA complex and membrane systems are not part of this section.\n")
A("### 9.4 Hybrid against new-coder with sphere (mdCATH, 16 domains, T = 1,000, mean ratio)\n")
A("The hybrid streams are still coded with the sphere off. 'with sphere' is the measured size if both hybrid coder streams used the sphere (each stream "
  "coded again with auto, bonds from the reconstructed positions); those files were not written.\n")
A("| max error | hybrid, packer only | packer only, with sphere | hybrid, full predictor | full predictor, with sphere | new-coder with sphere | MDCompress best setting |")
A("|---|---|---|---|---|---|---|")
for bd in B:
    hp = [oc.get(f"{s}/T1000/{bd}/hybrid") for s in DOMS]; hfull = [hf.get(f"{s}/T1000/{bd}/hybrid_full") for s in DOMS]
    if not all(h and "ratio" in h for h in hp): continue
    a = np.mean([h["ratio"] for h in hp]); b = np.mean([h["raw_bytes"] / h["hybrid_auto_bytes"] for h in hp])
    okf = all(h and "ratio" in h for h in hfull)
    kf = sum(1 for h in hfull if h and "ratio" in h); pend = f"running, {kf} of 16 domains done" if kf else "running"
    c = f2(np.mean([h["ratio"] for h in hfull])) if okf else pend; d = f2(np.mean([h["raw_bytes"] / h["hybrid_auto_bytes"] for h in hfull])) if okf else pend
    n = f2(np.mean([best(nc, s, 1000, bd, "auto") for s in DOMS])) if all(f"{s}/T1000/{bd}/nc_raw_auto" in nc for s in DOMS) else "not run"
    md = np.mean([oc[f"{s}/T1000/{bd}/mdc_l4a"]["ratio"] for s in DOMS])
    A(f"| {BL[bd]} | {f2(a)} | {f2(b)} | {c} | {d} | {n} | {f2(md)} |")
A("\n### 9.5 Timings (median per file, seconds, 2 core machine, encode includes the codec's verification decode)\n")
A("| T | max error | encode, sphere off | encode, with sphere | decode, sphere off | decode, with sphere |")
A("|---|---|---|---|---|---|")
for T in (1000, 250):
    for bd in ("0.2", "0.1", "lossless"):
        ks = [(s, p) for s in DOMS for p in ("raw", "mean") if f"{s}/T{T}/{bd}/nc_{p}_auto" in nc]
        if len(ks) < 32: continue
        eo = mt([oc[f"{s}/T{T}/{bd}/nc_{p}_off"]["encode_s"] for s, p in ks]); do = mt([oc[f"{s}/T{T}/{bd}/nc_{p}_off"]["decode_s"] for s, p in ks])
        en = mt([nc[f"{s}/T{T}/{bd}/nc_{p}_auto"]["encode_s"] - (nc[f"{s}/T{T}/{bd}/nc_{p}_auto"].get("diag") or {}).get("t_sub", 0.0) for s, p in ks])
        dn = mt([nc[f"{s}/T{T}/{bd}/nc_{p}_auto"]["decode_s"] for s, p in ks])
        A(f"| {T:,} | {BL[bd]} | {f2(eo)} | {f2(en)} | {f2(do)} | {f2(dn)} |")
A("\nThe encode with sphere is measured in an analysis run that also coded a 128 frame subsample of every mode; that subsample time is subtracted. Decoding needs no extra pass; its extra cost is the sphere and circle arithmetic.\n")
A("### 9.6 Portability\n")
A("A bundle of 24 coded blobs (8 in the experimental format, 16 in coder blob version 5, lossless and 0.1 Å, sphere and sphere plus circle, zero and nonzero base) "
  "was decoded in a fresh process from the files alone with gcc and clang at -O0, -O2, -O3 and -O3 -march=native and with aarch64 gcc at -O0, -O2 and -O3 under "
  "qemu: every decode matched the stored SHA-256 of the integers. The coder must be compiled with -ffp-contract=off -fno-fast-math; an aarch64 build with "
  "FMA contraction allowed decoded 5 of the 8 experimental blobs wrongly. A second physical machine has not run the bundle yet.\n")
open(OUT, "w").write("\n".join(L) + "\n"); print(OUT, len(L), "lines")
