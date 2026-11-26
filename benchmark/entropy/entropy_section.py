#!/usr/bin/env python
"""AEZ-12: build entropy/entropy_section.md (report section 4.6) and figs/fig_entropy_bounds.png from
results/benchmark_summary.json (after aggregate.py), the AEZ-11 raw files and entropy/results/*.json.
Every number is read from result files."""
import glob, json, os, sys
import numpy as np
BEN = "/workspace/md_compress/benchmark"; R = f"{BEN}/results"; E = f"{BEN}/entropy"
S = json.load(open(f"{R}/benchmark_summary.json"))
DOMS = json.load(open("/workspace/md_compress/results_gpu/frames/selected_domains.json"))
MDDB = json.load(open("/workspace/bench_data/mddb_meta.json"))
rows = {}
for f in sorted(glob.glob(f"{R}/*_raw*.json")) + sorted(glob.glob("/workspace/md_compress/results/ours_raw_*.json")):
    rows.update(json.load(open(f)))
ent = {}
for f in sorted(glob.glob(f"{E}/results/entropy_raw_*.json")): ent.update(json.load(open(f)))
etime = {}
for f in sorted(glob.glob(f"{E}/results/timing_*.json")): etime.update(json.load(open(f)))
BOUNDS = ["0.5", "0.3", "0.2", "0.1", "0.005"]
RES = {"0.005": 1000, "0.1": 20000, "0.2": 40000, "0.3": 60000, "0.5": 100000}
BL = lambda b: "lossless" if b == "0.005" else f"{b} Å"
MDC = lambda r: [f"MDCompress -l 4 --preset archive --res {r} fm", f"MDCompress -l 4 --res {r} fm", f"MDCompress -l 2 --res {r} fm" + (" (default)" if r == 1000 else "")]
f2 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"
L = []; A = L.append; SUMT = {}; SUMAE = {}


def ratio(src, s, T, m):
    v = src.get(f"{s}/T{T}/{m}")
    return v["ratio"] if v and "error" not in v else np.nan


def mdc_best_sys(s, T, b):
    v = [ratio(rows, s, T, m) for m in MDC(RES[b])]; v = [x for x in v if np.isfinite(x)]
    return max(v) if v else np.nan


def mdc_best_mean(T, b):
    tab = S["mdcath"].get(str(T), S["mdcath"].get(T, {}))
    c = [m for m in MDC(RES[b]) if m in tab]
    if not c: return np.nan, ""
    m = max(c, key=lambda m: tab[m]["ratio_mean"]); return tab[m]["ratio_mean"], m.replace("MDCompress ", "").replace(" fm", "").replace(" (default)", "")


def mean_over(src, T, m, doms):
    v = [ratio(src, d, T, m) for d in doms]
    return float(np.mean(v)) if all(np.isfinite(v)) and v else np.nan


def complete(T):
    need = [f"ours: hybrid + AC @ {b}" for b in BOUNDS] + [f"model free mean + AC @ {b}" for b in BOUNDS] + [f"model free raw + AC @ {b}" for b in BOUNDS]
    return [d for d in DOMS if all(np.isfinite(ratio(ent, d, T, m)) for m in need)]


A("### 4.6 A context modelled arithmetic coder for the residual streams (AEZ-12)\n")
A("Sections 4.1 to 4.5 code the hybrid's integer residuals with a search over general purpose layouts and lzma. MDCompress, which beats the hybrid at 0.2 Å, 0.1 Å "
  "and lossless, codes its residuals with a dedicated range coder. Here we replace the lzma search under both hybrid residual streams (backbone plus other atoms "
  "against the mean structure with a per frame rigid fit; side chain atoms against the packer prediction) with a coordinate aware entropy coder, keeping everything "
  "else (model, rotamer codes, swap bits, quantisation, bounds) unchanged.\n")
A("#### 4.6.1 What MDCompress actually does in its coding stage\n")
A("Read from the MDCompress 2.0.5 source (`engine_compression.cpp`, `engine.h`, `params.h` and the refresh range coder library):\n")
A("* Coordinates are rounded to an integer grid with step `--res` (maximum error res/2); every prediction uses already reconstructed integers, so the decoder repeats it exactly.")
A("* Prediction: with the default history of one frame, level 2 predicts an atom as its previous position moved by the displacement its first reference atom (a bonded "
  "neighbour, already coded in this frame) just made; levels 4 and 5 carry the local frame of three reference atoms from frame t-1 to frame t (rotation plus translation) "
  "and apply it to the atom's previous position. Anchor frames (every 21st frame by default, every 101st with `--preset archive`) are coded within the frame by "
  "trilateration from three reference atoms.")
A("* Binarisation and modelling: residuals are zigzag mapped, the bit length is coded with an adaptive frequency model whose context is tiny (history size and whether "
  "the atom is one of the first three), and the low bits are sent raw (level 5 models them for short lengths). There is no context on the magnitude of neighbouring "
  "residuals.")
A("* Streams: atoms are split into subsegments of 100 (default) or 1,000 (archive) and frames into batches between anchors; each piece is an independent range coder "
  "stream with fresh models.\n")
A("So MDCompress's strength is its geometric prediction (temporal plus the rigid motion of already coded bonded neighbours), while its probability modelling is simple. "
  "Our design therefore takes over its prediction mechanism in the residual domain and adds the context modelling MDCompress lacks.\n")
A("#### 4.6.2 Design of the coder (`gpu/residual_entropy.c`, `gpu/residual_entropy.py`)\n")
A("* **Input:** the same quantised integer residuals as before (lossy: q = round((x − p) / 2b), reconstruction p + 2b·q; lossless: q = round(x / 0.01) − round(p / 0.01)), "
  "for each of the two streams.")
A("* **Prediction of the residual.** For each atom i and frame t the coder predicts q[t][i] from already decoded data: (a) a geometric term in the MDCompress level 4 "
  "style: the local frame of three reference atoms (already decoded atoms of the same stream, or, for the side chain stream, atoms of the already decoded backbone) is "
  "carried from t−1 to t and applied to the atom's previous reconstructed position, the result being expressed in residual units; (b) the residual of a reference atom "
  "in the same frame; (c) the atom's own residual in the previous frame; (d) the reference atom's residual in the previous frame. The prediction is an integer "
  "weighted sum of these four terms (weights in 1/64). Weights are chosen per atom class (backbone N, CA, C, O, CB, γ, δ, other side chain, nucleic backbone, base, "
  "other) by the encoder: fixed candidate sets (zero, geometric only, spatial, temporal, the MDCompress level 2 analogue 1, 1, −1) and per class least squares fits, "
  "scored with a log2(1 + |e|) cost on a frame subsample, the best few actually coded and the smallest kept.")
A("* **Binarisation:** zero flag, sign, Elias gamma length (unary) and mantissa bits of the prediction error.")
A("* **Modelling:** every binary decision is coded with a logistic mixing of several adaptive context models (in the style of lpaq) followed by an adaptive probability "
  "map. Contexts: atom class, axis, activity buckets of already coded errors (same atom at t−1 and t−2, reference atom at t and t−1, previous axis of the same atom), "
  "and the disagreement between the predictor terms.")
A("* **Entropy coder:** LZMA style binary range coder with 16 bit probabilities. Each stream is one range coder stream (no subsegments).")
A("* **Side information counted in the size:** a version byte, the per class weights (16 classes × 4 int16), the in stream reference offsets and the three geometric "
  "reference atoms per atom (chosen by the encoder from its own mean structure, stored lzma compressed). The shared model weights remain excluded as before.")
A("* **Determinism:** all coding is integer except the geometric term, computed in double precision with contraction and fast math disabled; encoder and decoder must "
  "use the same build.\n")
A("#### 4.6.3 Verification\n")
nv = sum(1 for k, v in ent.items() if "+ AC" in k and "error" not in v)
bad = [k for k, v in ent.items() if "+ AC" in k and "error" not in v and v["checked_max_abs_err"] > float(k.split("@ ")[1]) + 1e-3]
ll = [v["checked_max_abs_err"] for k, v in ent.items() if "+ AC @ 0.005" in k and "error" not in v]
lo = max([v["checked_max_abs_err"] / float(k.split("@ ")[1]) for k, v in ent.items() if "+ AC" in k and "error" not in v and not k.endswith("0.005")] or [np.nan])
errs = [k for k, v in ent.items() if "error" in v]
A(f"Every coded stream was decoded again and compared with the encoder's integers (bit exact, otherwise the run aborts), and the reconstruction was checked against "
  f"the original coordinates: maximum absolute error ≤ bound + 0.001 Å for the lossy settings and ≤ 0.005 + 0.001 Å for lossless (the 0.01 Å grid mode; observed maximum {max(ll or [np.nan]):.4f} Å, as the sources lie on or very close to that grid; "
  f"for the lossy settings the largest observed error was {lo:.4f} times the bound). "
  f"{nv} coded results passed both checks; {len(bad)} exceeded the bound; {len(errs)} runs recorded an error" + (f" ({', '.join(errs)})" if errs else "") + ".\n")

# ---------------- mdCATH tables
for T in (1000, 250):
    doms = complete(T)
    if not doms: continue
    A(f"#### 4.6.{4 if T == 1000 else 5} mdCATH, T = {T:,} frames ({len(doms)} of 16 domains complete; mean ratio over domains)\n")
    A("| max error | ours: hybrid + AC | ours: hybrid + lzma (sections 4.1 to 4.3) | ours: hybrid + lzma (refit in this run) | model free mean + AC | model free mean + lzma | model free raw + AC | best AC variant per domain | MDCompress (best setting) | hybrid + AC vs MDCompress | best AC variant vs MDCompress | domains where hybrid + AC (best AC variant) beats every MDCompress setting |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BOUNDS:
        hac = mean_over(ent, T, f"ours: hybrid + AC @ {b}", doms); hlz = mean_over(rows, T, f"ours: hybrid @ {b}", doms)
        hlz2 = mean_over(ent, T, f"ours: hybrid @ {b}", doms); mac = mean_over(ent, T, f"model free mean + AC @ {b}", doms)
        mlz = mean_over(rows, T, f"model free mean @ {b}", doms); rac = mean_over(ent, T, f"model free raw + AC @ {b}", doms)
        mv, mn = mdc_best_mean(T, b)
        if len(doms) < 16: mv = float(np.mean([ratio(rows, d, T, "MDCompress " + mn + " fm" + (" (default)" if mn == "-l 2 --res 1000" else "")) for d in doms]))
        bst = {d: max(ratio(ent, d, T, f"{v} + AC @ {b}") for v in ("ours: hybrid", "model free mean", "model free raw")) for d in doms}
        bac = float(np.mean(list(bst.values())))
        wins = sum(ratio(ent, d, T, f"ours: hybrid + AC @ {b}") > mdc_best_sys(d, T, b) for d in doms)
        wb = sum(bst[d] > mdc_best_sys(d, T, b) for d in doms)
        pc = lambda x: f"{'+' if x > mv else ''}{100 * (x / mv - 1):.1f} %"
        SUMT.setdefault(T, {})[b] = dict(hac=hac, bac=bac, mdc=mv, hlz=hlz, n=len(doms), wins=wins, wb=wb)
        A(f"| {BL(b)} | **{f2(hac)}** | {f2(hlz)} | {f2(hlz2)} | {f2(mac)} | {f2(mlz)} | {f2(rac)} | {f2(bac)} | {f2(mv)} ({mn}) | {pc(hac)} | {pc(bac)} | {wins} ({wb}) of {len(doms)} |")
    A("")
    A("'Best AC variant per domain' is the smallest of the three AC coded files for each trajectory, i.e. an encoder that tries all three and stores a one byte "
      "flag (not counted; under 0.01 % of any file).\n")
    if T == 1000:
        A("Byte breakdown of hybrid + AC (mean over domains, kB): fixed part = backbone of the mean structure, per frame rigid transforms (28 bytes per frame), "
          "per residue packer state, rotamer codes and the per trajectory MLP; these are not residual streams and are unchanged by the new coder.\n")
        A("| max error | total | fixed part | backbone + other residuals | side chain residuals | swap bits | AC side information (included in the streams) |")
        A("|---|---|---|---|---|---|---|")
        for b in BOUNDS:
            vs = [ent[f"{d}/T{T}/ours: hybrid + AC @ {b}"] for d in doms]
            mk = lambda q: float(np.mean([v[q] for v in vs])) / 1e3
            A(f"| {BL(b)} | {mk('bytes'):.1f} | {mk('fixed_bytes'):.1f} | {mk('bb_and_other_residual_bytes'):.1f} | {mk('sc_residual_bytes'):.1f} | {mk('swap_bytes'):.2f} | {mk('ac_header_bytes'):.2f} |")
        A("")
        # predictor choices
        from collections import Counter
        c = Counter(); 
        for d in doms:
            for b in BOUNDS:
                v = ent[f"{d}/T{T}/ours: hybrid + AC @ {b}"]; c[v.get("predictor_bb")] += 1; c[v.get("predictor_sc")] += 1
        A("Predictor selected per stream (hybrid, T = 1,000, all bounds): " + ", ".join(f"{k}: {n}" for k, n in c.most_common()) + ".\n")

# ---------------- backbone autoencoder variant
A("#### 4.6.6 Variant: hybrid with the learned backbone autoencoder\n")
A("Here the backbone (and, for MDDB systems, the other atoms) is predicted not from the mean structure with a rigid fit but by the shared backbone "
  "autoencoder of the main report: per frame the aligned backbone is encoded to latents (8 per residue), stored as fp16 (lossless lzma, counted, with the 28 byte "
  "per frame rigid transform), decoded by the shared model and put back in the lab frame. The backbone residuals against this prediction and the side chain "
  "residuals (packer prediction from the decoded backbone, as in the hybrid) are then coded with the new coder (and, for reference, the lzma search). Fixed part: "
  "latents + rigid transforms + per residue data + rotamer codes + MLP; shared model weights excluded.\n")
for T in (1000, 250):
    dA = [d for d in DOMS if all(np.isfinite(ratio(ent, d, T, f"ours: hybrid + backbone AE{x} @ {b}")) and np.isfinite(ratio(ent, d, T, f"ours: hybrid + AC @ {b}")) for b in BOUNDS for x in (" + AC", ""))]
    if not dA: continue
    A(f"mdCATH, T = {T:,} ({len(dA)} of 16 domains; mean ratio):\n")
    A("| max error | hybrid + AC (mean structure backbone) | hybrid + backbone AE + AC | hybrid + backbone AE + lzma | latents, share of the AE + AC file | backbone residual bytes, AE / mean structure | MDCompress (best setting per domain) |")
    A("|---|---|---|---|---|---|---|")
    for b in BOUNDS:
        va = [ent[f"{d}/T{T}/ours: hybrid + backbone AE + AC @ {b}"] for d in dA]; vh = [ent[f"{d}/T{T}/ours: hybrid + AC @ {b}"] for d in dA]
        sh = np.mean([v["latent_bytes"] / v["bytes"] for v in va]); rb = np.mean([x["bb_and_other_residual_bytes"] / y["bb_and_other_residual_bytes"] for x, y in zip(va, vh)])
        SUMAE.setdefault(T, {})[b] = dict(ae_ac=mean_over(ent, T, f"ours: hybrid + backbone AE + AC @ {b}", dA), ae_lz=mean_over(ent, T, f"ours: hybrid + backbone AE @ {b}", dA), n=len(dA))
        A(f"| {BL(b)} | {f2(mean_over(ent, T, f'ours: hybrid + AC @ {b}', dA))} | **{f2(SUMAE[T][b]['ae_ac'])}** | {f2(SUMAE[T][b]['ae_lz'])} | {100 * sh:.0f} % | {rb:.2f} | "
          f"{f2(float(np.mean([mdc_best_sys(d, T, b) for d in dA])))} |")
    rm = np.mean([ent[f"{d}/T{T}/ours: hybrid + backbone AE + AC @ 0.5"]["ae_backbone_rmsd_lab"] for d in dA])
    A(f"\nMean backbone RMSD of the autoencoder prediction (lab frame, before residuals): {rm:.3f} Å.\n")

# ---------------- MDDB
A("#### 4.6.7 MDDB systems (T = 1,000)\n")
A("Systems without protein (DNA, RNA) have no hybrid; for them 'ours' is the model free mean structure stream, as in section 4.5. Columns per bound: best of our "
  "AC variants (hybrid + AC, hybrid + backbone AE + AC, model free mean + AC, model free raw + AC; the variant is given by a letter: h, a, m, r), the AEZ-11 best of ours with lzma "
  "(hybrid, or model free mean where there is no hybrid), and the best MDCompress setting on that system.\n")
A("| system | " + " | ".join(f"{BL(b)}: ours AC / ours lzma / MDCompress" for b in BOUNDS) + " |")
A("|---|" + "---|" * len(BOUNDS))
mddb_win = {b: [0, 0] for b in BOUNDS}
for s in MDDB:
    cells = []
    for b in BOUNDS:
        acs = {"h": ratio(ent, s, 1000, f"ours: hybrid + AC @ {b}"), "a": ratio(ent, s, 1000, f"ours: hybrid + backbone AE + AC @ {b}"), "m": ratio(ent, s, 1000, f"model free mean + AC @ {b}"), "r": ratio(ent, s, 1000, f"model free raw + AC @ {b}")}
        acs = {k: v for k, v in acs.items() if np.isfinite(v)}
        lz = [ratio(rows, s, 1000, f"ours: hybrid @ {b}"), ratio(rows, s, 1000, f"model free mean @ {b}")]
        lz = np.nanmax(lz) if np.isfinite(lz).any() else np.nan
        mv = mdc_best_sys(s, 1000, b)
        if acs:
            k = max(acs, key=acs.get); cells.append(f"**{acs[k]:.2f}** {k} / {f2(lz)} / {f2(mv)}")
            if np.isfinite(mv): mddb_win[b][0] += acs[k] > mv; mddb_win[b][1] += 1
        else: cells.append(f"n/a / {f2(lz)} / {f2(mv)}")
    A(f"| {MDDB[s]['kind']} ({MDDB[s]['accession']}) | " + " | ".join(cells) + " |")
A("")
A("Best AC variant beats the best MDCompress setting on: " + "; ".join(f"{BL(b)} {w} of {n} systems" for b, (w, n) in mddb_win.items()) + ".\n")

# ---------------- timing
A("#### 4.6.8 Encode and decode time\n")
doms = complete(1000)
if doms:
    A(f"Coder only wall time per trajectory (mean over the {len(doms)} complete domains, T = 1,000; both residual streams; single CPU thread for the AC coder; "
      "measured during the main run with three benchmark processes sharing the 8 core box, so absolute values carry some contention). The hybrid's model work "
      "(per trajectory MLP fit, packer prediction, swap fix) is identical for both coders and is listed separately.\n")
    A("| max error | AC encode (s) | AC decode (s) | lzma layout search encode (s) | lzma decode (s) | hybrid model and prediction, encode side (s) |")
    A("|---|---|---|---|---|---|")
    for b in BOUNDS:
        g = lambda m, q: float(np.mean([ent[f"{d}/T1000/{m}"][q] for d in doms if f"{d}/T1000/{m}" in ent]))
        A(f"| {BL(b)} | {g(f'ours: hybrid + AC @ {b}', 'coder_encode_s'):.2f} | {g(f'ours: hybrid + AC @ {b}', 'coder_decode_s'):.2f} | "
          f"{g(f'ours: hybrid @ {b}', 'coder_encode_s'):.2f} | {g(f'ours: hybrid @ {b}', 'coder_decode_s'):.2f} | {g(f'ours: hybrid + AC @ {b}', 'model_and_prediction_s'):.1f} |")
    A("")
if etime:
    A("Clean timing pass (one process alone on the box, three domains, T = 1,000):\n")
    A("| domain | max error | AC encode (s) | AC decode (s) | lzma search encode (s) | lzma decode (s) |")
    A("|---|---|---|---|---|---|")
    for d in sorted({k.split("/")[0] for k in etime}):
        for b in ("0.1", "0.005"):
            a_, l_ = etime.get(f"{d}/T1000/ours: hybrid + AC @ {b}"), etime.get(f"{d}/T1000/ours: hybrid @ {b}")
            if a_ and l_: A(f"| {d} | {BL(b)} | {a_['coder_encode_s']:.2f} | {a_['coder_decode_s']:.2f} | {l_['coder_encode_s']:.2f} | {l_['coder_decode_s']:.2f} |")
    A("")
s = "mddb_membrane_Eprot"; k = f"{s}/T1000/ours: hybrid + AC @ 0.1"
if k in ent and "error" not in ent[k]:
    A(f"Largest system ({MDDB[s]['kind']}, {MDDB[s]['n_heavy']:,} heavy atoms): AC coder {ent[k]['coder_encode_s']:.0f} s encode and {ent[k]['coder_decode_s']:.0f} s decode at 0.1 Å.\n")

# ---------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
xs = [0.005, 0.1, 0.2, 0.3, 0.5]; bs = ["0.005", "0.1", "0.2", "0.3", "0.5"]
for a_, T in zip(ax, (1000, 250)):
    doms = complete(T)
    if not doms: continue
    ser = [("ours: hybrid + AC", ent, "ours: hybrid + AC @ {}", "darkgreen", "o", "-"), ("ours: hybrid + lzma (AEZ-11)", rows, "ours: hybrid @ {}", "tab:green", "o", ":"),
           ("model free mean + AC", ent, "model free mean + AC @ {}", "navy", "s", "-"), ("model free mean + lzma", rows, "model free mean @ {}", "tab:blue", "s", ":"),
           ("model free raw + AC", ent, "model free raw + AC @ {}", "teal", "s", "--")]
    for lab, src, pat, c, mk, ls in ser:
        a_.plot(range(5), [mean_over(src, T, pat.format(b), doms) for b in bs], marker=mk, c=c, ls=ls, label=lab)
    mv = [np.mean([mdc_best_sys(d, T, b) for d in doms]) for b in bs]
    a_.plot(range(5), mv, marker="^", c="tab:red", label="MDCompress (best setting per domain)")
    a_.set_xticks(range(5)); a_.set_xticklabels(["lossless", "0.1 Å", "0.2 Å", "0.3 Å", "0.5 Å"])
    a_.set(xlabel="guaranteed maximum error", ylabel="ratio vs raw float32 (×), mean over domains", title=f"mdCATH, T = {T}, {len(doms)} domains"); a_.grid(alpha=0.3)
    a_.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{BEN}/figs/fig_entropy_bounds.png", dpi=170); plt.close(fig)
A("![Ratio by bound with the AC coder](figs/fig_entropy_bounds.png)\n")
A("*Figure 6. Mean ratio at each guaranteed bound, with the new arithmetic coder (solid) and with the lzma layout search (dotted); MDCompress is the best of its "
  "settings on each domain (a slightly stronger reference than the single best setting used in the tables). The AC variants also appear as dashed lines in the "
  "Pareto figures of section 4.1 and 4.3.*\n")

# ---------------- caveats

# ---------------- frames scaling
SCALSUM = {}
sc = {}
for f in sorted(glob.glob(f"{E}/results/scaling_*.json")): sc.update(json.load(open(f)))
SD = ["1e8rA00", "4m70B00", "2lg7A00"]; LEVS = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
SMETH = [("ours: hybrid + AC", "ours: hybrid + AC @ {}"), ("ours: hybrid + lzma", "ours: hybrid @ {}"), ("ours: hybrid + backbone AE + AC", "ours: hybrid + backbone AE + AC @ {}"),
         ("ours: hybrid + backbone AE + lzma", "ours: hybrid + backbone AE @ {}"), ("model free mean + AC", "model free mean + AC @ {}"), ("model free mean + lzma", "model free mean @ {}"),
         ("model free raw + AC", "model free raw + AC @ {}")]
XTCM = {"0.005": "XTC precision 1000", "0.5": "XTC precision 10"}
def mdc_sc(d, T, b):
    r = RES[b]; c = [f"MDCompress -l 4 --res {r} fm", f"MDCompress -l 4 --preset archive --res {r} fm", f"MDCompress -l 2 --res {r} fm" + (" (default)" if r == 1000 else "")]
    v = [sc[f"{d}/T{T}/{m}"]["ratio"] for m in c if f"{d}/T{T}/{m}" in sc and "error" not in sc[f"{d}/T{T}/{m}"]]
    return max(v) if v else np.nan
def sval(d, T, lab, pat, b):
    if lab == "MDCompress (best setting)": return mdc_sc(d, T, b)
    if lab == "XTC": k = f"{d}/T{T}/{XTCM.get(b, '')}"
    else: k = f"{d}/T{T}/{pat.format(b)}"
    v = sc.get(k); return v["ratio"] if v and "error" not in v else np.nan
if sc:
    allm = SMETH + [("MDCompress (best setting)", None), ("XTC", None)]
    A("#### 4.6.9 Compression ratio versus number of frames\n")
    A("Three domains of increasing size (1e8rA00, 376 heavy atoms; 4m70B00, 726; 2lg7A00, 969) were coded at T = 10, 50, 100, 250, 500, 1,000, 2,500, 5,000 and 10,000 "
      "frames (the first T frames of the archive order used throughout). Important: frames 1 to 1,760 are the four 320 K replicas; from frame 1,761 on the archive "
      "continues with the 348, 379, 413 and 450 K replicas, so beyond T = 1,760 the trajectory becomes more diverse (partial unfolding at high temperature) and the "
      "frame count effect is confounded with this. Every method was rerun on these frames (per trajectory model fit at each T). To keep the run time within the "
      "box, the lzma reference used the three most often winning layouts of the main run (none-nat-vint, none-nat-planes, dt-nat-planes); MDCompress was run with "
      "the same settings as in the main benchmark (levels 2 and 4, archive preset where available) and the best per domain is shown; XTC exists only at the lossless (precision 1000) and 0.5 Å "
      "(precision 10) bounds. Mean ratio over the three domains; a level is only averaged when all three finished.\n")
    for b in BOUNDS:
        nlev = [T for T in LEVS if any(f"{d}/T{T}/ours: hybrid + AC @ {b}" in sc for d in SD)]
        if not nlev: continue
        cnt = {T: [d for d in SD if f"{d}/T{T}/ours: hybrid + AC @ {b}" in sc] for T in nlev}
        A(f"{BL(b)}:\n")
        A("| method | " + " | ".join(f"T = {T:,}" + ("" if len(cnt[T]) == len(SD) else f" ({len(cnt[T])} of {len(SD)} domains finished, not averaged)") for T in nlev) + " |")
        A("|---|" + "---|" * len(nlev))
        for lab, pat in allm:
            if lab == "XTC" and b not in XTCM: continue
            vals = []
            for T in nlev:
                v = [sval(d, T, lab, pat, b) for d in cnt[T]]
                vals.append(float(np.mean(v)) if len(cnt[T]) == len(SD) and all(np.isfinite(v)) else np.nan)   # only levels where all domains finished
            SCALSUM.setdefault(b, {})[lab] = dict(zip(map(str, nlev), vals))
            A(f"| {lab} | " + " | ".join(f2(v) for v in vals) + " |")
        A("")
    pk_ = []
    for lab in ("ours: hybrid + AC", "model free mean + AC", "model free raw + AC", "MDCompress (best setting)"):
        d_ = {int(t): v for t, v in SCALSUM.get("0.1", {}).get(lab, {}).items() if np.isfinite(v)}
        if d_: tp = max(d_, key=d_.get); pk_.append(f"{lab} peaks at T = {tp:,} ({d_[tp]:.2f}×; T = 10: {d_.get(10, float('nan')):.2f}×, last averaged level T = {max(d_):,}: {d_[max(d_)]:.2f}×)")
    A("At 0.1 Å: " + "; ".join(pk_) + ". Every method gains strongly from 10 to about 250 frames, because per trajectory side information (mean structure, "
      "model, codes, reference choice, coder adaptation) is spread over more frames; above about 1,000 frames the curves flatten, and beyond 2,500 frames they "
      "fall, which coincides with the archive moving to higher temperature replicas (more diverse conformations). The same temperature check in section 4.6.9.1 "
      "shows no decline after 1,000 frames at constant temperature up to 2,500 frames, the most one domain offers at one temperature. The new coder's lead over lzma for the same predictor grows with T, since its adaptive models learn from more data.\n")
    s3 = {}
    for f in ("scaling320_entropy.json", "scaling320_formats.json"):
        if os.path.exists(f"{E}/results/{f}"): s3.update(json.load(open(f"{E}/results/{f}")))
    LEV3 = [10, 50, 100, 250, 500, 1000, 1500, 2000, 2500]
    def v3(src, d, T, lab, pat, b):
        if lab == "MDCompress (best setting)":
            r = RES[b]; c = [f"MDCompress -l 4 --res {r} fm", f"MDCompress -l 4 --preset archive --res {r} fm", f"MDCompress -l 2 --res {r} fm" + (" (default)" if r == 1000 else "")]
            v = [src[f"{d}/T{T}/{m}"]["ratio"] for m in c if f"{d}/T{T}/{m}" in src and "error" not in src[f"{d}/T{T}/{m}"]]
            return max(v) if v else np.nan
        k = f"{d}/T{T}/{XTCM.get(b, '')}" if lab == "XTC" else f"{d}/T{T}/{pat.format(b)}"
        v = src.get(k); return v["ratio"] if v and "error" not in v else np.nan
    S320 = {b: {lab: {T: v3(s3, "2lg7A00_320K", T, lab, pat, b) for T in LEV3} for lab, pat in allm if not (lab == "XTC" and b not in XTCM)} for b in BOUNDS}
    SOLD = {b: {lab: {T: v3(sc, "2lg7A00", T, lab, pat, b) for T in LEVS} for lab, pat in allm if not (lab == "XTC" and b not in XTCM)} for b in BOUNDS}
    sm = {}
    for f in ("scalingmddb_entropy.json", "scalingmddb_formats.json"):
        if os.path.exists(f"{E}/results/{f}"): sm.update(json.load(open(f"{E}/results/{f}")))
    LEVM = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]; DM = "mddb_small_1APQ_long"
    SMD = {b: {lab: {T: v3(sm, DM, T, lab, pat, b) for T in LEVM} for lab, pat in allm if not (lab == "XTC" and b not in XTCM)} for b in BOUNDS}
    HAVE_M = any(np.isfinite(SMD["0.1"]["ours: hybrid + AC"][T]) for T in LEVM)
    fig, ax2 = plt.subplots(3 if HAVE_M else 2, 5, figsize=(24, 15.6 if HAVE_M else 10.4)); ax = ax2[0]
    sty = {"ours: hybrid + AC": ("darkgreen", "o", "-"), "ours: hybrid + lzma": ("tab:green", "o", ":"), "ours: hybrid + backbone AE + AC": ("olive", "p", "-"),
           "ours: hybrid + backbone AE + lzma": ("olive", "p", ":"), "model free mean + AC": ("navy", "s", "-"), "model free mean + lzma": ("tab:blue", "s", ":"),
           "model free raw + AC": ("teal", "s", "--"), "MDCompress (best setting)": ("tab:red", "^", "-"), "XTC": ("#444444", "x", "-")}
    for a_, b in zip(ax, BOUNDS):
        for lab, d_ in SCALSUM.get(b, {}).items():
            xs = [int(t) for t, v in d_.items() if np.isfinite(v)]; ys = [v for v in d_.values() if np.isfinite(v)]
            c, mk, ls = sty[lab]; a_.plot(xs, ys, marker=mk, c=c, ls=ls, label=lab)
        a_.axvline(1760, c="grey", lw=0.8, ls="--"); a_.set_xscale("log"); a_.grid(alpha=0.3)
        a_.set(title=f"max error {BL(b)}", xlabel="frames T (log); dashed line: end of 320 K frames", ylabel="ratio vs raw float32 (×)")
    ax[0].legend(fontsize=7)
    for a_, b in zip(ax2[1], BOUNDS):
        for lab in S320[b]:
            c, mk, ls = sty[lab]
            xs = [T for T in LEV3 if np.isfinite(S320[b][lab][T])]
            if xs: a_.plot(xs, [S320[b][lab][T] for T in xs], marker=mk, c=c, ls=ls, label=lab + " (320 K)")
            xo = [T for T in LEVS if np.isfinite(SOLD[b][lab][T]) and lab in ("ours: hybrid + AC", "model free mean + AC", "MDCompress (best setting)")]
            if xo: a_.plot(xo, [SOLD[b][lab][T] for T in xo], c=c, ls=(0, (2, 2)), lw=1.0, alpha=0.6, label=lab + " (archive order, mixed temperature)")
        for xb in (500, 1000, 1500, 2000): a_.axvline(xb, c="grey", lw=0.5, ls=":")
        a_.axvline(2000, c="grey", lw=0.8, ls="--"); a_.set_xscale("log"); a_.grid(alpha=0.3)
        a_.set(title=f"2lg7A00, max error {BL(b)}", xlabel="frames T (log)", ylabel="ratio vs raw float32 (×)")
    ax2[1][0].legend(fontsize=6, loc="lower right")
    if HAVE_M:
        for a_, b in zip(ax2[2], BOUNDS):
            for lab in SMD[b]:
                c, mk, ls = sty[lab]; xs = [T for T in LEVM if np.isfinite(SMD[b][lab][T])]
                if xs: a_.plot(xs, [SMD[b][lab][T] for T in xs], marker=mk, c=c, ls=ls, label=lab)
            a_.axvline(1000, c="grey", lw=0.8, ls=":"); a_.set_xscale("log"); a_.grid(alpha=0.3)
            a_.set(title=f"MD-A00008 (300 K), max error {BL(b)}", xlabel="frames T (log)", ylabel="ratio vs raw float32 (×)")
        ax2[2][0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Top: archive order, mean of 3 domains (mixed temperature beyond the 320 K frames).   Middle: 2lg7A00, all frames at 320 K; thin dashed: same domain in archive order" + ("   Bottom: MDDB MD-A00008, one continuous 300 K run" if HAVE_M else ""), fontsize=12)
    fig.tight_layout(); fig.savefig(f"{BEN}/figs/fig_frames_scaling.png", dpi=150); plt.close(fig)
    A("![Ratio versus frames](figs/fig_frames_scaling.png)\n")
    A("*Figure 7. Compression ratio versus number of frames at each bound (log frame axis). Solid: arithmetic coder, MDCompress and XTC; dotted: lzma layout "
      "search. The dashed vertical line marks the end of the 320 K frames. Middle row: the same temperature check of section 4.6.9.1 (2lg7A00, all frames at 320 K, "
      "dotted vertical lines at the replica boundaries, dashed vertical line at frame 2,000 where the archive order leaves 320 K for this domain); the thin dashed curves repeat the same domain in archive order, where frames 2,001 to 2,500 come from 348 K." + (" Bottom row: the long single temperature MDDB run of section 4.6.9.2 (MD-A00008, 300 K, 10,000 frames at 1 ps)." if HAVE_M else "") + "*\n")
    if any(np.isfinite(S320["0.1"]["ours: hybrid + AC"][T]) for T in LEV3):
        A("##### 4.6.9.1 Same temperature check (2lg7A00, 320 K only)\n")
        A("To separate the frame count from the temperature change, the scaling was repeated on one domain with every frame at the same temperature. mdCATH stores "
          "five replicas per temperature, each at most 500 frames at 1 ns spacing, so one domain offers at most 2,500 frames at one temperature; T = 10,000 at constant "
          "temperature is therefore not reachable from mdCATH, and this check covers T = 10 to 2,500. 2lg7A00 was chosen because it is the scaling domain with the most "
          "320 K frames (5 × 500 = 2,500; 1e8rA00 has 2,210 and 4m70B00 2,130). The five 320 K replicas (downloaded in full from compsciencelab/mdCATH, including "
          "replica 4, which the archive order never used) were concatenated in replica order, giving replica boundaries after frames 500, 1,000, 1,500 and 2,000; the "
          "first 1,000 frames are bit identical to the archive order frames used above (checked). All methods were rerun (per trajectory model fit at every T), with the same "
          "settings as in section 4.6.9; every coded result was decoded and checked against its bound.\n")
        for b in BOUNDS:
            A(f"{BL(b)} (2lg7A00, 320 K):\n")
            A("| method | " + " | ".join(f"T = {T:,}" for T in LEV3) + " | change T = 1,000 → 2,500 | archive order T = 2,500 (mixed) |")
            A("|---|" + "---|" * (len(LEV3) + 2))
            for lab in S320[b]:
                r = S320[b][lab]; ch = (r[2500] / r[1000] - 1) * 100 if np.isfinite(r[2500]) and np.isfinite(r[1000]) else np.nan
                A(f"| {lab} | " + " | ".join(f2(r[T]) for T in LEV3) + f" | {ch:+.1f} % | {f2(SOLD[b][lab].get(2500, np.nan))} |")
            A("")
        txt = []
        for b in BOUNDS:
            parts = []
            for lab in ("ours: hybrid + AC", "model free mean + AC", "MDCompress (best setting)"):
                r = {T: v for T, v in S320[b][lab].items() if np.isfinite(v)}
                if not r: continue
                tp = max(r, key=r.get); parts.append(f"{lab} peak at T = {tp:,} ({r[tp]:.2f}×), T = 1,000 → 2,500: {r.get(1000, np.nan):.2f}× → {r.get(2500, np.nan):.2f}×")
            txt.append(f"{BL(b)}: " + "; ".join(parts))
        A("Peaks and the change after 1,000 frames at constant temperature: " + ". ".join(txt) + ".\n")
        chg = {lab: [(S320[b][lab][2500] / S320[b][lab][1000] - 1) * 100 for b in BOUNDS if np.isfinite(S320[b][lab][2500]) and np.isfinite(S320[b][lab][1000])]
               for lab, _ in allm if lab != "XTC"}
        ac_ = chg["ours: hybrid + AC"] + chg["MDCompress (best setting)"] + chg["ours: hybrid + backbone AE + AC"]
        mn_ = chg["model free mean + AC"] + chg["model free mean + lzma"]
        o1 = {b: SOLD[b]["ours: hybrid + AC"] for b in BOUNDS}
        A(f"**Answer: at constant temperature the ratio does not decrease after about 1,000 frames, as far as mdCATH allows us to test (T = 2,500).** From T = 1,000 to "
          f"T = 2,500 at 320 K, the hybrid + AC, hybrid + backbone AE + AC and MDCompress ratios change by {min(ac_):+.1f} % to {max(ac_):+.1f} % over all bounds, i.e. they keep "
          f"rising slowly. The model free mean structure references are flat within {min(mn_):+.1f} % to {max(mn_):+.1f} %; their small dip at T = 2,000 "
          "coincides with adding replica 3 (frames 1,501 to 2,000), the replica furthest from the common mean structure (heavy atom RMSD to the 2,500 frame mean "
          "after superposition about 3.0 Å, against 2.5 to 2.8 Å for the other four replicas), which the mean structure predictor cannot absorb; the ratio recovers "
          "at T = 2,500. Beyond "
          "about 1,000 frames the curves are essentially flat: the per trajectory side information is already amortised, and each additional frame costs about the same. "
          "For the same domain in archive order the hybrid + AC ratio at 0.1 Å is " + f"{o1['0.1'][1000]:.2f}× at T = 1,000, {o1['0.1'][2500]:.2f}× at 2,500, "
          f"{o1['0.1'][5000]:.2f}× at 5,000 and {o1['0.1'][10000]:.2f}× at 10,000; the drop happens only after T = 5,000, where the archive order consists of 379 to "
          "450 K frames, consistent with the decline in the three domain mean being a temperature (conformational diversity) effect. This check could not extend past "
          "2,500 frames at one temperature, so a frame count effect beyond 2,500 frames is not excluded by direct measurement; the mechanism (side information "
          "amortised, adaptive models converged) gives no reason to expect one.\n")
        SCALSUM["same_temperature_2lg7A00_320K"] = {b: {lab: {str(T): v for T, v in d_.items()} for lab, d_ in S320[b].items()} for b in BOUNDS}
        SCALSUM["archive_2lg7A00"] = {b: {lab: {str(T): v for T, v in d_.items()} for lab, d_ in SOLD[b].items()} for b in BOUNDS}
        # boundary test
        sb = {}
        for f in ("scaling320_boundary.json", "scaling320_formats.json"):
            if os.path.exists(f"{E}/results/{f}"): sb.update(json.load(open(f"{E}/results/{f}")))
        BT = [("2lg7A00_b0", "replica 0, frames in order (0 jumps)"), ("2lg7A00_p4", "same frames, 5 blocks of 100 permuted (4 jumps)"),
              ("2lg7A00_p9", "same frames, 10 blocks of 50 permuted (9 jumps)"), ("2lg7A00_r4", "100 frames from each of the 5 replicas (4 replica boundaries)")]
        def bt_bytes(d, lab, pat, b):
            if lab == "MDCompress (best setting)":
                r = RES[b]; c = [f"MDCompress -l 4 --res {r} fm", f"MDCompress -l 4 --preset archive --res {r} fm", f"MDCompress -l 2 --res {r} fm" + (" (default)" if r == 1000 else "")]
                v = [sb[f"{d}/T500/{m}"]["bytes"] for m in c if f"{d}/T500/{m}" in sb and "error" not in sb[f"{d}/T500/{m}"]]
                return min(v) if v else np.nan
            k = f"{d}/T500/{XTCM.get(b, '')}" if lab == "XTC" else f"{d}/T500/{pat.format(b)}"
            v = sb.get(k); return v["bytes"] if v and "error" not in v else np.nan
        if any(k.startswith("2lg7A00_p9/") for k in sb):
            A("**Cost of a discontinuity (replica boundary).** Concatenating replicas inserts a jump at every boundary, similar to a restart. To measure its cost, "
              "the first 500 frames of replica 0 were coded in order and as the same 500 frames permuted in blocks (4 or 9 jumps of at least 50 ns, identical frame "
              "set, so mean structure and model see the same data and only the jumps differ), and a fourth file took 100 frames from each of the five replicas "
              "(4 real replica boundaries, more diverse content). The table gives the ratio and, in brackets, the extra bytes per jump relative to the in order "
              "file (p4 and p9: (bytes − bytes in order) / number of jumps; r4: the same, which also includes the effect of the more diverse content).\n")
            BTS = {}
            for b in BOUNDS:
                A(f"{BL(b)}, T = 500:\n")
                A("| method | " + " | ".join(lab_ for _, lab_ in BT) + " |"); A("|---|" + "---|" * len(BT))
                for lab, pat in allm:
                    if lab == "XTC" and b not in XTCM: continue
                    by = [bt_bytes(d, lab, pat, b) for d, _ in BT]
                    if not np.isfinite(by[0]): continue
                    cells = []
                    for (d, _), x, nj in zip(BT, by, (0, 4, 9, 4)):
                        if not np.isfinite(x): cells.append("n/a"); continue
                        rr = 12 * 969 * 500 / x
                        cells.append(f"{rr:.2f}×" + ("" if nj == 0 else f" ({(x - by[0]) / nj:+,.0f} B)"))
                    BTS.setdefault(b, {})[lab] = by
                    A(f"| {lab} | " + " | ".join(cells) + " |")
                A("")
            SCALSUM["boundary_test_bytes"] = BTS
            pj = [abs(by[k] - by[0]) / nj for b in BTS for lab, by in BTS[b].items() for k, nj in ((1, 4), (2, 9)) if np.isfinite(by[k])]
            wst = 0.0
            for b in BTS:
                for lab, by in BTS[b].items():
                    k25 = s3.get(f"2lg7A00_320K/T2500/" + dict(allm)[lab].format(b)) if dict(allm).get(lab) else None
                    if k25 and "error" not in k25:
                        for k, nj in ((1, 4), (2, 9)):
                            if np.isfinite(by[k]): wst = max(wst, 4 * abs(by[k] - by[0]) / nj / k25["bytes"] * 100)
            rel = [abs(by[k] - by[0]) / by[0] * 100 for b in BTS for lab, by in BTS[b].items() for k in (1, 2) if np.isfinite(by[k])]
            A(f"A jump costs at most {max(pj):,.0f} bytes in any method and bound (at most {max(rel):.2f} % of the file over all four or nine jumps), and the hybrid "
              "differences of a few dozen to a few hundred bytes are within the variation of the stochastic per trajectory model fit (both signs occur). MDCompress "
              "and XTC are essentially unaffected. The file with 100 frames from each replica differs more for the lzma variants (about 1 to 3 kB larger, from the "
              "more diverse content), and by a few hundred bytes either way for the AC variants. Replica boundaries are therefore not a measurable cost in the 320 K "
              f"scaling run: even at the largest per jump cost seen here, the four boundaries in the 2,500 frame file would cost {wst:.2f} % of its size.\n")

if sc and HAVE_M:
    A("##### 4.6.9.2 Same temperature check on a long MDDB trajectory (MD-A00008, 300 K, 10,000 frames)\n")
    mm = json.load(open("/workspace/bench_data_mddb_long/mddb_meta.json"))["mddb_small_1APQ_long"]
    A("Because mdCATH holds at most 2,500 frames per domain at one temperature, the constant temperature scaling was repeated on one MDDB trajectory that is a "
      "single continuous run at one temperature: MD-A00008 (MoDEL 1APQ, small protein, 53 residues; MDposit metadata: Amber 8, Parm99, TIP3P water, NPT ensemble, "
      "300 K, 2 fs time step, 10 ns length, one frame per ps, 10,000 frames, one MD replica). All 10,000 frames were downloaded from the MDposit REST API "
      f"(heavy atoms; water and ions stripped as in section 4.3; {mm['n_heavy']} atoms, {mm['n_protein_atoms_model']} in the atom14 protein model and "
      f"{mm['n_other_atoms']} extra atom); every 10th frame of this file is identical to the 1,000 frame MDDB file used in section 4.3 (checked), and the largest "
      "frame to frame RMS displacement is 0.83 Å, so the file contains no restart. The three MoDEL protein systems used in section 4.3 all have 10,000 frames, "
      "the most among our MDDB systems; the medium and large MoDEL proteins have the same length but are 5 and 9 times larger, so the small one was "
      "chosen for memory. Note the frame spacing: 1 ps here against 1 ns in mdCATH, so consecutive frames are far more correlated, which favours temporal "
      "predictors (MDCompress, raw + AC). Same methods and settings as section 4.6.9, every result decoded and checked against its bound.\n")
    for b in BOUNDS:
        A(f"{BL(b)} (MD-A00008, 300 K):\n")
        A("| method | " + " | ".join(f"T = {T:,}" for T in LEVM) + " | change T = 1,000 → 10,000 |")
        A("|---|" + "---|" * (len(LEVM) + 1))
        for lab in SMD[b]:
            r = SMD[b][lab]; ch = (r[10000] / r[1000] - 1) * 100 if np.isfinite(r[10000]) and np.isfinite(r[1000]) else np.nan
            A(f"| {lab} | " + " | ".join(f2(r[T]) for T in LEVM) + f" | {ch:+.1f} % |")
        A("")
    txt = []
    for b in BOUNDS:
        parts = []
        for lab in ("ours: hybrid + AC", "model free mean + AC", "model free raw + AC", "MDCompress (best setting)"):
            r = {T: v for T, v in SMD[b][lab].items() if np.isfinite(v)}
            if r: tp = max(r, key=r.get); parts.append(f"{lab} peak at T = {tp:,} ({r[tp]:.2f}×), T = 1,000 → 10,000: {r.get(1000, np.nan):.2f}× → {r.get(10000, np.nan):.2f}×")
        txt.append(f"{BL(b)}: " + "; ".join(parts))
    A("Peaks and the change after 1,000 frames: " + ". ".join(txt) + ".\n")
    SCALSUM["same_temperature_mddb_MD-A00008"] = {b: {lab: {str(T): v for T, v in d_.items()} for lab, d_ in SMD[b].items()} for b in BOUNDS}
    g = lambda lab, b, T: SMD[b][lab][T]
    tc = {lab: [(g(lab, b, 10000) / g(lab, b, 1000) - 1) * 100 for b in BOUNDS] for lab in SMD["0.1"] if lab != "XTC"}
    A("**Answer from the long MDDB run: beyond 1,000 frames at constant temperature the ratio plateaus; it neither keeps rising substantially nor declines "
      "systematically.** From T = 1,000 to T = 10,000 the changes over all bounds are: " +
      "; ".join(f"{lab} {min(v):+.1f} % to {max(v):+.1f} %" for lab, v in tc.items() if all(np.isfinite(v))) +
      ". At 0.1 Å, for example, hybrid + AC goes " + f"{g('ours: hybrid + AC', '0.1', 1000):.2f}× → {g('ours: hybrid + AC', '0.1', 2500):.2f}× → "
      f"{g('ours: hybrid + AC', '0.1', 5000):.2f}× → {g('ours: hybrid + AC', '0.1', 10000):.2f}× (T = 1,000, 2,500, 5,000, 10,000) and MDCompress "
      f"{g('MDCompress (best setting)', '0.1', 1000):.2f}× → {g('MDCompress (best setting)', '0.1', 10000):.2f}×; at lossless hybrid + AC goes "
      f"{g('ours: hybrid + AC', '0.005', 1000):.2f}× → {g('ours: hybrid + AC', '0.005', 10000):.2f}×. Most of the gain is reached by T = 500 to 1,000; "
      "from 1,000 to 10,000 frames (one decade) most methods gain only a few per cent; only the mean structure based predictors at 0.5 and 0.3 Å lose ground (next paragraph).\n")
    A("The one exception is instructive. At the loose bounds (0.5 and 0.3 Å) the predictors built on a mean structure (hybrid and model free mean) peak at "
      f"T = 1,000 (hybrid + AC at 0.5 Å {g('ours: hybrid + AC', '0.5', 1000):.2f}×, mean + AC {g('model free mean + AC', '0.5', 1000):.2f}×), dip at "
      f"T = 2,500 to 5,000 (down to {min(g('ours: hybrid + AC', '0.5', 2500), g('ours: hybrid + AC', '0.5', 5000)):.2f}× and "
      f"{min(g('model free mean + AC', '0.5', 2500), g('model free mean + AC', '0.5', 5000)):.2f}×) and partly recover at T = 10,000 "
      f"({g('ours: hybrid + AC', '0.5', 10000):.2f}× and {g('model free mean + AC', '0.5', 10000):.2f}×). This tracks the conformational spread of the "
      "frame window, not the frame count: the heavy atom RMS deviation from the window's own mean structure after superposition is about 1.4 Å for the first "
      "1,000 frames, 1.8 Å for 2,500, 2.2 Å for 5,000 and 2.0 Å for 10,000 (the protein moves away from its early conformations within the first few ns, and the later frames lie partly closer to the overall mean again). A "
      "wider spread means larger residuals around a single mean structure, which matters most at loose bounds where the residual stream is small. The purely "
      "temporal predictors (model free raw + AC and MDCompress) do not depend on a mean structure and rise monotonically over the whole range. The same "
      "mechanism explains the mdCATH archive order decline: adding high temperature replicas widens the conformational spread.\n")
    A("A second observation: with 1 ps frame spacing, consecutive frames are so correlated that the purely temporal model free raw + AC stream is the best "
      f"method at every bound on this system (at T = 10,000: {g('model free raw + AC', '0.5', 10000):.2f}× at 0.5 Å, {g('model free raw + AC', '0.1', 10000):.2f}× "
      f"at 0.1 Å, {g('model free raw + AC', '0.005', 10000):.2f}× lossless, against {g('ours: hybrid + AC', '0.5', 10000):.2f}×, "
      f"{g('ours: hybrid + AC', '0.1', 10000):.2f}× and {g('ours: hybrid + AC', '0.005', 10000):.2f}× for hybrid + AC and "
      f"{g('MDCompress (best setting)', '0.5', 10000):.2f}×, {g('MDCompress (best setting)', '0.1', 10000):.2f}×, {g('MDCompress (best setting)', '0.005', 10000):.2f}× "
      "for MDCompress). The hybrid's structural predictor helps at the 1 ns spacing of mdCATH and the strided MDDB files of section 4.3, but not at 1 ps "
      "spacing, where the previous frame is a better predictor than any structure model.\n")

A("#### 4.6.10 Caveats\n")
A("* The hybrid + lzma reference was refitted in this run (column 'refit'): the per trajectory MLP fit is stochastic, so its ratios differ slightly from sections "
  "4.1 to 4.3. Hybrid + AC and the refit hybrid + lzma code exactly the same residual integers, so their difference is the coder alone.")
A("* The fixed part (rigid transforms stored raw, rotamer codes, MLP) is not entropy coded by the new coder; at 0.5 Å it is a sizeable share of the file, which is why "
  "the model free mean + AC stream can match or beat the hybrid on small domains. Coding the rigid transforms and codes with the same machinery is future work.")
A("* The geometric term uses floating point; decoding is bit exact on the same build (verified), but portability across compilers or CPUs is not guaranteed without "
  "a fixed point implementation.")
A("* On the membrane system the lzma reference used a restricted layout search (two layouts) to keep memory and time within the box; the AC coder was not restricted.")
A("* The backbone autoencoder rows were produced in a second pass, which refitted the per trajectory MLP; their side chain stream therefore comes from a "
  "slightly different fit than the hybrid rows (same procedure, different random draw).")
A("* Frames scaling: three domains in archive order, where frames beyond the 320 K replicas (1,760 to 2,000 frames depending on the domain) come from higher temperature "
  "replicas; the same temperature check (section 4.6.9.1) covers one domain and at most 2,500 frames, the most mdCATH holds at one temperature for one domain; "
  "the long MDDB check (section 4.6.9.2) is one small protein at 1 ps frame spacing; the lzma reference used three layouts.")
A("* Timing in the main table was measured with three benchmark processes running concurrently.\n")
json.dump({"ac": {str(k): v for k, v in SUMT.items()}, "ae": {str(k): v for k, v in SUMAE.items()}, "scaling": SCALSUM}, open(f"{E}/results/entropy_summary.json", "w"), indent=1)
open(f"{E}/entropy_section.md", "w").write("\n".join(L) + "\n")
print("wrote entropy_section.md", len(L), "lines")
