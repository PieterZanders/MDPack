"""Write an untracked AEZ-12 results block from results/entropy_summary.json. Does not change the README."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
EE = json.load(open(os.path.join(HERE, "results/entropy_summary.json"))); E = EE["ac"]; AE = EE.get("ae", {}); SC = EE.get("scaling", {})
bb = ("0.5", "0.3", "0.2", "0.1", "0.005"); bl = lambda b: "lossless" if b == "0.005" else f"{b} Å"
L = ["<!-- AEZ12 start -->", "## Context modelled arithmetic coder for the residual streams (`gpu/residual_entropy.c`, `gpu/residual_entropy.py`)", "",
     "Replaces the lzma layout search under the hybrid's two residual streams (backbone plus other atoms; side chains). Design follows MDCompress's coding stage "
     "(read from its source, notes in `benchmark/entropy/mdcompress_notes.md`): an MDCompress level 4 style translate and rotate prediction from three already decoded "
     "reference atoms, combined with the reference atom's and the atom's own previous residuals by per atom class integer weights chosen by the encoder; zero flag, sign, "
     "Elias gamma binarisation; logistic mixing of adaptive context models (class, axis, activity of neighbouring errors, predictor disagreement) plus an APM; LZMA style "
     "binary range coder. All coder side information is counted. Every stream is decoded and compared bit for bit, and max error ≤ bound + 0.001 Å is checked.", "",
     "Usage: `from gpu import residual_entropy as RE; blob, info = RE.encode(q, cls, base, s, ref, R3, Xc); q2 = RE.decode(blob, q.shape, cls, base, s, Xc)` "
     "(see `benchmark/entropy/bench_entropy.py`). Full results: section 4.6 of `benchmark/benchmark_report.pdf`.", ""]
for T in ("1000", "250"):
    if T not in E: continue
    e = E[T]
    L += [f"mdCATH, T = {T}, mean ratio over {e['0.5']['n']} domains:", "", "| max error | hybrid + AC | hybrid + lzma | best AC variant per trajectory | MDCompress best setting |", "|---|---|---|---|---|"]
    L += [f"| {bl(b)} | {e[b]['hac']:.2f} | {e[b]['hlz']:.2f} | {e[b]['bac']:.2f} | {e[b]['mdc']:.2f} |" for b in bb] + [""]
if "1000" in AE:
    L += ["Variant hybrid + backbone autoencoder (backbone predicted by the shared autoencoder from fp16 latents, counted), new coder, T = 1000: " +
          ", ".join(f"{bl(b)} {AE['1000'][b]['ae_ac']:.2f}x" for b in bb) + " (latents cost more than the residuals they save).", ""]
if SC:
    L += ["Frames scaling (1e8rA00, 4m70B00, 2lg7A00; T = 10 to 10000; frames beyond 1760 are higher temperature replicas): see section 4.6.9 and `benchmark/figs/fig_frames_scaling.png`.", ""]
    S3 = SC.get("same_temperature_2lg7A00_320K", {})
    if S3:
        h = S3["0.1"]["ours: hybrid + AC"]
        L += [f"Same temperature check (2lg7A00, 5 replicas at 320 K concatenated, T up to 2500, the most mdCATH offers at one temperature; section 4.6.9.1): no decrease after 1000 frames (hybrid + AC at 0.1 A: {h['1000']:.2f}x at T = 1000, {h['2500']:.2f}x at T = 2500); replica boundaries cost at most a few hundred bytes each. The decline in the archive order curves comes with the higher temperature frames.", ""]
    SM3 = SC.get("same_temperature_mddb_MD-A00008", {})
    if SM3:
        h = SM3["0.1"]["ours: hybrid + AC"]; m5 = SM3["0.5"]["ours: hybrid + AC"]
        L += [f"Long single temperature run (MDDB MD-A00008, one continuous 10 ns run at 300 K, 10000 frames at 1 ps; section 4.6.9.2): the ratio plateaus beyond 1000 frames (hybrid + AC at 0.1 A: {h['1000']:.2f}x at T = 1000, {h['10000']:.2f}x at T = 10000; at 0.5 A {m5['1000']:.2f}x -> {m5['10000']:.2f}x, a small dip that follows the conformational spread of the window). At 1 ps spacing the model free raw + AC stream is best at every bound.", ""]
    for b in ("0.1", "0.005"):
        if b not in SC: continue
        Ts = list(next(iter(SC[b].values())).keys())
        L += [f"{bl(b)}:", "", "| method | " + " | ".join(f"T={t}" for t in Ts) + " |", "|---|" + "---|" * len(Ts)]
        L += [f"| {m} | " + " | ".join("n/a" if v != v else f"{v:.2f}" for v in d.values()) + " |" for m, d in SC[b].items()] + [""]
_ap = os.path.join(HERE, "results/ablation_summary.json")
if os.path.exists(_ap):
    AB = json.load(open(_ap)); t1, t2 = AB.get("mdcath_T1000"), AB.get("mdcath_T250")
    if t1 and t2:
        L += ["Ablation of the side chain predictor (section 4.7.6, `benchmark/figs/fig_ablation.png`), mdCATH T = 1000, new coder, full / no rotamer codes / no MLP / packer only: " +
              "; ".join(f"{bl(b)} " + " / ".join(f"{t1['ratio_ac'][v][b]:.2f}x" for v in ("full", "no codes", "no MLP", "neither")) for b in ("0.5", "0.1", "0.005")) +
              ". Codes and MLP pay only together; at T = 250 the packer alone is best at every bound.", ""]
L.append("<!-- AEZ12 end -->")
blk = "\n".join(L) + "\n"
outp = os.path.join(HERE, "readme_results_block.md")
open(outp, "w").write(blk); print("wrote", outp)
