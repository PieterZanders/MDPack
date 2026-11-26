#!/bin/bash
# Reproduce benchmark/benchmark_report.{md,html,pdf} (md_compress benchmark report).
# Generated reports, figures and result JSON are written locally and are not tracked.
#
#   ./reproduce.sh --stage report            # rebuild the report from result JSON produced by earlier stages (no GPU)
#   ./reproduce.sh --stage data,baselines    # comma separated stages, run in the order given
#   ./reproduce.sh --all                     # every stage in order: data train baselines entropy scaling ablation codec report
#   ./reproduce.sh --stage entropy --dry-run # print the commands only
#   ./reproduce.sh --list                    # stages with GPU need and the environments they expect
#
# See the README section "Reproducing the report". Lines marked  [HAND]  were done by hand in the original run; lines
# marked  [INFERRED]  were launched ad hoc and are reconstructed from notes and logs.
# All benchmark scripts are resumable: rows already present in their output JSON are skipped. Delete a result file to
# recompute it. This stage runner does not modify the README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # /workspace/md_compress
BEN="$ROOT/benchmark"; ENT="$BEN/entropy"; R="$BEN/results"; ER="$ENT/results"
P="${BENCH_PY:-/workspace/conda/envs/bench/bin/python}"          # conda env "bench" (python 3.11, CPU pytorch, MDAnalysis, mdtraj, ...)
SYS="${SYS_PY:-python3}"                                         # system python3: markdown + weasyprint (PDF), matplotlib, torch (bench_ours.py)
DATA=/workspace/md_compress_gpu_data                             # mdCATH conversions (box)
D8A=1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00
D8B=1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00
D16=$D8A,$D8B
MDDB6=mddb_small_1APQ,mddb_dna_1d89,mddb_rna_2l8f,mddb_protdna_FOXA3,mddb_medium_12CA,mddb_large_1AYX
MDDB7=$MDDB6,mddb_membrane_Eprot
LZ3=none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma; LZ2=none-nat-vint-lzma,none-nat-planes-lzma
LEV=10,50,100,250,500,1000,2500,5000,10000
ALL="data train baselines entropy scaling ablation codec report"
DRY=0; STAGES=""

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }
list() { cat <<'L'
stage      GPU   needs
data       no    internet (HuggingFace mdCATH, MDposit)
train      YES   a GPU machine; runs gpu/pipeline.sh
baselines  part  GROMACS, MDCompress and pyPcazip locally; compresstraj and cg2all are set up to run on Colab
entropy    no    results_gpu/runs/e2e_S_prior and a built gpu/_residual_entropy.so
scaling    no    outputs of the data stage
ablation   no    the shared checkpoint and the built coder
codec      no    GROMACS, MDCompress, cg2all, pyPcazip, hdf5plugin
report     no    result JSON from earlier stages (not tracked); writes the report locally (not tracked); does not change the README
L
}

while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGES="${2//,/ }"; shift 2;;
    --stage=*) STAGES="${1#--stage=}"; STAGES="${STAGES//,/ }"; shift;;
    --all) STAGES="$ALL"; shift;;
    --dry-run|-n) DRY=1; shift;;
    --list) list; exit 0;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument $1"; usage; exit 2;;
  esac
done
[ -n "$STAGES" ] || { usage; exit 2; }
for s in $STAGES; do case " $ALL " in *" $s "*) ;; *) echo "unknown stage $s (choose from: $ALL)"; exit 2;; esac; done

run() { echo "+ $*"; [ "$DRY" = 1 ] || eval "$@"; }
note() { echo "# $*"; }
need() { for f in "$@"; do [ -e "$f" ] || { [ "$DRY" = 1 ] && echo "# (missing now: $f)" || { echo "missing input: $f"; exit 1; }; }; done; }
stamp() { echo; echo "=== stage $1 ($(date '+%F %H:%M %Z'))"; }

# ------------------------------------------------------------------------------------------------- data
stage_data() {
  stamp data
  cd "$ROOT"
  note "mdCATH test/val split + 60 train domains on the box (same superfamily split as the GPU run; test and val pools do not depend on --n_train)"
  run "$P gpu/prep_mdcath.py --out $DATA/data_mdcath --n_train 60 --workers 12"         # [INFERRED] prep_box.log: 50 test, 30 val, 60 train, 173 s
  run "$P gpu/prep_external.py $DATA/data_external"                                       # [INFERRED] output dir name from the box layout
  run "$P gpu/probe_mdcath_lengths.py"                                                    # -> results_gpu/frames/mdcath_lengths.json
  note "[HAND] results_gpu/frames/selected_domains.json: 16 of the 47 test domains with >= 10,000 frames outside the held-out replica,"
  note "       picked evenly by size; no script wrote it. The domain id list is kept in the repository and used as is."
  need results_gpu/frames/selected_domains.json
  run "$P gpu/fetch_long.py"                                                              # -> $DATA/data_long/<dom>.npz (16 domains, 2.8 GB, ~5 min)
  run "$P $BEN/scripts/export_data.py"                                                    # -> /workspace/bench_data/<dom>.pdb, <dom>_T1000.npy, meta.json
  run "$P $BEN/scripts/fetch_mddb.py"                                                     # -> /workspace/bench_data/mddb_*.{pdb,npy,ours.npz}, mddb_meta.json (MDposit REST API)
  run "$P $ENT/export_long.py"                                                            # -> /workspace/bench_data_long (10,000 frames of 1e8rA00, 4m70B00, 2lg7A00)
  note "[HAND] full mdCATH HDF5 file of 2lg7A00 for the 320 K export (the original run also fetched 1e8rA00 and 4m70B00, unused); URL as in gpu/fetch_long.py"
  run "mkdir -p /workspace/mdcath_h5 && [ -s /workspace/mdcath_h5/mdcath_dataset_2lg7A00.h5 ] || curl -sL -o /workspace/mdcath_h5/mdcath_dataset_2lg7A00.h5 https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_2lg7A00.h5"
  run "$P $ENT/export_320K.py"                                                            # -> /workspace/bench_data_320K
  run "$P $BEN/scripts/fetch_mddb_long.py mddb_small_1APQ_long"                           # -> /workspace/bench_data_mddb_long (MD-A00008, 10,000 frames)
  note "MDposit and HuggingFace serve the data live; a changed upstream record changes the inputs (the report used downloads made 22 to 25 Nov 2026)."
}

# ------------------------------------------------------------------------------------------------- train
stage_train() {
  stamp train
  note "GPU stage. Original run: Colab L4 VMs, repository copied to /content/md_compress, gpu/pipeline.sh (data prep with 300 train domains,"
  note "buffers, codec + 4 packers, end-to-end fine tuning, eval). The first two VMs were reclaimed; the final checkpoints came from a"
  note "resume script on a third VM (packers to 30 min, e2e S prior / no prior 10 min) that is NOT in the repository (gap), so"
  note "retraining gives a comparable, not an identical, checkpoint. The benchmark uses results_gpu/runs/e2e_S_prior/best.pt (kept)."
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && [ -d /content/md_compress ]; then
    run "cd /content/md_compress && T_CODEC=15 T_PACK=35 T_E2E=15 bash gpu/pipeline.sh"   # writes /content/runs, /content/export
    note "then copy /content/export/runs/* to $ROOT/results_gpu/runs/ on the box (best_weights.pt per run, history, config)"
  else
    note "no GPU VM with /content/md_compress here: skipped. On a GPU VM run:"
    note "  cd /content/md_compress && T_CODEC=15 T_PACK=35 T_E2E=15 bash gpu/pipeline.sh"
  fi
  need "$ROOT/results_gpu/runs/e2e_S_prior/best.pt"
  note "optional CPU evaluation on the box (main report sections, not needed by the benchmark report):"
  run "cd $ROOT && $P gpu/eval_gpu.py --data $DATA/data_mdcath --runs results_gpu/runs"   # [INFERRED] STATE: 'CPU eval running: eval/results.json'
}

# ------------------------------------------------------------------------------------------------- baselines
stage_baselines() {
  stamp baselines
  cd "$BEN"
  need /workspace/bench_data/meta.json /workspace/bench_data/mddb_meta.json /workspace/conda/envs/gmx/bin/gmx /workspace/bench_tools/mdcbin/mdcompress
  note "standard formats + MDCompress (box CPU)"
  run "$P scripts/bench_formats.py --systems $D16 > logs/formats_mdcath.log 2>&1"          # [INFERRED] default --T 1000,250 -> results/formats_raw.json
  run "$P scripts/bench_formats.py --systems $MDDB7 --T 1000 > logs/formats_mddb.log 2>&1" # scripts/queue_cpu.sh (membrane: PDB streamed, H5MD gzip 9 aborted, see report)
  run "$P scripts/bench_formats.py --systems $D16,$MDDB6 --only 'MDCompress -l 4 --preset archive' --out $R/formats_mdc_archive_raw.json > logs/formats_mdc_archive.log 2>&1"   # [INFERRED]
  run "$P scripts/bench_formats.py --systems mddb_membrane_Eprot --T 1000 --only 'MDCompress -l 4 --preset archive' --out $R/formats_mdc_archive_raw.json > logs/formats_mdc_archive_membrane.log 2>&1"   # [INFERRED]
  run "$P scripts/bench_pcazip.py --systems $D16,$MDDB7 --T 1000,250 > logs/pcazip.log 2>&1"   # scripts/queue_cpu.sh
  note "our methods with the lzma coders (system python3 as in the original queue files; results land in $ROOT/results, aggregate reads them there)"
  run "$SYS scripts/bench_ours.py --systems $D8A --out $ROOT/results/ours_raw_partA.json --threads 4 > logs/ours_A.log 2>&1"   # [INFERRED] STATE: part A, cwd quirk
  run "$SYS scripts/bench_ours.py --systems $D8B --out $ROOT/results/ours_raw_partB.json --threads 4 > logs/ours_B.log 2>&1"   # [INFERRED]
  run "$SYS scripts/bench_ours.py --systems $MDDB6 --T 1000 --out $ROOT/results/ours_raw_mddb.json --threads 4 > logs/ours_mddb.log 2>&1"   # scripts/queue_ours_mddb.sh
  run "$SYS scripts/bench_ours.py --systems mddb_membrane_Eprot --T 1000 --coders $LZ2 --no-raw --out $ROOT/results/ours_raw_mddb.json --threads 4 > logs/ours_mddb_membrane.log 2>&1"   # STATE Resume 3
  run "bash scripts/run_timing.sh"                                                       # clean timing pass -> results/timing_*.json
  note "[HAND] compresstraj and cg2all ran on Colab T4 GPUs (sessions driven by hand, two were lost; cutoff 06:45 CET 25 Nov):"
  note "  upload /workspace/bench_colab/up (drivers, compresstraj source, data) to /content/bench, run benchmark/scripts/colab_run_all.sh there,"
  note "  pull /content/bench/out and /content/bench/out_cg to /workspace/bench_colab_out/{out,out_cg}. Coverage in the report: compresstraj 14/16 mdCATH,"
  note "  no MDDB, no latent 16; cg2all CA p100 on 2/5 MDDB protein systems. The pulled outputs are kept in /workspace/bench_colab_out."
  need /workspace/bench_colab_out
  run "$P scripts/import_colab.py"                                                      # -> results/compresstraj_raw.json, results/cg2all_raw.json
  note "[HAND] results/report_extra.json (narrative: summary bullets, MDDB paragraphs, discussion, limitations) may be written by hand; it is optional and is not tracked."
}

# ------------------------------------------------------------------------------------------------- entropy (sections 4.6, 4.6.6, 4.7)
stage_entropy() {
  stamp entropy
  cd "$ENT"; need "$ROOT/results_gpu/runs/e2e_S_prior/best.pt"
  run "$P -c 'import sys; sys.path.insert(0, \"$ROOT/gpu\"); import residual_entropy'"   # builds gpu/_residual_entropy.so if missing (box refresh)
  note "main AC benchmark + backbone AE rows (benchmark/entropy/run_all.sh, run_ae.sh first part, membrane retry of run_final.sh / run_resume5.sh)"
  run "( $P bench_entropy.py --threads 3 --systems $D8A --T 1000,250 --out $ER/entropy_raw_A.json; $P bench_entropy.py --threads 3 --systems mddb_small_1APQ,mddb_dna_1d89,mddb_medium_12CA,mddb_rna_2l8f,mddb_protdna_FOXA3 --T 1000 --out $ER/entropy_raw_A.json ) > logs/run_A.log 2>&1 &
       ( $P bench_entropy.py --threads 3 --systems $D8B --T 1000,250 --out $ER/entropy_raw_B.json; $P bench_entropy.py --threads 3 --systems mddb_large_1AYX --T 1000 --out $ER/entropy_raw_B.json ) > logs/run_B.log 2>&1 &
       wait"
  run "$P bench_entropy.py --threads 8 --systems mddb_membrane_Eprot --T 1000 --lzma-coders $LZ2 --out $ER/entropy_raw_C.json > logs/r5_membrane.log 2>&1"   # alone: OOM when run next to others
  note "predictor analysis (section 4.7)"
  run "$P analyze_residuals.py --systems $D8A --T 1000 --out $ER/predictor_analysis_1.json > logs/pred_1.log 2>&1 &
       $P analyze_residuals.py --systems $D8B --T 1000 --out $ER/predictor_analysis_2.json > logs/pred_2.log 2>&1 &
       wait"   # [INFERRED] STATE ~12:43: 16 domains, two processes
  run "$P bench_entropy.py --threads 8 --no-ae --systems 1e8rA00,4m70B00,1b5eA00 --T 1000 --out $ER/timing_clean.json > logs/r5_timing.log 2>&1"   # run_tail.sh / run_resume5.sh; alone for timing
}

# ------------------------------------------------------------------------------------------------- scaling (section 4.6.9)
stage_scaling() {
  stamp scaling
  cd "$ENT"; need /workspace/bench_data_long /workspace/bench_data_320K /workspace/bench_data_mddb_long
  run "( cd ../scripts && $P bench_formats.py --systems 1e8rA00,4m70B00,2lg7A00 --T $LEV --data /workspace/bench_data_long --only 'XTC precision|MDCompress -l 4' --out $ER/scaling_formats.json ) > logs/scaling_formats.log 2>&1"   # run_ae.sh
  run "( cd ../scripts && $P bench_formats.py --systems 1e8rA00,4m70B00,2lg7A00 --T $LEV --data /workspace/bench_data_long --only 'MDCompress -l 2' --out $ER/scaling_formats.json ) > logs/scaling_formats_l2.log 2>&1"   # [INFERRED] STATE line 'MDCompress -l 2 must be added'
  run "$P bench_entropy.py --threads 4 --systems 1e8rA00,4m70B00 --T $LEV --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $ER/scaling_entropy_1.json > logs/scaling_1.log 2>&1"   # run_ae.sh
  run "$P bench_entropy.py --threads 4 --systems 2lg7A00 --T 10,50,100,250,500,1000,2500,5000 --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $ER/scaling_entropy_2b.json > logs/scaling_2b.log 2>&1"   # [INFERRED] levels from the JSON
  run "$P bench_entropy.py --threads 8 --systems 2lg7A00 --T 10000 --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $ER/scaling_entropy_2a.json > logs/scaling_2a_retry.log 2>&1"   # run_final2.sh (alone: OOM otherwise)
  run "bash run_320K.sh > logs/run_320K.log 2>&1"                                          # 320 K scaling + replica boundary test (4.6.9.1)
  run "bash run_mddb_long.sh > logs/run_mddb_long.log 2>&1"                                # MDDB MD-A00008 constant temperature (4.6.9.2)
}

# ------------------------------------------------------------------------------------------------- ablation (section 4.7.6)
stage_ablation() {
  stamp ablation
  cd "$ENT"
  run "bash run_ablation.sh > logs/run_ablation.log 2>&1"                                  # -> results/ablation_{A,B,C,D}.json
  note "the original B chain was split: its T = 250 half went to a separate file (STATE 19:28); run_ablation.sh writes it into ablation_B.json instead,"
  note "which ablation_section.py reads the same way. The original separate command was:"
  run "$P ablate.py --threads 3 --systems $D8B --T 250 --out $ER/ablation_B250.json > logs/abl_B250.log 2>&1"   # [INFERRED] from the log and result names
}

# ------------------------------------------------------------------------------------------------- codec (section 4.7.7)
stage_codec() {
  stamp codec
  cd "$ROOT"
  run "$P tests/test_codec.py > $ENT/logs/codec_tests.log 2>&1"                            # -> results/codec_tests.json (106 checks)
  run "$P tests/test_codec.py --methods-only > $ENT/logs/codec_method_tests.log 2>&1"     # -> results/codec_method_tests.json (all --method cases)
  run "$P tests/test_report.py > $ENT/logs/report_tests.log 2>&1"                          # -> results/report_tests.json (metadata / JSON report)
}

# ------------------------------------------------------------------------------------------------- report
stage_report() {
  stamp report
  need "$ER/entropy_raw_A.json" "$ER/ablation_A.json" "$ER/codec_tests.json"
  note "report_extra.json is optional (make_report.py skips it if absent). Report outputs are written locally and are not tracked."
  run "cd $BEN/scripts && $P aggregate.py"                     # -> results/benchmark_summary.json, results/tables.md, figs/fig_pareto_*, fig_bars_*, fig_timing, fig_mddb_systems
  run "cd $ENT && $P entropy_section.py"                       # -> entropy_section.md (4.6, 4.6.6, 4.6.9), figs/fig_entropy_bounds.png, fig_frames_scaling.png
  run "cd $ENT && $P predictor_section.py"                     # -> predictor_section.md (4.7), figs/fig_residual_hist.png, fig_entropy_vs_bound.png
  run "cd $ENT && $P ablation_section.py"                      # -> ablation_section.md (4.7.6), figs/fig_ablation.png
  run "cd $ENT && $SYS codec_section.py"                       # -> codec_section.md (4.7.7); does not change the README
  run "cd $BEN/scripts && $SYS make_report.py"                 # -> benchmark/benchmark_report.{md,html,pdf}
  run "cd $ENT && $SYS readme_section.py"                      # -> readme_results_block.md (untracked); does not change the README
}

for s in $STAGES; do "stage_$s"; done
echo; echo "done: $STAGES ($(date '+%F %H:%M %Z'))"
