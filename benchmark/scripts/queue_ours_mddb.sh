#!/bin/bash
cd /workspace/md_compress/benchmark
while pgrep -f "bench_ours.py --systems 1e8r" >/dev/null || pgrep -f "bench_ours.py --systems 1jyo" >/dev/null; do sleep 20; done
python3 scripts/bench_ours.py --systems mddb_small_1APQ,mddb_dna_1d89,mddb_rna_2l8f,mddb_protdna_FOXA3,mddb_medium_12CA,mddb_large_1AYX,mddb_membrane_Eprot --T 1000 --out results/ours_raw_mddb.json --threads 4 > logs/ours_mddb.log 2>&1
