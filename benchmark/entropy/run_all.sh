#!/bin/bash
# AEZ-12 full run: three nohup queues with separate incremental result files (resumable: finished rows are skipped)
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results
( $P bench_entropy.py --threads 3 --systems 1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00 --T 1000,250 --out $R/entropy_raw_A.json
  $P bench_entropy.py --threads 3 --systems mddb_small_1APQ,mddb_dna_1d89,mddb_medium_12CA,mddb_rna_2l8f,mddb_protdna_FOXA3 --T 1000 --out $R/entropy_raw_A.json ) > logs/run_A.log 2>&1 &
( $P bench_entropy.py --threads 3 --systems 1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00 --T 1000,250 --out $R/entropy_raw_B.json
  $P bench_entropy.py --threads 3 --systems mddb_large_1AYX --T 1000 --out $R/entropy_raw_B.json ) > logs/run_B.log 2>&1 &
( $P bench_entropy.py --threads 2 --systems mddb_membrane_Eprot --T 1000 --lzma-coders none-nat-vint-lzma,none-nat-planes-lzma --out $R/entropy_raw_C.json ) > logs/run_C.log 2>&1 &
wait
