#!/bin/bash
# 14:37 resume after box refresh: missing rows only (bench_entropy skips existing keys)
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results; LZ3=none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma
$P bench_entropy.py --threads 4 --systems 4m70B00 --T 10000 --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $R/scaling_entropy_1.json > logs/r5_4m70.log 2>&1 &
$P bench_entropy.py --threads 4 --systems 2lg7A00 --T 10000 --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $R/scaling_entropy_2a.json > logs/r5_2lg7.log 2>&1 &
wait
$P bench_entropy.py --threads 8 --systems mddb_membrane_Eprot --T 1000 --lzma-coders none-nat-vint-lzma,none-nat-planes-lzma --out $R/entropy_raw_C.json > logs/r5_membrane.log 2>&1
$P bench_entropy.py --threads 8 --no-ae --systems 1e8rA00,4m70B00,1b5eA00 --T 1000 --out $R/timing_clean.json > logs/r5_timing.log 2>&1
echo finished
