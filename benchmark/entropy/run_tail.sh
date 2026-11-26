#!/bin/bash
# after ae_B finishes: membrane AE rows, then 2lg7A00 T=10000 scaling, then the clean timing pass (alone as far as possible)
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results; LZ3=none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma
while kill -0 1221068 2>/dev/null; do sleep 30; done
$P bench_entropy.py --threads 4 --systems mddb_membrane_Eprot --T 1000 --lzma-coders none-nat-vint-lzma,none-nat-planes-lzma --out $R/entropy_raw_C.json > logs/ae_C.log 2>&1
$P bench_entropy.py --threads 4 --systems 2lg7A00 --T 10000 --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $R/scaling_entropy_2a.json > logs/scaling_2a.log 2>&1
while pgrep -f "bench_entropy.py|analyze_residuals.py" >/dev/null; do sleep 30; done
$P bench_entropy.py --threads 8 --no-ae --systems 1e8rA00,4m70B00,1b5eA00 --T 1000 --out $R/timing_clean.json > logs/timing_clean.log 2>&1
echo finished
