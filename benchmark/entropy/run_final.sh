#!/bin/bash
# after run_tail.sh: membrane AE rows alone (the first attempt was OOM killed at 13:3x while other jobs ran)
cd /workspace/md_compress/benchmark/entropy
while kill -0 1243300 2>/dev/null; do sleep 30; done
/workspace/conda/envs/bench/bin/python bench_entropy.py --threads 8 --systems mddb_membrane_Eprot --T 1000 --lzma-coders none-nat-vint-lzma,none-nat-planes-lzma --out /workspace/md_compress/benchmark/entropy/results/entropy_raw_C.json > logs/ae_C2.log 2>&1
echo finished
