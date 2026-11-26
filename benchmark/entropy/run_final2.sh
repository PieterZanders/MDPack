#!/bin/bash
# 2lg7A00 T=10000 scaling retry alone (first attempt OOM killed in the hybrid phase at ~13:50 while other jobs ran)
cd /workspace/md_compress/benchmark/entropy
while kill -0 1265404 2>/dev/null; do sleep 30; done
/workspace/conda/envs/bench/bin/python bench_entropy.py --threads 8 --systems 2lg7A00 --T 10000 --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma --out /workspace/md_compress/benchmark/entropy/results/scaling_entropy_2a.json > logs/scaling_2a_retry.log 2>&1
echo finished
