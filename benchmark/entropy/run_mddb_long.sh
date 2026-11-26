#!/bin/bash
# AEZ-12 constant temperature frames scaling on MDDB MD-A00008 (MoDEL 1APQ, 300 K NPT, 10 ns, 10,000 frames at 1 ps), all frames
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results; D=/workspace/bench_data_mddb_long
LZ3=none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma; LEV=10,50,100,250,500,1000,2500,5000,10000
( cd ../scripts && $P bench_formats.py --systems mddb_small_1APQ_long --T $LEV --data $D --only "XTC precision|MDCompress -l 4|MDCompress -l 2" --out $R/scalingmddb_formats.json ) > logs/smddb_formats.log 2>&1 &
$P bench_entropy.py --threads 4 --systems mddb_small_1APQ_long --T $LEV --data $D --lzma-coders $LZ3 --out $R/scalingmddb_entropy.json > logs/smddb_entropy.log 2>&1 &
wait; echo finished
