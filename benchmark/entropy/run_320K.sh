#!/bin/bash
# AEZ-12 follow up: same temperature (320 K) frames scaling on 2lg7A00 + replica boundary test at T = 500
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results; D=/workspace/bench_data_320K
LZ3=none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma; LEV=10,50,100,250,500,1000,1500,2000,2500
( cd ../scripts && $P bench_formats.py --systems 2lg7A00_320K --T $LEV --data $D --only "XTC precision|MDCompress -l 4|MDCompress -l 2" --out $R/scaling320_formats.json
  $P bench_formats.py --systems 2lg7A00_b0,2lg7A00_p4,2lg7A00_p9,2lg7A00_r4 --T 500 --data $D --only "XTC precision|MDCompress -l 4|MDCompress -l 2" --out $R/scaling320_formats.json ) > logs/s320_formats.log 2>&1 &
( $P bench_entropy.py --threads 4 --systems 2lg7A00_320K --T $LEV --data $D --lzma-coders $LZ3 --out $R/scaling320_entropy.json
  $P bench_entropy.py --threads 4 --systems 2lg7A00_b0,2lg7A00_p4,2lg7A00_p9,2lg7A00_r4 --T 500 --data $D --lzma-coders $LZ3 --out $R/scaling320_boundary.json ) > logs/s320_entropy.log 2>&1 &
wait; echo finished
