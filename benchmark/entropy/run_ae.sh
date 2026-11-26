#!/bin/bash
# AEZ-12 added scope: (1) backbone AE variant rows in the main benchmark (existing rows are skipped), (2) frames scaling
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results; LZ3=none-nat-vint-lzma,none-nat-planes-lzma,dt-nat-planes-lzma
SC=1e8rA00,4m70B00,2lg7A00; LEV=10,50,100,250,500,1000,2500,5000,10000
( $P bench_entropy.py --threads 3 --systems 1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00 --T 1000,250 --out $R/entropy_raw_A.json
  $P bench_entropy.py --threads 3 --systems mddb_small_1APQ,mddb_medium_12CA,mddb_protdna_FOXA3 --T 1000 --out $R/entropy_raw_A.json ) > logs/ae_A.log 2>&1 &
( $P bench_entropy.py --threads 3 --systems 1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00 --T 1000,250 --out $R/entropy_raw_B.json
  $P bench_entropy.py --threads 3 --systems mddb_large_1AYX --T 1000 --out $R/entropy_raw_B.json ) > logs/ae_B.log 2>&1 &
( cd ../scripts && $P bench_formats.py --systems $SC --T $LEV --data /workspace/bench_data_long --only "XTC precision|MDCompress -l 4" --out $R/scaling_formats.json ) > logs/scaling_formats.log 2>&1 &
wait
$P bench_entropy.py --threads 6 --systems mddb_membrane_Eprot --T 1000 --lzma-coders none-nat-vint-lzma,none-nat-planes-lzma --out $R/entropy_raw_C.json > logs/ae_C.log 2>&1
( $P bench_entropy.py --threads 4 --systems 2lg7A00 --T $LEV --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $R/scaling_entropy_2.json ) > logs/scaling_2.log 2>&1 &
( $P bench_entropy.py --threads 4 --systems 1e8rA00,4m70B00 --T $LEV --data /nonexistent --pdb-dir /workspace/bench_data_long --lzma-coders $LZ3 --out $R/scaling_entropy_1.json ) > logs/scaling_1.log 2>&1 &
wait
echo finished
