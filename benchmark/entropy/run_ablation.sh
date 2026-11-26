#!/bin/bash
# AEZ-12 ablation: full / no codes / no MLP / neither (Pieter). Membrane skipped (20k atoms, memory); DNA and RNA have no protein (hybrid not applicable).
cd /workspace/md_compress/benchmark/entropy
P=/workspace/conda/envs/bench/bin/python; R=/workspace/md_compress/benchmark/entropy/results
( $P ablate.py --threads 3 --systems 1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00 --T 1000 --out $R/ablation_A.json
  $P ablate.py --threads 3 --systems 1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00 --T 250 --out $R/ablation_A.json ) > logs/abl_A.log 2>&1 &
( $P ablate.py --threads 3 --systems 1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00 --T 1000 --out $R/ablation_B.json
  $P ablate.py --threads 3 --systems 1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00 --T 250 --out $R/ablation_B.json ) > logs/abl_B.log 2>&1 &
( $P ablate.py --threads 3 --systems mddb_small_1APQ,mddb_medium_12CA,mddb_protdna_FOXA3,mddb_large_1AYX --T 1000 --out $R/ablation_C.json ) > logs/abl_C.log 2>&1 &
( $P ablate.py --threads 3 --systems mddb_small_1APQ_long --T 1000,10000 --data /workspace/bench_data_mddb_long --out $R/ablation_D.json ) > logs/abl_D.log 2>&1 &
wait; echo finished
