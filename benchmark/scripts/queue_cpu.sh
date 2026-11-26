#!/bin/bash
# CPU queue on the box (after the first mdCATH format and our-method runs)
cd /workspace/md_compress/benchmark
P=/workspace/conda/envs/bench/bin/python
while pgrep -f bench_formats.py >/dev/null; do sleep 20; done
$P scripts/bench_formats.py --systems mddb_small_1APQ,mddb_dna_1d89,mddb_rna_2l8f,mddb_protdna_FOXA3,mddb_medium_12CA,mddb_large_1AYX,mddb_membrane_Eprot --T 1000 > logs/formats_mddb.log 2>&1
$P scripts/bench_pcazip.py --systems 1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00,1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00,mddb_small_1APQ,mddb_dna_1d89,mddb_rna_2l8f,mddb_protdna_FOXA3,mddb_medium_12CA,mddb_large_1AYX,mddb_membrane_Eprot --T 1000,250 > logs/pcazip.log 2>&1
