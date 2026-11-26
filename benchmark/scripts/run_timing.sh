#!/bin/bash
# Clean timing pass: waits until no other benchmark job runs, then times three mdCATH domains (small, medium, large)
# at T = 1000 with nothing else running. Results: results/timingrun_*.json -> results/timing_*.json (sys/T/method keys).
cd /workspace/md_compress/benchmark
# (no wait: run concurrently with the membrane encode, see report)
P=/workspace/conda/envs/bench/bin/python; D=1e8rA00,4m70B00,1b5eA00; R=/workspace/md_compress/benchmark/results
echo "start $(date)" > logs/timing.log; uptime >> logs/timing.log
$P scripts/bench_formats.py --systems $D --T 1000 --out $R/timingrun_formats.json >> logs/timing.log 2>&1
python3 scripts/bench_ours.py --systems $D --T 1000 --out $R/timingrun_ours.json --threads 4 >> logs/timing.log 2>&1
$P scripts/bench_pcazip.py --systems $D --T 1000 --out $R/timingrun_pcazip.json >> logs/timing.log 2>&1
python3 scripts/timing_convert.py >> logs/timing.log 2>&1
echo "end $(date)" >> logs/timing.log
