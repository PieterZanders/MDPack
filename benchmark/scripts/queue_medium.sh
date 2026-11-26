#!/usr/bin/env bash
# medium protein queue (default system 4m70B00): chain A main -> angle -> frames (main JSON);
# chain B tune, then slow after main has finished (side JSON, merged by make_mdpack_report.py) so main timings are not disturbed
cd "$(dirname "$0")/.."
P=${BENCH_PY:-/workspace/conda/envs/bench/bin/python}; S=${1:-4m70B00}; mkdir -p logs results
( for s in main angle frames; do $P scripts/bench_medium.py --system $S --stage $s > logs/medium_${S}_$s.log 2>&1; touch logs/medium_${S}_$s.done; done ) &
( $P scripts/bench_medium.py --system $S --stage tune --out results/medium_${S}_side.json > logs/medium_${S}_tune.log 2>&1
  while [ ! -e logs/medium_${S}_main.done ]; do sleep 20; done
  $P scripts/bench_medium.py --system $S --stage slow --out results/medium_${S}_side.json > logs/medium_${S}_slow.log 2>&1 ) &
wait
