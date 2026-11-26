#!/bin/bash
# Full GPU schedule (resumable: finished runs exit immediately, interrupted ones resume from last.pt).
# Stage 1 (parallel): shared backbone codec + four packers (S/L x prior/no prior) on true backbones.
# Stage 2 (parallel): end-to-end fine-tuning of codec + packer for the four packer variants.
# Then evaluation on held-out superfamilies + the external set.
set -u
cd /content/md_compress
D=/content/data/mdcath; RUNS=/content/runs; LOG=/content/runs/logs; mkdir -p $LOG
T_CODEC=${T_CODEC:-15}; T_PACK=${T_PACK:-35}; T_E2E=${T_E2E:-15}
# batch (frames of one protein per step): L at 32 ran out of memory with 5 runs sharing the 24 GB L4
declare -A BS=([S]=32 [L]=16)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date +%T) data"
python -u gpu/prep_external.py /content/data/external > $LOG/prep_external.log 2>&1 &
python -u gpu/prep_mdcath.py --workers 16 > $LOG/prep_mdcath.log 2>&1
wait
echo "$(date +%T) buffers"; python -u gpu/train_gpu.py buffers --data $D --runs $RUNS > $LOG/buffers.log 2>&1
echo "$(date +%T) stage 1"
python -u gpu/train_gpu.py codec --data $D --runs $RUNS --minutes $T_CODEC --bs 32 >> $LOG/codec.log 2>&1 &
for v in S L; do
  python -u gpu/train_gpu.py packer --variant $v --data $D --runs $RUNS --minutes $T_PACK --bs ${BS[$v]} >> $LOG/packer_${v}_prior.log 2>&1 &
  python -u gpu/train_gpu.py packer --variant $v --no_prior --data $D --runs $RUNS --minutes $T_PACK --bs ${BS[$v]} >> $LOG/packer_${v}_noprior.log 2>&1 &
done
wait
echo "$(date +%T) stage 2"
for v in S L; do
  for p in prior noprior; do
    flag=""; [ $p = noprior ] && flag="--no_prior"
    python -u gpu/train_gpu.py e2e --variant $v $flag --data $D --runs $RUNS --minutes $T_E2E --bs 16 \
      --init_codec $RUNS/codec_S_prior/best.pt --init_packer $RUNS/packer_${v}_${p}/best.pt >> $LOG/e2e_${v}_${p}.log 2>&1 &
  done
done
wait
echo "$(date +%T) eval"
python -u gpu/eval_gpu.py --data $D --runs $RUNS >> $LOG/eval.log 2>&1
echo "$(date +%T) package"
mkdir -p /content/export/runs /content/export/eval
for r in $RUNS/*/; do n=$(basename $r); [ $n = logs ] || [ $n = eval ] && continue
  mkdir -p /content/export/runs/$n; cp $r/config.json $r/history.json /content/export/runs/$n/ 2>/dev/null
  python - <<PY
import torch; ck = torch.load("$r/best.pt", weights_only=False); torch.save(dict(weights=ck["weights"], config=ck["config"], state={k: v for k, v in ck["state"].items() if k != "history"}), "/content/export/runs/$n/best_weights.pt")
PY
done
cp $RUNS/eval/* /content/export/eval/; cp -r $LOG /content/export/; cp $D/manifest.json /content/export/mdcath_manifest.json
cp $D/buffers.pt /content/export/ 2>/dev/null
mkdir -p /content/export/dyn; for d in $(python -c "import json;m=json.load(open('$D/manifest.json'));t=sorted((v['n_res'],k) for k,v in m['domains'].items() if v['split']=='test');print(' '.join(k for _,k in [t[5],t[len(t)//2],t[-5]]))"); do cp $D/domains/$d.npz /content/export/dyn/; done; true
echo "$(date +%T) all done"
