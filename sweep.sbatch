#!/bin/bash

NX=10
NY=10

python3 run_worker.py \
  --nx "$NX" --ny "$NY" \
  --steps 6000 \
  --coarse-gx 25 --coarse-gy 25 \
  --dt 0.02 --mass 1.0 --gamma 0.8 \
  --seed 123 \
  --outdir results