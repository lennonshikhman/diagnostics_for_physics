#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-data}
OUTDIR=${OUTDIR:-outputs_toy_smoke}

diagnostic-suite --make-toy --data-root "$DATA_ROOT" --toy-grid 24 --toy-steps 6

diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --tasks toy2d \
  --models fno unet resnet \
  --smoke \
  --device cpu \
  --outdir "$OUTDIR"
