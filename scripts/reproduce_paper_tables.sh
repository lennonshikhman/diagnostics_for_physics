#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the benchmark data root.}
OUTDIR=${OUTDIR:-outputs_diagnostic_suite}

diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --tasks ns2d shallow2d active2d cns3d mhd3d \
  --models fno deeponet unet resnet \
  --epochs-well 25 \
  --epochs-underfit 2 \
  --max-train-pairs 4096 \
  --max-val-pairs 1024 \
  --max-test-windows 512 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --outdir "$OUTDIR"
