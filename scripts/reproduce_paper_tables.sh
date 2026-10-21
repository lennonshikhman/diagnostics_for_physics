#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the benchmark data root.}
OUTDIR=${OUTDIR:-outputs_diagnostic_suite}

diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --tasks adv2d shallow2d active2d cns3d mhd3d \
  --models fno deeponet unet resnet \
  --seeds 0 1 2 \
  --eval-horizons 1 2 4 \
  --epochs-well 25 \
  --epochs-underfit 2 \
  --max-train-pairs 4096 \
  --max-val-pairs 1024 \
  --max-test-windows 512 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --hash-data \
  --resume \
  --outdir "$OUTDIR"

diagnostic-suite-tables \
  --metrics "$OUTDIR/metrics_summary.csv" \
  --outdir "$OUTDIR/paper_tables" \
  --horizon 2
