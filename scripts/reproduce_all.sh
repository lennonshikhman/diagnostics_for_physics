#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:-"${DATA_ROOT:-$HOME/Desktop/research_work/COMET/comet_ncs_codebase/data/raw}"}
OUTDIR=${2:-"outputs_diagnostic_suite"}
diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --tasks ns2d shallow2d active2d cns3d mhd3d \
  --models fno deeponet unet resnet
