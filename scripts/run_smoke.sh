#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:-"${DATA_ROOT:-$HOME/Desktop/research_work/COMET/comet_ncs_codebase/data/raw}"}
diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --tasks ns2d shallow2d \
  --models fno unet \
  --smoke
