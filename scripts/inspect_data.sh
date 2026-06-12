#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:-data}

diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --tasks ns2d shallow2d active2d cns3d mhd3d \
  --inspect-data
