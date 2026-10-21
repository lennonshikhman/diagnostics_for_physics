#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${1:-${DATA_ROOT:?Pass the data root as the first argument or set DATA_ROOT.}}
diagnostic-suite \
  --data-root "$DATA_ROOT" \
  --tasks adv2d shallow2d \
  --models fno unet \
  --smoke
